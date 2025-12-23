"""
评估训练好的 PPO 策略在 Branin 函数上的性能
对比方法：TabPFN+RL, GP+RL, GP+EI, TabPFN+EI
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from train_rl import (
    CandidateSelector,
    BraninVariantFunction,
)
from select_candidates import (
    SelectionConfig,
    BraninFunction,
    select_candidates,
    compute_ei,
)


class RLAgent:
    """加载训练好的 RL 策略"""
    
    def __init__(self, model_path: str, feature_dim: int = 4, device: str = "cpu"):
        self.device = device
        self.policy = CandidateSelector(
            feature_dim=feature_dim,
            hidden_dim=64,
            n_layers=2,
            n_heads=4,
        ).to(device)
        
        # 加载模型
        state_dict = torch.load(model_path, map_location=device)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        print(f"已加载模型: {model_path}")
    
    def select(self, X_selected: np.ndarray, pred_mean: np.ndarray, pred_std: np.ndarray,
               y_best: float = None, bounds: Tuple[np.ndarray, np.ndarray] = None) -> int:
        """
        选择候选点
        
        Args:
            X_selected: 候选点坐标
            pred_mean: 预测均值
            pred_std: 预测标准差
            y_best: 当前最优值（用于相对特征）
            bounds: (lower, upper) 坐标边界
        """
        # 构建归一化的相对特征
        X = np.array(X_selected)
        mean = np.array(pred_mean)
        std = np.array(pred_std)
        
        # 1. 坐标归一化到 [0, 1]
        if bounds is not None:
            lower, upper = bounds
            X_norm = (X - lower) / (upper - lower + 1e-8)
        else:
            X_norm = X
        
        # 2. 均值转为相对特征
        if y_best is not None:
            mean_rel = mean - y_best
        else:
            mean_rel = mean
        
        mean_std_val = mean_rel.std()
        if mean_std_val > 1e-8:
            mean_norm = (mean_rel - mean_rel.mean()) / mean_std_val
        else:
            mean_norm = mean_rel - mean_rel.mean()
        
        # 3. 标准差标准化
        std_std_val = std.std()
        if std_std_val > 1e-8:
            std_norm = (std - std.mean()) / std_std_val
        else:
            std_norm = std - std.mean()
        
        X_t = torch.FloatTensor(X_norm)
        mean_t = torch.FloatTensor(mean_norm).unsqueeze(-1)
        std_t = torch.FloatTensor(std_norm).unsqueeze(-1)
        features = torch.cat([X_t, mean_t, std_t], dim=-1)
        features = features.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, _ = self.policy(features)
            action = torch.argmax(logits, dim=-1).item()
        
        return action


def get_gp_predictions(X_context: np.ndarray, y_context: np.ndarray, 
                       X_candidates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """使用 GP 进行预测"""
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    gp.fit(X_context, y_context)
    mean, std = gp.predict(X_candidates, return_std=True)
    return mean, std


def run_tabpfn_rl(func, agent, tabpfn_path: str, X_context: np.ndarray, 
                  y_context: np.ndarray, max_steps: int, rng, device: str,
                  selection_config: SelectionConfig) -> Tuple[List[float], np.ndarray]:
    """TabPFN + RL 方法"""
    X_context = X_context.copy()
    y_context = y_context.copy()
    best_y = float(y_context.min())
    global_min = func.optimal_value
    bounds = func.bounds
    
    regrets = [best_y - global_min]
    trajectory = [X_context.copy()]
    
    for step in range(max_steps):
        selection_config.random_seed = int(rng.integers(0, 100000))
        
        result = select_candidates(
            func=func,
            model_path=tabpfn_path,
            config=selection_config,
            X_context=X_context,
        )
        
        X_selected = result["X_selected"]
        pred_mean = result["y_selected_pred_mean"]
        pred_std = result["y_selected_pred_std"]
        
        action = agent.select(X_selected, pred_mean, pred_std, y_best=best_y, bounds=bounds)
        x_new = X_selected[action:action+1]
        y_new = func(x_new)[0]
        
        X_context = np.vstack([X_context, x_new])
        y_context = np.concatenate([y_context, [y_new]])
        best_y = min(best_y, float(y_new))
        regrets.append(best_y - global_min)
        trajectory.append(x_new.copy())
    
    return regrets, np.vstack(trajectory)


def run_gp_rl(func, agent, X_context: np.ndarray, y_context: np.ndarray, 
              max_steps: int, rng, selection_config: SelectionConfig) -> Tuple[List[float], np.ndarray]:
    """GP + RL 方法"""
    from scipy.stats.qmc import Sobol
    
    X_context = X_context.copy()
    y_context = y_context.copy()
    best_y = float(y_context.min())
    global_min = func.optimal_value
    lower, upper = func.bounds
    bounds = (lower, upper)
    
    regrets = [best_y - global_min]
    trajectory = [X_context.copy()]
    
    for step in range(max_steps):
        # Sobol 采样候选点
        sobol_sampler = Sobol(d=func.dim, scramble=True, seed=int(rng.integers(0, 100000)))
        sobol_samples = sobol_sampler.random(selection_config.n_candidates)
        X_candidates = (sobol_samples * (upper - lower) + lower).astype(np.float32)
        
        # GP 预测
        gp_mean, gp_std = get_gp_predictions(X_context, y_context, X_candidates)
        
        # EI 选择 top-k
        ei_values = compute_ei(gp_mean, gp_std, y_context.min(), xi=selection_config.xi)
        top_k_ei_indices = np.argsort(ei_values)[-selection_config.top_k_ei:][::-1]
        
        # Variance 选择 top-k
        remaining_mask = np.ones(len(X_candidates), dtype=bool)
        remaining_mask[top_k_ei_indices] = False
        remaining_indices = np.where(remaining_mask)[0]
        var_remaining = gp_std[remaining_indices] ** 2
        top_k_var_local = np.argsort(var_remaining)[-selection_config.top_k_var:][::-1]
        top_k_var_indices = remaining_indices[top_k_var_local]
        
        # 随机选择
        remaining_mask[top_k_var_indices] = False
        remaining_indices = np.where(remaining_mask)[0]
        random_indices = rng.choice(remaining_indices, size=selection_config.n_random, replace=False)
        
        # 合并
        selected_indices = np.concatenate([top_k_ei_indices, top_k_var_indices, random_indices])
        X_selected = X_candidates[selected_indices]
        pred_mean = gp_mean[selected_indices]
        pred_std = gp_std[selected_indices]
        
        # RL 选择（传入 y_best 和 bounds）
        action = agent.select(X_selected, pred_mean, pred_std, y_best=best_y, bounds=bounds)
        x_new = X_selected[action:action+1]
        y_new = func(x_new)[0]
        
        X_context = np.vstack([X_context, x_new])
        y_context = np.concatenate([y_context, [y_new]])
        best_y = min(best_y, float(y_new))
        regrets.append(best_y - global_min)
        trajectory.append(x_new.copy())
    
    return regrets, np.vstack(trajectory)


def run_gp_ei(func, X_context: np.ndarray, y_context: np.ndarray, 
              max_steps: int, rng, n_candidates: int = 512) -> Tuple[List[float], np.ndarray]:
    """GP + EI (标准贝叶斯优化)"""
    from scipy.stats.qmc import Sobol
    
    X_context = X_context.copy()
    y_context = y_context.copy()
    best_y = float(y_context.min())
    global_min = func.optimal_value
    lower, upper = func.bounds
    
    regrets = [best_y - global_min]
    trajectory = [X_context.copy()]
    
    for step in range(max_steps):
        # Sobol 采样候选点
        sobol_sampler = Sobol(d=func.dim, scramble=True, seed=int(rng.integers(0, 100000)))
        sobol_samples = sobol_sampler.random(n_candidates)
        X_candidates = (sobol_samples * (upper - lower) + lower).astype(np.float32)
        
        # GP 预测
        gp_mean, gp_std = get_gp_predictions(X_context, y_context, X_candidates)
        
        # EI 选择最大值
        ei_values = compute_ei(gp_mean, gp_std, y_context.min(), xi=0.01)
        best_idx = np.argmax(ei_values)
        
        x_new = X_candidates[best_idx:best_idx+1]
        y_new = func(x_new)[0]
        
        X_context = np.vstack([X_context, x_new])
        y_context = np.concatenate([y_context, [y_new]])
        best_y = min(best_y, float(y_new))
        regrets.append(best_y - global_min)
        trajectory.append(x_new.copy())
    
    return regrets, np.vstack(trajectory)


def run_tabpfn_ei(func, tabpfn_path: str, X_context: np.ndarray, y_context: np.ndarray, 
                  max_steps: int, rng, device: str, 
                  selection_config: SelectionConfig) -> Tuple[List[float], np.ndarray]:
    """TabPFN + EI"""
    X_context = X_context.copy()
    y_context = y_context.copy()
    best_y = float(y_context.min())
    global_min = func.optimal_value
    
    regrets = [best_y - global_min]
    trajectory = [X_context.copy()]
    
    for step in range(max_steps):
        selection_config.random_seed = int(rng.integers(0, 100000))
        
        result = select_candidates(
            func=func,
            model_path=tabpfn_path,
            config=selection_config,
            X_context=X_context,
        )
        
        # 从所有候选点中选 EI 最大的
        ei_values = result["ei_values"]
        best_idx = np.argmax(ei_values)
        
        x_new = result["X_candidates"][best_idx:best_idx+1]
        y_new = func(x_new)[0]
        
        X_context = np.vstack([X_context, x_new])
        y_context = np.concatenate([y_context, [y_new]])
        best_y = min(best_y, float(y_new))
        regrets.append(best_y - global_min)
        trajectory.append(x_new.copy())
    
    return regrets, np.vstack(trajectory)


def evaluate_on_branin(
    model_path: str,
    tabpfn_path: str,
    n_runs: int = 10,
    max_steps: int = 20,
    n_init: int = 2,
    seed: int = 42,
    device: str = "cpu",
    save_dir: str = "./results",
):
    """
    在标准 Branin 函数上评估所有方法
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Comparison on Branin Function")
    print(f"{'='*60}")
    print(f"Methods: TabPFN+RL, GP+RL, GP+EI, TabPFN+EI")
    print(f"Runs: {n_runs}, Steps: {max_steps}, Init: {n_init}")
    print(f"{'='*60}\n")
    
    # 加载 RL 策略
    agent = RLAgent(model_path, feature_dim=4, device=device)
    
    # 创建 Branin 函数
    func = BraninFunction()
    global_min = func.optimal_value
    lower, upper = func.bounds
    
    print(f"Branin global minimum: {global_min:.6f}")
    
    # 候选点选择配置
    selection_config = SelectionConfig(
        n_candidates=512,
        top_k_ei=20,
        top_k_var=10,
        n_random=10,
        verbose=False,
        device=device,
    )
    
    # 存储结果
    results = {
        "TabPFN+RL": {"regrets": [], "trajectories": []},
        "GP+RL": {"regrets": [], "trajectories": []},
        "GP+EI": {"regrets": [], "trajectories": []},
        "TabPFN+EI": {"regrets": [], "trajectories": []},
    }
    
    rng = np.random.default_rng(seed)
    
    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}")
        run_seed = int(rng.integers(0, 100000))
        
        # 相同的初始点
        init_rng = np.random.default_rng(run_seed)
        X_init = init_rng.uniform(lower, upper, size=(n_init, func.dim)).astype(np.float32)
        y_init = func(X_init)
        
        # TabPFN + RL
        rng1 = np.random.default_rng(run_seed)
        regrets, traj = run_tabpfn_rl(func, agent, tabpfn_path, X_init, y_init, 
                                       max_steps, rng1, device, selection_config)
        results["TabPFN+RL"]["regrets"].append(regrets)
        results["TabPFN+RL"]["trajectories"].append(traj)
        print(f"  TabPFN+RL: final regret = {regrets[-1]:.6f}")
        
        # GP + RL
        rng2 = np.random.default_rng(run_seed)
        regrets, traj = run_gp_rl(func, agent, X_init, y_init, max_steps, rng2, selection_config)
        results["GP+RL"]["regrets"].append(regrets)
        results["GP+RL"]["trajectories"].append(traj)
        print(f"  GP+RL: final regret = {regrets[-1]:.6f}")
        
        # GP + EI
        rng3 = np.random.default_rng(run_seed)
        regrets, traj = run_gp_ei(func, X_init, y_init, max_steps, rng3, n_candidates=512)
        results["GP+EI"]["regrets"].append(regrets)
        results["GP+EI"]["trajectories"].append(traj)
        print(f"  GP+EI: final regret = {regrets[-1]:.6f}")
        
        # TabPFN + EI
        rng4 = np.random.default_rng(run_seed)
        regrets, traj = run_tabpfn_ei(func, tabpfn_path, X_init, y_init, 
                                       max_steps, rng4, device, selection_config)
        results["TabPFN+EI"]["regrets"].append(regrets)
        results["TabPFN+EI"]["trajectories"].append(traj)
        print(f"  TabPFN+EI: final regret = {regrets[-1]:.6f}")
    
    # 绘制对比曲线
    plot_comparison(results, max_steps, save_dir)
    
    # 绘制轨迹图（使用第一次运行的轨迹）
    plot_trajectories(func, results, save_dir)
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("Final Results Summary")
    print(f"{'='*60}")
    for method, data in results.items():
        final_regrets = [r[-1] for r in data["regrets"]]
        print(f"{method}:")
        print(f"  Mean: {np.mean(final_regrets):.6f} +/- {np.std(final_regrets):.6f}")
    
    return results


def plot_comparison(results: Dict, max_steps: int, save_dir: str):
    """绘制对比曲线"""
    plt.figure(figsize=(10, 6))
    
    colors = {
        "TabPFN+RL": "tab:blue",
        "GP+RL": "tab:orange", 
        "GP+EI": "tab:green",
        "TabPFN+EI": "tab:red",
    }
    
    x = np.arange(max_steps + 1)
    
    for method, data in results.items():
        regrets = np.array(data["regrets"])
        mean = regrets.mean(axis=0)
        std = regrets.std(axis=0)
        
        plt.plot(x, mean, label=method, color=colors[method], linewidth=2)
        plt.fill_between(x, mean - std, mean + std, color=colors[method], alpha=0.2)
    
    plt.xlabel("Number of Function Evaluations", fontsize=12)
    plt.ylabel("Simple Regret", fontsize=12)
    plt.title("Comparison on Branin Function", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison_curve.png", dpi=150)
    plt.savefig(f"{save_dir}/comparison_curve.pdf")
    plt.close()
    print(f"\nSaved comparison curve to {save_dir}/comparison_curve.png")


def plot_trajectories(func, results: Dict, save_dir: str):
    """绘制轨迹图，一行四列"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 创建 Branin 等高线背景
    lower, upper = func.bounds
    x1 = np.linspace(lower[0], upper[0], 100)
    x2 = np.linspace(lower[1], upper[1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = func(np.array([[X1[i, j], X2[i, j]]]))[0]
    
    methods = ["TabPFN+RL", "GP+RL", "GP+EI", "TabPFN+EI"]
    
    for ax, method in zip(axes, methods):
        # 等高线
        contour = ax.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.8)
        ax.contour(X1, X2, Z, levels=20, colors='white', alpha=0.3, linewidths=0.5)
        
        # 轨迹（使用第一次运行）
        traj = results[method]["trajectories"][0]
        
        # 初始点
        n_init = 2
        ax.scatter(traj[:n_init, 0], traj[:n_init, 1], c='white', s=100, 
                   marker='o', edgecolors='black', linewidths=1.5, label='Init', zorder=5)
        
        # 后续点
        ax.scatter(traj[n_init:, 0], traj[n_init:, 1], c='red', s=50, 
                   marker='x', linewidths=1.5, label='Query', zorder=5)
        
        # 连线
        ax.plot(traj[:, 0], traj[:, 1], 'r--', alpha=0.5, linewidth=1)
        
        # 全局最优点
        # Branin 有三个全局最优点
        optima = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
        ax.scatter(optima[:, 0], optima[:, 1], c='yellow', s=150, marker='*', 
                   edgecolors='black', linewidths=1, label='Optimum', zorder=6)
        
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_xlabel("$x_1$", fontsize=11)
        ax.set_ylabel("$x_2$", fontsize=11)
        ax.set_title(method, fontsize=12)
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/trajectories.png", dpi=150)
    plt.savefig(f"{save_dir}/trajectories.pdf")
    plt.close()
    print(f"Saved trajectories to {save_dir}/trajectories.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL policy on Branin")
    parser.add_argument("--model_path", type=str, default="/gpfs/radev/pi/cohan/yz979/lin/test/Branin/runs/ppo_bo_1217_v1/ppo_ep5000.pt",
                        help="PPO model path")
    parser.add_argument("--tabpfn_path", type=str, default="./model/finetuned_tabpfn_branin_family.ckpt",
                        help="TabPFN model path")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--max_steps", type=int, default=20, help="Max steps per run")
    parser.add_argument("--n_init", type=int, default=2, help="Initial context points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="./results", help="Save directory")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = evaluate_on_branin(
        model_path=args.model_path,
        tabpfn_path=args.tabpfn_path,
        n_runs=args.n_runs,
        max_steps=args.max_steps,
        n_init=args.n_init,
        seed=args.seed,
        device=device,
        save_dir=args.save_dir,
    )
