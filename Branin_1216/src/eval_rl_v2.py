"""
评估双塔 PPO 策略在 Branin 函数上的性能
对比方法：TabPFN+RL(v2), GP+RL(v2), GP+EI, TabPFN+EI
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from train_rl_v2 import (
    DualTowerCandidateSelector,
    BraninVariantFunction,
)
from select_candidates import (
    SelectionConfig,
    BraninFunction,
    select_candidates,
    compute_ei,
    compute_std_from_tabpfn_output,
)
from tabpfn import TabPFNRegressor


class RLAgentV2:
    """加载双塔 RL 策略"""
    
    def __init__(self, model_path: str, coord_dim: int = 2, hidden_dim: int = 64,
                 n_self_attn_layers: int = 2, n_cross_attn_layers: int = 2,
                 n_heads: int = 4, device: str = "cpu"):
        self.device = device
        self.coord_dim = coord_dim
        
        self.policy = DualTowerCandidateSelector(
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            n_self_attn_layers=n_self_attn_layers,
            n_cross_attn_layers=n_cross_attn_layers,
            n_heads=n_heads,
        ).to(device)
        
        # 加载模型
        state_dict = torch.load(model_path, map_location=device)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        print(f"已加载双塔模型: {model_path}")
    
    def select(
        self, 
        X_context: np.ndarray, 
        y_context: np.ndarray,
        X_candidates: np.ndarray, 
        pred_mean: np.ndarray, 
        pred_std: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray]
    ) -> int:
        """选择候选点"""
        context_feat, candidate_feat = self._build_features(
            X_context, y_context, X_candidates, pred_mean, pred_std, bounds
        )
        
        context_feat = context_feat.unsqueeze(0).to(self.device)
        candidate_feat = candidate_feat.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, _ = self.policy(context_feat, candidate_feat)
            action = torch.argmax(logits, dim=-1).item()
        
        return action
    
    def _build_features(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_candidates: np.ndarray,
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建归一化特征"""
        lower, upper = bounds
        
        # Context 特征
        X_ctx_norm = (X_context - lower) / (upper - lower + 1e-8)
        y_min, y_max = y_context.min(), y_context.max()
        if y_max - y_min > 1e-8:
            y_ctx_norm = (y_context - y_min) / (y_max - y_min)
        else:
            y_ctx_norm = np.zeros_like(y_context)
        
        context_feat = np.concatenate([
            X_ctx_norm, 
            y_ctx_norm.reshape(-1, 1)
        ], axis=-1)
        
        # Candidate 特征
        X_cand_norm = (X_candidates - lower) / (upper - lower + 1e-8)
        
        y_best = y_context.min()
        mean_rel = pred_mean - y_best
        mean_std_val = mean_rel.std()
        if mean_std_val > 1e-8:
            mean_norm = (mean_rel - mean_rel.mean()) / mean_std_val
        else:
            mean_norm = mean_rel - mean_rel.mean()
        
        std_std_val = pred_std.std()
        if std_std_val > 1e-8:
            std_norm = (pred_std - pred_std.mean()) / std_std_val
        else:
            std_norm = pred_std - pred_std.mean()
        
        candidate_feat = np.concatenate([
            X_cand_norm,
            mean_norm.reshape(-1, 1),
            std_norm.reshape(-1, 1)
        ], axis=-1)
        
        return (
            torch.FloatTensor(context_feat),
            torch.FloatTensor(candidate_feat)
        )


def get_gp_predictions(X_context: np.ndarray, y_context: np.ndarray, 
                       X_candidates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """使用 GP 进行预测"""
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    gp.fit(X_context, y_context)
    mean, std = gp.predict(X_candidates, return_std=True)
    return mean, std


def get_tabpfn_predictions(regressor, X_context: np.ndarray, y_context: np.ndarray,
                           X_candidates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """使用 TabPFN 进行预测"""
    regressor.fit(X_context, y_context)
    full_out = regressor.predict(X_candidates, output_type="full")
    pred_mean = full_out.get("mean", None)
    pred_std = compute_std_from_tabpfn_output(full_out)
    return pred_mean, pred_std


def run_tabpfn_rl_v2(func, agent, tabpfn_path: str, X_context: np.ndarray, 
                     y_context: np.ndarray, max_steps: int, rng, device: str,
                     n_candidates: int = 128) -> Tuple[List[float], np.ndarray]:
    """TabPFN + RL(v2) 双塔方法"""
    X_context = X_context.copy()
    y_context = y_context.copy()
    best_y = float(y_context.min())
    global_min = func.optimal_value
    lower, upper = func.bounds
    bounds = (lower, upper)
    
    # 初始化 TabPFN
    regressor = TabPFNRegressor(
        device=device,
        n_estimators=1,
        random_state=42,
        inference_precision=torch.float32,
        ignore_pretraining_limits=True,
        model_path=tabpfn_path,
    )
    
    regrets = [best_y - global_min]
    trajectory = [X_context.copy()]
    
    for step in range(max_steps):
        # Sobol 采样候选点
        sobol_seed = int(rng.integers(0, 100000))
        sobol_sampler = Sobol(d=func.dim, scramble=True, seed=sobol_seed)
        sobol_samples = sobol_sampler.random(n_candidates)
        X_candidates = (sobol_samples * (upper - lower) + lower).astype(np.float32)
        
        # TabPFN 预测
        pred_mean, pred_std = get_tabpfn_predictions(regressor, X_context, y_context, X_candidates)
        
        # RL 选择
        action = agent.select(X_context, y_context, X_candidates, pred_mean, pred_std, bounds)
        x_new = X_candidates[action:action+1]
        y_new = func(x_new)[0]
        
        X_context = np.vstack([X_context, x_new])
        y_context = np.concatenate([y_context, [y_new]])
        best_y = min(best_y, float(y_new))
        regrets.append(best_y - global_min)
        trajectory.append(x_new.copy())
    
    return regrets, np.vstack(trajectory)


def run_gp_rl_v2(func, agent, X_context: np.ndarray, y_context: np.ndarray, 
                 max_steps: int, rng, n_candidates: int = 128) -> Tuple[List[float], np.ndarray]:
    """GP + RL(v2) 双塔方法"""
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
        sobol_seed = int(rng.integers(0, 100000))
        sobol_sampler = Sobol(d=func.dim, scramble=True, seed=sobol_seed)
        sobol_samples = sobol_sampler.random(n_candidates)
        X_candidates = (sobol_samples * (upper - lower) + lower).astype(np.float32)
        
        # GP 预测
        gp_mean, gp_std = get_gp_predictions(X_context, y_context, X_candidates)
        
        # RL 选择
        action = agent.select(X_context, y_context, X_candidates, gp_mean, gp_std, bounds)
        x_new = X_candidates[action:action+1]
        y_new = func(x_new)[0]
        
        X_context = np.vstack([X_context, x_new])
        y_context = np.concatenate([y_context, [y_new]])
        best_y = min(best_y, float(y_new))
        regrets.append(best_y - global_min)
        trajectory.append(x_new.copy())
    
    return regrets, np.vstack(trajectory)


def run_gp_ei(func, X_context: np.ndarray, y_context: np.ndarray, 
              max_steps: int, rng, n_candidates: int = 128) -> Tuple[List[float], np.ndarray]:
    """GP + EI (标准贝叶斯优化)"""
    X_context = X_context.copy()
    y_context = y_context.copy()
    best_y = float(y_context.min())
    global_min = func.optimal_value
    lower, upper = func.bounds
    
    regrets = [best_y - global_min]
    trajectory = [X_context.copy()]
    
    for step in range(max_steps):
        sobol_seed = int(rng.integers(0, 100000))
        sobol_sampler = Sobol(d=func.dim, scramble=True, seed=sobol_seed)
        sobol_samples = sobol_sampler.random(n_candidates)
        X_candidates = (sobol_samples * (upper - lower) + lower).astype(np.float32)
        
        gp_mean, gp_std = get_gp_predictions(X_context, y_context, X_candidates)
        
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
                  n_candidates: int = 128) -> Tuple[List[float], np.ndarray]:
    """TabPFN + EI"""
    X_context = X_context.copy()
    y_context = y_context.copy()
    best_y = float(y_context.min())
    global_min = func.optimal_value
    lower, upper = func.bounds
    
    regressor = TabPFNRegressor(
        device=device,
        n_estimators=1,
        random_state=42,
        inference_precision=torch.float32,
        ignore_pretraining_limits=True,
        model_path=tabpfn_path,
    )
    
    regrets = [best_y - global_min]
    trajectory = [X_context.copy()]
    
    for step in range(max_steps):
        sobol_seed = int(rng.integers(0, 100000))
        sobol_sampler = Sobol(d=func.dim, scramble=True, seed=sobol_seed)
        sobol_samples = sobol_sampler.random(n_candidates)
        X_candidates = (sobol_samples * (upper - lower) + lower).astype(np.float32)
        
        pred_mean, pred_std = get_tabpfn_predictions(regressor, X_context, y_context, X_candidates)
        
        ei_values = compute_ei(pred_mean, pred_std, y_context.min(), xi=0.01)
        best_idx = np.argmax(ei_values)
        
        x_new = X_candidates[best_idx:best_idx+1]
        y_new = func(x_new)[0]
        
        X_context = np.vstack([X_context, x_new])
        y_context = np.concatenate([y_context, [y_new]])
        best_y = min(best_y, float(y_new))
        regrets.append(best_y - global_min)
        trajectory.append(x_new.copy())
    
    return regrets, np.vstack(trajectory)


def run_random(func, X_context: np.ndarray, y_context: np.ndarray, 
               max_steps: int, rng) -> Tuple[List[float], np.ndarray]:
    """随机采样基线"""
    X_context = X_context.copy()
    y_context = y_context.copy()
    best_y = float(y_context.min())
    global_min = func.optimal_value
    lower, upper = func.bounds
    
    regrets = [best_y - global_min]
    trajectory = [X_context.copy()]
    
    for step in range(max_steps):
        x_new = rng.uniform(lower, upper, size=(1, func.dim)).astype(np.float32)
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
    n_candidates: int = 128,
    seed: int = 42,
    device: str = "cpu",
    save_dir: str = "./results_v2",
    # 网络参数
    hidden_dim: int = 64,
    n_self_attn_layers: int = 2,
    n_cross_attn_layers: int = 2,
    n_heads: int = 4,
):
    """在标准 Branin 函数上评估所有方法"""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Comparison on Branin Function (v2 Dual-Tower)")
    print(f"{'='*60}")
    print(f"Methods: TabPFN+RL(v2), GP+RL(v2), GP+EI, TabPFN+EI, Random")
    print(f"Runs: {n_runs}, Steps: {max_steps}, Init: {n_init}, Candidates: {n_candidates}")
    print(f"{'='*60}\n")
    
    # 加载双塔 RL 策略
    agent = RLAgentV2(
        model_path, 
        coord_dim=2, 
        hidden_dim=hidden_dim,
        n_self_attn_layers=n_self_attn_layers,
        n_cross_attn_layers=n_cross_attn_layers,
        n_heads=n_heads,
        device=device
    )
    
    # 创建 Branin 函数
    func = BraninFunction()
    global_min = func.optimal_value
    lower, upper = func.bounds
    
    print(f"Branin global minimum: {global_min:.6f}")
    
    # 存储结果
    results = {
        "TabPFN+RL(v2)": {"regrets": [], "trajectories": []},
        "GP+RL(v2)": {"regrets": [], "trajectories": []},
        "GP+EI": {"regrets": [], "trajectories": []},
        "TabPFN+EI": {"regrets": [], "trajectories": []},
        "Random": {"regrets": [], "trajectories": []},
    }
    
    rng = np.random.default_rng(seed)
    
    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}")
        run_seed = int(rng.integers(0, 100000))
        
        # 相同的初始点
        init_rng = np.random.default_rng(run_seed)
        X_init = init_rng.uniform(lower, upper, size=(n_init, func.dim)).astype(np.float32)
        y_init = func(X_init)
        
        # TabPFN + RL(v2)
        rng1 = np.random.default_rng(run_seed)
        regrets, traj = run_tabpfn_rl_v2(func, agent, tabpfn_path, X_init, y_init, 
                                          max_steps, rng1, device, n_candidates)
        results["TabPFN+RL(v2)"]["regrets"].append(regrets)
        results["TabPFN+RL(v2)"]["trajectories"].append(traj)
        print(f"  TabPFN+RL(v2): final regret = {regrets[-1]:.6f}")
        
        # GP + RL(v2)
        rng2 = np.random.default_rng(run_seed)
        regrets, traj = run_gp_rl_v2(func, agent, X_init, y_init, max_steps, rng2, n_candidates)
        results["GP+RL(v2)"]["regrets"].append(regrets)
        results["GP+RL(v2)"]["trajectories"].append(traj)
        print(f"  GP+RL(v2): final regret = {regrets[-1]:.6f}")
        
        # GP + EI
        rng3 = np.random.default_rng(run_seed)
        regrets, traj = run_gp_ei(func, X_init, y_init, max_steps, rng3, n_candidates)
        results["GP+EI"]["regrets"].append(regrets)
        results["GP+EI"]["trajectories"].append(traj)
        print(f"  GP+EI: final regret = {regrets[-1]:.6f}")
        
        # TabPFN + EI
        rng4 = np.random.default_rng(run_seed)
        regrets, traj = run_tabpfn_ei(func, tabpfn_path, X_init, y_init, 
                                       max_steps, rng4, device, n_candidates)
        results["TabPFN+EI"]["regrets"].append(regrets)
        results["TabPFN+EI"]["trajectories"].append(traj)
        print(f"  TabPFN+EI: final regret = {regrets[-1]:.6f}")
        
        # Random
        rng5 = np.random.default_rng(run_seed)
        regrets, traj = run_random(func, X_init, y_init, max_steps, rng5)
        results["Random"]["regrets"].append(regrets)
        results["Random"]["trajectories"].append(traj)
        print(f"  Random: final regret = {regrets[-1]:.6f}")
    
    # 绘制对比曲线
    plot_comparison(results, max_steps, save_dir)
    
    # 绘制轨迹图
    plot_trajectories(func, results, save_dir, n_init)
    
    # 保存结果
    save_results(results, save_dir)
    
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
        "TabPFN+RL(v2)": "tab:blue",
        "GP+RL(v2)": "tab:orange", 
        "GP+EI": "tab:green",
        "TabPFN+EI": "tab:red",
        "Random": "tab:gray",
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
    plt.title("Comparison on Branin Function (Dual-Tower v2)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison_curve.png", dpi=150)
    plt.savefig(f"{save_dir}/comparison_curve.pdf")
    plt.close()
    print(f"\nSaved comparison curve to {save_dir}/comparison_curve.png")


def plot_trajectories(func, results: Dict, save_dir: str, n_init: int = 2):
    """绘制轨迹图"""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # 创建 Branin 等高线背景
    lower, upper = func.bounds
    x1 = np.linspace(lower[0], upper[0], 100)
    x2 = np.linspace(lower[1], upper[1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = func(np.array([[X1[i, j], X2[i, j]]]))[0]
    
    methods = ["TabPFN+RL(v2)", "GP+RL(v2)", "GP+EI", "TabPFN+EI", "Random"]
    
    for ax, method in zip(axes, methods):
        contour = ax.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.8)
        ax.contour(X1, X2, Z, levels=20, colors='white', alpha=0.3, linewidths=0.5)
        
        traj = results[method]["trajectories"][0]
        
        ax.scatter(traj[:n_init, 0], traj[:n_init, 1], c='white', s=100, 
                   marker='o', edgecolors='black', linewidths=1.5, label='Init', zorder=5)
        
        ax.scatter(traj[n_init:, 0], traj[n_init:, 1], c='red', s=50, 
                   marker='x', linewidths=1.5, label='Query', zorder=5)
        
        ax.plot(traj[:, 0], traj[:, 1], 'r--', alpha=0.5, linewidth=1)
        
        # 全局最优点
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


def save_results(results: Dict, save_dir: str):
    """保存结果"""
    import json
    
    # 保存 regrets
    regrets_dict = {}
    for method, data in results.items():
        regrets_dict[method] = [list(r) for r in data["regrets"]]
    
    with open(f"{save_dir}/regrets.json", "w") as f:
        json.dump(regrets_dict, f, indent=2)
    
    # 保存为 npz
    np.savez(
        f"{save_dir}/results.npz",
        **{f"{m}_regrets": np.array(d["regrets"]) for m, d in results.items()}
    )
    
    print(f"Saved results to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dual-Tower RL policy on Branin")
    parser.add_argument("--model_path", type=str, 
                        default="/gpfs/radev/pi/cohan/yz979/lin/test/Branin_1216/runs/ppo_bo_1216v1/ppo_final.pt",
                        help="PPO model path")
    parser.add_argument("--tabpfn_path", type=str, 
                        default="./model/finetuned_tabpfn_branin_family.ckpt",
                        help="TabPFN model path")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--max_steps", type=int, default=20, help="Max steps per run")
    parser.add_argument("--n_init", type=int, default=2, help="Initial context points")
    parser.add_argument("--n_candidates", type=int, default=128, help="Number of candidates")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="./results_v2", help="Save directory")
    # 网络参数
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_self_attn_layers", type=int, default=2)
    parser.add_argument("--n_cross_attn_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = evaluate_on_branin(
        model_path=args.model_path,
        tabpfn_path=args.tabpfn_path,
        n_runs=args.n_runs,
        max_steps=args.max_steps,
        n_init=args.n_init,
        n_candidates=args.n_candidates,
        seed=args.seed,
        device=device,
        save_dir=args.save_dir,
        hidden_dim=args.hidden_dim,
        n_self_attn_layers=args.n_self_attn_layers,
        n_cross_attn_layers=args.n_cross_attn_layers,
        n_heads=args.n_heads,
    )
