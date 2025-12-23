"""
评估双塔 PPO 策略在 **训练分布内** 的 Branin 变体上的性能
目的：验证 RL 策略是否在训练分布内有效
"""
import os
import json
import argparse
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
        lower, upper = bounds
        
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
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    gp.fit(X_context, y_context)
    mean, std = gp.predict(X_candidates, return_std=True)
    return mean, std


def get_tabpfn_predictions(regressor, X_context: np.ndarray, y_context: np.ndarray,
                           X_candidates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    regressor.fit(X_context, y_context)
    full_out = regressor.predict(X_candidates, output_type="full")
    pred_mean = full_out.get("mean", None)
    pred_std = compute_std_from_tabpfn_output(full_out)
    return pred_mean, pred_std


def run_tabpfn_rl_v2(func, agent, tabpfn_path: str, X_context: np.ndarray, 
                     y_context: np.ndarray, max_steps: int, rng, device: str,
                     n_candidates: int = 128) -> List[float]:
    X_context = X_context.copy()
    y_context = y_context.copy()
    best_y = float(y_context.min())
    global_min = func.optimal_value
    lower, upper = func.bounds
    bounds = (lower, upper)
    
    regressor = TabPFNRegressor(
        device=device,
        n_estimators=1,
        random_state=42,
        inference_precision=torch.float32,
        ignore_pretraining_limits=True,
        model_path=tabpfn_path,
    )
    
    regrets = [best_y - global_min]
    
    for step in range(max_steps):
        sobol_seed = int(rng.integers(0, 100000))
        sobol_sampler = Sobol(d=func.dim, scramble=True, seed=sobol_seed)
        sobol_samples = sobol_sampler.random(n_candidates)
        X_candidates = (sobol_samples * (upper - lower) + lower).astype(np.float32)
        
        pred_mean, pred_std = get_tabpfn_predictions(regressor, X_context, y_context, X_candidates)
        
        action = agent.select(X_context, y_context, X_candidates, pred_mean, pred_std, bounds)
        x_new = X_candidates[action:action+1]
        y_new = func(x_new)[0]
        
        X_context = np.vstack([X_context, x_new])
        y_context = np.concatenate([y_context, [y_new]])
        best_y = min(best_y, float(y_new))
        regrets.append(best_y - global_min)
    
    return regrets


def run_gp_ei(func, X_context: np.ndarray, y_context: np.ndarray, 
              max_steps: int, rng, n_candidates: int = 128) -> List[float]:
    X_context = X_context.copy()
    y_context = y_context.copy()
    best_y = float(y_context.min())
    global_min = func.optimal_value
    lower, upper = func.bounds
    
    regrets = [best_y - global_min]
    
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
    
    return regrets


def run_random(func, X_context: np.ndarray, y_context: np.ndarray, 
               max_steps: int, rng) -> List[float]:
    X_context = X_context.copy()
    y_context = y_context.copy()
    best_y = float(y_context.min())
    global_min = func.optimal_value
    lower, upper = func.bounds
    
    regrets = [best_y - global_min]
    
    for step in range(max_steps):
        x_new = rng.uniform(lower, upper, size=(1, func.dim)).astype(np.float32)
        y_new = func(x_new)[0]
        
        X_context = np.vstack([X_context, x_new])
        y_context = np.concatenate([y_context, [y_new]])
        best_y = min(best_y, float(y_new))
        regrets.append(best_y - global_min)
    
    return regrets


def evaluate_in_distribution(
    model_path: str,
    tabpfn_path: str,
    variants_path: str,
    n_runs: int = 10,
    max_steps: int = 20,
    n_init: int = 2,
    n_candidates: int = 128,
    seed: int = 42,
    device: str = "cpu",
    save_dir: str = "./results_v2_indist",
    hidden_dim: int = 64,
    n_self_attn_layers: int = 2,
    n_cross_attn_layers: int = 2,
    n_heads: int = 4,
):
    """在训练分布内的 Branin 变体上评估"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载变体
    data = np.load(variants_path, allow_pickle=True)
    variants = data["variants"].tolist()
    
    print(f"\n{'='*60}")
    print("In-Distribution Evaluation (Training Variants)")
    print(f"{'='*60}")
    print(f"Methods: TabPFN+RL(v2), GP+EI, Random")
    print(f"Variants: {len(variants)}, Runs per variant: {n_runs}")
    print(f"Steps: {max_steps}, Init: {n_init}, Candidates: {n_candidates}")
    print(f"{'='*60}\n")
    
    # 加载 RL 策略
    agent = RLAgentV2(
        model_path, 
        coord_dim=2, 
        hidden_dim=hidden_dim,
        n_self_attn_layers=n_self_attn_layers,
        n_cross_attn_layers=n_cross_attn_layers,
        n_heads=n_heads,
        device=device
    )
    
    results = {
        "TabPFN+RL(v2)": {"regrets": []},
        "GP+EI": {"regrets": []},
        "Random": {"regrets": []},
    }
    
    rng = np.random.default_rng(seed)
    
    # 对每个变体测试
    n_test_variants = min(10, len(variants))  # 测试前10个变体
    
    for var_idx in range(n_test_variants):
        variant_params = variants[var_idx]
        func = BraninVariantFunction(variant_params, device)
        lower, upper = func.bounds
        global_min = func.optimal_value
        
        print(f"\nVariant {var_idx+1}/{n_test_variants}: global_min={global_min:.4f}")
        print(f"  Params: dx1={variant_params['dx1']:.2f}, dx2={variant_params['dx2']:.2f}, beta={variant_params['beta']:.2f}")
        
        var_results = {"TabPFN+RL(v2)": [], "GP+EI": [], "Random": []}
        
        for run in range(n_runs):
            run_seed = int(rng.integers(0, 100000))
            
            # 相同初始点
            init_rng = np.random.default_rng(run_seed)
            X_init = init_rng.uniform(lower, upper, size=(n_init, func.dim)).astype(np.float32)
            y_init = func(X_init)
            
            # TabPFN + RL(v2)
            rng1 = np.random.default_rng(run_seed)
            regrets = run_tabpfn_rl_v2(func, agent, tabpfn_path, X_init, y_init, 
                                       max_steps, rng1, device, n_candidates)
            var_results["TabPFN+RL(v2)"].append(regrets)
            
            # GP + EI
            rng2 = np.random.default_rng(run_seed)
            regrets = run_gp_ei(func, X_init, y_init, max_steps, rng2, n_candidates)
            var_results["GP+EI"].append(regrets)
            
            # Random
            rng3 = np.random.default_rng(run_seed)
            regrets = run_random(func, X_init, y_init, max_steps, rng3)
            var_results["Random"].append(regrets)
        
        # 汇总该变体结果
        for method in results:
            results[method]["regrets"].extend(var_results[method])
            final_regrets = [r[-1] for r in var_results[method]]
            print(f"  {method}: {np.mean(final_regrets):.4f} +/- {np.std(final_regrets):.4f}")
    
    # 绘图
    plot_comparison_indist(results, max_steps, save_dir)
    
    # 总结
    print(f"\n{'='*60}")
    print("Overall Results (In-Distribution)")
    print(f"{'='*60}")
    for method, data in results.items():
        all_regrets = np.array(data["regrets"])
        final_regrets = all_regrets[:, -1]
        print(f"{method}:")
        print(f"  Final Regret: {np.mean(final_regrets):.4f} +/- {np.std(final_regrets):.4f}")
        print(f"  Mean Regret@5: {np.mean(all_regrets[:, 5]):.4f}")
        print(f"  Mean Regret@10: {np.mean(all_regrets[:, 10]):.4f}")
    
    return results


def plot_comparison_indist(results: Dict, max_steps: int, save_dir: str):
    """绘制对比曲线"""
    plt.figure(figsize=(10, 6))
    
    colors = {
        "TabPFN+RL(v2)": "tab:blue",
        "GP+EI": "tab:green",
        "Random": "tab:gray",
    }
    
    x = np.arange(max_steps + 1)
    
    for method, data in results.items():
        regrets = np.array(data["regrets"])
        mean = np.mean(regrets, axis=0)
        std = np.std(regrets, axis=0)
        
        plt.plot(x, mean, label=method, color=colors[method], linewidth=2)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=colors[method])
    
    plt.xlabel("Number of Function Evaluations", fontsize=12)
    plt.ylabel("Simple Regret", fontsize=12)
    plt.title("In-Distribution Evaluation (Training Variants)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison_indist.png", dpi=150)
    plt.savefig(f"{save_dir}/comparison_indist.pdf")
    plt.close()
    print(f"\nSaved comparison curve to {save_dir}/comparison_indist.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, 
                        default="/gpfs/radev/pi/cohan/yz979/lin/test/Branin_1216/runs/ppo_bo_1216v1/ppo_final.pt")
    parser.add_argument("--tabpfn_path", type=str, 
                        default="./model/finetuned_tabpfn_branin_family.ckpt")
    parser.add_argument("--variants_path", type=str,
                        default="./data/branin_family_variants.npz")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--n_init", type=int, default=2)
    parser.add_argument("--n_candidates", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./results_v2_indist")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_self_attn_layers", type=int, default=2)
    parser.add_argument("--n_cross_attn_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    evaluate_in_distribution(
        model_path=args.model_path,
        tabpfn_path=args.tabpfn_path,
        variants_path=args.variants_path,
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
