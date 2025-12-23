import os
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from tabpfn import TabPFNRegressor

plt.switch_backend("agg")


# =========================================================
# 1. Goldstein-Price 及其变体
# =========================================================

def goldstein_price_np(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    标准 Goldstein-Price 函数，定义域通常为 x1, x2 ∈ [-2, 2]
    全局最小值在 (0, -1)，值为 3
    """
    part1 = 1 + (x1 + x2 + 1) ** 2 * (
        19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2
    )
    part2 = 30 + (2 * x1 - 3 * x2) ** 2 * (
        18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2
    )
    return (part1 * part2).astype(np.float32)


def goldstein_price_family_torch(
    x: torch.Tensor,
    dx1: float = 0.0,
    dx2: float = 0.0,
    sx1: float = 1.0,
    sx2: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> torch.Tensor:
    """
    Goldstein-Price 变体族：在输入上做线性变换 (sx, dx)，在输出上做仿射变换 (alpha, beta)。
    x: (..., 2) 的 Tensor，最后一维为 [x1, x2]
    """
    x1 = x[..., 0]
    x2 = x[..., 1]

    # 输入平移 + 缩放
    x1_t = sx1 * x1 + dx1
    x2_t = sx2 * x2 + dx2

    # Goldstein-Price 主体
    part1 = 1 + (x1_t + x2_t + 1) ** 2 * (
        19 - 14 * x1_t + 3 * x1_t ** 2 - 14 * x2_t + 6 * x1_t * x2_t + 3 * x2_t ** 2
    )
    part2 = 30 + (2 * x1_t - 3 * x2_t) ** 2 * (
        18 - 32 * x1_t + 12 * x1_t ** 2 + 48 * x2_t - 36 * x1_t * x2_t + 27 * x2_t ** 2
    )
    y = part1 * part2

    # 输出仿射变换
    y = alpha * y + beta
    return y


def goldstein_price_family_numpy_from_params(X, variant_params, device="cpu"):
    x_tensor = torch.from_numpy(X).float().unsqueeze(0).to(device)
    with torch.no_grad():
        y = goldstein_price_family_torch(x_tensor, **variant_params).cpu().numpy()[0]
    return y.astype(np.float32)


# =========================================================
# 2. RL 环境 (Goldstein-Price)
# =========================================================

class TabPFN_GoldsteinPriceEnv:
    def __init__(
        self,
        tabpfn_model_path: str,
        variants_path: str,
        grid_size: int = 32,
        device: str = "cpu",
        max_steps: int = 20,
        seed: int = 42,
    ):
        self.device = device
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # 加载 TabPFN
        self.tabpfn = TabPFNRegressor(
            model_path=tabpfn_model_path,
            device=self.device,
            n_estimators=1,
            random_state=seed,
            inference_precision=torch.float32,
            fit_mode="batched",
            differentiable_input=False,
        )

        # 加载变体
        data = np.load(variants_path, allow_pickle=True)
        self.variants = data["variants"].tolist()
        self.num_variants = len(self.variants)

        # Goldstein-Price 定义域: x1, x2 ∈ [-2, 2]
        self.x1_min, self.x1_max = -2.0, 2.0
        self.x2_min, self.x2_max = -2.0, 2.0

        x1 = np.linspace(self.x1_min, self.x1_max, self.grid_size)
        x2 = np.linspace(self.x2_min, self.x2_max, self.grid_size)
        self.X1, self.X2 = np.meshgrid(x1, x2)
        self.grid_points = np.stack([self.X1.flatten(), self.X2.flatten()], axis=1).astype(np.float32)

        self.observation_space_shape = (2, grid_size, grid_size)
        self.action_space_shape = (2,)

        # Episode 状态
        self.current_variant = None
        self.global_min_y = None
        self.X_ctx = None
        self.y_ctx = None
        self.best_y = None
        self.step_count = 0
        
        # 缓存当前变体在网格上的真值
        self.true_y_grid = None

    def _find_global_min(self, variant_params):
        y_grid = goldstein_price_family_numpy_from_params(self.grid_points, variant_params, device=self.device)
        min_idx = int(np.argmin(y_grid))
        x0 = self.grid_points[min_idx]

        def func(x):
            x = np.array(x, dtype=np.float32).reshape(1, 2)
            return float(goldstein_price_family_numpy_from_params(x, variant_params, device=self.device)[0])

        bounds = [(self.x1_min, self.x1_max), (self.x2_min, self.x2_max)]
        res = minimize(func, x0, bounds=bounds, method="L-BFGS-B")
        return float(res.fun)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_variant_idx = int(self.rng.integers(0, self.num_variants))
        self.current_variant = self.variants[self.current_variant_idx]
        self.global_min_y = self._find_global_min(self.current_variant)
        
        # 缓存真值网格
        self.true_y_grid = goldstein_price_family_numpy_from_params(
            self.grid_points, self.current_variant, device=self.device
        ).reshape(self.grid_size, self.grid_size)

        self.X_ctx = np.empty((0, 2), dtype=np.float32)
        self.y_ctx = np.empty((0,), dtype=np.float32)
        self.best_y = float("inf")
        self.step_count = 0

        obs = self._get_observation()
        return obs

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        x1 = (action[0] + 1) / 2 * (self.x1_max - self.x1_min) + self.x1_min
        x2 = (action[1] + 1) / 2 * (self.x2_max - self.x2_min) + self.x2_min
        x = np.array([[x1, x2]], dtype=np.float32)

        y_val = goldstein_price_family_numpy_from_params(x, self.current_variant, device=self.device)[0]

        self.X_ctx = np.concatenate([self.X_ctx, x], axis=0)
        self.y_ctx = np.concatenate([self.y_ctx, np.array([y_val], dtype=np.float32)])
        
        old_best = self.best_y
        self.best_y = min(self.best_y, float(y_val))
        self.step_count += 1

        # 计算奖励
        reward = self._compute_reward(y_val, old_best)

        done = self.step_count >= self.max_steps
        obs = self._get_observation()

        return obs, float(reward), done, {}

    def _compute_reward(self, y_new, old_best):
        """
        简化的奖励函数：
        Goldstein-Price 最小值为 3，最大值可能超过 1e6
        """
        current_best = self.best_y
        reward = 0.0
        
        # 接近全局最小值时给予大奖励
        if current_best - self.global_min_y < 1.0:
            reward = 50
        else:
            # print(current_best)
            # 使用 log scale 来处理 Goldstein-Price 的大范围值
            reward = -current_best / 1e4  # 值越小，奖励越高
        
        return reward

    def _compute_maps(self):
        """计算均值和不确定性 map"""
        n_ctx = self.X_ctx.shape[0]

        if n_ctx == 0:
            # 无观测：用先验
            mean_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            uncert_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
            return mean_map, uncert_map

        # 准备数据
        if n_ctx == 1:
            X_fit = np.repeat(self.X_ctx, 2, axis=0)
            y_fit_raw = np.repeat(self.y_ctx, 2, axis=0)
        else:
            X_fit = self.X_ctx
            y_fit_raw = self.y_ctx

        # Goldstein-Price 值范围：最小约 3，最大可能 > 1e6
        # 使用 log scale 归一化
        y_global_min = 0.0
        y_global_max = 1e6
        y_range = y_global_max - y_global_min
        
        y_fit_norm = (y_fit_raw - y_global_min) / y_range

        self.tabpfn.fit(X_fit, y_fit_norm)
        full_out = self.tabpfn.predict(self.grid_points, output_type="full")

        # 均值反归一化
        mean_norm = np.asarray(full_out["mean"]).reshape(-1)
        mean_pred = mean_norm * y_range + y_global_min
        mean_map = mean_pred.reshape(self.grid_size, self.grid_size).astype(np.float32)

        # 方差
        criterion = full_out.get("criterion", None)
        logits = full_out.get("logits", None)
        
        if criterion is not None and logits is not None:
            if not isinstance(logits, torch.Tensor):
                logits_t = torch.from_numpy(np.asarray(logits)).float()
            else:
                logits_t = logits.float()
            # 将 criterion 内部的参数也转换为 float32
            try:
                criterion = criterion.float()
            except AttributeError:
                pass  # 如果 criterion 没有 .float() 方法，则跳过  
            with torch.no_grad():
                var_t = criterion.variance(logits_t)
            if isinstance(var_t, torch.Tensor):
                var = var_t.detach().cpu().numpy().reshape(-1)
            else:
                var = np.asarray(var_t).reshape(-1)
            
            var = var * (y_range ** 2)
            uncert_map = np.sqrt(np.maximum(var, 1e-6))
            uncert_map = uncert_map.reshape(self.grid_size, self.grid_size).astype(np.float32)
        else:
            dists = pairwise_distances(self.grid_points, self.X_ctx)
            min_dist = dists.min(axis=1)
            uncert_map = min_dist.reshape(self.grid_size, self.grid_size).astype(np.float32)

        return mean_map, uncert_map

    def _get_observation(self, return_raw=False):
        """
        构造观测：
        - 通道 0: 归一化的均值预测
        - 通道 1: 归一化的不确定性
        """
        mean_map, uncert_map = self._compute_maps()
        
        if return_raw:
            return (
                mean_map[None, :, :].astype(np.float32),
                uncert_map[None, :, :].astype(np.float32),
            )
        
        # Goldstein-Price 值范围调整
        mean_norm = (mean_map - 0.0) / 1e5
        mean_norm = np.clip(mean_norm, -1.0, 2.0)
        
        uncert_norm = uncert_map / 1e4
        uncert_norm = np.clip(uncert_norm, 0.0, 3.0)
        
        # 添加已采样点的位置信息
        if self.X_ctx.shape[0] > 0:
            for pt in self.X_ctx:
                i = int((pt[1] - self.x2_min) / (self.x2_max - self.x2_min) * (self.grid_size - 1))
                j = int((pt[0] - self.x1_min) / (self.x1_max - self.x1_min) * (self.grid_size - 1))
                i = np.clip(i, 0, self.grid_size - 1)
                j = np.clip(j, 0, self.grid_size - 1)
                uncert_norm[i, j] = -1.0
        
        obs = np.stack([mean_norm, uncert_norm], axis=0).astype(np.float32)
        return obs


# =========================================================
# 3. CNN Agent
# =========================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNAgent(nn.Module):
    def __init__(self, grid_size=32):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(2, 32, 3, 1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            layer_init(nn.Conv2d(32, 64, 3, 1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            layer_init(nn.Conv2d(64, 128, 3, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        
        self.feature_dim = 128 * 4 * 4
        
        self.fc = nn.Sequential(
            layer_init(nn.Linear(self.feature_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        )
        
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)
        self.actor_mean = layer_init(nn.Linear(128, 2), std=0.5)
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))

    def get_value(self, x):
        return self.critic(self.fc(self.network(x)))

    def get_action_and_value(self, x, action=None):
        feat = self.fc(self.network(x))
        action_mean = torch.tanh(self.actor_mean(feat))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        action_std = torch.clamp(action_std, min=0.1)
        
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(feat),
        )


# =========================================================
# 4. 可视化函数
# =========================================================

def visualize_episode(agent, test_env, save_dir, update_idx, device):
    """
    可视化一个完整 episode 的采样过程
    """
    obs = test_env.reset()

    raw_means = []
    raw_uncs = []
    history_x1 = []
    history_x2 = []

    # 收集初始状态
    raw_mean, raw_unc = test_env._get_observation(return_raw=True)
    raw_means.append(raw_mean[0])
    raw_uncs.append(raw_unc[0])

    done = False
    step_count = 0
    
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)

        real_action = action.cpu().numpy()[0]
        clipped_action = np.clip(real_action, -1.0, 1.0)

        # 转回物理坐标
        x1_phys = (clipped_action[0] + 1) / 2 * (
            test_env.x1_max - test_env.x1_min
        ) + test_env.x1_min
        x2_phys = (clipped_action[1] + 1) / 2 * (
            test_env.x2_max - test_env.x2_min
        ) + test_env.x2_min
        history_x1.append(x1_phys)
        history_x2.append(x2_phys)

        obs, _, done, _ = test_env.step(real_action)
        step_count += 1

        raw_mean, raw_unc = test_env._get_observation(return_raw=True)
        raw_means.append(raw_mean[0])
        raw_uncs.append(raw_unc[0])

    # 选择要绘制的步数
    steps_to_plot = [4, 8, 12, 16, 20]
    idxs_to_plot = [s - 1 for s in steps_to_plot if s <= len(history_x1)]
    
    if len(idxs_to_plot) == 0:
        print(f"[Visual] Episode 太短 ({len(history_x1)} steps)，跳过可视化")
        return

    fig, axes = plt.subplots(2, len(idxs_to_plot), figsize=(4.5 * len(idxs_to_plot), 9))
    
    if len(idxs_to_plot) == 1:
        axes = axes.reshape(2, 1)
    
    X, Y = test_env.X1, test_env.X2

    # 计算全局颜色范围（使用 log scale）
    all_means = np.array(raw_means)
    all_uncs = np.array(raw_uncs)
    
    # 对 mean 使用 log scale
    all_means_log = np.log10(np.maximum(all_means, 1.0))
    mean_vmin, mean_vmax = np.percentile(all_means_log, [2, 98])
    unc_vmax = np.percentile(all_uncs, 98)

    cf1 = cf2 = None

    for i, step_idx in enumerate(idxs_to_plot):
        cur_x1 = history_x1[: step_idx + 1]
        cur_x2 = history_x2[: step_idx + 1]

        # Row 1: mean map（使用 log scale）
        ax_mean = axes[0, i]
        mean_log = np.log10(np.maximum(raw_means[step_idx], 1.0))
        cf1 = ax_mean.contourf(
            X, Y, mean_log, 
            levels=np.linspace(mean_vmin, mean_vmax, 20),
            extend='both'
        )
        if len(cur_x1) > 1:
            ax_mean.scatter(cur_x1[:-1], cur_x2[:-1], c="white", marker="o", s=40, alpha=0.7, edgecolors='black', linewidths=0.5)
        ax_mean.scatter(cur_x1[-1], cur_x2[-1], c="red", marker="x", s=120, linewidths=3)
        ax_mean.set_title(f"Step {steps_to_plot[i]}\nMean Prediction (log10)", fontsize=12)
        ax_mean.set_xlim(test_env.x1_min, test_env.x1_max)
        ax_mean.set_ylim(test_env.x2_min, test_env.x2_max)
        if i == 0:
            ax_mean.set_ylabel("x2", fontsize=11)
        else:
            ax_mean.tick_params(left=False, labelleft=False)
        ax_mean.tick_params(bottom=False, labelbottom=False)

        # Row 2: uncertainty map
        ax_unc = axes[1, i]
        cf2 = ax_unc.contourf(
            X, Y, raw_uncs[step_idx],
            levels=np.linspace(0, unc_vmax, 20),
            extend='max'
        )
        ax_unc.scatter(cur_x1, cur_x2, c="white", marker="o", s=30, alpha=0.6, edgecolors='black', linewidths=0.5)
        ax_unc.set_title("Uncertainty (Std)", fontsize=12)
        ax_unc.set_xlim(test_env.x1_min, test_env.x1_max)
        ax_unc.set_ylim(test_env.x2_min, test_env.x2_max)
        ax_unc.set_xlabel("x1", fontsize=11)
        if i == 0:
            ax_unc.set_ylabel("x2", fontsize=11)
        else:
            ax_unc.tick_params(left=False, labelleft=False)

    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    
    if cf1 is not None:
        cax1 = fig.add_axes([0.93, 0.55, 0.015, 0.35])
        fig.colorbar(cf1, cax=cax1, label="Mean (log10)")
    if cf2 is not None:
        cax2 = fig.add_axes([0.93, 0.1, 0.015, 0.35])
        fig.colorbar(cf2, cax=cax2, label="Uncertainty")

    final_regret = max(1e-8, test_env.best_y - test_env.global_min_y)
    plt.suptitle(
        f"Update {update_idx}: Final Regret={final_regret:.2e} "
        f"(Log10={np.log10(final_regret):.2f}), Best_y={test_env.best_y:.4f}",
        fontsize=14,
        y=0.98,
    )

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vis_update_{update_idx:04d}.png")
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[Visual] 可视化已保存: {save_path}")


# =========================================================
# 5. PPO 训练
# =========================================================

def train_ppo(args):
    run_name = f"TabPFN_GoldsteinPrice_CNN_PPO_{int(time.time())}"
    vis_dir = f"visualizations/{run_name}"
    save_dir = f"runs/{run_name}"
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    env = TabPFN_GoldsteinPriceEnv(
        tabpfn_model_path=args.tabpfn_model_path,
        variants_path=args.variants_path,
        grid_size=args.grid_size,
        device=str(device),
        max_steps=args.max_steps,
        seed=args.seed,
    )
    
    # 创建独立的测试环境用于可视化
    test_env = TabPFN_GoldsteinPriceEnv(
        tabpfn_model_path=args.tabpfn_model_path,
        variants_path=args.variants_path,
        grid_size=args.grid_size,
        device=str(device),
        max_steps=args.max_steps,
        seed=args.seed + 1000,
    )

    agent = CNNAgent(grid_size=args.grid_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # 打印初始参数统计
    print("\n=== 初始参数统计 ===")
    print(f"actor_mean.weight: mean={agent.actor_mean.weight.mean():.4f}, std={agent.actor_mean.weight.std():.4f}")
    print(f"actor_mean.bias: {agent.actor_mean.bias.data}")
    print(f"actor_logstd: {agent.actor_logstd.data}")
    print("=" * 50)

    obs_buf = torch.zeros((args.num_steps, 2, args.grid_size, args.grid_size)).to(device)
    actions = torch.zeros((args.num_steps, 2)).to(device)
    logprobs = torch.zeros(args.num_steps).to(device)
    rewards = torch.zeros(args.num_steps).to(device)
    dones = torch.zeros(args.num_steps).to(device)
    values = torch.zeros(args.num_steps).to(device)

    global_step = 0
    next_obs = env.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_done = torch.zeros(1).to(device)

    num_updates = args.total_timesteps // args.num_steps

    for update in range(1, num_updates + 1):
        episode_rewards = []
        episode_regrets = []

        for step in range(args.num_steps):
            global_step += 1
            obs_buf[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            real_action = action.cpu().numpy()[0]
            next_obs_np, reward, done, _ = env.step(real_action)

            rewards[step] = torch.tensor(reward).to(device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
            next_done = torch.tensor([done], dtype=torch.float32).to(device)

            if done:
                final_regret = max(1e-20, env.best_y - env.global_min_y)
                episode_regrets.append(final_regret)
                episode_rewards.append(rewards[max(0, step-env.max_steps+1):step+1].sum().item())
                next_obs_np = env.reset()
                next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten
        b_obs = obs_buf.reshape((-1,) + env.observation_space_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_space_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO update
        b_inds = np.arange(args.num_steps)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.num_steps, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # 监控梯度和参数
        if update % 1 == 0:
            with torch.no_grad():
                actor_w = agent.actor_mean.weight
                actor_b = agent.actor_mean.bias
                print(f"\nUpdate {update}:")
                print(f"  actor_mean.weight: mean={actor_w.mean():.4f}, std={actor_w.std():.4f}, "
                      f"min={actor_w.min():.4f}, max={actor_w.max():.4f}")
                print(f"  actor_mean.bias: {actor_b.data.cpu().numpy()}")
                print(f"  actor_logstd: {agent.actor_logstd.data.cpu().numpy()}, "
                      f"std={torch.exp(agent.actor_logstd).data.cpu().numpy()}")
                
                avg_regret = np.mean(episode_regrets) if episode_regrets else float('inf')
                avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
                print(f"  Avg Regret: {avg_regret:.4f}, Avg Reward: {avg_reward:.2f}")
                print(f"  Loss: {loss.item():.4f}, Entropy: {entropy_loss.item():.4f}")

        # 保存模型
        if update % 5 == 0:
            save_path = os.path.join(save_dir, f"agent_{update}.pt")
            torch.save({
                "model_state_dict": agent.state_dict(),
                "config": vars(args)
            }, save_path)
            print(f"  Saved to {save_path}")
            
        # 可视化 episode
        if update % 10 == 0 or update == 1:
            agent.eval()
            visualize_episode(agent, test_env, vis_dir, update, device)
            agent.train()

    print(f"\n训练结束。模型保存在 {save_dir}/")
    print(f"可视化保存在 {vis_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabpfn_model_path", type=str, default="./model/finetuned_tabpfn_goldstein_price_family.ckpt")
    parser.add_argument("--variants_path", type=str, default="./data/goldstein_price_family_variants.npz")
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true", default=True)
    
    # 超参数
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_steps", type=int, default=400)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.05)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    args = parser.parse_args()
    train_ppo(args)