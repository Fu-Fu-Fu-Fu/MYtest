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
# 1. Branin 及其变体
# =========================================================

def branin_np(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return (
        a * (x2 - b * x1**2 + c * x1 - r) ** 2
        + s * (1 - t) * np.cos(x1)
        + s
    ).astype(np.float32)


def branin_family_torch(
    x: torch.Tensor,
    dx1: float = 0.0, dx2: float = 0.0,
    sx1: float = 1.0, sx2: float = 1.0,
    alpha: float = 1.0, beta: float = 0.0,
    a: float = 1.0, b: float = 5.1 / (4.0 * np.pi**2),
    c: float = 5.0 / np.pi, r: float = 6.0,
    s: float = 10.0, t: float = 1.0 / (8.0 * np.pi),
) -> torch.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]
    x1_t = sx1 * x1 + dx1
    x2_t = sx2 * x2 + dx2
    y = a * (x2_t - b * x1_t**2 + c * x1_t - r) ** 2 + s * (1.0 - t) * torch.cos(x1_t) + s
    return alpha * y + beta


def branin_family_numpy_from_params(X, variant_params, device="cpu"):
    x_tensor = torch.from_numpy(X).float().unsqueeze(0).to(device)
    with torch.no_grad():
        y = branin_family_torch(x_tensor, **variant_params).cpu().numpy()[0]
    return y.astype(np.float32)


# =========================================================
# 2. 修复版 RL 环境
# =========================================================

class TabPFN_BraninEnv:
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

        # 定义域
        self.x1_min, self.x1_max = -5.0, 10.0
        self.x2_min, self.x2_max = 0.0, 15.0

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
        
        # === 关键修复：缓存当前变体在网格上的真值 ===
        self.true_y_grid = None

    def _find_global_min(self, variant_params):
        y_grid = branin_family_numpy_from_params(self.grid_points, variant_params, device=self.device)
        min_idx = int(np.argmin(y_grid))
        x0 = self.grid_points[min_idx]

        def func(x):
            x = np.array(x, dtype=np.float32).reshape(1, 2)
            return float(branin_family_numpy_from_params(x, variant_params, device=self.device)[0])

        bounds = [(self.x1_min, self.x1_max), (self.x2_min, self.x2_max)]
        res = minimize(func, x0, bounds=bounds, method="L-BFGS-B")
        return float(res.fun)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_variant_idx = int(self.rng.integers(0, self.num_variants))
        self.current_variant = self.variants[self.current_variant_idx]
        self.global_min_y = self._find_global_min(self.current_variant)
        
        # === 缓存真值网格 ===
        self.true_y_grid = branin_family_numpy_from_params(
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

        y_val = branin_family_numpy_from_params(x, self.current_variant, device=self.device)[0]

        self.X_ctx = np.concatenate([self.X_ctx, x], axis=0)
        self.y_ctx = np.concatenate([self.y_ctx, np.array([y_val], dtype=np.float32)])
        
        old_best = self.best_y
        self.best_y = min(self.best_y, float(y_val))
        self.step_count += 1

        # === 关键修复：改进 reward 设计 ===
        reward = self._compute_reward(y_val, old_best)

        done = self.step_count >= self.max_steps
        obs = self._get_observation()

        return obs, float(reward), done, {}


    def _compute_reward(self, y_new, old_best):
        """
        简化的奖励函数：
        1. 奖励 = -当前最小值（鼓励越早找到越小的值）
        2. 边界惩罚
        
        由于 Branin 最小值约 0.4，最大值约 300+，我们对奖励进行缩放
        """
        # === 核心奖励：当前最小值的负数 ===
        # Branin 值范围大约 [0.4, 300]，进行归一化
        # 使用 log scale 来压缩范围，使奖励更稳定
        current_best = self.best_y
        
        # 方案1：直接使用负值并缩放
        # reward = -current_best / 50.0  # 缩放到合理范围 [-6, ~0]
        # reward = 0.0
        # 方案2：使用 log scale（推荐，因为 Branin 值范围很大）
        # 加 1 防止 log(0)，然后取负
        # if current_best - self.global_min_y < 1.0:
        #     reward = 50
        # else:
        #     reward = - current_best  # best=0.4 -> -0.34, best=100 -> -4.6
        regret = max(1e-20, self.best_y - self.global_min_y)
        reward = -np.log10(regret)
        if regret < 1.0:
            reward += 5.0
        # === 边界惩罚 ===
        # if self.X_ctx.shape[0] > 0:
        #     last_x = self.X_ctx[-1]
        #     x1_norm = (last_x[0] - self.x1_min) / (self.x1_max - self.x1_min)
        #     x2_norm = (last_x[1] - self.x2_min) / (self.x2_max - self.x2_min)
            
        #     # 计算到边界的最小距离
        #     min_margin = min(x1_norm, 1 - x1_norm, x2_norm, 1 - x2_norm)
            
        #     # 边界惩罚：距离边界 5% 以内时惩罚
        #     boundary_threshold = 0.05
        #     if min_margin < boundary_threshold:
        #         # 惩罚与距离边界的程度成正比
        #         boundary_penalty = 0.5 * (1.0 - min_margin / boundary_threshold)
        #         reward -= boundary_penalty
        
        # # === 安全检查 ===
        # reward = float(np.clip(reward, -10.0, 10.0))
        # if not np.isfinite(reward):
        #     reward = -5.0
        
        return reward




    def _compute_maps(self):
        """计算均值和不确定性 map"""
        n_ctx = self.X_ctx.shape[0]

        if n_ctx == 0:
            # 无观测：用先验（真值的模糊版本 + 高不确定性）
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

        # === 关键：使用固定的归一化参数，而不是每次都变 ===
        # 用整个 Branin 的典型范围来归一化
        y_global_min = 0.0    # Branin 最小约 0.4
        y_global_max = 300.0  # Branin 在边界处可能很大
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
                logits_t = torch.from_numpy(np.asarray(logits))
            else:
                logits_t = logits
            with torch.no_grad():
                var_t = criterion.variance(logits_t)
            if isinstance(var_t, torch.Tensor):
                var = var_t.detach().cpu().numpy().reshape(-1)
            else:
                var = np.asarray(var_t).reshape(-1)
            
            var = var * (y_range ** 2)  # 反归一化方差
            uncert_map = np.sqrt(np.maximum(var, 1e-6))  # 用标准差
            uncert_map = uncert_map.reshape(self.grid_size, self.grid_size).astype(np.float32)
        else:
            dists = pairwise_distances(self.grid_points, self.X_ctx)
            min_dist = dists.min(axis=1)
            uncert_map = min_dist.reshape(self.grid_size, self.grid_size).astype(np.float32)

        return mean_map, uncert_map

    # 在 TabPFN_BraninEnv 类中修改 _get_observation 方法

    def _get_observation(self, return_raw=False):
        """
        构造观测：
        - 通道 0: 归一化的均值预测（标识哪里可能是低值区域）
        - 通道 1: 归一化的不确定性（标识哪里需要探索）
        
        Args:
            return_raw: 如果为 True，返回原始的 (mean_map, uncert_map) 用于可视化
        """
        mean_map, uncert_map = self._compute_maps()
        
        if return_raw:
            return (
                mean_map[None, :, :].astype(np.float32),
                uncert_map[None, :, :].astype(np.float32),
            )
        
        # === 使用固定范围归一化，而不是 instance norm ===
        # 均值：用 Branin 的典型范围
        mean_norm = (mean_map - 0.0) / 300.0  # 归一化到 [0, 1] 附近
        mean_norm = np.clip(mean_norm, -1.0, 2.0)  # 防止极端值
        
        # 不确定性：用合理的范围
        uncert_norm = uncert_map / 50.0  # 归一化到 [0, ~2] 附近
        uncert_norm = np.clip(uncert_norm, 0.0, 3.0)
        
        # === 添加已采样点的位置信息作为"mask"通道 ===
        if self.X_ctx.shape[0] > 0:
            for pt in self.X_ctx:
                i = int((pt[1] - self.x2_min) / (self.x2_max - self.x2_min) * (self.grid_size - 1))
                j = int((pt[0] - self.x1_min) / (self.x1_max - self.x1_min) * (self.grid_size - 1))
                i = np.clip(i, 0, self.grid_size - 1)
                j = np.clip(j, 0, self.grid_size - 1)
                uncert_norm[i, j] = -1.0  # 已采样点不确定性设为负值
        
        obs = np.stack([mean_norm, uncert_norm], axis=0).astype(np.float32)
        return obs


# =========================================================
# 3. 修复版 CNN Agent
# =========================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNAgent(nn.Module):
    def __init__(self, grid_size=32):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(2, 32, 3, 1, 1)),  # 增加通道数
            nn.ReLU(),
            nn.MaxPool2d(2),
            layer_init(nn.Conv2d(32, 64, 3, 1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            layer_init(nn.Conv2d(64, 128, 3, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 固定输出大小
            nn.Flatten(),
        )
        
        self.feature_dim = 128 * 4 * 4
        
        self.fc = nn.Sequential(
            layer_init(nn.Linear(self.feature_dim, 256)),
            nn.ReLU(),  # 改用 ReLU 而不是 Tanh
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        )
        
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)
        
        # === 关键修复：更大的初始化 std ===
        self.actor_mean = layer_init(nn.Linear(128, 2), std=0.5)  # 从 0.01 改为 0.5
        
        # === 初始探索 std 更大 ===
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))  # std = 1.0

    def get_value(self, x):
        return self.critic(self.fc(self.network(x)))

    def get_action_and_value(self, x, action=None):
        feat = self.fc(self.network(x))
        action_mean = torch.tanh(self.actor_mean(feat))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        # === 关键：限制 std 不要太小 ===
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

# 在 train_ppo 函数之前添加此函数

def visualize_episode(agent, test_env, save_dir, update_idx, device):
    """
    可视化一个完整 episode 的采样过程：
    - 第一行：TabPFN 预测的 mean map
    - 第二行：TabPFN 预测的 uncertainty map
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

        # 转回物理坐标，仅用于画点
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
    
    # 确保 axes 是 2D 数组
    if len(idxs_to_plot) == 1:
        axes = axes.reshape(2, 1)
    
    X, Y = test_env.X1, test_env.X2

    # 计算全局颜色范围
    all_means = np.array(raw_means)
    all_uncs = np.array(raw_uncs)
    mean_vmin, mean_vmax = np.percentile(all_means, [2, 98])
    unc_vmax = np.percentile(all_uncs, 98)

    cf1 = cf2 = None

    for i, step_idx in enumerate(idxs_to_plot):
        cur_x1 = history_x1[: step_idx + 1]
        cur_x2 = history_x2[: step_idx + 1]

        # Row 1: mean map（TabPFN 预测）
        ax_mean = axes[0, i]
        cf1 = ax_mean.contourf(
            X, Y, raw_means[step_idx], 
            levels=np.linspace(mean_vmin, mean_vmax, 20),
            extend='both'
        )
        if len(cur_x1) > 1:
            ax_mean.scatter(cur_x1[:-1], cur_x2[:-1], c="white", marker="o", s=40, alpha=0.7, edgecolors='black', linewidths=0.5)
        ax_mean.scatter(cur_x1[-1], cur_x2[-1], c="red", marker="x", s=120, linewidths=3)
        ax_mean.set_title(f"Step {steps_to_plot[i]}\nMean Prediction", fontsize=12)
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
    
    # 添加 colorbar
    if cf1 is not None:
        cax1 = fig.add_axes([0.93, 0.55, 0.015, 0.35])
        fig.colorbar(cf1, cax=cax1, label="Mean")
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
# 4. PPO 训练（带梯度监控）
# =========================================================

# 在 train_ppo 函数的训练循环中添加可视化调用
# 找到 "# 保存模型" 部分，在其后添加可视化代码

def train_ppo(args):
    run_name = f"TabPFN_CNN_PPO_FIXED_{int(time.time())}"
    vis_dir = f"visualizations/{run_name}"
    save_dir = f"runs/{run_name}"
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    env = TabPFN_BraninEnv(
        tabpfn_model_path=args.tabpfn_model_path,
        variants_path=args.variants_path,
        grid_size=args.grid_size,
        device=str(device),
        max_steps=args.max_steps,
        seed=args.seed,
    )
    
    # === 创建一个独立的测试环境用于可视化 ===
    test_env = TabPFN_BraninEnv(
        tabpfn_model_path=args.tabpfn_model_path,
        variants_path=args.variants_path,
        grid_size=args.grid_size,
        device=str(device),
        max_steps=args.max_steps,
        seed=args.seed + 1000,  # 使用不同的种子
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

        # === 监控梯度和参数 ===
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
            
        # === 可视化 episode ===
        if update % 10 == 0 or update == 1:
            agent.eval()
            visualize_episode(agent, test_env, vis_dir, update, device)
            agent.train()

    print(f"\n训练结束。模型保存在 {save_dir}/")
    print(f"可视化保存在 {vis_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabpfn_model_path", type=str, default="./model/finetuned_tabpfn_branin_family.ckpt")
    parser.add_argument("--variants_path", type=str, default="./data/branin_family_variants.npz")
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true", default=True)
    
    # === 调整后的超参数 ===
    parser.add_argument("--total_timesteps", type=int, default=100000)  # 更多训练
    parser.add_argument("--learning_rate", type=float, default=1e-4)    # 更大学习率
    parser.add_argument("--num_steps", type=int, default=400)           # 每次收集更多
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.05)         # 更多探索
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    args = parser.parse_args()
    train_ppo(args)