import os
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
# from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from tabpfn import TabPFNRegressor

# 用于 GPU 训练时，不弹窗
plt.switch_backend("agg")

# =========================================================
# 1. Branin 及其变体（和微调代码保持一致的定义）
# =========================================================

def branin_np(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """标准 Branin 函数，用 numpy 版本方便 debug"""
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
    dx1: float = 0.0,
    dx2: float = 0.0,
    sx1: float = 1.0,
    sx2: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.0,
    a: float = 1.0,
    b: float = 5.1 / (4.0 * np.pi**2),
    c: float = 5.0 / np.pi,
    r: float = 6.0,
    s: float = 10.0,
    t: float = 1.0 / (8.0 * np.pi),
) -> torch.Tensor:
    """
    和微调脚本中一致的 Branin 变体族定义。
    x: (..., 2)
    """
    x1 = x[..., 0]
    x2 = x[..., 1]

    x1_t = sx1 * x1 + dx1
    x2_t = sx2 * x2 + dx2

    y = (
        a * (x2_t - b * x1_t**2 + c * x1_t - r) ** 2
        + s * (1.0 - t) * torch.cos(x1_t)
        + s
    )

    y = alpha * y + beta
    return y


def branin_family_numpy_from_params(
    X: np.ndarray,
    variant_params: dict,
    device: str = "cpu",
) -> np.ndarray:
    """
    方便在 numpy 上调用上面的 torch 版变体。
    X: (N,2)
    返回: (N,)
    """
    x_tensor = torch.from_numpy(X).float().unsqueeze(0).to(device)
    with torch.no_grad():
        y = branin_family_torch(x_tensor, **variant_params).cpu().numpy()[0]
    return y.astype(np.float32)


# =========================================================
# 2. 使用 TabPFN 的 RL 环境
# =========================================================

class TabPFN_BraninEnv:
    """
    环境的设计思想：

    - hidden true function: 某个 Branin 变体 f_v(x)
    - agent 的观测：来自 **微调好的 TabPFN** 在网格上的预测：
        - 通道 1: TabPFN 的均值预测 map (grid_size x grid_size)
        - 通道 2: 一个简单的“不确定性”proxy：到最近观测点的距离 map

    - reward: 仍然基于真实变体的 global minimum 计算 regret：
        regret = best_y_so_far - global_min_y
        reward = -log10(regret) (+ 小奖励 for regret < 1)
    """

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

        # ----- 1) 加载微调好的 TabPFN -----
        if not os.path.exists(tabpfn_model_path):
            raise FileNotFoundError(f"找不到 TabPFN 模型: {tabpfn_model_path}")

        # 关键：用 model_path=... 方式加载微调后的模型
        self.tabpfn = TabPFNRegressor(
            model_path=tabpfn_model_path,
            device=self.device,
            n_estimators=1,
            random_state=seed,
            inference_precision=torch.float32,
            fit_mode="batched",
            differentiable_input=False,
        )
        print(f"[Env] Loaded fine-tuned TabPFN from {tabpfn_model_path}")

        # ----- 2) 加载变体（和微调阶段一致） -----
        if not os.path.exists(variants_path):
            raise FileNotFoundError(
                f"找不到变体文件: {variants_path}\n"
                f"请在微调脚本中保存 variants 到该 npz，再运行 RL（见说明）。"
            )

        data = np.load(variants_path, allow_pickle=True)
        self.variants = data["variants"].tolist()
        self.num_variants = len(self.variants)
        print(f"[Env] Loaded {self.num_variants} variants from {variants_path}")

        # ----- 3) 定义物理坐标域和观测网格 -----
        self.x1_min, self.x1_max = -5.0, 10.0
        self.x2_min, self.x2_max = 0.0, 15.0

        x1 = np.linspace(self.x1_min, self.x1_max, self.grid_size)
        x2 = np.linspace(self.x2_min, self.x2_max, self.grid_size)
        self.X1, self.X2 = np.meshgrid(x1, x2)  # (G,G)

        # 网格点 (N_grid, 2)，用于 TabPFN 预测
        self.grid_points = np.stack(
            [self.X1.flatten(), self.X2.flatten()],
            axis=1,
        ).astype(np.float32)

        # 观测 / 动作空间的 shape
        self.observation_space_shape = (2, grid_size, grid_size)
        self.action_space_shape = (2,)

        # 当前 episode 的状态
        self.current_variant_idx = None
        self.current_variant = None
        self.global_min_y = None
        self.X_ctx = None  # (n_ctx, 2)
        self.y_ctx = None  # (n_ctx,)
        self.best_y = None
        self.step_count = 0

    # -----------------------------------------------------
    # 内部：计算某个变体的“近似全局最小值”，用于定义 regret
    # -----------------------------------------------------
    def _find_global_min(self, variant_params: dict) -> float:
        """
        和你原来 PFN 环境一样思路：
        1) 在网格上粗暴搜索一个好初始点
        2) 用 L-BFGS-B 在连续空间上做局部精细搜索
        """
        # 1. 网格粗搜
        y_grid = branin_family_numpy_from_params(
            self.grid_points,
            variant_params,
            device=self.device,
        )
        min_idx = int(np.argmin(y_grid))
        x0 = self.grid_points[min_idx]  # (2,)

        # 2. 局部精搜
        def func(x):
            x = np.array(x, dtype=np.float32).reshape(1, 2)
            y = branin_family_numpy_from_params(
                x,
                variant_params,
                device=self.device,
            )[0]
            return float(y)

        bounds = [
            (self.x1_min, self.x1_max),
            (self.x2_min, self.x2_max),
        ]

        res = minimize(func, x0, bounds=bounds, method="L-BFGS-B")
        best_val = res.fun

        # 再加几个经典 Branin 坐标点作为补充起点（虽然有变体，但有时依旧有帮助）
        candidates = [
            [-np.pi, 12.275],
            [np.pi, 2.275],
            [9.42478, 2.475],
        ]
        for c in candidates:
            res_cand = minimize(func, c, bounds=bounds, method="L-BFGS-B")
            if res_cand.fun < best_val:
                best_val = res_cand.fun

        return float(best_val)

    # -----------------------------------------------------
    # reset / step
    # -----------------------------------------------------
    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            random.seed(seed)

        # 随机挑一个变体
        self.current_variant_idx = int(self.rng.integers(0, self.num_variants))
        self.current_variant = self.variants[self.current_variant_idx]

        # 计算该变体的（近似）全局最小值
        self.global_min_y = self._find_global_min(self.current_variant)

        # 清空上下文
        self.X_ctx = np.empty((0, 2), dtype=np.float32)
        self.y_ctx = np.empty((0,), dtype=np.float32)
        self.best_y = float("inf")
        self.step_count = 0

        obs, _ = self._get_observation(return_raw=False)
        return obs

    def step(self, action: np.ndarray):
        # 1. clip action 到 [-1,1]^2
        action = np.clip(action, -1.0, 1.0)

        # 2. 映射到物理坐标
        x1 = (action[0] + 1) / 2 * (self.x1_max - self.x1_min) + self.x1_min
        x2 = (action[1] + 1) / 2 * (self.x2_max - self.x2_min) + self.x2_min
        x = np.array([[x1, x2]], dtype=np.float32)  # (1,2)

        # 3. 用真实变体函数评估 y
        y_val = branin_family_numpy_from_params(
            x,
            self.current_variant,
            device=self.device,
        )[0]

        # 更新上下文与当前 best
        self.X_ctx = np.concatenate([self.X_ctx, x], axis=0)
        self.y_ctx = np.concatenate([self.y_ctx, np.array([y_val], dtype=np.float32)])
        self.best_y = min(self.best_y, float(y_val))
        self.step_count += 1

        # 4. 计算 regret 和 reward
        regret = max(1e-20, self.best_y - self.global_min_y)
        reward = -np.log10(regret)
        if regret < 1.0:
            reward += 5.0

        done = self.step_count >= self.max_steps

        # 5. 生成新的观测（来自 TabPFN 的预测）
        obs, _ = self._get_observation(return_raw=False)
        info = {}

        return obs, float(reward), done, info

    # -----------------------------------------------------
    # 用 TabPFN + 当前上下文，构造观测
    # -----------------------------------------------------
    # 修改 _compute_mean_and_uncertainty_maps 方法

    def _compute_mean_and_uncertainty_maps(self):
        """
        修复版：
        1. 不对 y 做标准化（或者只做，然后反标准化回来）
        2. 使用更稳定的归一化方式
        """
        n_ctx = self.X_ctx.shape[0]

        # 0) 没有观测时，给先验
        if n_ctx == 0:
            mean_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            # 先验不确定性：离中心越远越不确定（给策略一个探索信号）
            center = np.array([
                (self.x1_max + self.x1_min) / 2,
                (self.x2_max + self.x2_min) / 2
            ])
            dists = np.sqrt(np.sum((self.grid_points - center) ** 2, axis=1))
            uncert_map = dists.reshape(self.grid_size, self.grid_size).astype(np.float32)
            return mean_map, uncert_map

        # 1) 准备数据
        if n_ctx == 1:
            X_fit = np.repeat(self.X_ctx, 2, axis=0)
            y_fit_raw = np.repeat(self.y_ctx, 2, axis=0)
        else:
            X_fit = self.X_ctx
            y_fit_raw = self.y_ctx

        # === 关键修复 1：记录标准化参数，用于反标准化 ===
        y_mean = y_fit_raw.mean()
        y_std = y_fit_raw.std()
        if y_std < 1e-6:
            y_std = 1.0
        
        y_fit = (y_fit_raw - y_mean) / y_std

        # 2) TabPFN fit & predict
        self.tabpfn.fit(X_fit, y_fit)
        full_out = self.tabpfn.predict(self.grid_points, output_type="full")

        # === 关键修复 2：均值预测反标准化 ===
        mean_pred_norm = np.asarray(full_out["mean"]).reshape(-1)
        mean_pred = mean_pred_norm * y_std + y_mean  # 反标准化
        mean_map = mean_pred.reshape(self.grid_size, self.grid_size).astype(np.float32)

        # === 关键修复 3：方差预测反标准化 ===
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
            
            # 方差反标准化：Var(aX) = a^2 * Var(X)
            var = var * (y_std ** 2)
            uncert_map = var.reshape(self.grid_size, self.grid_size).astype(np.float32)
        else:
            # 兜底：用距离
            dists = pairwise_distances(self.grid_points, self.X_ctx)
            min_dist = dists.min(axis=1)
            uncert_map = min_dist.reshape(self.grid_size, self.grid_size).astype(np.float32)

        return mean_map, uncert_map


    def _get_observation(self, return_raw: bool = False):
        """
        修复版：使用更稳健的归一化
        """
        mean_map, uncert_map = self._compute_mean_and_uncertainty_maps()

        if return_raw:
            return (
                np.nan_to_num(mean_map[None, :, :], nan=0.0),
                np.nan_to_num(uncert_map[None, :, :], nan=0.0),
            )

        # === 关键修复 4：使用 robust 归一化 ===
        def robust_normalize(arr):
            """使用百分位数归一化，避免极值影响"""
            p5, p95 = np.percentile(arr, [5, 95])
            range_val = p95 - p5
            if range_val < 1e-6:
                range_val = 1.0
            return (arr - p5) / range_val - 0.5  # 归一化到 [-0.5, 0.5] 附近
        
        mean_norm = robust_normalize(mean_map)
        uncert_norm = robust_normalize(uncert_map)

        # 额外：在已采样点位置标记（给策略一个明确的"已探索"信号）
        if self.X_ctx.shape[0] > 0:
            # 找到每个已采样点对应的网格索引
            for pt in self.X_ctx:
                i = int((pt[1] - self.x2_min) / (self.x2_max - self.x2_min) * (self.grid_size - 1))
                j = int((pt[0] - self.x1_min) / (self.x1_max - self.x1_min) * (self.grid_size - 1))
                i = np.clip(i, 0, self.grid_size - 1)
                j = np.clip(j, 0, self.grid_size - 1)
                # 在方差图上将已采样点位置设为低不确定性
                uncert_norm[i, j] = -2.0  # 明确标记

        obs = np.stack([mean_norm, uncert_norm], axis=0).astype(np.float32)
        return obs, None


# =========================================================
# 3. CNN Agent（和你之前 PPO 代码一样）
# =========================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNAgent(nn.Module):
    def __init__(self, grid_size=32):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(2, 16, 3, 1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            layer_init(nn.Conv2d(16, 32, 3, 1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            layer_init(nn.Conv2d(32, 64, 3, 1, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 2, grid_size, grid_size)
            self.feature_dim = self.network(dummy).shape[1]
        self.fc = nn.Sequential(layer_init(nn.Linear(self.feature_dim, 256)), nn.Tanh())
        self.critic = layer_init(nn.Linear(256, 1), std=1.0)
        self.actor_mean = layer_init(nn.Linear(256, 2), std=0.01)
        # 初始探索 std ~ 0.36
        self.actor_logstd = nn.Parameter(torch.ones(1, 2) * -1.0)

    def get_value(self, x):
        return self.critic(self.fc(self.network(x)))

    def get_action_and_value(self, x, action=None):
        feat = self.fc(self.network(x))
        action_mean = torch.tanh(self.actor_mean(feat))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
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
# 4. 可视化（沿用你原来的接口）
# =========================================================

def visualize_episode(agent, test_env, save_dir, update_idx, device):
    obs = test_env.reset()

    raw_means = []
    raw_uncs = []
    history_x1 = []
    history_x2 = []

    # 初始
    raw_mean, raw_unc = test_env._get_observation(return_raw=True)
    # unc_max = np.max(raw_unc) if np.max(raw_unc) > 1e-6 else 1.0
    # print(f"[Visual] Uncertainty range: min={unc_min:.4f}, max={unc_max:.4f}")

    done = False
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

        raw_mean, raw_unc = test_env._get_observation(return_raw=True)
        raw_means.append(raw_mean[0])
        raw_uncs.append(raw_unc[0])

    steps_to_plot = [4, 8, 12, 16, 20]
    idxs_to_plot = [s - 1 for s in steps_to_plot]

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    X, Y = test_env.X1, test_env.X2

    for i, step_idx in enumerate(idxs_to_plot):
        if step_idx >= len(history_x1):
            break

        cur_x1 = history_x1[: step_idx + 1]
        cur_x2 = history_x2[: step_idx + 1]

        # Row 1: mean map（TabPFN 预测）
        ax_mean = axes[0, i]
        cf1 = ax_mean.contourf(X, Y, raw_means[step_idx], levels=20)
        ax_mean.scatter(cur_x1[:-1], cur_x2[:-1], c="white", marker=".", s=50, alpha=0.7)
        ax_mean.scatter(cur_x1[-1], cur_x2[-1], c="red", marker="x", s=120, linewidths=3)
        ax_mean.set_title(f"Step {steps_to_plot[i]}\nMean", fontsize=12)
        if i > 0:
            ax_mean.tick_params(left=False, labelleft=False)
        ax_mean.tick_params(bottom=False, labelbottom=False)

        # Row 2: “不确定性” map（距离最近观测点的距离）
        ax_unc = axes[1, i]
        # print(raw_uncs[step_idx])
        unc_max = np.max(raw_uncs)
        cf2 = ax_unc.contourf(
            X,
            Y,
            raw_uncs[step_idx],
            levels=np.linspace(0, unc_max, 20)
        )
        ax_unc.scatter(cur_x1, cur_x2, c="white", marker=".", s=30, alpha=0.5)
        ax_unc.set_title("Uncertainty (dist)", fontsize=12)
        if i > 0:
            ax_unc.tick_params(left=False, labelleft=False)
        ax_unc.set_xlabel("X1")
        if i == 0:
            ax_unc.set_ylabel("X2")

    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    cax1 = fig.add_axes([0.93, 0.55, 0.015, 0.35])
    fig.colorbar(cf1, cax=cax1, label="Mean")
    cax2 = fig.add_axes([0.93, 0.1, 0.015, 0.35])
    fig.colorbar(cf2, cax=cax2, label="Uncertainty")

    final_regret = max(1e-8, test_env.best_y - test_env.global_min_y)
    plt.suptitle(
        f"Update {update_idx}: Final Regret={final_regret:.2e} "
        f"(Log={np.log10(final_regret):.2f})",
        fontsize=16,
        y=0.98,
    )

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vis_update_{update_idx:04d}.png")
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[Visual] 可视化已保存: {save_path}")


# =========================================================
# 5. PPO 训练循环（改动很小，只是换了 Env）
# =========================================================

def train_ppo(args):
    run_name = f"TabPFN_CNN_PPO_{int(time.time())}"
    # writer = SummaryWriter(f"runs/{run_name}")
    vis_dir = f"visualizations/{run_name}"
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = TabPFN_BraninEnv(
        tabpfn_model_path=args.tabpfn_model_path,
        variants_path=args.variants_path,
        grid_size=args.grid_size,
        device=str(device),
        max_steps=args.max_steps,
        seed=args.seed,
    )
    test_env = TabPFN_BraninEnv(
        tabpfn_model_path=args.tabpfn_model_path,
        variants_path=args.variants_path,
        grid_size=args.grid_size,
        device=str(device),
        max_steps=args.max_steps,
        seed=args.seed + 123,
    )

    agent = CNNAgent(grid_size=args.grid_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros(
        (args.num_steps, 2, args.grid_size, args.grid_size)
    ).to(device)
    actions = torch.zeros((args.num_steps, 2)).to(device)
    logprobs = torch.zeros((args.num_steps)).to(device)
    rewards = torch.zeros((args.num_steps)).to(device)
    dones = torch.zeros((args.num_steps)).to(device)
    values = torch.zeros((args.num_steps)).to(device)

    global_step = 0
    next_obs = env.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_done = torch.zeros(1).to(device)

    num_updates = args.total_timesteps // args.num_steps

    print(f"\n[Train] PPO 启动: {args.total_timesteps} steps")
    print("[Info] Env uses fine-tuned TabPFN for observation.")

    for update in range(1, num_updates + 1):
        if update % 5 == 0:
            visualize_episode(agent, test_env, vis_dir, update, device)

        episode_regrets = []

        for step in range(args.num_steps):
            if step % 100 == 0:
                print("step:", step)
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs.unsqueeze(0)
                )
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            real_action = action.cpu().numpy()[0]
            next_obs_np, reward, done, _ = env.step(real_action)

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
            next_done = torch.tensor([done], dtype=torch.float32).to(device)

            if done:
                final_regret = max(1e-20, env.best_y - env.global_min_y)
                episode_regrets.append(final_regret)
                # writer.add_scalar("charts/ep_regret", final_regret, global_step)
                next_obs_np = env.reset()
                next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)

        # GAE & returns
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
                delta = (
                    rewards[t]
                    + args.gamma * nextvalues * nextnonterminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape((-1,) + env.observation_space_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_space_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

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

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                pg_loss = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if update % 1 == 0:
            avg_regret = (
                float(np.mean(episode_regrets)) if len(episode_regrets) > 0 else 0.0
            )
            log_regret = np.log10(avg_regret) if avg_regret > 0 else -20
            print(
                f"Update {update} | Loss: {loss.item():.4f} "
                f"| Avg Regret: {avg_regret:.2e} (Log {log_regret:.2f}) "
                f"| R/Step: {rewards.mean().item():.4f}"
            )
            if update % 5 == 0:
                save_path = f"runs/{run_name}/agent_{update}.pt"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(
                    {"model_state_dict": agent.state_dict(), "config": vars(args)},
                    save_path,
                )

    print(f"训练结束。模型已保存至 runs/{run_name}/")
    # writer.close()


# =========================================================
# 6. main & argparse
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tabpfn_model_path",
        type=str,
        default="./model/finetuned_tabpfn_branin_family.ckpt",
        help="微调 TabPFN 模型 ckpt 路径",
    )
    parser.add_argument(
        "--variants_path",
        type=str,
        default="./data/branin_family_variants.npz",
        help="微调阶段保存的 Branin 变体 npz 路径",
    )
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=20)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true", default=True)

    parser.add_argument("--total_timesteps", type=int, default=50000)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--minibatch_size", type=int, default=32)
    parser.add_argument("--update_epochs", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.02)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    args = parser.parse_args()

    train_ppo(args)
