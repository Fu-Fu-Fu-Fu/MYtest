import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

from tabpfn import TabPFNRegressor

plt.switch_backend("agg")  # 跟训练脚本一样，不弹窗

# =============================
# 0. 标准 Branin 真值函数
# =============================

def branin_np(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    和你微调 / RL 代码里的 branin_np 一致
    """
    a = 1.0
    b = 5.1 / (4.0 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)

    return (
        a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
        + s * (1 - t) * np.cos(x1)
        + s
    ).astype(np.float32)


# =============================
# 1. CNN Agent（和你 RL 训练代码保持一致）
# =============================

import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# 替换 CNNAgent 类

class CNNAgent(nn.Module):
    def __init__(self, grid_size=32):
        super().__init__()
        # === 与 train_rl.py 保持一致 ===
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

    def get_deterministic_action(self, x):
        feat = self.fc(self.network(x))
        action_mean = torch.tanh(self.actor_mean(feat))
        return action_mean


# =============================
# 2. RL 策略测试器（基于 TabPFNRegressor）
# =============================

class RLPolicyTesterCNN:
    """
    用你 **微调好的 TabPFNRegressor** 和 **已训练好的 CNN PPO 策略**，
    在标准 Branin 上做四种方法对比：

      1) TabPFN + RL
      2) GP + RL
      3) GP-BO
      4) TabPFN-BO
    """

    def __init__(self, policy_path, tabpfn_model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.policy_path = policy_path

        # ---------- 加载 RL 策略（CNNAgent） ----------
        print(f"加载 RL 策略: {policy_path}")
        ckpt = torch.load(policy_path, map_location=self.device)
        config = ckpt.get('config', {})
        self.grid_size = config.get('grid_size', 32)

        self.agent = CNNAgent(grid_size=self.grid_size).to(self.device)
        self.agent.load_state_dict(ckpt['model_state_dict'])
        self.agent.eval()
        print("✓ RL 策略加载完成")

        # ---------- 加载 TabPFNRegressor ----------
        print(f"加载 TabPFNRegressor（用于 RL 观测）: {tabpfn_model_path}")
        self.tabpfn_rl = TabPFNRegressor(
            model_path=tabpfn_model_path,
            device=str(self.device),
            n_estimators=1,
            random_state=config.get('seed', 42),
            inference_precision=torch.float32,
            fit_mode="batched",
            differentiable_input=False,
        )

        print(f"加载 TabPFNRegressor（用于 BO）: {tabpfn_model_path}")
        self.tabpfn_bo = TabPFNRegressor(
            model_path=tabpfn_model_path,
            device=str(self.device),
            n_estimators=1,
            random_state=config.get('seed', 43),
            inference_precision=torch.float32,
            fit_mode="batched",
            differentiable_input=False,
        )
        print("✓ TabPFNRegressor 加载完成")

        # ---------- Branin 定义域（和训练时一致） ----------
        self.x1_min, self.x1_max = -5.0, 10.0
        self.x2_min, self.x2_max = 0.0, 15.0

        # 预先在物理坐标上建好网格
        x1 = np.linspace(self.x1_min, self.x1_max, self.grid_size, dtype=np.float32)
        x2 = np.linspace(self.x2_min, self.x2_max, self.grid_size, dtype=np.float32)
        X1, X2 = np.meshgrid(x1, x2)
        self.X1, self.X2 = X1, X2
        self.grid_points = np.stack([X1.flatten(), X2.flatten()], axis=1).astype(np.float32)

        # 标准 Branin 的全局最小值（Ground Truth）
        self.global_min_y = 0.397887

    # ---------- 一些小工具 ----------

    def _set_seed(self, seed):
        if seed is None:
            return
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _evaluate(self, x):
        """真值函数：标准 Branin，直接用物理坐标评估"""
        x = np.asarray(x, dtype=np.float32)
        return float(branin_np(x[0], x[1]))

    def _action_to_physical(self, a):
        """把策略输出的 [-1,1]^2 action 映射回物理坐标"""
        a = np.clip(a, -1.0, 1.0)
        x1 = (a[0] + 1) / 2 * (self.x1_max - self.x1_min) + self.x1_min
        x2 = (a[1] + 1) / 2 * (self.x2_max - self.x2_min) + self.x2_min
        return np.array([x1, x2], dtype=np.float32)

    # 修改 _tabpfn_maps 和 _maps_to_obs 方法

    # 更新 _tabpfn_maps 和 _maps_to_obs 方法，与 train_rl.py 保持一致

    def _tabpfn_maps(self, X_ctx, y_ctx):
        """与训练时的 _compute_maps 保持一致"""
        n_ctx = X_ctx.shape[0]
        
        if n_ctx == 0:
            mean_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            uncert_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
            return mean_map, uncert_map

        if n_ctx == 1:
            X_fit = np.repeat(X_ctx, 2, axis=0)
            y_fit_raw = np.repeat(y_ctx, 2, axis=0)
        else:
            X_fit = X_ctx
            y_fit_raw = y_ctx

        # === 使用固定的归一化参数（与训练一致）===
        y_global_min = 0.0
        y_global_max = 300.0
        y_range = y_global_max - y_global_min
        
        y_fit_norm = (y_fit_raw - y_global_min) / y_range

        self.tabpfn_rl.fit(X_fit, y_fit_norm)
        full_out = self.tabpfn_rl.predict(self.grid_points, output_type="full")

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
            
            var = var * (y_range ** 2)
            uncert_map = np.sqrt(np.maximum(var, 1e-6))
            uncert_map = uncert_map.reshape(self.grid_size, self.grid_size).astype(np.float32)
        else:
            from sklearn.metrics import pairwise_distances
            dists = pairwise_distances(self.grid_points, X_ctx)
            min_dist = dists.min(axis=1)
            uncert_map = min_dist.reshape(self.grid_size, self.grid_size).astype(np.float32)

        return mean_map, uncert_map


    def _maps_to_obs(self, mean_map, uncert_map, X_ctx=None):
        """与训练时的 _get_observation 保持一致"""
        # 使用固定范围归一化
        mean_norm = (mean_map - 0.0) / 300.0
        mean_norm = np.clip(mean_norm, -1.0, 2.0)
        
        uncert_norm = uncert_map / 50.0
        uncert_norm = np.clip(uncert_norm, 0.0, 3.0)
        
        # 标记已采样点
        if X_ctx is not None and len(X_ctx) > 0:
            for pt in X_ctx:
                i = int((pt[1] - self.x2_min) / (self.x2_max - self.x2_min) * (self.grid_size - 1))
                j = int((pt[0] - self.x1_min) / (self.x1_max - self.x1_min) * (self.grid_size - 1))
                i = np.clip(i, 0, self.grid_size - 1)
                j = np.clip(j, 0, self.grid_size - 1)
                uncert_norm[i, j] = -1.0
        
        obs = np.stack([mean_norm, uncert_norm], axis=0).astype(np.float32)
        return obs

    # ---------- 2.1 TabPFN + RL ----------

    def run_rl_tabpfn_optimization(self, max_steps=20, seed=None, verbose=False):
        """
        和原来的 run_rl_optimization 类似，只是：
        - PFN 模型换成 TabPFNRegressor（微调后的）
        - 直接在 Branin 真值上评估
        """
        self._set_seed(seed)
        x_history, y_history = [], []

        # 初始随机点（物理坐标）
        x_init = np.array(
            [np.random.uniform(self.x1_min, self.x1_max),
             np.random.uniform(self.x2_min, self.x2_max)],
            dtype=np.float32,
        )
        y_init = self._evaluate(x_init)
        x_history.append(x_init)
        y_history.append(y_init)

        X_ctx = np.array(x_history, dtype=np.float32)
        y_ctx = np.array(y_history, dtype=np.float32)

        for step in range(max_steps - 1):
            mean_map, uncert_map = self._tabpfn_maps(X_ctx, y_ctx)
            obs = self._maps_to_obs(mean_map, uncert_map, X_ctx=X_ctx)  # 传入 X_ctx

            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _, _, _ = self.agent.get_action_and_value(obs_tensor)

            x_new = self._action_to_physical(action.cpu().numpy()[0])
            y_new = self._evaluate(x_new)

            x_history.append(x_new)
            y_history.append(y_new)

            X_ctx = np.vstack([X_ctx, x_new[None, :]])
            y_ctx = np.concatenate([y_ctx, np.array([y_new], dtype=np.float32)])

            if verbose:
                best = np.min(y_history)
                print(f"[TabPFN+RL] step {step+1}: y={y_new:.4f}, best={best:.4f}")

        return np.array(x_history), np.array(y_history)

    # ---------- 2.2 GP + RL ----------

    # def run_gp_rl_optimization(self, max_steps=20, seed=None, verbose=False):
    #     """
    #     和原来一样：用 GP 在网格上算 (mu, sigma)，喂给同一个 CNN 策略。
    #     """
    #     self._set_seed(seed)
    #     x_history, y_history = [], []

    #     x_init = np.array(
    #         [np.random.uniform(self.x1_min, self.x1_max),
    #          np.random.uniform(self.x2_min, self.x2_max)],
    #         dtype=np.float32,
    #     )
    #     y_init = self._evaluate(x_init)
    #     x_history.append(x_init)
    #     y_history.append(y_init)

    #     for step in range(max_steps - 1):
    #         kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-2)
    #         gp = GaussianProcessRegressor(
    #             kernel=kernel,
    #             alpha=1e-6,
    #             normalize_y=True,
    #             n_restarts_optimizer=5,
    #             random_state=seed,
    #         )
    #         gp.fit(np.array(x_history), np.array(y_history).reshape(-1, 1))
    #         mu, sigma = gp.predict(self.grid_points, return_std=True)
    #         mu_map = mu.reshape(self.grid_size, self.grid_size)
    #         sigma_map = sigma.reshape(self.grid_size, self.grid_size)

    #         m_mean, m_std = mu_map.mean(), mu_map.std()
    #         s_mean, s_std = sigma_map.mean(), sigma_map.std()
    #         if m_std < 1e-6:
    #             m_std = 1.0
    #         if s_std < 1e-6:
    #             s_std = 1.0
    #         mu_norm = (mu_map - m_mean) / m_std
    #         sigma_norm = (sigma_map - s_mean) / s_std
    #         obs = np.stack([mu_norm, sigma_norm], axis=0).astype(np.float32)

    #         obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
    #         with torch.no_grad():
    #             action, _, _, _ = self.agent.get_action_and_value(obs_tensor)

    #         x_new = self._action_to_physical(action.cpu().numpy()[0])
    #         y_new = self._evaluate(x_new)

    #         x_history.append(x_new)
    #         y_history.append(y_new)

    #         if verbose:
    #             best = np.min(y_history)
    #             print(f"[GP+RL] step {step+1}: y={y_new:.4f}, best={best:.4f}")

    #     return np.array(x_history), np.array(y_history)

    # def run_gp_rl_optimization(self, max_steps=20, seed=None, verbose=False):
    #     """
    #     用 GP 在 Branin 真值上做 surrogate，在网格上算 (mu, sigma)，
    #     再把 [mu_norm, sigma_norm] 喂给已经训练好的 CNN 策略。
    #     """
    #     # 设随机种子
    #     self._set_seed(seed)

    #     x_history, y_history = [], []

    #     # 1) 初始随机点（物理坐标）
    #     x_init = np.array(
    #         [
    #             np.random.uniform(self.x1_min, self.x1_max),
    #             np.random.uniform(self.x2_min, self.x2_max),
    #         ],
    #         dtype=np.float32,
    #     )
    #     y_init = self._evaluate(x_init)  # 真正的 Branin 函数
    #     x_history.append(x_init)
    #     y_history.append(y_init)

    #     # 2) 迭代优化
    #     for step in range(max_steps - 1):
    #         # 2.1 用历史 (x,y) 拟合 GP（完全在物理坐标上）
    #         kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.01)
    #         gp = GaussianProcessRegressor(
    #             kernel=kernel,
    #             alpha=1e-6,
    #             normalize_y=True,
    #             n_restarts_optimizer=5,
    #             random_state=seed,
    #         )
    #         gp.fit(np.array(x_history), np.array(y_history).reshape(-1, 1))

    #         # 2.2 在固定网格上算 mu / sigma（仍然是物理坐标）
    #         mu, sigma = gp.predict(self.grid_points, return_std=True)
    #         mu_map = mu.reshape(self.grid_size, self.grid_size)
    #         sigma_map = sigma.reshape(self.grid_size, self.grid_size)

    #         # 2.3 和训练时一样做标准化
    #         m_mean, m_std = mu_map.mean(), mu_map.std()
    #         s_mean, s_std = sigma_map.mean(), sigma_map.std()
    #         if m_std < 1e-6:
    #             m_std = 1.0
    #         if s_std < 1e-6:
    #             s_std = 1.0
    #         mu_norm = (mu_map - m_mean) / m_std
    #         sigma_norm = (sigma_map - s_mean) / s_std

    #         # 2.4 拼成 obs（2, G, G），喂给 CNN 策略
    #         obs = np.stack([mu_norm, sigma_norm], axis=0).astype(np.float32)
    #         obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
    #         with torch.no_grad():
    #             action, _, _, _ = self.agent.get_action_and_value(obs_tensor)

    #         # 2.5 动作 [-1,1]^2 → 物理坐标
    #         x_new = self._action_to_physical(action.cpu().numpy()[0])

    #         # 2.6 用真函数评估（不要再通过 env.step）
    #         y_new = self._evaluate(x_new)
    #         x_history.append(x_new)
    #         y_history.append(y_new)

    #         if verbose:
    #             print(
    #                 f"[GP+RL] step {step+1}: y={y_new:.4f}, "
    #                 f"best={np.min(y_history):.4f}"
    #             )

    #     return np.array(x_history), np.array(y_history)
    # ---------- 2.2 GP + RL (修正版) ----------

    def run_gp_rl_optimization(self, max_steps=20, seed=None, verbose=False):
        """
        修正点：
        1. 对 X 进行归一化 (MinMax Scaling)，解决 GP ConvergenceWarning。
        2. 将 GP 输出的 std 转换为 variance，与 TabPFN 训练时的输入保持一致。
        """
        self._set_seed(seed)
        
        # 定义 GP 的输入归一化函数
        def normalize_x(x):
            # 映射到 [0, 1]
            x_norm = np.copy(x)
            x_norm[:, 0] = (x[:, 0] - self.x1_min) / (self.x1_max - self.x1_min)
            x_norm[:, 1] = (x[:, 1] - self.x2_min) / (self.x2_max - self.x2_min)
            return x_norm

        # 预先计算好归一化后的网格点，用于 GP 预测
        grid_points_norm = normalize_x(self.grid_points)

        x_history, y_history = [], []

        # 1) 初始随机点
        x_init = np.array(
            [
                np.random.uniform(self.x1_min, self.x1_max),
                np.random.uniform(self.x2_min, self.x2_max),
            ],
            dtype=np.float32,
        )
        y_init = self._evaluate(x_init)
        x_history.append(x_init)
        y_history.append(y_init)

        for step in range(max_steps - 1):
            # 准备训练数据
            X_train = np.array(x_history)
            y_train = np.array(y_history).reshape(-1, 1)
            
            # GP 输入归一化
            X_train_norm = normalize_x(X_train)

            # 2.1 定义更鲁棒的 Kernel
            # 增大 length_scale_bounds 的范围，防止撞墙警告
            # 增加 ConstantKernel 以适应不同的均值幅度
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=0.5, length_scale_bounds=(1e-2, 1e2), nu=2.5) + \
                     WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))
            
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=0.0, # 噪声由 WhiteKernel 处理
                normalize_y=True, # 这是一个好习惯，保持开启
                n_restarts_optimizer=5,
                random_state=seed, # 保持 GP 拟合确定性
            )
            
            # 拟合 GP
            with np.errstate(all='ignore'): # 偶尔的警告可以忽略，只要不是 ConvergenceWarning
                gp.fit(X_train_norm, y_train)

            # 2.2 在网格上预测 (使用归一化的网格)
            mu, sigma = gp.predict(grid_points_norm, return_std=True)
            
            # 关键修正：RL Agent 是基于 Variance 训练的，而 GP 输出的是 Std
            # 所以这里需要平方，转为方差图
            variance = sigma ** 2
            
            mu_map = mu.reshape(self.grid_size, self.grid_size)
            var_map = variance.reshape(self.grid_size, self.grid_size) # 注意这里用 var_map

            # 2.3 标准化 (Instance Normalization)
            m_mean, m_std = mu_map.mean(), mu_map.std()
            v_mean, v_std = var_map.mean(), var_map.std()
            
            if m_std < 1e-6: m_std = 1.0
            if v_std < 1e-6: v_std = 1.0
            
            mu_norm = (mu_map - m_mean) / m_std
            var_norm = (var_map - v_mean) / v_std # 使用 var_norm

            # 2.4 构造 Observation
            obs = np.stack([mu_norm, var_norm], axis=0).astype(np.float32)
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, _, _, _ = self.agent.get_action_and_value(obs_tensor)

            # 2.5 动作映射回物理坐标
            x_new = self._action_to_physical(action.cpu().numpy()[0])
            y_new = self._evaluate(x_new)

            x_history.append(x_new)
            y_history.append(y_new)

            if verbose:
                print(f"[GP+RL] step {step+1}: y={y_new:.4f}, best={np.min(y_history):.4f}")

        return np.array(x_history), np.array(y_history)


    # ---------- 2.3 GP-BO ----------

    def run_bo_gp_optimization(self, max_steps=20, seed=None, verbose=False):
        self._set_seed(seed)
        x_history, y_history = [], []

        x_init = np.array(
            [np.random.uniform(self.x1_min, self.x1_max),
             np.random.uniform(self.x2_min, self.x2_max)],
            dtype=np.float32,
        )
        y_init = self._evaluate(x_init)
        x_history.append(x_init)
        y_history.append(y_init)

        for step in range(max_steps - 1):
            kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=seed,
            )
            gp.fit(np.array(x_history), np.array(y_history).reshape(-1, 1))

            def ei(x):
                mu, sigma = gp.predict(np.atleast_2d(x), return_std=True)
                mu = mu.ravel()
                sigma = sigma.ravel()
                f_best = np.min(y_history)
                with np.errstate(divide="warn"):
                    z = (f_best - mu) / (sigma + 1e-9)
                    val = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                    val[sigma == 0.0] = 0.0
                return val

            x_candidates = np.random.uniform(
                low=[self.x1_min, self.x2_min],
                high=[self.x1_max, self.x2_max],
                size=(1000, 2),
            )
            ei_vals = ei(x_candidates)
            x_next = x_candidates[np.argmax(ei_vals)]
            y_next = self._evaluate(x_next)

            x_history.append(x_next.astype(np.float32))
            y_history.append(y_next)

            if verbose:
                best = np.min(y_history)
                print(f"[GP-BO] step {step+1}: y={y_next:.4f}, best={best:.4f}")

        return np.array(x_history), np.array(y_history)

    # ---------- 2.4 TabPFN-BO ----------

    def run_bo_tabpfn_optimization(self, max_steps=20, seed=None, verbose=False):
        """
        用微调好的 TabPFN 做 surrogate，基于 EI 选择下一点。
        """
        self._set_seed(seed)
        x_history, y_history = [], []

        x_init = np.array(
            [np.random.uniform(self.x1_min, self.x1_max),
             np.random.uniform(self.x2_min, self.x2_max)],
            dtype=np.float32,
        )
        y_init = self._evaluate(x_init)
        x_history.append(x_init)
        y_history.append(y_init)

        for step in range(max_steps - 1):
            X_train = np.array(x_history, dtype=np.float32)
            y_train = np.array(y_history, dtype=np.float32)

            # TabPFN 至少 2 个样本
            if X_train.shape[0] == 1:
                X_fit = np.repeat(X_train, 2, axis=0)
                y_fit = np.repeat(y_train, 2, axis=0)
            else:
                X_fit, y_fit = X_train, y_train

            self.tabpfn_bo.fit(X_fit, y_fit)

            x_candidates = np.random.uniform(
                low=[self.x1_min, self.x2_min],
                high=[self.x1_max, self.x2_max],
                size=(1000, 2),
            ).astype(np.float32)

            full_out = self.tabpfn_bo.predict(x_candidates, output_type="full")
            mu = np.asarray(full_out["mean"], dtype=np.float32).reshape(-1)

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
                sigma = np.sqrt(np.maximum(var, 1e-12))
            else:
                sigma = np.ones_like(mu, dtype=np.float32)

            f_best = np.min(y_history)
            with np.errstate(divide="warn"):
                z = (f_best - mu) / (sigma + 1e-9)
                ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                ei[sigma == 0.0] = 0.0

            best_idx = np.argmax(ei)
            x_next = x_candidates[best_idx]
            y_next = self._evaluate(x_next)

            x_history.append(x_next.astype(np.float32))
            y_history.append(y_next)

            if verbose:
                best = np.min(y_history)
                print(f"[TabPFN-BO] step {step+1}: y={y_next:.4f}, best={best:.4f}")

        return np.array(x_history), np.array(y_history)

    # ---------- 2.3 Random + RL ----------

    def run_random_rl_optimization(self, max_steps=20, seed=None, verbose=False):
        """
        检查策略是否真的利用了输入的 (mean, var)。
        这里完全不看 TabPFN / GP，只在每个 step 随机生成 mean_map 和 var_map，
        然后喂给同一个 CNN 策略，看采样轨迹是否还和 TabPFN+RL / GP+RL 很像。
        """
        self._set_seed(seed)
        rng = np.random.default_rng(seed)

        x_history, y_history = [], []

        # 初始随机点（物理坐标），和其他方法保持一致的写法
        x_init = np.array(
            [
                np.random.uniform(self.x1_min, self.x1_max),
                np.random.uniform(self.x2_min, self.x2_max),
            ],
            dtype=np.float32,
        )
        y_init = self._evaluate(x_init)
        x_history.append(x_init)
        y_history.append(y_init)

        for step in range(max_steps - 1):
            # 随机生成一对 (mean_map, var_map)
            mean_map = rng.normal(size=(self.grid_size, self.grid_size)).astype(np.float32)
            var_map = rng.normal(size=(self.grid_size, self.grid_size)).astype(np.float32)

            # 用和训练时一样的 _maps_to_obs 做 instance norm
            obs = self._maps_to_obs(mean_map, var_map)
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, _, _, _ = self.agent.get_action_and_value(obs_tensor)

            # 动作映射到物理坐标，并在真 Branin 上评估
            x_new = self._action_to_physical(action.cpu().numpy()[0])
            y_new = self._evaluate(x_new)

            x_history.append(x_new)
            y_history.append(y_new)

            if verbose:
                print(
                    f"[Random+RL] step {step+1}: y={y_new:.4f}, "
                    f"best={np.min(y_history):.4f}"
                )

        return np.array(x_history), np.array(y_history)

    def debug_compare_means_three_methods(
        self,
        max_steps=20,
        seed=42,
        save_dir="./results/debug_means_three_methods",
    ):
        """
        画 3 行 × 5 列图：
          第 1 行：TabPFN+RL，在 step=4,8,12,16,20 时 TabPFN mean map + 已采样点
          第 2 行：GP+RL，在同样步数时 GP mean map + 已采样点
          第 3 行：Random+RL，在同样步数时 随机 mean map + 已采样点
        如果 Random+RL 的采样点和前两种几乎一致，就说明策略基本没用上输入观测。
        """
        import os
        import matplotlib.pyplot as plt
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            Matern,
            WhiteKernel,
            ConstantKernel as C,
        )

        os.makedirs(save_dir, exist_ok=True)

        # 先各自跑一条 RL 轨迹
        x_tab, y_tab = self.run_rl_tabpfn_optimization(
            max_steps=max_steps, seed=seed, verbose=False
        )
        x_gp, y_gp = self.run_gp_rl_optimization(
            max_steps=max_steps, seed=seed, verbose=False
        )
        x_rand, y_rand = self.run_random_rl_optimization(
            max_steps=max_steps, seed=seed, verbose=False
        )

        # ---------- GP 拟合 & 网格预测（和之前一样） ----------
        def normalize_x(x):
            x = np.asarray(x, dtype=np.float32)
            x_norm = np.empty_like(x)
            x_norm[:, 0] = (x[:, 0] - self.x1_min) / (self.x1_max - self.x1_min)
            x_norm[:, 1] = (x[:, 1] - self.x2_min) / (self.x2_max - self.x2_min)
            return x_norm

        grid_points_norm = normalize_x(self.grid_points)

        def gp_mean_on_grid(X_ctx, y_ctx):
            X_train = np.asarray(X_ctx, dtype=np.float32)
            y_train = np.asarray(y_ctx, dtype=np.float32).reshape(-1, 1)
            X_train_norm = normalize_x(X_train)

            kernel = (
                C(1.0, (1e-3, 1e3))
                * Matern(length_scale=0.5, length_scale_bounds=(1e-2, 1e2), nu=2.5)
                + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=0.0,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=seed,
            )
            with np.errstate(all="ignore"):
                gp.fit(X_train_norm, y_train)

            mu, _ = gp.predict(grid_points_norm, return_std=True)
            mu_map = mu.reshape(self.grid_size, self.grid_size).astype(np.float32)
            return mu_map

        # Random+RL 这边，背景本身就是随机的，我们重新造一份即可
        rng_bg = np.random.default_rng(seed + 123)

        steps_to_plot = [4, 8, 12, 16, 20]
        fig, axes = plt.subplots(3, len(steps_to_plot), figsize=(4 * len(steps_to_plot), 12))

        X_grid, Y_grid = self.X1, self.X2

        cf1 = cf2 = cf3 = None

        for col, step in enumerate(steps_to_plot):
            # 防御：步数别超过轨迹长度
            if step > len(x_tab):
                continue

            # ===== 第 1 行：TabPFN+RL =====
            X_ctx_tab = x_tab[:step]
            y_ctx_tab = y_tab[:step]
            mean_tab, _ = self._tabpfn_maps(X_ctx_tab, y_ctx_tab)

            ax_tab = axes[0, col]
            cf1 = ax_tab.contourf(X_grid, Y_grid, mean_tab, levels=20)
            pts_tab = np.array(X_ctx_tab)

            if len(pts_tab) > 1:
                ax_tab.scatter(
                    pts_tab[:-1, 0],
                    pts_tab[:-1, 1],
                    c="white",
                    s=25,
                    alpha=0.7,
                )
            ax_tab.scatter(
                pts_tab[-1, 0],
                pts_tab[-1, 1],
                c="red",
                marker="x",
                s=80,
                linewidths=2,
            )
            ax_tab.set_title(f"Step {step} - TabPFN mean")
            ax_tab.set_xticklabels([])
            if col > 0:
                ax_tab.set_yticklabels([])

            # ===== 第 2 行：GP+RL =====
            if step > len(x_gp):
                continue
            X_ctx_gp = x_gp[:step]
            y_ctx_gp = y_gp[:step]
            mu_gp = gp_mean_on_grid(X_ctx_gp, y_ctx_gp)

            ax_gp = axes[1, col]
            cf2 = ax_gp.contourf(X_grid, Y_grid, mu_gp, levels=20)
            pts_gp = np.array(X_ctx_gp)

            if len(pts_gp) > 1:
                ax_gp.scatter(
                    pts_gp[:-1, 0],
                    pts_gp[:-1, 1],
                    c="white",
                    s=25,
                    alpha=0.7,
                )
            ax_gp.scatter(
                pts_gp[-1, 0],
                pts_gp[-1, 1],
                c="red",
                marker="x",
                s=80,
                linewidths=2,
            )
            ax_gp.set_title(f"Step {step} - GP mean")
            if col == 0:
                ax_gp.set_ylabel("x2")
            ax_gp.set_xticklabels([])
            if col > 0:
                ax_gp.set_yticklabels([])

            # ===== 第 3 行：Random+RL =====
            if step > len(x_rand):
                continue
            X_ctx_rand = x_rand[:step]
            pts_rand = np.array(X_ctx_rand)

            # 为了可视化，单独生成一个随机 mean_map
            mean_rand = rng_bg.normal(
                size=(self.grid_size, self.grid_size)
            ).astype(np.float32)

            ax_rand = axes[2, col]
            cf3 = ax_rand.contourf(X_grid, Y_grid, mean_rand, levels=20)
            if len(pts_rand) > 1:
                ax_rand.scatter(
                    pts_rand[:-1, 0],
                    pts_rand[:-1, 1],
                    c="white",
                    s=25,
                    alpha=0.7,
                )
            ax_rand.scatter(
                pts_rand[-1, 0],
                pts_rand[-1, 1],
                c="red",
                marker="x",
                s=80,
                linewidths=2,
            )
            ax_rand.set_title(f"Step {step} - Random mean")
            ax_rand.set_xlabel("x1")
            if col == 0:
                ax_rand.set_ylabel("x2")
            else:
                ax_rand.set_yticklabels([])

        plt.tight_layout(rect=[0, 0, 0.92, 0.95])
        if cf1 is not None:
            cax1 = fig.add_axes([0.93, 0.67, 0.015, 0.25])
            fig.colorbar(cf1, cax=cax1, label="TabPFN mean")
        if cf2 is not None:
            cax2 = fig.add_axes([0.93, 0.37, 0.015, 0.25])
            fig.colorbar(cf2, cax=cax2, label="GP mean")
        if cf3 is not None:
            cax3 = fig.add_axes([0.93, 0.07, 0.015, 0.25])
            fig.colorbar(cf3, cax=cax3, label="Random mean")

        save_path = os.path.join(
            save_dir,
            f"means_tabpfn_gp_random_steps{max_steps}_seed{seed}.png",
        )
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[Debug] means comparison (3 methods) saved to {save_path}")

    # 在 RLPolicyTesterCNN 类中添加以下方法（在 debug_compare_means_three_methods 方法之后）

    def plot_trajectory_comparison(
        self,
        x_tabpfn, y_tabpfn,
        x_gp, y_gp,
        run_idx,
        seed,
        save_dir="./results/trajectory_comparison",
    ):
        """
        画 TabPFN+RL 和 GP+RL 的轨迹对比图：
        - 第 1 行：TabPFN+RL，在 step=4,8,12,16,20 时的 mean map + 采样点
        - 第 2 行：GP+RL，在同样步数时的 GP mean map + 采样点
        """
        import os
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            Matern,
            WhiteKernel,
            ConstantKernel as C,
        )

        os.makedirs(save_dir, exist_ok=True)

        # GP 输入归一化函数
        def normalize_x(x):
            x = np.asarray(x, dtype=np.float32)
            x_norm = np.empty_like(x)
            x_norm[:, 0] = (x[:, 0] - self.x1_min) / (self.x1_max - self.x1_min)
            x_norm[:, 1] = (x[:, 1] - self.x2_min) / (self.x2_max - self.x2_min)
            return x_norm

        grid_points_norm = normalize_x(self.grid_points)

        def gp_mean_on_grid(X_ctx, y_ctx):
            X_train = np.asarray(X_ctx, dtype=np.float32)
            y_train = np.asarray(y_ctx, dtype=np.float32).reshape(-1, 1)
            X_train_norm = normalize_x(X_train)

            kernel = (
                C(1.0, (1e-3, 1e3))
                * Matern(length_scale=0.5, length_scale_bounds=(1e-2, 1e2), nu=2.5)
                + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=0.0,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=seed,
            )
            with np.errstate(all="ignore"):
                gp.fit(X_train_norm, y_train)

            mu, _ = gp.predict(grid_points_norm, return_std=True)
            mu_map = mu.reshape(self.grid_size, self.grid_size).astype(np.float32)
            return mu_map

        steps_to_plot = [4, 8, 12, 16, 20]
        fig, axes = plt.subplots(2, len(steps_to_plot), figsize=(4.5 * len(steps_to_plot), 9))

        X_grid, Y_grid = self.X1, self.X2

        # 计算全局颜色范围（为了两行颜色一致）
        all_means_tabpfn = []
        all_means_gp = []
        
        for step in steps_to_plot:
            if step <= len(x_tabpfn):
                mean_tab, _ = self._tabpfn_maps(x_tabpfn[:step], y_tabpfn[:step])
                all_means_tabpfn.append(mean_tab)
            if step <= len(x_gp):
                mean_gp = gp_mean_on_grid(x_gp[:step], y_gp[:step])
                all_means_gp.append(mean_gp)

        # 统一颜色范围
        if all_means_tabpfn and all_means_gp:
            vmin = min(np.min(all_means_tabpfn), np.min(all_means_gp))
            vmax = max(np.max(all_means_tabpfn), np.max(all_means_gp))
        else:
            vmin, vmax = 0, 100

        cf1 = cf2 = None

        for col, step in enumerate(steps_to_plot):
            # ===== 第 1 行：TabPFN+RL =====
            if step <= len(x_tabpfn):
                X_ctx_tab = x_tabpfn[:step]
                y_ctx_tab = y_tabpfn[:step]
                mean_tab, _ = self._tabpfn_maps(X_ctx_tab, y_ctx_tab)

                ax_tab = axes[0, col]
                cf1 = ax_tab.contourf(
                    X_grid, Y_grid, mean_tab,
                    levels=np.linspace(vmin, vmax, 20),
                    extend='both'
                )
                
                # 画采样点
                if len(X_ctx_tab) > 1:
                    ax_tab.scatter(
                        X_ctx_tab[:-1, 0], X_ctx_tab[:-1, 1],
                        c="white", s=40, alpha=0.8,
                        edgecolors='black', linewidths=0.5,
                        zorder=10
                    )
                # 最新采样点用红色 X
                ax_tab.scatter(
                    X_ctx_tab[-1, 0], X_ctx_tab[-1, 1],
                    c="red", marker="x", s=100, linewidths=3,
                    zorder=11
                )
                
                # 标注全局最优位置
                ax_tab.scatter(
                    [np.pi, -np.pi, np.pi],
                    [2.275, 12.275, 2.275],
                    c="lime", marker="*", s=150, edgecolors='black',
                    linewidths=1, zorder=12, label='Global Optima'
                )
                
                best_so_far = np.min(y_ctx_tab)
                ax_tab.set_title(f"Step {step}\nTabPFN+RL (best={best_so_far:.2f})", fontsize=11)
                ax_tab.set_xlim(self.x1_min, self.x1_max)
                ax_tab.set_ylim(self.x2_min, self.x2_max)
                
                if col == 0:
                    ax_tab.set_ylabel("x2", fontsize=11)
                else:
                    ax_tab.tick_params(left=False, labelleft=False)
                ax_tab.tick_params(bottom=False, labelbottom=False)

            # ===== 第 2 行：GP+RL =====
            if step <= len(x_gp):
                X_ctx_gp = x_gp[:step]
                y_ctx_gp = y_gp[:step]
                mean_gp = gp_mean_on_grid(X_ctx_gp, y_ctx_gp)

                ax_gp = axes[1, col]
                cf2 = ax_gp.contourf(
                    X_grid, Y_grid, mean_gp,
                    levels=np.linspace(vmin, vmax, 20),
                    extend='both'
                )
                
                # 画采样点
                if len(X_ctx_gp) > 1:
                    ax_gp.scatter(
                        X_ctx_gp[:-1, 0], X_ctx_gp[:-1, 1],
                        c="white", s=40, alpha=0.8,
                        edgecolors='black', linewidths=0.5,
                        zorder=10
                    )
                ax_gp.scatter(
                    X_ctx_gp[-1, 0], X_ctx_gp[-1, 1],
                    c="red", marker="x", s=100, linewidths=3,
                    zorder=11
                )
                
                # 标注全局最优位置
                ax_gp.scatter(
                    [np.pi, -np.pi, np.pi],
                    [2.275, 12.275, 2.275],
                    c="lime", marker="*", s=150, edgecolors='black',
                    linewidths=1, zorder=12
                )
                
                best_so_far = np.min(y_ctx_gp)
                ax_gp.set_title(f"Step {step}\nGP+RL (best={best_so_far:.2f})", fontsize=11)
                ax_gp.set_xlim(self.x1_min, self.x1_max)
                ax_gp.set_ylim(self.x2_min, self.x2_max)
                ax_gp.set_xlabel("x1", fontsize=11)
                
                if col == 0:
                    ax_gp.set_ylabel("x2", fontsize=11)
                else:
                    ax_gp.tick_params(left=False, labelleft=False)

        plt.tight_layout(rect=[0, 0, 0.92, 0.93])

        # 添加 colorbar
        if cf1 is not None:
            cax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
            cbar = fig.colorbar(cf1, cax=cax)
            cbar.set_label("Mean Prediction", fontsize=11)

        # 添加总标题
        final_best_tabpfn = np.min(y_tabpfn)
        final_best_gp = np.min(y_gp)
        plt.suptitle(
            f"Run {run_idx} (seed={seed}): "
            f"TabPFN+RL final={final_best_tabpfn:.4f}, GP+RL final={final_best_gp:.4f}, "
            f"Global Opt=0.3979",
            fontsize=13, y=0.98
        )

        save_path = os.path.join(save_dir, f"trajectory_run{run_idx:02d}_seed{seed}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  [Trajectory] 保存到 {save_path}")

    # ---------- 一点小工具：从策略 ckpt 名字里提取编号 ----------

    def get_policy_identifier(self):
        import re
        basename = os.path.basename(self.policy_path)
        match = re.search(r'agent_(\d+)', basename)
        if match:
            return match.group(1)
        return 'unknown'


# =============================
# 3. 画图 & 统计（直接沿用你原来的）
# =============================

def plot_comparison_results(
    rl_results,
    gp_rl_results,
    bo_gp_results,
    bo_tabpfn_results,
    save_dir='./results/eval_tabpfn_cnn',
    policy_id='',
):
    os.makedirs(save_dir, exist_ok=True)
    suffix = f"_policy_{policy_id}" if policy_id else ""

    num_runs = len(rl_results)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- 左图：所有轨迹（Spaghetti Plot） ---
    ax = axes[0]
    for i in range(num_runs):
        rl = np.minimum.accumulate(rl_results[i]['y_history'])
        gp_rl = np.minimum.accumulate(gp_rl_results[i]['y_history'])
        bo_gp = np.minimum.accumulate(bo_gp_results[i]['y_history'])
        bo_tab = np.minimum.accumulate(bo_tabpfn_results[i]['y_history'])

        ax.plot(rl, 'b-', alpha=0.1)
        ax.plot(gp_rl, color='purple', linestyle=':', alpha=0.1)
        ax.plot(bo_gp, 'r--', alpha=0.1)
        ax.plot(bo_tab, 'g-.', alpha=0.1)

    ax.plot([], [], 'b-', label='TabPFN+RL', linewidth=2)
    ax.plot([], [], color='purple', linestyle=':', label='GP+RL', linewidth=2)
    ax.plot([], [], 'r--', label='GP-BO', linewidth=2)
    ax.plot([], [], 'g-.', label='TabPFN-BO', linewidth=2)
    ax.axhline(y=0.3979, color='lime', linestyle=':', linewidth=2, label='Global Optimum')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Function Value')
    ax.set_title(f'Optimization Trails ({num_runs} runs)')
    ax.legend()
    ax.grid(True)

    # --- 右图：平均性能曲线（Mean ± SE） ---
    ax = axes[1]

    def get_curve_stats(res_list):
        max_len = max(len(r['y_history']) for r in res_list)
        all_curves = []
        for r in res_list:
            curve = np.minimum.accumulate(r['y_history'])
            if len(curve) < max_len:
                curve = np.pad(curve, (0, max_len - len(curve)), 'edge')
            all_curves.append(curve)
        all_curves = np.array(all_curves)
        mean = np.mean(all_curves, axis=0)
        std = np.std(all_curves, axis=0)
        se = std / np.sqrt(len(res_list))
        return mean, se

    rl_mean, rl_se = get_curve_stats(rl_results)
    gp_rl_mean, gp_rl_se = get_curve_stats(gp_rl_results)
    bo_gp_mean, bo_gp_se = get_curve_stats(bo_gp_results)
    bo_tab_mean, bo_tab_se = get_curve_stats(bo_tabpfn_results)

    steps = np.arange(len(rl_mean))

    ax.plot(steps, rl_mean, 'b-', linewidth=2.5, label='TabPFN+RL')
    ax.fill_between(steps, rl_mean - rl_se, rl_mean + rl_se, color='blue', alpha=0.15)

    ax.plot(steps, gp_rl_mean, color='purple', linestyle=':', linewidth=2.5, label='GP+RL')
    ax.fill_between(steps, gp_rl_mean - gp_rl_se, gp_rl_mean + gp_rl_se, color='purple', alpha=0.15)

    ax.plot(steps, bo_gp_mean, 'r--', linewidth=2.5, label='GP-BO')
    ax.fill_between(steps, bo_gp_mean - bo_gp_se, bo_gp_mean + bo_gp_se, color='red', alpha=0.15)

    ax.plot(steps, bo_tab_mean, 'g-.', linewidth=2.5, label='TabPFN-BO')
    ax.fill_between(steps, bo_tab_mean - bo_tab_se, bo_tab_mean + bo_tab_se, color='green', alpha=0.15)

    ax.axhline(y=0.3979, color='lime', linestyle=':', linewidth=2, label='Global Optimum')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Value (Mean ± SE)')
    ax.set_title('Average Performance')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'optimization_curves{suffix}.png'), dpi=150)
    plt.close()

    # --- 单独保存箱线图 ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    data = [
        [r['best_y'] for r in rl_results],
        [r['best_y'] for r in gp_rl_results],
        [r['best_y'] for r in bo_gp_results],
        [r['best_y'] for r in bo_tabpfn_results],
    ]
    ax2.boxplot(data, labels=['TabPFN+RL', 'GP+RL', 'GP-BO', 'TabPFN-BO'])
    ax2.axhline(y=0.3979, color='lime', linestyle=':', linewidth=2, label='Global Optimum')
    ax2.set_ylabel('Best Function Value')
    ax2.set_title('Final Performance Distribution')
    ax2.grid(True)
    plt.savefig(os.path.join(save_dir, f'performance_boxplot{suffix}.png'), dpi=150)
    plt.close()

    print(f"✓ 结果图已保存到 {save_dir} （curves & boxplot, 后缀: {suffix}）")


def print_statistics(rl_results, gp_rl_results, bo_gp_results, bo_tabpfn_results):
    def get_stats(res_list):
        vals = np.array([r['best_y'] for r in res_list])
        return vals.mean(), vals.std(), np.median(vals)

    print("\n" + "=" * 80)
    print("测试结果统计 (Mean ± Std | Median)")
    print("=" * 80)

    m, s, med = get_stats(rl_results)
    print(f"  TabPFN+RL : {m:.4f} ± {s:.4f} | {med:.4f}")

    m, s, med = get_stats(gp_rl_results)
    print(f"  GP+RL     : {m:.4f} ± {s:.4f} | {med:.4f}")

    m, s, med = get_stats(bo_gp_results)
    print(f"  GP-BO     : {m:.4f} ± {s:.4f} | {med:.4f}")

    m, s, med = get_stats(bo_tabpfn_results)
    print(f"  TabPFN-BO : {m:.4f} ± {s:.4f} | {med:.4f}")
    print("=" * 80)


# =============================
# 4. 主入口：在 Ground Truth Branin 上测试
# =============================



# 修改 test_on_ground_truth 函数

def test_on_ground_truth():
    print("=" * 80)
    print("Test on Ground Truth Branin (TabPFN + CNN PPO)")
    print("=" * 80)

    # 这里改成你实际的路径
    policy_path = '/gpfs/radev/pi/cohan/yz979/lin/test/runs/TabPFN_CNN_PPO_FIXED_1765460868/agent_70.pt'
    tabpfn_model_path = './model/finetuned_tabpfn_branin_family.ckpt'

    if not os.path.exists(policy_path):
        print(f"❌ 错误：找不到 RL 模型文件 {policy_path}")
        return
    if not os.path.exists(tabpfn_model_path):
        print(f"❌ 错误：找不到 TabPFN 模型文件 {tabpfn_model_path}")
        return

    tester = RLPolicyTesterCNN(policy_path, tabpfn_model_path)

    # 现在画 3×5：TabPFN+RL, GP+RL, Random+RL
    tester.debug_compare_means_three_methods(max_steps=20, seed=42)

    policy_id = tester.get_policy_identifier()
    print(f"策略标识: {policy_id}")

    rl_res, gp_rl_res, bo_gp_res, bo_tabpfn_res = [], [], [], []
    num_runs = 10
    max_steps = 20
    
    # === 新增：轨迹图保存目录 ===
    trajectory_save_dir = f"./results/trajectory_comparison_policy_{policy_id}"

    for i in range(num_runs):
        seed = 42 + i
        print(f"\nRun {i+1}/{num_runs} (seed={seed})")

        # 1. TabPFN + RL
        x_tabpfn, y_tabpfn = tester.run_rl_tabpfn_optimization(max_steps=max_steps, seed=seed, verbose=False)
        rl_res.append({'y_history': y_tabpfn, 'best_y': y_tabpfn.min()})

        # 2. GP + RL
        x_gp, y_gp = tester.run_gp_rl_optimization(max_steps=max_steps, seed=seed, verbose=False)
        gp_rl_res.append({'y_history': y_gp, 'best_y': y_gp.min()})
        
        print(
            f"  max |TabPFN+RL - GP+RL| = "
            f"{np.max(np.abs(rl_res[i]['y_history'] - gp_rl_res[i]['y_history'])):.4e}"
        )

        # === 新增：画轨迹对比图 ===
        tester.plot_trajectory_comparison(
            x_tabpfn, y_tabpfn,
            x_gp, y_gp,
            run_idx=i + 1,
            seed=seed,
            save_dir=trajectory_save_dir,
        )

        # 3. GP-BO
        x, y = tester.run_bo_gp_optimization(max_steps=max_steps, seed=seed, verbose=False)
        bo_gp_res.append({'y_history': y, 'best_y': y.min()})

        # 4. TabPFN-BO
        x, y = tester.run_bo_tabpfn_optimization(max_steps=max_steps, seed=seed, verbose=False)
        bo_tabpfn_res.append({'y_history': y, 'best_y': y.min()})

    print("\n")
    print_statistics(rl_res, gp_rl_res, bo_gp_res, bo_tabpfn_res)
    plot_comparison_results(
        rl_res,
        gp_rl_res,
        bo_gp_res,
        bo_tabpfn_res,
        policy_id=policy_id,
    )
    print(f"\n✅ 测试完成! 轨迹图保存在: {trajectory_save_dir}")


if __name__ == "__main__":
    test_on_ground_truth()