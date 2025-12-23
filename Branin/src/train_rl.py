"""
贝叶斯优化候选点选择的 PPO 实现

输入: 候选点的 [坐标, 预测均值, 预测标准差]
输出: 选择哪个候选点（离散动作）

使用 Self-Attention 处理候选点集合，保证排列不变性
"""

import os
import json
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, Any, Optional, Tuple
from scipy.optimize import minimize
import math
from collections import deque

# 忽略 Sobol 警告（已确保 n_candidates 是 2 的幂次方）
warnings.filterwarnings('ignore', message='The balance properties of Sobol')

from select_candidates import (
    SelectionConfig, 
    ObjectiveFunction, 
    select_candidates,
)


# ==================== Branin 变体函数 ====================
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


def branin_family_numpy(X: np.ndarray, variant_params: dict, device: str = "cpu") -> np.ndarray:
    """计算 Branin 变体函数值"""
    x_tensor = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        y = branin_family_torch(x_tensor, **variant_params).cpu().numpy()
    return y.astype(np.float32)


class BraninVariantFunction(ObjectiveFunction):
    """Branin 变体函数（用于 select_candidates）"""
    
    def __init__(self, variant_params: dict, device: str = "cpu"):
        self.variant_params = variant_params
        self._device = device
        self._lower = np.array([-5.0, 0.0], dtype=np.float32)
        self._upper = np.array([10.0, 15.0], dtype=np.float32)
        self._global_min = None
    
    @property
    def dim(self) -> int:
        return 2
    
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lower, self._upper
    
    @property
    def optimal_value(self) -> Optional[float]:
        if self._global_min is None:
            self._global_min = self._find_global_min()
        return self._global_min
    
    def _find_global_min(self) -> float:
        """搜索全局最小值（多起点优化，更可靠）"""
        # 1. 密集网格搜索
        x1 = np.linspace(-5, 10, 100)
        x2 = np.linspace(0, 15, 100)
        X1, X2 = np.meshgrid(x1, x2)
        grid_points = np.stack([X1.flatten(), X2.flatten()], axis=1).astype(np.float32)
        y_grid = self(grid_points)
        
        # 2. 选择多个起点（网格最优 + top-k 局部最优）
        sorted_indices = np.argsort(y_grid)
        n_starts = 20  # 多起点
        start_points = grid_points[sorted_indices[:n_starts]]
        
        # 3. 多起点 L-BFGS-B 优化
        def func(x):
            x = np.array(x, dtype=np.float32).reshape(1, 2)
            return float(self(x)[0])
        
        bounds = [(-5, 10), (0, 15)]
        best_min = float('inf')
        
        for x0 in start_points:
            try:
                res = minimize(func, x0, bounds=bounds, method="L-BFGS-B")
                if res.fun < best_min:
                    best_min = res.fun
            except:
                pass
        
        # 4. 确保返回的是网格搜索和优化中的最小值
        best_min = min(best_min, float(y_grid.min()))
        
        return best_min
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X).astype(np.float32)
        return branin_family_numpy(X, self.variant_params, self._device)


# ==================== Self-Attention 模块 ====================
class MultiHeadSelfAttention(nn.Module):
    """多头自注意力"""
    
    def __init__(self, hidden_dim, n_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_candidates, hidden_dim)
        Returns:
            out: (batch, n_candidates, hidden_dim)
        """
        batch, n, d = x.shape
        
        # 计算 Q, K, V
        q = self.q_proj(x).view(batch, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, n, self.n_heads, self.head_dim).transpose(1, 2)
        # shape: (batch, n_heads, n, head_dim)
        
        # 注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch, n_heads, n, n)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        out = torch.matmul(attn, v)  # (batch, n_heads, n, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, n, d)
        out = self.out_proj(out)
        
        return out


class TransformerBlock(nn.Module):
    """Transformer 块: Self-Attention + FFN"""
    
    def __init__(self, hidden_dim, n_heads=4, ffn_dim=None, dropout=0.1):
        super().__init__()
        ffn_dim = ffn_dim or hidden_dim * 2
        
        self.attn = MultiHeadSelfAttention(hidden_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        # Self-Attention + Residual
        x = x + self.attn(self.norm1(x))
        # FFN + Residual
        x = x + self.ffn(self.norm2(x))
        return x


class AttentionPooling(nn.Module):
    """Attention-based Pooling: 让网络学习关注哪些候选点"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_candidates, hidden_dim)
        Returns:
            pooled: (batch, hidden_dim)
        """
        # 计算注意力权重
        attn_weights = self.attention(x)  # (batch, n, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # 加权求和
        pooled = (x * attn_weights).sum(dim=1)  # (batch, hidden_dim)
        return pooled


# ==================== 策略网络 ====================
class CandidateSelector(nn.Module):
    """
    基于 Self-Attention 的候选点选择网络
    
    输入: (batch, n_candidates, feature_dim)
    输出: 
         - logits: (batch, n_candidates) 每个候选点的选择概率
         - value: (batch,) 状态价值
    """
    
    def __init__(self, feature_dim, hidden_dim=64, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        
        # 输入嵌入
        self.input_embed = nn.Linear(feature_dim, hidden_dim)
        
        # Self-Attention 层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, hidden_dim * 2, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Actor: 输出每个候选点的 score（排列等变）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Critic: 使用 Attention Pooling 聚合信息（替代 Mean Pooling）
        self.attn_pool = AttentionPooling(hidden_dim)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, candidates):
        """
        Args:
            candidates: (batch, n_candidates, feature_dim)
        Returns:
            logits: (batch, n_candidates)
            value: (batch,)
        """
        # 输入嵌入
        x = self.input_embed(candidates)  # (batch, n, hidden_dim)
        
        # Self-Attention 层
        for layer in self.transformer_layers:
            x = layer(x)
        
        x = self.norm(x)
        
        logits = self.actor(x).squeeze(-1)
        pooled = self.attn_pool(x)  # Attention Pooling 替代 Mean Pooling
        value = self.critic(pooled).squeeze(-1)
        
        return logits, value
    
    def get_action(self, candidates):
        """采样动作"""
        with torch.no_grad():
            logits, value = self.forward(candidates)
            dist = Categorical(logits=logits)
            action = dist.sample()#按照概率随机采样一个索引，评估时可以改为logits.argmax()
            log_prob = dist.log_prob(action)
        return action, log_prob, value
    
    def evaluate(self, candidates, actions):
        """评估动作"""
        logits, values = self.forward(candidates)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy


# ==================== PPO 算法 ====================
class PPO:
    """候选点选择的 PPO"""
    
    def __init__(
        self,
        feature_dim,
        hidden_dim=64,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        device="cpu",
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        self.policy = CandidateSelector(
            feature_dim, hidden_dim, n_layers, n_heads, dropout
        ).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def select_candidate(self, X_selected, pred_mean, pred_std, y_best=None, bounds=None):
        """
        从候选点中选择一个
        
        Args:
            X_selected: (n_candidates, coord_dim) 候选点坐标
            pred_mean: (n_candidates,) 预测均值
            pred_std: (n_candidates,) 预测标准差
            y_best: 当前最优值（用于计算相对均值）
            bounds: (lower, upper) 坐标边界（用于归一化坐标）
            
        Returns:
            action: 选中的候选点索引
            log_prob: 动作的 log 概率
            value: 状态价值估计
        """
        # 构建特征: [归一化坐标, 相对均值, 标准化标准差]
        features = self._build_features(X_selected, pred_mean, pred_std, y_best, bounds)
        features = features.unsqueeze(0).to(self.device)
        
        action, log_prob, value = self.policy.get_action(features)
        return action.item(), log_prob.item(), value.item()
    
    def _build_features(self, X_selected, pred_mean, pred_std, y_best=None, bounds=None):
        """
        构建归一化的相对特征
        
        Args:
            X_selected: (n, dim) 候选点坐标
            pred_mean: (n,) 预测均值
            pred_std: (n,) 预测标准差
            y_best: 当前最优值（用于计算相对均值）
            bounds: (lower, upper) 坐标边界（用于归一化坐标）
        """
        X = np.array(X_selected)
        mean = np.array(pred_mean)
        std = np.array(pred_std)
        
        # 1. 坐标归一化到 [0, 1]
        if bounds is not None:
            lower, upper = bounds
            X_norm = (X - lower) / (upper - lower + 1e-8)
        else:
            X_norm = X
        
        # 2. 均值转为相对特征：相对于 y_best 的差值，并标准化
        if y_best is not None:
            mean_rel = mean - y_best  # 相对于当前最优的差异（越小越好）
        else:
            mean_rel = mean
        
        # 在候选点间标准化
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
        
        return torch.cat([X_t, mean_t, std_t], dim=-1)
    
    def compute_gae(self, rewards, values, dones, last_value):
        """计算 GAE"""
        advantages = []
        gae = 0
        
        values = values + [last_value]
        
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values[:-1]).to(self.device)
        
        return advantages, returns
    
    def update(self, rollout, n_epochs=4, batch_size=64):
        states = rollout["states"]
        actions = torch.LongTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        
        # 计算 GAE
        advantages, returns = self.compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            rollout["last_value"],
        )
        
        # 标准化 advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        n_samples = len(states)
        
        for _ in range(n_epochs):
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                
                # 批次数据
                b_states = torch.stack([states[i] for i in batch_idx]).to(self.device)
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]
                
                # 评估
                new_log_probs, new_values, entropy = self.policy.evaluate(b_states, b_actions)
                
                # PPO clip loss
                ratio = (new_log_probs - b_old_log_probs).exp()
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(new_values, b_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # 总 loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
        }


# ==================== Rollout Buffer ====================
class RolloutBuffer:
    """存储一个 episode 的经验"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(float(done))
    
    def get(self, last_value=0.0):
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "values": self.values,
            "log_probs": self.log_probs,
            "dones": self.dones,
            "last_value": last_value,
        }
    
    def __len__(self):
        return len(self.states)


# ==================== 训练环境 ====================
class BraninBOEnv:
    """贝叶斯优化环境"""
    
    def __init__(
        self,
        variants_path: str,
        model_path: str,
        max_steps: int = 20,
        n_init_context: int = 2,
        selection_config: Optional[SelectionConfig] = None,
        device: str = "cpu",
        seed: int = 42,
    ):
        # 加载变体
        data = np.load(variants_path, allow_pickle=True)
        self.variants = data["variants"].tolist()
        self.num_variants = len(self.variants)
        
        self.model_path = model_path
        self.max_steps = max_steps
        self.n_init_context = n_init_context
        self.device = device
        self.rng = np.random.default_rng(seed)
        
        self.selection_config = selection_config or SelectionConfig(
            n_candidates=512,  # 必须是 2 的幂次方，Sobol 序列要求
            top_k_ei=20,
            top_k_var=10,
            n_random=10,
            verbose=False,
            device=device,
        )
        
        # Episode 状态
        self.current_func = None
        self.X_context = None
        self.y_context = None
        self.best_y = None
        self.global_min = None
        self.step_count = 0
        self._current_candidates = None
    
    def reset(self, seed=None):
        """重置环境，随机选择一个变体"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # 随机选择变体
        variant_idx = int(self.rng.integers(0, self.num_variants))
        variant_params = self.variants[variant_idx]
        
        self.current_func = BraninVariantFunction(variant_params, self.device)
        self.global_min = self.current_func.optimal_value
        
        # 随机初始化上下文点
        lower, upper = self.current_func.bounds
        self.X_context = self.rng.uniform(
            lower, upper, 
            size=(self.n_init_context, self.current_func.dim)
        ).astype(np.float32)
        self.y_context = self.current_func(self.X_context)
        
        self.best_y = float(self.y_context.min())
        self.step_count = 0
        self._current_candidates = self._get_candidates()
        # 获取初始候选点信息
        return self._current_candidates
    
    def _get_candidates(self):
        """调用 select_candidates 获取候选点"""
        # 更新 seed 避免每次相同
        self.selection_config.random_seed = int(self.rng.integers(0, 100000))
        self.selection_config.verbose = False
        result = select_candidates(
            func=self.current_func,
            model_path=self.model_path,
            config=self.selection_config,
            X_context=self.X_context,
        )
        
        return {
            "X_selected": result["X_selected"],
            "pred_mean": result["y_selected_pred_mean"],
            "pred_std": result["y_selected_pred_std"],
        }
    
    def step(self, action_idx: int):
        """
        执行动作：选择第 action_idx 个候选点进行评估
        
        Returns:
            next_obs: 下一个状态的候选点信息
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 获取当前候选点
        # candidates = self._get_candidates()
        # X_selected = candidates["X_selected"]
        X_selected = self._current_candidates["X_selected"]
        
        # 获取选中的点
        x_new = X_selected[action_idx:action_idx+1]
        y_new = self.current_func(x_new)[0]
        
        # 更新上下文
        self.X_context = np.vstack([self.X_context, x_new])
        self.y_context = np.concatenate([self.y_context, [y_new]])
        
        # 更新最优值
        old_best = self.best_y
        self.best_y = min(self.best_y, float(y_new))
        
        # 计算奖励
        reward = self._compute_reward(y_new, old_best)
        
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        # 获取下一个状态
        if not done:
            self._current_candidates = self._get_candidates()
            next_obs = self._current_candidates
        else:
            next_obs = None
        
        info = {
            "y_new": y_new,
            "best_y": self.best_y,
            "global_min": self.global_min,
            "regret": self.best_y - self.global_min,
            "step": self.step_count,
        }
        
        return next_obs, reward, done, info
    
    def _compute_reward(self, y_new: float, old_best: float) -> float:
        """
        计算奖励：基于 improvement + 时间惩罚
        
        设计目标：
        - 找到更好的点给正奖励（体现"好"）
        - 每步有小惩罚（体现"快"，鼓励尽早找到好点）
        - 奖励与 improvement 成正比
        """
        # 基础时间惩罚：每步 -0.1，鼓励快速收敛
        reward = -0.1
        
        # 计算 improvement
        improvement = max(0, old_best - self.best_y)
        
        if improvement > 0:
            # 找到更好的点，给予正奖励
            # 使用相对 improvement：improvement / (old_best - global_min)
            # 这样无论 regret 大小，都能得到 0-1 之间的相对改进
            gap = old_best - self.global_min
            if gap > 1e-8:
                relative_improvement = improvement / gap
                # 奖励 = 1 + 额外奖励（最多 +5）
                reward += 1.0 + 5.0 * relative_improvement
            else:
                # 已经接近最优，给予固定奖励
                reward += 2.0
        
        # 额外奖励：如果 regret 很小（接近最优）
        regret = self.best_y - self.global_min
        if regret < 0.1:
            reward += 0.5
        if regret < 0.01:
            reward += 1.0
        
        return float(reward)


# ==================== 训练监控类 ====================
class TrainingMonitor:
    """训练监控：记录指标、保存日志、支持 TensorBoard"""
    
    def __init__(self, save_dir: str, use_tensorboard: bool = True):
        self.save_dir = save_dir
        self.use_tensorboard = use_tensorboard
        self.writer = None
        
        # 尝试导入 TensorBoard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))
                print(f"TensorBoard 日志目录: {os.path.join(save_dir, 'tensorboard')}")
            except ImportError:
                print("警告: tensorboard 未安装，将只使用文本日志")
                self.use_tensorboard = False
        
        # 指标历史
        self.metrics_history = {
            'episode_reward': [],
            'episode_regret': [],
            'best_regret': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
        }
        
        # 滑动窗口统计
        self.window_size = 100
        self.reward_window = deque(maxlen=self.window_size)
        self.regret_window = deque(maxlen=self.window_size)
        
        # 最佳指标
        self.best_avg_regret = float('inf')
        self.best_episode = 0
        
        # 计时
        self.start_time = time.time()
        self.episode_start_time = None
    
    def start_episode(self):
        """开始一个 episode"""
        self.episode_start_time = time.time()
    
    def log_episode(self, episode: int, reward: float, regret: float, info: dict = None):
        """记录 episode 结果"""
        self.metrics_history['episode_reward'].append(reward)
        self.metrics_history['episode_regret'].append(regret)
        self.reward_window.append(reward)
        self.regret_window.append(regret)
        
        # 更新最佳指标
        if len(self.regret_window) >= 10:
            avg_regret = np.mean(self.regret_window)
            if avg_regret < self.best_avg_regret:
                self.best_avg_regret = avg_regret
                self.best_episode = episode
        
        self.metrics_history['best_regret'].append(self.best_avg_regret)
        
        # TensorBoard 记录
        if self.writer:
            self.writer.add_scalar('Episode/Reward', reward, episode)
            self.writer.add_scalar('Episode/Regret', regret, episode)
            if len(self.reward_window) > 0:
                self.writer.add_scalar('Episode/AvgReward_100', np.mean(self.reward_window), episode)
                self.writer.add_scalar('Episode/AvgRegret_100', np.mean(self.regret_window), episode)
            if info:
                self.writer.add_scalar('Episode/BestY', info.get('best_y', 0), episode)
                self.writer.add_scalar('Episode/GlobalMin', info.get('global_min', 0), episode)
    
    def log_update(self, episode: int, losses: dict):
        """记录 PPO 更新"""
        self.metrics_history['policy_loss'].append(losses.get('policy_loss', 0))
        self.metrics_history['value_loss'].append(losses.get('value_loss', 0))
        self.metrics_history['entropy'].append(losses.get('entropy', 0))
        
        if self.writer:
            self.writer.add_scalar('Loss/Policy', losses.get('policy_loss', 0), episode)
            self.writer.add_scalar('Loss/Value', losses.get('value_loss', 0), episode)
            self.writer.add_scalar('Loss/Entropy', losses.get('entropy', 0), episode)
    
    def print_progress(self, episode: int, total_episodes: int, update_every: int):
        """打印训练进度"""
        elapsed = time.time() - self.start_time
        eps_per_sec = episode / elapsed if elapsed > 0 else 0
        eta = (total_episodes - episode) / eps_per_sec if eps_per_sec > 0 else 0
        
        avg_reward = np.mean(list(self.reward_window)) if self.reward_window else 0
        avg_regret = np.mean(list(self.regret_window)) if self.regret_window else 0
        min_regret = min(list(self.regret_window)) if self.regret_window else 0
        
        # 获取最近的 loss
        policy_loss = self.metrics_history['policy_loss'][-1] if self.metrics_history['policy_loss'] else 0
        value_loss = self.metrics_history['value_loss'][-1] if self.metrics_history['value_loss'] else 0
        entropy = self.metrics_history['entropy'][-1] if self.metrics_history['entropy'] else 0
        
        print(f"\n{'='*70}")
        print(f"Episode {episode}/{total_episodes} | 进度: {episode/total_episodes*100:.1f}%")
        print(f"时间: {elapsed/60:.1f}分钟 | ETA: {eta/60:.1f}分钟 | 速度: {eps_per_sec:.2f} ep/s")
        print(f"-"*70)
        print(f"最近{len(self.reward_window)}轮: Avg Reward={avg_reward:.3f} | Avg Regret={avg_regret:.4f} | Min Regret={min_regret:.4f}")
        print(f"历史最佳: Avg Regret={self.best_avg_regret:.4f} @ Episode {self.best_episode}")
        print(f"Loss: Policy={policy_loss:.4f} | Value={value_loss:.4f} | Entropy={entropy:.4f}")
        print(f"{'='*70}")
    
    def save_metrics(self):
        """保存指标到文件"""
        metrics_path = os.path.join(self.save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # 保存为 numpy 格式方便绘图
        np_path = os.path.join(self.save_dir, 'metrics.npz')
        np.savez(np_path, **{k: np.array(v) for k, v in self.metrics_history.items()})
    
    def close(self):
        """关闭监控"""
        self.save_metrics()
        if self.writer:
            self.writer.close()
        
        total_time = time.time() - self.start_time
        print(f"\n训练完成！总耗时: {total_time/60:.1f}分钟")
        print(f"最佳平均 Regret: {self.best_avg_regret:.4f} @ Episode {self.best_episode}")
        print(f"指标已保存到: {self.save_dir}")


# ==================== 训练函数 ====================
def train(
    variants_path: str = "./data/branin_family_variants.npz",
    model_path: str = "./model/finetuned_tabpfn_branin_family.ckpt",
    total_episodes: int = 1000,
    max_steps: int = 20,
    update_every: int = 10,
    save_every: int = 100,
    save_dir: str = "./runs/ppo_bo",
    seed: int = 42,
    use_tensorboard: bool = True,
):
    """训练 PPO"""
    os.makedirs(save_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"PPO 贝叶斯优化训练")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Max Steps per Episode: {max_steps}")
    print(f"Update Every: {update_every} episodes")
    print(f"Save Dir: {save_dir}")
    print(f"{'='*70}\n")
    
    # 创建监控器
    monitor = TrainingMonitor(save_dir, use_tensorboard=use_tensorboard)
    
    # 创建环境
    env = BraninBOEnv(
        variants_path=variants_path,
        model_path=model_path,
        max_steps=max_steps,
        device=device,
        seed=seed,
    )
    print(f"环境已创建: {env.num_variants} 个 Branin 变体")
    
    # 创建 PPO
    coord_dim = 2
    feature_dim = coord_dim + 2  # 坐标 + 均值 + 标准差
    
    ppo = PPO(
        feature_dim=feature_dim,
        hidden_dim=64,
        n_layers=2,
        n_heads=4,
        device=device,
    )
    print(f"PPO 网络参数量: {sum(p.numel() for p in ppo.policy.parameters()):,}")
    
    # Rollout buffer
    buffer = RolloutBuffer()
    
    # 训练统计
    episode_rewards = []
    episode_regrets = []
    
    print(f"\n开始训练...\n")
    
    for episode in range(1, total_episodes + 1):
        monitor.start_episode()
        
        # 重置环境
        obs = env.reset()
        episode_reward = 0
        
        # 获取坐标边界用于特征归一化
        bounds = env.current_func.bounds
        
        for step in range(max_steps):
            X_selected = obs["X_selected"]
            pred_mean = obs["pred_mean"]
            pred_std = obs["pred_std"]
            y_best = env.best_y  # 当前最优值
            
            # 选择候选点（传入 y_best 和 bounds 用于特征归一化）
            action, log_prob, value = ppo.select_candidate(
                X_selected, pred_mean, pred_std, y_best=y_best, bounds=bounds
            )
            
            # 执行动作
            next_obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            
            # 存储经验（使用归一化特征）
            features = ppo._build_features(X_selected, pred_mean, pred_std, y_best=y_best, bounds=bounds)
            buffer.add(features, action, reward, value, log_prob, done)
            
            if done:
                break
            
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_regrets.append(info["regret"])
        
        # 记录 episode 指标
        monitor.log_episode(episode, episode_reward, info["regret"], info)
        
        # PPO 更新
        if episode % update_every == 0 and len(buffer) > 0:
            rollout = buffer.get(last_value=0.0)
            losses = ppo.update(rollout, n_epochs=4, batch_size=32)
            buffer.reset()
            
            # 记录更新指标
            monitor.log_update(episode, losses)
            
            # 打印进度
            monitor.print_progress(episode, total_episodes, update_every)
        
        # 保存模型
        if episode % save_every == 0:
            save_path = os.path.join(save_dir, f"ppo_ep{episode}.pt")
            torch.save(ppo.policy.state_dict(), save_path)
            print(f"\n模型已保存: {save_path}")
            
            # 保存当前最佳模型
            if monitor.best_avg_regret < float('inf'):
                best_path = os.path.join(save_dir, "ppo_best.pt")
                torch.save(ppo.policy.state_dict(), best_path)
            
            # 保存指标
            monitor.save_metrics()
    
    # 保存最终模型
    final_path = os.path.join(save_dir, "ppo_final.pt")
    torch.save(ppo.policy.state_dict(), final_path)
    print(f"\n最终模型已保存: {final_path}")
    
    # 关闭监控
    monitor.close()
    
    return ppo, monitor


# ==================== 示例 ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants_path", type=str, default="./data/branin_family_variants.npz")
    parser.add_argument("--model_path", type=str, default="./model/finetuned_tabpfn_branin_family.ckpt")
    parser.add_argument("--total_episodes", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--update_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./runs/ppo_bo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_tensorboard", action="store_true", default=True)
    parser.add_argument("--no_tensorboard", dest="use_tensorboard", action="store_false")
    
    args = parser.parse_args()
    
    train(
        variants_path=args.variants_path,
        model_path=args.model_path,
        total_episodes=args.total_episodes,
        max_steps=args.max_steps,
        update_every=args.update_every,
        save_every=args.save_every,
        save_dir=args.save_dir,
        seed=args.seed,
        use_tensorboard=args.use_tensorboard,
    )
