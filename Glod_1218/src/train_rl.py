"""
改进版 PPO 训练 - 针对 Goldstein-Price 函数

改进点:
1. 更丰富的特征：加入 EI 值、当前步数、PI 值等
2. 更好的奖励设计：加入 shaping reward
3. 更大的训练规模
4. 更好的探索策略
"""
import os
import math
import json
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Dict, Any, Optional, Tuple
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from collections import deque
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore', message='The balance properties of Sobol')

from select_candidates import (
    SelectionConfig,
    ObjectiveFunction,
    compute_std_from_tabpfn_output,
    compute_ei,  # 导入 EI 计算
)
from tabpfn import TabPFNRegressor
from scipy.optimize import minimize


# ==================== Goldstein-Price 变体函数 ====================
def goldstein_price_family_torch(
    x: torch.Tensor,
    dx1: float = 0.0, dx2: float = 0.0,
    sx1: float = 1.0, sx2: float = 1.0,
    alpha: float = 1.0, beta: float = 0.0,
) -> torch.Tensor:
    """
    Goldstein-Price 函数族（支持变体参数）

    标准定义域: x1, x2 ∈ [-2, 2]
    全局最小值: f(0, -1) = 3
    """
    x1 = x[..., 0]
    x2 = x[..., 1]

    # 应用变换
    x1_t = sx1 * x1 + dx1
    x2_t = sx2 * x2 + dx2

    # Goldstein-Price 公式
    term1 = 1 + (x1_t + x2_t + 1)**2 * (19 - 14*x1_t + 3*x1_t**2 - 14*x2_t + 6*x1_t*x2_t + 3*x2_t**2)
    term2 = 30 + (2*x1_t - 3*x2_t)**2 * (18 - 32*x1_t + 12*x1_t**2 + 48*x2_t - 36*x1_t*x2_t + 27*x2_t**2)

    y = term1 * term2
    return alpha * y + beta


def goldstein_price_family_numpy(X: np.ndarray, variant_params: dict, device: str = "cpu") -> np.ndarray:
    """计算 Goldstein-Price 变体函数值"""
    x_tensor = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        y = goldstein_price_family_torch(x_tensor, **variant_params).cpu().numpy()
    return y.astype(np.float32)


class GoldsteinPriceVariantFunction(ObjectiveFunction):
    """Goldstein-Price 变体函数"""

    def __init__(self, variant_params: dict, device: str = "cpu"):
        self.variant_params = variant_params
        self._device = device
        self._lower = np.array([-2.0, -2.0], dtype=np.float32)
        self._upper = np.array([2.0, 2.0], dtype=np.float32)
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
        """搜索全局最小值"""
        x1 = np.linspace(-2, 2, 100)
        x2 = np.linspace(-2, 2, 100)
        X1, X2 = np.meshgrid(x1, x2)
        grid_points = np.stack([X1.flatten(), X2.flatten()], axis=1).astype(np.float32)
        y_grid = self(grid_points)

        sorted_indices = np.argsort(y_grid)
        n_starts = 20
        start_points = grid_points[sorted_indices[:n_starts]]

        def func(x):
            x = np.array(x, dtype=np.float32).reshape(1, 2)
            return float(self(x)[0])

        bounds = [(-2, 2), (-2, 2)]
        best_min = float('inf')

        for x0 in start_points:
            try:
                res = minimize(func, x0, bounds=bounds, method="L-BFGS-B")
                if res.fun < best_min:
                    best_min = res.fun
            except:
                pass

        best_min = min(best_min, float(y_grid.min()))
        return best_min

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X).astype(np.float32)
        return goldstein_price_family_numpy(X, self.variant_params, self._device)


# ==================== 注意力模块 ====================
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

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) optional padding mask (1=valid, 0=padding)
        Returns:
            out: (batch, seq_len, hidden_dim)
        """
        batch, n, d = x.shape

        q = self.q_proj(x).view(batch, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, n, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, n, d)
        out = self.out_proj(out)

        return out


class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力: Query 来自一个序列，Key/Value 来自另一个序列"""

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

    def forward(self, query, key_value, kv_mask=None):
        """
        Args:
            query: (batch, n_query, hidden_dim) - 候选点
            key_value: (batch, n_kv, hidden_dim) - 上下文点
            kv_mask: (batch, n_kv) optional mask for key/value (1=valid, 0=padding)
        Returns:
            out: (batch, n_query, hidden_dim)
        """
        batch, n_q, d = query.shape
        n_kv = key_value.shape[1]

        q = self.q_proj(query).view(batch, n_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch, n_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch, n_kv, self.n_heads, self.head_dim).transpose(1, 2)

        # (batch, n_heads, n_q, n_kv)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if kv_mask is not None:
            # kv_mask: (batch, n_kv) -> (batch, 1, 1, n_kv)
            kv_mask = kv_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(kv_mask == 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (batch, n_heads, n_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, n_q, d)
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

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-Attention 块: Cross-Attention + FFN"""

    def __init__(self, hidden_dim, n_heads=4, ffn_dim=None, dropout=0.1):
        super().__init__()
        ffn_dim = ffn_dim or hidden_dim * 2

        self.cross_attn = MultiHeadCrossAttention(hidden_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query, key_value, kv_mask=None):
        # Cross-Attention + Residual
        query = query + self.cross_attn(self.norm1(query), self.norm_kv(key_value), kv_mask)
        # FFN + Residual
        query = query + self.ffn(self.norm2(query))
        return query


class AttentionPooling(nn.Module):
    """Attention-based Pooling"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) optional (1=valid, 0=padding)
        Returns:
            pooled: (batch, hidden_dim)
        """
        attn_weights = self.attention(x)  # (batch, n, 1)

        if mask is not None:
            mask = mask.unsqueeze(-1)  # (batch, n, 1)
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        return pooled


# ==================== 改进的双塔网络 ====================
class ImprovedDualTowerSelector(nn.Module):
    """
    改进的双塔 Cross-Attention 候选点选择网络

    改进点:
    1. Candidate 特征增加 EI, PI, 相对排名
    2. 加入当前步数信息
    3. 更深的网络
    """

    def __init__(
        self,
        coord_dim: int = 2,
        hidden_dim: int = 128,  # 增大隐藏维度
        n_self_attn_layers: int = 3,
        n_cross_attn_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_steps: int = 20,
    ):
        super().__init__()

        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps

        # Context 特征: [x, y] -> (coord_dim + 1)
        context_input_dim = coord_dim + 1

        # Candidate 特征: [x, μ, σ, ei, pi, rank, remaining_budget] -> (coord_dim + 6)
        candidate_input_dim = coord_dim + 6

        # Embedding layers
        self.context_embed = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.candidate_embed = nn.Sequential(
            nn.Linear(candidate_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Step embedding
        self.step_embed = nn.Embedding(max_steps + 1, hidden_dim)

        # Context 塔: Self-Attention
        self.context_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_self_attn_layers)
        ])

        # Candidate 塔: Self-Attention + Cross-Attention
        self.candidate_self_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_self_attn_layers)
        ])

        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_cross_attn_layers)
        ])

        # Actor Head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Critic Head (使用 attention pooling)
        self.context_pool = AttentionPooling(hidden_dim)
        self.candidate_pool = AttentionPooling(hidden_dim)

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, context, candidates, step, context_mask=None):
        """
        Args:
            context: (batch, n_context, context_dim)
            candidates: (batch, n_candidates, candidate_dim)
            step: (batch,) 当前步数
            context_mask: optional
        """
        batch = context.shape[0]

        # Embedding
        ctx_emb = self.context_embed(context)
        cand_emb = self.candidate_embed(candidates)

        # 加入步数信息
        step_emb = self.step_embed(step)  # (batch, hidden_dim)
        ctx_emb = ctx_emb + step_emb.unsqueeze(1)
        cand_emb = cand_emb + step_emb.unsqueeze(1)

        # Context 塔
        for layer in self.context_layers:
            ctx_emb = layer(ctx_emb, mask=context_mask)

        # Candidate Self-Attention
        for layer in self.candidate_self_layers:
            cand_emb = layer(cand_emb)

        # Cross-Attention
        for layer in self.cross_layers:
            cand_emb = layer(cand_emb, ctx_emb, kv_mask=context_mask)

        # Actor: 每个候选点的 logit
        logits = self.actor_head(cand_emb).squeeze(-1)  # (batch, n_candidates)

        # Critic: pooled features
        ctx_pooled = self.context_pool(ctx_emb, mask=context_mask)
        cand_pooled = self.candidate_pool(cand_emb)

        combined = torch.cat([ctx_pooled, cand_pooled], dim=-1)
        value = self.critic_head(combined).squeeze(-1)

        return logits, value

    def get_action(self, context, candidates, step, context_mask=None):
        logits, value = self.forward(context, candidates, step, context_mask)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate(self, context, candidates, step, actions, context_mask=None):
        logits, value = self.forward(context, candidates, step, context_mask)

        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, entropy, value


# ==================== 改进的 PPO ====================
class ImprovedPPO:
    """改进的 PPO"""

    def __init__(
        self,
        coord_dim: int = 2,
        hidden_dim: int = 128,
        n_self_attn_layers: int = 3,
        n_cross_attn_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_steps: int = 20,
        lr: float = 1e-4,  # 降低学习率
        gamma: float = 0.95,  # 降低折扣因子，鼓励更快找到最优
        lam: float = 0.9,  # GAE lambda，降低以减少方差
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.02,  # 增加探索
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.max_steps = max_steps

        self.policy = ImprovedDualTowerSelector(
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            n_self_attn_layers=n_self_attn_layers,
            n_cross_attn_layers=n_cross_attn_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_steps=max_steps,
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2
        )

    def select_candidate(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_candidates: np.ndarray,
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
        current_step: int,
    ) -> Tuple[int, float, float]:
        """选择候选点"""
        context_feat, candidate_feat = self._build_features(
            X_context, y_context, X_candidates, pred_mean, pred_std, bounds, current_step
        )

        context_feat = context_feat.unsqueeze(0).to(self.device)
        candidate_feat = candidate_feat.unsqueeze(0).to(self.device)
        step_tensor = torch.tensor([current_step], device=self.device)

        action, log_prob, value = self.policy.get_action(
            context_feat, candidate_feat, step_tensor
        )
        return action.item(), log_prob.item(), value.item()

    def _build_features(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_candidates: np.ndarray,
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
        current_step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建改进的特征

        Context: [x_norm, y_norm]
        Candidate: [x_norm, μ_norm, σ_norm, ei_norm, pi_norm, rank_norm, remaining_budget]
        """
        lower, upper = bounds

        # ========== Context 特征 ==========
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

        # ========== Candidate 特征 ==========
        X_cand_norm = (X_candidates - lower) / (upper - lower + 1e-8)

        y_best = y_context.min()

        # 均值相对差值
        mean_rel = pred_mean - y_best
        mean_std_val = mean_rel.std()
        if mean_std_val > 1e-8:
            mean_norm = (mean_rel - mean_rel.mean()) / mean_std_val
        else:
            mean_norm = np.zeros_like(mean_rel)

        # 标准差
        std_std_val = pred_std.std()
        if std_std_val > 1e-8:
            std_norm = (pred_std - pred_std.mean()) / std_std_val
        else:
            std_norm = np.zeros_like(pred_std)

        # 计算 EI
        ei_values = compute_ei(pred_mean, pred_std, y_best, xi=0.01)
        ei_max = ei_values.max()
        if ei_max > 1e-8:
            ei_norm = ei_values / ei_max
        else:
            ei_norm = np.zeros_like(ei_values)

        # 计算 PI (Probability of Improvement)
        imp = y_best - pred_mean
        Z = np.zeros_like(pred_mean)
        mask = pred_std > 1e-8
        Z[mask] = imp[mask] / pred_std[mask]
        pi_values = norm.cdf(Z)
        pi_norm = pi_values  # PI 已经在 [0, 1]

        # 计算 EI 排名 (归一化)
        n_candidates = len(X_candidates)
        ei_rank = np.argsort(np.argsort(-ei_values))  # 降序排名
        rank_norm = (ei_rank / (n_candidates - 1)).astype(np.float32) if n_candidates > 1 else np.zeros(n_candidates, dtype=np.float32)

        # 剩余 budget (归一化到 [0, 1])
        remaining_budget = (self.max_steps - current_step) / self.max_steps
        remaining_budget_arr = np.full(n_candidates, remaining_budget, dtype=np.float32)

        candidate_feat = np.concatenate([
            X_cand_norm,
            mean_norm.reshape(-1, 1),
            std_norm.reshape(-1, 1),
            ei_norm.reshape(-1, 1),
            pi_norm.reshape(-1, 1),
            rank_norm.reshape(-1, 1),
            remaining_budget_arr.reshape(-1, 1),
        ], axis=-1)

        return (
            torch.FloatTensor(context_feat),
            torch.FloatTensor(candidate_feat)
        )

    def compute_gae(self, rewards, values, dones, last_value):
        advantages = []
        gae = 0
        values = values + [last_value]

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, rollout, n_epochs=4, batch_size=64):
        context_feats = rollout["context_feats"]
        candidate_feats = rollout["candidate_feats"]
        steps = rollout["steps"]
        actions = torch.LongTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        advantages = torch.FloatTensor(rollout["advantages"]).to(self.device)
        returns = torch.FloatTensor(rollout["returns"]).to(self.device)

        # 标准化 advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = len(actions)
        indices = np.arange(n_samples)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]

                # Pad sequences
                batch_ctx = self._pad_sequences([context_feats[i] for i in batch_indices])
                batch_cand = self._pad_sequences([candidate_feats[i] for i in batch_indices])
                batch_steps = torch.LongTensor([steps[i] for i in batch_indices]).to(self.device)
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 计算 context mask (1=valid, 0=padding)
                # padding 的位置所有特征都是 0，所以 sum(dim=-1) == 0
                batch_ctx_mask = (batch_ctx.abs().sum(dim=-1) > 1e-8).float()

                # Forward
                log_probs, entropy, values = self.policy.evaluate(
                    batch_ctx, batch_cand, batch_steps, batch_actions, batch_ctx_mask
                )

                # Policy loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = ((values - batch_returns) ** 2).mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        self.scheduler.step()

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "lr": self.optimizer.param_groups[0]['lr'],
        }

    def _pad_sequences(self, sequences):
        max_len = max(s.shape[0] for s in sequences)
        feat_dim = sequences[0].shape[1]
        batch = len(sequences)

        padded = torch.zeros(batch, max_len, feat_dim, device=self.device)
        for i, seq in enumerate(sequences):
            padded[i, :seq.shape[0], :] = seq.to(self.device)

        return padded


# ==================== 改进的奖励函数 ====================
class ImprovedGoldsteinPriceBOEnv:
    """改进的贝叶斯优化环境 - Goldstein-Price 函数"""

    def __init__(
        self,
        variants_path: str,
        model_path: str,
        max_steps: int = 20,
        n_init_context: int = 2,
        n_candidates: int = 128,
        device: str = "cpu",
        seed: int = 42,
    ):
        data = np.load(variants_path, allow_pickle=True)
        self.variants = data["variants"].tolist()
        self.num_variants = len(self.variants)

        self.model_path = model_path
        self.max_steps = max_steps
        self.n_init_context = n_init_context
        self.n_candidates = n_candidates
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.regressor = None
        self._init_regressor()

        self.current_func = None
        self.X_context = None
        self.y_context = None
        self.best_y = None
        self.global_min = None
        self.step_count = 0
        self.initial_regret = None

        # 缓存当前候选点，避免 step() 时重新生成
        self._cached_candidates = None
        self._cached_pred_mean = None
        self._cached_pred_std = None

    def _init_regressor(self):
        regressor_kwargs = {
            "device": self.device,
            "n_estimators": 1,
            "random_state": 42,
            "inference_precision": torch.float32,
            "ignore_pretraining_limits": True,
        }
        if self.model_path is not None:
            regressor_kwargs["model_path"] = self.model_path
        self.regressor = TabPFNRegressor(**regressor_kwargs)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        variant_idx = int(self.rng.integers(0, self.num_variants))
        variant_params = self.variants[variant_idx]

        self.current_func = GoldsteinPriceVariantFunction(variant_params, self.device)
        self.global_min = self.current_func.optimal_value

        lower, upper = self.current_func.bounds
        self.X_context = self.rng.uniform(
            lower, upper,
            size=(self.n_init_context, self.current_func.dim)
        ).astype(np.float32)
        self.y_context = self.current_func(self.X_context)

        self.best_y = float(self.y_context.min())
        self.initial_regret = self.best_y - self.global_min
        self.step_count = 0

        return self._get_observation()

    def _get_observation(self):
        """生成观测，并缓存候选点供 step() 使用"""
        lower, upper = self.current_func.bounds

        sobol_seed = int(self.rng.integers(0, 100000))
        sobol_sampler = Sobol(d=self.current_func.dim, scramble=True, seed=sobol_seed)
        sobol_samples = sobol_sampler.random(self.n_candidates)
        X_candidates = (sobol_samples * (upper - lower) + lower).astype(np.float32)

        self.regressor.fit(self.X_context, self.y_context)
        full_out = self.regressor.predict(X_candidates, output_type="full")
        pred_mean = full_out.get("mean", None)
        pred_std = compute_std_from_tabpfn_output(full_out)

        # 缓存候选点，供 step() 使用
        self._cached_candidates = X_candidates
        self._cached_pred_mean = pred_mean
        self._cached_pred_std = pred_std

        return {
            "X_context": self.X_context.copy(),
            "y_context": self.y_context.copy(),
            "X_candidates": X_candidates,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "step": self.step_count,
        }

    def step(self, action_idx: int):
        """
        执行动作：选择第 action_idx 个候选点进行评估

        重要：使用缓存的候选点，而不是重新生成！
        这确保了 RL 选择的 action 对应的点就是实际评估的点。
        """
        # 使用缓存的候选点（由之前的 _get_observation() 生成）
        if self._cached_candidates is None:
            raise RuntimeError("step() 调用前必须先调用 reset() 或确保有缓存的候选点")

        X_candidates = self._cached_candidates
        pred_mean = self._cached_pred_mean
        pred_std = self._cached_pred_std

        x_new = X_candidates[action_idx:action_idx+1]
        y_new = self.current_func(x_new)[0]

        old_best = self.best_y
        self.best_y = min(self.best_y, float(y_new))

        self.X_context = np.vstack([self.X_context, x_new])
        self.y_context = np.concatenate([self.y_context, [y_new]])

        self.step_count += 1
        done = self.step_count >= self.max_steps

        # 改进的奖励计算
        reward = self._compute_improved_reward(
            y_new, old_best, pred_mean, pred_std, action_idx
        )

        # 生成下一个观测（会缓存新的候选点）
        next_obs = None if done else self._get_observation()

        info = {
            "y_new": y_new,
            "best_y": self.best_y,
            "global_min": self.global_min,
            "regret": self.best_y - self.global_min,
            "step": self.step_count,
        }

        return next_obs, reward, done, info

    def _compute_improved_reward(
        self,
        y_new: float,
        old_best: float,
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        action_idx: int,
    ) -> float:
        """
        简洁清晰的奖励函数

        设计原则:
        1. 成功 -> 巨大奖励，直接结束（明确目标）
        2. 未成功 -> 时间惩罚 + 距离惩罚 + 进步奖励
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
        if regret < 100:
            reward += 0.5
        if regret < 50:
            reward += 1.0

        return float(reward)


# ==================== Rollout Buffer ====================
class ImprovedRolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.context_feats = []
        self.candidate_feats = []
        self.steps = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, context_feat, candidate_feat, step, action, reward, value, log_prob, done):
        self.context_feats.append(context_feat)
        self.candidate_feats.append(candidate_feat)
        self.steps.append(step)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(float(done))

    def get(self, last_value=0.0, gamma=0.99, lam=0.95):
        # GAE
        advantages = []
        gae = 0
        values = self.values + [last_value]

        for t in reversed(range(len(self.rewards))):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        return {
            "context_feats": self.context_feats,
            "candidate_feats": self.candidate_feats,
            "steps": self.steps,
            "actions": self.actions,
            "log_probs": self.log_probs,
            "advantages": advantages,
            "returns": returns,
        }

    def __len__(self):
        return len(self.actions)


# ==================== 训练函数 ====================
def train_improved(
    variants_path: str = "./data/goldstein_price_family_variants.npz",
    model_path: str = "./model/finetuned_tabpfn_goldstein_price_family.ckpt",
    total_episodes: int = 5000,  # 增加训练量
    max_steps: int = 20,
    n_candidates: int = 128,
    update_every: int = 20,  # 更频繁的更新
    save_every: int = 500,
    save_dir: str = "./runs/ppo_bo_goldstein_price",
    seed: int = 42,
    use_tensorboard: bool = True,  # TensorBoard 开关
    # 网络超参数
    hidden_dim: int = 128,
    n_self_attn_layers: int = 3,
    n_cross_attn_layers: int = 3,
    n_heads: int = 8,
):
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"Improved PPO Training for Bayesian Optimization (Goldstein-Price)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Max Steps per Episode: {max_steps}")
    print(f"Candidates per Step: {n_candidates}")
    print(f"Update Every: {update_every} episodes")
    print(f"网络: hidden={hidden_dim}, self_attn={n_self_attn_layers}, cross_attn={n_cross_attn_layers}, heads={n_heads}")
    print(f"{'='*70}\n")

    # 保存配置
    config = {
        "variants_path": variants_path,
        "model_path": model_path,
        "total_episodes": total_episodes,
        "max_steps": max_steps,
        "n_candidates": n_candidates,
        "hidden_dim": hidden_dim,
        "n_self_attn_layers": n_self_attn_layers,
        "n_cross_attn_layers": n_cross_attn_layers,
        "n_heads": n_heads,
        "seed": seed,
        "version": "goldstein_price",
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 初始化 TensorBoard
    writer = None
    if use_tensorboard:
        tb_dir = os.path.join(save_dir, "tensorboard")
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"TensorBoard 日志目录: {tb_dir}")

    # 创建环境
    env = ImprovedGoldsteinPriceBOEnv(
        variants_path=variants_path,
        model_path=model_path,
        max_steps=max_steps,
        n_candidates=n_candidates,
        device=device,
        seed=seed,
    )
    print(f"环境已创建: {env.num_variants} 个 Goldstein-Price 变体")

    # 创建 PPO
    ppo = ImprovedPPO(
        coord_dim=2,
        hidden_dim=hidden_dim,
        n_self_attn_layers=n_self_attn_layers,
        n_cross_attn_layers=n_cross_attn_layers,
        n_heads=n_heads,
        max_steps=max_steps,
        device=device,
    )
    print(f"PPO 网络参数量: {sum(p.numel() for p in ppo.policy.parameters()):,}")

    buffer = ImprovedRolloutBuffer()

    # 训练记录
    metrics = {
        "episode_reward": [],
        "episode_regret": [],
        "best_regret": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
    }

    best_avg_regret = float('inf')
    recent_regrets = deque(maxlen=100)

    print(f"\n开始训练...\n")

    for episode in range(1, total_episodes + 1):
        obs = env.reset()
        episode_reward = 0
        bounds = env.current_func.bounds

        for step in range(max_steps):
            X_context = obs["X_context"]
            y_context = obs["y_context"]
            X_candidates = obs["X_candidates"]
            pred_mean = obs["pred_mean"]
            pred_std = obs["pred_std"]
            current_step = obs["step"]

            action, log_prob, value = ppo.select_candidate(
                X_context, y_context, X_candidates, pred_mean, pred_std, bounds, current_step
            )

            next_obs, reward, done, info = env.step(action)

            episode_reward += reward

            context_feat, candidate_feat = ppo._build_features(
                X_context, y_context, X_candidates, pred_mean, pred_std, bounds, current_step
            )
            buffer.add(context_feat, candidate_feat, current_step, action, reward, value, log_prob, done)

            if done:
                break

            obs = next_obs

        # 记录
        final_regret = info["regret"]
        metrics["episode_reward"].append(episode_reward)
        metrics["episode_regret"].append(final_regret)
        recent_regrets.append(final_regret)

        # TensorBoard 记录 episode 指标
        if writer is not None:
            writer.add_scalar("Episode/Reward", episode_reward, episode)
            writer.add_scalar("Episode/Regret", final_regret, episode)
            writer.add_scalar("Episode/BestY", info["best_y"], episode)

        if len(recent_regrets) >= 100:
            avg_regret = np.mean(recent_regrets)
            if avg_regret < best_avg_regret:
                best_avg_regret = avg_regret
                best_path = os.path.join(save_dir, "ppo_best.pt")
                torch.save(ppo.policy.state_dict(), best_path)

        metrics["best_regret"].append(best_avg_regret)

        # PPO 更新
        if episode % update_every == 0 and len(buffer) > 0:
            rollout = buffer.get(last_value=0.0, gamma=ppo.gamma, lam=ppo.lam)
            losses = ppo.update(rollout, n_epochs=4, batch_size=64)
            buffer.reset()

            metrics["policy_loss"].append(losses["policy_loss"])
            metrics["value_loss"].append(losses["value_loss"])
            metrics["entropy"].append(losses["entropy"])

            # TensorBoard 记录训练指标
            if writer is not None:
                writer.add_scalar("Train/PolicyLoss", losses["policy_loss"], episode)
                writer.add_scalar("Train/ValueLoss", losses["value_loss"], episode)
                writer.add_scalar("Train/Entropy", losses["entropy"], episode)
                writer.add_scalar("Train/LearningRate", losses["lr"], episode)
                writer.add_scalar("Train/AvgRegret_100ep", np.mean(list(recent_regrets)), episode)
                writer.add_scalar("Train/BestAvgRegret", best_avg_regret, episode)

            if episode % 10 == 0:
                avg_reward = np.mean(metrics["episode_reward"][-100:])
                avg_regret = np.mean(metrics["episode_regret"][-100:])
                print(f"Episode {episode:5d} | Reward: {avg_reward:6.2f} | Regret: {avg_regret:.4f} | "
                      f"Best: {best_avg_regret:.4f} | LR: {losses['lr']:.2e}")

        # 保存
        if episode % save_every == 0:
            save_path = os.path.join(save_dir, f"ppo_ep{episode}.pt")
            torch.save(ppo.policy.state_dict(), save_path)

            with open(os.path.join(save_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f)

            print(f"\n模型已保存: {save_path}\n")

    # 保存最终模型
    final_path = os.path.join(save_dir, "ppo_final.pt")
    torch.save(ppo.policy.state_dict(), final_path)
    print(f"\n最终模型已保存: {final_path}")

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    # 关闭 TensorBoard
    if writer is not None:
        writer.close()
        print("TensorBoard writer 已关闭")

    return ppo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants_path", type=str, default="./data/goldstein_price_family_variants.npz")
    parser.add_argument("--model_path", type=str, default="./model/finetuned_tabpfn_goldstein_price_family.ckpt")
    parser.add_argument("--total_episodes", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--n_candidates", type=int, default=128)
    parser.add_argument("--update_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="./runs/ppo_bo_goldstein_price")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_tensorboard", action="store_true", default=True)
    parser.add_argument("--no_tensorboard", dest="use_tensorboard", action="store_false")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_self_attn_layers", type=int, default=3)
    parser.add_argument("--n_cross_attn_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=8)

    args = parser.parse_args()

    train_improved(
        variants_path=args.variants_path,
        model_path=args.model_path,
        total_episodes=args.total_episodes,
        max_steps=args.max_steps,
        n_candidates=args.n_candidates,
        update_every=args.update_every,
        save_every=args.save_every,
        save_dir=args.save_dir,
        seed=args.seed,
        use_tensorboard=args.use_tensorboard,
        hidden_dim=args.hidden_dim,
        n_self_attn_layers=args.n_self_attn_layers,
        n_cross_attn_layers=args.n_cross_attn_layers,
        n_heads=args.n_heads,
    )
