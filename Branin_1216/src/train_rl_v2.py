"""
贝叶斯优化候选点选择的 PPO 实现 v2

改进点：
1. 抛弃 EI 筛选，直接使用 Sobol 候选点作为决策空间
2. 双塔 Cross-Attention 架构：
   - Context 塔: 编码历史观测点 [x_i, y_i]
   - Candidate 塔: 通过 Cross-Attention 查询历史信息
3. RL 可以学习 "给定历史观测模式 → 如何决策"

架构:
Context 塔:  [x_i, y_i] → Self-Attention → Context Embeddings
                                              ↓ (Key, Value)
Candidate 塔: [x_j, μ_j, σ_j] → Self-Attention → Cross-Attention → Candidate Embeddings
                                              ↑ (Query)
                                              
输出: Candidate Embeddings → Actor Head → 选择概率
      Pooled Features → Critic Head → 状态价值
"""

import os
import json
import time
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.distributions import Categorical
from typing import Dict, Any, Optional, Tuple
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol
from collections import deque

# 忽略 Sobol 警告
warnings.filterwarnings('ignore', message='The balance properties of Sobol')

from select_candidates import (
    SelectionConfig, 
    ObjectiveFunction, 
    compute_std_from_tabpfn_output,
)
from tabpfn import TabPFNRegressor


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
    """Branin 变体函数"""
    
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
        """搜索全局最小值"""
        x1 = np.linspace(-5, 10, 100)
        x2 = np.linspace(0, 15, 100)
        X1, X2 = np.meshgrid(x1, x2)
        grid_points = np.stack([X1.flatten(), X2.flatten()], axis=1).astype(np.float32)
        y_grid = self(grid_points)
        
        sorted_indices = np.argsort(y_grid)
        n_starts = 20
        start_points = grid_points[sorted_indices[:n_starts]]
        
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
        
        best_min = min(best_min, float(y_grid.min()))
        return best_min
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X).astype(np.float32)
        return branin_family_numpy(X, self.variant_params, self._device)


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
            mask: (batch, seq_len) optional padding mask
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
            kv_mask: (batch, n_kv) optional mask for key/value
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
            mask: (batch, seq_len) optional
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


# ==================== 双塔 Cross-Attention 策略网络 ====================
class DualTowerCandidateSelector(nn.Module):
    """
    双塔 Cross-Attention 候选点选择网络
    
    架构:
    1. Context 塔: 编码历史观测点 [x_i, y_i] -> Context Embeddings
    2. Candidate 塔: 编码候选点 [x_j, μ_j, σ_j]，通过 Cross-Attention 融合 Context
    
    输入:
        - context: (batch, n_context, context_dim)  历史观测 [x, y]
        - candidates: (batch, n_candidates, candidate_dim)  候选点 [x, μ, σ]
        
    输出:
        - logits: (batch, n_candidates) 每个候选点的选择 score
        - value: (batch,) 状态价值
    """
    
    def __init__(
        self, 
        coord_dim: int = 2,
        hidden_dim: int = 64, 
        n_self_attn_layers: int = 2,
        n_cross_attn_layers: int = 2,
        n_heads: int = 4, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        
        # Context 特征: [x (coord_dim), y (1)] = coord_dim + 1
        context_dim = coord_dim + 1
        # Candidate 特征: [x (coord_dim), μ (1), σ (1)] = coord_dim + 2
        candidate_dim = coord_dim + 2
        
        # ========== Context 塔 ==========
        self.context_embed = nn.Linear(context_dim, hidden_dim)
        self.context_transformer = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, hidden_dim * 2, dropout)
            for _ in range(n_self_attn_layers)
        ])
        self.context_norm = nn.LayerNorm(hidden_dim)
        
        # ========== Candidate 塔 ==========
        self.candidate_embed = nn.Linear(candidate_dim, hidden_dim)
        self.candidate_self_attn = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, hidden_dim * 2, dropout)
            for _ in range(n_self_attn_layers)
        ])
        
        # Cross-Attention: Candidate (Query) attends to Context (Key, Value)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, hidden_dim * 2, dropout)
            for _ in range(n_cross_attn_layers)
        ])
        self.candidate_norm = nn.LayerNorm(hidden_dim)
        
        # ========== Actor Head ==========
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # ========== Critic Head ==========
        # 融合 Context 和 Candidate 信息
        self.context_pool = AttentionPooling(hidden_dim)
        self.candidate_pool = AttentionPooling(hidden_dim)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, context, candidates, context_mask=None):
        """
        Args:
            context: (batch, n_context, coord_dim + 1)  [x, y]
            candidates: (batch, n_candidates, coord_dim + 2)  [x, μ, σ]
            context_mask: (batch, n_context) optional padding mask
            
        Returns:
            logits: (batch, n_candidates)
            value: (batch,)
        """
        # ========== Context 塔 ==========
        ctx = self.context_embed(context)  # (batch, n_ctx, hidden)
        for layer in self.context_transformer:
            ctx = layer(ctx, context_mask)
        ctx = self.context_norm(ctx)
        
        # ========== Candidate 塔 ==========
        cand = self.candidate_embed(candidates)  # (batch, n_cand, hidden)
        for layer in self.candidate_self_attn:
            cand = layer(cand)
        
        # Cross-Attention: Candidate queries Context
        for cross_layer in self.cross_attn_layers:
            cand = cross_layer(cand, ctx, context_mask)
        cand = self.candidate_norm(cand)
        
        # ========== Actor: 输出每个候选点的 score ==========
        logits = self.actor(cand).squeeze(-1)  # (batch, n_cand)
        
        # ========== Critic: 全局状态价值 ==========
        ctx_pooled = self.context_pool(ctx, context_mask)  # (batch, hidden)
        cand_pooled = self.candidate_pool(cand)  # (batch, hidden)
        global_feat = torch.cat([ctx_pooled, cand_pooled], dim=-1)  # (batch, hidden*2)
        value = self.critic(global_feat).squeeze(-1)  # (batch,)
        
        return logits, value
    
    def get_action(self, context, candidates, context_mask=None):
        """采样动作"""
        with torch.no_grad():
            logits, value = self.forward(context, candidates, context_mask)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob, value
    
    def evaluate(self, context, candidates, actions, context_mask=None):
        """评估动作"""
        logits, values = self.forward(context, candidates, context_mask)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy


# ==================== PPO 算法 ====================
class PPO:
    """候选点选择的 PPO（双塔版本）"""
    
    def __init__(
        self,
        coord_dim: int = 2,
        hidden_dim: int = 64,
        n_self_attn_layers: int = 2,
        n_cross_attn_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.coord_dim = coord_dim
        
        self.policy = DualTowerCandidateSelector(
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            n_self_attn_layers=n_self_attn_layers,
            n_cross_attn_layers=n_cross_attn_layers,
            n_heads=n_heads,
            dropout=dropout,
        ).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def select_candidate(
        self, 
        X_context: np.ndarray, 
        y_context: np.ndarray,
        X_candidates: np.ndarray, 
        pred_mean: np.ndarray, 
        pred_std: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[int, float, float]:
        """
        从候选点中选择一个
        
        Args:
            X_context: (n_context, coord_dim) 历史观测坐标
            y_context: (n_context,) 历史观测值
            X_candidates: (n_candidates, coord_dim) 候选点坐标
            pred_mean: (n_candidates,) 预测均值
            pred_std: (n_candidates,) 预测标准差
            bounds: (lower, upper) 坐标边界
            
        Returns:
            action: 选中的候选点索引
            log_prob: 动作的 log 概率
            value: 状态价值估计
        """
        context_feat, candidate_feat = self._build_features(
            X_context, y_context, X_candidates, pred_mean, pred_std, bounds
        )
        
        context_feat = context_feat.unsqueeze(0).to(self.device)
        candidate_feat = candidate_feat.unsqueeze(0).to(self.device)
        
        action, log_prob, value = self.policy.get_action(context_feat, candidate_feat)
        return action.item(), log_prob.item(), value.item()
    
    def _build_features(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_candidates: np.ndarray,
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建归一化特征
        
        Returns:
            context_feat: (n_context, coord_dim + 1)
            candidate_feat: (n_candidates, coord_dim + 2)
        """
        lower, upper = bounds
        
        # ========== Context 特征 ==========
        # 坐标归一化到 [0, 1]
        X_ctx_norm = (X_context - lower) / (upper - lower + 1e-8)
        
        # y 值标准化（相对于上下文的 min/max）
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
        
        # 均值相对于 y_best 的差值，并标准化
        y_best = y_context.min()
        mean_rel = pred_mean - y_best
        mean_std_val = mean_rel.std()
        if mean_std_val > 1e-8:
            mean_norm = (mean_rel - mean_rel.mean()) / mean_std_val
        else:
            mean_norm = mean_rel - mean_rel.mean()
        
        # 标准差标准化
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
        """PPO 更新"""
        context_states = rollout["context_states"]
        candidate_states = rollout["candidate_states"]
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
        
        n_samples = len(context_states)
        
        for _ in range(n_epochs):
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                
                # 批次数据（需要 padding 处理不同长度的 context）
                b_context = self._pad_sequences([context_states[i] for i in batch_idx])
                b_candidates = torch.stack([candidate_states[i] for i in batch_idx]).to(self.device)
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]
                
                # Context mask (1 = valid, 0 = padding)
                b_context_mask = (b_context.sum(dim=-1) != 0).float().to(self.device)
                b_context = b_context.to(self.device)
                
                # 评估
                new_log_probs, new_values, entropy = self.policy.evaluate(
                    b_context, b_candidates, b_actions, b_context_mask
                )
                
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
    
    def _pad_sequences(self, sequences):
        """Pad context sequences to same length"""
        max_len = max(s.shape[0] for s in sequences)
        batch_size = len(sequences)
        feat_dim = sequences[0].shape[1]
        
        padded = torch.zeros(batch_size, max_len, feat_dim)
        for i, seq in enumerate(sequences):
            padded[i, :seq.shape[0], :] = seq
        
        return padded


# ==================== Rollout Buffer ====================
class RolloutBuffer:
    """存储经验（双塔版本）"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.context_states = []
        self.candidate_states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, context_feat, candidate_feat, action, reward, value, log_prob, done):
        self.context_states.append(context_feat)
        self.candidate_states.append(candidate_feat)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(float(done))
    
    def get(self, last_value=0.0):
        return {
            "context_states": self.context_states,
            "candidate_states": self.candidate_states,
            "actions": self.actions,
            "rewards": self.rewards,
            "values": self.values,
            "log_probs": self.log_probs,
            "dones": self.dones,
            "last_value": last_value,
        }
    
    def __len__(self):
        return len(self.actions)


# ==================== 训练环境 ====================
class BraninBOEnv:
    """贝叶斯优化环境（直接使用 Sobol 候选点，不经过 EI 筛选）"""
    
    def __init__(
        self,
        variants_path: str,
        model_path: str,
        max_steps: int = 20,
        n_init_context: int = 2,
        n_candidates: int = 128,  # 候选点数量
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
        self.n_candidates = n_candidates
        self.device = device
        self.rng = np.random.default_rng(seed)
        
        # TabPFN Regressor（复用）
        self.regressor = None
        self._init_regressor()
        
        # Episode 状态
        self.current_func = None
        self.X_context = None
        self.y_context = None
        self.best_y = None
        self.global_min = None
        self.step_count = 0
    
    def _init_regressor(self):
        """初始化 TabPFN"""
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
        
        # 获取初始观测
        return self._get_observation()
    
    def _get_observation(self):
        """获取当前观测（Sobol 候选点 + TabPFN 预测）"""
        lower, upper = self.current_func.bounds
        
        # Sobol 采样候选点
        sobol_seed = int(self.rng.integers(0, 100000))
        sobol_sampler = Sobol(d=self.current_func.dim, scramble=True, seed=sobol_seed)
        sobol_samples = sobol_sampler.random(self.n_candidates)
        X_candidates = (sobol_samples * (upper - lower) + lower).astype(np.float32)
        
        # TabPFN 预测
        self.regressor.fit(self.X_context, self.y_context)
        full_out = self.regressor.predict(X_candidates, output_type="full")
        pred_mean = full_out.get("mean", None)
        pred_std = compute_std_from_tabpfn_output(full_out)
        
        return {
            "X_context": self.X_context.copy(),
            "y_context": self.y_context.copy(),
            "X_candidates": X_candidates,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
        }
    
    def step(self, action_idx: int):
        """
        执行动作：选择第 action_idx 个候选点进行评估
        """
        # 获取当前候选点
        obs = self._get_observation()
        X_candidates = obs["X_candidates"]
        
        # 获取选中的点
        x_new = X_candidates[action_idx:action_idx+1]
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
            next_obs = self._get_observation()
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
        """计算奖励"""
        # 基础时间惩罚
        reward = -0.1
        
        # 计算 improvement
        improvement = max(0, old_best - self.best_y)
        
        if improvement > 0:
            gap = old_best - self.global_min
            if gap > 1e-8:
                relative_improvement = improvement / gap
                reward += 1.0 + 5.0 * relative_improvement
            else:
                reward += 2.0
        
        # 额外奖励：接近最优
        regret = self.best_y - self.global_min
        if regret < 0.1:
            reward += 0.5
        if regret < 0.01:
            reward += 1.0
        
        return float(reward)


# ==================== 训练监控类 ====================
class TrainingMonitor:
    """训练监控"""
    
    def __init__(self, save_dir: str, use_tensorboard: bool = True):
        self.save_dir = save_dir
        self.use_tensorboard = use_tensorboard
        self.writer = None
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))
                print(f"TensorBoard 日志目录: {os.path.join(save_dir, 'tensorboard')}")
            except ImportError:
                print("警告: tensorboard 未安装")
                self.use_tensorboard = False
        
        self.metrics_history = {
            'episode_reward': [],
            'episode_regret': [],
            'best_regret': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
        }
        
        self.window_size = 100
        self.reward_window = deque(maxlen=self.window_size)
        self.regret_window = deque(maxlen=self.window_size)
        
        self.best_avg_regret = float('inf')
        self.best_episode = 0
        self.start_time = time.time()
    
    def start_episode(self):
        pass
    
    def log_episode(self, episode: int, reward: float, regret: float, info: dict = None):
        self.metrics_history['episode_reward'].append(reward)
        self.metrics_history['episode_regret'].append(regret)
        self.reward_window.append(reward)
        self.regret_window.append(regret)
        
        if len(self.regret_window) >= 10:
            avg_regret = np.mean(self.regret_window)
            if avg_regret < self.best_avg_regret:
                self.best_avg_regret = avg_regret
                self.best_episode = episode
        
        self.metrics_history['best_regret'].append(self.best_avg_regret)
        
        if self.writer:
            self.writer.add_scalar('Episode/Reward', reward, episode)
            self.writer.add_scalar('Episode/Regret', regret, episode)
            if len(self.reward_window) > 0:
                self.writer.add_scalar('Episode/AvgReward_100', np.mean(self.reward_window), episode)
                self.writer.add_scalar('Episode/AvgRegret_100', np.mean(self.regret_window), episode)
    
    def log_update(self, episode: int, losses: dict):
        self.metrics_history['policy_loss'].append(losses.get('policy_loss', 0))
        self.metrics_history['value_loss'].append(losses.get('value_loss', 0))
        self.metrics_history['entropy'].append(losses.get('entropy', 0))
        
        if self.writer:
            self.writer.add_scalar('Loss/Policy', losses.get('policy_loss', 0), episode)
            self.writer.add_scalar('Loss/Value', losses.get('value_loss', 0), episode)
            self.writer.add_scalar('Loss/Entropy', losses.get('entropy', 0), episode)
    
    def print_progress(self, episode: int, total_episodes: int, update_every: int):
        elapsed = time.time() - self.start_time
        eps_per_sec = episode / elapsed if elapsed > 0 else 0
        eta = (total_episodes - episode) / eps_per_sec if eps_per_sec > 0 else 0
        
        avg_reward = np.mean(list(self.reward_window)) if self.reward_window else 0
        avg_regret = np.mean(list(self.regret_window)) if self.regret_window else 0
        min_regret = min(list(self.regret_window)) if self.regret_window else 0
        
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
        metrics_path = os.path.join(self.save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        np_path = os.path.join(self.save_dir, 'metrics.npz')
        np.savez(np_path, **{k: np.array(v) for k, v in self.metrics_history.items()})
    
    def close(self):
        self.save_metrics()
        if self.writer:
            self.writer.close()
        
        total_time = time.time() - self.start_time
        print(f"\n训练完成！总耗时: {total_time/60:.1f}分钟")
        print(f"最佳平均 Regret: {self.best_avg_regret:.4f} @ Episode {self.best_episode}")


# ==================== 训练函数 ====================
def train(
    variants_path: str = "./data/branin_family_variants.npz",
    model_path: str = "./model/finetuned_tabpfn_branin_family.ckpt",
    total_episodes: int = 1000,
    max_steps: int = 20,
    n_candidates: int = 128,
    update_every: int = 10,
    save_every: int = 100,
    save_dir: str = "./runs/ppo_bo_v2",
    seed: int = 42,
    use_tensorboard: bool = True,
    # 网络超参数
    hidden_dim: int = 64,
    n_self_attn_layers: int = 2,
    n_cross_attn_layers: int = 2,
    n_heads: int = 4,
):
    """训练双塔 PPO"""
    os.makedirs(save_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"PPO 双塔 Cross-Attention 贝叶斯优化训练 (v2)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Max Steps per Episode: {max_steps}")
    print(f"Candidates per Step: {n_candidates}")
    print(f"Update Every: {update_every} episodes")
    print(f"Save Dir: {save_dir}")
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
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # 创建监控器
    monitor = TrainingMonitor(save_dir, use_tensorboard=use_tensorboard)
    
    # 创建环境
    env = BraninBOEnv(
        variants_path=variants_path,
        model_path=model_path,
        max_steps=max_steps,
        n_candidates=n_candidates,
        device=device,
        seed=seed,
    )
    print(f"环境已创建: {env.num_variants} 个 Branin 变体")
    
    # 创建 PPO
    coord_dim = 2
    ppo = PPO(
        coord_dim=coord_dim,
        hidden_dim=hidden_dim,
        n_self_attn_layers=n_self_attn_layers,
        n_cross_attn_layers=n_cross_attn_layers,
        n_heads=n_heads,
        device=device,
    )
    print(f"PPO 网络参数量: {sum(p.numel() for p in ppo.policy.parameters()):,}")
    
    # Rollout buffer
    buffer = RolloutBuffer()
    
    print(f"\n开始训练...\n")
    
    for episode in range(1, total_episodes + 1):
        monitor.start_episode()
        
        # 重置环境
        obs = env.reset()
        episode_reward = 0
        bounds = env.current_func.bounds
        
        for step in range(max_steps):
            X_context = obs["X_context"]
            y_context = obs["y_context"]
            X_candidates = obs["X_candidates"]
            pred_mean = obs["pred_mean"]
            pred_std = obs["pred_std"]
            
            # 选择候选点
            action, log_prob, value = ppo.select_candidate(
                X_context, y_context, X_candidates, pred_mean, pred_std, bounds
            )
            
            # 执行动作
            next_obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            
            # 存储经验
            context_feat, candidate_feat = ppo._build_features(
                X_context, y_context, X_candidates, pred_mean, pred_std, bounds
            )
            buffer.add(context_feat, candidate_feat, action, reward, value, log_prob, done)
            
            if done:
                break
            
            obs = next_obs
        
        # 记录 episode 指标
        monitor.log_episode(episode, episode_reward, info["regret"], info)
        
        # PPO 更新
        if episode % update_every == 0 and len(buffer) > 0:
            rollout = buffer.get(last_value=0.0)
            losses = ppo.update(rollout, n_epochs=4, batch_size=32)
            buffer.reset()
            
            monitor.log_update(episode, losses)
            monitor.print_progress(episode, total_episodes, update_every)
        
        # 保存模型
        if episode % save_every == 0:
            save_path = os.path.join(save_dir, f"ppo_ep{episode}.pt")
            torch.save(ppo.policy.state_dict(), save_path)
            print(f"\n模型已保存: {save_path}")
            
            if monitor.best_avg_regret < float('inf'):
                best_path = os.path.join(save_dir, "ppo_best.pt")
                torch.save(ppo.policy.state_dict(), best_path)
            
            monitor.save_metrics()
    
    # 保存最终模型
    final_path = os.path.join(save_dir, "ppo_final.pt")
    torch.save(ppo.policy.state_dict(), final_path)
    print(f"\n最终模型已保存: {final_path}")
    
    monitor.close()
    
    return ppo, monitor


# ==================== 主函数 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dual-Tower PPO for Bayesian Optimization")
    parser.add_argument("--variants_path", type=str, default="./data/branin_family_variants.npz")
    parser.add_argument("--model_path", type=str, default="./model/finetuned_tabpfn_branin_family.ckpt")
    parser.add_argument("--total_episodes", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--n_candidates", type=int, default=128, help="Number of Sobol candidates")
    parser.add_argument("--update_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./runs/ppo_bo_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_tensorboard", action="store_true", default=True)
    parser.add_argument("--no_tensorboard", dest="use_tensorboard", action="store_false")
    # 网络超参数
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_self_attn_layers", type=int, default=2)
    parser.add_argument("--n_cross_attn_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    
    args = parser.parse_args()
    
    train(
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
