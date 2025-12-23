"""
用 EI (Expected Improvement) 监督预训练 CNN 策略
"""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm

from tabpfn import TabPFNRegressor

plt.switch_backend("agg")


# =========================================================
# 1. Branin 函数族
# =========================================================

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
# 2. EI 计算
# =========================================================

def compute_ei(mean, std, best_y, xi=0.01):
    """
    计算 Expected Improvement
    
    Args:
        mean: 预测均值 (N,)
        std: 预测标准差 (N,)
        best_y: 当前最好的函数值
        xi: 探索参数
    
    Returns:
        ei: Expected Improvement (N,)
    """
    std = np.maximum(std, 1e-9)
    
    # 对于最小化问题
    imp = best_y - mean - xi
    Z = imp / std
    ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
    ei[std < 1e-9] = 0.0
    
    return ei


# =========================================================
# 3. 数据生成器：用 TabPFN 生成 (obs, ei_action) 对
# =========================================================

# 替换 EIExpertDataGenerator 类中的相关方法

class EIExpertDataGenerator:
    """生成 EI 专家的采样轨迹数据"""
    
    def __init__(
        self,
        tabpfn_model_path: str,
        variants_path: str,
        grid_size: int = 32,
        device: str = "cpu",
    ):
        self.device = device
        self.grid_size = grid_size
        
        # 加载 TabPFN
        self.tabpfn = TabPFNRegressor(
            model_path=tabpfn_model_path,
            device=self.device,
            n_estimators=1,
            inference_precision=torch.float32,
            fit_mode="batched",
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
        
        # 归一化参数
        self.y_global_min = 0.0
        self.y_global_max = 300.0
        self.y_range = self.y_global_max - self.y_global_min

    def _compute_maps(self, X_ctx, y_ctx):
        """计算 TabPFN 的均值和标准差 map"""
        n_ctx = X_ctx.shape[0]
        
        if n_ctx == 0:
            # 初始状态：均匀的均值，带空间变化的不确定性
            mean_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32) * 50.0
            std_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32) * 100.0
            return mean_map, std_map
        
        if n_ctx == 1:
            X_fit = np.repeat(X_ctx, 2, axis=0)
            y_fit_raw = np.repeat(y_ctx, 2, axis=0)
        else:
            X_fit = X_ctx
            y_fit_raw = y_ctx
        
        y_fit_norm = (y_fit_raw - self.y_global_min) / self.y_range
        
        self.tabpfn.fit(X_fit, y_fit_norm)
        full_out = self.tabpfn.predict(self.grid_points, output_type="full")
        
        # 均值
        mean_norm = np.asarray(full_out["mean"]).reshape(-1)
        mean_pred = mean_norm * self.y_range + self.y_global_min
        mean_map = mean_pred.reshape(self.grid_size, self.grid_size).astype(np.float32)
        
        # 标准差
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
            
            var = var * (self.y_range ** 2)
            std_map = np.sqrt(np.maximum(var, 1e-6))
            std_map = std_map.reshape(self.grid_size, self.grid_size).astype(np.float32)
        else:
            # 没有方差信息时，用距离已采样点的距离作为不确定性
            from sklearn.metrics import pairwise_distances
            dists = pairwise_distances(self.grid_points, X_ctx)
            min_dist = dists.min(axis=1)
            std_map = (min_dist * 10 + 1.0).reshape(self.grid_size, self.grid_size).astype(np.float32)
        
        return mean_map, std_map

    def _maps_to_obs(self, mean_map, std_map, X_ctx):
        """将 map 转换为 CNN 输入"""
        mean_norm = (mean_map - 0.0) / 300.0
        mean_norm = np.clip(mean_norm, -1.0, 2.0)
        
        uncert_norm = std_map / 50.0
        uncert_norm = np.clip(uncert_norm, 0.0, 3.0)
        
        # 标记已采样点
        if len(X_ctx) > 0:
            for pt in X_ctx:
                i = int((pt[1] - self.x2_min) / (self.x2_max - self.x2_min) * (self.grid_size - 1))
                j = int((pt[0] - self.x1_min) / (self.x1_max - self.x1_min) * (self.grid_size - 1))
                i = np.clip(i, 0, self.grid_size - 1)
                j = np.clip(j, 0, self.grid_size - 1)
                uncert_norm[i, j] = -1.0
        
        obs = np.stack([mean_norm, uncert_norm], axis=0).astype(np.float32)
        return obs

    def _get_ei_action(self, mean_map, std_map, best_y):
        """用 EI 选择下一个采样点，返回归一化后的动作 [-1, 1]"""
        mean_flat = mean_map.flatten()
        std_flat = std_map.flatten()
        
        # 计算 EI
        ei = compute_ei(mean_flat, std_flat, best_y)
        
        # 添加小量随机噪声打破平局
        ei = ei + np.random.uniform(0, 1e-6, size=ei.shape)
        
        best_idx = np.argmax(ei)
        best_point = self.grid_points[best_idx]
        
        # 转换为动作空间 [-1, 1]
        a1 = 2.0 * (best_point[0] - self.x1_min) / (self.x1_max - self.x1_min) - 1.0
        a2 = 2.0 * (best_point[1] - self.x2_min) / (self.x2_max - self.x2_min) - 1.0
        
        return np.array([a1, a2], dtype=np.float32)

    def generate_trajectory(self, variant_idx, max_steps=20, seed=None):
        """
        生成一条完整轨迹的 (obs, action) 数据
        """
        if seed is not None:
            np.random.seed(seed)
        
        variant = self.variants[variant_idx]
        
        X_ctx = np.empty((0, 2), dtype=np.float32)
        y_ctx = np.empty((0,), dtype=np.float32)
        best_y = float("inf")
        
        observations = []
        actions = []
        
        for step in range(max_steps):
            # 计算当前状态
            mean_map, std_map = self._compute_maps(X_ctx, y_ctx)
            obs = self._maps_to_obs(mean_map, std_map, X_ctx)
            
            # 第一步随机选择，之后用 EI
            if step == 0:
                # 随机选择第一个采样点
                x1 = np.random.uniform(self.x1_min, self.x1_max)
                x2 = np.random.uniform(self.x2_min, self.x2_max)
                a1 = 2.0 * (x1 - self.x1_min) / (self.x1_max - self.x1_min) - 1.0
                a2 = 2.0 * (x2 - self.x2_min) / (self.x2_max - self.x2_min) - 1.0
                action = np.array([a1, a2], dtype=np.float32)
            else:
                # 用 EI 选择动作
                action = self._get_ei_action(mean_map, std_map, best_y)
            
            observations.append(obs)
            actions.append(action)
            
            # 执行动作，获取真实函数值
            x1 = (action[0] + 1) / 2 * (self.x1_max - self.x1_min) + self.x1_min
            x2 = (action[1] + 1) / 2 * (self.x2_max - self.x2_min) + self.x2_min
            x = np.array([[x1, x2]], dtype=np.float32)
            
            y_val = branin_family_numpy_from_params(x, variant, device=self.device)[0]
            
            X_ctx = np.concatenate([X_ctx, x], axis=0)
            y_ctx = np.concatenate([y_ctx, np.array([y_val], dtype=np.float32)])
            best_y = min(best_y, float(y_val))
        
        return observations, actions

    def generate_dataset(self, num_trajectories=500, max_steps=20, seed=42):
        """生成预训练数据集"""
        np.random.seed(seed)
        
        all_obs = []
        all_actions = []
        
        print(f"生成 {num_trajectories} 条 EI 专家轨迹...")
        for i in tqdm(range(num_trajectories)):
            variant_idx = np.random.randint(0, self.num_variants)
            traj_seed = seed + i
            
            obs_list, action_list = self.generate_trajectory(
                variant_idx, max_steps=max_steps, seed=traj_seed
            )
            
            all_obs.extend(obs_list)
            all_actions.extend(action_list)
        
        return np.array(all_obs), np.array(all_actions)


# =========================================================
# 4. PyTorch Dataset
# =========================================================

class EIPretrainDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = torch.from_numpy(observations).float()
        self.actions = torch.from_numpy(actions).float()
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


# =========================================================
# 5. CNN Agent (与 train_rl.py 一致)
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

    def get_action_mean(self, x):
        """只获取动作均值（用于监督学习）"""
        feat = self.fc(self.network(x))
        action_mean = torch.tanh(self.actor_mean(feat))
        return action_mean

    def get_action_and_value(self, x, action=None):
        from torch.distributions.normal import Normal
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
# 6. 监督预训练
# =========================================================

def pretrain_with_ei(args):
    """Phase 1: 用 EI 专家数据监督预训练"""
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # 1. 生成数据集
    generator = EIExpertDataGenerator(
        tabpfn_model_path=args.tabpfn_model_path,
        variants_path=args.variants_path,
        grid_size=args.grid_size,
        device=str(device),
    )
    
    # 检查是否有缓存的数据
    cache_path = f"./data/ei_pretrain_data_{args.num_trajectories}traj_{args.max_steps}steps.npz"
    
    if os.path.exists(cache_path) and not args.regenerate_data:
        print(f"加载缓存的数据: {cache_path}")
        data = np.load(cache_path)
        observations = data["observations"]
        actions = data["actions"]
    else:
        observations, actions = generator.generate_dataset(
            num_trajectories=args.num_trajectories,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        os.makedirs("./data", exist_ok=True)
        np.savez(cache_path, observations=observations, actions=actions)
        print(f"数据已缓存到: {cache_path}")
    
    print(f"数据集大小: {len(observations)} 个样本")
    
    # 2. 创建 DataLoader
    dataset = EIPretrainDataset(observations, actions)
    
    # 划分训练/验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. 创建模型和优化器
    agent = CNNAgent(grid_size=args.grid_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.pretrain_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs)
    
    # 4. 训练循环
    save_dir = f"runs/EI_Pretrain_{int(time.time())}"
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    
    print(f"\n开始监督预训练...")
    for epoch in range(1, args.pretrain_epochs + 1):
        # Training
        agent.train()
        epoch_loss = 0.0
        for obs, target_action in train_loader:
            obs = obs.to(device)
            target_action = target_action.to(device)
            
            pred_action = agent.get_action_mean(obs)
            
            # MSE Loss
            loss = nn.functional.mse_loss(pred_action, target_action)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * obs.size(0)
        
        train_loss = epoch_loss / len(train_dataset)
        train_losses.append(train_loss)
        
        # Validation
        agent.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs, target_action in val_loader:
                obs = obs.to(device)
                target_action = target_action.to(device)
                pred_action = agent.get_action_mean(obs)
                loss = nn.functional.mse_loss(pred_action, target_action)
                val_loss += loss.item() * obs.size(0)
        
        val_loss = val_loss / len(val_dataset)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": agent.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, os.path.join(save_dir, "best_pretrained.pt"))
    
    # 5. 保存最终模型
    torch.save({
        "model_state_dict": agent.state_dict(),
        "epoch": args.pretrain_epochs,
        "val_loss": val_losses[-1],
    }, os.path.join(save_dir, "final_pretrained.pt"))
    
    # 6. 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("EI Supervised Pretraining")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "pretrain_loss.png"), dpi=150)
    plt.close()
    
    print(f"\n预训练完成!")
    print(f"  最佳验证损失: {best_val_loss:.6f}")
    print(f"  模型保存在: {save_dir}/")
    
    return os.path.join(save_dir, "best_pretrained.pt")


# =========================================================
# 7. 可视化预训练效果
# =========================================================

def visualize_pretrained_vs_ei(pretrained_path, args):
    """对比预训练策略 vs EI 专家的采样轨迹"""
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # 加载预训练模型
    agent = CNNAgent(grid_size=args.grid_size).to(device)
    checkpoint = torch.load(pretrained_path, map_location=device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    
    # 创建数据生成器
    generator = EIExpertDataGenerator(
        tabpfn_model_path=args.tabpfn_model_path,
        variants_path=args.variants_path,
        grid_size=args.grid_size,
        device=str(device),
    )
    
    # 随机选择一个变体
    np.random.seed(123)
    variant_idx = np.random.randint(0, generator.num_variants)
    variant = generator.variants[variant_idx]
    
    # 运行 EI 专家
    ei_obs, ei_actions = generator.generate_trajectory(variant_idx, max_steps=20, seed=123)
    
    # 运行预训练策略
    X_ctx = np.empty((0, 2), dtype=np.float32)
    y_ctx = np.empty((0,), dtype=np.float32)
    best_y = float("inf")
    
    pretrain_actions = []
    pretrain_points = []
    
    for step in range(20):
        mean_map, std_map = generator._compute_maps(X_ctx, y_ctx)
        obs = generator._maps_to_obs(mean_map, std_map, X_ctx)
        
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action = agent.get_action_mean(obs_tensor).cpu().numpy()[0]
        
        pretrain_actions.append(action)
        
        x1 = (action[0] + 1) / 2 * (generator.x1_max - generator.x1_min) + generator.x1_min
        x2 = (action[1] + 1) / 2 * (generator.x2_max - generator.x2_min) + generator.x2_min
        x = np.array([[x1, x2]], dtype=np.float32)
        pretrain_points.append([x1, x2])
        
        y_val = branin_family_numpy_from_params(x, variant, device=str(device))[0]
        
        X_ctx = np.concatenate([X_ctx, x], axis=0)
        y_ctx = np.concatenate([y_ctx, np.array([y_val], dtype=np.float32)])
        best_y = min(best_y, float(y_val))
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 计算真值网格
    true_y = branin_family_numpy_from_params(
        generator.grid_points, variant, device=str(device)
    ).reshape(generator.grid_size, generator.grid_size)
    
    # EI 轨迹
    ax = axes[0]
    cf = ax.contourf(generator.X1, generator.X2, true_y, levels=20)
    
    ei_points = []
    for a in ei_actions:
        x1 = (a[0] + 1) / 2 * (generator.x1_max - generator.x1_min) + generator.x1_min
        x2 = (a[1] + 1) / 2 * (generator.x2_max - generator.x2_min) + generator.x2_min
        ei_points.append([x1, x2])
    ei_points = np.array(ei_points)
    
    ax.plot(ei_points[:, 0], ei_points[:, 1], 'w.-', markersize=8, linewidth=1.5, alpha=0.8)
    ax.scatter(ei_points[-1, 0], ei_points[-1, 1], c='red', marker='x', s=100, linewidths=3, zorder=10)
    ax.set_title("EI Expert", fontsize=14)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    
    # 预训练策略轨迹
    ax = axes[1]
    ax.contourf(generator.X1, generator.X2, true_y, levels=20)
    
    pretrain_points = np.array(pretrain_points)
    ax.plot(pretrain_points[:, 0], pretrain_points[:, 1], 'w.-', markersize=8, linewidth=1.5, alpha=0.8)
    ax.scatter(pretrain_points[-1, 0], pretrain_points[-1, 1], c='red', marker='x', s=100, linewidths=3, zorder=10)
    ax.set_title(f"Pretrained Policy (Best y = {best_y:.4f})", fontsize=14)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    
    plt.tight_layout()
    save_path = os.path.dirname(pretrained_path) + "/pretrain_vs_ei.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"对比图保存到: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabpfn_model_path", type=str, default="./model/finetuned_tabpfn_branin_family.ckpt")
    parser.add_argument("--variants_path", type=str, default="./data/branin_family_variants.npz")
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true", default=True)
    
    # 预训练参数
    parser.add_argument("--num_trajectories", type=int, default=500)  # 500 条轨迹
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--regenerate_data", action="store_true", default=False)
    
    args = parser.parse_args()
    
    # 1. 预训练
    pretrained_path = pretrain_with_ei(args)
    
    # 2. 可视化
    visualize_pretrained_vs_ei(pretrained_path, args)