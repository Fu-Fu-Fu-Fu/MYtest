import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# 复用你 eval_rl.py 中的类
from eval_rl import RLPolicyTesterCNN, CNNAgent

def debug_policy_sensitivity():
    """
    诊断策略是否真的在利用输入观测。
    方法：固定多个完全不同的输入，看输出动作的方差。
    """
    policy_path = '/gpfs/radev/pi/cohan/yz979/lin/test/runs/TabPFN_CNN_PPO_1765353463/agent_15.pt'
    tabpfn_model_path = './model/finetuned_tabpfn_branin_family.ckpt'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载策略
    ckpt = torch.load(policy_path, map_location=device)
    config = ckpt.get('config', {})
    grid_size = config.get('grid_size', 32)
    
    agent = CNNAgent(grid_size=grid_size).to(device)
    agent.load_state_dict(ckpt['model_state_dict'])
    agent.eval()
    
    print("=" * 60)
    print("诊断 1: 检查不同输入下的动作输出")
    print("=" * 60)
    
    # 生成多种极端不同的输入
    test_inputs = []
    labels = []
    
    # 1. 全零输入
    test_inputs.append(np.zeros((2, grid_size, grid_size), dtype=np.float32))
    labels.append("全零")
    
    # 2. 全一输入
    test_inputs.append(np.ones((2, grid_size, grid_size), dtype=np.float32))
    labels.append("全一")
    
    # 3. 随机高斯
    for i in range(5):
        rng = np.random.default_rng(seed=i)
        test_inputs.append(rng.normal(size=(2, grid_size, grid_size)).astype(np.float32))
        labels.append(f"随机高斯-{i}")
    
    # 4. 极端值：通道1高、通道2低
    extreme1 = np.zeros((2, grid_size, grid_size), dtype=np.float32)
    extreme1[0] = 10.0
    extreme1[1] = -10.0
    test_inputs.append(extreme1)
    labels.append("通道1高/通道2低")
    
    # 5. 极端值：通道1低、通道2高
    extreme2 = np.zeros((2, grid_size, grid_size), dtype=np.float32)
    extreme2[0] = -10.0
    extreme2[1] = 10.0
    test_inputs.append(extreme2)
    labels.append("通道1低/通道2高")
    
    # 6. 左上角高、右下角低的梯度
    gradient = np.zeros((2, grid_size, grid_size), dtype=np.float32)
    for i in range(grid_size):
        for j in range(grid_size):
            gradient[0, i, j] = (i + j) / (2 * grid_size) * 2 - 1
            gradient[1, i, j] = -(i + j) / (2 * grid_size) * 2 + 1
    test_inputs.append(gradient)
    labels.append("对角梯度")
    
    actions = []
    for inp in test_inputs:
        obs_tensor = torch.from_numpy(inp).unsqueeze(0).to(device)
        with torch.no_grad():
            # 使用确定性动作（mean）
            feat = agent.fc(agent.network(obs_tensor))
            action_mean = torch.tanh(agent.actor_mean(feat))
            actions.append(action_mean.cpu().numpy()[0])
    
    actions = np.array(actions)
    
    print("\n输入类型 -> 动作输出 (x1, x2)")
    print("-" * 50)
    for label, action in zip(labels, actions):
        print(f"{label:20s} -> ({action[0]:+.4f}, {action[1]:+.4f})")
    
    print("\n" + "=" * 60)
    print("动作输出统计:")
    print(f"  x1: mean={actions[:,0].mean():.4f}, std={actions[:,0].std():.6f}")
    print(f"  x2: mean={actions[:,1].mean():.4f}, std={actions[:,1].std():.6f}")
    print("=" * 60)
    
    if actions[:, 0].std() < 0.01 and actions[:, 1].std() < 0.01:
        print("\n⚠️  警告: 策略输出几乎不随输入变化！策略已崩塌。")
    else:
        print("\n✓ 策略对不同输入有不同响应。")
    
    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, (inp, label, action) in enumerate(zip(test_inputs[:10], labels[:10], actions[:10])):
        row, col = i // 5, i % 5
        ax = axes[row, col]
        ax.imshow(inp[0], cmap='viridis')  # 只显示通道1
        ax.set_title(f"{label}\na=({action[0]:.2f},{action[1]:.2f})", fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('./results/debug_policy_sensitivity.png', dpi=150)
    plt.close()
    print("\n诊断图已保存到 ./results/debug_policy_sensitivity.png")


def debug_training_observations():
    """
    检查训练环境中实际产生的观测是否有足够的区分度。
    """
    import sys
    sys.path.append('.')
    from train_rl import TabPFN_BraninEnv
    
    print("\n" + "=" * 60)
    print("诊断 2: 检查训练环境产生的观测")
    print("=" * 60)
    
    env = TabPFN_BraninEnv(
        tabpfn_model_path='./model/finetuned_tabpfn_branin_family.ckpt',
        variants_path='./data/branin_family_variants.npz',
        grid_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_steps=20,
        seed=42,
    )
    
    obs = env.reset(seed=42)
    print(f"\n初始观测统计:")
    print(f"  通道0 (mean_map): mean={obs[0].mean():.4f}, std={obs[0].std():.4f}")
    print(f"  通道1 (var_map):  mean={obs[1].mean():.4f}, std={obs[1].std():.4f}")
    
    # 模拟几步随机动作
    observations = [obs]
    for step in range(5):
        action = np.random.uniform(-1, 1, size=2)
        obs, reward, done, _ = env.step(action)
        observations.append(obs)
        
        print(f"\nStep {step+1}:")
        print(f"  通道0: mean={obs[0].mean():.4f}, std={obs[0].std():.4f}, "
              f"min={obs[0].min():.4f}, max={obs[0].max():.4f}")
        print(f"  通道1: mean={obs[1].mean():.4f}, std={obs[1].std():.4f}, "
              f"min={obs[1].min():.4f}, max={obs[1].max():.4f}")
    
    # 检查观测之间的差异
    print("\n" + "-" * 50)
    print("相邻观测之间的差异:")
    for i in range(1, len(observations)):
        diff = np.abs(observations[i] - observations[i-1])
        print(f"  Step {i-1} -> {i}: max_diff={diff.max():.4f}, mean_diff={diff.mean():.4f}")


def debug_actor_logstd():
    """检查 actor_logstd 是否变得极负（导致几乎没有探索）"""
    policy_path = '/gpfs/radev/pi/cohan/yz979/lin/test/runs/TabPFN_CNN_PPO_1765353463/agent_15.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ckpt = torch.load(policy_path, map_location=device)
    
    print("\n" + "=" * 60)
    print("诊断 3: 检查策略网络参数")
    print("=" * 60)
    
    # 检查 actor_logstd
    actor_logstd = ckpt['model_state_dict']['actor_logstd']
    print(f"actor_logstd: {actor_logstd.cpu().numpy()}")
    print(f"对应的 std:   {torch.exp(actor_logstd).cpu().numpy()}")
    
    # 检查 actor_mean 的权重
    actor_mean_weight = ckpt['model_state_dict']['actor_mean.weight']
    actor_mean_bias = ckpt['model_state_dict']['actor_mean.bias']
    print(f"\nactor_mean.weight 统计:")
    print(f"  shape: {actor_mean_weight.shape}")
    print(f"  mean: {actor_mean_weight.mean():.6f}, std: {actor_mean_weight.std():.6f}")
    print(f"  min: {actor_mean_weight.min():.6f}, max: {actor_mean_weight.max():.6f}")
    print(f"\nactor_mean.bias: {actor_mean_bias.cpu().numpy()}")
    
    # 如果 bias 很大且 weight 很小，tanh 会饱和
    if abs(actor_mean_bias.cpu().numpy()).max() > 2:
        print("\n⚠️  警告: actor_mean.bias 过大，可能导致 tanh 饱和！")


if __name__ == "__main__":
    os.makedirs('./results', exist_ok=True)
    debug_policy_sensitivity()
    debug_actor_logstd()
    # 如果你能运行训练环境，也运行这个：
    # debug_training_observations()