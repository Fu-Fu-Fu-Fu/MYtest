"""
简单测试 TabPFN 回归功能
"""
import numpy as np
import torch
from tabpfn import TabPFNRegressor


def main():
    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 创建简单的测试数据
    np.random.seed(42)
    X_train = np.random.rand(20, 2).astype(np.float32)  # 20 个训练点，2 个特征
    y_train = (np.sin(X_train[:, 0] * 2 * np.pi) + X_train[:, 1]).astype(np.float32)
    
    X_test = np.random.rand(10, 2).astype(np.float32)  # 10 个测试点
    y_test = (np.sin(X_test[:, 0] * 2 * np.pi) + X_test[:, 1]).astype(np.float32)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # 创建 TabPFN 回归器
    regressor = TabPFNRegressor(
        model_path="./model/finetuned_tabpfn_branin_family.ckpt",
        device=device,
        n_estimators=1,
        random_state=42,
        ignore_pretraining_limits=True,
    )
    
    # 拟合（In-Context Learning，不更新权重）
    print("\nFitting TabPFN...")
    regressor.fit(X_train, y_train)
    
    # 预测均值
    print("Predicting...")
    full_out = regressor.predict(X_test, output_type="full")
    
    # 获取 criterion 和 logits
    criterion = full_out.get("criterion", None)
    logits = full_out.get("logits", None)
    y_pred = full_out.get("mean", None)
    
    print(f"\n========== Logits 分析 ==========")
    print(f"logits type: {type(logits)}")
    if isinstance(logits, torch.Tensor):
        print(f"logits shape: {logits.shape}")  # 通常是 (n_samples, n_bins)
        print(f"logits 范围: [{logits.min():.4f}, {logits.max():.4f}]")
        print(f"logits 中 -inf 数量: {(logits == float('-inf')).sum().item()}")
        print(f"logits 中有限值数量: {torch.isfinite(logits).sum().item()}")
    
    # ========== 解释 -inf 值 ==========
    print(f"\n========== 关于 -inf 值 ==========")
    print("TabPFN 使用 BarDistribution（离散化概率分布）")
    print("logits 是各个 bin 的对数概率（log-probability）")
    print("-inf 表示该 bin 的概率为 0（因为 log(0) = -inf）")
    print("这是完全正常的！表示模型认为预测值不可能落在该 bin 范围内")
    
    # ========== 从 criterion 计算方差 ==========
    print(f"\n========== 从 criterion 计算方差 ==========")
    print(f"criterion type: {type(criterion)}")
    
    if criterion is not None:
        # criterion 是 BarDistribution 对象，可以用其方法计算统计量
        print(f"criterion 属性: {[attr for attr in dir(criterion) if not attr.startswith('_')]}")
        
        # 方法1: 使用 criterion 的内置方法（如果有）
        if hasattr(criterion, 'mean') and callable(criterion.mean):
            try:
                mean_from_criterion = criterion.mean(logits)
                print(f"\n方法1 - criterion.mean(): {mean_from_criterion[:5]}")
            except Exception as e:
                print(f"criterion.mean() 错误: {e}")
        
        # 方法2: 手动从 logits 计算均值和方差
        print(f"\n方法2 - 手动计算:")
        if isinstance(logits, torch.Tensor):
            # 获取 bin 的边界/中心点
            if hasattr(criterion, 'borders'):
                borders = criterion.borders  # bin 边界
                print(f"borders shape: {borders.shape if hasattr(borders, 'shape') else len(borders)}")
                # bin 中心点
                bin_centers = (borders[:-1] + borders[1:]) / 2
                print(f"bin_centers: {bin_centers[:5]}...{bin_centers[-5:]}")
            elif hasattr(criterion, 'bucket_widths'):
                print(f"bucket_widths: {criterion.bucket_widths}")
            
            # softmax 得到概率
            probs = torch.softmax(logits, dim=-1)  # (n_samples, n_bins)
            print(f"probs shape: {probs.shape}")
            print(f"probs sum (应该≈1): {probs.sum(dim=-1)[:3]}")
            
            # 如果有 bin_centers，计算期望和方差
            if hasattr(criterion, 'borders'):
                borders = criterion.borders
                bin_centers = (borders[:-1] + borders[1:]) / 2
                bin_centers = bin_centers.to(probs.device)
                
                # 期望 E[X] = sum(p_i * x_i)
                mean_manual = (probs * bin_centers).sum(dim=-1)
                
                # 方差 Var[X] = E[X^2] - E[X]^2
                mean_sq = (probs * bin_centers ** 2).sum(dim=-1)
                var_manual = mean_sq - mean_manual ** 2
                std_manual = torch.sqrt(var_manual.clamp(min=1e-8))
                
                print(f"\n手动计算的均值: {mean_manual[:5]}")
                print(f"full_out 的均值: {y_pred[:5]}")
                print(f"手动计算的方差: {var_manual[:5]}")
                print(f"手动计算的标准差: {std_manual[:5]}")
    
    # ========== 方法3: 使用分位数估计方差 ==========
    print(f"\n========== 方法3 - 分位数估计方差 ==========")
    try:
        # 获取 16% 和 84% 分位数（对应正态分布的 ±1σ）
        quantiles = regressor.predict(X_test, output_type="quantiles", quantiles=[0.159, 0.5, 0.841])
        q_low = quantiles[:, 0]   # 15.9% 分位数
        q_mid = quantiles[:, 1]   # 50% 分位数 (中位数)
        q_high = quantiles[:, 2]  # 84.1% 分位数
        
        # 标准差 ≈ (q_84 - q_16) / 2
        std_from_quantiles = (q_high - q_low) / 2.0
        
        print(f"15.9% 分位数: {q_low[:5]}")
        print(f"84.1% 分位数: {q_high[:5]}")
        print(f"分位数估计的标准差: {std_from_quantiles[:5]}")
    except Exception as e:
        print(f"分位数方法错误: {e}")
    
    # ========== 对比结果 ==========
    print(f"\n========== 预测结果对比 ==========")
    print(f"y_pred (mean): {y_pred[:5]}")
    print(f"y_test (true): {y_test[:5]}")
    
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"\nMSE: {mse:.6f}")
    
    print("\n✅ 测试完成!")


if __name__ == "__main__":
    main()

