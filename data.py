import numpy as np
import pandas as pd

# -----------------------------
# 1. 定义 Branin 函数
# -----------------------------
def branin(x1, x2):
    # 常用的 Branin 函数定义（通常用于 [-5, 10] x [0, 15]）
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
    )


# -----------------------------
# 2. 生成 Branin 数据集并保存
# -----------------------------
def generate_branin_dataset(n_samples=10_000,
                            x1_range=(-5.0, 10.0),
                            x2_range=(0.0, 15.0),
                            save_path="./data/branin_dataset.csv"):
    """
    生成基于 Branin 函数的微调数据集：
    每一行数据为 x1, x2, y，并保存为 CSV。
    """
    rng = np.random.default_rng(seed=42)

    # 在给定范围内均匀采样 x1, x2
    x1 = rng.uniform(x1_range[0], x1_range[1], size=n_samples)
    x2 = rng.uniform(x2_range[0], x2_range[1], size=n_samples)

    # 计算 y = branin(x1, x2)
    y = branin(x1, x2)

    # 组装为 DataFrame，列顺序：x1, x2, y
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "y": y,
    })

    # 保存到 CSV 文件
    df.to_csv(save_path, index=False)
    print(f"Saved Branin dataset with {n_samples} samples to: {save_path}")


if __name__ == "__main__":
    generate_branin_dataset()
