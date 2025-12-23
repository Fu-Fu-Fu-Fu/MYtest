import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os

from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from tabpfn import TabPFNRegressor


# ---- Hartmann-3D & 数据准备 ----
def hartmann_3d(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    """
    标准 3D Hartmann 函数，定义域通常为 x1, x2, x3 ∈ [0, 1]
    全局最小值在 (0.114614, 0.555649, 0.852547)，值约为 -3.86278

    Hartmann-3D:
    f(x) = -∑_{i=1}^{4} α_i * exp(-∑_{j=1}^{3} A_{ij} * (x_j - P_{ij})^2)
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2], dtype=np.float32)
    A = np.array([
        [3.0, 10, 30],
        [0.1, 10, 35],
        [3.0, 10, 30],
        [0.1, 10, 35]
    ], dtype=np.float32)
    P = np.array([
        [0.3689, 0.1170, 0.2673],
        [0.4699, 0.4387, 0.7470],
        [0.1091, 0.8732, 0.5547],
        [0.0381, 0.5743, 0.8828]
    ], dtype=np.float32)

    # 将输入组合成 (N, 3) 数组
    X = np.stack([x1, x2, x3], axis=-1)  # (N, 3)

    result = np.zeros(X.shape[0], dtype=np.float32)
    for i in range(4):
        inner = np.sum(A[i] * (X - P[i])**2, axis=-1)
        result -= alpha[i] * np.exp(-inner)

    return result.astype(np.float32)


def prepare_data_for_eval(config: dict):
    df = pd.read_csv(config.get("data_path", "./data/hartmann_3d_dataset.csv"))
    X_all = df[["x1", "x2", "x3"]].values.astype(np.float32)
    y_all = df["y"].values.astype(np.float32)

    rng = np.random.default_rng(config.get("random_seed", 42))
    num_use = min(config.get("num_samples_to_use", len(y_all)), len(y_all))
    idx = rng.choice(len(y_all), size=num_use, replace=False)

    X_ctx_pool = X_all[idx]
    y_ctx_pool = y_all[idx]

    x1_min, x1_max = config.get("x1_range", (0.0, 1.0))
    x2_min, x2_max = config.get("x2_range", (0.0, 1.0))
    x3_min, x3_max = config.get("x3_range", (0.0, 1.0))
    grid_size = config.get("grid_size", 20)

    x1_lin = np.linspace(x1_min, x1_max, grid_size, dtype=np.float32)
    x2_lin = np.linspace(x2_min, x2_max, grid_size, dtype=np.float32)
    x3_lin = np.linspace(x3_min, x3_max, grid_size, dtype=np.float32)
    X1, X2, X3 = np.meshgrid(x1_lin, x2_lin, x3_lin, indexing='ij')

    X_test_grid = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=1).astype(np.float32)
    y_test_grid = hartmann_3d(X_test_grid[:, 0], X_test_grid[:, 1], X_test_grid[:, 2])

    return X_ctx_pool, y_ctx_pool, X_test_grid, y_test_grid, X1, X2, X3


def prepare_2d_slice_for_plot(config: dict, x3_fixed: float = 0.5):
    """
    准备用于可视化的2D切片数据（固定x3）
    """
    x1_min, x1_max = config.get("x1_range", (0.0, 1.0))
    x2_min, x2_max = config.get("x2_range", (0.0, 1.0))
    grid_size = config.get("grid_size", 20)

    x1_lin = np.linspace(x1_min, x1_max, grid_size, dtype=np.float32)
    x2_lin = np.linspace(x2_min, x2_max, grid_size, dtype=np.float32)
    X1_2d, X2_2d = np.meshgrid(x1_lin, x2_lin)

    # 固定 x3
    X3_2d = np.full_like(X1_2d, x3_fixed)

    X_slice = np.stack([X1_2d.ravel(), X2_2d.ravel(), X3_2d.ravel()], axis=1).astype(np.float32)
    y_slice = hartmann_3d(X_slice[:, 0], X_slice[:, 1], X_slice[:, 2])

    return X_slice, y_slice, X1_2d, X2_2d


# ---- 画等高线 + 计算 MSE，包含 GP（可视化前两维，固定x3） ----
def eval_and_plot_for_context_sizes(
    base_regressor: TabPFNRegressor,
    tuned_regressor: TabPFNRegressor,
    regressor_config: dict,
    X_ctx_pool: np.ndarray,
    y_ctx_pool: np.ndarray,
    X_test_grid: np.ndarray,
    y_test_grid: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    X3: np.ndarray,
    config: dict,
    context_sizes=(4, 8, 12, 16, 20),
    x3_fixed: float = 0.5,
):
    grid_size = config["grid_size"]

    # 准备2D切片用于可视化
    X_slice, y_slice, X1_2d, X2_2d = prepare_2d_slice_for_plot(config, x3_fixed)
    y_true_2d = y_slice.reshape(grid_size, grid_size)

    rng = np.random.default_rng(config.get("eval_seed", 123))
    os.makedirs("./figs", exist_ok=True)
    base_regressor._initialize_model_variables()
    for k in context_sizes:
        # 1) 采样 k 个上下文点（对 base / tuned / GP 用同一组）
        idx = rng.choice(np.arange(X_ctx_pool.shape[0]), size=k, replace=False)
        X_ctx = X_ctx_pool[idx]
        y_ctx = y_ctx_pool[idx]

        # ========== 2) TabPFN 预测（基线 & 微调） ==========
        # 用 TabPFN 自带的 fit 做 ICL 拟合上下文（不会做梯度更新）
        base_regressor.fit(X_ctx, y_ctx)
        tuned_regressor.fit(X_ctx, y_ctx)

        # 在2D切片上预测
        y_pred_base_slice = base_regressor.predict(X_slice)   # (N_slice,)
        y_pred_tuned_slice = tuned_regressor.predict(X_slice) # (N_slice,)

        # 在完整3D网格上预测（用于计算MSE）
        y_pred_base = base_regressor.predict(X_test_grid)
        y_pred_tuned = tuned_regressor.predict(X_test_grid)

        # ========== 3) 高斯过程回归(GP) ==========
        # Hartmann-3D 值域约为 [-3.86, 0]，适当调整 kernel 参数
        kernel = (
            C(1.0, (1e-3, 1e3))
            * RBF(length_scale=[0.5, 0.5, 0.5], length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,              # 噪声由 WhiteKernel 控制
            normalize_y=True,
            n_restarts_optimizer=3, # 稍微稳一点
            random_state=0,
        )
        gp.fit(X_ctx, y_ctx)
        y_pred_gp_slice, y_std_gp_slice = gp.predict(X_slice, return_std=True)
        y_pred_gp = gp.predict(X_test_grid)

        # ========== 4) MSE 计算（在完整3D网格上） ==========
        mse_base = mean_squared_error(y_test_grid, y_pred_base)
        mse_tuned = mean_squared_error(y_test_grid, y_pred_tuned)
        mse_gp = mean_squared_error(y_test_grid, y_pred_gp)

        # 重塑为 (grid_size, grid_size) 画等高线（2D切片）
        y_pred_base_2d = y_pred_base_slice.reshape(grid_size, grid_size)
        y_pred_tuned_2d = y_pred_tuned_slice.reshape(grid_size, grid_size)
        y_pred_gp_2d = y_pred_gp_slice.reshape(grid_size, grid_size)

        # 统一颜色范围，方便肉眼比较
        vmin = min(
            y_true_2d.min(),
            y_pred_base_2d.min(),
            y_pred_tuned_2d.min(),
            y_pred_gp_2d.min(),
        )
        vmax = max(
            y_true_2d.max(),
            y_pred_base_2d.max(),
            y_pred_tuned_2d.max(),
            y_pred_gp_2d.max(),
        )

        # ========== 5) 画 4 列图：True / Base / Tuned / GP ==========
        fig, axes = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)
        levels = 30

        # 上下文点在2D切片上的投影（只显示x1, x2）
        X_ctx_2d = X_ctx[:, :2]

        # 真值
        cs0 = axes[0].contourf(X1_2d, X2_2d, y_true_2d, levels=levels, vmin=vmin, vmax=vmax)
        axes[0].scatter(X_ctx_2d[:, 0], X_ctx_2d[:, 1],
                        c="white", edgecolors="black", s=30)
        axes[0].set_title(f"True Hartmann-3D (k={k}, x3={x3_fixed})")

        # Base TabPFN
        cs1 = axes[1].contourf(X1_2d, X2_2d, y_pred_base_2d, levels=levels, vmin=vmin, vmax=vmax)
        axes[1].scatter(X_ctx_2d[:, 0], X_ctx_2d[:, 1],
                        c="white", edgecolors="black", s=30)
        axes[1].set_title(f"Base TabPFN\n(k={k}, MSE={mse_base:.3f})")

        # Finetuned TabPFN
        cs2 = axes[2].contourf(X1_2d, X2_2d, y_pred_tuned_2d, levels=levels, vmin=vmin, vmax=vmax)
        axes[2].scatter(X_ctx_2d[:, 0], X_ctx_2d[:, 1],
                        c="white", edgecolors="black", s=30)
        axes[2].set_title(f"Finetuned TabPFN\n(k={k}, MSE={mse_tuned:.3f})")

        # GP
        cs3 = axes[3].contourf(X1_2d, X2_2d, y_pred_gp_2d, levels=levels, vmin=vmin, vmax=vmax)
        axes[3].scatter(X_ctx_2d[:, 0], X_ctx_2d[:, 1],
                        c="white", edgecolors="black", s=30)
        axes[3].set_title(f"GP\n(k={k}, MSE={mse_gp:.3f})")

        for ax in axes:
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")

        # 共用一个 colorbar（用最后一幅的 handle 即可）
        fig.colorbar(cs3, ax=axes, orientation="vertical",
                     fraction=0.03, pad=0.04)

        plt.suptitle(f"Hartmann-3D predictions with {k} context points (x3={x3_fixed})", fontsize=14)

        save_path = f"./figs/hartmann_3d_contours_k{k}_with_gp.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

        print(
            f"[k={k}] Base MSE: {mse_base:.4f} | "
            f"Finetuned MSE: {mse_tuned:.4f} | "
            f"GP MSE: {mse_gp:.4f}"
        )
        print(f"Saved figure to {save_path}")


def main():
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_seed": 42,
        "eval_seed": 123,

        "data_path": "./data/hartmann_3d_dataset.csv",
        "num_samples_to_use": 10_000,
        # Hartmann-3D 定义域为 [0, 1] x [0, 1] x [0, 1]
        "x1_range": (0.0, 1.0),
        "x2_range": (0.0, 1.0),
        "x3_range": (0.0, 1.0),
        "grid_size": 20,

        "n_inference_context_samples": 20,
    }

    # 1) 数据
    X_ctx_pool, y_ctx_pool, X_test_grid, y_test_grid, X1, X2, X3 = \
        prepare_data_for_eval(config)

    # 2) baseline 模型（预训练 TabPFN）
    regressor_config = {
        "ignore_pretraining_limits": True,
        "device": config["device"],
        "n_estimators": 1,
        "random_state": config["random_seed"],
        "inference_precision": torch.float32,
    }
    base_regressor = TabPFNRegressor(
        **regressor_config,
        differentiable_input=False,
    )

    # 3) 加载微调好的 TabPFN（训练好的 ckpt）
    tuned_regressor = TabPFNRegressor(
        model_path="./model/finetuned_tabpfn_hartmann_3d_family.ckpt",
        device=config["device"],
        n_estimators=1,
        random_state=config["random_seed"],
        inference_precision=torch.float32,
        differentiable_input=False,
    )
    print("Loaded fine-tuned backbone.")

    # 4) 画等高线 + 打印 MSE（含 GP），可视化前两维（固定x3=0.5）
    eval_and_plot_for_context_sizes(
        base_regressor=base_regressor,
        tuned_regressor=tuned_regressor,
        regressor_config=regressor_config,
        X_ctx_pool=X_ctx_pool,
        y_ctx_pool=y_ctx_pool,
        X_test_grid=X_test_grid,
        y_test_grid=y_test_grid,
        X1=X1,
        X2=X2,
        X3=X3,
        config=config,
        context_sizes=[4, 8, 12, 16, 20],
        x3_fixed=0.5,
    )


if __name__ == "__main__":
    main()
