import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os

from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from tabpfn import TabPFNRegressor


# ---- Six-Hump Camel & 数据准备 ----
def six_hump_camel(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    标准 2D Six-Hump Camel 函数，定义域通常为 x1 ∈ [-3, 3], x2 ∈ [-2, 2]
    全局最小值在 (0.0898, -0.7126) 和 (-0.0898, 0.7126)，值为 ≈ -1.0316

    Six-Hump Camel:
    f(x1, x2) = (4 - 2.1*x1^2 + x1^4/3)*x1^2 + x1*x2 + (-4 + 4*x2^2)*x2^2
    """
    term1 = (4 - 2.1*x1**2 + x1**4/3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4*x2**2) * x2**2
    return (term1 + term2 + term3).astype(np.float32)


def prepare_data_for_eval(config: dict):
    df = pd.read_csv(config.get("data_path", "./data/six_hump_camel_dataset.csv"))
    X_all = df[["x1", "x2"]].values.astype(np.float32)
    y_all = df["y"].values.astype(np.float32)

    rng = np.random.default_rng(config.get("random_seed", 42))
    num_use = min(config.get("num_samples_to_use", len(y_all)), len(y_all))
    idx = rng.choice(len(y_all), size=num_use, replace=False)

    X_ctx_pool = X_all[idx]
    y_ctx_pool = y_all[idx]

    x1_min, x1_max = config.get("x1_range", (-3.0, 3.0))
    x2_min, x2_max = config.get("x2_range", (-2.0, 2.0))
    grid_size = config.get("grid_size", 20)

    x1_lin = np.linspace(x1_min, x1_max, grid_size, dtype=np.float32)
    x2_lin = np.linspace(x2_min, x2_max, grid_size, dtype=np.float32)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)

    X_test_grid = np.stack([X1.ravel(), X2.ravel()], axis=1).astype(np.float32)
    y_test_grid = six_hump_camel(X_test_grid[:, 0], X_test_grid[:, 1])

    return X_ctx_pool, y_ctx_pool, X_test_grid, y_test_grid, X1, X2


# ---- 画等高线 + 计算 MSE，包含 GP ----
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
    config: dict,
    context_sizes=(4, 8, 12, 16, 20),
):
    grid_size = config["grid_size"]
    y_true_grid = y_test_grid.reshape(grid_size, grid_size)

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

        y_pred_base = base_regressor.predict(X_test_grid)   # (N_grid,)
        y_pred_tuned = tuned_regressor.predict(X_test_grid) # (N_grid,)

        # ========== 3) 高斯过程回归(GP) ==========
        # Six-Hump Camel 值域约为 [-1.03, 20+]，适当调整 kernel 参数
        kernel = (
            C(1.0, (1e-3, 1e3))
            * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,              # 噪声由 WhiteKernel 控制
            normalize_y=True,
            n_restarts_optimizer=3, # 稍微稳一点
            random_state=0,
        )
        gp.fit(X_ctx, y_ctx)
        y_pred_gp, y_std_gp = gp.predict(X_test_grid, return_std=True)

        # ========== 4) MSE 计算 ==========
        mse_base = mean_squared_error(y_test_grid, y_pred_base)
        mse_tuned = mean_squared_error(y_test_grid, y_pred_tuned)
        mse_gp = mean_squared_error(y_test_grid, y_pred_gp)

        # 重塑为 (grid_size, grid_size) 画等高线
        y_pred_base_grid = y_pred_base.reshape(grid_size, grid_size)
        y_pred_tuned_grid = y_pred_tuned.reshape(grid_size, grid_size)
        y_pred_gp_grid = y_pred_gp.reshape(grid_size, grid_size)

        # 统一颜色范围，方便肉眼比较
        vmin = min(
            y_true_grid.min(),
            y_pred_base_grid.min(),
            y_pred_tuned_grid.min(),
            y_pred_gp_grid.min(),
        )
        vmax = max(
            y_true_grid.max(),
            y_pred_base_grid.max(),
            y_pred_tuned_grid.max(),
            y_pred_gp_grid.max(),
        )

        # ========== 5) 画 4 列图：True / Base / Tuned / GP ==========
        fig, axes = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)
        levels = 30

        # 真值
        cs0 = axes[0].contourf(X1, X2, y_true_grid, levels=levels, vmin=vmin, vmax=vmax)
        axes[0].scatter(X_ctx[:, 0], X_ctx[:, 1],
                        c="white", edgecolors="black", s=30)
        axes[0].set_title(f"True Six-Hump Camel (k={k})")

        # Base TabPFN
        cs1 = axes[1].contourf(X1, X2, y_pred_base_grid, levels=levels, vmin=vmin, vmax=vmax)
        axes[1].scatter(X_ctx[:, 0], X_ctx[:, 1],
                        c="white", edgecolors="black", s=30)
        axes[1].set_title(f"Base TabPFN\n(k={k}, MSE={mse_base:.3f})")

        # Finetuned TabPFN
        cs2 = axes[2].contourf(X1, X2, y_pred_tuned_grid, levels=levels, vmin=vmin, vmax=vmax)
        axes[2].scatter(X_ctx[:, 0], X_ctx[:, 1],
                        c="white", edgecolors="black", s=30)
        axes[2].set_title(f"Finetuned TabPFN\n(k={k}, MSE={mse_tuned:.3f})")

        # GP
        cs3 = axes[3].contourf(X1, X2, y_pred_gp_grid, levels=levels, vmin=vmin, vmax=vmax)
        axes[3].scatter(X_ctx[:, 0], X_ctx[:, 1],
                        c="white", edgecolors="black", s=30)
        axes[3].set_title(f"GP\n(k={k}, MSE={mse_gp:.3f})")

        for ax in axes:
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")

        # 共用一个 colorbar（用最后一幅的 handle 即可）
        fig.colorbar(cs3, ax=axes, orientation="vertical",
                     fraction=0.03, pad=0.04)

        plt.suptitle(f"Six-Hump Camel predictions with {k} context points", fontsize=14)

        save_path = f"./figs/six_hump_camel_contours_k{k}_with_gp.png"
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

        "data_path": "./data/six_hump_camel_dataset.csv",
        "num_samples_to_use": 10_000,
        # Six-Hump Camel 定义域为 [-3, 3] x [-2, 2]
        "x1_range": (-3.0, 3.0),
        "x2_range": (-2.0, 2.0),
        "grid_size": 20,

        "n_inference_context_samples": 20,
    }

    # 1) 数据
    X_ctx_pool, y_ctx_pool, X_test_grid, y_test_grid, X1, X2 = \
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
        model_path="./model/finetuned_tabpfn_six_hump_camel_family.ckpt",
        device=config["device"],
        n_estimators=1,
        random_state=config["random_seed"],
        inference_precision=torch.float32,
        differentiable_input=False,
    )
    print("Loaded fine-tuned backbone.")

    # 4) 画等高线 + 打印 MSE（含 GP）
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
        config=config,
        context_sizes=[4, 8, 12, 16, 20],
    )


if __name__ == "__main__":
    main()
