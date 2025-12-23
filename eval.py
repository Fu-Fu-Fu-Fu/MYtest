import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import clone_model_for_evaluation


# ---- Branin & 数据准备（和你之前基本一样） ----
def branin(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
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
    ).astype(np.float32)


def prepare_data_for_eval(config: dict):
    df = pd.read_csv(config.get("data_path", "./data/branin_dataset.csv"))
    X_all = df[["x1", "x2"]].values.astype(np.float32)
    y_all = df["y"].values.astype(np.float32)

    rng = np.random.default_rng(config.get("random_seed", 42))
    num_use = min(config.get("num_samples_to_use", len(y_all)), len(y_all))
    idx = rng.choice(len(y_all), size=num_use, replace=False)

    X_ctx_pool = X_all[idx]
    y_ctx_pool = y_all[idx]

    x1_min, x1_max = config.get("x1_range", (-5.0, 10.0))
    x2_min, x2_max = config.get("x2_range", (0.0, 15.0))
    grid_size = config.get("grid_size", 20)

    x1_lin = np.linspace(x1_min, x1_max, grid_size, dtype=np.float32)
    x2_lin = np.linspace(x2_min, x2_max, grid_size, dtype=np.float32)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)

    X_test_grid = np.stack([X1.ravel(), X2.ravel()], axis=1).astype(np.float32)
    y_test_grid = branin(X_test_grid[:, 0], X_test_grid[:, 1])

    return X_ctx_pool, y_ctx_pool, X_test_grid, y_test_grid, X1, X2


# ---- 画等高线 + 计算 MSE（注意这里用 clone_model_for_evaluation） ----
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

    eval_config = {
        **regressor_config,
        "inference_config": {
            "SUBSAMPLE_SAMPLES": config["n_inference_context_samples"],
        },
    }

    rng = np.random.default_rng(config.get("eval_seed", 123))

    for k in context_sizes:
        # 1) 采样 k 个上下文点（对 base / tuned 用同一组）
        idx = rng.choice(np.arange(X_ctx_pool.shape[0]), size=k, replace=False)
        X_ctx = X_ctx_pool[idx]
        y_ctx = y_ctx_pool[idx]

        # 2) 用 clone_model_for_evaluation 构造评估模型

        # base_eval = clone_model_for_evaluation(
        #     base_regressor, eval_config, TabPFNRegressor
        # )
        # tuned_eval = clone_model_for_evaluation(
        #     tuned_regressor, eval_config, TabPFNRegressor
        # )

        # # 3) 在相同上下文上 fit（这里只是 ICL，不是梯度更新）
        # base_eval.fit(X_ctx, y_ctx)
        # tuned_eval.fit(X_ctx, y_ctx)
        # IMPORTANT: Initialize base model before using  
        base_regressor._initialize_model_variables()  
          
        # Use models directly without cloning  
        base_regressor.fit(X_ctx, y_ctx)  
        tuned_regressor.fit(X_ctx, y_ctx) 
        y_pred_base = base_regressor.predict(X_test_grid)  
        y_pred_tuned = tuned_regressor.predict(X_test_grid)
        # 4) 在全局网格上预测
        # y_pred_base = base_eval.predict(X_test_grid)
        # y_pred_tuned = tuned_eval.predict(X_test_grid)

        mse_base = mean_squared_error(y_test_grid, y_pred_base)
        mse_tuned = mean_squared_error(y_test_grid, y_pred_tuned)

        y_pred_base_grid = y_pred_base.reshape(grid_size, grid_size)
        y_pred_tuned_grid = y_pred_tuned.reshape(grid_size, grid_size)

        vmin = min(y_true_grid.min(), y_pred_base_grid.min(), y_pred_tuned_grid.min())
        vmax = max(y_true_grid.max(), y_pred_base_grid.max(), y_pred_tuned_grid.max())

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        levels = 30

        cs0 = axes[0].contourf(X1, X2, y_true_grid, levels=levels, vmin=vmin, vmax=vmax)
        axes[0].scatter(X_ctx[:, 0], X_ctx[:, 1],
                        c="white", edgecolors="black", s=30)
        axes[0].set_title(f"True Branin (k={k})")

        cs1 = axes[1].contourf(X1, X2, y_pred_base_grid, levels=levels, vmin=vmin, vmax=vmax)
        axes[1].scatter(X_ctx[:, 0], X_ctx[:, 1],
                        c="white", edgecolors="black", s=30)
        axes[1].set_title(f"Base TabPFN (k={k})\nMSE={mse_base:.3f}")

        cs2 = axes[2].contourf(X1, X2, y_pred_tuned_grid, levels=levels, vmin=vmin, vmax=vmax)
        axes[2].scatter(X_ctx[:, 0], X_ctx[:, 1],
                        c="white", edgecolors="black", s=30)
        axes[2].set_title(f"Finetuned TabPFN (k={k})\nMSE={mse_tuned:.3f}")

        for ax in axes:
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")

        fig.colorbar(cs2, ax=axes, orientation="vertical",
                     fraction=0.03, pad=0.04)

        plt.suptitle(f"Branin predictions with {k} context points", fontsize=14)
        # plt.show()
        # === 新增：保存 png ===
        import os
        os.makedirs("./figs", exist_ok=True)
        save_path = f"./figs/branin_contours_k{k}.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

        print(f"[k={k}] Base MSE: {mse_base:.4f} | Finetuned MSE: {mse_tuned:.4f}")
        print(f"Saved figure to {save_path}")
        # print(f"[k={k}] Base MSE: {mse_base:.4f} | Finetuned MSE: {mse_tuned:.4f}")


def main():
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_seed": 42,
        "eval_seed": 123,

        "data_path": "./data/branin_dataset.csv",
        "num_samples_to_use": 10_000,
        "x1_range": (-5.0, 10.0),
        "x2_range": (0.0, 15.0),
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
        fit_mode="batched",
        differentiable_input=False,
    )

    # from tabpfn.model_loading import load_fitted_tabpfn_model  
  
    # Load the fine-tuned model  
    tuned_regressor = TabPFNRegressor(  
        model_path="./model/finetuned_tabpfn_branin_family.ckpt" ,  # Load from the checkpoint  
        device=config["device"],  
        n_estimators=1,  
        random_state=config["random_seed"],  
        inference_precision=torch.float32,  
        fit_mode="batched",  
        differentiable_input=False,  
    )
    print("✅ Loaded fine-tuned backbone.")

    # 4) 画等高线 + 打印 MSE
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
