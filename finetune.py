import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator
import matplotlib.pyplot as plt


# Branin 函数（标准定义，域：x1 ∈ [-5, 10], x2 ∈ [0, 15]）
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


def prepare_data(
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    准备 Branin 微调数据。

    返回四个 ndarray：
    - X_ctx_pool:  用于后续随机采样上下文点的“池”（来自 branin_dataset.csv）
    - X_test_grid: 全局 30x30 网格的输入点
    - y_ctx_pool:  X_ctx_pool 对应的 y
    - y_test_grid: X_test_grid 对应的 y（用解析 Branin 函数计算）

    上下文长度 2–20 的变化 **不在这里实现**，
    而是后续在采样任务时，从 X_ctx_pool / y_ctx_pool 里按需抽取。
    """
    print("--- 1. Data Preparation (Branin) ---")

    # ========== 1) 读取 branin_dataset.csv ==========
    data_path = config.get("data_path", "./data/branin_dataset.csv")
    df = pd.read_csv(data_path)

    # 假定列名为: x1, x2, y
    X_all = df[["x1", "x2"]].values.astype(np.float32)
    y_all = df["y"].values.astype(np.float32)

    # 允许通过 num_samples_to_use 下采样
    rng = np.random.default_rng(config.get("random_seed", 42))
    num_samples_to_use = min(config.get("num_samples_to_use", len(y_all)), len(y_all))
    indices = rng.choice(len(y_all), size=num_samples_to_use, replace=False)

    X_ctx_pool = X_all[indices]
    y_ctx_pool = y_all[indices]

    print(f"Context pool: {X_ctx_pool.shape[0]} samples")
    print(f"  X_ctx_pool shape: {X_ctx_pool.shape}")  # (N_ctx_pool, 2)
    print(f"  y_ctx_pool shape: {y_ctx_pool.shape}")  # (N_ctx_pool,)

    # ========== 2) 构造全局 30x30 网格作为 test ==========
    x1_min, x1_max = config.get("x1_range", (-5.0, 10.0))
    x2_min, x2_max = config.get("x2_range", (0.0, 15.0))
    grid_size = config.get("grid_size", 30)

    x1_lin = np.linspace(x1_min, x1_max, grid_size, dtype=np.float32)
    x2_lin = np.linspace(x2_min, x2_max, grid_size, dtype=np.float32)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)  # 每个都是 (grid_size, grid_size)

    # 展平成 (grid_size^2, 2)
    X_test_grid = np.stack([X1.ravel(), X2.ravel()], axis=1).astype(np.float32)
    y_test_grid = branin(X_test_grid[:, 0], X_test_grid[:, 1])

    print(f"Global test grid: {X_test_grid.shape[0]} points")
    print(f"  X_test_grid shape: {X_test_grid.shape}")  # (grid_size^2, 2)
    print(f"  y_test_grid shape: {y_test_grid.shape}")  # (grid_size^2,)

    # 简单展示前几行数据，帮助直观查看
    print("\n[Sample of context pool X, y]:")
    print(np.concatenate(
        [X_ctx_pool[:5], y_ctx_pool[:5, None]],
        axis=1
    ))

    print("\n[Sample of test grid X, y]:")
    print(np.concatenate(
        [X_test_grid[:5], y_test_grid[:5, None]],
        axis=1
    ))

    print("---------------------------\n")

    return X_ctx_pool, X_test_grid, y_ctx_pool, y_test_grid


import numpy as np
from torch.utils.data import Dataset, DataLoader


class BraninMetaDataset(Dataset):
    """
    每个 __getitem__ 返回一个“源任务”：
      - X_ctx:  (n_ctx, 2)   随机采样的上下文点，n_ctx ∈ [min_ctx, max_ctx]
      - y_ctx:  (n_ctx,)
      - X_test: (n_test, 2)  全局网格点（固定）
      - y_test: (n_test,)
    """
    def __init__(
        self,
        X_ctx_pool: np.ndarray,
        y_ctx_pool: np.ndarray,
        X_test_grid: np.ndarray,
        y_test_grid: np.ndarray,
        num_tasks: int = 1000,
        min_ctx: int = 2,
        max_ctx: int = 20,
        random_seed: int = 42,
    ):
        assert X_ctx_pool.shape[0] == y_ctx_pool.shape[0], "ctx pool size mismatch"
        assert X_test_grid.shape[0] == y_test_grid.shape[0], "test grid size mismatch"
        assert min_ctx >= 1, "min_ctx must be >= 1"
        assert max_ctx >= min_ctx, "max_ctx must be >= min_ctx"

        self.X_ctx_pool = X_ctx_pool
        self.y_ctx_pool = y_ctx_pool
        self.X_test_grid = X_test_grid
        self.y_test_grid = y_test_grid

        self.num_tasks = num_tasks
        self.min_ctx = min_ctx
        self.max_ctx = max_ctx

        self.rng = np.random.default_rng(random_seed)

    def __len__(self) -> int:
        # 数据集中一共多少个“源任务”
        return self.num_tasks

    def __getitem__(self, idx: int):
        # 1) 随机决定这个任务的上下文长度 n_ctx ∈ [min_ctx, max_ctx]
        n_pool = self.X_ctx_pool.shape[0]
        n_ctx = int(self.rng.integers(self.min_ctx, self.max_ctx + 1))

        # 2) 在 context pool 中无放回随机采样 n_ctx 个点
        indices = self.rng.choice(n_pool, size=n_ctx, replace=False)
        X_ctx = self.X_ctx_pool[indices]   # (n_ctx, 2)
        y_ctx = self.y_ctx_pool[indices]   # (n_ctx,)

        # 3) test 直接用固定的全局网格
        X_test = self.X_test_grid         # (n_test, 2)
        y_test = self.y_test_grid         # (n_test,)

        return X_ctx, y_ctx, X_test, y_test

def demo_meta_dataset(
    X_ctx_pool,
    X_test_grid,
    y_ctx_pool,
    y_test_grid,
):
    # 创建一个包含 5 个任务的小数据集先试试
    meta_ds = BraninMetaDataset(
        X_ctx_pool=X_ctx_pool,
        y_ctx_pool=y_ctx_pool,
        X_test_grid=X_test_grid,
        y_test_grid=y_test_grid,
        num_tasks=5,      # 先来 5 个任务看一眼
        min_ctx=2,
        max_ctx=20,
        random_seed=123,  # 为了可复现
    )

    # 简单 DataLoader，batch_size=1 表示一次取一个“任务”
    loader = DataLoader(meta_ds, batch_size=1, shuffle=False)

    print("--- 2. Inspect a few sampled tasks ---")
    for batch_idx, batch in enumerate(loader):
        X_ctx, y_ctx, X_test, y_test = batch  # 注意这是加了 batch 维度的

        # batch_size=1，所以可以去掉第一维更直观
        X_ctx = X_ctx[0].numpy()
        y_ctx = y_ctx[0].numpy()
        X_test = X_test[0].numpy()
        y_test = y_test[0].numpy()

        print(f"\nTask {batch_idx}:")
        print(f"  X_ctx shape:  {X_ctx.shape}")   # (n_ctx, 2)
        print(f"  y_ctx shape:  {y_ctx.shape}")   # (n_ctx,)")
        print(f"  X_test shape: {X_test.shape}")  # (n_test, 2)
        print(f"  y_test shape: {y_test.shape}")  # (n_test,)")

        # 展示前几个上下文点和网格点，直观感受一下
        print("  Sample context points (x1, x2, y):")
        print(np.concatenate([X_ctx[:3], y_ctx[:3, None]], axis=1))

        print("  Sample test grid points (x1, x2, y):")
        print(np.concatenate([X_test[:3], y_test[:3, None]], axis=1))

        if batch_idx >= 2:
            # 只看前三个任务，够检查结构了
            break

    print("---------------------------\n")

def setup_regressor(config: dict) -> tuple[TabPFNRegressor, dict]:
    """初始化 TabPFNRegressor，并返回模型本身和它的配置字典。"""
    print("--- 3. Model Setup (TabPFNRegressor) ---")

    regressor_config = {
        "ignore_pretraining_limits": True,
        "device": config["device"],
        "n_estimators": 1,  # 只用单模型，方便微调
        "random_state": config["random_seed"],
        "inference_precision": torch.float32,
    }

    regressor = TabPFNRegressor(
        **regressor_config,
        fit_mode="batched",
        differentiable_input=False,
    )

    print(f"Using device: {config['device']}")
    print("---------------------------\n")
    return regressor, regressor_config

import numpy as np


def make_branin_splitter(
    X_test_grid: np.ndarray,
    y_test_grid: np.ndarray,
    config: dict,
):
    """
    返回一个自定义 splitter，用在 TabPFNRegressor.get_preprocessed_datasets 里。

    作用：
      - 输入: (X_all, y_all) —— 实际上就是我们的 context pool
      - 输出: (X_ctx, X_test, y_ctx, y_test)
        其中:
          * X_ctx, y_ctx: 从 X_all, y_all 中随机抽取 k∈[min_context, max_context] 个点
          * X_test, y_test: 固定为全局 Branin 网格
    """
    min_c = config["min_context"]
    max_c = config["max_context"]
    rng = np.random.default_rng(config["random_seed"])

    def splitter(X_all: np.ndarray, y_all: np.ndarray):
        n = X_all.shape[0]
        ctx_size = rng.integers(min_c, max_c + 1)
        indices = rng.choice(np.arange(n), size=ctx_size, replace=False)

        X_ctx = X_all[indices]
        y_ctx = y_all[indices]

        # test 部分直接用全局网格
        X_test = X_test_grid
        y_test = y_test_grid

        return X_ctx, X_test, y_ctx, y_test

    return splitter

def create_finetuning_dataloader(
    regressor: TabPFNRegressor,
    X_ctx_pool: np.ndarray,
    y_ctx_pool: np.ndarray,
    X_test_grid: np.ndarray,
    y_test_grid: np.ndarray,
    config: dict,
) -> DataLoader:
    print("--- 4. Build finetuning datasets & dataloader ---")

    splitter = make_branin_splitter(X_test_grid, y_test_grid, config)

    # 这里的 max_data_size 对我们自定义的 splitter 实际影响不大，
    # 它主要作用在 TabPFN 内部的预处理；设得略大一点即可。
    max_data_size = config["finetuning"]["max_data_size"]

    training_datasets = regressor.get_preprocessed_datasets(
        X_ctx_pool,
        y_ctx_pool,
        splitter,
        max_data_size=max_data_size,
    )

    print(f"Number of meta-datasets from get_preprocessed_datasets: {len(training_datasets)}")

    finetuning_dataloader = DataLoader(
        training_datasets,
        batch_size=config["finetuning"]["meta_batch_size"],
        collate_fn=meta_dataset_collator,
    )

    # —— 打印一个 batch，确认形状是否合理 ——
    first_batch = next(iter(finetuning_dataloader))

    (
        X_trains_preprocessed,
        X_tests_preprocessed,
        y_trains_znorm,
        y_test_znorm,
        cat_ixs,
        confs,
        raw_space_bardist_,
        znorm_space_bardist_,
        _,
        _y_test_raw,
    ) = first_batch

    print("Inspect one preprocessed task (after collate, meta_batch_size=1):")
    print("  X_trains_preprocessed[0].shape:", X_trains_preprocessed[0].shape)
    print("  X_tests_preprocessed[0].shape :", X_tests_preprocessed[0].shape)
    print("  y_trains_znorm[0].shape       :", y_trains_znorm[0].shape)
    print("  y_test_znorm[0].shape         :", y_test_znorm[0].shape)
    print("---------------------------\n")

    return finetuning_dataloader

def evaluate_regressor_on_branin_grid(
    regressor: TabPFNRegressor,
    eval_config: dict,
    X_ctx_pool: np.ndarray,
    y_ctx_pool: np.ndarray,
    X_test_grid: np.ndarray,
    y_test_grid: np.ndarray,
    config: dict,
) -> tuple[float, float, float]:
    """在全局 Branin 网格上评估当前（微调后的）TabPFN。"""
    eval_regressor = clone_model_for_evaluation(regressor, eval_config, TabPFNRegressor)

    rng = np.random.default_rng(config["random_seed"])
    n_ctx_eval = config.get("eval_context_size", config["max_context"])

    idx = rng.choice(np.arange(X_ctx_pool.shape[0]), size=n_ctx_eval, replace=False)
    X_ctx_eval = X_ctx_pool[idx]
    y_ctx_eval = y_ctx_pool[idx]

    # 这里的 fit 是“给上下文”，不会再做梯度更新，只是用于 ICL 推理
    eval_regressor.fit(X_ctx_eval, y_ctx_eval)

    preds = eval_regressor.predict(X_test_grid)

    mse = mean_squared_error(y_test_grid, preds)
    mae = mean_absolute_error(y_test_grid, preds)
    r2 = r2_score(y_test_grid, preds)

    return mse, mae, r2

def run_finetuning(
    regressor: TabPFNRegressor,
    regressor_config: dict,
    finetuning_dataloader: DataLoader,
    X_ctx_pool: np.ndarray,
    y_ctx_pool: np.ndarray,
    X_test_grid: np.ndarray,
    y_test_grid: np.ndarray,
    config: dict,
) -> None:
    # 注意：必须先调用 get_preprocessed_datasets，再拿 models_[0] 初始化优化器
    if len(regressor.models_) > 1:
        raise ValueError(
            f"Your TabPFNRegressor uses multiple models ({len(regressor.models_)}). "
            "Finetuning is only supported for a single model."
        )

    model = regressor.models_[0]
    optimizer = Adam(model.parameters(), lr=config["finetuning"]["learning_rate"])

    print(
        f"--- Optimizer Initialized: Adam, LR: {config['finetuning']['learning_rate']} ---\n"
    )

    # 评估时的配置
    eval_config = {
        **regressor_config,
        "inference_config": {
            # 这里给个上界，Branin 上我们其实只用到 <= max_context
            "SUBSAMPLE_SAMPLES": config["n_inference_context_samples"],
        },
    }

    print("--- 5. Starting Finetuning & Evaluation (Branin) ---")

    num_epochs = config["finetuning"]["epochs"]

    for epoch in range(num_epochs + 1):
        # 1) 先在 Branin 全局网格上评估当前模型
        mse, mae, r2 = evaluate_regressor_on_branin_grid(
            regressor,
            eval_config,
            X_ctx_pool,
            y_ctx_pool,
            X_test_grid,
            y_test_grid,
            config,
        )
        status = "Initial" if epoch == 0 else f"Epoch {epoch}"
        print(
            f"📊 {status} Evaluation on Branin grid | "
            f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}"
        )

        # 2) 从 epoch=1 开始做微调
        if epoch == 0:
            print("---------------------------")
            continue

        progress_bar = tqdm(finetuning_dataloader, desc=f"Finetuning Epoch {epoch}")
        for data_batch in progress_bar:
            optimizer.zero_grad()

            (
                X_trains_preprocessed,
                X_tests_preprocessed,
                y_trains_znorm,
                y_test_znorm,
                cat_ixs,
                confs,
                raw_space_bardist_,
                znorm_space_bardist_,
                _,
                _y_test_raw,
            ) = data_batch

            # 取出 batch 中的第 0 个任务（因为 meta_batch_size=1）
            regressor.raw_space_bardist_ = raw_space_bardist_[0]
            regressor.znorm_space_bardist_ = znorm_space_bardist_[0]

            regressor.fit_from_preprocessed(
                X_trains_preprocessed,
                y_trains_znorm,
                cat_ixs,
                confs,
            )

            logits, _, _ = regressor.forward(X_tests_preprocessed)

            # 回归任务的 loss function 已经包含在 znorm_space_bardist_ 里
            loss_fn = znorm_space_bardist_[0]
            y_target = y_test_znorm

            loss = loss_fn(logits, y_target.to(config["device"])).mean()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        print("---------------------------")

    # # 在文件顶部添加导入  
    from tabpfn.model_loading import save_tabpfn_model  
    
    # 修改main函数末尾的保存部分  
    save_path = "./model/finetuned_tabpfn_branin.ckpt"  
    save_tabpfn_model(regressor, save_path)  
    print(f"Saved fine-tuned model weights to: {save_path}")


    print("--- ✅ Finetuning Finished ---")


def main() -> None:
    # —— 全局配置 —— 
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_seed": 42,

        # Branin 数据配置
        "num_context_samples": 10_000,  # context pool 大小
        "grid_size": 20,                # 20×20 网格 -> 400 个 test 点

        # 变长上下文
        "min_context": 2,
        "max_context": 20,

        # 评估时使用多少个上下文点
        "eval_context_size": 20,

        # 评估时 TabPFN 推理的最大上下文数量上界
        "n_inference_context_samples": 20,
    }

    # 微调超参数
    config["finetuning"] = {
        "epochs": 10,              # 先来几个 epoch 试试
        "learning_rate": 1.5e-6,  # 官方推荐的小学习率
        "meta_batch_size": 1,     # 目前必须为 1
        "max_data_size": 20,     # 传给 get_preprocessed_datasets 的 max_data_size
    }

    # 1) 构造 Branin 数据（context pool + 全局 test 网格）
    X_ctx_pool, X_test_grid, y_ctx_pool, y_test_grid = prepare_data(config)

    # 2) 用我们自己定义的 BraninMetaDataset 再抽几个任务看看形状是否合理
    demo_meta_dataset(X_ctx_pool, X_test_grid, y_ctx_pool, y_test_grid)

    # 3) 初始化 TabPFNRegressor
    regressor, regressor_config = setup_regressor(config)

    # 4) 构建 fine-tuning dataloader（带自定义 Branin splitter）
    finetuning_dataloader = create_finetuning_dataloader(
        regressor,
        X_ctx_pool,
        y_ctx_pool,
        X_test_grid,
        y_test_grid,
        config,
    )

    # 5) 运行微调 + 每个 epoch 后在全局 Branin 网格上评估
    run_finetuning(
        regressor,
        regressor_config,
        finetuning_dataloader,
        X_ctx_pool,
        y_ctx_pool,
        X_test_grid,
        y_test_grid,
        config,
    )


if __name__ == "__main__":
    main()
