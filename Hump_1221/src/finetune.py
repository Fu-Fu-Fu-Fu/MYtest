import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.signal import correlate2d
from scipy.ndimage import shift
from scipy.stats import spearmanr

from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator
from tabpfn.model_loading import save_tabpfn_model


# ============================================================
# 0. 标准 Six-Hump Camel 函数（numpy 版，主要用于检查和对比）
# ============================================================
def six_hump_camel_np(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
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


# ============================================================
# 0.1 生成 Six-Hump Camel 数据集
# ============================================================
def generate_six_hump_camel_dataset(
    n_samples: int = 10_000,
    x1_range: Tuple[float, float] = (-3.0, 3.0),
    x2_range: Tuple[float, float] = (-2.0, 2.0),
    save_path: str = "./data/six_hump_camel_dataset.csv",
    seed: int = 42,
) -> None:
    """
    生成基于 Six-Hump Camel 函数的微调数据集：
    每一行数据为 x1, x2, y，并保存为 CSV。
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rng = np.random.default_rng(seed=seed)

    # 在给定范围内均匀采样 x1, x2
    x1 = rng.uniform(x1_range[0], x1_range[1], size=n_samples).astype(np.float32)
    x2 = rng.uniform(x2_range[0], x2_range[1], size=n_samples).astype(np.float32)

    # 计算 y = six_hump_camel(x1, x2)
    y = six_hump_camel_np(x1, x2)

    # 组装为 DataFrame，列顺序：x1, x2, y
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "y": y,
    })

    # 保存到 CSV 文件
    df.to_csv(save_path, index=False)
    print(f"Saved Six-Hump Camel dataset with {n_samples} samples to: {save_path}")
    print(f"   x1 range: [{x1.min():.4f}, {x1.max():.4f}]")
    print(f"   x2 range: [{x2.min():.4f}, {x2.max():.4f}]")
    print(f"   y  range: [{y.min():.4f}, {y.max():.4f}]")
    print("---------------------------\n")


# ============================================================
# 1. Six-Hump Camel 变体族（torch 版），用于 similarity 与生成变体函数值
# ============================================================
def six_hump_camel_family_torch(
    x: torch.Tensor,
    dx1: float = 0.0,
    dx2: float = 0.0,
    sx1: float = 1.0,
    sx2: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> torch.Tensor:
    """
    Six-Hump Camel 变体族：在输入上做线性变换 (sx, dx)，在输出上做仿射变换 (alpha, beta)。
    x: (..., 2) 的 Tensor，最后一维为 [x1, x2]
    """
    # 支持 (N,2) 或 (B,N,2) 等形状
    x1 = x[..., 0]
    x2 = x[..., 1]

    # 输入平移 + 缩放
    x1_t = sx1 * x1 + dx1
    x2_t = sx2 * x2 + dx2

    # Six-Hump Camel 主体
    term1 = (4 - 2.1*x1_t**2 + x1_t**4/3) * x1_t**2
    term2 = x1_t * x2_t
    term3 = (-4 + 4*x2_t**2) * x2_t**2
    y = term1 + term2 + term3

    # 输出仿射变换
    y = alpha * y + beta
    return y


def six_hump_camel_family_numpy_from_params(
    X: np.ndarray,
    variant_params: Dict[str, float],
    device: str = "cpu",
) -> np.ndarray:
    """
    方便在 numpy 上调用 six_hump_camel_family_torch：
    X: (N, 2) numpy array
    返回: (N,) numpy array
    """
    x_tensor = torch.from_numpy(X).float().unsqueeze(0).to(device)  # (1, N, 2)
    with torch.no_grad():
        y = six_hump_camel_family_torch(x_tensor, **variant_params).cpu().numpy()[0]
    return y.astype(np.float32)


# ============================================================
# 2. 相似度度量：对齐 Spearman
# ============================================================
def find_best_alignment_1d(
    y_source: np.ndarray,
    y_target: np.ndarray,
    grid_shape: Tuple[int, int],
) -> Tuple[float, np.ndarray]:
    """
    使用 2D 互相关找到最佳对齐位置（1D 数据需转回 2D 网格）
    返回: (aligned_spearman, shift_vec)
    """
    # 将 1D 数组转为 2D 网格
    Z_source = y_source.reshape(grid_shape)
    Z_target = y_target.reshape(grid_shape)

    # 标准化
    Z_source_norm = (Z_source - np.mean(Z_source)) / (np.std(Z_source) + 1e-8)
    Z_target_norm = (Z_target - np.mean(Z_target)) / (np.std(Z_target) + 1e-8)

    # 2D 互相关
    correlation = correlate2d(Z_source_norm, Z_target_norm, mode="same")

    # 找到最大相关位置
    max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    center = np.array(correlation.shape) // 2
    shift_vec = np.array(max_idx) - center

    # 对目标函数进行位移
    Z_target_shifted = shift(Z_target, shift_vec, mode="constant", cval=np.nan)

    # 只在有效区域计算 Spearman（排除 NaN）
    valid_mask = ~np.isnan(Z_target_shifted)

    total_points = Z_source.size
    overlap_points = np.sum(valid_mask)
    overlap_ratio = overlap_points / total_points

    if np.sum(valid_mask) < 10:
        # 有效点太少就放弃
        return 0.0, shift_vec

    source_flat = Z_source[valid_mask].flatten()
    target_flat = Z_target_shifted[valid_mask].flatten()

    # 计算对齐后的 Spearman
    rho, _ = spearmanr(source_flat, target_flat)
    if rho is None:
        rho = 0.0
    rho = float(rho) * float(overlap_ratio)  # 乘以重叠率作为惩罚

    return rho, shift_vec


def calculate_similarity(
    variant_params: Dict[str, float],
    grid_points: np.ndarray,
    y_base: np.ndarray,
    grid_shape: Tuple[int, int],
    device: str = "cpu",
) -> float:
    """
    计算变体与标准 Six-Hump Camel 在同一网格上的对齐 Spearman 相关系数。
    """
    x_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        y_var = (
            six_hump_camel_family_torch(x_tensor, **variant_params)
            .cpu()
            .numpy()
            .flatten()
            .astype(np.float32)
        )

    rho_aligned, _ = find_best_alignment_1d(y_base, y_var, grid_shape)
    return float(rho_aligned)


# ============================================================
# 3. 生成基础网格 & 选择 32 个变体
# ============================================================
def build_base_grid_and_y(config: dict):
    """
    构造标准 Six-Hump Camel 的全局测试网格：
      - X_test_grid: (G^2, 2)
      - X1, X2: (G, G) meshgrid
      - y_base: (G^2,) 标准 Six-Hump Camel 在该网格上的值
    """
    x1_min, x1_max = config.get("x1_range", (-3.0, 3.0))
    x2_min, x2_max = config.get("x2_range", (-2.0, 2.0))
    grid_size = config.get("grid_size", 20)

    x1_lin = np.linspace(x1_min, x1_max, grid_size, dtype=np.float32)
    x2_lin = np.linspace(x2_min, x2_max, grid_size, dtype=np.float32)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)

    X_test_grid = np.stack([X1.ravel(), X2.ravel()], axis=1).astype(np.float32)
    y_base = six_hump_camel_np(X_test_grid[:, 0], X_test_grid[:, 1])  # (G^2,)

    print(f"Base grid: {X_test_grid.shape[0]} points, grid_size={grid_size}")
    print(f"  X_test_grid shape: {X_test_grid.shape}")
    print(f"  y_base shape      : {y_base.shape}")
    print("---------------------------\n")

    return X_test_grid, X1, X2, y_base


def sample_variant_params(rng: np.random.Generator) -> Dict[str, float]:
    """
    随机采样一组 Six-Hump Camel 变体参数。
    Six-Hump Camel 定义域为 x1 ∈ [-3, 3], x2 ∈ [-2, 2]，值域约为 [-1.03, 20+]
    """
    params = {
        # 输入平移（范围较小以保持在定义域内）
        "dx1": float(rng.uniform(-0.5, 0.5)),
        "dx2": float(rng.uniform(-0.3, 0.3)),
        # 输入缩放
        "sx1": float(rng.uniform(0.8, 1.2)),
        "sx2": float(rng.uniform(0.8, 1.2)),
        # 输出缩放与平移（Six-Hump Camel 值域约为 [-1.03, 20+]）
        "alpha": float(rng.uniform(0.8, 1.2)),
        "beta": float(rng.uniform(-2.0, 2.0)),
    }
    return params


def select_six_hump_camel_variants(
    config: dict,
    grid_points: np.ndarray,
    y_base: np.ndarray,
    grid_shape: Tuple[int, int],
) -> Tuple[List[Dict[str, float]], np.ndarray, List[float]]:
    """
    用拒绝采样的方式，选出若干个相似度在 [rho_min, rho_max] 的 Six-Hump Camel 变体。
    返回：
      - variants:       长度 num_variants 的参数字典列表
      - variant_y_grids: shape (num_variants, G^2)，每行是一个变体在全局网格上的值
      - variant_rhos:   每个变体对应的相似度 rho
    """
    device = config["device"]
    rng = np.random.default_rng(config.get("variant_seed", config["random_seed"]))

    num_variants = config.get("num_variants", 32)
    rho_min, rho_max = config.get("similarity_range", (0.4, 0.6))
    max_trials = config.get("max_variant_trials", num_variants * 200)

    variants: List[Dict[str, float]] = []
    variant_y_list: List[np.ndarray] = []
    variant_rhos: List[float] = []

    print(
        f"--- Select Six-Hump Camel variants ---\n"
        f"Target similarity range: [{rho_min:.2f}, {rho_max:.2f}], "
        f"num_variants={num_variants}, max_trials={max_trials}"
    )

    trials = 0
    while len(variants) < num_variants and trials < max_trials:
        trials += 1
        params = sample_variant_params(rng)
        rho = calculate_similarity(params, grid_points, y_base, grid_shape, device=device)

        if rho_min <= rho <= rho_max:
            # 计算并缓存变体在网格上的值
            x_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                y_var = (
                    six_hump_camel_family_torch(x_tensor, **params)
                    .cpu()
                    .numpy()
                    .flatten()
                    .astype(np.float32)
                )

            variants.append(params)
            variant_y_list.append(y_var)
            variant_rhos.append(rho)

            print(
                f"  Accepted variant {len(variants):2d}/{num_variants} "
                f"(trial={trials:4d}), rho={rho:.3f}"
            )

    if len(variants) < num_variants:
        print(
            f"WARNING: Only found {len(variants)} variants in range [{rho_min},{rho_max}] "
            f"after {trials} trials."
        )

    variant_y_grids = np.stack(variant_y_list, axis=0)  # (num_variants, G^2)

    print("\nSelected variants summary:")
    for i, rho in enumerate(variant_rhos):
        print(f"  Variant {i:2d}: rho={rho:.3f}")
    print("---------------------------\n")

    return variants, variant_y_grids, variant_rhos


# ============================================================
# 4. 为每个变体构造上下文池（基于已有 six_hump_camel_dataset.csv 的 X）
# ============================================================
def load_base_context_points(config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 six_hump_camel_dataset.csv 中读取基础上下文点：
      - X_ctx_base: (N_ctx_base, 2)
      - y_ctx_base: (N_ctx_base,) —— 标准 Six-Hump Camel 在这些点上的值
    """
    data_path = config.get("data_path", "./data/six_hump_camel_dataset.csv")
    # 检查数据集是否存在，不存在则生成
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}, generating...")
        generate_six_hump_camel_dataset(
            n_samples=config.get("num_context_samples", 10_000),
            x1_range=config.get("x1_range", (-3.0, 3.0)),
            x2_range=config.get("x2_range", (-2.0, 2.0)),
            save_path=data_path,
            seed=config.get("random_seed", 42),
        )

    df = pd.read_csv(data_path)

    X_all = df[["x1", "x2"]].values.astype(np.float32)
    y_all = df["y"].values.astype(np.float32)

    rng = np.random.default_rng(config.get("random_seed", 42))
    num_use = min(config.get("num_context_samples", len(y_all)), len(y_all))
    idx = rng.choice(len(y_all), size=num_use, replace=False)

    X_ctx_base = X_all[idx]
    y_ctx_base = y_all[idx]

    print("--- Base context pool from CSV ---")
    print(f"Base context pool size: {X_ctx_base.shape[0]}")
    print(f"  X_ctx_base shape: {X_ctx_base.shape}")
    print(f"  y_ctx_base shape: {y_ctx_base.shape}")
    print("---------------------------\n")

    return X_ctx_base, y_ctx_base


def build_variant_context_pools(
    X_ctx_base: np.ndarray,
    variants: List[Dict[str, float]],
    config: dict,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    对每个变体 i：
      - X_ctx_pools[i] = X_ctx_base（所有变体共用同一批坐标）
      - y_ctx_pools[i] = 该变体在 X_ctx_base 上的函数值
    """
    device = config["device"]
    X_ctx_pools: List[np.ndarray] = []
    y_ctx_pools: List[np.ndarray] = []

    print("--- Build variant context pools ---")
    for i, params in enumerate(variants):
        y_ctx = six_hump_camel_family_numpy_from_params(X_ctx_base, params, device=device)
        X_ctx_pools.append(X_ctx_base.copy())
        y_ctx_pools.append(y_ctx)

        if i < 3:  # 打印前几个变体的前几条样本看看
            print(f"\nVariant {i} context sample (first 5 points):")
            print(
                np.concatenate(
                    [X_ctx_base[:5], y_ctx[:5, None]], axis=1
                )
            )

    print("\nTotal variant context pools:", len(X_ctx_pools))
    print("---------------------------\n")

    return X_ctx_pools, y_ctx_pools


# ============================================================
# 5. 多变体 meta-dataset（路径 B）—— 用于直观查看 task 结构
# ============================================================
class SixHumpCamelFamilyMetaDataset(Dataset):
    """
    每个 __getitem__ 返回一个"源任务"（meta-task）：
      - X_ctx:  (n_ctx, 2)   —— 来自某一个变体的上下文点
      - y_ctx:  (n_ctx,)
      - X_test: (n_test, 2)  —— 公共全局网格
      - y_test: (n_test,)    —— 对应变体在该网格上的函数值
    """

    def __init__(
        self,
        X_ctx_pools: List[np.ndarray],
        y_ctx_pools: List[np.ndarray],
        X_test_grid: np.ndarray,
        y_test_grids: np.ndarray,  # shape (num_variants, n_test)
        num_tasks: int,
        min_ctx: int,
        max_ctx: int,
        random_seed: int = 42,
    ):
        assert len(X_ctx_pools) == len(y_ctx_pools) == y_test_grids.shape[0]
        assert min_ctx >= 1 and max_ctx >= min_ctx

        self.X_ctx_pools = X_ctx_pools
        self.y_ctx_pools = y_ctx_pools
        self.X_test_grid = X_test_grid
        self.y_test_grids = y_test_grids

        self.num_variants = len(X_ctx_pools)
        self.num_tasks = num_tasks
        self.min_ctx = min_ctx
        self.max_ctx = max_ctx

        self.rng = np.random.default_rng(random_seed)

    def __len__(self) -> int:
        return self.num_tasks

    def __getitem__(self, idx: int):
        # 1) 随机选择一个变体
        v = int(self.rng.integers(0, self.num_variants))

        X_pool = self.X_ctx_pools[v]
        y_pool = self.y_ctx_pools[v]
        n_pool = X_pool.shape[0]

        # 2) 随机选择上下文长度 n_ctx ∈ [min_ctx, max_ctx]
        n_ctx = int(self.rng.integers(self.min_ctx, self.max_ctx + 1))

        # 3) 在该变体的上下文池中采样 n_ctx 个点
        indices = self.rng.choice(n_pool, size=n_ctx, replace=False)
        X_ctx = X_pool[indices]
        y_ctx = y_pool[indices]

        # 4) test 使用公共网格 + 该变体在网格上的函数值
        X_test = self.X_test_grid
        y_test = self.y_test_grids[v]

        return (
            X_ctx.astype(np.float32),
            y_ctx.astype(np.float32),
            X_test.astype(np.float32),
            y_test.astype(np.float32),
        )


def demo_family_meta_dataset(
    X_ctx_pools: List[np.ndarray],
    y_ctx_pools: List[np.ndarray],
    X_test_grid: np.ndarray,
    y_test_grids: np.ndarray,
    config: dict,
):
    """
    打印几个 meta-task 看看结构是否符合预期。
    """
    num_tasks_demo = 5
    ds = SixHumpCamelFamilyMetaDataset(
        X_ctx_pools=X_ctx_pools,
        y_ctx_pools=y_ctx_pools,
        X_test_grid=X_test_grid,
        y_test_grids=y_test_grids,
        num_tasks=num_tasks_demo,
        min_ctx=config["min_context"],
        max_ctx=config["max_context"],
        random_seed=config["random_seed"],
    )

    loader = DataLoader(ds, batch_size=1, shuffle=False)

    print("--- Demo SixHumpCamelFamilyMetaDataset tasks ---")
    for batch_idx, batch in enumerate(loader):
        X_ctx, y_ctx, X_test, y_test = batch

        # batch_size = 1，去掉 batch 维度方便打印
        X_ctx = X_ctx[0].numpy()
        y_ctx = y_ctx[0].numpy()
        X_test = X_test[0].numpy()
        y_test = y_test[0].numpy()

        print(f"\nTask {batch_idx}:")
        print(f"  X_ctx shape:  {X_ctx.shape}")
        print(f"  y_ctx shape:  {y_ctx.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_test shape: {y_test.shape}")

        print("  Sample context points (x1, x2, y):")
        print(np.concatenate([X_ctx[:3], y_ctx[:3, None]], axis=1))

        print("  Sample test grid points (x1, x2, y):")
        print(np.concatenate([X_test[:3], y_test[:3, None]], axis=1))

    print("---------------------------\n")


# ============================================================
# 6. TabPFNRegressor 初始化 & 多变体 splitter + dataloader
# ============================================================
def setup_regressor(config: dict) -> Tuple[TabPFNRegressor, dict]:
    print("--- TabPFNRegressor Setup ---")
    regressor_config = {
        "ignore_pretraining_limits": True,
        "device": config["device"],
        "n_estimators": 1,
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


def make_six_hump_camel_family_splitter(
    X_ctx_pools: List[np.ndarray],
    y_ctx_pools: List[np.ndarray],
    X_test_grid: np.ndarray,
    y_test_grids: np.ndarray,
    config: dict,
):
    """
    自定义 splitter，用于 TabPFNRegressor.get_preprocessed_datasets。
    """
    num_variants = len(X_ctx_pools)
    min_c = config["min_context"]
    max_c = config["max_context"]
    rng = np.random.default_rng(config["random_seed"])

    def splitter(X_all: np.ndarray, y_all: np.ndarray):
        # 1) 随机选择变体
        v = int(rng.integers(0, num_variants))

        X_pool = X_ctx_pools[v]
        y_pool = y_ctx_pools[v]
        n_pool = X_pool.shape[0]

        # 2) 随机上下文长度
        ctx_size = int(rng.integers(min_c, max_c + 1))
        indices = rng.choice(n_pool, size=ctx_size, replace=False)

        X_ctx = X_pool[indices]
        y_ctx = y_pool[indices]

        # 3) test 固定为公共网格 + 该变体对应的 y
        X_test = X_test_grid
        y_test = y_test_grids[v]

        return X_ctx, X_test, y_ctx, y_test

    return splitter


def create_finetuning_dataloader_family(
    regressor: TabPFNRegressor,
    X_ctx_base: np.ndarray,
    y_ctx_base: np.ndarray,
    X_ctx_pools: List[np.ndarray],
    y_ctx_pools: List[np.ndarray],
    X_test_grid: np.ndarray,
    y_test_grids: np.ndarray,
    config: dict,
) -> DataLoader:
    print("--- Build finetuning datasets & dataloader (family) ---")

    splitter = make_six_hump_camel_family_splitter(
        X_ctx_pools,
        y_ctx_pools,
        X_test_grid,
        y_test_grids,
        config,
    )

    max_data_size = config["finetuning"]["max_data_size"]

    training_datasets = regressor.get_preprocessed_datasets(
        X_ctx_base,
        y_ctx_base,
        splitter,
        max_data_size=max_data_size,
    )

    print(
        f"Number of meta-datasets from get_preprocessed_datasets: "
        f"{len(training_datasets)}"
    )

    finetuning_dataloader = DataLoader(
        training_datasets,
        batch_size=config["finetuning"]["meta_batch_size"],
        collate_fn=meta_dataset_collator,
    )

    # 看一个 batch 确认形状
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

    print(
        "Inspect one preprocessed task (after collate, meta_batch_size=1):"
    )
    print("  X_trains_preprocessed[0].shape:", X_trains_preprocessed[0].shape)
    print("  X_tests_preprocessed[0].shape :", X_tests_preprocessed[0].shape)
    print("  y_trains_znorm[0].shape       :", y_trains_znorm[0].shape)
    print("  y_test_znorm[0].shape         :", y_test_znorm[0].shape)
    print("---------------------------\n")

    return finetuning_dataloader


# ============================================================
# 7. 在标准 Six-Hump Camel 网格上的简单评估（用于训练过程监控）
# ============================================================
def evaluate_regressor_on_base_six_hump_camel_grid(
    regressor: TabPFNRegressor,
    regressor_config: dict,
    X_ctx_base: np.ndarray,
    y_ctx_base: np.ndarray,
    X_test_grid: np.ndarray,
    y_base_grid: np.ndarray,
    config: dict,
) -> Tuple[float, float, float]:
    """
    在标准 Six-Hump Camel 的全局网格上评估当前 TabPFN。
    """
    eval_regressor = clone_model_for_evaluation(
        regressor,
        {
            **regressor_config,
            "inference_config": {
                "SUBSAMPLE_SAMPLES": config["n_inference_context_samples"],
            },
        },
        TabPFNRegressor,
    )

    rng = np.random.default_rng(config["random_seed"])
    n_ctx_eval = config.get("eval_context_size", config["max_context"])
    n_base = X_ctx_base.shape[0]
    idx = rng.choice(np.arange(n_base), size=n_ctx_eval, replace=False)
    X_ctx_eval = X_ctx_base[idx]
    y_ctx_eval = y_ctx_base[idx]

    eval_regressor.fit(X_ctx_eval, y_ctx_eval)
    preds = eval_regressor.predict(X_test_grid)

    mse = mean_squared_error(y_base_grid, preds)
    mae = mean_absolute_error(y_base_grid, preds)
    r2 = r2_score(y_base_grid, preds)
    return mse, mae, r2


# ============================================================
# 8. 微调主循环（在 Six-Hump Camel 变体族上 finetune TabPFN）
# ============================================================
def run_finetuning_family(
    regressor: TabPFNRegressor,
    regressor_config: dict,
    finetuning_dataloader: DataLoader,
    X_ctx_base: np.ndarray,
    y_ctx_base: np.ndarray,
    X_test_grid: np.ndarray,
    y_base_grid: np.ndarray,
    config: dict,
) -> None:
    # 确保只有一个底层模型可微调
    if hasattr(regressor, "models_"):
        if len(regressor.models_) > 1:
            raise ValueError(
                f"Your TabPFNRegressor uses multiple models ({len(regressor.models_)}). "
                "Finetuning is only supported for a single model."
            )
        model = regressor.models_[0]
    else:
        model = regressor.model_

    optimizer = Adam(model.parameters(), lr=config["finetuning"]["learning_rate"])
    print(
        f"--- Optimizer Initialized: Adam, LR: {config['finetuning']['learning_rate']} ---\n"
    )

    print("--- Start finetuning on Six-Hump Camel family ---")
    num_epochs = config["finetuning"]["epochs"]

    for epoch in range(num_epochs + 1):
        # 1) 每个 epoch 前先在标准 Six-Hump Camel 网格上评估一次
        mse, mae, r2 = evaluate_regressor_on_base_six_hump_camel_grid(
            regressor,
            regressor_config,
            X_ctx_base,
            y_ctx_base,
            X_test_grid,
            y_base_grid,
            config,
        )
        status = "Initial" if epoch == 0 else f"Epoch {epoch}"
        print(
            f"{status} Evaluation on base Six-Hump Camel grid | "
            f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}"
        )

        if epoch == 0:
            print("---------------------------")
            continue

        # 2) 在变体族构成的 meta-tasks 上做一次 epoch 的微调
        progress_bar = tqdm(
            finetuning_dataloader, desc=f"Finetuning Epoch {epoch}"
        )
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

            regressor.raw_space_bardist_ = raw_space_bardist_[0]
            regressor.znorm_space_bardist_ = znorm_space_bardist_[0]

            regressor.fit_from_preprocessed(
                X_trains_preprocessed,
                y_trains_znorm,
                cat_ixs,
                confs,
            )

            logits, _, _ = regressor.forward(X_tests_preprocessed)

            loss_fn = znorm_space_bardist_[0]
            y_target = y_test_znorm

            loss = loss_fn(logits, y_target.to(config["device"])).mean()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        print("---------------------------")

    # 训练结束后，保存整个 TabPFN 模型
    os.makedirs("./model", exist_ok=True)
    save_path = "./model/finetuned_tabpfn_six_hump_camel_family.ckpt"
    save_tabpfn_model(regressor, save_path)
    print(f"Saved fine-tuned TabPFNRegressor to: {save_path}")
    print("--- Finetuning Finished ---")


# ============================================================
# 9. main：串起来步骤 0–5
# ============================================================
def main():
    # 全局配置
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_seed": 42,
        "variant_seed": 123,

        # Six-Hump Camel 域 & 网格（定义域为 [-3, 3] x [-2, 2]）
        "x1_range": (-3.0, 3.0),
        "x2_range": (-2.0, 2.0),
        "grid_size": 20,  # 20×20 网格

        # 从 CSV 读取的基础上下文池大小（标准 Six-Hump Camel）
        "data_path": "./data/six_hump_camel_dataset.csv",
        "num_context_samples": 10_000,

        # 变体相关
        "num_variants": 32,
        "similarity_range": (0.4, 0.6),
        "max_variant_trials": 32 * 300,

        # 变长上下文
        "min_context": 2,
        "max_context": 20,

        # 评估时用多少上下文点
        "eval_context_size": 20,
        "n_inference_context_samples": 20,

        # 微调超参数
        "finetuning": {
            "epochs": 10,
            "learning_rate": 1.5e-6,
            "meta_batch_size": 1,
            "max_data_size": 20,
        },
    }

    # --- 步骤 0：生成数据集（如果不存在） ---
    if not os.path.exists(config["data_path"]):
        print("=" * 50)
        print("Step 0: Generating Six-Hump Camel dataset...")
        print("=" * 50)
        generate_six_hump_camel_dataset(
            n_samples=config["num_context_samples"],
            x1_range=config["x1_range"],
            x2_range=config["x2_range"],
            save_path=config["data_path"],
            seed=config["random_seed"],
        )

    # --- 步骤 1：构造基础网格 & 标准 Six-Hump Camel 网格值 ---
    X_test_grid, X1, X2, y_base_grid = build_base_grid_and_y(config)

    # --- 步骤 2：从 CSV 读取基础上下文点（标准 Six-Hump Camel） ---
    X_ctx_base, y_ctx_base = load_base_context_points(config)

    # --- 步骤 3：在基础网格上选 32 个相似度在 [0.4,0.6] 的变体 ---
    grid_shape = (config["grid_size"], config["grid_size"])
    variants, variant_y_grids, variant_rhos = select_six_hump_camel_variants(
        config,
        X_test_grid,
        y_base_grid,
        grid_shape,
    )

    os.makedirs("./data", exist_ok=True)
    np.savez(
        "./data/six_hump_camel_family_variants.npz",
        variants=np.array(variants, dtype=object),
        variant_y_grids=variant_y_grids,
        variant_rhos=np.array(variant_rhos, dtype=np.float32),
    )
    print("Saved Six-Hump Camel family variants to ./data/six_hump_camel_family_variants.npz")

    # --- 步骤 4：为每个变体构造上下文池 ---
    X_ctx_pools, y_ctx_pools = build_variant_context_pools(
        X_ctx_base,
        variants,
        config,
    )

    # --- 步骤 5（路径 B）：构造 SixHumpCamelFamilyMetaDataset，打印几个 task 看看 ---
    demo_family_meta_dataset(
        X_ctx_pools,
        y_ctx_pools,
        X_test_grid,
        variant_y_grids,
        config,
    )

    # --- 步骤 6：TabPFN 初始化 + finetuning dataloader + 微调 ---
    regressor, regressor_config = setup_regressor(config)

    finetuning_dataloader = create_finetuning_dataloader_family(
        regressor,
        X_ctx_base,
        y_ctx_base,
        X_ctx_pools,
        y_ctx_pools,
        X_test_grid,
        variant_y_grids,
        config,
    )

    run_finetuning_family(
        regressor,
        regressor_config,
        finetuning_dataloader,
        X_ctx_base,
        y_ctx_base,
        X_test_grid,
        y_base_grid,
        config,
    )


if __name__ == "__main__":
    main()
