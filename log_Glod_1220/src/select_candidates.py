"""
Train RL: 使用 TabPFN 做贝叶斯优化的候选点选择
- 支持任意目标函数（通过 ObjectiveFunction 基类）
- 随机采样上下文点
- Sobol 采样候选点
- 计算 EI，选择 top-k EI / top-k variance / 随机 候选点
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Callable, Optional, List, Tuple, Dict, Any
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from tabpfn import TabPFNRegressor


# ==================== 配置数据类 ====================
# @dataclass#自动生成 __init__ 等方法
class SelectionConfig:
    """候选点选择配置"""
    
    def __init__(
        self,
        n_candidates: int = 512,          # Sobol 候选池大小
        top_k_ei: int = 20,               # top-k EI 候选点
        top_k_var: int = 10,              # top-k variance 候选点
        n_random: int = 10,               # 随机候选点
        xi: float = 0.01,                 # 探索-利用权衡参数
        random_seed: int = 42,
        device: str = None,
        verbose: bool = True,
    ):
        self.n_candidates = n_candidates
        self.top_k_ei = top_k_ei
        self.top_k_var = top_k_var
        self.n_random = n_random
        self.xi = xi
        self.random_seed = random_seed
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose


# ==================== 目标函数基类 ====================
class ObjectiveFunction(ABC):#ABC = Abstract Base Class（抽象基类）
    """目标函数抽象基类"""
    
    @property#将方法变成"属性"，调用时不需要加括号
    @abstractmethod#标记这个方法必须被子类实现，否则子类不能实例化
    def dim(self) -> int:
        """返回输入维度"""
        pass
    
    @property
    @abstractmethod
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回定义域边界 (lower, upper)，每个都是 (dim,) 的数组"""
        pass
    
    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        计算目标函数值
        
        Args:
            X: 输入点 (N, dim)
            
        Returns:
            y: 函数值 (N,)
        """
        pass
    
    @property
    def name(self) -> str:
        """函数名称（可选覆盖）"""
        return self.__class__.__name__
    
    @property
    def optimal_value(self) -> Optional[float]:
        """全局最优值（可选覆盖，用于参考）"""
        return None


# ==================== Branin 函数实现 ====================
class BraninFunction(ObjectiveFunction):
    """
    Branin 测试函数
    定义域: x1 ∈ [-5, 10], x2 ∈ [0, 15]
    全局最小值: f(x*) ≈ 0.397887
    """
    
    @property
    def dim(self) -> int:
        return 2
    
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.array([-5.0, 0.0], dtype=np.float32)
        upper = np.array([10.0, 15.0], dtype=np.float32)
        return lower, upper
    
    @property
    def optimal_value(self) -> float:
        return 0.397887
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        x1, x2 = X[:, 0], X[:, 1]
        
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


# ==================== 更多测试函数示例 ====================
class SphereFunction(ObjectiveFunction):
    """
    Sphere 函数: f(x) = sum(x_i^2)
    全局最小值: f(0, ..., 0) = 0
    """
    
    def __init__(self, dim: int = 2, bound: float = 5.0):
        self._dim = dim
        self._bound = bound
    
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.full(self._dim, -self._bound, dtype=np.float32)
        upper = np.full(self._dim, self._bound, dtype=np.float32)
        return lower, upper
    
    @property
    def optimal_value(self) -> float:
        return 0.0
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.sum(X ** 2, axis=1).astype(np.float32)


class RastriginFunction(ObjectiveFunction):
    """
    Rastrigin 函数
    全局最小值: f(0, ..., 0) = 0
    """
    
    def __init__(self, dim: int = 2, bound: float = 5.12):
        self._dim = dim
        self._bound = bound
    
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.full(self._dim, -self._bound, dtype=np.float32)
        upper = np.full(self._dim, self._bound, dtype=np.float32)
        return lower, upper
    
    @property
    def optimal_value(self) -> float:
        return 0.0
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        A = 10
        n = X.shape[1]
        return (A * n + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)).astype(np.float32)


# ==================== EI 计算 ====================
def compute_ei(mean: np.ndarray, std: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    """
    计算 Expected Improvement (EI)
    
    Args:
        mean: 预测均值 (N,)
        std: 预测标准差 (N,)
        y_best: 当前最优值（最小化问题取最小 y）
        xi: 探索-利用权衡参数
    
    Returns:
        ei: Expected Improvement (N,)
    """
    # 对于最小化问题
    imp = y_best - mean - xi
    Z = np.zeros_like(mean)
    
    # 避免除以 0
    mask = std > 1e-8
    Z[mask] = imp[mask] / std[mask]
    
    ei = np.zeros_like(mean)
    ei[mask] = imp[mask] * norm.cdf(Z[mask]) + std[mask] * norm.pdf(Z[mask])
    ei[ei < 0] = 0.0
    
    return ei


# ==================== 从 TabPFN 输出计算方差 ====================
def compute_std_from_tabpfn_output(full_out: Dict[str, Any]) -> np.ndarray:
    """
    从 TabPFN 的 full output 中计算标准差
    
    Args:
        full_out: regressor.predict(X, output_type="full") 的输出
        
    Returns:
        std: 标准差 (N,) numpy array
    """
    criterion = full_out.get("criterion", None)
    logits = full_out.get("logits", None)
    
    if criterion is None or logits is None:
        raise ValueError("full_out 必须包含 'criterion' 和 'logits'")
    
    borders = criterion.borders
    bin_centers = (borders[:-1] + borders[1:]) / 2
    
    # softmax 转换为概率
    probs = torch.softmax(logits, dim=-1)  # shape: (N, n_bins)
    
    # 均值: E[X] = Σ p_i * x_i
    mean = (probs * bin_centers).sum(dim=-1)
    
    # 二阶矩: E[X²] = Σ p_i * x_i²
    mean_sq = (probs * bin_centers ** 2).sum(dim=-1)
    
    # 方差: Var[X] = E[X²] - E[X]²
    variance = mean_sq - mean ** 2
    std = torch.sqrt(variance.clamp(min=1e-8))
    
    return std.cpu().numpy()


# ==================== 主函数：候选点选择 ====================
def select_candidates(
    func: ObjectiveFunction,
    model_path: Optional[str] = None,
    config: Optional[SelectionConfig] = None,
    X_context: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    使用 TabPFN 进行贝叶斯优化风格的候选点选择
    
    Args:
        func: 目标函数对象（继承自 ObjectiveFunction）
        model_path: TabPFN 模型路径，None 则使用预训练模型
        config: 选择配置，None 则使用默认配置
        X_context: 上下文点 (n_context, dim)，None 则随机采样 2 个点
        
    Returns:
        result: 包含所有选择结果的字典
    """
    if config is None:
        config = SelectionConfig()
    
    rng = np.random.default_rng(config.random_seed)
    lower, upper = func.bounds
    
    def log(msg: str):
        if config.verbose:
            print(msg)
    
    # ========== 1. 准备上下文点 ==========
    log("=" * 60)
    log(f"目标函数: {func.name} (dim={func.dim})")
    if func.optimal_value is not None:
        log(f"全局最优值: {func.optimal_value:.6f}")
    log("=" * 60)
    log("\nStep 1: 准备上下文点")
    
    if X_context is None:
        # 默认随机采样 2 个点作为上下文
        X_context = rng.uniform(
            lower, upper, 
            size=(2, func.dim)
        ).astype(np.float32)
    else:
        X_context = np.atleast_2d(X_context).astype(np.float32)
    
    # 根据传入的点计算函数值作为上下文
    y_context = func(X_context)
    
    log(f"上下文点数量: {len(X_context)}")
    log(f"上下文点 y 范围: [{y_context.min():.4f}, {y_context.max():.4f}]")
    log(f"当前最优值 y_best: {y_context.min():.4f}")
    
    # ========== 2. Sobol 采样候选点 ==========
    log("\nStep 2: Sobol 采样候选池")
    
    sobol_sampler = Sobol(d=func.dim, scramble=True, seed=config.random_seed)
    sobol_samples = sobol_sampler.random(config.n_candidates)
    X_candidates = (sobol_samples * (upper - lower) + lower).astype(np.float32)
    
    log(f"候选池大小: {len(X_candidates)}")
    
    # ========== 3. 使用 TabPFN 预测 ==========
    log("\nStep 3: TabPFN 预测")
    
    regressor_kwargs = {
        "device": config.device,
        "n_estimators": 1,
        "random_state": config.random_seed,
        "inference_precision": torch.float32,
        "ignore_pretraining_limits": True,
    }
    if model_path is not None:
        regressor_kwargs["model_path"] = model_path
    
    regressor = TabPFNRegressor(**regressor_kwargs)
    
    # fit 上下文点
    regressor.fit(X_context, y_context)
    
    # 预测候选点
    full_out = regressor.predict(X_candidates, output_type="full")
    y_pred_mean = full_out.get("mean", None)
    y_pred_std = compute_std_from_tabpfn_output(full_out)
    
    log(f"预测均值范围: [{y_pred_mean.min():.4f}, {y_pred_mean.max():.4f}]")
    log(f"预测标准差范围: [{y_pred_std.min():.4f}, {y_pred_std.max():.4f}]")
    
    # ========== 4. 计算 EI ==========
    log("\nStep 4: 计算 EI")
    
    y_best = y_context.min()
    ei_values = compute_ei(y_pred_mean, y_pred_std, y_best, xi=config.xi)
    
    log(f"EI 范围: [{ei_values.min():.6f}, {ei_values.max():.6f}]")
    
    # ========== 5. 选择候选点 ==========
    log("\nStep 5: 选择候选点")
    
    # Top-k EI
    top_k_ei_indices = np.argsort(ei_values)[-config.top_k_ei:][::-1]
    X_top_ei = X_candidates[top_k_ei_indices]
    
    log(f"  Top-{config.top_k_ei} EI: EI=[{ei_values[top_k_ei_indices].min():.6f}, {ei_values[top_k_ei_indices].max():.6f}]")
    
    # Top-k Variance（排除已选的）
    remaining_mask = np.ones(len(X_candidates), dtype=bool)
    remaining_mask[top_k_ei_indices] = False
    remaining_indices = np.where(remaining_mask)[0]
    
    var_values = y_pred_std ** 2
    var_remaining = var_values[remaining_indices]
    top_k_var_local_indices = np.argsort(var_remaining)[-config.top_k_var:][::-1]
    top_k_var_indices = remaining_indices[top_k_var_local_indices]
    X_top_var = X_candidates[top_k_var_indices]
    
    log(f"  Top-{config.top_k_var} Var: Var=[{var_values[top_k_var_indices].min():.6f}, {var_values[top_k_var_indices].max():.6f}]")
    
    # 随机（排除已选的）
    remaining_mask[top_k_var_indices] = False
    remaining_indices = np.where(remaining_mask)[0]
    random_indices = rng.choice(remaining_indices, size=config.n_random, replace=False)
    X_random = X_candidates[random_indices]
    
    log(f"  Random: {config.n_random} 点")
    
    # ========== 6. 汇总结果 ==========
    log("\nStep 6: 汇总结果")
    
    X_selected = np.vstack([X_top_ei, X_top_var, X_random])
    y_selected_true = func(X_selected)
    
    # 选中点的预测均值和方差
    selected_indices = np.concatenate([top_k_ei_indices, top_k_var_indices, random_indices])
    y_selected_pred_mean = y_pred_mean[selected_indices]
    y_selected_pred_std = y_pred_std[selected_indices]
    y_selected_pred_var = y_selected_pred_std ** 2
    
    log(f"总选中: {len(X_selected)} (EI:{len(X_top_ei)} + Var:{len(X_top_var)} + Rand:{len(X_random)})")
    log(f"选中点真实值范围: [{y_selected_true.min():.4f}, {y_selected_true.max():.4f}]")
    log(f"选中点预测均值范围: [{y_selected_pred_mean.min():.4f}, {y_selected_pred_mean.max():.4f}]")
    log(f"选中点预测方差范围: [{y_selected_pred_var.min():.6f}, {y_selected_pred_var.max():.6f}]")
    log(f"选中点最优真实值: {y_selected_true.min():.4f}")
    
    return {
        "X_context": X_context,
        "y_context": y_context,
        "X_candidates": X_candidates,
        "y_pred_mean": y_pred_mean,
        "y_pred_std": y_pred_std,
        "ei_values": ei_values,
        "X_top_ei": X_top_ei,
        "X_top_var": X_top_var,
        "X_random": X_random,
        "X_selected": X_selected,
        "y_selected_true": y_selected_true,
        "y_selected_pred_mean": y_selected_pred_mean,      # 选中点的预测均值
        "y_selected_pred_std": y_selected_pred_std,        # 选中点的预测标准差
        "y_selected_pred_var": y_selected_pred_var,        # 选中点的预测方差
        "selected_indices": selected_indices,              # 选中点在候选池中的索引
        "top_k_ei_indices": top_k_ei_indices,
        "top_k_var_indices": top_k_var_indices,
        "random_indices": random_indices,
        "config": config,
        "func": func,
    }


# ==================== 示例 ====================
def main():
    """示例：使用不同的目标函数"""
    
    # 方式1：使用 Branin 函数
    print("\n" + "=" * 60)
    print("示例 1: Branin 函数")
    print("=" * 60)
    
    branin_func = BraninFunction()
    
    # 手动指定上下文点（10个随机点）
    rng = np.random.default_rng(42)
    lower, upper = branin_func.bounds
    X_context = rng.uniform(lower, upper, size=(10, branin_func.dim)).astype(np.float32)
    
    result = select_candidates(
        func=branin_func,
        model_path="./model/finetuned_tabpfn_branin_family.ckpt",
        config=SelectionConfig(
            n_candidates=512,  # 必须是 2 的幂次方，Sobol 序列要求
            top_k_ei=20,
            top_k_var=10,
            n_random=10,
        ),
        X_context=X_context,  # 传入上下文点
    )
    
    # 方式2：使用 Sphere 函数（无需微调模型）
    # print("\n" + "=" * 60)
    # print("示例 2: Sphere 函数 (3D)")
    # print("=" * 60)
    # 
    # sphere_func = SphereFunction(dim=3, bound=5.0)
    # result2 = select_candidates(
    #     func=sphere_func,
    #     model_path=None,  # 使用预训练模型
    #     config=SelectionConfig(n_context=15),
    # )
    
    return result


if __name__ == "__main__":
    result = main()
