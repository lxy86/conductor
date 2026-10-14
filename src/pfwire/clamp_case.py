"""6.5 基准工况的装配、W' 加载搜索与起裂/转向度量（D-67 §3–§5）。

本模块的两个核心都做成**纯函数**，好让单元测试不必付有限元网格的代价：

- :func:`secant_delta`：割线法求 W(δ) = W_target。注入解析 W(δ) 就能验证算法本身
  （线性、非线性、不收敛三类），真实 FEM 的 W(δ) 由 :func:`solve_contact_at` 提供。
- :func:`crack_metrics`：在合成 d 场上验证（竖直、倾斜转向、表面连通性过滤）。

其余（:class:`ClampCase`、:func:`build_clamp_case`、:func:`run_clamp_case`）是生产侧
的装配与推进，验收判据按 D-67 §5：**任一条不满足就停下报告，不调参**（硬约束 6）。

单位：SI。深度向下为正（与 postproc 的快照约定一致）。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "secant_delta",
    "CrackMetrics",
    "crack_metrics",
    "ClampCase",
    "build_clamp_case",
    "solve_contact_at",
]


# --------------------------------------------------------------------------
# T4：W' 的 δ 搜索（割线法，纯函数）
# --------------------------------------------------------------------------
def secant_delta(W_of_delta: Callable[[float], float], W_target: float, *,
                 delta0: float = 17.0e-6, delta1: float | None = None,
                 rtol: float = 1.0e-2, max_iter: int = 20,
                 delta_max: float | None = None) -> float:
    """割线法求 W(δ) = W_target 的 δ，返回达到容差的 δ。

    D-67 §3：δ₀ = 17 µm 取自 6.1 长加载结果（P ≈ W'）；相对容差 1%，最多 20 次；
    **不收敛即抛异常**（硬约束 6：不得把未收敛的结果当结果用）。

    ``delta1`` 为 None 时取 delta0·(1 + 1e-3) 作第二点——1e-3 的相对扰动足够让割线
    有斜率，又不至于一步跨进另一个接触状态（W(δ) 在 δ 增大时单调增，见 Hertz）。

    ``delta_max`` 是外推的**物理上界**（None 时取 δ₀ 的 100 倍，线夹下压量不可能
    到那个量级）。必须给：W(δ) 有上界而目标够不到时，割线法会一路外推到荒谬大的
    δ（实测推到 km 量级），那时 W(δ) 已数值饱和成常数，报错会退化成「W 不单调」
    ——把「目标不可达」错报成「W 坏了」，排查方向就错了。越界立即以
    「未收敛（外推出界）」退出，错误信息对准真因。

    W(δ) 由调用方给出：生产里是 :func:`solve_contact_at`（每次一个 FE 解），
    测试里可以是解析函数，这样算法本身能被独立验证。
    """
    W_target = float(W_target)
    if not math.isfinite(W_target):
        raise ValueError("W_target 必须为有限值，收到 %r" % (W_target,))
    if rtol <= 0.0:
        raise ValueError("rtol 必须为正，收到 %r" % (rtol,))
    if max_iter < 1:
        raise ValueError("max_iter 必须 >= 1，收到 %r" % (max_iter,))
    d0 = float(delta0)
    d1 = float(delta1) if delta1 is not None else d0 * (1.0 + 1.0e-3)
    d_max = float(delta_max) if delta_max is not None else 100.0 * abs(d0)
    f0 = float(W_of_delta(d0)) - W_target
    if abs(f0) <= rtol * abs(W_target):
        return d0
    f1 = float(W_of_delta(d1)) - W_target
    if abs(f1) <= rtol * abs(W_target):
        return d1
    for _ in range(int(max_iter)):
        if f1 == f0:
            raise RuntimeError(
                "割线法退化：两个试探点的 W 相同（W(δ0)=W(δ1)=%.6g），"
                "W(δ) 在 [%.4g, %.4g] 上不是单调的" % (f0 + W_target, d0, d1))
        d2 = d1 - f1 * (d1 - d0) / (f1 - f0)
        if not math.isfinite(d2) or abs(d2) > d_max:
            raise RuntimeError(
                "割线法未收敛（外推出界）：下一步 δ = %.6g 已越过物理上界 %.6g"
                "（末次 W = %.6g，目标 %.6g）。这通常意味着 W(δ) 达不到目标："
                "检查 W' 是否超过了线夹在该 δ 范围内能给出的最大反力。"
                % (d2, d_max, f1 + W_target, W_target))
        f2 = float(W_of_delta(d2)) - W_target
        if abs(f2) <= rtol * abs(W_target):
            return d2
        d0, f0, d1, f1 = d1, f1, d2, f2
    raise RuntimeError(
        "割线法 %d 次未收敛到 |W − W'| <= %.1f%%·W'（末次 δ = %.6g，W = %.6g，"
        "目标 %.6g）。检查 W(δ) 是否单调、δ₀ 是否离工作点太远。"
        % (max_iter, 100 * rtol, d1, f1 + W_target, W_target))


# --------------------------------------------------------------------------
# T5：起裂与转向度量（纯函数）
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CrackMetrics:
    """起裂位置与裂纹方向（D-67 §4）。角度单位为度，相对表面法向、带符号。"""

    x_init: float
    """起裂点 x（m）：表面连通的 d ≥ threshold 区域在表面上的位置。"""
    centroid_6: tuple[float, float]
    """nucleation_depth 深度切片上连通区域的质心 (x, depth)（m）。"""
    centroid_100: tuple[float, float]
    """turn_depth 深度切片上连通区域的质心 (x, depth)（m）。"""
    theta_init_deg: float
    """起裂点 → 6 µm 质心 的连线与表面法向的夹角（度，带符号）。"""
    theta_later_deg: float
    """6 µm → 100 µm 质心 的连线与表面法向的夹角（度，带符号）。"""
    reached_nuc: bool
    """d ≥ threshold 是否达到 nucleation_depth（未达到则不算起裂）。"""
    dmax: float
    """诊断：全区最大损伤。"""
    depth_max: float
    """诊断：表面连通裂纹的最大深度（m）。"""


def _surface_connected_mask(mask: np.ndarray) -> np.ndarray:
    """从二值 mask 中保留**与表面（第 0 行）连通**的部分（4-连通 flood fill）。

    D-67 §4 要求「表面连通」：内部孤立的损伤斑块不算裂纹（那是假损伤，非物理）。
    用逐行扫描的连通标记，避免 import scipy。
    """
    n_depth, n_x = mask.shape
    labels = np.zeros_like(mask, dtype=np.int32)
    if n_depth == 0 or n_x == 0:
        return np.zeros_like(mask, dtype=bool)
    # 逐行并查集式的标记：同一行内相邻 -> 合并；与上一行同列相邻 -> 继承
    next_label = 1
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n_depth):
        prev = -1
        for j in range(n_x):
            if not mask[i, j]:
                prev = -1
                continue
            if prev >= 0:
                labels[i, j] = labels[i, prev]
            elif i > 0 and mask[i - 1, j] and labels[i - 1, j]:
                labels[i, j] = labels[i - 1, j]
            else:
                labels[i, j] = next_label
                parent[next_label] = next_label
                next_label += 1
            if prev >= 0 and labels[i, prev] != labels[i, j]:
                union(labels[i, prev], labels[i, j])
            # 与上一行的不同标签合并（上一行同列已被继承，这里处理左右邻居的跨行）
            if i > 0 and mask[i - 1, j] and labels[i - 1, j] and labels[i - 1, j] != labels[i, j]:
                union(labels[i - 1, j], labels[i, j])
            prev = j
    # 归一到根标签
    roots = np.vectorize(lambda v: find(int(v)) if v else 0)(labels)
    surface_labels = set(int(v) for v in roots[0, :] if v)
    if not surface_labels:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(roots, list(surface_labels))


def _centroid_at_depth(mask_conn: np.ndarray, x_coords: np.ndarray,
                       depth_coords: np.ndarray, depth_target: float
                       ) -> tuple[float, float] | None:
    """在给定深度的切片上取连通区域的质心 (x, depth)。

    切片取「最接近 depth_target 的那一行」；该行若无连通点则向前（更深）找
    最近的有连通点的行——裂纹在两个判定深度之间是连续的，这样取不会跳到别的分支。
    """
    n_depth = len(depth_coords)
    order = np.argsort(np.abs(depth_coords - depth_target))
    for i in order:
        row = mask_conn[i, :]
        if row.any():
            return float(np.mean(x_coords[row])), float(depth_coords[i])
    return None


def _angle_from_normal(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    """两点连线与表面法向（深度方向）的夹角，度，带符号。

    θ = atan2(Δx, Δdepth)：正表示裂纹向 +x 倾斜。法向即深度增加的方向，
    故 Δdepth > 0 时 atan2 的分母为正，角度在 (−90°, 90°) 内。
    """
    dx = p1[0] - p0[0]
    ddepth = p1[1] - p0[1]
    return math.degrees(math.atan2(dx, ddepth))


def crack_metrics(d_field: np.ndarray, x_coords: np.ndarray, depth_coords: np.ndarray, *,
                  threshold: float = 0.95,
                  nucleation_depth: float = 6.0e-6,
                  turn_depth: float = 100.0e-6) -> CrackMetrics | None:
    """由 d 场给出起裂位置与裂纹方向（D-67 §4）。

    ``d_field`` 为 (n_depth, n_x)，``x_coords`` (n_x,)、``depth_coords`` (n_depth,)，
    深度向下为正、表面为 depth = 0（与 postproc 的快照约定一致）。

    返回 None 表示「d ≥ threshold 未达到 nucleation_depth」——此时还没有起裂，
    θ 无从定义，调用方（run_clamp_case）据此继续下一个循环步而不记录 N_nuc。

    步骤：二值化 → 取与表面连通的部分 → 在该连通域内在 nucleation/turn 深度切片
    取质心 → 用两点连线算 θ_init / θ_later（D-67 §4：两点连线对网格扰动稳定，
    不采用等值线切向）。
    """
    d = np.asarray(d_field, dtype=float)
    xs = np.asarray(x_coords, dtype=float)
    depths = np.asarray(depth_coords, dtype=float)
    if d.ndim != 2 or d.shape != (len(depths), len(xs)):
        raise ValueError(
            "d_field 形状应为 (n_depth, n_x) = (%d, %d)，收到 %r"
            % (len(depths), len(xs), d.shape))
    mask = d >= float(threshold)
    dmax = float(d.max()) if d.size else 0.0
    if not mask.any():
        return CrackMetrics(x_init=float("nan"), centroid_6=(float("nan"),) * 2,
                            centroid_100=(float("nan"),) * 2, theta_init_deg=float("nan"),
                            theta_later_deg=float("nan"), reached_nuc=False,
                            dmax=dmax, depth_max=0.0)

    conn = _surface_connected_mask(mask)
    # 表面连通域在表面上的 x 范围的中点即起裂点（D-67 §4 的 x_init）
    surf = conn[0, :]
    if not surf.any():
        # 有损伤但没有一个与表面连通：孤立斑块，不算裂纹
        return CrackMetrics(x_init=float("nan"), centroid_6=(float("nan"),) * 2,
                            centroid_100=(float("nan"),) * 2, theta_init_deg=float("nan"),
                            theta_later_deg=float("nan"), reached_nuc=False,
                            dmax=dmax, depth_max=0.0)
    x_init = 0.5 * (xs[surf].min() + xs[surf].max())
    depth_max = float(depths[np.nonzero(conn.any(axis=1))[0]].max()) if conn.any() else 0.0
    reached = depth_max >= float(nucleation_depth)

    c6 = _centroid_at_depth(conn, xs, depths, nucleation_depth)
    c100 = _centroid_at_depth(conn, xs, depths, turn_depth)
    if c6 is None or c100 is None:
        return CrackMetrics(x_init=x_init, centroid_6=c6 or (float("nan"),) * 2,
                            centroid_100=c100 or (float("nan"),) * 2,
                            theta_init_deg=float("nan"), theta_later_deg=float("nan"),
                            reached_nuc=reached, dmax=dmax, depth_max=depth_max)
    theta_init = _angle_from_normal((x_init, 0.0), c6)
    theta_later = _angle_from_normal(c6, c100)
    return CrackMetrics(x_init=x_init, centroid_6=c6, centroid_100=c100,
                        theta_init_deg=theta_init, theta_later_deg=theta_later,
                        reached_nuc=reached, dmax=dmax, depth_max=depth_max)


# --------------------------------------------------------------------------
# 生产侧：装配与推进（T6/T7 用）
# --------------------------------------------------------------------------
@dataclass
class ClampCase:
    """基准工况的全部部件。"""
    mesh: object
    facet_tags: object
    clamp: object                  # ClampContact
    loads: object                  # ClampLoads
    solver: object                 # PhaseFieldSolver（plastic + contact）
    config: dict = field(default_factory=dict)

    def set_delta(self, delta: float) -> None:
        self.clamp.set_delta(delta)


def build_clamp_case(config) -> ClampCase:
    """由配置装配基准工况。

    **尚未实现**（T6 前完成）：要读 configs/clamp_base.yaml 建 mesh（布局 C）、
    ClampContact、ClampLoads、PhaseFieldSolver（plastic + contact）。
    """
    raise NotImplementedError("build_clamp_case 在 T6 实现")


def solve_contact_at(case: ClampCase, delta: float) -> float:
    """把线夹下压到 δ 并解一次平衡，返回接触反力 W(δ)（N/m，单位厚度）。

    给 :func:`secant_delta` 当 ``W_of_delta`` 用（T4 的 FEM 侧）。
    """
    case.set_delta(delta)
    case.solver.solve_displacement()
    return case.clamp.reaction(case.solver.u)
