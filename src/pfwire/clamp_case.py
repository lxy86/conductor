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
    clamp: object = None            # ClampContact
    loads: object = None            # ClampLoads
    solver: object = None           # PhaseFieldSolver（plastic + contact）
    state: object = None            # FatigueState
    driver: object = None           # CycleJumpDriver
    grid: object = None             # DamageGrid
    config: dict = field(default_factory=dict)
    delta: float = float("nan")     # 工作点 δ（W' 搜索的结果）

    def set_delta(self, delta: float) -> None:
        self.delta = float(delta)
        self.clamp.set_delta(delta)


# --------------------------------------------------------------------------
# T6：判据（纯函数，可单测）+ 装配 + 推进
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AcceptanceReport:
    """D-67 §5 的验收判据结果。``passed`` 为 False 时**停下报告，不调参**。"""
    passed: bool
    N_nuc: float | None
    """起裂寿命（d ≥ threshold 首次达到 nucleation_depth 的 N）。"""
    x_init: float
    theta_init_deg: float
    theta_later_deg: float
    contact_edges: tuple[float, float]
    """接触边缘的 x（±a_contact），起裂位置要与它比。"""
    a_pl: float
    checks: dict
    """逐条判据 -> bool。"""
    details: str


def judge_acceptance(metrics: "CrackMetrics | None", *, N_nuc, a_contact: float,
                     a_pl: float, incline_deg: float = 10.0,
                     turn_deg: float = 20.0) -> AcceptanceReport:
    """D-67 §5 的验收判据（纯函数，不依赖 FEM）。

    判据：
      1. 起裂位置落在接触边缘 ±a_pl 内（a_pl 由 6.1 实测给出，≈ 2.389 mm）；
      2. |θ_init| ≥ incline_deg（10°）——裂纹是倾斜的，不是沿法向直裂；
      3. |θ_later − θ_init| ≥ turn_deg（20°）——之后发生转向。

    ``metrics`` 为 None（还没起裂）时三条都不成立，passed=False。
    """
    edges = (-abs(a_contact), abs(a_contact))
    if metrics is None or not metrics.reached_nuc:
        return AcceptanceReport(
            passed=False, N_nuc=None, x_init=float("nan"), theta_init_deg=float("nan"),
            theta_later_deg=float("nan"), contact_edges=edges, a_pl=a_pl,
            checks={"nucleated": False, "x_init_in_window": False,
                    "inclined": False, "turned": False},
            details="未起裂：d ≥ threshold 未达到 nucleation_depth")
    # 起裂位置与**两个**接触边缘比（裂纹可出现在任一侧）
    near_edge = min(abs(metrics.x_init - e) for e in edges)
    checks = {
        "nucleated": True,
        "x_init_in_window": bool(near_edge <= a_pl),
        "inclined": bool(abs(metrics.theta_init_deg) >= incline_deg),
        "turned": bool(abs(metrics.theta_later_deg - metrics.theta_init_deg) >= turn_deg),
    }
    passed = all(checks.values())
    return AcceptanceReport(
        passed=passed, N_nuc=float(N_nuc) if N_nuc is not None else None,
        x_init=metrics.x_init, theta_init_deg=metrics.theta_init_deg,
        theta_later_deg=metrics.theta_later_deg, contact_edges=edges, a_pl=a_pl,
        checks=checks,
        details=("起裂 N=%.6g，x_init=%.4g mm（距最近接触边缘 %.4g mm，窗口 ±%.4g mm）；"
                 "θ_init=%.2f°，θ_later=%.2f°，转向 %.2f°"
                 % (float(N_nuc) if N_nuc is not None else float("nan"),
                    1e3 * metrics.x_init, 1e3 * near_edge, 1e3 * a_pl,
                    metrics.theta_init_deg, metrics.theta_later_deg,
                    metrics.theta_later_deg - metrics.theta_init_deg)))


def build_clamp_case(config_path, comm=None, *, resume: bool = False) -> ClampCase:
    """由配置装配基准工况（网格、接触、载荷、求解器、cycle-jump、采样网格）。

    ``resume=True`` 时从 ``output.dir`` 的断点恢复（硬约束 5）：断点里存的是
    **网格本身**而非重新生成——adios4dolfinx 按 input 编号存值，重建的网格
    顺序不同会静默读出乱序的值（见 pfwire/checkpoint.py）。
    """
    import pathlib

    import yaml
    from dolfinx import fem
    from mpi4py import MPI

    from pfwire import checkpoint
    from pfwire.clamp_contact import ClampContact
    from pfwire.clamp_loads import ClampLoads
    from pfwire.cyclejump import CycleJumpConfig, CycleJumpDriver, FatigueState, ShakedownConfig
    from pfwire.failure import DamageGrid, GridSpec
    from pfwire.material import load_material
    from pfwire.mesh import block_mesh
    from pfwire.phasefield import PhaseFieldModel, PhaseFieldSolver, SolverConfig
    from pfwire.plasticity import PlasticLaw, load_j2_params

    comm = comm or MPI.COMM_WORLD
    path = pathlib.Path(config_path)
    # 配置里的 material / mesh.from_config 都是**仓库相对**路径，故 root 取本文件的
    # 上两级（src/pfwire -> 仓库根），而不是配置自身的父目录——测试把配置写到
    # tmp_path 下时后者会算出越界的路径。
    root = pathlib.Path(__file__).resolve().parents[2]
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    material = load_material(root / raw["material"])

    # --- 网格：布局 C 的参数从 clamp_mesh.yaml 读（不在这里重复写死）---
    mraw = yaml.safe_load((root / raw["mesh"]["from_config"]).read_text(encoding="utf-8"))
    geo, el = mraw["geometry"], mraw["element"]
    lay = mraw["layouts"][raw["mesh"]["layout"]]
    a_contact = float(mraw["inputs"]["a_contact"])
    H = float(geo["depth"])
    w, d = float(lay["crack_band_halfwidth"]), float(lay["crack_band_depth"])
    boxes = [(a_contact - w, a_contact + w, -d, 0.0),
             (-a_contact - w, -a_contact + w, -d, 0.0)]
    workdir = root / raw["mesh"]["workdir"]

    resumed = None
    if resume:
        outdir = root / raw["output"]["dir"]
        resumed = checkpoint.resume(outdir, comm)
        msh, facet_tags = resumed.mesh, resumed.read_facet_tags()
    else:
        # ``mesh.tiny`` 只给单测用：把布局 C 的两条 0.5 µm 加密带换成中心一个小盒，
        # 单元数从 145 万降到几千，好让装配/搜索/推进在秒级走通。
        if raw["mesh"].get("tiny"):
            boxes = [(-0.5e-3, 0.5e-3, -0.3e-3, 0.0)]
            h_fine, h_coarse, transition = 50.0e-6, 750.0e-6, 0.3e-3
        else:
            # ``mesh.edge_target`` 可覆盖 clamp_mesh.yaml 的单元边长（降规模用）。
            # gmsh 的设定值 h 与实际边长之比是 edge_ratio（实测 0.45，D-58）。
            edge = float(raw["mesh"].get("edge_target", el["edge_target"]))
            h_fine = edge / float(el["edge_ratio"])
            h_coarse, transition = float(el["h_coarse"]), float(el["transition"])
            # ``mesh.refine_boxes`` 可覆盖布局 C 的两条裂纹带（6.1 尺度下用一整个
            # 覆盖接触区的盒子即可，单元数从 145 万降到 7 千）。
            if raw["mesh"].get("refine_boxes"):
                boxes = [tuple(float(v) for v in b)
                         for b in raw["mesh"]["refine_boxes"]]
        msh, _ct, facet_tags = block_mesh(
            workdir, width=float(geo["width"]), depth=H, refine_boxes=boxes,
            h_fine=h_fine, h_coarse=h_coarse, transition=transition, comm=comm)
    h_fine_eff = h_fine

    # --- 接触与载荷 ---
    cc = raw["contact"]
    # h_contact 是**接触面的实际单元边长**，罚刚度 ε = κE/h 必须与它匹配：
    # 按 0.5 µm 算而网格实际是 50 µm 时，ε 大了 100 倍，首个延拓步的罚压力就到
    # 1e12 Pa 量级、Newton 直接出 NaN（实测 SNES -9）。故 tiny 与降规模网格都跟随
    # 实际的加密带尺寸（h_fine_eff × edge_ratio = 实际单元边长）。
    edge_eff = float(raw["mesh"].get("edge_target", el["edge_target"]))
    h_contact = (float(cc["h_contact"]) if not raw["mesh"].get("tiny")
                 and edge_eff == float(el["edge_target"])
                 else h_fine_eff * float(el["edge_ratio"]))
    # bottom_support：下侧平面支撑的 y 坐标（对称夹持的下颚）。None = 不加。
    bs = cc.get("bottom_support")
    if bs is not None and str(bs) in ("auto", "bottom"):
        bs = -H                       # 单丝底面
    sym = bool(cc.get("symmetric", False))
    clamp = ClampContact(msh, facet_tags, E=material.elastic.E,
                         R_clamp=float(cc["R_clamp"]), kappa=float(cc["kappa"]),
                         h_contact=h_contact,
                         bottom_support=None if bs is None else float(bs),
                         symmetric=sym, depth=H)
    ld = raw["loading"]
    self_eq = bool(ld.get("self_equilibrated", False))
    loads = ClampLoads(msh, facet_tags, H=float(ld["H"]),
                       sigma_a=float(ld["sigma_a"]), T=float(ld["T_prime"]),
                       side=str(ld.get("side", "right")),
                       self_equilibrated=self_eq)

    # --- 求解器：plastic + contact，必须 newton ---
    model = PhaseFieldModel(material=material, split="spectral", k_res=1.0e-6)
    sv = raw.get("solver", {})
    cfg = SolverConfig(u_solver=str(sv.get("u_solver", "newton")),
                       tol_d=float(sv.get("tol_d", 1.0e-4)),
                       max_iter=int(sv.get("max_iter", 50)),
                       u_linesearch=str(sv.get("u_linesearch", "bt")),
                       # d 用 snes_atol/rtol（默认 1e-8/1e-6，见 SolverConfig 注释）；
                       # u 另设（6.1 标定值 atol=1e-6/rtol=1e-10），两者量级差很远。
                       snes_rtol=float(sv.get("snes_rtol_d", 1.0e-6)),
                       snes_atol=float(sv.get("snes_atol_d", 1.0e-8)),
                       snes_rtol_u=float(sv.get("snes_rtol", 1.0e-10)),
                       snes_atol_u=float(sv.get("snes_atol", 1.0e-6)),
                       # max_it 也要分开：6.1 的 60 是给 u 的，d 的 SNES 默认 200。
                       # 实测 d 在 53/60 次时报 -6（还没到 200 就被截断）。
                       snes_max_it=int(sv.get("snes_max_it_d", 200)))

    def bcs_u(V):
        # 构型由 loading.support 选：simply_supported（两端简支）/ bottom（竖向支撑）
        # / free（D-67 §3 原文）。见 end_bcs 的实测对比。
        support = str(raw["loading"].get("support", "free"))
        return end_bcs(V, msh, facet_tags,
                       simply_supported=(support == "simply_supported"),
                       bottom_fixed=(support == "bottom"),
                       axial_anchor=(support == "axial"))

    alpha_T, form, kappa_f = material.require_fatigue_params()
    state = FatigueState(msh, model, alpha_T=alpha_T, form=form, kappa=kappa_f)
    solver = PhaseFieldSolver(
        msh, model, bcs_u=bcs_u, external_work=loads.traction(),
        f_alpha=state.f_alpha, config=cfg,
        plastic=(None if raw.get("no_plastic")
                 else PlasticLaw(msh, load_j2_params(root / raw["material"]))),
        contact=clamp)
    state.bind_solver(solver)

    # --- cycle-jump ---
    c = raw["loading"]["cycle"]
    sd = raw["loading"]["shakedown"]
    jump_cfg = CycleJumpConfig(
        n_sub=int(c["n_sub"]), dN_init=float(c["dN_init"]), dN_min=float(c["dN_min"]),
        dN_max=float(c["dN_max"]), dd_max=float(c["dd_max"]),
        grow_below=float(c["grow_below"]), factor=float(c["factor"]),
        N_max=float(c["N_max"]),
        peak_phase=None if c.get("peak_phase") is None else float(c["peak_phase"]),
        shakedown=ShakedownConfig(zero_tol=float(sd["zero_tol"]),
                                  decay_ratio=float(sd["decay_ratio"]),
                                  n_check=int(sd["n_check"])))
    import math
    phase = [0.0]

    def set_phase(t_over_T: float) -> None:
        value = math.sin(2.0 * math.pi * float(t_over_T))
        loads.set_phase(value)

    driver = CycleJumpDriver(solver, state, set_phase, jump_cfg)

    # --- 采样网格 ---
    # spacing 必须 <= h_fine（DamageGrid 的校验：粗于网格会低估裂纹深度）。
    # tiny 模式下 h_fine 是 50 µm，故 spacing 也要跟着放宽，否则会在 40 mm 宽、
    # 150 µm 深的窗口上铺出上亿个采样点。
    sp = raw["sampling"]
    spacing = float(sp["spacing"]) if not raw["mesh"].get("tiny") else 10.0e-6
    grid = DamageGrid(msh, GridSpec(x_min=-float(sp["halfwidth"]),
                                    x_max=float(sp["halfwidth"]),
                                    depth_max=float(sp["depth_max"]),
                                    spacing=spacing,
                                    y_surface=0.0),
                      h_fine=h_fine_eff)

    case = ClampCase(mesh=msh, facet_tags=facet_tags, clamp=clamp, loads=loads,
                     solver=solver, state=state, driver=driver, grid=grid,
                     config=raw)
    # root 与输出目录存进 config：run_clamp_case 要用，别在各处再拼一遍
    case.config["_root"] = root
    case.config["_outdir"] = root / raw["output"]["dir"]
    case.a_contact = a_contact
    case.h_contact = h_contact
    if resumed is not None:
        resumed.read_fields(solver, state)
        driver.N = float(resumed.meta["N"])
        driver.delta_N = float(resumed.meta["delta_N"])
    return case


def end_bcs(V, msh, facet_tags, *, bottom_fixed: bool = False,
            simply_supported: bool = False, axial_anchor: bool = False):
    """基准工况的位移约束。三种构型（互斥）：

    ``simply_supported=True``：**两端简支**。左端 u_x = u_y = 0，右端 u_y = 0
    （u_x 自由，让轴力 T' 能传）。接触反力由远端支座承担，而支座离接触区
    20 mm（5.3 H），圣维南衰减后不干扰接触区与弯曲——这是局部压入 + 弯曲的
    标准构型，也是唯一一个同时满足「接触顶得住」与「弯曲不被约束」的构型。

    ``bottom_fixed=True``：底面竖向支撑（u_y = 0、u_x 自由）。接触能顶住
    （W 正常），但实测这种支撑下 σ_a 只剩目标的 15%——底边全长不能变形，
    弯曲被支座约束掉了。

    两者都 False（D-67 §3 原文）：左端 u_x = 0、底边中点 u_y = 0。块体近似
    自由，接触力把它推走、压力建立不起来（实测 W 只有 1/23）。
    """
    import math

    import numpy as np
    from dolfinx import default_scalar_type, fem

    from pfwire.mesh import BLOCK_BOTTOM, BLOCK_LEFT, BLOCK_RIGHT

    fdim = msh.topology.dim - 1
    if simply_supported:
        # 两端简支：左端全固定，右端只压 y（u_x 自由，轴力可传）
        return [
            fem.dirichletbc(
                np.zeros(msh.geometry.dim, dtype=default_scalar_type),
                fem.locate_dofs_topological(V, fdim, facet_tags.find(BLOCK_LEFT)), V),
            fem.dirichletbc(default_scalar_type(0.0),
                            fem.locate_dofs_topological(
                                V.sub(1), fdim, facet_tags.find(BLOCK_RIGHT)),
                            V.sub(1)),
        ]
    if axial_anchor:
        # **只**锚定轴向（Lévesque 2011 的 yz 对称面：u_x = 0）。法向不设任何
        # 约束，由对称双夹持自平衡。这是唯一一个既不干扰弯曲、又让接触压力
        # 能建立的构型——见 end_bcs 的 docstring 里的实测对比。
        return [fem.dirichletbc(default_scalar_type(0.0),
                                fem.locate_dofs_topological(
                                    V.sub(0), fdim, facet_tags.find(BLOCK_LEFT)),
                                V.sub(0))]
    if bottom_fixed:
        # **只压 u_y，u_x 自由**（简支梁式竖向支撑）：
        #   - 竖向约束给接触提供了"顶住"的反力（否则接触压力建立不起来，实测 W
        #     只有 1/23）；
        #   - 轴向必须自由，否则底边不能伸长/缩短，弯曲被约束掉——底面全约束
        #     （u_x=u_y=0）实测让 σ_a 只剩目标的 15%。
        return [fem.dirichletbc(default_scalar_type(0.0),
                                fem.locate_dofs_topological(
                                    V.sub(1), fdim, facet_tags.find(BLOCK_BOTTOM)),
                                V.sub(1))]
    Vx, Vy = V.sub(0), V.sub(1)
    bcs = [fem.dirichletbc(default_scalar_type(0.0),
                           fem.locate_dofs_topological(Vx, fdim, facet_tags.find(BLOCK_LEFT)),
                           Vx)]
    # 底边中点：并行下先各进程找 |x| 最小的底边节点，再取全局最优
    dofs_y = fem.locate_dofs_topological(Vy, fdim, facet_tags.find(BLOCK_BOTTOM))
    coords = V.tabulate_dof_coordinates()
    cand = dofs_y[dofs_y < Vy.dofmap.index_map.size_local]
    if cand.size:
        i = int(np.argmin(np.abs(coords[cand, 0])))
        x_best, dof_best = float(abs(coords[cand[i], 0])), int(cand[i])
    else:
        x_best, dof_best = math.inf, -1
    owner = min(msh.comm.allgather((x_best, dof_best, msh.comm.rank)))
    if owner[2] == msh.comm.rank and owner[1] >= 0:
        bcs.append(fem.dirichletbc(default_scalar_type(0.0),
                                   np.array([owner[1]], dtype=np.int32), Vy))
    return bcs


def find_working_delta(case: ClampCase, W_target: float, *, delta0: float = 17.0e-6,
                       rtol: float = 1.0e-2, max_iter: int = 20) -> float:
    """T4 的 FEM 侧：割线法搜 δ 使接触反力 = W'，并把结果留在 case 上。"""
    delta = secant_delta(lambda d: solve_contact_at(case, d), W_target,
                         delta0=delta0, rtol=rtol, max_iter=max_iter)
    case.set_delta(delta)
    return delta


#: δ 延拓的步数。必须延拓：一次性把线夹压到 δ 会让初始（u = 0）的罚压力达到
#: p = ε·δ ≈ (κE/h)·δ ≈ 1.4e18 × 17e-6 ≈ 2e13 Pa —— Newton 从这样的初值起步
#: 直接发散（实测 SNES -5，残差停在 1e11 不降）。6.1 的 indent_plastic_obstacle
#: 同样用 δ 序列（1→15 µm）而非一步到位。延拓不改变解，只改抵达路径。
DELTA_STEPS = 12


#: δ 延拓的**第一步**的罚压力目标（Pa）。
#:
#: 必须延拓，且**首步必须极小**：初值 u = 0 时线夹已压到 δ，穿透就是 δ 本身，
#: 罚压力 p = ε·δ = (κE/h)·δ。h = 0.5 µm（布局 C）时 ε ≈ 1.4e18，δ = 1 µm 就给出
#: 1.4e12 Pa —— 比 σ_y 高 4 个数量级，Newton 从这样的初值起步必然发散
#: （实测 SNES -6/-9）。6.1 的 h_contact = 50 µm 时 ε 小 10 万倍，故它用 12 步
#: 等间距延拓就够了；细网格上必须让首步的罚压力落在材料承载量级。
RAMP_P0 = 1.0e7
#: 延拓步长的几何增长因子（δ_{k+1} = factor · δ_k）。
RAMP_FACTOR = 1.6
#: 延拓步数上限（兜底，防 δ₀ 极大时步数失控）。
RAMP_MAX_STEPS = 80


def _ramp_plan(delta: float, eps: float, p0: float = RAMP_P0,
               factor: float = RAMP_FACTOR, max_steps: int = RAMP_MAX_STEPS,
               h_contact: float | None = None, R_clamp: float | None = None
               ) -> list[float]:
    """生成 δ 延拓序列：首步取两个下限里的大者，之后按 ``factor`` 几何增长。

    首步的两个下限：
      1. **罚压力**：ε·d₁ ≈ p0（避免 Newton 从超 σ_y 的初值起步）；
      2. **接触区可分辨**：a = √(R·d₁) ≥ 5·h_contact，即 d₁ ≥ 25 h²/R。

    为什么要第二个下限：只按罚压力定首步时，d₁ 会小到接触区不足一个单元
    ——积分点上 g 的符号剧烈跳变，残差不连续，Newton 报 -6（实测：δ 到
    1.2e-8 时 W ≡ 0 完全没接触，下一步 δ=1.9e-8 首次接触即 -6）。

    ⚠ 两个下限必须**相容**，否则无解：
        25h²/R ≤ 2σ_y h/(κE)   ⟺   h ≤ 2σ_y R/(25κE)
    σ_y = 1.67e8、R = 0.33 m、E = 69e9 时，κ = 10 要求 h ≤ 6.4 µm——
    **6.1 的 h = 50 µm 在 κ = 10 下无解**（这就是单侧线夹跑不通的根因）；
    κ = 1 时要求 h ≤ 64 µm，50 µm 才落进去。κ 的取值因此不是自由的。
    """
    if delta <= 0.0:
        return []
    d1 = p0 / eps
    if h_contact is not None and R_clamp is not None and R_clamp > 0.0:
        d1 = max(d1, 25.0 * h_contact**2 / R_clamp)
    if d1 >= delta:
        return [delta]
    n = int(math.ceil(math.log(delta / d1) / math.log(factor)))
    n = max(1, min(n, int(max_steps)))
    seq = list(d1 * factor ** np.arange(n))
    if seq[-1] < delta * (1.0 - 1.0e-12):
        seq.append(delta)
    else:
        seq[-1] = delta
    return [float(x) for x in seq]


def _ramp_to(case: ClampCase, delta: float, steps: int = DELTA_STEPS) -> None:
    """把 δ 从当前位置沿延拓序列推到 ``delta``，每步解一次平衡。

    只在目标比当前**更深**时推进；变浅（割线法回退）时直接设值——接触是
    单调加载下才需要延拓，卸载不发散。
    """
    current = case.delta
    if not (current == current):          # NaN：还没设过
        current = 0.0
    if delta <= current:
        case.set_delta(delta)
        case.solver.solve_displacement()
        return
    eps = case.clamp.eps
    for d in _ramp_plan(delta, eps, h_contact=getattr(case, "h_contact", None),
                        R_clamp=case.clamp.R_clamp):
        if d <= current:
            continue
        case.set_delta(float(d))
        case.solver.solve_displacement()


def solve_contact_at(case: ClampCase, delta: float) -> float:
    """把线夹下压到 δ 并解一次平衡，返回接触反力 W(δ)（N/m，单位厚度）。

    给 :func:`secant_delta` 当 ``W_of_delta`` 用（T4 的 FEM 侧）。
    δ 是**延拓**着加上去的（见 :data:`DELTA_STEPS`），不是一步到位。
    """
    _ramp_to(case, delta)
    return case.clamp.reaction(case.solver.u)


def kappa_sensitivity(case: ClampCase, kappa_alt: float) -> dict:
    """κ 敏感性：换 κ 重算**接触反力**，报告相对变化（D-67 §3，6.1 的口径，阈值 3%）。

    ⚠ 不能「换掉 solver.contact 再让相场求解器重解」：``NonlinearProblem`` 的残差
    form 在**构造时**就已编译，换属性不会改变已装好的 form，那样重解出的还是
    原来的 κ（静默给出 rel = 0，看起来「完全不敏感」）。这里另建一个独立的
    弹性接触求解器：同一网格、同一 δ、同一材料，只装「线弹性 + 接触」两个项。

    只比接触反力，不跑第二遍完整疲劳——那要付与本体相同的算例代价。
    """
    import numpy as np
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import NonlinearProblem

    from pfwire.clamp_contact import ClampContact

    cc = case.config["contact"]
    msh, ft = case.mesh, case.facet_tags
    E = case.solver.model.material.elastic.E
    nu = case.solver.model.material.elastic.nu

    def reaction_with(kappa: float) -> float:
        # 必须与主 case 同构型（对称双夹持 / 平面支撑），否则单侧接触在只锚
        # u_x 的约束下"顶不住"，延拓第一步就发散（实测 SNES -6）。
        clamp = ClampContact(msh, ft, E=E, R_clamp=float(cc["R_clamp"]), kappa=kappa,
                             h_contact=float(getattr(case, "h_contact", cc["h_contact"])),
                             bottom_support=case.clamp.bottom_support,
                             symmetric=case.clamp.symmetric,
                             depth=getattr(case.clamp, "depth", None))
        clamp.set_delta(0.0)          # 延拓起点：见 DELTA_STEPS
        V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
        u = fem.Function(V)
        v = ufl.TestFunction(V)
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        eps = ufl.sym(ufl.grad(u))
        sig = 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(msh.geometry.dim)
        F = (ufl.inner(sig, ufl.sym(ufl.grad(v))) * ufl.dx
             - clamp.residual(u, v))
        support = str(case.config["loading"].get("support", "free"))
        bcs = end_bcs(V, msh, ft,
                      simply_supported=(support == "simply_supported"),
                      bottom_fixed=(support == "bottom"),
                      axial_anchor=(support == "axial"))
        problem = NonlinearProblem(
            F, u, bcs=bcs, petsc_options_prefix="clamp_kappa_",
            petsc_options={"snes_type": "newtonls", "snes_linesearch_type": "bt",
                           "snes_rtol": 1.0e-10, "snes_atol": 1.0e-10,
                           "snes_max_it": 60, "ksp_type": "preonly", "pc_type": "lu",
                           "pc_factor_mat_solver_type": "mumps"})
        # δ 延拓：一次性压到 δ 会让初始罚压力远超 σ_y，Newton 从 u=0 起步发散。
        # 首步按「罚压力 ≈ RAMP_P0」定，之后几何增长（见 _ramp_plan）。
        for d in _ramp_plan(case.delta, clamp.eps,
                            h_contact=clamp.h_contact if hasattr(clamp, "h_contact") else None,
                            R_clamp=clamp.R_clamp):
            clamp.set_delta(float(d))
            problem.solve()
            if problem.solver.getConvergedReason() <= 0:
                raise RuntimeError(
                    "κ 敏感性的接触求解未收敛（κ=%g，δ=%.4g，SNES reason=%d）"
                    % (kappa, d, problem.solver.getConvergedReason()))
        return clamp.reaction(u)

    W1 = reaction_with(float(cc["kappa"]))
    W2 = reaction_with(float(kappa_alt))
    return {"kappa": float(cc["kappa"]), "kappa_sens": float(kappa_alt),
            "W": W1, "W_sens": W2, "delta": case.delta,
            "rel": abs(W2 - W1) / W1 if W1 else float("nan")}


def run_clamp_case(case: ClampCase, *, log=None, resume_meta: dict | None = None) -> dict:
    """推进 cycle-jump，直到 θ_later 可测（d ≥ threshold 达 turn_depth）或 N_max。

    D-67 §5：验收判据全是**起裂与转向**，转向需要 θ_later（100 µm 深度的质心），
    故终止条件是「裂纹走到 turn_depth」而不是「起裂（6 µm）就停」。

    T7：每个 jump 的 shakedown 标签记进 history；``RatchetDetected``
    （not_decaying）**停下报告，不调参**（硬约束 6）。
    """
    import json
    import time

    from mpi4py import MPI

    from pfwire import checkpoint
    from pfwire.cyclejump import RatchetDetected

    comm = case.mesh.comm
    is_root = comm.rank == 0

    def say(level, msg, *args):
        if log is not None:
            getattr(log, level)(msg, *args)
        elif is_root:
            print(msg % args if args else msg, flush=True)

    raw = case.config
    nc = raw["nucleation"]
    outdir = raw["_outdir"]
    a_contact = float(case.a_contact)
    save_every = int(raw["output"].get("save_every", 25))
    dump_every = int(raw["output"].get("dump_every", 10))
    turn_depth = float(nc["turn_depth"])

    history: list[dict] = list((resume_meta or {}).get("history", []))
    shakedown_tags: list[str] = list((resume_meta or {}).get("shakedown_tags", []))
    N_nuc = (resume_meta or {}).get("N_nuc")
    t0 = time.perf_counter()
    status = "running"

    def measure():
        return crack_metrics_from_field(case)

    def record(report):
        m = measure()
        history.append({
            "N": report.N, "delta_N": report.delta_N, "max_d": report.max_d,
            "max_alpha_bar": report.max_alpha_bar,
            "shakedown": report.shakedown,
            "depth_max": None if m is None else m.depth_max,
            "x_init": None if m is None else m.x_init,
            "theta_init_deg": None if m is None else m.theta_init_deg,
            "theta_later_deg": None if m is None else m.theta_later_deg,
        })
        if report.shakedown is not None:
            shakedown_tags.append(str(report.shakedown))
        return m

    def stop(report) -> bool:
        nonlocal N_nuc, status
        m = record(report)
        if N_nuc is None and m is not None and m.reached_nuc:
            N_nuc = report.N
            say("info", "起裂：N=%.6g，x_init=%.4g mm，θ_init=%.2f°",
                report.N, 1e3 * m.x_init, m.theta_init_deg)
        if len(history) % dump_every == 0:
            say("info", "N=%.5g ΔN=%.4g max_d=%.4f 深度=%.3g µm 已用 %.0f s",
                report.N, report.delta_N, report.max_d,
                float("nan") if m is None else 1e6 * m.depth_max,
                time.perf_counter() - t0)
        if save_every > 0 and len(history) % save_every == 0:
            _save(case, outdir, history, shakedown_tags, N_nuc)
        # 终止：裂纹走到 turn_depth（θ_later 可测）
        return bool(m is not None and m.depth_max >= turn_depth)

    try:
        reports = case.driver.run(stop=stop)
    except RatchetDetected as exc:
        # T7：棘轮 -> 停下报告，绝不调参（硬约束 6）
        status = "ratcheting"
        say("error", "检测到棘轮（ratcheting），停下报告：%s", exc)
        reports = list(case.driver.reports)
    finally:
        # 退出前一定存一次断点（D-30 的教训：只按倍数存会丢掉失败时刻的状态）
        if save_every > 0:
            try:
                _save(case, outdir, history, shakedown_tags, N_nuc)
            except Exception as exc:
                say("error", "退出前存断点失败：%s", exc)

    final = reports[-1] if reports else None
    if status == "running":
        if final is not None and not final.accepted:
            status = "not_converged"
            say("error", "cycle-jump 在 N=%.5g 未收敛：%s", final.N, final.failure_mode)
        elif final is not None and final.runout:
            status = "runout"
        else:
            status = "completed"

    metrics = measure()
    acc = judge_acceptance(
        metrics, N_nuc=N_nuc, a_contact=a_contact,
        a_pl=float(nc["x_init_tol"]), incline_deg=float(nc["incline_deg"]),
        turn_deg=float(nc["turn_deg"]))

    tags: dict[str, int] = {}
    for t in shakedown_tags:
        tags[t] = tags.get(t, 0) + 1
    result = {
        "case": raw.get("case", "clamp_base"),
        "case_type": "clamp",
        "status": status,
        "delta": case.delta,
        "W_target": float(raw["loading"]["W_target"]),
        "N_final": float(history[-1]["N"]) if history else 0.0,
        "N_nuc": None if N_nuc is None else float(N_nuc),
        "acceptance": {
            "passed": bool(acc.passed),
            "checks": {k: bool(v) for k, v in acc.checks.items()},
            "x_init": acc.x_init,
            "theta_init_deg": acc.theta_init_deg,
            "theta_later_deg": acc.theta_later_deg,
            "a_pl": acc.a_pl,
            "contact_edges": list(acc.contact_edges),
            "details": acc.details,
        },
        "shakedown_tags": tags,
        "n_jumps": len(history),
        "timing_s": {"solve": time.perf_counter() - t0},
        "history": history,
    }
    if is_root:
        outdir.mkdir(parents=True, exist_ok=True)
        tmp = outdir / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(outdir / "result.json")
    return result


def _save(case, outdir, history, shakedown_tags, N_nuc) -> None:
    """存断点（adios4dolfinx）。"""
    from pfwire import checkpoint
    checkpoint.save(outdir, case.solver, case.state,
                    {"N": case.driver.N, "delta_N": case.driver.delta_N,
                     "n_jumps": len(history), "history": history,
                     "shakedown_tags": shakedown_tags, "N_nuc": N_nuc},
                    case.mesh.comm, facet_tags=case.facet_tags)


def crack_metrics_from_field(case: ClampCase) -> "CrackMetrics | None":
    """采样当前 d 场并算起裂/转向度量（T5 与 T6 共用同一函数）。"""
    nc = case.config["nucleation"]
    values = case.grid.sample(case.solver.d)
    return crack_metrics(values, case.grid.x_coords, case.grid.depth_coords,
                         threshold=float(nc["threshold"]),
                         nucleation_depth=float(nc["nucleation_depth"]),
                         turn_depth=float(nc["turn_depth"]))
