"""失效判据与裂纹几何提取（PLAN.md Phase 1.4）。

判据（PLAN.md 第 2 节 + DECISIONS.md D-1）
------------------------------------------
- d >= d_threshold（默认 0.95）的连通区域**从上表面**向下贯穿到给定深度 -> N_f；
- 深度阈值按 D-1 改为**绝对深度 100 µm**（≈5a），不再是 0.5·R_wire = 800 µm；
  仍支持 PLAN 原来的 ``depth_fraction`` 写法（乘 R_wire），二者只能给一个。
- N > N_max 未失效 -> runout（由 cyclejump.CycleJumpDriver 负责）。

连通性怎么判
-----------
在感兴趣区域上铺一层**固定的规则网格**采样 d，再用 ``scipy.ndimage.label``
做连通域标记，只保留触及表面那一行的连通域。

为什么不用网格拓扑上的连通域：那需要在 MPI 下做跨 rank 的标签传播，正确实现代价高；
而「采样 + allreduce」天然并行安全（每个 rank 只算自己拥有的点，再取全局 MAX），
且单元定位只做一次即可缓存复用（网格在整个算例中不变）。
副产物：Phase 3 画裂纹路径与 a–N 曲线正好需要这样一张规则网格。

**代价**：分辨率是网格间距 ``spacing``，故 spacing 必须 <= h_fine，否则会低估深度。
构造时会检查并在 spacing 过大时报错。连通性用 8 连通（含对角），
否则网格上的斜裂纹会被判成断开的一串孤立点。

单位：SI。深度自表面向下为正（表面在 y = y_surface，体内 y 减小）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from dolfinx import fem, geometry
from dolfinx import mesh as dmesh
from mpi4py import MPI
from scipy import ndimage

__all__ = [
    "GridSpec",
    "DamageGrid",
    "CrackMetrics",
    "FailureCriterion",
    "extract_crack",
]

#: 8 连通结构元（含对角）。斜裂纹在规则网格上常只有对角相邻，
#: 用 4 连通会把它判成断开。
_CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)

#: 深度比较的相对容差。裂纹深度由 spacing*行号 算得，二进制下会差 1 ulp
#: （例如 10*1e-6 = 9.999999999999999e-06）。判据阈值往往正好落在网格深度上
#: （如 100 µm 配 1 µm 间距），不吸收这点噪声会让失效判定晚整整一行。
#: 1e-12 的相对容差远低于任何物理分辨率，只用来抹平浮点误差。
_DEPTH_RTOL = 1.0e-12


@dataclass(frozen=True)
class GridSpec:
    """采样网格：x ∈ [x_min, x_max]，深度 ∈ [0, depth_max]，间距 spacing。"""

    x_min: float
    x_max: float
    depth_max: float
    spacing: float
    y_surface: float = 0.0

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max 必须大于 x_min")
        if self.depth_max <= 0.0 or self.spacing <= 0.0:
            raise ValueError("depth_max 与 spacing 必须为正")
        if self.spacing > 0.5 * self.depth_max:
            raise ValueError(
                "spacing (%g) 相对 depth_max (%g) 太粗，至少要能分出 2 行"
                % (self.spacing, self.depth_max)
            )

    @property
    def x_coords(self) -> np.ndarray:
        n = int(math.floor((self.x_max - self.x_min) / self.spacing)) + 1
        return self.x_min + self.spacing * np.arange(n)

    @property
    def depth_coords(self) -> np.ndarray:
        n = int(math.floor(self.depth_max / self.spacing)) + 1
        return self.spacing * np.arange(n)


class DamageGrid:
    """把 d 采样到固定规则网格上。单元定位在构造时做一次并缓存。"""

    def __init__(self, mesh: dmesh.Mesh, spec: GridSpec, *, h_fine: float | None = None):
        self.mesh = mesh
        self.spec = spec
        self.comm = mesh.comm
        if h_fine is not None and spec.spacing > h_fine:
            raise ValueError(
                "采样间距 spacing = %g 大于网格尺寸 h_fine = %g，会低估裂纹深度。"
                "请把 spacing 取到 h_fine 或更细。" % (spec.spacing, h_fine)
            )
        self.x_coords = spec.x_coords
        self.depth_coords = spec.depth_coords
        self.shape = (self.depth_coords.size, self.x_coords.size)

        xx, dd = np.meshgrid(self.x_coords, self.depth_coords)
        points = np.zeros((xx.size, 3))
        points[:, 0] = xx.ravel()
        points[:, 1] = spec.y_surface - dd.ravel()
        self._points = points

        # 表面那一行正好落在边界上，给 bb_tree 一点 padding 免得浮点误差漏掉
        padding = 1.0e-6 * spec.spacing
        tree = geometry.bb_tree(mesh, mesh.topology.dim, padding=padding)
        candidates = geometry.compute_collisions_points(tree, points)
        colliding = geometry.compute_colliding_cells(mesh, candidates, points)
        owned_idx: list[int] = []
        owned_cells: list[int] = []
        for i in range(points.shape[0]):
            links = colliding.links(i)
            if links.size > 0:
                owned_idx.append(i)
                owned_cells.append(links[0])
        self._owned_idx = np.asarray(owned_idx, dtype=np.int32)
        self._owned_cells = np.asarray(owned_cells, dtype=np.int32)

        # 全局检查：每个采样点都必须被某个 rank 认领，否则网格区域超出了计算域
        claimed = np.zeros(points.shape[0], dtype=np.int32)
        claimed[self._owned_idx] = 1
        total = np.zeros_like(claimed)
        self.comm.Allreduce(claimed, total, op=MPI.SUM)
        missing = int(np.count_nonzero(total == 0))
        if missing:
            raise ValueError(
                "采样网格有 %d/%d 个点落在计算域之外（GridSpec 超出了网格范围）。"
                "请检查 x_min/x_max/depth_max 与 y_surface。"
                % (missing, points.shape[0])
            )

    def sample(self, function: fem.Function) -> np.ndarray:
        """返回形状 (n_depth, n_x) 的全局数组，所有 rank 上一致。

        行 0 是表面，行 i 对应深度 i*spacing。
        """
        local = np.full(self._points.shape[0], -np.inf)
        if self._owned_idx.size:
            values = function.eval(self._points[self._owned_idx], self._owned_cells)
            local[self._owned_idx] = np.asarray(values).ravel()
        glob = np.empty_like(local)
        self.comm.Allreduce(local, glob, op=MPI.MAX)
        return glob.reshape(self.shape)


@dataclass(frozen=True)
class CrackMetrics:
    """从 d 场提取的裂纹几何。没有表面连通裂纹时 depth = 0、位置量为 None。"""

    depth: float
    x_surface: float | None
    x_tip: float | None
    inclination_deg: float | None
    area: float
    n_surface_components: int
    max_damage: float

    @property
    def has_surface_crack(self) -> bool:
        return self.n_surface_components > 0


def extract_crack(
    values: np.ndarray,
    x_coords: np.ndarray,
    depth_coords: np.ndarray,
    *,
    threshold: float = 0.95,
    inclination_depth: float | None = None,
    min_rows_for_angle: int = 3,
) -> CrackMetrics:
    """在采样网格上提取「与表面连通」的裂纹几何。

    - 连通域用 8 连通标记；只保留含有第 0 行（表面）像素的连通域；
    - ``depth`` 取这些连通域中最深的那个的最大深度（PLAN：沿 d>=0.95 区域最深点）；
    - ``x_surface`` 是该连通域在表面那一行的 x 形心（萌生位置）；
    - ``inclination_deg`` 是裂纹在表层窗口内相对**竖直方向**的倾角（朝 +x 为正），
      由各行 x 形心对深度做线性拟合得到；窗口默认取 10 个间距。
      Phase 2.4 要判「初始倾斜后转向垂直」就用这个量。
    """
    values = np.asarray(values)
    if values.shape != (depth_coords.size, x_coords.size):
        raise ValueError(
            "values 形状 %r 与坐标 (%d, %d) 不匹配"
            % (values.shape, depth_coords.size, x_coords.size)
        )
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold 必须在 (0,1) 内，收到 %r" % (threshold,))

    max_damage = float(np.max(values)) if values.size else 0.0
    damaged = values >= threshold
    empty = CrackMetrics(
        depth=0.0,
        x_surface=None,
        x_tip=None,
        inclination_deg=None,
        area=0.0,
        n_surface_components=0,
        max_damage=max_damage,
    )
    if not damaged.any():
        return empty

    labels, _ = ndimage.label(damaged, structure=_CONNECTIVITY_8)
    surface_labels = np.unique(labels[0, :])
    surface_labels = surface_labels[surface_labels > 0]
    if surface_labels.size == 0:
        # 有损伤但不与表面连通（例如内部孤立损伤斑）——不算裂纹
        return empty

    spacing = float(depth_coords[1] - depth_coords[0])
    best = None
    for label in surface_labels:
        mask = labels == label
        rows = np.nonzero(mask.any(axis=1))[0]
        depth = float(depth_coords[rows.max()])
        if best is None or depth > best[0]:
            best = (depth, label, mask)
    depth, label, mask = best  # type: ignore[misc]

    surface_cols = np.nonzero(mask[0, :])[0]
    x_surface = float(np.mean(x_coords[surface_cols]))
    deepest_row = int(np.nonzero(mask.any(axis=1))[0].max())
    tip_cols = np.nonzero(mask[deepest_row, :])[0]
    x_tip = float(np.mean(x_coords[tip_cols]))
    area = float(np.count_nonzero(mask) * spacing**2)

    # 倾角：表层窗口内各行 x 形心对深度线性拟合，取相对竖直方向的角度
    window = 10.0 * spacing if inclination_depth is None else float(inclination_depth)
    window = min(window, depth)
    inclination: float | None = None
    rows_in_window = [
        r
        for r in range(mask.shape[0])
        if depth_coords[r] <= window + 1e-12 and mask[r, :].any()
    ]
    if len(rows_in_window) >= min_rows_for_angle:
        centroids = np.array(
            [float(np.mean(x_coords[np.nonzero(mask[r, :])[0]])) for r in rows_in_window]
        )
        depths = depth_coords[rows_in_window]
        slope = float(np.polyfit(depths, centroids, 1)[0])
        inclination = math.degrees(math.atan(slope))

    return CrackMetrics(
        depth=depth,
        x_surface=x_surface,
        x_tip=x_tip,
        inclination_deg=inclination,
        area=area,
        n_surface_components=int(surface_labels.size),
        max_damage=max_damage,
    )


@dataclass(frozen=True)
class FailureCriterion:
    """d >= d_threshold 的表面连通裂纹达到 depth 即判失效。

    ``depth`` 是绝对深度（m），默认 100 µm，见 DECISIONS.md D-1。
    """

    d_threshold: float = 0.95
    depth: float = 1.0e-4

    def __post_init__(self) -> None:
        if not 0.0 < self.d_threshold < 1.0:
            raise ValueError("d_threshold 必须在 (0,1) 内，收到 %r" % (self.d_threshold,))
        if self.depth <= 0.0:
            raise ValueError("depth 必须为正，收到 %r" % (self.depth,))

    @classmethod
    def from_config(
        cls, cfg: dict, *, R_wire: float | None = None
    ) -> "FailureCriterion":
        """从工况 yaml 的 ``failure`` 段构造。

        支持两种写法，但只能给一个：
        - ``depth``（绝对深度，m）—— D-1 采用的写法；
        - ``depth_fraction``（乘 R_wire）—— PLAN.md 第 5 节的原写法。
        """
        unknown = set(cfg) - {"d_threshold", "depth", "depth_fraction"}
        if unknown:
            raise KeyError("failure 段有未知键 %r" % (sorted(unknown),))
        has_abs = cfg.get("depth") is not None
        has_frac = cfg.get("depth_fraction") is not None
        if has_abs and has_frac:
            raise ValueError(
                "failure 段同时给了 depth 与 depth_fraction，无法判断以哪个为准，"
                "请只留一个（D-1 采用绝对 depth）。"
            )
        if not has_abs and not has_frac:
            raise ValueError(
                "failure 段必须给 depth（绝对深度，m）或 depth_fraction（乘 R_wire）。"
            )
        if has_frac:
            if R_wire is None:
                raise ValueError("用 depth_fraction 时必须提供 R_wire")
            depth = float(cfg["depth_fraction"]) * float(R_wire)
        else:
            depth = float(cfg["depth"])
        return cls(
            d_threshold=float(cfg.get("d_threshold", 0.95)),
            depth=depth,
        )

    def is_failed(self, metrics: CrackMetrics) -> bool:
        return metrics.has_surface_crack and metrics.depth >= self.depth * (
            1.0 - _DEPTH_RTOL
        )

    def describe(self) -> str:
        return "d >= %.3g 的表面连通裂纹深度 >= %.4g m" % (self.d_threshold, self.depth)
