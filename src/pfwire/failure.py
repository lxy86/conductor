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

两种采样网格
-----------
- :class:`DamageGrid`：裂纹自试样**上表面向下**（−y）生长 —— 微动疲劳的口径；
- :class:`CrackPathGrid`：生长方向是**任意单位向量** —— 验证 2 的 CT 试样裂纹
  自缺口尖端沿 +x 生长，用 DamageGrid 的 −y 约定就表达不出来。

两者的采样结果都是 (n_行, n_列) 数组，行方向即生长方向、行 0 是起始面，
故可直接交给同一个 :func:`extract_crack`。

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
    "CrackPathGrid",
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


def _locate_points(mesh: dmesh.Mesh, points: np.ndarray, spacing: float):
    """把采样点定位到单元上，返回 (点下标, 单元, 该点到该单元的距离平方)。

    这段是并行正确性的关键：每个 rank 认领落在自己分区（含 ghost 层）里的点，
    采样时按「距离最小者胜」合并（见 :func:`_sample_points`）。
    定位只做一次、结果缓存复用。
    最后做一次**全局**检查：任何点若没被任何 rank 认领，说明采样网格伸出了
    计算域 —— 这必须报错，否则那些点会永远保持 −inf 并被当成「无损伤」。
    """
    # 起始面那一行常正好落在边界上，给 bb_tree 一点 padding 免得浮点误差漏掉
    padding = 1.0e-6 * spacing
    tdim = mesh.topology.dim
    tree = geometry.bb_tree(mesh, tdim, padding=padding)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(mesh, candidates, points)
    owned_idx: list[int] = []
    owned_cells: list[int] = []
    owned_dist: list[float] = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if links.size == 0:
            continue
        # **不能直接取 links[0]**：dolfinx 的 compute_colliding_cells 用的是
        # **绝对**距离容差（约 1e-6，按米算就是 1 µm），而微动网格的单元只有
        # 0.5 µm —— 于是「碰撞」列表里会混进离该点一两个单元远的邻居，
        # 取第一个就等于在单元外**外插**。d 在裂纹带上 0→1 只跨 ell=2 µm，
        # 外插半个单元就够把 P1 值顶到 1.03 以上（实测采样最大值 1.074>1，
        # 而 P1 单元内必是节点值的凸组合，这正是发现该 bug 的线索）。
        #
        # 所以取**距离最小**的单元——真正含该点的那个距离恰为 0。
        # 距离本身要留着：并行下还得跨 rank 再比一次（见 _sample_points），
        # 否则某个 rank 在自己 ghost 层上外插出的偏高值会被 MAX 合并选中。
        #
        # 早先这里是「距离 > 容差就不认领」，用的是绝对容差 (1e-9*spacing)^2。
        # 那是错的：容差写在**距离平方**上，量纲与直觉差一个平方，
        # 生产网格上直接判掉 73/33740 个点、算例一启动就报「采样点在计算域外」。
        # 改成跨 rank 比距离，就不需要任何绝对尺度了。
        dist2 = geometry.squared_distance(
            mesh, tdim, np.asarray(links, dtype=np.int32),
            np.tile(points[i], (links.size, 1)),
        )
        best = int(np.argmin(dist2))
        owned_idx.append(i)
        owned_cells.append(int(links[best]))
        owned_dist.append(float(dist2[best]))
    owned_idx_arr = np.asarray(owned_idx, dtype=np.int32)

    claimed = np.zeros(points.shape[0], dtype=np.int32)
    claimed[owned_idx_arr] = 1
    total = np.zeros_like(claimed)
    mesh.comm.Allreduce(claimed, total, op=MPI.SUM)
    missing = int(np.count_nonzero(total == 0))
    if missing:
        raise ValueError(
            "采样网格有 %d/%d 个点落在计算域之外（采样区域超出了网格范围）。"
            % (missing, points.shape[0])
        )
    return (
        owned_idx_arr,
        np.asarray(owned_cells, dtype=np.int32),
        np.asarray(owned_dist, dtype=np.float64),
    )


def _sample_points(
    comm,
    function: fem.Function,
    points: np.ndarray,
    owned_idx: np.ndarray,
    owned_cells: np.ndarray,
    owned_dist: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """在缓存好的采样点上求值，allreduce 成所有 rank 一致的 (n_行, n_列) 数组。

    合并规则是**距离最小者胜**，不是单纯取 MAX：一个点可能同时落在若干 rank
    的 ghost 层里，而只有真正含它的那个 rank 距离为 0，其余 rank 是在单元外
    **外插**。直接 MAX 会挑中外插出来的偏高值（这正是 D-29 的后一半）。
    先按距离选出胜者，再对并列者取 MAX —— 并列只发生在距离同为 0 的单元边界上，
    那里 P1 连续、两边给的值本就相同。
    """
    n = points.shape[0]
    local_d = np.full(n, np.inf)
    if owned_idx.size:
        local_d[owned_idx] = owned_dist
    best_d = np.empty_like(local_d)
    comm.Allreduce(local_d, best_d, op=MPI.MIN)

    local = np.full(n, -np.inf)
    if owned_idx.size:
        values = np.asarray(function.eval(points[owned_idx], owned_cells)).ravel()
        # 只有「距离等于全局最小」的 rank 才把值投进去
        keep = local_d[owned_idx] <= best_d[owned_idx]
        local[owned_idx[keep]] = values[keep]
    glob = np.empty_like(local)
    comm.Allreduce(local, glob, op=MPI.MAX)
    return glob.reshape(shape)


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
        self._owned_idx, self._owned_cells, self._owned_dist = _locate_points(
            mesh, points, spec.spacing
        )

    def sample(self, function: fem.Function) -> np.ndarray:
        """返回形状 (n_depth, n_x) 的全局数组，所有 rank 上一致。

        行 0 是表面，行 i 对应深度 i*spacing。
        """
        return _sample_points(
            self.comm,
            function,
            self._points,
            self._owned_idx,
            self._owned_cells,
            self._owned_dist,
            self.shape,
        )


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
    tip_direction_deg: float | None = None
    """裂纹**尖端附近**的局部扩展方向（相对竖直，朝 +x 为正）。

    与 ``inclination_deg`` 的区别是窗口位置：后者固定在表层（量「初始倾角」），
    尖端长过窗口之后就不再反映当前方向。判「是否转向垂直」必须用这个量。
    见 DECISIONS.md OPEN-2。
    """

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
    tip_window: float | None = None,
    min_rows_for_angle: int = 3,
) -> CrackMetrics:
    """在采样网格上提取「与表面连通」的裂纹几何。

    - 连通域用 8 连通标记；只保留含有第 0 行（表面）像素的连通域；
    - ``depth`` 取这些连通域中最深的那个的最大深度（PLAN：沿 d>=0.95 区域最深点）；
    - ``x_surface`` 是该连通域在表面那一行的 x 形心（萌生位置）；
    - ``inclination_deg`` 是裂纹在表层窗口内相对**竖直方向**的倾角（朝 +x 为正），
      由各行 x 形心对深度做线性拟合得到；窗口默认取 10 个间距。
      它量的是**表层那一段**，用于 Phase 2.4 判据 2「初始倾斜」；
    - ``tip_direction_deg`` 是同样的拟合但窗口贴着**最深处**，量当前扩展方向，
      用于判据 3「转向垂直」。两者不可混用，见 DECISIONS.md OPEN-2。
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
        tip_direction_deg=None,
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

    # 尖端局部方向：窗口贴着**最深处**往上取，随裂纹一起走
    tip_win = window if tip_window is None else float(tip_window)
    tip_direction: float | None = None
    tip_rows = [
        r
        for r in range(mask.shape[0])
        if depth_coords[r] >= depth - tip_win - 1e-12 and mask[r, :].any()
    ]
    if len(tip_rows) >= min_rows_for_angle:
        tip_centroids = np.array(
            [float(np.mean(x_coords[np.nonzero(mask[r, :])[0]])) for r in tip_rows]
        )
        tip_slope = float(np.polyfit(depth_coords[tip_rows], tip_centroids, 1)[0])
        tip_direction = math.degrees(math.atan(tip_slope))

    return CrackMetrics(
        depth=depth,
        x_surface=x_surface,
        x_tip=x_tip,
        inclination_deg=inclination,
        area=area,
        n_surface_components=int(surface_labels.size),
        max_damage=max_damage,
        tip_direction_deg=tip_direction,
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


class CrackPathGrid:
    """沿**任意方向**生长的裂纹的采样网格（验证 2 的 CT 试样用）。

    :class:`DamageGrid` 把「深度」写死成 −y 方向（微动件表面在上、裂纹向下），
    而 CT 的裂纹自缺口尖端沿 +x 生长，用那套约定表达不出来。本类把生长方向
    抽象成单位向量 ``growth_dir``，横向方向取其左旋 90°：

        point(i, j) = origin + s_i · g + t_j · n,
        s_i = i·spacing  (i = 0..n_growth)，t_j = −halfwidth + j·spacing

    于是 :meth:`sample` 返回的 (n_行, n_列) 数组里**行方向就是生长方向**、
    行 0 是起始面（CT 里即缺口尖端所在的横截面），可直接交给
    :func:`extract_crack` —— 它返回的 ``depth`` 就是裂纹自 ``origin`` 沿
    ``growth_dir`` 的**扩展量**。

    注意 ``halfwidth``：它决定横向能容纳多宽的裂纹带，必须显著大于相场带宽
    （几个 ℓ），否则裂纹一偏折就跑出采样区、长度被低估；但也不能大到伸出
    计算域（构造时有全局检查会报错）。
    """

    def __init__(
        self,
        mesh: dmesh.Mesh,
        *,
        origin: tuple[float, float],
        growth_length: float,
        halfwidth: float,
        spacing: float,
        growth_dir: tuple[float, float] = (1.0, 0.0),
        h_fine: float | None = None,
    ) -> None:
        if growth_length <= 0.0 or halfwidth <= 0.0 or spacing <= 0.0:
            raise ValueError("growth_length、halfwidth、spacing 必须为正")
        if spacing > 0.5 * growth_length:
            raise ValueError(
                "spacing (%g) 相对 growth_length (%g) 太粗，至少要能分出 2 行"
                % (spacing, growth_length)
            )
        if h_fine is not None and spacing > h_fine:
            raise ValueError(
                "采样间距 spacing = %g 大于网格尺寸 h_fine = %g，会低估裂纹长度。"
                "请把 spacing 取到 h_fine 或更细。" % (spacing, h_fine)
            )
        norm = math.hypot(growth_dir[0], growth_dir[1])
        if norm <= 0.0:
            raise ValueError("growth_dir 不能是零向量")

        self.mesh = mesh
        self.comm = mesh.comm
        self.origin = (float(origin[0]), float(origin[1]))
        self.spacing = float(spacing)
        gx, gy = growth_dir[0] / norm, growth_dir[1] / norm
        self.growth_dir = (gx, gy)
        nx, ny = -gy, gx  # 左旋 90°

        n_growth = int(math.floor(growth_length / spacing)) + 1
        n_trans = 2 * int(math.floor(halfwidth / spacing)) + 1
        self.growth_coords = spacing * np.arange(n_growth)
        self.transverse_coords = spacing * (
            np.arange(n_trans) - (n_trans - 1) // 2
        )
        self.shape = (n_growth, n_trans)

        ss, tt = np.meshgrid(self.growth_coords, self.transverse_coords, indexing="ij")
        points = np.zeros((ss.size, 3))
        points[:, 0] = self.origin[0] + gx * ss.ravel() + nx * tt.ravel()
        points[:, 1] = self.origin[1] + gy * ss.ravel() + ny * tt.ravel()
        self._points = points
        (self._owned_idx, self._owned_cells,
         self._owned_dist) = _locate_points(mesh, points, spacing)

    def sample(self, function: fem.Function) -> np.ndarray:
        """返回形状 (n_growth, n_transverse) 的全局数组，所有 rank 上一致。"""
        return _sample_points(
            self.comm,
            function,
            self._points,
            self._owned_idx,
            self._owned_cells,
            self._owned_dist,
            self.shape,
        )

    def crack_extension(self, d: fem.Function, *, threshold: float = 0.95) -> CrackMetrics:
        """采样 d 并提取裂纹几何。``metrics.depth`` 即自 origin 的扩展量。"""
        values = self.sample(d)
        return extract_crack(
            values,
            self.transverse_coords,
            self.growth_coords,
            threshold=threshold,
        )
