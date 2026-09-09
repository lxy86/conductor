"""gmsh 网格生成（PLAN.md Phase 1 验证算例 / Phase 2.1 微动试样）。

当前实现
--------
- :func:`sent_mesh`：单边缺口拉伸（SENT）试样，用于验证 1（Miehe et al. 2010 第 5.1 节）。
  几何为 W×H 方板，中高度处自左边缘向内的水平缺口。
- :func:`ct_mesh`：紧凑拉伸（CT）试样，用于验证 2（Carrara et al. 2020 §5.2.1 Fig. 12a）。

两个缺口的做法不一样，别混
--------------------------
SENT 的缺口是**零厚度切缝**（见下），而 CT 的缺口按原文 Fig. 12a 是**有宽度的机加槽**
（高 W/16）+ 尖端锥收，所以 CT 直接用 OCC 布尔差（矩形挖两个销孔和一个槽多边形）即可，
不需要重合点那套技巧。把 CT 也做成零厚度切缝是错的：槽宽 W/16 = 3.125 mm 远大于
ℓ = 0.2 mm，槽壁的自由边界会实质改变尖端场。

缺口为什么必须是**真实的零厚度切缝**
------------------------------------
Miehe 的缺口是几何不连续（Fig. 6a 的粗黑线），不是薄槽、也不是「把 d 预置为 1」。
三者的峰值载荷不同，而验证 1 的口径是峰值载荷偏差 < 5%，所以必须做成真切缝。

gmsh 里的做法：把方板拆成上下两个面，沿中线
- 缺口段（x < a）用**两条重合但互相独立**的线（起点是两个坐标相同、编号不同的点），
- 韧带段（x > a）用**同一条共享**线。
于是网格在缺口段的节点是成对重复的，两个裂纹面可以自由张开；韧带段则是连通的。
关键是用 built-in（geo）内核且**绝不调用 removeAllDuplicates**，否则重合点会被合并、
切缝消失（那时缺口段的面会变成内部面，而不是外边界）。
:func:`sent_mesh` 返回后可用「缺口段的面是否为外边界」来自检，
tests/test_mesh.py 正是这么测的。

单位：SI（m）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

__all__ = [
    "SentSpec",
    "SentMesh",
    "SENT_TAG_DOMAIN",
    "SENT_TAG_BOTTOM",
    "SENT_TAG_TOP",
    "SENT_TAG_NOTCH",
    "sent_mesh",
    "CtSpec",
    "CtMesh",
    "CT_TAG_DOMAIN",
    "CT_TAG_HOLE_TOP",
    "CT_TAG_HOLE_BOTTOM",
    "CT_TAG_NOTCH",
    "CT_TAG_OUTER",
    "ct_mesh",
]

#: 物理组标签。facet 组用于施加边界条件，cell 组标识材料域。
SENT_TAG_DOMAIN = 1
SENT_TAG_BOTTOM = 11
SENT_TAG_TOP = 12
SENT_TAG_NOTCH = 13

CT_TAG_DOMAIN = 2
CT_TAG_HOLE_TOP = 21
CT_TAG_HOLE_BOTTOM = 22
CT_TAG_NOTCH = 23
CT_TAG_OUTER = 24


@dataclass(frozen=True)
class SentSpec:
    """SENT 试样几何与网格尺寸。默认值取 Miehe et al. 2010 第 5.1 节（1×1 mm）。"""

    W: float = 1.0e-3
    H: float = 1.0e-3
    notch_length: float = 0.5e-3
    h_fine: float = 3.75e-6
    h_coarse: float = 2.0e-5
    refine_halfheight: float = 1.0e-4
    """加密带半高：|y − H/2| <= 该值的区域用 h_fine。裂纹沿中线水平走，
    故只需加密一条水平带；带外按 h_coarse 渐变。"""

    def __post_init__(self) -> None:
        for name in ("W", "H", "notch_length", "h_fine", "h_coarse", "refine_halfheight"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("%s 必须为有限正数，收到 %r" % (name, value))
        if self.notch_length >= self.W:
            raise ValueError(
                "notch_length (%g) 必须小于 W (%g)，否则试样被完全切断"
                % (self.notch_length, self.W)
            )
        if self.h_fine > self.h_coarse:
            raise ValueError("h_fine 不应大于 h_coarse")
        if 2.0 * self.refine_halfheight >= self.H:
            raise ValueError("加密带高度不应达到整个试样高度")


@dataclass(frozen=True)
class SentMesh:
    """:func:`sent_mesh` 的返回值。``facet_tags`` 用于按物理组施加边界条件。"""

    mesh: object
    cell_tags: object
    facet_tags: object
    spec: SentSpec


def sent_mesh(
    spec: SentSpec | None = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
    *,
    verbose: bool = False,
) -> SentMesh:
    """生成 SENT 试样网格（真实零厚度切缝）。

    坐标：x ∈ [0, W]，y ∈ [0, H]，缺口在 y = H/2、自 x = 0 到 x = notch_length。
    物理组见模块常量。
    """
    import gmsh

    spec = spec or SentSpec()
    W, H, a = spec.W, spec.H, spec.notch_length
    y_mid = 0.5 * H

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("sent")
        geo = gmsh.model.geo

        # 角点与中线点。pL_low / pL_up 坐标相同但是两个独立点 —— 切缝就靠它们。
        p1 = geo.addPoint(0.0, 0.0, 0.0)
        p2 = geo.addPoint(W, 0.0, 0.0)
        p3 = geo.addPoint(W, H, 0.0)
        p4 = geo.addPoint(0.0, H, 0.0)
        pR = geo.addPoint(W, y_mid, 0.0)
        pT = geo.addPoint(a, y_mid, 0.0)  # 缺口尖端，上下共享
        pL_low = geo.addPoint(0.0, y_mid, 0.0)
        pL_up = geo.addPoint(0.0, y_mid, 0.0)

        l_bottom = geo.addLine(p1, p2)
        l_right_low = geo.addLine(p2, pR)
        l_right_up = geo.addLine(pR, p3)
        l_top = geo.addLine(p3, p4)
        l_left_up = geo.addLine(p4, pL_up)
        l_left_low = geo.addLine(pL_low, p1)
        l_notch_low = geo.addLine(pL_low, pT)  # 缺口下表面
        l_notch_up = geo.addLine(pL_up, pT)  # 缺口上表面（与上一条重合但独立）
        l_ligament = geo.addLine(pT, pR)  # 韧带段，上下共享 -> 连通

        loop_low = geo.addCurveLoop(
            [l_bottom, l_right_low, -l_ligament, -l_notch_low, l_left_low]
        )
        loop_up = geo.addCurveLoop(
            [l_ligament, l_right_up, l_top, l_left_up, l_notch_up]
        )
        surf_low = geo.addPlaneSurface([loop_low])
        surf_up = geo.addPlaneSurface([loop_up])

        # 绝不调用 geo.removeAllDuplicates()：那会把 pL_low/pL_up 合并、切缝消失
        geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [surf_low, surf_up], tag=SENT_TAG_DOMAIN)
        gmsh.model.addPhysicalGroup(1, [l_bottom], tag=SENT_TAG_BOTTOM)
        gmsh.model.addPhysicalGroup(1, [l_top], tag=SENT_TAG_TOP)
        gmsh.model.addPhysicalGroup(
            1, [l_notch_low, l_notch_up], tag=SENT_TAG_NOTCH
        )

        # 尺寸场：中线附近的水平带用 h_fine，带外渐变到 h_coarse
        field = gmsh.model.mesh.field
        box = field.add("Box")
        field.setNumber(box, "XMin", 0.0)
        field.setNumber(box, "XMax", W)
        field.setNumber(box, "YMin", y_mid - spec.refine_halfheight)
        field.setNumber(box, "YMax", y_mid + spec.refine_halfheight)
        field.setNumber(box, "ZMin", -1.0)
        field.setNumber(box, "ZMax", 1.0)
        field.setNumber(box, "VIn", spec.h_fine)
        field.setNumber(box, "VOut", spec.h_coarse)
        field.setNumber(box, "Thickness", 4.0 * spec.refine_halfheight)
        field.setAsBackgroundMesh(box)
        # 让尺寸完全由场控制，不受点/曲率/边界外推影响
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        gmsh.model.mesh.generate(2)

        data = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    finally:
        gmsh.finalize()

    return SentMesh(
        mesh=data.mesh,
        cell_tags=data.cell_tags,
        facet_tags=data.facet_tags,
        spec=spec,
    )


@dataclass(frozen=True)
class CtSpec:
    """CT（紧凑拉伸）试样几何与网格尺寸。

    默认值取 Carrara et al. 2020 (CMAME 361:112731 / arXiv:1811.02244) §5.2.1
    Fig. 12a，即标准 ASTM E647 C(T) 比例：

    - ``W`` 是**载荷线到背面**的距离（不是试样总宽），原文 W = 50 mm；
    - 总宽 1.25W，总高 1.2W（中面上下各 0.6W）；
    - 两个销孔直径 0.25W，孔心在载荷线上、距裂纹面 ±0.275W；
    - 缺口是高 W/16 的机加槽，自前端面切入，尖端锥收；
    - ``a0`` 自**载荷线**量起（Fig. 12a 的 a0 标注），原文 a0/W = 0.22。

    坐标系：**原点取在载荷线与裂纹面的交点**。于是
    x ∈ [−0.25W, +W]（前端面到背面），y ∈ [−0.6W, +0.6W]，缺口尖端在 (a0, 0)。
    这样 ``a`` 和 ``a0`` 直接就是 x 坐标，与 ΔK 标定式 Y(a/W) 的自变量同一口径，
    省掉一次容易搞错的平移。
    """

    W: float = 50.0e-3
    a0_over_W: float = 0.22
    total_width_over_W: float = 1.25
    halfheight_over_W: float = 0.6
    hole_diameter_over_W: float = 0.25
    hole_offset_over_W: float = 0.275
    notch_height_over_W: float = 1.0 / 16.0
    notch_taper_length: float | None = None
    """缺口尖端锥收段长度。``None`` 表示取缺口半高，即 45° V 形尖端。

    原文 Fig. 12a 只画了锥收、未标尺寸，故这是我们的建模选择（见 DECISIONS D-19）。
    锥收长度默认 W/32 = 1.56 mm ≈ 8ℓ，尖端场由相场自己解析，锥收细节不敏感。
    """
    h_fine: float = 0.04e-3
    """裂纹路径加密带的单元尺寸。默认 = ℓ/5（原文 p17/p20 的取值）。"""
    h_coarse: float = 1.5e-3
    refine_halfheight: float = 1.0e-3
    """加密带半高（裂纹沿 y = 0 水平走）。默认 1 mm = 5ℓ。"""
    refine_x_margin: float = 2.0e-3
    """加密带自 ``a0 − refine_x_margin`` 起，保证缺口尖端本身也在带内。"""

    def __post_init__(self) -> None:
        for name in (
            "W",
            "a0_over_W",
            "total_width_over_W",
            "halfheight_over_W",
            "hole_diameter_over_W",
            "hole_offset_over_W",
            "notch_height_over_W",
            "h_fine",
            "h_coarse",
            "refine_halfheight",
            "refine_x_margin",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("%s 必须为有限正数，收到 %r" % (name, value))
        if self.notch_taper_length is not None:
            taper = float(self.notch_taper_length)
            if not np.isfinite(taper) or taper <= 0.0:
                raise ValueError("notch_taper_length 必须为有限正数或 None")
        if self.h_fine > self.h_coarse:
            raise ValueError("h_fine 不应大于 h_coarse")
        if self.a0 <= 0.0:
            raise ValueError(
                "a0 (%g) 必须为正 —— a0 自载荷线量起，缺口尖端要在载荷线右侧"
                % self.a0
            )
        if self.a0 >= self.W:
            raise ValueError(
                "a0 (%g) 必须小于 W (%g)，否则缺口贯穿到背面" % (self.a0, self.W)
            )
        # 先查几何冲突（销孔/缺口互相穿透），再查锥收 —— 否则锥收默认取缺口半高，
        # 缺口一放大就先被锥收那条拦下，报出的原因是错的。
        if self.notch_half_height >= self.hole_offset - self.hole_radius:
            raise ValueError(
                "缺口槽半高 (%g) 已达销孔下缘 (%g)，两者相交"
                % (self.notch_half_height, self.hole_offset - self.hole_radius)
            )
        if self.hole_offset + self.hole_radius >= self.halfheight:
            raise ValueError("销孔超出试样上下边界")
        # 缺口槽必须从前端面切入并穿过载荷线，才谈得上「a0 自载荷线量起」
        if self.a0 <= self.taper_length:
            raise ValueError(
                "a0 (%g) 必须大于锥收长度 (%g)" % (self.a0, self.taper_length)
            )

    # ---- 派生几何量（全部 SI，m）----
    @property
    def a0(self) -> float:
        """初始缺口长度，自载荷线量起（= 尖端的 x 坐标）。"""
        return self.a0_over_W * self.W

    @property
    def x_front(self) -> float:
        """前端面 x 坐标（负值）。总宽 1.25W 减去 W 即载荷线到前端面的距离。"""
        return -(self.total_width_over_W - 1.0) * self.W

    @property
    def x_back(self) -> float:
        return self.W

    @property
    def halfheight(self) -> float:
        return self.halfheight_over_W * self.W

    @property
    def hole_radius(self) -> float:
        return 0.5 * self.hole_diameter_over_W * self.W

    @property
    def hole_offset(self) -> float:
        return self.hole_offset_over_W * self.W

    @property
    def notch_half_height(self) -> float:
        return 0.5 * self.notch_height_over_W * self.W

    @property
    def taper_length(self) -> float:
        return (
            self.notch_half_height
            if self.notch_taper_length is None
            else float(self.notch_taper_length)
        )

    @property
    def area(self) -> float:
        """试样面积（矩形 − 两个销孔 − 缺口槽），用于网格自检。"""
        rect = (self.x_back - self.x_front) * 2.0 * self.halfheight
        holes = 2.0 * np.pi * self.hole_radius**2
        slot_x = self.a0 - self.taper_length - self.x_front
        slot = 2.0 * self.notch_half_height * slot_x
        taper = self.notch_half_height * self.taper_length  # 三角形 ×2 = 半高×锥长
        return float(rect - holes - slot - taper)


@dataclass(frozen=True)
class CtMesh:
    """:func:`ct_mesh` 的返回值。``facet_tags`` 用于按物理组施加载荷/约束。"""

    mesh: object
    cell_tags: object
    facet_tags: object
    spec: CtSpec


def ct_mesh(
    spec: CtSpec | None = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
    *,
    verbose: bool = False,
) -> CtMesh:
    """生成 CT 试样网格（Carrara 2020 Fig. 12a 几何）。

    用 OCC 内核做布尔差：矩形 − 上下两个销孔 − 缺口槽（带锥收尖端的多边形）。
    物理组见模块常量：上孔（施加 P）、下孔（约束 u = 0）、缺口面、其余外边界。
    """
    import gmsh

    spec = spec or CtSpec()
    x0, x1 = spec.x_front, spec.x_back
    hh = spec.halfheight
    nh = spec.notch_half_height
    tip_x = spec.a0
    taper = spec.taper_length

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("ct")
        occ = gmsh.model.occ

        body = occ.addRectangle(x0, -hh, 0.0, x1 - x0, 2.0 * hh)
        hole_up = occ.addDisk(0.0, spec.hole_offset, 0.0, spec.hole_radius, spec.hole_radius)
        hole_lo = occ.addDisk(0.0, -spec.hole_offset, 0.0, spec.hole_radius, spec.hole_radius)

        # 缺口槽：自前端面切入的矩形段 + 锥收到尖端的三角段。
        # 稍微越过前端面（x0 - pad）以保证布尔差干净地开口，不留薄壁。
        pad = 0.05 * (x1 - x0)
        slot_pts = [
            (x0 - pad, -nh),
            (tip_x - taper, -nh),
            (tip_x, 0.0),
            (tip_x - taper, nh),
            (x0 - pad, nh),
        ]
        pt_tags = [occ.addPoint(px, py, 0.0) for px, py in slot_pts]
        line_tags = [
            occ.addLine(pt_tags[i], pt_tags[(i + 1) % len(pt_tags)])
            for i in range(len(pt_tags))
        ]
        slot = occ.addPlaneSurface([occ.addCurveLoop(line_tags)])

        out, _ = occ.cut([(2, body)], [(2, hole_up), (2, hole_lo), (2, slot)])
        occ.synchronize()
        surfaces = [tag for dim, tag in out if dim == 2]
        if len(surfaces) != 1:
            raise RuntimeError(
                "布尔差得到 %d 个面（应为 1）—— 检查缺口槽是否把试样切断" % len(surfaces)
            )

        gmsh.model.addPhysicalGroup(2, surfaces, tag=CT_TAG_DOMAIN)

        # 按几何位置给边界曲线分组：销孔用「质心到孔心的距离」判，最稳。
        groups: dict[int, list[int]] = {
            CT_TAG_HOLE_TOP: [],
            CT_TAG_HOLE_BOTTOM: [],
            CT_TAG_NOTCH: [],
            CT_TAG_OUTER: [],
        }
        tol = 0.25 * spec.hole_radius
        for dim, curve in gmsh.model.getBoundary(
            [(2, surfaces[0])], oriented=False, recursive=False
        ):
            if dim != 1:
                continue
            cx, cy, _ = occ.getCenterOfMass(1, curve)
            if abs(cx) < tol and abs(cy - spec.hole_offset) < tol:
                groups[CT_TAG_HOLE_TOP].append(curve)
            elif abs(cx) < tol and abs(cy + spec.hole_offset) < tol:
                groups[CT_TAG_HOLE_BOTTOM].append(curve)
            elif cx < tip_x + 1e-12 and abs(cy) <= nh + 1e-12:
                groups[CT_TAG_NOTCH].append(curve)
            else:
                groups[CT_TAG_OUTER].append(curve)
        for tag, curves in groups.items():
            if not curves:
                raise RuntimeError("物理组 %d 没有匹配到任何边界曲线" % tag)
            gmsh.model.addPhysicalGroup(1, curves, tag=tag)

        # 尺寸场：裂纹路径（自缺口尖端到背面）的水平带用 h_fine
        field = gmsh.model.mesh.field
        box = field.add("Box")
        field.setNumber(box, "XMin", tip_x - spec.refine_x_margin)
        field.setNumber(box, "XMax", x1)
        field.setNumber(box, "YMin", -spec.refine_halfheight)
        field.setNumber(box, "YMax", spec.refine_halfheight)
        field.setNumber(box, "ZMin", -1.0)
        field.setNumber(box, "ZMax", 1.0)
        field.setNumber(box, "VIn", spec.h_fine)
        field.setNumber(box, "VOut", spec.h_coarse)
        field.setNumber(box, "Thickness", 8.0 * spec.refine_halfheight)
        field.setAsBackgroundMesh(box)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        gmsh.model.mesh.generate(2)

        data = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    finally:
        gmsh.finalize()

    return CtMesh(
        mesh=data.mesh,
        cell_tags=data.cell_tags,
        facet_tags=data.facet_tags,
        spec=spec,
    )
