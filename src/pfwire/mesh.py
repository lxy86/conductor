"""gmsh 网格生成（PLAN.md Phase 1 验证算例 / Phase 2.1 微动试样）。

当前实现
--------
- :func:`sent_mesh`：单边缺口拉伸（SENT）试样，用于验证 1（Miehe et al. 2010 第 5.1 节）。
  几何为 W×H 方板，中高度处自左边缘向内的水平缺口。

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
]

#: 物理组标签。facet 组用于施加边界条件，cell 组标识材料域。
SENT_TAG_DOMAIN = 1
SENT_TAG_BOTTOM = 11
SENT_TAG_TOP = 12
SENT_TAG_NOTCH = 13


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
