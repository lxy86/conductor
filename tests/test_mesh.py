"""mesh.py 单测：SENT 试样，重点验证缺口是**真实零厚度切缝**。

测试意图
--------
1. 面积等于 W×H（零厚度切缝不吃掉面积；若缺口被做成薄槽，面积会偏小）；
2. **缺口段的面是外边界**：切缝两侧节点重复、裂纹面自由 -> 这些 facet 属于外边界；
   而韧带段（x > a）的中线 facet 必须是**内部面**（上下连通）。
   这是「切缝到底做成了没有」的判据 —— 若 gmsh 把重合点合并，本测试立刻失败；
3. 缺口段外边界 facet 的数量约为 2×(a/h_fine)（两个裂纹面各一层）；
4. 纯拉伸下缺口**真的张开**：x = a/2 处中线上下的 u_y 存在跳跃，
   而韧带段同一位置无跳跃。这是功能性验证，比只看拓扑更有说服力；
5. 物理组标签可用于施加边界条件，且尺寸场生效（加密带内单元明显小于带外）;
6. 参数校验：缺口不得贯穿、h_fine 不得大于 h_coarse 等。

CT 试样（验证 2，Carrara 2020 Fig. 12a）
--------------------------------------
7. 派生几何量与原文标注逐条对表（1.25W / 1.2W / Ø0.25W / ±0.275W / W/16、
   a0 自**载荷线**量起）—— 这些是「几何搭错了」最容易出的地方；
8. 面积：矩形 − 两销孔 − 缺口槽。注意销孔是多边形离散的，网格面积必然**略大于**
   解析值（内接多边形挖得少），故只能给上界并检查符号；
   而**外边界周长**是直线段构成的，可以精确对表（矩形周长 − 缺口开口宽）；
9. 缺口是**有宽度的机加槽**（与 SENT 的零厚度切缝不同）：槽内是挖空的，
   故缺口段中线上不应有任何单元，而韧带段中线必须连通；
10. 销孔周长与解析弧长比对（差应等于弦/弧比，由 facet 数决定）。

微动试样（Phase 2.1，DECISIONS.md D-1）
--------------------------------------
11. 面积与**四条边长全部可精确对表**（矩形，全是直线，没有圆弧/切缝那些离散误差）；
12. **接触区分辨率**：上表面 |x| <= a 段的 facet 数必须 ≈ 2a/h_fine。
    这是 D-1 把 ℓ 从 40 µm 改成 4 µm 的全部理由 —— 接触半宽分辨不出来，
    「裂纹在 x ≈ ±a 的滑移区萌生」（Phase 2.4 的硬关卡）就在数值上被抹平；
13. 上表面必须**平整无缺口**：所有 TOP facet 都在 y = 0 上。微动裂纹要靠接触
    面力自己萌生，预置任何几何缺口都等于把答案写进输入；
14. 加密区内外的单元尺寸分档正确；
15. 生产默认参数必须满足 D-1 的分辨率要求（2a/h >= 40），这条只查 spec 不建网格。
"""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pfwire import mesh as pfmesh

# 用比 Miehe 更粗的网格跑测试（几何逻辑与尺寸无关）
COARSE = pfmesh.SentSpec(
    W=1.0e-3,
    H=1.0e-3,
    notch_length=0.5e-3,
    h_fine=2.0e-5,
    h_coarse=8.0e-5,
    refine_halfheight=1.0e-4,
)


@pytest.fixture(scope="module")
def sent():
    return pfmesh.sent_mesh(COARSE)


def test_area_equals_full_rectangle(sent):
    msh = sent.mesh
    area = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ufl.dx)), op=MPI.SUM
    )
    assert area == pytest.approx(COARSE.W * COARSE.H, rel=1e-9)


def test_notch_faces_are_exterior_but_ligament_is_interior(sent):
    """切缝的判据：缺口段中线 facet 在外边界上，韧带段不在。"""
    msh = sent.mesh
    spec = sent.spec
    y_mid = 0.5 * spec.H
    tol = 0.1 * spec.h_fine
    fdim = msh.topology.dim - 1

    def on_midline_notch(x):
        return np.isclose(x[1], y_mid, atol=tol) & (x[0] < spec.notch_length - tol)

    def on_midline_ligament(x):
        return np.isclose(x[1], y_mid, atol=tol) & (x[0] > spec.notch_length + tol)

    notch_boundary = dmesh.locate_entities_boundary(msh, fdim, on_midline_notch)
    ligament_boundary = dmesh.locate_entities_boundary(msh, fdim, on_midline_ligament)
    n_notch = msh.comm.allreduce(notch_boundary.size, op=MPI.SUM)
    n_ligament = msh.comm.allreduce(ligament_boundary.size, op=MPI.SUM)

    assert n_notch > 0, (
        "缺口段中线上没有外边界 facet —— 说明重合点被合并、切缝没做成"
    )
    assert n_ligament == 0, (
        "韧带段中线上出现了外边界 facet —— 说明试样被整条切断了"
    )

    # 两个裂纹面各铺一层：facet 数约为 2*(a/h_fine)
    expected = 2.0 * spec.notch_length / spec.h_fine
    assert 0.5 * expected <= n_notch <= 2.0 * expected, (
        f"缺口 facet 数 {n_notch} 与预期 {expected:.0f} 差太多"
    )


def global_tag_set(msh, tags) -> set[int]:
    """所有 rank 上出现过的物理组标签的**并集**。

    并行下每个 rank 只拥有分区里的那部分边界，`np.unique(tags.values)` 是
    rank 局部的：2 进程下可能一个 rank 看到 {11,13}、另一个看到 {12,13}，
    于是「缺少物理组 12」这种假失败就冒出来了。
    这正是 CLAUDE.md「已知陷阱 3：并行下不要写按 rank 局部的断言」。
    """
    local = set(np.unique(tags.values).tolist()) if tags.values.size else set()
    return set().union(*msh.comm.allgather(local))


def test_physical_groups_present(sent):
    tags = sent.facet_tags
    assert tags is not None
    present = global_tag_set(sent.mesh, tags)
    for tag in (
        pfmesh.SENT_TAG_BOTTOM,
        pfmesh.SENT_TAG_TOP,
        pfmesh.SENT_TAG_NOTCH,
    ):
        assert tag in present, f"缺少物理组 {tag}"


def test_size_field_refines_the_midline_band(sent):
    """加密带内的单元面积应显著小于带外。"""
    msh = sent.mesh
    spec = sent.spec
    tdim = msh.topology.dim
    V0 = fem.functionspace(msh, ("DG", 0))
    area = fem.Function(V0)
    area.interpolate(fem.Expression(ufl.FacetArea(msh) * 0.0 + ufl.CellVolume(msh),
                                    V0.element.interpolation_points))
    centroids = fem.Function(V0)
    centroids.interpolate(
        fem.Expression(ufl.SpatialCoordinate(msh)[1], V0.element.interpolation_points)
    )
    n_local = V0.dofmap.index_map.size_local
    y = centroids.x.array[:n_local]
    A = area.x.array[:n_local]
    y_mid = 0.5 * spec.H
    inside = np.abs(y - y_mid) < 0.5 * spec.refine_halfheight
    outside = np.abs(y - y_mid) > 0.45 * spec.H
    mean_in = msh.comm.allreduce(float(A[inside].sum()), op=MPI.SUM) / max(
        msh.comm.allreduce(int(inside.sum()), op=MPI.SUM), 1
    )
    mean_out = msh.comm.allreduce(float(A[outside].sum()), op=MPI.SUM) / max(
        msh.comm.allreduce(int(outside.sum()), op=MPI.SUM), 1
    )
    assert mean_in < 0.5 * mean_out, (
        f"加密带没生效：带内平均单元面积 {mean_in:.3e} vs 带外 {mean_out:.3e}"
    )
    # 单元尺寸量级应与 h_fine/h_coarse 相符（三角形面积 ~ h^2/2）
    assert np.sqrt(2.0 * mean_in) == pytest.approx(spec.h_fine, rel=0.6)


def test_notch_opens_under_tension(sent):
    """功能性验证：纯拉伸下缺口张开，韧带段不张开。

    只解线弹性（无损伤），底边固定、顶边给竖向位移，
    比较 x = a/2 与 x = (a+W)/2 两处中线上下的 u_y 差。
    """
    msh = sent.mesh
    spec = sent.spec
    y_mid = 0.5 * spec.H
    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    fdim = msh.topology.dim - 1
    tags = sent.facet_tags

    u_top = 1.0e-6  # m
    bottom_dofs = fem.locate_dofs_topological(
        V, fdim, tags.find(pfmesh.SENT_TAG_BOTTOM)
    )
    top_dofs = fem.locate_dofs_topological(V, fdim, tags.find(pfmesh.SENT_TAG_TOP))
    bcs = [
        fem.dirichletbc(
            np.zeros(2, dtype=default_scalar_type), bottom_dofs, V
        ),
        fem.dirichletbc(
            np.array([0.0, u_top], dtype=default_scalar_type), top_dofs, V
        ),
    ]

    from dolfinx.fem.petsc import LinearProblem

    from pfwire.material import lame_parameters

    lmbda, mu = lame_parameters(210.0e9, 0.3)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    sigma = 2.0 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(2)
    a_form = ufl.inner(sigma, eps(v)) * ufl.dx
    zero = fem.Constant(msh, np.zeros(2, dtype=default_scalar_type))
    L_form = ufl.inner(zero, v) * ufl.dx
    problem = LinearProblem(
        a_form,
        L_form,
        bcs=bcs,
        petsc_options_prefix="notch_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    uh = problem.solve()
    assert problem.solver.getConvergedReason() > 0

    from pfwire.failure import DamageGrid, GridSpec

    def uy_jump(x_probe: float) -> float:
        """中线上下各取一点，返回 u_y 之差（用采样网格做点求值）。"""
        offset = 2.0 * spec.h_fine
        spec_grid = GridSpec(
            x_min=x_probe,
            x_max=x_probe + 1e-12,
            depth_max=2.0 * offset,
            spacing=offset,
            y_surface=y_mid + offset,
        )
        grid = DamageGrid(msh, spec_grid)
        Vy = fem.functionspace(msh, ("Lagrange", 1))
        uy = fem.Function(Vy)
        uy.interpolate(
            fem.Expression(uh[1], Vy.element.interpolation_points)
        )
        values = grid.sample(uy)  # 行 0 在 y_mid+offset，行 2 在 y_mid-offset
        return float(values[0, 0] - values[-1, 0])

    jump_notch = uy_jump(0.5 * spec.notch_length)
    jump_ligament = uy_jump(0.5 * (spec.notch_length + spec.W))
    # 均匀拉伸下 2*offset 高度对应的均匀伸长量，作为「无裂缝」参考
    uniform = u_top * (4.0 * spec.h_fine) / spec.H

    assert jump_notch > 5.0 * uniform, (
        f"缺口没张开：跳跃 {jump_notch:.3e} 仅为均匀伸长 {uniform:.3e} 的 "
        f"{jump_notch / uniform:.1f} 倍"
    )
    assert jump_ligament == pytest.approx(uniform, rel=1.0), (
        f"韧带段出现异常张开：{jump_ligament:.3e} vs 均匀 {uniform:.3e}"
    )


def test_spec_validation():
    with pytest.raises(ValueError, match="notch_length"):
        pfmesh.SentSpec(W=1.0e-3, notch_length=1.0e-3)
    with pytest.raises(ValueError, match="h_fine"):
        pfmesh.SentSpec(h_fine=1.0e-4, h_coarse=1.0e-5)
    with pytest.raises(ValueError, match="必须为有限正数"):
        pfmesh.SentSpec(W=-1.0)
    with pytest.raises(ValueError, match="加密带"):
        pfmesh.SentSpec(H=1.0e-3, refine_halfheight=1.0e-3)


# --------------------------------------------------------------------------
# CT 试样（验证 2）
# --------------------------------------------------------------------------
CT_COARSE = pfmesh.CtSpec(
    h_fine=0.4e-3, h_coarse=3.0e-3, refine_halfheight=2.0e-3
)


@pytest.fixture(scope="module")
def ct():
    return pfmesh.ct_mesh(CT_COARSE)


def test_ct_spec_matches_carrara_fig12a():
    """派生几何量与原文 Fig. 12a 的标注逐条对表。"""
    s = pfmesh.CtSpec()
    assert s.W == pytest.approx(50.0e-3)
    # 总宽 1.25W、总高 1.2W
    assert s.x_back - s.x_front == pytest.approx(1.25 * s.W)
    assert 2.0 * s.halfheight == pytest.approx(1.2 * s.W)
    # 原点在载荷线上 -> 前端面在 -0.25W，背面在 +W
    assert s.x_front == pytest.approx(-0.25 * s.W)
    assert s.x_back == pytest.approx(s.W)
    # 销孔 Ø = 0.25W、孔心距裂纹面 ±0.275W
    assert 2.0 * s.hole_radius == pytest.approx(0.25 * s.W)
    assert s.hole_offset == pytest.approx(0.275 * s.W)
    # 缺口槽高 W/16
    assert 2.0 * s.notch_half_height == pytest.approx(s.W / 16.0)
    # a0 自载荷线量起，a0/W = 0.22 -> 尖端 x 坐标就是 a0
    assert s.a0 == pytest.approx(0.22 * s.W)
    assert s.a0 == pytest.approx(11.0e-3)


def test_ct_area_bounded_by_hole_faceting(ct):
    """网格面积应略大于解析面积，差由销孔的多边形离散解释。"""
    msh = ct.mesh
    spec = ct.spec
    area = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ufl.dx)), op=MPI.SUM
    )
    # 内接多边形挖掉的面积小于圆 -> 网格面积必然偏大
    assert area > spec.area, (
        "网格面积 %.6e 不大于解析面积 %.6e —— 销孔多边形应挖得比圆少" % (area, spec.area)
    )
    # 上界：两个销孔各自最多少挖 (1 - 3/π 的量级)，这里给宽松但有意义的 1%
    assert (area - spec.area) / spec.area < 1.0e-2


def test_ct_outer_perimeter_is_exact(ct):
    """外边界全是直线段，可精确对表：矩形周长 − 缺口开口宽。"""
    msh = ct.mesh
    spec = ct.spec
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ct.facet_tags)
    L = msh.comm.allreduce(
        fem.assemble_scalar(
            fem.form(fem.Constant(msh, 1.0) * ds(pfmesh.CT_TAG_OUTER))
        ),
        op=MPI.SUM,
    )
    rect = 2.0 * (spec.x_back - spec.x_front) + 2.0 * (2.0 * spec.halfheight)
    expected = rect - 2.0 * spec.notch_half_height  # 前端面被缺口开口挖掉一段
    assert L == pytest.approx(expected, rel=1e-9)


def test_ct_hole_perimeters_match_chord_length(ct):
    """销孔周长应是内接多边形的弦长和：短于弧长，且与等弦估计同量级。

    不能对「等弦公式」2 n r sin(pi/n) 取机器精度 —— gmsh 在圆上布点并不严格均匀
    （实测差 6.5e-4），那是网格生成器的自由度，不是几何错误。
    真正的不变量是「内接 => 短于弧长」，以及半径正确时二者只差 O(1/n²)。
    半径若搭错，偏差是百分数量级，本判据照样能抓住。
    """
    msh = ct.mesh
    spec = ct.spec
    arc = 2.0 * np.pi * spec.hole_radius
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ct.facet_tags)
    for tag in (pfmesh.CT_TAG_HOLE_TOP, pfmesh.CT_TAG_HOLE_BOTTOM):
        n = msh.comm.allreduce(int(ct.facet_tags.find(tag).size), op=MPI.SUM)
        L = msh.comm.allreduce(
            fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ds(tag))), op=MPI.SUM
        )
        assert n > 8, "销孔只有 %d 个 facet，太粗" % n
        assert L < arc, "周长 %.6f 不短于弧长 %.6f —— 多边形应内接于圆" % (L, arc)
        equal_chord = 2.0 * n * spec.hole_radius * np.sin(np.pi / n)
        assert L == pytest.approx(equal_chord, rel=5.0e-3)
        # n 个弦时弦/弧比为 sin(pi/n)/(pi/n)，据此给出周长的下界
        assert L > arc * np.sin(np.pi / n) / (np.pi / n) * 0.99


def test_ct_notch_is_a_hollow_slot_and_ligament_is_connected(ct):
    """CT 缺口是**挖空的槽**：槽内中线上不该有单元；韧带段中线必须连通。"""
    msh = ct.mesh
    spec = ct.spec
    fdim = msh.topology.dim - 1
    tol = 0.2 * spec.h_fine

    def in_slot(x):
        return np.isclose(x[1], 0.0, atol=tol) & (
            x[0] < spec.a0 - spec.taper_length - tol
        )

    def in_ligament(x):
        return np.isclose(x[1], 0.0, atol=tol) & (x[0] > spec.a0 + tol)

    n_slot = msh.comm.allreduce(
        dmesh.locate_entities_boundary(msh, fdim, in_slot).size, op=MPI.SUM
    )
    n_lig = msh.comm.allreduce(
        dmesh.locate_entities_boundary(msh, fdim, in_ligament).size, op=MPI.SUM
    )
    assert n_slot == 0, (
        "槽内中线上出现了 %d 个边界 facet —— 槽没被挖空（若做成了零厚度切缝也会这样）"
        % n_slot
    )
    assert n_lig == 0, "韧带段中线上出现外边界 facet —— 试样被切断了"


def test_ct_physical_groups_present(ct):
    present = global_tag_set(ct.mesh, ct.facet_tags)
    for tag in (
        pfmesh.CT_TAG_HOLE_TOP,
        pfmesh.CT_TAG_HOLE_BOTTOM,
        pfmesh.CT_TAG_NOTCH,
        pfmesh.CT_TAG_OUTER,
    ):
        assert tag in present, "缺少物理组 %d" % tag


def test_ct_refines_the_crack_path_band(ct):
    """加密带（自缺口尖端到背面的水平带）内单元应显著小于带外。"""
    msh = ct.mesh
    spec = ct.spec
    V0 = fem.functionspace(msh, ("DG", 0))
    area = fem.Function(V0)
    area.interpolate(
        fem.Expression(ufl.CellVolume(msh), V0.element.interpolation_points)
    )
    ycen = fem.Function(V0)
    ycen.interpolate(
        fem.Expression(ufl.SpatialCoordinate(msh)[1], V0.element.interpolation_points)
    )
    xcen = fem.Function(V0)
    xcen.interpolate(
        fem.Expression(ufl.SpatialCoordinate(msh)[0], V0.element.interpolation_points)
    )
    n_local = V0.dofmap.index_map.size_local
    A = area.x.array[:n_local]
    y = ycen.x.array[:n_local]
    x = xcen.x.array[:n_local]
    inside = (np.abs(y) < 0.5 * spec.refine_halfheight) & (x > spec.a0)
    outside = np.abs(y) > 0.7 * spec.halfheight

    def mean(mask):
        tot = msh.comm.allreduce(float(A[mask].sum()), op=MPI.SUM)
        cnt = msh.comm.allreduce(int(mask.sum()), op=MPI.SUM)
        return tot / max(cnt, 1)

    mean_in, mean_out = mean(inside), mean(outside)
    assert mean_in < 0.5 * mean_out, (
        "加密带没生效：带内平均单元面积 %.3e vs 带外 %.3e" % (mean_in, mean_out)
    )
    assert np.sqrt(2.0 * mean_in) == pytest.approx(spec.h_fine, rel=0.6)


def test_ct_spec_validation():
    with pytest.raises(ValueError, match="贯穿到背面"):
        pfmesh.CtSpec(a0_over_W=1.2)
    with pytest.raises(ValueError, match="锥收长度"):
        pfmesh.CtSpec(a0_over_W=0.01)
    with pytest.raises(ValueError, match="销孔超出"):
        pfmesh.CtSpec(hole_offset_over_W=0.55)
    with pytest.raises(ValueError, match="相交"):
        pfmesh.CtSpec(notch_height_over_W=0.5)
    with pytest.raises(ValueError, match="h_fine"):
        pfmesh.CtSpec(h_fine=5.0e-3)
    with pytest.raises(ValueError, match="必须为有限正数"):
        pfmesh.CtSpec(W=-1.0)
    with pytest.raises(ValueError, match="notch_taper_length"):
        pfmesh.CtSpec(notch_taper_length=-1.0)


# --------------------------------------------------------------------------
# 微动试样（Phase 2.1）
# --------------------------------------------------------------------------
#: 测试用的缩小试样：几何逻辑与绝对尺寸无关，但保持 a 与 h_fine 的**比例**，
#: 这样接触区分辨率这条测的还是真实口径。
FRET_TEST = pfmesh.FrettingSpec(
    W=0.8e-3,
    H=0.4e-3,
    contact_halfwidth=24.2e-6,
    refine_depth=60.0e-6,
    h_fine=4.0e-6,
    h_coarse=5.0e-5,
)


@pytest.fixture(scope="module")
def fret():
    return pfmesh.fretting_mesh(FRET_TEST)


def test_fretting_area_and_edges_are_exact(fret):
    """矩形试样全是直线，面积与四条边长都应精确到机器精度。"""
    msh = fret.mesh
    spec = fret.spec
    area = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ufl.dx)), op=MPI.SUM
    )
    assert area == pytest.approx(spec.W * spec.H, rel=1e-10)

    ds = ufl.Measure("ds", domain=msh, subdomain_data=fret.facet_tags)
    one = fem.Constant(msh, 1.0)
    for tag, expected in (
        (pfmesh.FRETTING_TAG_TOP, spec.W),
        (pfmesh.FRETTING_TAG_BOTTOM, spec.W),
        (pfmesh.FRETTING_TAG_LEFT, spec.H),
        (pfmesh.FRETTING_TAG_RIGHT, spec.H),
    ):
        L = msh.comm.allreduce(
            fem.assemble_scalar(fem.form(one * ds(tag))), op=MPI.SUM
        )
        assert L == pytest.approx(expected, rel=1e-10)


def test_fretting_physical_groups_present(fret):
    present = global_tag_set(fret.mesh, fret.facet_tags)
    for tag in (
        pfmesh.FRETTING_TAG_TOP,
        pfmesh.FRETTING_TAG_BOTTOM,
        pfmesh.FRETTING_TAG_LEFT,
        pfmesh.FRETTING_TAG_RIGHT,
    ):
        assert tag in present, "缺少物理组 %d" % tag


def test_fretting_contact_zone_is_resolved(fret):
    """上表面 |x| <= a 段的 facet 数须 ≈ 2a/h_fine（D-1 的分辨率口径）。"""
    msh = fret.mesh
    spec = fret.spec
    fdim = msh.topology.dim - 1
    a = spec.contact_halfwidth
    facets = dmesh.locate_entities_boundary(
        msh,
        fdim,
        lambda x: np.isclose(x[1], 0.0, atol=0.1 * spec.h_fine)
        & (np.abs(x[0]) <= a),
    )
    n = msh.comm.allreduce(int(facets.size), op=MPI.SUM)
    expected = spec.elements_across_contact
    assert n == pytest.approx(expected, rel=0.15), (
        "接触区只有 %d 个 facet，预期约 %.0f —— 分辨不出 x ≈ ±a 的滑移区"
        % (n, expected)
    )


def test_fretting_top_surface_is_flat(fret):
    """上表面必须平整：所有 TOP facet 的节点都在 y = 0 上。

    微动裂纹要靠接触面力**自己**萌生，预置几何缺口等于把答案写进输入
    （Phase 2.4 的硬关卡就是「裂纹必须自己出现在 x ≈ ±a」）。
    """
    msh = fret.mesh
    fdim = msh.topology.dim - 1
    top = fret.facet_tags.find(pfmesh.FRETTING_TAG_TOP)
    # **不要**用 topology.connectivity(fdim, 0) 的顶点号去索引 geometry.x：
    # 拓扑顶点编号与几何节点编号只在串行下碰巧一致，并行下分区重排后就对不上，
    # 取到的是别处的坐标（实测偏差 3.4e-6，看起来像「表面不平」的假故障）。
    # facet 中点由 dolfinx 自己按几何算，没有这个坑，且直边 facet 的中点精确。
    mids = dmesh.compute_midpoints(msh, fdim, top) if top.size else np.zeros((0, 3))
    worst = float(np.abs(mids[:, 1]).max()) if mids.size else 0.0
    worst = msh.comm.allreduce(worst, op=MPI.MAX)
    assert worst < 1e-15, "TOP 上有 facet 中点偏离 y = 0 达 %.3e，上表面不是平的" % worst


def test_fretting_refinement_bands(fret):
    """加密区内 ≈ h_fine，远场 ≈ h_coarse。"""
    msh = fret.mesh
    spec = fret.spec
    V0 = fem.functionspace(msh, ("DG", 0))
    vol = fem.Function(V0)
    vol.interpolate(
        fem.Expression(ufl.CellVolume(msh), V0.element.interpolation_points)
    )
    xc = fem.Function(V0)
    xc.interpolate(
        fem.Expression(ufl.SpatialCoordinate(msh)[0], V0.element.interpolation_points)
    )
    yc = fem.Function(V0)
    yc.interpolate(
        fem.Expression(ufl.SpatialCoordinate(msh)[1], V0.element.interpolation_points)
    )
    n = V0.dofmap.index_map.size_local
    A, x, y = vol.x.array[:n], xc.x.array[:n], yc.x.array[:n]

    def mean_h(mask):
        tot = msh.comm.allreduce(float(A[mask].sum()), op=MPI.SUM)
        cnt = msh.comm.allreduce(int(mask.sum()), op=MPI.SUM)
        return np.sqrt(2.0 * tot / max(cnt, 1))

    inside = (np.abs(x) < 0.8 * spec.refine_halfwidth) & (
        y > -0.8 * spec.refine_depth
    )
    far = (np.abs(x) > 0.3 * spec.W) & (y < -0.5 * spec.H)
    h_in, h_out = mean_h(inside), mean_h(far)
    assert h_in == pytest.approx(spec.h_fine, rel=0.35)
    # 远场**不能**要求正好等于 h_coarse：从 h_fine 长到 h_coarse 需要一段
    # 有限的过渡距离（见 FrettingSpec.transition_factor 的推导），
    # 而这个缩小的测试试样还没来得及长完就到边界了。
    # 真正该测的是「分档确实生效」且远场不超过 h_coarse。
    assert h_out > 5.0 * h_in
    assert h_out <= spec.h_coarse * 1.05


def test_fretting_production_spec_meets_d1_resolution():
    """生产默认参数必须满足 D-1 的分辨率要求；这条只查 spec，不建网格。"""
    spec = pfmesh.FrettingSpec()
    assert spec.h_fine == pytest.approx(4.0e-6 / 4.0)  # = ell/4，ell = 4 µm
    assert spec.contact_halfwidth == pytest.approx(24.2e-6, rel=1e-3)
    assert spec.elements_across_contact >= 40.0, (
        "接触区只有 %.0f 个单元，D-1 要求足够分辨 x ≈ ±a"
        % spec.elements_across_contact
    )
    # 加密区要盖住滑移区并留出转向余量
    assert spec.refine_halfwidth >= 2.0 * spec.contact_halfwidth
    # 失效判据是绝对深度 100 µm，加密区必须比它深
    assert spec.refine_depth > 100.0e-6
    # St. Venant：左右边界离接触区足够远
    assert spec.W / spec.contact_halfwidth > 100.0


def test_fretting_spec_validation():
    with pytest.raises(ValueError, match="h_fine"):
        pfmesh.FrettingSpec(h_fine=1.0)
    with pytest.raises(ValueError, match="接触区分辨不出来"):
        pfmesh.FrettingSpec(contact_halfwidth=2.0e-6)
    with pytest.raises(ValueError, match="加密区深度"):
        pfmesh.FrettingSpec(refine_depth=4.0e-3)
    with pytest.raises(ValueError, match="加密区半宽"):
        pfmesh.FrettingSpec(contact_halfwidth=2.0e-3)
    with pytest.raises(ValueError, match="必须为有限正数"):
        pfmesh.FrettingSpec(W=-1.0)
