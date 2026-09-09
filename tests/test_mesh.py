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


def test_physical_groups_present(sent):
    tags = sent.facet_tags
    assert tags is not None
    present = set(np.unique(tags.values).tolist())
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
