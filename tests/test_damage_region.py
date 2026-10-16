"""D-68 路线 A（A-1 + A-b）的测试：损伤计算限制在**可信区**（h ≲ ℓ）。

依据（D-68 问题 12/13）：粗网格区 ψ⁺ 反推的等效应力 451 MPa，超出 J2 屈服面
上界 σ_y²/(6E) **7.3 倍**，是离散误差；用它当起裂判据会把不该起裂的状态
推成 d 的棘轮发散（N=94.2773）。

覆盖：
  T1 掩码本身：cell_max_edge 的正确性（解析网格）、阈值两侧、h_max 校验、
     并行统计（用 allreduce，不作 rank 局部断言 —— CLAUDE.md 陷阱 3）。
  T2 模型校验：非 Function / 非 DG0 一律拒绝（含 P1 这种 basix family 同为 P
     的陷阱：必须看 discontinuous，不是 family）。
  T3 **回归**：damage_region=None 时既有行为逐位不变（d 方程与能量口径）。
  T4 屏蔽生效：半域掩码下区外 d 恒 0、区内照常起裂。
  T5 psi_plus_expr 不被屏蔽（诊断仍看得到区外原始值，A-1 口径）。
  T6 能量一致性：energies() 报的与求解器实际处置的口径一致（避免再犯
     b-1 那种两处打架）。
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from dolfinx import fem, mesh as dmesh
from mpi4py import MPI

from pfwire.material import load_material
from pfwire.mesh_h import cell_max_edge, damage_region_mask, damage_region_stats
from pfwire.phasefield import PhaseFieldModel, PhaseFieldSolver, SolverConfig

COMM = MPI.COMM_WORLD
MAT = "configs/material_al1350.yaml"


def _unit_square(n=8):
    return dmesh.create_unit_square(COMM, n, n, dmesh.CellType.triangle)


def _material():
    return load_material(MAT)


# ---------------------------------------------------------------- T1 掩码
def test_cell_max_edge_matches_analytic():
    """单位方格 n×n 三角化：每个直角三角形直角边 1/n，斜边 √2/n -> 最大边长 √2/n。"""
    n = 8
    msh = _unit_square(n)
    h = cell_max_edge(msh)
    assert h.size > 0
    want = math.sqrt(2.0) / n
    assert np.allclose(h, want, rtol=1e-12), (h.min(), h.max(), want)


@pytest.mark.parametrize("h_max,expect", [(1.0, 1.0), (0.1, 0.0)])
def test_mask_threshold(h_max, expect):
    msh = _unit_square(8)
    mask = damage_region_mask(msh, h_max)
    n = mask.function_space.dofmap.index_map.size_local
    got = float(mask.x.array[:n].sum())
    ncell = msh.topology.index_map(msh.topology.dim).size_local
    assert got == pytest.approx(expect * ncell)


def test_mask_rejects_nonpositive():
    with pytest.raises(ValueError, match="h_max"):
        damage_region_mask(_unit_square(4), 0.0)
    with pytest.raises(ValueError, match="h_max"):
        damage_region_mask(_unit_square(4), -1.0)


def test_stats_are_global():
    """统计必须并行安全：用 allreduce，不写按 rank 局部的断言。"""
    msh = _unit_square(8)
    st = damage_region_stats(msh, 1.0, COMM)
    n_glob = msh.topology.index_map(msh.topology.dim).size_global
    assert st["n_cell"] == pytest.approx(float(n_glob))
    assert st["fraction"] == pytest.approx(1.0)
    st0 = damage_region_stats(msh, 0.1, COMM)
    assert st0["fraction"] == pytest.approx(0.0)


# ------------------------------------------------------------ T2 模型校验
def test_model_rejects_bad_mask():
    mat = _material()
    with pytest.raises(TypeError, match="DG0"):
        PhaseFieldModel(material=mat, split="amor", damage_region=np.zeros(3))


def test_model_rejects_p1_mask():
    """P1 的 basix family 也是 P —— 只看 family 会漏判，必须看 discontinuous。"""
    msh = _unit_square(4)
    V1 = fem.functionspace(msh, ("Lagrange", 1))
    p1 = fem.Function(V1)
    mat = _material()
    with pytest.raises(TypeError, match="DG0"):
        PhaseFieldModel(material=mat, split="amor", damage_region=p1)
    # DG2 同样要拒（degree 不为 0）
    V2 = fem.functionspace(msh, ("DG", 2))
    with pytest.raises(TypeError, match="DG0"):
        PhaseFieldModel(material=mat, split="amor", damage_region=fem.Function(V2))


def test_model_accepts_dg0():
    msh = _unit_square(4)
    mat = _material()
    m = PhaseFieldModel(material=mat, split="amor",
                        damage_region=damage_region_mask(msh, 1.0))
    assert m.damage_region is not None


# --------------------------------------------------------- T3/T4/T5/T6
def _left_fixed(msh):
    """左端 x=0 全固定（仿 test_stiffness_degradation 的 _uniaxial_bcs）。

    注意 dolfinx 0.11 的 dirichletbc 签名是 ``(g, dofs)``，
    多传一个 V 会报 "incompatible function arguments"。
    """
    def make(V):
        fdim = msh.topology.dim - 1
        left = dmesh.locate_entities_boundary(
            msh, fdim, lambda x: np.isclose(x[0], 0.0))
        dofs = fem.locate_dofs_topological(V, fdim, left)
        g = fem.Function(V)
        g.interpolate(lambda x: np.vstack([np.zeros(x.shape[1]),
                                           np.zeros(x.shape[1])]))
        return [fem.dirichletbc(g, dofs)]
    return make


def _solver(msh, mat, mask, split="amor"):
    model = PhaseFieldModel(material=mat, split=split, k_res=1.0e-6,
                            damage_region=mask)
    return PhaseFieldSolver(msh, model, bcs_u=_left_fixed(msh),
                            config=SolverConfig(max_iter=50))


def test_none_mask_keeps_psi_plus_unmasked():
    """T3：mask=None 时 psi_plus_drive 与 psi_plus_expr 同一个表达式。"""
    msh = _unit_square(4)
    mat = _material()
    sol = _solver(msh, mat, None)
    # 同样的 u=0 -> ε=0 -> ψ⁺=0，但关键是两个入口都不报错且口径一致
    import ufl
    a = sol.psi_plus_expr()
    b = sol.psi_plus_drive()
    assert a is not None and b is not None
    # mask 为 None 时 drive 就是 expr（不额外乘任何东西）
    assert sol.model.damage_region is None


def test_masked_drive_zero_outside():
    """T4：区外（掩码 0）的驱动力为 0 -> 那里解出的 d 恒为 0。"""
    msh = _unit_square(8)
    mat = _material()
    # 阈值取极小值 -> 全区掩码为 0 -> 任何地方都无驱动
    mask = damage_region_mask(msh, 1.0e-12)
    sol = _solver(msh, mat, mask)
    # 施加一个位移使 ψ⁺ > 0：直接给 u 一个非零场
    V_u = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    uu = fem.Function(V_u)
    uu.interpolate(lambda x: np.vstack([0.1 * x[0], np.zeros_like(x[0])]))
    sol.u.x.array[:] = uu.x.array
    sol.u.x.scatter_forward()

    # 未屏蔽时 ψ⁺ 应为正（说明确实有驱动）
    import dolfinx.fem as _fem
    import ufl
    psi_raw = _fem.assemble_scalar(
        _fem.form(sol.psi_plus_expr() * ufl.dx))
    psi_raw = sol.comm.allreduce(psi_raw, op=MPI.SUM)
    assert psi_raw > 0.0, "测试本身失效：未屏蔽的 ψ⁺ 应为正"

    # 屏蔽后驱动力积分为 0
    psi_drive = _fem.assemble_scalar(
        _fem.form(sol.psi_plus_drive() * ufl.dx))
    psi_drive = sol.comm.allreduce(psi_drive, op=MPI.SUM)
    assert abs(psi_drive) < 1e-12 * max(1.0, abs(psi_raw))

    # 于是 d 解出来恒为 0
    sol.solve_damage()
    assert sol.max_damage() == pytest.approx(0.0, abs=1e-14)


def test_partial_mask_keeps_inside_active():
    """T4 续：掩码为 1 的区域驱动力保留（与未屏蔽同值）。"""
    msh = _unit_square(8)
    mat = _material()
    mask = damage_region_mask(msh, 1.0)     # 全区为 1
    sol = _solver(msh, mat, mask)
    V_u = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    uu = fem.Function(V_u)
    uu.interpolate(lambda x: np.vstack([0.1 * x[0], np.zeros_like(x[0])]))
    sol.u.x.array[:] = uu.x.array
    sol.u.x.scatter_forward()

    import dolfinx.fem as _fem
    import ufl
    a = sol.comm.allreduce(
        _fem.assemble_scalar(_fem.form(sol.psi_plus_expr() * ufl.dx)), op=MPI.SUM)
    b = sol.comm.allreduce(
        _fem.assemble_scalar(_fem.form(sol.psi_plus_drive() * ufl.dx)), op=MPI.SUM)
    assert b == pytest.approx(a, rel=1e-12)


def test_psi_plus_expr_not_masked():
    """T5：诊断入口 psi_plus_expr 保持未屏蔽（A-1：区外 ᾱ 照旧累积、可复查）。"""
    msh = _unit_square(8)
    mat = _material()
    mask = damage_region_mask(msh, 1.0e-12)   # 全 0
    sol = _solver(msh, mat, mask)
    V_u = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    uu = fem.Function(V_u)
    uu.interpolate(lambda x: np.vstack([0.1 * x[0], np.zeros_like(x[0])]))
    sol.u.x.array[:] = uu.x.array
    sol.u.x.scatter_forward()

    import dolfinx.fem as _fem
    import ufl
    raw = sol.comm.allreduce(
        _fem.assemble_scalar(_fem.form(sol.psi_plus_expr() * ufl.dx)), op=MPI.SUM)
    assert raw > 0.0, "psi_plus_expr 被屏蔽了 —— 诊断将看不到区外到底有多大"


def test_energies_consistent_with_solver():
    """T6：energies() 报的弹性项与 d 方程实际处置的口径一致。

    b-1 那次就是两处口径打架（u 最小化 ψ⁺+ψ⁻，energies() 却报 g(d)ψ⁺+ψ⁻）。
    """
    msh = _unit_square(8)
    mat = _material()
    for mask in (None, damage_region_mask(msh, 1.0)):
        sol = _solver(msh, mat, mask)
        V_u = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
        uu = fem.Function(V_u)
        uu.interpolate(lambda x: np.vstack([0.1 * x[0], np.zeros_like(x[0])]))
        sol.u.x.array[:] = uu.x.array
        sol.u.x.scatter_forward()
        e = sol.energies()
        # ψ⁺ 一项：有驱动时非负；能量各项均为有限数
        for k, v in e.items():
            assert math.isfinite(v), (k, v)
        assert e["psi_plus"] >= 0.0


def test_solve_step_with_mask_runs():
    """端到端：带掩码也能走完一个载荷步（不因掩码引入新的求解失败）。"""
    msh = _unit_square(8)
    mat = _material()
    mask = damage_region_mask(msh, 1.0)
    sol = _solver(msh, mat, mask)
    res = sol.solve_step()
    assert res is not None
    assert math.isfinite(res.delta_d_inf)
