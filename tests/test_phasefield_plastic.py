"""PhaseFieldSolver 与塑性的耦合（D-61 设计，D-62/D-63 的 B 口径）。

测试意图：
  1. 弹性极限：屈服应力极大时，带塑性的解与无塑性解一致（ε_e = ε）；
  2. 均匀仿射场（全边界 Dirichlet，应力场处处相同）：提交后的 ε_p、p 与
     return_map(g = (1−d)² + k_res) 一致，d = 0 与 d = 0.5 两种情形；
  3. 历史只在收敛后提交：solve 之后、commit_step 之前 ε_p 保持为零；
  4. 冲突组合被拒绝：塑性 + linear 报错；cycle-jump 拒绝带塑性的求解器。
"""

import pathlib

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem, assemble_matrix, assemble_vector
from mpi4py import MPI

from pfwire.cyclejump import CycleJumpDriver
from pfwire.material import load_material
from pfwire.phasefield import PhaseFieldModel, PhaseFieldSolver, SolverConfig
from pfwire.plasticity import J2Params, PlasticLaw, load_j2_params, return_map

ROOT = pathlib.Path(__file__).resolve().parents[1]
MATERIAL = ROOT / "configs/material_al1350.yaml"
K_RES = 1.0e-6


def _affine_bcs(msh, exx, eyy):
    """全部边界上施加仿射位移 (exx·x, eyy·y)：均匀应变的精确离散解（无体力，应力均匀）。"""

    def make(V):
        fdim = msh.topology.dim - 1
        facets = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.full(x.shape[1], True))
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        g = fem.Function(V)
        g.interpolate(lambda x: np.vstack([exx * x[0], eyy * x[1]]))
        return [fem.dirichletbc(g, dofs)]

    return make


def _build(exx=2.0e-2, eyy=-5.0e-3, prm=None, plastic=True, u_solver="newton", split="spectral"):
    material = load_material(MATERIAL)
    prm = prm or load_j2_params(MATERIAL)
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 2, 2, dmesh.CellType.triangle)
    model = PhaseFieldModel(material=material, split=split, k_res=K_RES, dissipation="AT1")
    law = PlasticLaw(msh, prm) if plastic else None
    solver = PhaseFieldSolver(
        msh, model, bcs_u=_affine_bcs(msh, exx, eyy),
        config=SolverConfig(u_solver=u_solver), plastic=law,
    )
    return msh, prm, law, solver


def test_elastic_limit_matches_no_plasticity():
    """屈服应力极大：ε_e = ε，带塑性的 u 与无塑性的 u 一致。"""
    base = load_j2_params(MATERIAL)
    huge = J2Params(E=base.E, nu=base.nu, sigma_y0=1.0e15, H=base.H)
    _, _, _, s_pl = _build(exx=1.0e-4, eyy=-3.0e-5, prm=huge, plastic=True)
    _, _, _, s_el = _build(exx=1.0e-4, eyy=-3.0e-5, plastic=False)
    s_pl.solve_displacement()
    s_el.solve_displacement()
    assert np.allclose(s_pl.u.x.array, s_el.u.x.array, rtol=1e-10, atol=1e-16)


@pytest.mark.parametrize("d_value", [0.0, 0.5])
def test_homogeneous_state_commits_return_map_with_degradation(d_value):
    _, prm, law, solver = _build()
    exx, eyy = 2.0e-2, -5.0e-3
    solver.d.x.array[:] = d_value
    solver.solve_displacement()
    # 收敛但未提交：历史仍为零
    assert np.allclose(law.ep_n.x.array, 0.0)

    g = (1.0 - d_value) ** 2 + K_RES
    r = return_map(np.diag([exx, eyy, 0.0]), np.zeros((3, 3)), 0.0, prm, g=g)
    assert r.plastic, "测试状态应当屈服，否则不构成耦合检验"

    solver.commit_step()
    ep = law.ep_n.x.array.reshape(-1, 3)
    p = law.p_n.x.array
    scale = prm.sigma_y0 / prm.E
    assert np.allclose(ep[:, 0], r.eps_p[0, 0], rtol=1e-6, atol=1e-9 * scale)
    assert np.allclose(ep[:, 1], r.eps_p[1, 1], rtol=1e-6, atol=1e-9 * scale)
    assert np.allclose(ep[:, 2], r.eps_p[0, 1], rtol=1e-6, atol=1e-9 * scale)
    assert np.allclose(p, r.p, rtol=1e-6)


def test_plastic_with_linear_u_solver_is_refused():
    with pytest.raises(ValueError):
        _build(u_solver="linear", split="amor")


def test_cycle_jump_refuses_plastic_solver():
    _, _, _, solver = _build()
    with pytest.raises(NotImplementedError):
        CycleJumpDriver(solver, None, lambda phase: None)


def _edge_bcs(msh, ux_right):
    """左边全固定，右边 u_x = ux_right、u_y = 0：屈服区非均匀（夹持端附近）。"""

    def make(V):
        fdim = msh.topology.dim - 1
        left = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
        right = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
        g_left = fem.Function(V)
        g_right = fem.Function(V)
        g_right.interpolate(lambda x: np.vstack([ux_right * np.ones(x.shape[1]), np.zeros(x.shape[1])]))
        return [
            fem.dirichletbc(g_left, fem.locate_dofs_topological(V, fdim, left)),
            fem.dirichletbc(g_right, fem.locate_dofs_topological(V, fdim, right)),
        ]

    return make


def test_coupled_u_matches_standard_plastic_fe_on_nonuniform_field():
    """T-a2（D-65）：d ≡ 0 时，耦合求解器的 u 与标准塑性 FE（残差 ∫ σ:sym∇v）一致。

    二者唯一差别是 g = 1 + k_res：σ_deg = (1+k_res)σ⁺ + σ⁻，与标准 σ = σ⁺ + σ⁻ 差 O(k_res)。
    容差 1e-5（相对）；屈服效应在 u 上是 O(1) 量级，不会被这个容差掩盖。
    """
    prm = load_j2_params(MATERIAL)
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 4, 4, dmesh.CellType.triangle)
    bcs = _edge_bcs(msh, 6.0e-3)
    model = PhaseFieldModel(material=load_material(MATERIAL), split="spectral", k_res=K_RES)
    tight = SolverConfig(u_solver="newton", snes_rtol=1.0e-10, snes_atol=1.0e-14)
    solver = PhaseFieldSolver(msh, model, bcs_u=bcs, config=tight, plastic=PlasticLaw(msh, prm))
    solver.solve_displacement()

    law_ref = PlasticLaw(msh, prm)
    V = solver.V_u
    u_ref = fem.Function(V)
    v = ufl.TestFunction(V)
    F = ufl.inner(law_ref.sigma(u_ref), ufl.sym(ufl.grad(v))) * solver.dx
    prob = NonlinearProblem(
        F, u_ref, bcs=bcs(V), petsc_options_prefix="std_plastic_",
        petsc_options={
            "snes_type": "newtonls", "snes_linesearch_type": "bt",
            "snes_atol": 1.0e-14, "snes_rtol": 1.0e-10, "snes_max_it": 50,
            **tight.direct_solver_options,
        },
    )
    prob.solve()
    law_ref.commit(u_ref)
    assert np.max(np.abs(law_ref.ep_n.x.array)) > 0.0, "测试状态应当含屈服点"
    err = np.max(np.abs(solver.u.x.array - u_ref.x.array)) / np.max(np.abs(u_ref.x.array))
    assert err < 1e-5


def test_coupled_tangent_matches_finite_difference_of_residual():
    """T-a4（D-65）：AD 得到的 Jacobian 与残差的有限差分一致（屈服状态下）。"""
    prm = load_j2_params(MATERIAL)
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 4, 4, dmesh.CellType.triangle)
    model = PhaseFieldModel(material=load_material(MATERIAL), split="spectral", k_res=K_RES)
    tight = SolverConfig(u_solver="newton", snes_rtol=1.0e-10, snes_atol=1.0e-14)
    solver = PhaseFieldSolver(
        msh, model, bcs_u=_edge_bcs(msh, 6.0e-3), config=tight, plastic=PlasticLaw(msh, prm),
    )
    solver.solve_displacement()

    V = solver.V_u
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    F = solver._residual_u_form(v)
    F_form = fem.form(F)
    J_form = fem.form(ufl.derivative(F, solver.u, du))

    rng = np.random.default_rng(17)
    delta = fem.Function(V)
    delta.x.array[:] = rng.normal(size=delta.x.array.size)
    delta.x.scatter_forward()
    u0 = solver.u.x.array.copy()
    h = 1.0e-7 * np.max(np.abs(u0))

    def residual_at(sign):
        solver.u.x.array[:] = u0 + sign * h * delta.x.array
        solver.u.x.scatter_forward()
        return assemble_vector(F_form).array.copy()

    fd = (residual_at(+1.0) - residual_at(-1.0)) / (2.0 * h)
    solver.u.x.array[:] = u0
    solver.u.x.scatter_forward()
    Jm = assemble_matrix(J_form)
    Jm.assemble()
    out = Jm.createVecRight()
    Jm.mult(delta.x.petsc_vec, out)
    an = out.array.copy()
    assert np.linalg.norm(fd - an) < 1e-5 * np.linalg.norm(an)
