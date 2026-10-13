"""6.5 的线夹接触（D-67）：ClampContact 与相场求解器的耦合。

测试意图：
  T1. 切线：接触 + 相场（d ≡ 0）+ 塑性，屈服状态下 Jacobian 与残差的有限差分一致（含接触项）；
  T2. 与 6.1 交叉验证：弹性、无相场，同一 δ 下，用 ClampContact 装配的位移场与 ObstacleContact 一致（1e-8）；
      反力也一致。两边的边界条件与材料完全相同（底面全固定）。

网格为小尺寸的加密矩形（10 mm 长），只为测试速度；几何约定与 6.1 相同（上表面 y = 0、底面 y = −depth）。
"""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

import pathlib

from pfwire.clamp_contact import ClampContact
from pfwire.material import load_material
from pfwire.mesh import BLOCK_BOTTOM
from pfwire.obstacle import build_block_mesh
from pfwire.obstacle_contact import ObstacleContact, plane_strain_lame
from pfwire.phasefield import PhaseFieldModel, PhaseFieldSolver, SolverConfig
from pfwire.plasticity import PlasticLaw, load_j2_params

ROOT = pathlib.Path(__file__).resolve().parents[1]
MATERIAL = ROOT / "configs/material_al1350.yaml"
K_RES = 1.0e-6
KAPPA = 10.0
R_CLAMP = 330.0e-3
DELTA = 5.0e-6          # 线夹下压量：使接触区内的压力达到屈服量级
SOLVER = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_rtol": 1.0e-10, "snes_atol": 1.0e-10, "snes_max_it": 60,
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
}


def _mesh(tmp_path):
    msh, _, facet_tags = build_block_mesh(
        tmp_path, width=10.0e-3, depth=3.75e-3, h_fine=0.1e-3, h_coarse=0.5e-3,
        fine_halfwidth=2.0e-3, fine_depth=2.0e-3, transition=0.5e-3, comm=MPI.COMM_WORLD,
    )
    return msh, facet_tags


def _bottom_fixed(facet_tags, msh):
    """底面全固定（去刚体位移）。BC 建在求解器自己的空间上（CLAUDE.md 陷阱 1）。"""

    def make(V):
        fdim = msh.topology.dim - 1
        dofs = fem.locate_dofs_topological(V, fdim, facet_tags.find(BLOCK_BOTTOM))
        zero = fem.Function(V)
        return [fem.dirichletbc(zero, dofs)]

    return make


def test_clamp_contact_matches_obstacle_contact_elastic(tmp_path):
    """T2：弹性、无相场，ClampContact 装配的位移与 ObstacleContact 一致。"""
    msh, facet_tags = _mesh(tmp_path)
    E, nu = 69.0e9, 0.33
    h = 0.1e-3

    obst = ObstacleContact(msh, facet_tags, E=E, nu=nu, R_clamp=R_CLAMP, kappa=KAPPA,
                           h_contact=h, solver=SOLVER)
    obst.step(DELTA)
    w_ref = obst.reaction()

    clamp = ClampContact(msh, facet_tags, E=E, R_clamp=R_CLAMP, kappa=KAPPA, h_contact=h)
    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    mu, lmbda = plane_strain_lame(E, nu)
    eps = ufl.sym(ufl.grad(u))
    sig = 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(2)
    dx = ufl.Measure("dx", domain=msh)
    F = ufl.inner(sig, ufl.sym(ufl.grad(v))) * dx - clamp.residual(u, v)
    bcs = _bottom_fixed(facet_tags, msh)(V)
    problem = NonlinearProblem(F, u, bcs=bcs, petsc_options_prefix="t2_clamp_",
                               petsc_options=dict(SOLVER))
    clamp.set_delta(DELTA)
    problem.solve()

    assert problem.solver.getConvergedReason() > 0
    err = np.max(np.abs(u.x.array - obst.u.x.array)) / np.max(np.abs(obst.u.x.array))
    assert err < 1e-8
    assert clamp.reaction(u) == pytest.approx(w_ref, rel=1e-8)
    assert w_ref > 0.0, "测试状态应当有接触反力"


def test_coupled_tangent_with_contact_matches_finite_difference(tmp_path):
    """T1：接触 + 相场（d ≡ 0）+ 塑性：屈服状态下 Jacobian 与残差的有限差分一致。

    δ = 20 µm 使接触区应力越过 σ_y0（dgamma_max ~ 4.5e-3），塑性返回映射真正激活，
    切线的塑性分支被覆盖。差分步长 h = 1e-11（方向归一化后）是扫描出的甜点：
    再大则接触/屈服折点穿越的 O(h) 截断进来（罚刚度 ~1e16 放大），再小则差分
    相消到舍入地板（见 D-67）。
    """
    msh, facet_tags = _mesh(tmp_path)
    material = load_material(MATERIAL)
    prm = load_j2_params(MATERIAL)
    model = PhaseFieldModel(material=material, split="spectral", k_res=K_RES)
    clamp = ClampContact(msh, facet_tags, E=material.elastic.E, R_clamp=R_CLAMP, kappa=KAPPA,
                         h_contact=0.1e-3)
    cfg = SolverConfig(u_solver="newton", snes_rtol=1.0e-10, snes_atol=1.0e-14)
    solver = PhaseFieldSolver(msh, model, bcs_u=_bottom_fixed(facet_tags, msh), config=cfg,
                              plastic=PlasticLaw(msh, prm), contact=clamp)
    clamp.set_delta(20.0e-6)
    solver.solve_displacement()

    # 屈服确实激活（否则塑性切线分支没被覆盖，测试名不副实）
    dg_fn = fem.Function(fem.functionspace(msh, ("DG", 0)))
    upd = solver.plastic.update(solver.u, g=1.0)
    dg_fn.interpolate(fem.Expression(upd.dgamma,
                                      dg_fn.function_space.element.interpolation_points))
    dg_max = msh.comm.allreduce(float(np.max(dg_fn.x.array)), op=MPI.SUM)
    assert dg_max > 1.0e-4, "测试状态应当已屈服（dgamma > 1e-4），实际 %r" % (dg_max,)
    assert clamp.reaction(solver.u) > 0.0, "测试状态应当有接触"

    V = solver.V_u
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    F = solver._residual_u_form(v)
    F_form = fem.form(F)
    J_form = fem.form(ufl.derivative(F, solver.u, du))

    rng = np.random.default_rng(23)
    delta_vec = fem.Function(V)
    delta_vec.x.array[:] = rng.normal(size=delta_vec.x.array.size)
    delta_vec.x.scatter_forward()
    delta_vec.x.array[:] /= np.max(np.abs(delta_vec.x.array))  # 归一：方向最大分量 = 1
    u0 = solver.u.x.array.copy()
    h = 1.0e-11

    from dolfinx.fem.petsc import assemble_matrix, assemble_vector

    def residual_at(sign):
        solver.u.x.array[:] = u0 + sign * h * delta_vec.x.array
        solver.u.x.scatter_forward()
        return assemble_vector(F_form).array.copy()

    fd = (residual_at(+1.0) - residual_at(-1.0)) / (2.0 * h)
    solver.u.x.array[:] = u0
    solver.u.x.scatter_forward()
    Jm = assemble_matrix(J_form)
    Jm.assemble()
    out = Jm.createVecRight()
    Jm.mult(delta_vec.x.petsc_vec, out)
    an = out.array.copy()
    assert np.linalg.norm(fd - an) < 1e-5 * np.linalg.norm(an)
