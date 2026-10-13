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
from dolfinx import fem
from dolfinx import mesh as dmesh
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
