"""Phase 5.1/5.2 的 FE 层测试：UFL 返回映射 vs NumPy 参考、AD 切线、单元 Newton 收敛。"""
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

from pfwire.plasticity import J2Params, PlasticLaw, consistent_tangent, return_map, stress_update_ufl

PRM = J2Params(E=69.0e9, nu=0.33, sigma_y0=167.0e6, H=5.0e9)
EPS_Y = PRM.sigma_y0 / PRM.E


def _mesh(n=2, cell=dmesh.CellType.triangle):
    return dmesh.create_unit_square(MPI.COMM_WORLD, n, n, cell)


def _embed(e2):
    out = np.zeros((3, 3))
    out[:2, :2] = e2
    return out


def _embed_ep(ep):
    return np.array([[ep[0], ep[2], 0], [ep[2], ep[1], 0], [0, 0, -(ep[0] + ep[1])]])


def _eval_dg0(msh, expr, n_comp):
    """在 DG0 上对 UFL 表达式插值（每个单元一个值），返回 (n_cells, n_comp)。"""
    W = fem.functionspace(msh, ("DG", 0, (n_comp,)))
    f = fem.Function(W)
    f.interpolate(fem.Expression(expr, W.element.interpolation_points))
    return f.x.array.reshape(-1, n_comp)


@pytest.mark.parametrize("cell", [dmesh.CellType.triangle, dmesh.CellType.quadrilateral])
def test_ufl_update_matches_numpy_at_random_points(cell):
    """随机位移场 + 随机历史：UFL 的 sigma、eps_p_new、p_new 与 NumPy return_map 逐点一致。"""
    msh = _mesh(2, cell)
    law = PlasticLaw(msh, PRM)
    rng = np.random.default_rng(3)
    V = fem.functionspace(msh, ("Lagrange", 2, (2,)))
    u = fem.Function(V)
    u.x.array[:] = rng.normal(size=u.x.array.size) * EPS_Y * 10.0 ** rng.uniform(-2.0, 0.0, u.x.array.size)
    law.ep_n.x.array[:] = rng.normal(size=law.ep_n.x.array.size) * 0.3 * EPS_Y
    law.p_n.x.array[:] = np.abs(rng.normal(size=law.p_n.x.array.size)) * 0.2 * EPS_Y
    upd = law.update(u)
    eps2 = law.strain(u)
    W4 = fem.functionspace(msh, ("DG", 2, (4,)))

    def interp(expr, W):
        f = fem.Function(W)
        f.interpolate(fem.Expression(expr, W.element.interpolation_points))
        return f.x.array

    n_dof = law.p_n.x.array.size
    sig = interp(ufl.as_vector([upd.sigma[0, 0], upd.sigma[1, 1], upd.sigma[0, 1], upd.sigma3[2, 2]]), W4).reshape(n_dof, 4)
    eps = interp(ufl.as_vector([eps2[0, 0], eps2[1, 1], eps2[0, 1], 0.0]), W4).reshape(n_dof, 4)
    ep_new = interp(upd.ep_new, law.V_ep).reshape(n_dof, 3)
    p_new = interp(upd.p_new, law.V_p)
    epo = law.ep_n.x.array.reshape(n_dof, 3)
    p_old = law.p_n.x.array
    n_plastic = 0
    scale = PRM.sigma_y0
    for i in range(n_dof):
        e = np.array([[eps[i, 0], eps[i, 2], 0], [eps[i, 2], eps[i, 1], 0], [0, 0, 0.0]])
        r = return_map(e, _embed_ep(epo[i]), p_old[i], PRM)
        n_plastic += r.plastic
        assert abs(sig[i, 0] - r.sigma[0, 0]) < 1e-9 * scale
        assert abs(sig[i, 1] - r.sigma[1, 1]) < 1e-9 * scale
        assert abs(sig[i, 2] - r.sigma[0, 1]) < 1e-9 * scale
        assert abs(sig[i, 3] - r.sigma[2, 2]) < 1e-9 * scale
        assert abs(ep_new[i, 0] - r.eps_p[0, 0]) < 1e-12
        assert abs(ep_new[i, 1] - r.eps_p[1, 1]) < 1e-12
        assert abs(ep_new[i, 2] - r.eps_p[0, 1]) < 1e-12
        assert abs(p_new[i] - r.p) < 1e-12
    assert 0 < n_plastic < n_dof          # 两种状态（弹性/塑性）都被覆盖


@pytest.mark.parametrize("scale,label", [(0.4, "elastic"), (3.0, "plastic")])
def test_ad_tangent_matches_analytic_consistent_tangent(scale, label):
    """UFL 自动微分得到的切线 = 解析一致切线（平面内分量）。"""
    msh = _mesh(1)
    x0 = fem.Function(fem.functionspace(msh, ("DG", 0)))      # 零函数：让表达式带上网格 / 通信子
    eps = np.array([[1.0, 0.3], [0.3, -0.5]]) * scale * EPS_Y
    ep = np.array([0.2, -0.1, 0.05]) * EPS_Y
    pn = 0.3 * EPS_Y
    eps_var = ufl.variable(ufl.as_tensor(eps.tolist()) + x0 * ufl.Identity(2))
    upd = stress_update_ufl(eps_var, ufl.as_vector(list(ep)), pn, PRM)
    C = consistent_tangent(_embed(eps), _embed_ep(ep), pn, PRM)
    for (i, j), tag in (((0, 0), "xx"), ((1, 1), "yy"), ((0, 1), "xy")):
        d = ufl.diff(upd.sigma[i, j], eps_var)           # d sigma_ij / d eps_kl（2x2）
        got = _eval_dg0(msh, ufl.as_vector([d[0, 0], d[1, 1], d[0, 1] + d[1, 0]]), 3)[0]
        # ufl.diff 对对称张量的 off-diagonal 视作独立分量：d[0,1] + d[1,0] 才是对「对称扰动」的响应
        assert got[0] == pytest.approx(C[i, j, 0, 0], rel=1e-8), (label, tag, "xx")
        assert got[1] == pytest.approx(C[i, j, 1, 1], rel=1e-8), (label, tag, "yy")
        assert got[2] == pytest.approx(2.0 * C[i, j, 0, 1], rel=1e-8), (label, tag, "xy")


def _uniaxial_tension_problem(n=1):
    """单位正方形，P2；左边 ux=0、下边 uy=0、右边 ux=δ；上边自由（平面应变）。"""
    msh = _mesh(n, dmesh.CellType.quadrilateral)
    V = fem.functionspace(msh, ("Lagrange", 2, (2,)))
    u = fem.Function(V)
    law = PlasticLaw(msh, PRM)
    v = ufl.TestFunction(V)
    F = ufl.inner(law.sigma(u), law.strain(v)) * law.dx
    fdim = msh.topology.dim - 1
    left = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    right = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    bottom = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    g_right = fem.Constant(msh, np.float64(0.0))
    zero = fem.Constant(msh, np.float64(0.0))
    bcs = [
        fem.dirichletbc(zero, fem.locate_dofs_topological(V.sub(0), fdim, left), V.sub(0)),
        fem.dirichletbc(g_right, fem.locate_dofs_topological(V.sub(0), fdim, right), V.sub(0)),
        fem.dirichletbc(zero, fem.locate_dofs_topological(V.sub(1), fdim, bottom), V.sub(1)),
    ]
    prob = NonlinearProblem(F, u, bcs=bcs, petsc_options_prefix="pl_test_",
                            petsc_options={"snes_type": "newtonls", "snes_linesearch_type": "basic",
                                           "snes_atol": 1e-6, "snes_rtol": 1e-14, "snes_max_it": 20,
                                           "ksp_type": "preonly", "pc_type": "lu"})
    return msh, u, law, g_right, prob


def test_newton_converges_in_few_iterations_with_consistent_tangent():
    """每个载荷步（含屈服后）都收敛且 Newton 迭代数 <= 5。
    一致切线若有误（例如误用弹性切线），这里会退化成线性收敛、迭代数明显更多。"""
    msh, u, law, g_right, prob = _uniaxial_tension_problem()
    iters = []
    for k in range(1, 9):
        g_right.value = k * EPS_Y
        prob.solve()
        snes = prob.solver
        assert snes.getConvergedReason() > 0, "第 %d 步 Newton 未收敛" % k
        iters.append(snes.getIterationNumber())
        law.commit(u)
    assert max(iters) <= 5, iters
    assert law.p_n.x.array.max() > 0.0                      # 确实进过塑性


def test_uniform_state_matches_numpy_return_map_through_unload_reverse():
    """均匀场：FE 与 NumPy 沿同一应变历史（加载-卸载-反向）累积的 (eps_p, p)、应力一致。"""
    msh, u, law, g_right, prob = _uniaxial_tension_problem()
    eps_p_np, p_np = np.zeros((3, 3)), 0.0
    for g in list(np.linspace(0.75, 6.0, 8)) + [3.0, 0.0, -3.0, -6.0]:
        g_right.value = g * EPS_Y
        prob.solve()
        assert prob.solver.getConvergedReason() > 0
        e2 = _eval_dg0(msh, ufl.as_vector([law.strain(u)[0, 0], law.strain(u)[1, 1], law.strain(u)[0, 1]]), 3)[0]
        eps2 = np.array([[e2[0], e2[2]], [e2[2], e2[1]]])
        r = return_map(_embed(eps2), eps_p_np, p_np, PRM)
        eps_p_np, p_np = r.eps_p, r.p
        law.commit(u)
        # 提交后 FE 历史 = NumPy 历史（单元内均匀）
        ep_fe = law.ep_n.x.array.reshape(-1, 3)
        assert np.allclose(ep_fe[:, 0], eps_p_np[0, 0], atol=1e-12)
        assert np.allclose(ep_fe[:, 1], eps_p_np[1, 1], atol=1e-12)
        assert np.allclose(law.p_n.x.array, p_np, atol=1e-12)
        # 提交后的应力是对已更新历史的弹性应力 = r.sigma
        s = _eval_dg0(msh, ufl.as_vector([law.sigma(u)[0, 0], law.sigma(u)[1, 1], law.sigma(u)[0, 1]]), 3)[0]
        assert abs(s[0] - r.sigma[0, 0]) < 1e-6 * PRM.sigma_y0, g
        assert abs(s[1] - r.sigma[1, 1]) < 1e-6 * PRM.sigma_y0, g
