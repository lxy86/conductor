"""损伤退化屈服（D-62，B 口径）的塑性返回映射：pfwire.plasticity 的 g 参数。

屈服函数 F = g(d) · sqrt(3/2)|s| − (σ_y0 + H p)，屈服按**实际（退化后）**应力判断。

测试意图：
  1. g = 1 与旧调用（不传 g）逐位一致（回归：无损伤路径不变）；
  2. g < 1：返回态落在退化屈服面上，F = 0 数值成立；
  3. 屈服阈值：未退化应力须达 σ_y/g 才屈服。这是 D-62 修正后的含义——
     损伤**不是**让材料更易屈服，而是把未退化应力空间里的屈服面外扩；
  4. g = 0：实际应力恒为零，永不屈服；
  5. g 越界报错；
  6. 一致切线（g < 1）与返回映射的中心差分一致；
  7. UFL 返回映射（g 为 DG2 场）与 NumPy 逐点一致。
"""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pfwire.plasticity import J2Params, PlasticLaw, consistent_tangent, return_map

PRM = J2Params(E=69.0e9, nu=0.33, sigma_y0=167.0e6, H=3.0e9)
ZERO = np.zeros((3, 3))


def _dev(a):
    return a - np.trace(a) / 3.0 * np.eye(3)


def _eqv(sig):
    return np.sqrt(1.5) * np.linalg.norm(_dev(sig))


def _eps_for_eq(sig_eq_tr):
    """ε = c·diag(1, −1, 0) 给出的未退化试探 σ_eq,tr = 2√3 G c，据此反解 c。"""
    c = sig_eq_tr / (2.0 * np.sqrt(3.0) * PRM.G)
    return np.diag([c, -c, 0.0])


def test_g1_is_bitwise_the_undegraded_map():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.normal(size=(3, 3)) * 3e-3
        eps = 0.5 * (a + a.T)
        ep = np.zeros((3, 3))
        ep[:2, :2] = 1e-4 * rng.normal(size=(2, 2))
        ep = 0.5 * (ep + ep.T)
        p = abs(rng.normal()) * 1e-4
        r0 = return_map(eps, ep, p, PRM)
        r1 = return_map(eps, ep, p, PRM, g=1.0)
        assert np.array_equal(r0.sigma, r1.sigma)
        assert np.array_equal(r0.eps_p, r1.eps_p)
        assert r0.p == r1.p and r0.plastic == r1.plastic


@pytest.mark.parametrize("g", [0.9, 0.5, 0.2])
def test_degraded_return_lands_on_degraded_yield(g):
    # 取未退化应力为 2σ_y/g，使退化后的实际应力仍超过屈服，确保发生塑性
    r = return_map(_eps_for_eq(2.0 * PRM.sigma_y0 / g), ZERO, 0.0, PRM, g=g)
    assert r.plastic
    F = g * _eqv(r.sigma) - (PRM.sigma_y0 + PRM.H * r.p)
    assert abs(F) < 1e-9 * PRM.sigma_y0


@pytest.mark.parametrize("g", [0.9, 0.5])
def test_yield_threshold_is_on_undegraded_stress_sigma_y_over_g(g):
    """D-62 修正：未退化 σ_eq,tr 须超过 σ_y/g 才屈服（不是 σ_y）。"""
    target = PRM.sigma_y0 / g
    below = return_map(_eps_for_eq(target * (1.0 - 1e-4)), ZERO, 0.0, PRM, g=g)
    above = return_map(_eps_for_eq(target * (1.0 + 1e-4)), ZERO, 0.0, PRM, g=g)
    assert not below.plastic
    assert above.plastic


def test_fully_degraded_never_yields():
    """g = 0：实际应力恒为零，F = −(σ_y + H p) < 0，任何未退化应力都不屈服。"""
    r = return_map(_eps_for_eq(1.0e10), ZERO, 0.0, PRM, g=0.0)
    assert not r.plastic
    assert np.allclose(r.eps_p, 0.0)


@pytest.mark.parametrize("g", [-0.1, 1.5])
def test_degradation_out_of_range_is_rejected(g):
    with pytest.raises(ValueError):
        return_map(_eps_for_eq(2.0 * PRM.sigma_y0), ZERO, 0.0, PRM, g=g)
    with pytest.raises(ValueError):
        consistent_tangent(_eps_for_eq(2.0 * PRM.sigma_y0), ZERO, 0.0, PRM, g=g)


@pytest.mark.parametrize("g", [0.5, 0.2])
def test_consistent_tangent_matches_central_difference_degraded(g):
    eps0 = _eps_for_eq(1.5 * PRM.sigma_y0 / g)
    r0 = return_map(eps0, ZERO, 0.0, PRM, g=g)
    assert r0.plastic
    rng = np.random.default_rng(7)
    d = rng.normal(size=(3, 3))
    d = 0.5 * (d + d.T)
    d /= np.linalg.norm(d)
    h = 1e-9
    sp = return_map(eps0 + h * d, ZERO, 0.0, PRM, g=g).sigma
    sm = return_map(eps0 - h * d, ZERO, 0.0, PRM, g=g).sigma
    fd = (sp - sm) / (2.0 * h)
    C = consistent_tangent(eps0, ZERO, 0.0, PRM, g=g)
    an = np.einsum("ijkl,kl->ij", C, d)
    assert np.linalg.norm(fd - an) < 1e-5 * np.linalg.norm(an)


def _eval_at_dg2_nodes(msh, expr, n_comp, degree):
    W = fem.functionspace(msh, ("DG", degree, (n_comp,)))
    f = fem.Function(W)
    f.interpolate(fem.Expression(expr, W.element.interpolation_points))
    return f.x.array.reshape(-1, n_comp)


def test_ufl_update_with_degradation_matches_numpy_pointwise():
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 2, 2, dmesh.CellType.triangle)
    law = PlasticLaw(msh, PRM)
    rng = np.random.default_rng(11)
    V = fem.functionspace(msh, ("Lagrange", 2, (2,)))
    u = fem.Function(V)
    u.x.array[:] = rng.normal(size=u.x.array.size) * 2e-3
    law.ep_n.x.array[:] = rng.normal(size=law.ep_n.x.array.size) * 1e-4
    law.p_n.x.array[:] = np.abs(rng.normal(size=law.p_n.x.array.size)) * 1e-4
    g = fem.Function(law.V_p)
    g.x.array[:] = rng.uniform(0.2, 1.0, size=g.x.array.size)

    upd = law.update(u, g=g)
    eps2 = law.strain(u)
    n_dof = law.p_n.x.array.size
    sig = _eval_at_dg2_nodes(msh, ufl.as_vector([upd.sigma[0, 0], upd.sigma[1, 1], upd.sigma[0, 1], upd.sigma3[2, 2]]), 4, 2)
    eps = _eval_at_dg2_nodes(msh, ufl.as_vector([eps2[0, 0], eps2[1, 1], eps2[0, 1], 0.0]), 4, 2)
    ep_new = _eval_at_dg2_nodes(msh, upd.ep_new, 3, 2)
    p_new = _eval_at_dg2_nodes(msh, upd.p_new, 1, 2)[:, 0]
    epo = law.ep_n.x.array.reshape(n_dof, 3)
    n_plastic = 0
    for i in range(n_dof):
        e = np.array([[eps[i, 0], eps[i, 2], 0], [eps[i, 2], eps[i, 1], 0], [0, 0, 0.0]])
        ep = np.array([[epo[i, 0], epo[i, 2], 0], [epo[i, 2], epo[i, 1], 0],
                       [0, 0, -(epo[i, 0] + epo[i, 1])]])
        r = return_map(e, ep, law.p_n.x.array[i], PRM, g=g.x.array[i])
        n_plastic += r.plastic
        assert abs(sig[i, 0] - r.sigma[0, 0]) < 1e-9 * PRM.sigma_y0
        assert abs(sig[i, 1] - r.sigma[1, 1]) < 1e-9 * PRM.sigma_y0
        assert abs(sig[i, 2] - r.sigma[0, 1]) < 1e-9 * PRM.sigma_y0
        assert abs(ep_new[i, 0] - r.eps_p[0, 0]) < 1e-12
        assert abs(ep_new[i, 2] - r.eps_p[0, 1]) < 1e-12
        assert abs(p_new[i] - r.p) < 1e-12
    assert 0 < n_plastic < n_dof
