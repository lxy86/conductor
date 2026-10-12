"""Phase 5.2：J2 各向同性硬化的逐点测试（NumPy 参考实现）。

单轴路径与**手算**的分段线性应力–应变关系逐点对拍；其余测试核对返回映射的结构性质
（一致性条件、塑性不可压、耗散非负、一致切线 vs 中心差分）。
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from pfwire.plasticity import J2Params, consistent_tangent, return_map

PRM = J2Params(E=69.0e9, nu=0.33, sigma_y0=167.0e6, H=5.0e9)    # 合成参数，H 取大以便区分硬化段
Z = np.zeros((3, 3))


def _uniaxial_state(eps_axial, eps_p, p, prm):
    """单轴应力：解侧向应变 e 使 sigma_22 = sigma_33 = 0。返回 (sigma_11, eps_p, p)。"""
    def lateral(e):
        eps = np.diag([eps_axial, e, e])
        return return_map(eps, eps_p, p, prm).sigma[1, 1]

    # sigma_22 对 e 单调增（正刚度）；残余塑性应变下侧向应变可远大于当前轴向应变，括号取宽
    L = 0.1
    e = brentq(lateral, -L, L, xtol=1e-16, rtol=1e-14)
    r = return_map(np.diag([eps_axial, e, e]), eps_p, p, prm)
    return r.sigma[0, 0], r.eps_p, r.p


def _hand_uniaxial(eps, prm, eps_a):
    """手算：拉到 eps_a -> 卸载 -> 反向加载的 sigma(eps)。分段线性，eps 为轴向总应变。"""
    E, H, sy = prm.E, prm.H, prm.sigma_y0
    Et = E * H / (E + H)
    eps_y = sy / E
    s_max = sy + Et * (eps_a - eps_y)               # 拉伸峰值应力
    eps_ry = eps_a - 2.0 * s_max / E                # 反向屈服（各向同性硬化：|sigma| = s_max）
    return s_max, eps_ry, Et


def test_uniaxial_tension_unload_reverse_matches_hand_path():
    prm = PRM
    eps_a, eps_end = 0.010, -0.012
    s_max, eps_ry, Et = _hand_uniaxial(None, prm, eps_a)
    load = np.linspace(0, eps_a, 41)
    unload = np.linspace(eps_a, eps_end, 89)[1:]
    eps_p, p = Z.copy(), 0.0
    E, sy = prm.E, prm.sigma_y0
    worst = 0.0
    for phase, seq in (("load", load), ("unload", unload)):
        for e in seq:
            s, eps_p, p = _uniaxial_state(e, eps_p, p, prm)
            if phase == "load":
                hand = E * e if e <= sy / E else sy + Et * (e - sy / E)
            elif e >= eps_ry:                             # 卸载弹性段
                hand = s_max - E * (eps_a - e)
            else:                                         # 反向硬化段
                hand = -s_max + Et * (e - eps_ry)
            worst = max(worst, abs(s - hand) / sy)
    assert worst < 1e-9, worst


def test_uniaxial_reverse_yield_is_symmetric_expansion():
    """各向同性硬化：反向屈服应力的绝对值 = 拉伸峰值应力。"""
    prm = PRM
    eps_a = 0.010
    s_max, eps_ry, _ = _hand_uniaxial(None, prm, eps_a)
    eps_p, p = Z.copy(), 0.0
    for e in np.linspace(0, eps_a, 21):
        s, eps_p, p = _uniaxial_state(e, eps_p, p, prm)
    assert s == pytest.approx(s_max, rel=1e-10)
    s, eps_p2, p2 = _uniaxial_state(eps_ry + 1e-7, eps_p, p, prm)   # 刚好未反向屈服
    assert s == pytest.approx(-s_max, rel=1e-4) and p2 == pytest.approx(p, abs=1e-14)
    s, eps_p3, p3 = _uniaxial_state(eps_ry - 1e-4, eps_p, p, prm)   # 过了反向屈服
    assert p3 > p and abs(s) > s_max


def test_pure_shear_yields_at_sigma_y_over_sqrt3():
    prm = PRM
    G, tau_y = prm.G, prm.sigma_y0 / np.sqrt(3.0)
    gamma_y = tau_y / G                                 # 工程剪应变 = 2 eps_xy
    def eps_shear(g):
        e = np.zeros((3, 3)); e[0, 1] = e[1, 0] = g / 2.0
        return e
    below = return_map(eps_shear(0.999 * gamma_y), Z, 0.0, prm)
    above = return_map(eps_shear(1.001 * gamma_y), Z, 0.0, prm)
    assert not below.plastic and below.sigma[0, 1] == pytest.approx(0.999 * tau_y, rel=1e-12)
    assert above.plastic and above.p > 0.0


def test_perfect_plasticity_caps_equivalent_stress():
    prm = J2Params(E=69e9, nu=0.33, sigma_y0=167e6, H=0.0)
    for scale in (2.0, 5.0, 50.0):
        eps = scale * prm.sigma_y0 / prm.E * np.array([[1.0, 0.3, 0.0], [0.3, -0.4, 0.2], [0.0, 0.2, -0.2]])
        r = return_map(eps, Z, 0.0, prm)
        s = r.sigma - np.trace(r.sigma) / 3.0 * np.eye(3)
        assert np.sqrt(1.5) * np.linalg.norm(s) == pytest.approx(prm.sigma_y0, rel=1e-12)


def _random_strain(rng, scale):
    a = rng.normal(size=(3, 3))
    return scale * prm_eps_y() * 0.5 * (a + a.T)


def prm_eps_y():
    return PRM.sigma_y0 / PRM.E


def test_consistency_condition_and_incompressible_plastic_flow():
    rng = np.random.default_rng(1)
    for _ in range(50):
        eps = _random_strain(rng, rng.uniform(1.5, 8.0))
        r = return_map(eps, Z, 0.0, PRM)
        assert r.plastic
        s = r.sigma - np.trace(r.sigma) / 3.0 * np.eye(3)
        f = np.sqrt(1.5) * np.linalg.norm(s) - (PRM.sigma_y0 + PRM.H * r.p)
        assert abs(f) < 1e-12 * PRM.sigma_y0                 # f = 0
        assert abs(np.trace(r.eps_p)) < 1e-15                # tr eps_p = 0


def test_p_monotone_and_dissipation_nonnegative_along_random_history():
    rng = np.random.default_rng(2)
    eps_p, p = Z.copy(), 0.0
    for _ in range(300):
        eps = _random_strain(rng, rng.uniform(0.0, 6.0))
        r = return_map(eps, eps_p, p, PRM)
        assert r.p >= p
        # 耗散 sigma : d(eps_p) - (sigma_y0 + H p_mid)... 取 sigma : d_eps_p >= 0
        assert np.sum(r.sigma * (r.eps_p - eps_p)) >= -1e-9 * PRM.sigma_y0 * prm_eps_y()
        eps_p, p = r.eps_p, r.p


def test_elastic_step_returns_state_unchanged_and_hooke():
    eps = 0.5 * prm_eps_y() * np.array([[1.0, 0.2, 0.0], [0.2, -0.3, 0.0], [0.0, 0.0, -0.3]])
    ep0 = 1e-4 * np.diag([1.0, -0.5, -0.5])
    r = return_map(eps, ep0, 0.002, J2Params(E=69e9, nu=0.33, sigma_y0=1e12, H=1e9))
    assert not r.plastic and r.dgamma == 0.0 and r.p == 0.002
    assert np.array_equal(r.eps_p, ep0)
    ee = eps - ep0
    hooke = PRM.lam * np.trace(ee) * np.eye(3) + 2 * PRM.G * ee
    assert np.allclose(r.sigma, hooke, rtol=1e-12, atol=1e-3)


def test_plane_strain_embedding_keeps_axial_stress_and_plastic_strain():
    """eps_zz = 0 的平面应变加载：sigma_zz != 0，且 eps_e,zz = -eps_p,zz。"""
    eps = np.zeros((3, 3)); eps[0, 0], eps[1, 1], eps[0, 1], eps[1, 0] = 0.004, -0.002, 0.0015, 0.0015
    r = return_map(eps, Z, 0.0, PRM)
    assert r.plastic and abs(r.sigma[2, 2]) > 1e5
    ee_zz = eps[2, 2] - r.eps_p[2, 2]
    assert ee_zz == pytest.approx(-r.eps_p[2, 2], abs=1e-18)
    assert r.eps_p[2, 2] != 0.0                          # zz 塑性应变不为 0（迹为零）


@pytest.mark.parametrize("scale", [0.4, 2.5, 7.0])         # 弹性 / 塑性 / 深塑性
def test_consistent_tangent_matches_central_difference(scale):
    rng = np.random.default_rng(int(10 * scale))
    eps = _random_strain(rng, scale)
    eps_p = 0.3 * _random_strain(rng, 1.0)
    eps_p = eps_p - np.trace(eps_p) / 3.0 * np.eye(3)
    p_n = 0.0008
    C = consistent_tangent(eps, eps_p, p_n, PRM)
    h = 1e-9
    for _ in range(5):
        d = rng.normal(size=(3, 3)); d = 0.5 * (d + d.T)
        fd = (return_map(eps + h * d, eps_p, p_n, PRM).sigma
              - return_map(eps - h * d, eps_p, p_n, PRM).sigma) / (2 * h)
        an = np.einsum("ijkl,kl->ij", C, d)
        assert np.linalg.norm(an - fd) <= 1e-6 * np.linalg.norm(an)


def test_tangent_is_symmetric_positive_definite_in_plastic_state():
    eps = 4.0 * prm_eps_y() * np.diag([1.0, -0.3, -0.3])
    C = consistent_tangent(eps, Z, 0.0, PRM)
    M = C.reshape(9, 9)
    sym = 0.5 * (M + M.T)
    assert np.allclose(M, M.T, rtol=1e-10, atol=1e-3)     # 关联流动 + 各向同性硬化 -> 对称
    # 对称 9x9 在反对称空间里有零特征值；只看对称应变子空间
    w = np.linalg.eigvalsh(sym)
    assert np.sum(w > 1e-6 * w.max()) == 6                # 6 个正特征值（对称应变空间），其余为 0


@pytest.mark.parametrize("kwargs", [dict(E=-1.0), dict(nu=0.5), dict(nu=-1.0), dict(sigma_y0=0.0), dict(H=-1.0)])
def test_params_validation(kwargs):
    base = dict(E=69e9, nu=0.33, sigma_y0=167e6, H=1e9)
    base.update(kwargs)
    with pytest.raises(ValueError):
        J2Params(**base)


def test_load_j2_params_from_material_yaml():
    import pathlib
    from pfwire.plasticity import load_j2_params
    path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "material_al1350.yaml"
    prm = load_j2_params(path)
    assert prm.E == 69.0e9 and prm.nu == 0.33
    assert prm.sigma_y0 == 167.0e6                    # D-50
    assert 2.0e9 <= prm.H <= 6.0e9                    # D-53 的派生估计区间


def test_load_j2_params_rejects_missing_H(tmp_path):
    import pathlib
    import yaml
    from pfwire.plasticity import load_j2_params
    src = pathlib.Path(__file__).resolve().parents[1] / "configs" / "material_al1350.yaml"
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    raw["plastic"]["H"] = None
    bad = tmp_path / "m.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_j2_params(bad)
