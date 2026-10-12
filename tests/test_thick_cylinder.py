"""Phase 5.4：厚壁圆筒解析解的自洽性 + 粗网格 FE 对拍（完整两网格、两工况见 scripts/verify_plastic_cylinder.py）。"""
import importlib.util
import pathlib

import numpy as np
import pytest

from pfwire import thick_cylinder as tc

A, B, SY = 1.0, 2.0, 1.0


def test_yield_onset_and_limit_pressures():
    assert tc.pressure_from_c(A, SY, A, B) == pytest.approx(tc.yield_onset_pressure(SY, A, B), rel=1e-14)
    assert tc.pressure_from_c(B, SY, A, B) == pytest.approx(tc.limit_pressure(SY, A, B), rel=1e-14)
    assert tc.yield_onset_pressure(SY, A, B) == pytest.approx(0.75 / np.sqrt(3.0), rel=1e-14)


def test_c_from_pressure_inverts_pressure_from_c_and_is_monotone():
    cs = np.linspace(1.01, 1.99, 25)
    ps = np.array([tc.pressure_from_c(c, SY, A, B) for c in cs])
    assert np.all(np.diff(ps) > 0)
    for c, p in zip(cs, ps):
        assert tc.c_from_pressure(p, SY, A, B) == pytest.approx(c, rel=1e-10)
    assert tc.c_from_pressure(0.1, SY, A, B) == A                  # 全弹性
    with pytest.raises(ValueError):
        tc.c_from_pressure(tc.limit_pressure(SY, A, B) * 1.001, SY, A, B)


@pytest.mark.parametrize("p", [0.2, 0.5, 0.65, 0.78])
def test_stress_field_boundary_conditions_continuity_and_equilibrium(p):
    sr_a, _ = tc.plastic_stress(np.array([A]), p, SY, A, B)
    sr_b, _ = tc.plastic_stress(np.array([B]), p, SY, A, B)
    assert sr_a[0] == pytest.approx(-p, rel=1e-12) and sr_b[0] == pytest.approx(0.0, abs=1e-12)
    c = tc.c_from_pressure(p, SY, A, B)
    eps = 1e-9
    sl, tl = tc.plastic_stress(np.array([c - eps]), p, SY, A, B)
    sr_, tr_ = tc.plastic_stress(np.array([c + eps]), p, SY, A, B)
    assert sl[0] == pytest.approx(sr_[0], abs=1e-7) and tl[0] == pytest.approx(tr_[0], abs=1e-7)  # 界面连续
    # 平衡 d(sr)/dr = (st - sr)/r（中心差分，避开 c 处的分段点）
    r = np.linspace(A + 0.05, B - 0.05, 41)
    r = r[np.abs(r - c) > 0.02]
    h = 1e-6
    sp, _ = tc.plastic_stress(r + h, p, SY, A, B)
    sm, _ = tc.plastic_stress(r - h, p, SY, A, B)
    s0, t0 = tc.plastic_stress(r, p, SY, A, B)
    assert np.allclose((sp - sm) / (2 * h), (t0 - s0) / r, atol=1e-5)


def test_plastic_zone_satisfies_mises_with_plane_strain_axial_stress():
    """塑性区 σθ − σr = 2σy/√3（σz = (σr+σθ)/2 时 Mises 屈服的等价形式）。"""
    p = 0.6
    c = tc.c_from_pressure(p, SY, A, B)
    r = np.linspace(A, c, 20)
    sr, st = tc.plastic_stress(r, p, SY, A, B)
    sz = 0.5 * (sr + st)
    seq = np.sqrt(0.5 * ((sr - st) ** 2 + (st - sz) ** 2 + (sz - sr) ** 2))
    assert np.allclose(seq, SY, rtol=1e-12)


def test_lame_u_inner_incompressible_limit_matches_plastic_formula_at_yield():
    """nu -> 1/2 的 Lamé u_r(a) 在 p = p_y 处应等于弹塑性公式（c = a）。"""
    E = 3.0e0
    p_y = tc.yield_onset_pressure(SY, A, B)
    G = E / 3.0
    assert tc.lame_u_inner(p_y, E, 0.5 - 1e-12, A, B) == pytest.approx(
        tc.plastic_u_inner(p_y, SY, G, A, B), rel=1e-9)


def _load_script():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "verify_plastic_cylinder.py"
    spec = importlib.util.spec_from_file_location("verify_plastic_cylinder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fe_coarse_mesh_matches_analytic_solution():
    """12x20 粗网格、nu = 0.499：起屈压力、u_r(a)、边界 c、偏量应力对解析解（宽于完整脚本的阈值）。"""
    pytest.importorskip("dolfinx")
    from mpi4py import MPI
    from pfwire.verify_contact_common import load_config
    V = _load_script()
    cfg = load_config(pathlib.Path(V.__file__).resolve().parents[1] / "configs" / "verify_plastic_cylinder.yaml")
    res = V.run_level(cfg, 12, 20, MPI.COMM_WORLD, lambda *_: None, nu=0.499)
    py = tc.yield_onset_pressure(res["sy"], res["a"], res["b"])
    assert abs(res["p_y_fe"] - py) / py < 0.02
    assert max(res["iters"]) <= 40
    for L, snap in res["snaps"].items():
        e = V.errors_at(L, snap, res["a"], res["b"], res["E"], res["nu"], res["sy"])
        assert e["u_err"] < 3e-3
        assert e["dev_l2"] < 1e-2
        if e["kind"] == "plastic":
            assert e["c_err"] < 0.05
