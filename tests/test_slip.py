"""pfwire.slip 的单元测试（合成数据，不跑 FE）。"""
import numpy as np
import pytest

from pfwire import slip as sl

R = 8.0


def _arcs(n=81, xmax=1.5):
    """上弧（圆心 (0,0)，底部在 y=-R）与下弧（圆心 (0,-2R)，顶部在 y=-R），参考构形。"""
    x = np.linspace(-xmax, xmax, n)
    up = np.stack([x, -np.sqrt(R**2 - x**2)], axis=1)
    lo = np.stack([x, -2 * R + np.sqrt(R**2 - x**2)], axis=1)
    return up, lo


def test_arc_tangent_unit_and_flat_at_centre():
    _, lo = _arcs()
    tau = sl.arc_tangent(lo, (0.0, -2 * R))
    assert np.allclose(np.linalg.norm(tau, axis=1), 1.0)
    assert np.allclose(tau[len(lo) // 2], (1.0, 0.0), atol=1e-12)


def test_rigid_relative_translation_gives_delta_everywhere():
    up, lo = _arcs()
    tau = sl.arc_tangent(lo, (0.0, -2 * R))
    zeros = np.zeros_like(up)
    s = sl.pair_material_points(up, zeros, lo, zeros)
    g0 = sl.relative_tangential_displacement(up, zeros, lo, zeros, s, tau)
    delta = 0.01
    u_up = np.tile([delta, 0.0], (len(up), 1))
    inc = sl.slip_increment(sl.relative_tangential_displacement(up, u_up, lo, zeros, s, tau), g0)
    # u_x = δ 沿 τ 的分量是 δ cosθ；|x|<=1.5, R=8 -> cosθ >= 0.982
    assert np.all(inc > 0.98 * delta) and np.all(inc <= delta + 1e-15)


def test_baseline_subtraction_zero_at_baseline():
    up, lo = _arcs()
    tau = sl.arc_tangent(lo, (0.0, -2 * R))
    rng = np.random.default_rng(0)
    u_up, u_lo = rng.normal(0, 1e-3, up.shape), rng.normal(0, 1e-3, lo.shape)
    s = sl.pair_material_points(up, u_up, lo, u_lo)
    g = sl.relative_tangential_displacement(up, u_up, lo, u_lo, s, tau)
    inc = sl.slip_increment(g, g)
    ok = np.isfinite(s)
    assert ok.sum() > 0.9 * len(s)                    # 噪声可能让端点配不上（nan），其余必须恰为 0
    assert np.all(inc[ok] == 0.0) and np.all(np.isnan(inc[~ok]))


def test_pairing_matches_current_x():
    up, lo = _arcs()
    u_up = np.stack([0.03 + 0.01 * up[:, 0], np.zeros(len(up))], axis=1)   # 基线时上弧已有 u_x
    s = sl.pair_material_points(up, u_up, lo, np.zeros_like(lo))
    ok = np.isfinite(s)
    assert ok.sum() > 0.9 * len(s)
    # s + u_up_x(s) = X_lo_x（u_up_x 对 x 线性，插值精确）
    assert np.allclose(s[ok] + 0.03 + 0.01 * s[ok], lo[ok, 0], atol=1e-12)


def test_pairing_out_of_range_is_nan():
    up, lo = _arcs()
    u_up = np.tile([0.5, 0.0], (len(up), 1))          # 上弧整体右移 0.5，下弧左端无对应点
    s = sl.pair_material_points(up, u_up, lo, np.zeros_like(lo))
    assert np.isnan(s[0]) and np.isfinite(s[-1])
    assert np.all(np.isnan(s) == (lo[:, 0] < up[0, 0] + 0.5))


def test_tangent_projection_of_normal_relative_motion():
    """纯法向（y 向）相对位移 Δu_y 经倾斜的 τ 投影，应得 τ_y·Δu_y = ∓sinθ·Δu_y，量级 x/R·Δu_y。"""
    up, lo = _arcs()
    tau = sl.arc_tangent(lo, (0.0, -2 * R))
    zeros = np.zeros_like(up)
    s = sl.pair_material_points(up, zeros, lo, zeros)
    dy = 0.02
    g = sl.relative_tangential_displacement(up, np.tile([0.0, dy], (len(up), 1)), lo, zeros, s, tau)
    assert np.allclose(g, tau[:, 1] * dy, atol=1e-15)
    assert abs(g[-1]) == pytest.approx(dy * 1.5 / R, rel=0.02)
    assert abs(g[len(g) // 2]) < 1e-12                 # x=0 处切向水平，法向位移不进滑移


def _profile(x, c, k=0.05):
    """中间黏着 |x|<c，外侧 |slip| 线性增。"""
    return np.where(np.abs(x) < c, 0.0, k * (np.abs(x) - c) * np.sign(x))


def test_classify_stick_zone_and_monotone():
    x = np.linspace(-1, 1, 201)
    r = sl.classify_slip(x, _profile(x, 0.4), stick_rel=0.05)
    assert r.stick_lo == pytest.approx(-0.4, abs=0.05) and r.stick_hi == pytest.approx(0.4, abs=0.05)
    assert r.monotone_outward and r.n_slip > 0 and r.n_stick > 0


def test_classify_detects_non_monotone():
    x = np.linspace(-1, 1, 201)
    s = _profile(x, 0.4)
    s[(x > 0.7) & (x < 0.8)] *= 0.3                    # 滑移区内出现凹坑
    assert not sl.classify_slip(x, s, stick_rel=0.05).monotone_outward


def test_classify_mirror_symmetry():
    x = np.linspace(-1, 1, 201)
    s = _profile(x, 0.4)
    a = sl.classify_slip(x, s, stick_rel=0.05)
    b = sl.classify_slip(x, -s[::-1], stick_rel=0.05)
    assert (a.stick_lo, a.stick_hi) == pytest.approx((b.stick_lo, b.stick_hi))
    assert a.monotone_outward == b.monotone_outward


def test_classify_gross_sliding_has_no_stick():
    x = np.linspace(-1, 1, 101)
    r = sl.classify_slip(x, np.full_like(x, 0.01), stick_rel=0.05)
    assert r.n_stick == 0 and np.isnan(r.stick_lo)
