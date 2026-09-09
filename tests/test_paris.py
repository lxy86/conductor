"""paris.py 单测：几何因子与 Paris 拟合。

测试意图
--------
1. **几何因子对表**：``ct_geometric_factor`` 与 ASTM E647 标定式的公开取值逐点比对
   （Y 在 γ→1 时发散，故也测边界拒绝）；
2. **合成幂律必须精确回收**：喂一条由给定 C、m 解析构造的 a(N)，拟合必须还原
   出那对 C、m。这条最关键 —— 它把「拟合器写错」与「物理算错」彻底分开：
   端到端跑出来的 m 不对时，有了这条才能断定问题在模型而不在后处理；
3. **单位换算**：原文 Tab. 2 用 mm/cycle–MPa√m，SI 口径的 C 与之差 1e3·1e6^m，
   搞错会差十几个数量级，故显式测住；
4. **中间三分之一窗口**：给一条「前段萌生慢、中段幂律、末段失稳陡」的三段曲线，
   默认窗口必须只取中段、拟合出中段的 m；若整段回归则 m 明显偏大。
   这是原文 Appendix B 的口径，也是 m 最容易被系统性高估的地方；
5. 输入校验：a 不可回退、长度不一致、delta_a 过大导致点数不足等。
"""

from __future__ import annotations

import numpy as np
import pytest

from pfwire import paris

# Carrara §5.2 / Tab. 2 的 CT 算例参数（SI）
W = 50.0e-3
T = 1.0e-3
DELTA_P = 20.0


# --------------------------------------------------------------------------
# 1. 几何因子
# --------------------------------------------------------------------------
def test_ct_geometric_factor_matches_astm_e647():
    """与 ASTM E647 标定式直接展开的值比对（同一多项式，逐点核算）。"""
    for g in (0.22, 0.30, 0.40, 0.50, 0.60, 0.70):
        poly = 0.886 + 4.64 * g - 13.32 * g**2 + 14.72 * g**3 - 5.6 * g**4
        expected = (2.0 + g) / (1.0 - g) ** 1.5 * poly
        assert paris.ct_geometric_factor(g) == pytest.approx(expected, rel=1e-12)


def test_ct_geometric_factor_is_increasing_and_diverges():
    """Y(γ) 随裂纹增长单调上升（故定 ΔP 下 ΔK 单调上升），且 γ→1 时发散。"""
    g = np.linspace(0.2, 0.9, 40)
    Y = paris.ct_geometric_factor(g)
    assert np.all(np.diff(Y) > 0.0)
    assert paris.ct_geometric_factor(0.99) > 100.0 * paris.ct_geometric_factor(0.22)


def test_ct_geometric_factor_rejects_out_of_range():
    for bad in (0.0, -0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="gamma"):
            paris.ct_geometric_factor(bad)


def test_tpb_geometric_factor_matches_astm_e1820():
    s_over_w = 4.0
    for g in (0.30, 0.45, 0.60):
        bracket = 1.99 - g * (1.0 - g) * (2.15 - 3.93 * g + 2.7 * g**2)
        expected = (
            3.0 * s_over_w * g / (2.0 * (1.0 + 2.0 * g) * (1.0 - g) ** 1.5) * bracket
        )
        assert paris.tpb_geometric_factor(g, span_over_W=s_over_w) == pytest.approx(
            expected, rel=1e-12
        )


# --------------------------------------------------------------------------
# 2-3. 合成幂律的精确回收
# --------------------------------------------------------------------------
def synthetic_a_of_N(C: float, m: float, a0: float, a_end: float, n: int = 4000):
    """由 da/dN = C (ΔK)^m 反解 N(a)：N(a) = ∫ da / (C ΔK(a)^m)。

    ΔK(a) = ΔP/(T√W) Y(a/W) 已知，故用梯形法积分即可得到任意精度的 a(N)。
    """
    a = np.linspace(a0, a_end, n)
    dK = DELTA_P / (T * np.sqrt(W)) * paris.ct_geometric_factor(a / W)
    integrand = 1.0 / (C * dK**m)
    N = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(a))])
    return a, N


@pytest.mark.parametrize("C_true,m_true", [(1.0e-30, 3.42), (5.0e-27, 3.0), (2.0e-33, 4.0)])
def test_paris_fit_recovers_synthetic_power_law(C_true, m_true):
    """**核心测试**：解析构造的幂律必须被精确回收。

    这条通过后，端到端算例的 m 不对就只能是模型/求解的问题，不是后处理。
    """
    a, N = synthetic_a_of_N(C_true, m_true, 0.22 * W, 0.60 * W)
    fit = paris.paris_fit(a, N, W=W, T=T, delta_P=DELTA_P)
    assert fit.m == pytest.approx(m_true, rel=2e-3)
    assert fit.C == pytest.approx(C_true, rel=5e-2)
    assert fit.r_squared > 0.9999
    assert fit.n_points_used >= 2


def test_paris_fit_unit_conversion_to_paper_convention():
    """C_mm_MPa 必须等于 C·1e3·1e6^m（mm/cycle–MPa√m 口径）。"""
    C_true, m_true = 1.0e-30, 3.42
    a, N = synthetic_a_of_N(C_true, m_true, 0.22 * W, 0.60 * W)
    fit = paris.paris_fit(a, N, W=W, T=T, delta_P=DELTA_P)
    assert fit.C_mm_MPa == pytest.approx(fit.C * 1.0e3 * (1.0e6**fit.m), rel=1e-12)
    # 用换算后的 C、m 直接算一个点，应与 SI 口径算的同一个点一致
    dK_MPa = 1.5
    rate_mm = fit.C_mm_MPa * dK_MPa**fit.m
    rate_si = fit.C * (dK_MPa * 1.0e6) ** fit.m
    assert rate_mm == pytest.approx(rate_si * 1.0e3, rel=1e-9)


# --------------------------------------------------------------------------
# 4. 中间三分之一窗口
# --------------------------------------------------------------------------
def test_fit_window_isolates_the_stable_branch():
    """三段曲线（萌生慢 / 幂律 / 失稳陡）下，默认窗口须只拟合中段。"""
    m_mid = 3.42
    C_mid = 1.0e-30
    a, N = synthetic_a_of_N(C_mid, m_mid, 0.22 * W, 0.60 * W)

    # 人为在两端加进「非 Paris」的行为：前 15% 放慢 5 倍、后 15% 加快 20 倍。
    # 做法是改各段的 dN 再重新累加，保证 N 仍单调。
    dN = np.diff(N)
    n = dN.size
    head, tail = int(0.15 * n), int(0.85 * n)
    dN[:head] *= 5.0
    dN[tail:] /= 20.0
    N_mod = np.concatenate([[0.0], np.cumsum(dN)])

    fit = paris.paris_fit(a, N_mod, W=W, T=T, delta_P=DELTA_P)
    full = paris.paris_fit(a, N_mod, W=W, T=T, delta_P=DELTA_P, window=(0.0, 1.0))
    # 中段窗口应还原真实斜率；整段回归会被两端污染
    assert fit.m == pytest.approx(m_mid, rel=0.05)
    assert abs(full.m - m_mid) > abs(fit.m - m_mid), (
        "整段回归 m=%.3f 竟不比中段窗口 m=%.3f 更偏 —— 窗口没起作用" % (full.m, fit.m)
    )


def test_fit_window_is_reported_and_validated():
    a, N = synthetic_a_of_N(1.0e-30, 3.42, 0.22 * W, 0.60 * W)
    fit = paris.paris_fit(a, N, W=W, T=T, delta_P=DELTA_P)
    k_min, k_max = fit.delta_K_range
    lo, hi = fit.fit_window
    span = k_max - k_min
    assert lo == pytest.approx(k_min + span / 3.0, rel=1e-12)
    assert hi == pytest.approx(k_min + 2.0 * span / 3.0, rel=1e-12)
    with pytest.raises(ValueError, match="window"):
        paris.paris_fit(a, N, W=W, T=T, delta_P=DELTA_P, window=(0.7, 0.3))


# --------------------------------------------------------------------------
# 5. 增长率采样与输入校验
# --------------------------------------------------------------------------
def test_crack_growth_rate_uses_fixed_delta_a():
    """Δa 固定：相邻采样点的 a_mid 间距应恰为 delta_a。"""
    a, N = synthetic_a_of_N(1.0e-30, 3.42, 0.22 * W, 0.60 * W)
    delta_a = 0.25e-3
    r = paris.crack_growth_rate(a, N, W=W, T=T, delta_P=DELTA_P, delta_a=delta_a)
    assert r.n_intervals > 5
    assert np.allclose(np.diff(r.a_mid), delta_a, rtol=1e-12)
    # ΔK 随 a 单调上升（定 ΔP、Y 单调）
    assert np.all(np.diff(r.delta_K) > 0.0)
    # da/dN 为正
    assert np.all(r.da_dN > 0.0)


def test_crack_growth_rate_handles_stalled_crack():
    """裂纹停滞时 N(a) 取**首次到达**，停滞期算进正在走的区间。

    构造：11.0 -> 11.5 mm 用了 100 个循环，随后在 11.5 mm 停滞到 N=300，
    再到 12.0 mm 时 N=400。于是
      区间 1（11.0->11.5）= 100 个循环 -> 5.0e-6 m/cycle
      区间 2（11.5->12.0）= 300 个循环 -> 1.667e-6 m/cycle（含停滞）
    若 N(a) 误取「最后离开」，两段的速率会**对调**，这条测试就是锁死这一点的。
    """
    a = np.array([11.0e-3, 11.5e-3, 11.5e-3, 11.5e-3, 12.0e-3])
    N = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    r = paris.crack_growth_rate(a, N, W=W, T=T, delta_P=DELTA_P, delta_a=0.5e-3)
    assert r.n_intervals == 2
    assert r.da_dN[0] == pytest.approx(0.5e-3 / 100.0, rel=1e-12)
    assert r.da_dN[1] == pytest.approx(0.5e-3 / 300.0, rel=1e-12)
    # 停滞使速率下降，故第二段必须比第一段慢
    assert r.da_dN[1] < r.da_dN[0]


def test_crack_growth_rate_too_short_returns_empty():
    a = np.array([11.0e-3, 11.1e-3])
    N = np.array([0.0, 50.0])
    r = paris.crack_growth_rate(a, N, W=W, T=T, delta_P=DELTA_P, delta_a=0.25e-3)
    assert r.n_intervals == 0
    with pytest.raises(ValueError, match="无法拟合"):
        paris.paris_fit(a, N, W=W, T=T, delta_P=DELTA_P)


def test_input_validation():
    a = np.array([11.0e-3, 12.0e-3, 13.0e-3])
    N = np.array([0.0, 100.0, 200.0])
    with pytest.raises(ValueError, match="单调不减"):
        paris.crack_growth_rate(
            np.array([13.0e-3, 12.0e-3, 11.0e-3]), N, W=W, T=T, delta_P=DELTA_P
        )
    with pytest.raises(ValueError, match="单调不减"):
        paris.crack_growth_rate(
            a, np.array([200.0, 100.0, 0.0]), W=W, T=T, delta_P=DELTA_P
        )
    with pytest.raises(ValueError, match="等长"):
        paris.crack_growth_rate(a, N[:2], W=W, T=T, delta_P=DELTA_P)
    with pytest.raises(ValueError, match="必须为正"):
        paris.crack_growth_rate(a, N, W=W, T=T, delta_P=DELTA_P, delta_a=-1.0)
