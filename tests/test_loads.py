"""loads.py 单测（PLAN Phase 2.2）。

测试意图
--------
A. **Hertz 三种写法必须互相自洽**（机器精度）：a 由 P 算与由 p0 算必须相同，
   p0 反推也要回到原值。a = 2Rp0/E* 是 DECISIONS.md D-1 整个尺度论证的依据，
   它若与 a = sqrt(4PR/(πE*)) 不一致，D-1 的结论就塌了；
B. **面力积分对拍解析合力**（PLAN 要求误差 < 1e-6）：
   ∫p dx = P、∫q dx = Q = μP[1−(c/a)²]。这是最能抓住分布写错的一条 ——
   形状错了、系数错了、黏着区那一项的符号或缩放错了，合力都会变；
C. **极限情形**：Q/(μP) = 1 -> c = 0 且 q = μp（全滑移）；
   Q/(μP) = 0 -> c = a 且 q ≡ 0（全黏着，不该有微动损伤）；
   以及 q 在黏着区边界 |x| = c 处连续；
D. **与网格同源**：loads 算出的接触半宽必须与 mesh.FrettingSpec 的默认值一致。
   两处各算各的迟早会漂，而 a 一旦不一致，「裂纹在 x ≈ ±a 萌生」的判据就没意义了；
E. **P–S 对拍 D-6 的表**：三个张力下的 p 与 K 必须复现 D-6 记录的数值
   （代码不得硬编码这些值，只能由公式算出来再比）；
F. **CIGRE 分段**：按配置忠实取值；并**显式记录分段点处的不连续** ——
   TB 332 的多层系数在 N = 2e7 处跳变约 −3.2%（原文称为人为折点）
   尚未完成，测试把现状钉住，免得日后有人误以为已经核对过。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate

from pfwire import loads
from pfwire.material import contact_pair_properties
from pfwire.mesh import FrettingSpec

# 生产工况（PLAN Phase 2.4 + D-1）
E_AL, NU_AL, R_WIRE = 69.0e9, 0.33, 1.875e-3
# D-28：基准 p0 由 500 降到 240 MPa（500 MPa 下接触区静态起裂，见 D-27）
P0, MU, Q_RATIO = 240.0e6, 0.7, 0.6
CONDUCTOR_YAML = "configs/conductor_jl1_lha1_465_210.yaml"


@pytest.fixture(scope="module")
def pair():
    return contact_pair_properties(E_AL, NU_AL, R_WIRE, "aluminum")


@pytest.fixture(scope="module")
def contact(pair):
    E_star, R = pair
    return loads.ContactLoads.from_peak_pressure(
        p0=P0, mu=MU, Q_over_mu_P=Q_RATIO, E_star=E_star, R=R
    )


# --------------------------------------------------------------------------
# A. Hertz 自洽
# --------------------------------------------------------------------------
def test_hertz_three_forms_agree(pair, contact):
    """a(p0) 与 a(P) 必须相同；p0 反推回原值。"""
    E_star, R = pair
    P = contact.normal_load
    a_from_P = loads.hertz_half_width(P, R, E_star)
    assert a_from_P == pytest.approx(contact.a, rel=1e-14)
    assert loads.hertz_peak_pressure(P, contact.a) == pytest.approx(P0, rel=1e-14)
    # a = 2 R p0 / E* 的显式复核（D-1 的依据）
    assert contact.a == pytest.approx(2.0 * R * P0 / E_star, rel=1e-14)


def test_hertz_pressure_shape_and_support(contact):
    a, p0 = contact.a, contact.p0
    assert contact.pressure(0.0) == pytest.approx(p0, rel=1e-14)
    assert contact.pressure(a) == pytest.approx(0.0, abs=1e-9)
    # 接触区外必须是 0，不能是负数或 nan
    outside = contact.pressure(np.array([-1.5 * a, 1.5 * a, 10.0 * a]))
    assert np.all(outside == 0.0)
    # 半椭圆：p(a/2) = p0 sqrt(3)/2
    assert contact.pressure(0.5 * a) == pytest.approx(
        p0 * math.sqrt(3.0) / 2.0, rel=1e-12
    )


# --------------------------------------------------------------------------
# B. 积分对拍（PLAN：误差 < 1e-6）
# --------------------------------------------------------------------------
def test_pressure_integrates_to_normal_load(contact):
    P_analytic = contact.normal_load
    P_num, _ = integrate.quad(
        lambda t: float(contact.pressure(t)), -contact.a, contact.a, limit=200
    )
    assert P_num == pytest.approx(P_analytic, rel=1e-6)
    # 解析式本身：P = pi a p0 / 2
    assert P_analytic == pytest.approx(
        0.5 * math.pi * contact.a * contact.p0, rel=1e-14
    )


def test_shear_integrates_to_tangential_load(contact):
    """∫q dx = μP[1−(c/a)²]，且它确实等于 Q_ratio·μP。"""
    Q_analytic = contact.tangential_load
    Q_num, _ = integrate.quad(
        lambda t: float(contact.shear(t)),
        -contact.a,
        contact.a,
        points=[-contact.c, contact.c],
        limit=400,
    )
    assert Q_num == pytest.approx(Q_analytic, rel=1e-6)
    assert Q_analytic == pytest.approx(
        Q_RATIO * MU * contact.normal_load, rel=1e-12
    )


# --------------------------------------------------------------------------
# C. 极限与连续性
# --------------------------------------------------------------------------
def test_full_sliding_limit(pair):
    """Q/(μP) = 1：黏着区消失，切向面力退化为 μ p(x)。"""
    E_star, R = pair
    cl = loads.ContactLoads.from_peak_pressure(
        p0=P0, mu=MU, Q_over_mu_P=1.0, E_star=E_star, R=R
    )
    assert cl.c == pytest.approx(0.0, abs=1e-18)
    x = np.linspace(-cl.a, cl.a, 41)
    assert np.allclose(cl.shear(x), MU * cl.pressure(x), rtol=1e-12, atol=1e-6)
    assert cl.tangential_load == pytest.approx(MU * cl.normal_load, rel=1e-12)


def test_full_stick_limit(pair):
    """Q/(μP) = 0：全黏着，没有滑移区，切向面力恒为 0。

    物理上这一档不该产生微动损伤 —— 微动裂纹靠的是滑移区的往复微滑。
    """
    E_star, R = pair
    cl = loads.ContactLoads.from_peak_pressure(
        p0=P0, mu=MU, Q_over_mu_P=0.0, E_star=E_star, R=R
    )
    assert cl.c == pytest.approx(cl.a, rel=1e-14)
    x = np.linspace(-cl.a, cl.a, 41)
    assert np.allclose(cl.shear(x), 0.0, atol=1e-9)
    assert cl.tangential_load == pytest.approx(0.0, abs=1e-9)


def test_shear_is_continuous_at_stick_boundary_with_sqrt_cusp(contact):
    """|x| = c 处 q 连续，但带**平方根尖点**（导数无穷）。

    黏着项含 sqrt(1 − (x/c)²)，自内侧趋近 c 时它按 sqrt(距离) 归零，
    而不是线性归零。所以「取 eps 邻域看左右值是否相等」必须按 sqrt(eps)
    定容差：eps = 1e-9·a 时左右仍差 4.6e-5 相对量 —— 这不是实现 bug，
    是 Cattaneo–Mindlin 解本身的结构（滑移区边界处切向面力有竖直切线）。

    故这里测两件更有意义的事：
    1. 在 x = c 处黏着项**恰好**为 0，即 q(c) = μ p(c)（精确）；
    2. 左极限按 sqrt 收敛：eps 缩小 100 倍，偏差应缩小约 10 倍。
    """
    c = contact.c
    # 1. 端点处精确
    assert float(contact.shear(c)) == pytest.approx(
        MU * float(contact.pressure(c)), rel=1e-12
    )
    right = float(contact.shear(c + 1e-12 * contact.a))

    # 2. sqrt 收敛率
    diffs = []
    for eps in (1e-7 * contact.a, 1e-9 * contact.a, 1e-11 * contact.a):
        diffs.append(abs(float(contact.shear(c - eps)) - right) / right)
    assert diffs[0] > diffs[1] > diffs[2], "左极限没有收敛"
    # 每次 eps 缩小 100 倍 -> 偏差缩小约 10 倍（sqrt）
    assert diffs[0] / diffs[1] == pytest.approx(10.0, rel=0.25)
    assert diffs[1] / diffs[2] == pytest.approx(10.0, rel=0.25)
    assert diffs[2] < 1e-5


def test_slip_zone_is_where_cracks_should_nucleate(contact):
    """滑移区 c ≤ |x| ≤ a 非空，且 c/a = sqrt(1 − Q/(μP))。"""
    c, a = contact.slip_zone
    assert 0.0 < c < a
    assert c / a == pytest.approx(math.sqrt(1.0 - Q_RATIO), rel=1e-14)


# --------------------------------------------------------------------------
# D. 与网格同源
# --------------------------------------------------------------------------
def test_contact_half_width_matches_mesh_spec(contact):
    """loads 算的 a 必须与 mesh.FrettingSpec 的默认值一致。

    两处各算各的迟早会漂；a 一旦不一致，「裂纹在 x ≈ ±a 萌生」这条
    Phase 2.4 的硬关卡就失去判据。D-1 记的是四舍五入后的 24.2 µm，
    故这里只要求 0.1% 以内；**生产工况应当把 loads 算出的 a 传给 FrettingSpec**，
    而不是依赖这个默认值。
    """
    assert contact.a == pytest.approx(FrettingSpec().contact_halfwidth, rel=1e-3)
    assert contact.a == pytest.approx(11.62e-6, rel=1e-3)


# --------------------------------------------------------------------------
# E. Poffenberger–Swart
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def conductor():
    return loads.load_conductor(CONDUCTOR_YAML)


@pytest.mark.parametrize(
    "label,fraction,p_ref,K_ref_MPa_per_mm",
    [
        ("T15", 0.15, 22.63, 28.87),
        ("T20", 0.20, 26.14, 31.04),
        ("T25", 0.25, 29.22, 32.98),
    ],
)
def test_poffenberger_swart_matches_d6_table(
    conductor, label, fraction, p_ref, K_ref_MPa_per_mm
):
    """三个张力下的 p 与 K 必须复现 DECISIONS.md D-6 记录的数值。

    D-6 那张表是**独立手算**的，所以这是一次真正的对拍，不是自证。
    代码里不得硬编码这些值（只能由公式算出来再比）。
    """
    T = conductor.tension(fraction)
    assert T == pytest.approx(fraction * conductor.RTS, rel=1e-14)
    p = loads.poffenberger_swart_wavenumber(T, conductor.EI_min)
    assert p == pytest.approx(p_ref, rel=2e-3)
    K = conductor.ps_coefficient(fraction)
    assert K / 1e9 == pytest.approx(K_ref_MPa_per_mm, rel=2e-3)


def test_poffenberger_swart_round_trip(conductor):
    """σ_a ↔ Y_b 互逆；并复核 D-6 的量级检查（35 MPa @ T20 -> 约 1.13 mm）。"""
    K = conductor.ps_coefficient(0.20)
    sigma_a = 35.0e6
    Y_b = loads.bending_amplitude_from_stress(sigma_a, K)
    assert loads.stress_from_bending_amplitude(Y_b, K) == pytest.approx(
        sigma_a, rel=1e-14
    )
    assert Y_b == pytest.approx(1.13e-3, rel=5e-3)


def test_ps_coefficient_grows_with_tension(conductor):
    """张力越大，同样的弯曲振幅对应的应力越高。"""
    ks = [conductor.ps_coefficient(f) for f in (0.15, 0.20, 0.25)]
    assert ks[0] < ks[1] < ks[2]


# --------------------------------------------------------------------------
# F. CIGRE 安全边界线
# --------------------------------------------------------------------------
def test_cigre_segments_follow_config(conductor):
    """两段各自按 A N^(−b) 取值。"""
    assert conductor.safe_border_stress(1.0e5) == pytest.approx(
        450.0e6 * 1.0e5 ** (-0.20), rel=1e-12
    )
    assert conductor.safe_border_stress(1.0e8) == pytest.approx(
        263.0e6 * 1.0e8 ** (-0.17), rel=1e-12
    )
    # 数组输入
    out = conductor.safe_border_stress(np.array([1e6, 1e8]))
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


def test_cigre_is_monotonically_decreasing_within_each_segment(conductor):
    for N in (np.logspace(4, 7.15), np.logspace(7.3, 9)):
        s = conductor.safe_border_stress(N)
        assert np.all(np.diff(s) < 0.0)


def test_cigre_breakpoint_is_discontinuous_and_that_is_recorded(conductor):
    """**把现状钉住**：TB 332 的多层系数在分段点处不连续，约 −3.2%。

    这不是实现 bug —— 两段分别是 450 N^−0.20 与 263 N^−0.17，它们的交点在
    N ≈ 5.96e7，而分段点取在 2e7，所以必然跳变。TB 332 汇编也写明这是
    「分段定义的人为折点，非物理现象」。
    本测试防的是有人图「好看」去平滑它 —— 那等于拿一条自己编的曲线跟文献比。

    2026-09-23 已对照 TB 332 原文核对（D-25）：A、b 两段均与原文一致，
    **只有分段点原先错写成 1.56e7**（二手文献的说法），改为 2e7 后跳变
    由 −3.94% 变成 −3.22%，与原文「约 −3.1%（多层）」吻合。
    """
    N_b = 2.0e7
    upper = 450.0e6 * N_b ** (-0.20)
    lower = 263.0e6 * N_b ** (-0.17)
    jump = (lower - upper) / upper
    assert jump == pytest.approx(-0.0322, abs=5e-4)
    # 分段点本身按定义归第一段（N <= N_max）
    assert conductor.safe_border_stress(N_b) == pytest.approx(upper, rel=1e-12)
    # 两支的交点远在分段点之后
    cross = math.exp(math.log(450.0 / 263.0) / (0.20 - 0.17))
    assert cross == pytest.approx(5.96e7, rel=1e-2)


def test_cigre_requires_terminating_segment():
    with pytest.raises(ValueError, match="N_max"):
        loads.cigre_safe_border_stress(
            1e9, [loads.CigreSegment(N_max=1e7, A=450e6, b=0.2)]
        )


# --------------------------------------------------------------------------
# 输入校验
# --------------------------------------------------------------------------
def test_input_validation(pair):
    E_star, R = pair
    with pytest.raises(ValueError, match="Q/\\(μP\\)"):
        loads.mindlin_stick_half_width(1.0e-5, 1.5)
    with pytest.raises(ValueError, match="Q/\\(μP\\)"):
        loads.mindlin_stick_half_width(1.0e-5, -0.1)
    with pytest.raises(ValueError, match="必须为有限正数"):
        loads.hertz_half_width(-1.0, R, E_star)
    with pytest.raises(ValueError, match="必须落在"):
        loads.mindlin_shear(0.0, P0, 1.0e-5, 2.0e-5, MU)
    with pytest.raises(ValueError, match="N 必须为正"):
        loads.cigre_safe_border_stress(
            -1.0, [loads.CigreSegment(None, 450e6, 0.2)]
        )
    with pytest.raises(ValueError, match="fraction_RTS"):
        loads.load_conductor(CONDUCTOR_YAML).tension(1.5)
