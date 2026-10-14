"""6.5 T4（W' 的 δ 搜索）与 T5（起裂/转向度量）的单元测试（D-67 §4、§6）。

两者都做成纯函数，故这里**不跑有限元**：T4 注入解析 W(δ)，T5 喂合成 d 场。

T4：
  T4a 割线法在线性 W(δ) 上收敛到 < 1%；
  T4b 在非线性（Hertz 型 W ∝ δ^{3/2}）上同样收敛；
  T4c 不收敛即抛异常（恒值 W(δ)，永远到不了目标）。
T5：
  T5a 竖直裂纹：θ_init、θ_later 均 ≈ 0；
  T5b 倾斜 + 转向：先斜 15° 再拐 35°；
  T5c 表面连通性：孤立斑块不算裂纹（reached_nuc=False）；
  T5d 质心位置与已知几何一致。
"""

import math

import numpy as np
import pytest

from pfwire.clamp_case import crack_metrics, secant_delta

W_TARGET = 1.75e6          # W'（N/m，单位厚度），D-67 §3
K_LINEAR = 1.0e11          # 线性 W(δ) 的刚度


# --------------------------------------------------------------------------
# T4
# --------------------------------------------------------------------------
def test_secant_converges_on_linear_response():
    """T4a：线性 W(δ) = k δ 上，割线法收敛到 < 1%（D-67 §3 的容差）。

    只判「容差内」，不判「等于解析根」：割线法在 1% 容差满足时就返回，
    这是设计意图（省 FE 解），断言到 1e-12 是自相矛盾。
    """
    W = lambda d: K_LINEAR * d
    d = secant_delta(W, W_TARGET, delta0=17.0e-6, rtol=1.0e-2, max_iter=20)
    assert abs(W(d) - W_TARGET) <= 1.0e-2 * W_TARGET
    # 线性 W(δ) 上割线法一步到位（两点确定一条直线），故这里可以严格比
    assert d == pytest.approx(W_TARGET / K_LINEAR, rel=1.0e-12)


def test_secant_converges_on_nonlinear_response():
    """T4b：Hertz 型 W(δ) = k δ^{3/2} 上同样收敛到 < 1%。"""
    K = 1.0e13
    W = lambda d: K * d**1.5
    d = secant_delta(W, W_TARGET, delta0=17.0e-6, rtol=1.0e-2, max_iter=20)
    assert abs(W(d) - W_TARGET) <= 1.0e-2 * W_TARGET
    # 非线性下割线法在容差处就停，故与解析根只比到 1% 量级
    assert d == pytest.approx((W_TARGET / K) ** (2.0 / 3.0), rel=1.0e-2)
    # 收紧容差应当收敛得更准（验证容差确实起作用，不是碰巧）
    d_tight = secant_delta(W, W_TARGET, delta0=17.0e-6, rtol=1.0e-8, max_iter=20)
    assert d_tight == pytest.approx((W_TARGET / K) ** (2.0 / 3.0), rel=1.0e-6)


def test_secant_raises_when_target_is_unreachable():
    """T4c：W(δ) 单调但**渐近饱和**、目标够不到时抛「未收敛」（硬约束 6）。

    注入 W = W_max(1 − e^{−δ/d_s})：单调增但上界 W_max < W_target，割线法会一路
    外推到很大的 δ 却永远差着 (W_target − W_max)，最终被 max_iter 拦住。
    这是「目标不可达」这一类失败，与「W 不单调」区分开（后者由下一条测）。
    """
    W_max = 0.5 * W_TARGET
    W = lambda d: W_max * (1.0 - math.exp(-d / 1.0e-6))
    with pytest.raises(RuntimeError, match="未收敛"):
        secant_delta(W, W_TARGET, delta0=1.0e-6, rtol=1.0e-2, max_iter=20)


def test_secant_raises_when_W_is_not_monotone():
    """T4 辅助：W(δ) 恒值（两点 W 相同）时报「不是单调」，不是悄悄返回坏结果。"""
    with pytest.raises(RuntimeError, match="不是单调"):
        secant_delta(lambda d: 0.0, W_TARGET, delta0=17.0e-6, rtol=1.0e-2, max_iter=5)


def test_secant_returns_immediately_when_start_is_on_target():
    """T4 辅助：起点已在容差内则不再迭代（省一次 FE 解）。"""
    calls = []

    def W(d):
        calls.append(d)
        return K_LINEAR * d

    d = secant_delta(W, W_TARGET, delta0=W_TARGET / K_LINEAR, rtol=1.0e-2)
    assert d == pytest.approx(W_TARGET / K_LINEAR)
    assert len(calls) == 1, "起点已达标时应只调用一次 W(δ)，实际 %d 次" % len(calls)


def test_secant_validates_arguments():
    """T4 辅助：非法容差/迭代数直接拒绝。"""
    with pytest.raises(ValueError):
        secant_delta(lambda d: d, W_TARGET, rtol=0.0)
    with pytest.raises(ValueError):
        secant_delta(lambda d: d, W_TARGET, max_iter=0)


# --------------------------------------------------------------------------
# T5：合成 d 场
# --------------------------------------------------------------------------
def _grid(n_x=2001, depth_max=150.0e-6, n_depth=1501):
    """规则的 (x, depth) 采样网格，深度向下为正、表面为 0。

    分辨率要足够细：6 µm 切片上若只有几个网格点，质心的离散误差会吃掉 0.5°
    的角度容差（实测 dx = 1 µm 时 θ_init 读出 14.04° 而非 15°）。取 dx = 0.2 µm、
    dd = 0.1 µm，离散误差降到 0.05° 以下。
    """
    xs = np.linspace(-200.0e-6, 200.0e-6, n_x)
    depths = np.linspace(0.0, depth_max, n_depth)
    return xs, depths


def _vertical_band(xs, depths, x0=0.0, halfwidth=3.0e-6, top=0.0, bottom=120.0e-6):
    """一条竖直带：|x − x0| <= halfwidth 且 top <= depth <= bottom 处 d = 1。"""
    d = np.zeros((len(depths), len(xs)))
    X, D = np.meshgrid(xs, depths)
    d[(np.abs(X - x0) <= halfwidth) & (D >= top) & (D <= bottom)] = 1.0
    return d


def _kinked_band(xs, depths, x0=0.0, halfwidth=3.0e-6,
                 theta1_deg=15.0, theta2_deg=35.0, kink_depth=6.0e-6,
                 bottom=120.0e-6):
    """先斜 theta1、在 kink_depth 处拐成 theta2 的两段带（深度向下为正）。"""
    d = np.zeros((len(depths), len(xs)))
    t1, t2 = math.radians(theta1_deg), math.radians(theta2_deg)
    X, D = np.meshgrid(xs, depths)
    # 中心线 x_c(depth)：depth <= kink 时按 theta1，之后按 theta2
    xc = np.where(D <= kink_depth,
                  x0 + np.tan(t1) * D,
                  x0 + np.tan(t1) * kink_depth + np.tan(t2) * (D - kink_depth))
    d[np.abs(X - xc) <= halfwidth / math.cos(min(abs(t1), abs(t2)))] = 1.0
    d[D > bottom] = 0.0
    d[D < 0.0] = 0.0
    return d


def test_vertical_crack_has_zero_angles():
    """T5a：竖直裂纹的 θ_init、θ_later 均 ≈ 0（< 0.5°）。"""
    xs, depths = _grid()
    d = _vertical_band(xs, depths)
    m = crack_metrics(d, xs, depths)
    assert m is not None and m.reached_nuc
    assert m.theta_init_deg == pytest.approx(0.0, abs=0.5)
    assert m.theta_later_deg == pytest.approx(0.0, abs=0.5)
    assert m.x_init == pytest.approx(0.0, abs=1e-6)


def test_kinked_crack_gives_init_and_later_angles():
    """T5b：先斜 15° 再拐 35°，θ_init ≈ 15°、θ_later ≈ 35°（各 < 0.5°）。

    注意 θ_later 是「6 µm 质心 → 100 µm 质心」的连线：拐点在 6 µm 处，故第二段
    走的是 theta2 那一段，角度应读回 35°。
    """
    xs, depths = _grid()
    d = _kinked_band(xs, depths)
    m = crack_metrics(d, xs, depths)
    assert m is not None and m.reached_nuc
    assert m.theta_init_deg == pytest.approx(15.0, abs=0.5)
    assert m.theta_later_deg == pytest.approx(35.0, abs=0.5)


def test_isolated_damage_patch_is_not_a_crack():
    """T5c：与表面不连通的孤立斑块不算裂纹（reached_nuc=False）。"""
    xs, depths = _grid()
    d = np.zeros((len(depths), len(xs)))
    X, D = np.meshgrid(xs, depths)
    # 悬空斑块：深度 40–120 µm 的一块，表面（depth=0）没有损伤与之相连
    d[(np.abs(X) <= 10.0e-6) & (D >= 40.0e-6) & (D <= 120.0e-6)] = 1.0
    m = crack_metrics(d, xs, depths)
    assert m is not None
    assert not m.reached_nuc, "孤立斑块不应算作表面连通裂纹"
    assert m.depth_max == 0.0
    assert math.isnan(m.theta_init_deg)


def test_centroids_match_known_geometry():
    """T5d：质心坐标与已知几何一致（< 1 个网格间距）。"""
    xs, depths = _grid()
    d = _vertical_band(xs, depths, x0=20.0e-6)
    m = crack_metrics(d, xs, depths)
    dx = xs[1] - xs[0]
    dd = depths[1] - depths[0]
    assert m.centroid_6[0] == pytest.approx(20.0e-6, abs=dx)
    assert m.centroid_100[0] == pytest.approx(20.0e-6, abs=dx)
    # 切片深度取「最接近 6 µm / 100 µm 的那一行」
    assert m.centroid_6[1] == pytest.approx(6.0e-6, abs=dd)
    assert m.centroid_100[1] == pytest.approx(100.0e-6, abs=dd)


def test_no_damage_returns_no_nucleation():
    """T5 辅助：d 全为 0 时 reached_nuc=False，且不抛异常。"""
    xs, depths = _grid()
    m = crack_metrics(np.zeros((len(depths), len(xs))), xs, depths)
    assert m is not None
    assert not m.reached_nuc
    assert m.dmax == 0.0


def test_shape_mismatch_is_rejected():
    """T5 辅助：d 的形状与坐标不匹配时直接报错（静默错位比报错更难查）。"""
    xs, depths = _grid()
    with pytest.raises(ValueError, match="d_field 形状"):
        crack_metrics(np.zeros((3, 3)), xs, depths)
