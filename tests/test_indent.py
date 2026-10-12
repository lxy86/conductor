"""pfwire.indent 的单元测试（纯 numpy）。"""
import numpy as np
import pytest

from pfwire.indent import Saturation, contact_extent, mean_pressure, saturation_exponent

A, P0 = 1.0, 10.0


def _hertz(n=401, a=A, p0=P0):
    x = np.linspace(-2 * a, 2 * a, n)
    p = np.where(np.abs(x) < a, p0 * np.sqrt(np.clip(1 - (x / a) ** 2, 0, None)), 0.0)
    return x, p


def test_hertz_extent_pressure_and_integral():
    x, p = _hertz()
    a, p0, P = contact_extent(x, p)
    assert a == pytest.approx(A, rel=2e-2)          # 采样点离 a 至多半个网格
    assert p0 == pytest.approx(P0, rel=1e-12)
    assert P == pytest.approx(np.pi * P0 * A / 2.0, rel=5e-3)   # ∫p dx = π p0 a / 2


def test_flat_top_profile_gives_mean_equal_to_peak():
    x = np.linspace(-2, 2, 801)
    p = np.where(np.abs(x) <= 1.0, 7.0, 0.0)
    a, p0, P = contact_extent(x, p)
    assert a == pytest.approx(1.0, rel=1e-6)
    # 平顶 -> P/(2a) = p0；梯形法在跳变处有 O(h) 误差，故放到 1%
    assert mean_pressure(P, a) == pytest.approx(p0, rel=1e-2)


def test_extent_ignores_numerical_tail():
    """接触区外 CG1 抹出的数值残尾（p 很小但非零）不应扩大接触区。"""
    x, p = _hertz()
    p = p + 1e-4 * P0
    a, _, _ = contact_extent(x, p)
    assert a == pytest.approx(A, rel=2e-2)


def test_mean_pressure_rejects_zero_width():
    with pytest.raises(ValueError):
        mean_pressure(1.0, 0.0)


def test_saturation_exponent_hertz_is_half():
    P = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    s = saturation_exponent(P, 3.0 * np.sqrt(P))
    assert s.exponent == pytest.approx(0.5, abs=1e-12)
    assert isinstance(s, Saturation)


def test_saturation_exponent_saturated_is_zero():
    P = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    assert saturation_exponent(P, np.full_like(P, 2.97)).exponent == pytest.approx(0.0, abs=1e-12)


def test_saturation_exponent_penetrating_cylinder_is_two_thirds():
    """刚体压痕（平冲头）弹性段 p_mean 恒定、P ∝ a ⇒ p0 若按 p_mean 估计与 P 无关；
    这里用 p0 = c·P^(2/3) 检查指数能取到非 0.5 的值（拟合本身不应假设 0.5）。"""
    P = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    assert saturation_exponent(P, 2.0 * P ** (2.0 / 3.0)).exponent == pytest.approx(2 / 3, abs=1e-12)


def test_saturation_exponent_validates_input():
    P = np.array([1.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        saturation_exponent(P, np.ones_like(P))          # P 非严格递增
    with pytest.raises(ValueError):
        saturation_exponent(np.array([1.0]), np.array([1.0]))   # 长度 < 2
    with pytest.raises(ValueError):
        saturation_exponent(np.array([1.0, -2.0]), np.array([1.0, 1.0]))  # P <= 0
