"""厚壁圆筒内压弹塑性的解析解（Phase 5.4 的对拍基准）。

条件：平面应变、Mises、**弹性不可压（nu = 1/2）**、理想塑性（H = 0）。
推导（不依赖外部公式，只用平衡 + Mises + Lamé；与 Hill 1950, *The Mathematical Theory of
Plasticity*, ch. V 厚壁圆筒一节的结果一致 —— ⚠️ 具体节号待查书核对）：

  * 平衡 d(sr)/dr = (sθ − sr)/r；不可压平面应变 ⇒ σz = (σr + σθ)/2，
    Mises 屈服 ⇒ σθ − σr = (2/√3) σy（塑性区，σθ > σr）。
  * 塑性区 a ≤ r ≤ c：σr(r) = σr(c) + (2σy/√3) ln(r/c)，σθ = σr + 2σy/√3。
  * 弹性区 c ≤ r ≤ b（Lamé，外压为 0，界面压力 p_c）：
        σr = p_c c²/(b²−c²) (1 − b²/r²)，σθ = p_c c²/(b²−c²) (1 + b²/r²)
    在 r = c 处屈服 ⇒ p_c = (σy/√3)(1 − c²/b²)。
  * 内壁 σr(a) = −p ⇒ p(c) = (2σy/√3) [ ln(c/a) + ½ (1 − c²/b²) ]。
    c = a 时退回屈服起始压力 p_y = (σy/√3)(1 − a²/b²)；c = b 时得极限压力
    p_lim = (2σy/√3) ln(b/a)。
  * 位移：不可压 ⇒ r u_r = 常数（弹塑性区同）；弹性区 ε_θ = (σθ − σr)/(4G)，
    在 r = c 处 ε_θ = σy/(2√3 G) ⇒ u_r(r) = σy c²/(2√3 G r)，u_r(a) = σy c²/(2√3 G a)。

任意 nu 的弹性（Lamé）平面应变解 u_r(a) 单独给出（5.4 的弹性段用实际 nu 对拍）。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

SQRT3 = np.sqrt(3.0)


def yield_onset_pressure(sigma_y: float, a: float, b: float) -> float:
    return sigma_y / SQRT3 * (1.0 - a**2 / b**2)


def limit_pressure(sigma_y: float, a: float, b: float) -> float:
    return 2.0 * sigma_y / SQRT3 * np.log(b / a)


def pressure_from_c(c: float, sigma_y: float, a: float, b: float) -> float:
    """弹塑性边界半径 c 对应的内压 p（a <= c <= b）。"""
    return 2.0 * sigma_y / SQRT3 * (np.log(c / a) + 0.5 * (1.0 - c**2 / b**2))


def c_from_pressure(p: float, sigma_y: float, a: float, b: float) -> float:
    """内压 p -> 弹塑性边界 c；p <= p_y 返回 a（全弹性），p >= p_lim 报错。"""
    if p <= yield_onset_pressure(sigma_y, a, b):
        return a
    if p >= limit_pressure(sigma_y, a, b):
        raise ValueError("p = %g 超过极限压力 %g" % (p, limit_pressure(sigma_y, a, b)))
    return brentq(lambda c: pressure_from_c(c, sigma_y, a, b) - p, a, b, xtol=1e-14 * b)


def plastic_stress(r, p: float, sigma_y: float, a: float, b: float):
    """(σr, σθ) 在整个壁厚上（弹塑性）。r 为数组。"""
    r = np.asarray(r, float)
    if p <= yield_onset_pressure(sigma_y, a, b):                 # 全弹性
        return lame_stress(r, p, a, b)
    c = c_from_pressure(p, sigma_y, a, b)
    pc = sigma_y / SQRT3 * (1.0 - c**2 / b**2)                  # 弹塑性界面压力
    sr_plastic = -pc + 2.0 * sigma_y / SQRT3 * np.log(np.minimum(r, c) / c)
    if c < b:
        k = pc * c**2 / (b**2 - c**2)
        rr = np.maximum(r, c)
        sr_el, st_el = k * (1.0 - b**2 / rr**2), k * (1.0 + b**2 / rr**2)
    else:                                                       # 极限状态，整个壁厚塑性
        sr_el = st_el = np.zeros_like(r)
    sr = np.where(r <= c, sr_plastic, sr_el)
    st = np.where(r <= c, sr_plastic + 2.0 * sigma_y / SQRT3, st_el)
    return sr, st


def plastic_u_inner(p: float, sigma_y: float, G: float, a: float, b: float) -> float:
    """内壁径向位移 u_r(a)（nu = 1/2 极限，弹塑性）。"""
    c = c_from_pressure(p, sigma_y, a, b)
    return sigma_y * c**2 / (2.0 * SQRT3 * G * a)


def lame_u_inner(p: float, E: float, nu: float, a: float, b: float) -> float:
    """平面应变 Lamé 解的内壁位移（任意 nu）：u_r(a) = (1+ν) p a [(1−2ν) a² + b²] / (E (b² − a²))。"""
    return (1.0 + nu) * p * a * ((1.0 - 2.0 * nu) * a**2 + b**2) / (E * (b**2 - a**2))


def lame_stress(r, p: float, a: float, b: float):
    """弹性 Lamé 应力（与 nu 无关）：σr = p a²/(b²−a²)(1 − b²/r²)，σθ = p a²/(b²−a²)(1 + b²/r²)。"""
    r = np.asarray(r, float)
    k = p * a**2 / (b**2 - a**2)
    return k * (1.0 - b**2 / r**2), k * (1.0 + b**2 / r**2)
