"""压痕接触的后处理量：接触半宽、平均压力、饱和程度（Phase 6.1）。

纯 numpy，输入是从 FE 解提取的压力剖面 (x, p)。阈值口径与 4.3/4.4 一致：
接触半宽取「p > floor * p_max 的最外采样点」，不是首个零点 —— CG1 插值会在接触区外
留下数值残尾（D-49）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def contact_extent(xs, ps, *, floor: float = 0.01):
    """返回 (a, p0, P_int)：接触半宽、峰值压力、把 p 按 x 积分的总力（单位厚度）。

    `floor` 是接触区判据：p > floor * p_max 的点算作接触区。积分用 `np.trapezoid`，
    它是**积分量的估计**，用于快速诊断；力的精确值应当从加载面的面积分读（`ContactModel.resultant`）。
    """
    xs, ps = np.asarray(xs, float), np.asarray(ps, float)
    if xs.size == 0:
        return float("nan"), float("nan"), float("nan")
    p0 = float(ps.max())
    inside = ps > floor * p0
    a = float(np.max(np.abs(xs[inside]))) if inside.any() else 0.0
    o = np.argsort(xs)
    return a, p0, float(np.trapezoid(np.clip(ps[o], 0.0, None), xs[o]))


def mean_pressure(P: float, a: float) -> float:
    """平面应变压痕的平均接触压力 P / (2a)（单位厚度）。"""
    if a <= 0:
        raise ValueError("接触半宽必须 > 0，得到 %r" % (a,))
    return P / (2.0 * a)


@dataclass(frozen=True)
class Saturation:
    """末段 p0 ~ P^m 的拟合结果。"""

    exponent: float      # d ln p0 / d ln P
    ratio: float         # p0 / P 的末值（诊断用）


def saturation_exponent(P_hist, p0_hist, *, window: int = 3) -> Saturation:
    """用末 `window` 个 (P, p0) 点做 ln-ln 最小二乘，返回指数 m = d ln p0 / d ln P。

    Hertz 线接触 m = 0.5；压力完全饱和（p0 -> 常数）m -> 0。要求 P 严格递增且都为正。
    """
    P = np.asarray(P_hist, float)
    p0 = np.asarray(p0_hist, float)
    if P.size < 2 or P.size != p0.size:
        raise ValueError("P_hist/p0_hist 长度须相同且 >= 2，得到 %d / %d" % (P.size, p0.size))
    if np.any(P <= 0) or np.any(p0 <= 0):
        raise ValueError("P 与 p0 都必须 > 0")
    if np.any(np.diff(P) <= 0):
        raise ValueError("P_hist 必须严格递增（同一载荷下的重复点会污染指数）")
    k = min(window, P.size)
    m = float(np.polyfit(np.log(P[-k:]), np.log(p0[-k:]), 1)[0])
    r = float((p0[-k:] / P[-k:]).mean())
    return Saturation(exponent=m, ratio=r)
