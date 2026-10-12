"""接触面微滑幅：两侧配对材料点的切向相对位移（PLAN §4 Phase 4.5）。

为什么不用体应力反推：4.4 已证明接触区边缘处用体应力取面力不可靠（D-51），
位移是 FE 的基本未知量，没有这个问题。

口径（D-52）：
  * 基线 = 阶段 1（无摩擦法向压下）结束时的状态；滑移 = 之后的相对位移**增量**。
    阶段 1 无摩擦，两侧本来就会互相滑动，不属于微滑幅。
  * 配对按基线时的**当前 x**：下弧每个节点，找上弧中当前 x（参考 x + u_x）相同的材料点，
    记其参考 x 为 s，之后冻结 s，只读位移。
  * 切向取弧面真实切向 τ = (n_y, -n_x)，与 `ContactModel.surface_tractions` 同口径。

本模块只含纯 numpy 函数（不依赖 FE），输入是接触弧上的节点数据。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def arc_tangent(X_ref: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    """(N,2) 单位切向 τ = (n_y, -n_x)；n 是以 `center` 为圆心的外法向。x≈0 处 τ = (1, 0)。"""
    X_ref = np.asarray(X_ref, dtype=float)
    n = X_ref[:, :2] - np.asarray(center, dtype=float)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    return np.stack([n[:, 1], -n[:, 0]], axis=1)


def pair_material_points(X_up, u_up, X_lo, u_lo) -> np.ndarray:
    """对每个下弧节点，返回配对上弧材料点的**参考 x** s（浮点，上弧节点间线性插值）。

    匹配条件：s + u_up_x(s) = X_lo_x + u_lo_x（两者当前 x 相等）。
    当前 x 落在上弧范围外的返回 nan（不外推）。
    """
    X_up, u_up = np.asarray(X_up, float), np.asarray(u_up, float)
    xc = X_up[:, 0] + u_up[:, 0]
    o = np.argsort(xc)
    x_lo = np.asarray(X_lo, float)[:, 0] + np.asarray(u_lo, float)[:, 0]
    s = np.interp(x_lo, xc[o], X_up[o, 0])
    s[(x_lo < xc[o][0]) | (x_lo > xc[o][-1])] = np.nan
    return s


def relative_tangential_displacement(X_up, u_up, X_lo, u_lo, s, tau) -> np.ndarray:
    """g_i = τ_i · (u_up(s_i) − u_lo_i)。u_up(s) 沿上弧参考 x 线性插值；s 为 nan 处结果为 nan。"""
    X_up, u_up = np.asarray(X_up, float), np.asarray(u_up, float)
    o = np.argsort(X_up[:, 0])
    ux = np.interp(s, X_up[o, 0], u_up[o, 0])
    uy = np.interp(s, X_up[o, 0], u_up[o, 1])
    d = np.stack([ux, uy], axis=1) - np.asarray(u_lo, float)[:, :2]
    return np.sum(np.asarray(tau, float) * d, axis=1)


def slip_increment(g_now: np.ndarray, g_base: np.ndarray) -> np.ndarray:
    """滑移增量（带符号）：当前相对位移减去基线时的相对位移。"""
    return np.asarray(g_now, float) - np.asarray(g_base, float)


@dataclass(frozen=True)
class StickSlipSplit:
    """由滑移剖面判出的黏滑分区。"""

    stick_lo: float              # 黏着区 x 下界（无黏着点则 nan）
    stick_hi: float
    n_stick: int
    n_slip: int
    monotone_outward: bool       # 从黏着区向两侧外推，|slip| 是否单调不减（容差见 classify_slip）


def classify_slip(x, slip, *, stick_rel: float, monotone_tol: float = 0.0) -> StickSlipSplit:
    """按 |slip| < stick_rel * max|slip| 判黏着，并检查黏着区向外 |slip| 是否单调增。

    `x`、`slip` 为同长一维数组（nan 点忽略）。单调性：以黏着区中点为界，右侧按 x 升序、
    左侧按 x 降序，|slip| 的相邻回退不得超过 monotone_tol * max|slip|。
    没有黏着点时（整体滑移）以 |slip| 最小的点为中心判。
    """
    x, slip = np.asarray(x, float), np.asarray(slip, float)
    ok = np.isfinite(x) & np.isfinite(slip)
    x, slip = x[ok], slip[ok]
    o = np.argsort(x)
    x, a = x[o], np.abs(slip[o])
    amax = a.max() if a.size else 0.0
    stick = a < stick_rel * amax if amax > 0 else np.ones_like(a, dtype=bool)
    if stick.any():
        lo, hi = float(x[stick].min()), float(x[stick].max())
        centre = 0.5 * (lo + hi)
    else:
        lo = hi = float("nan")
        centre = float(x[np.argmin(a)])
    slack = monotone_tol * amax
    right, left = a[x >= centre], a[x <= centre][::-1]
    mono = bool(np.all(np.diff(right) >= -slack) and np.all(np.diff(left) >= -slack))
    return StickSlipSplit(lo, hi, int(stick.sum()), int((~stick).sum()), mono)
