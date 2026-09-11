"""接触面力与导线换算（PLAN.md Phase 2.2）。

本模块是**纯解析**的：只依赖 numpy，不 import dolfinx。分工与 material.py 相同 ——
公式放在这里，可以独立单测到机器精度；UFL 表达式在 run_case 里用这些参数去拼。
这样「公式对不对」和「装配对不对」是两件可以分开查的事。

公式出处（PLAN.md 第 1 节「载荷」，Nowell–Hills 的常规做法）
-----------------------------------------------------------
Hertz（二维线接触，单位长度法向力 P，平面应变）::

    E* = 1/[(1−ν₁²)/E₁ + (1−ν₂²)/E₂]     （material.contact_effective_modulus）
    R:  铝–铝 R = R_wire/2；铝–刚性 R = R_wire
    a  = sqrt(4 P R / (π E*))
    p0 = 2 P / (π a)
    p(x) = p0 sqrt(1 − (x/a)²)            |x| ≤ a，其余为 0

消去 P 得 **a = 2 R p0 / E***，这是 DECISIONS.md D-1 尺度论证的依据；
三种写法必须互相自洽，单测里按机器精度对拍。

Cattaneo–Mindlin 部分滑移（Q < μP）::

    c/a = sqrt(1 − Q/(μP))
    q(x) = μ p0 [sqrt(1 − (x/a)²) − (c/a) sqrt(1 − (x/c)²)]   |x| < c   （黏着区）
    q(x) = μ p0 sqrt(1 − (x/a)²)                              c ≤ |x| ≤ a（滑移区）

**滑移区正是微动裂纹萌生的地方**（PLAN Phase 2.4 要求复现裂纹出现在 x ≈ ±a），
所以 c 的位置必须算准，不能用「整体乘个系数」之类的近似。

Poffenberger–Swart（只用于把 σ_a 换算成导线弯曲振幅 Y_b）::

    K = E_al d_wire p² / (4 (e^{−p x} − 1 + p x)),  p = sqrt(T/EI_min),  x = 89 mm
    σ_a = K · Y_b

**口径**（DECISIONS.md D-6）：``Y_b`` 是 89 mm 处的**峰峰值（双振幅）**，
而 ``sigma_a`` 是**零峰值幅值**。两者定义不一致是这个公式最常见的用错方式。

CIGRE 安全边界线（分段幂律，多层导线系数）::

    σ_a = A N^(−b)

系数由导线卡给出。PLAN 与 D-6 都标注「实现前对照 CIGRE TB 332 核对系数」，
该 TODO **尚未完成**，故本模块只忠实实现配置里的分段，不做任何"修正"。

单位：一律 SI（m、Pa、N、N/m）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

__all__ = [
    "hertz_half_width",
    "hertz_half_width_from_peak_pressure",
    "hertz_peak_pressure",
    "hertz_normal_load",
    "hertz_pressure",
    "mindlin_stick_half_width",
    "mindlin_shear",
    "mindlin_tangential_load",
    "ContactLoads",
    "poffenberger_swart_wavenumber",
    "poffenberger_swart_coefficient",
    "bending_amplitude_from_stress",
    "stress_from_bending_amplitude",
    "CigreSegment",
    "cigre_safe_border_stress",
    "Conductor",
    "load_conductor",
]


def _positive(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError("%s 必须为有限正数，收到 %r" % (name, value))
    return v


# --------------------------------------------------------------------------
# Hertz
# --------------------------------------------------------------------------
def hertz_half_width(P: float, R: float, E_star: float) -> float:
    """接触半宽 a = sqrt(4 P R / (π E*))。``P`` 是**单位长度**法向力（N/m）。"""
    P = _positive(P, "P")
    R = _positive(R, "R")
    E_star = _positive(E_star, "E_star")
    return math.sqrt(4.0 * P * R / (math.pi * E_star))


def hertz_half_width_from_peak_pressure(p0: float, R: float, E_star: float) -> float:
    """由峰值压力反推接触半宽：a = 2 R p0 / E*。

    工况配置给的是 p0（PLAN Phase 2.4 取 500 MPa）而不是 P，故常用这一支。
    它与 :func:`hertz_half_width` 只是消去 P 的两种写法，单测按机器精度对拍。
    """
    p0 = _positive(p0, "p0")
    R = _positive(R, "R")
    E_star = _positive(E_star, "E_star")
    return 2.0 * R * p0 / E_star


def hertz_peak_pressure(P: float, a: float) -> float:
    """峰值压力 p0 = 2P/(π a)。"""
    return 2.0 * _positive(P, "P") / (math.pi * _positive(a, "a"))


def hertz_normal_load(p0: float, a: float) -> float:
    """单位长度法向力 P = π a p0 / 2（半椭圆压力分布的面积）。"""
    return 0.5 * math.pi * _positive(a, "a") * _positive(p0, "p0")


def hertz_pressure(x, p0: float, a: float):
    """Hertz 压力分布 p(x)。接触区外返回 0（不是负数，也不是 nan）。"""
    p0 = _positive(p0, "p0")
    a = _positive(a, "a")
    xi = np.asarray(x, dtype=float) / a
    inside = np.abs(xi) <= 1.0
    # np.where 的两支都会被求值，先把区外的自变量夹住，免得 sqrt 拿到负数报警
    return np.where(inside, p0 * np.sqrt(np.clip(1.0 - xi**2, 0.0, None)), 0.0)


# --------------------------------------------------------------------------
# Cattaneo–Mindlin
# --------------------------------------------------------------------------
def mindlin_stick_half_width(a: float, Q_over_mu_P: float) -> float:
    """黏着区半宽 c = a sqrt(1 − Q/(μP))。

    参数取无量纲的 ``Q_over_mu_P = Q/(μP)`` 而不是 (Q, μ, P) 三个量：
    工况配置里直接给的就是这个比值（PLAN Phase 2.4 的 ``Q_ratio = 0.6``），
    而且这样不可能出现「三个量互相矛盾」的输入。

    Q/(μP) = 0 -> c = a（全黏着，无滑移区，不会有微动损伤）；
    Q/(μP) = 1 -> c = 0（全滑移，整个接触区都滑）。
    """
    a = _positive(a, "a")
    ratio = float(Q_over_mu_P)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(
            "Q/(μP) 必须在 [0, 1] 内（超过 1 就是整体滑动，不再是部分滑移），"
            "收到 %r" % (Q_over_mu_P,)
        )
    return a * math.sqrt(1.0 - ratio)


def mindlin_shear(x, p0: float, a: float, c: float, mu: float):
    """Cattaneo–Mindlin 切向面力 q(x)。

    ``c`` 为黏着区半宽（见 :func:`mindlin_stick_half_width`）。
    c = 0 时退化为全滑移 q = μ p(x)。
    """
    p0 = _positive(p0, "p0")
    a = _positive(a, "a")
    mu = _positive(mu, "mu")
    c = float(c)
    if not 0.0 <= c <= a:
        raise ValueError("黏着区半宽 c=%r 必须落在 [0, a=%r] 内" % (c, a))

    xa = np.asarray(x, dtype=float) / a
    outer = np.where(
        np.abs(xa) <= 1.0, np.sqrt(np.clip(1.0 - xa**2, 0.0, None)), 0.0
    )
    if c <= 0.0:
        return mu * p0 * outer
    xc = np.asarray(x, dtype=float) / c
    inner = np.where(
        np.abs(xc) <= 1.0, np.sqrt(np.clip(1.0 - xc**2, 0.0, None)), 0.0
    )
    return mu * p0 * (outer - (c / a) * inner)


def mindlin_tangential_load(p0: float, a: float, c: float, mu: float) -> float:
    """单位长度切向力 Q = μ P [1 − (c/a)²]，P = π a p0/2。

    这是 q(x) 的解析积分；单测用数值积分与它对拍，误差要求 < 1e-6。
    """
    P = hertz_normal_load(p0, a)
    c = float(c)
    if not 0.0 <= c <= a:
        raise ValueError("黏着区半宽 c=%r 必须落在 [0, a=%r] 内" % (c, a))
    return _positive(mu, "mu") * P * (1.0 - (c / a) ** 2)


@dataclass(frozen=True)
class ContactLoads:
    """一组自洽的接触面力参数（p0、a、c、μ 及导出的 P、Q）。

    用 :meth:`from_peak_pressure` 构造：工况配置给的是 p0、μ、Q/(μP)，
    而 a 由 (p0, R, E*) 定。把它们收在一个对象里是为了**保证同源** ——
    网格用的 a（FrettingSpec.contact_halfwidth）与面力用的 a 必须是同一个数，
    各算各的迟早会漂。
    """

    p0: float
    a: float
    c: float
    mu: float

    @classmethod
    def from_peak_pressure(
        cls, *, p0: float, mu: float, Q_over_mu_P: float, E_star: float, R: float
    ) -> ContactLoads:
        a = hertz_half_width_from_peak_pressure(p0, R, E_star)
        c = mindlin_stick_half_width(a, Q_over_mu_P)
        return cls(p0=float(p0), a=a, c=c, mu=_positive(mu, "mu"))

    @property
    def normal_load(self) -> float:
        """单位长度法向力 P（N/m）。"""
        return hertz_normal_load(self.p0, self.a)

    @property
    def tangential_load(self) -> float:
        """单位长度切向力 Q（N/m）。"""
        return mindlin_tangential_load(self.p0, self.a, self.c, self.mu)

    @property
    def slip_zone(self) -> tuple[float, float]:
        """滑移区 c ≤ |x| ≤ a —— 微动裂纹应当在这里萌生（Phase 2.4 的关卡）。"""
        return (self.c, self.a)

    def pressure(self, x):
        return hertz_pressure(x, self.p0, self.a)

    def shear(self, x):
        return mindlin_shear(x, self.p0, self.a, self.c, self.mu)


# --------------------------------------------------------------------------
# Poffenberger–Swart
# --------------------------------------------------------------------------
def poffenberger_swart_wavenumber(T: float, EI_min: float) -> float:
    """p = sqrt(T / EI_min)（1/m）。"""
    return math.sqrt(_positive(T, "T") / _positive(EI_min, "EI_min"))


def poffenberger_swart_coefficient(
    T: float, EI_min: float, E_wire: float, d_wire: float, x: float = 89.0e-3
) -> float:
    """P–S 系数 K = E d p² / (4 (e^{−p x} − 1 + p x))，单位 Pa/m。

    **口径**：σ_a 是零峰值幅值，Y_b 是 89 mm 处的峰峰值（双振幅），见 D-6。
    """
    p = poffenberger_swart_wavenumber(T, EI_min)
    x = _positive(x, "x")
    denom = math.exp(-p * x) - 1.0 + p * x
    if denom <= 0.0:
        raise ValueError(
            "P–S 分母 e^{-px} - 1 + px = %r 非正，检查 T/EI_min/x" % (denom,)
        )
    return _positive(E_wire, "E_wire") * _positive(d_wire, "d_wire") * p**2 / (
        4.0 * denom
    )


def stress_from_bending_amplitude(Y_b: float, K: float) -> float:
    """σ_a = K Y_b。"""
    return float(K) * float(Y_b)


def bending_amplitude_from_stress(sigma_a: float, K: float) -> float:
    """Y_b = σ_a / K（峰峰值，m）。"""
    return float(sigma_a) / _positive(K, "K")


# --------------------------------------------------------------------------
# CIGRE 安全边界线
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CigreSegment:
    """分段幂律的一段：N ≤ ``N_max`` 时 σ_a = A N^(−b)。``N_max`` 为 None 表示最后一段。"""

    N_max: float | None
    A: float
    b: float


def cigre_safe_border_stress(N, segments: list[CigreSegment]):
    """CIGRE 安全边界线 σ_a(N)，按 ``segments`` 分段。

    ⚠️ 本函数**忠实实现配置里的系数，不做任何平滑或修正**。
    PLAN 与 D-6 都标注「实现前对照 CIGRE TB 332 核对系数」，该核对**尚未完成**；
    而按 PLAN 给的多层系数（450/0.20 与 263/0.17）在分段点 N = 1.56e7 处
    **并不连续**（见单测里记录的实测跳变）。这既可能是 CIGRE 原始定义如此，
    也可能是系数抄录有误 —— 在核对 TB 332 之前不得擅自"修正"，
    否则就是拿一条自己编的曲线去跟论文比。
    """
    if not segments:
        raise ValueError("segments 不能为空")
    N_arr = np.asarray(N, dtype=float)
    if np.any(N_arr <= 0.0):
        raise ValueError("N 必须为正")
    out = np.full(N_arr.shape, np.nan)
    remaining = np.ones(N_arr.shape, dtype=bool)
    for seg in segments:
        if seg.N_max is None:
            sel = remaining
        else:
            sel = remaining & (N_arr <= float(seg.N_max))
        out[sel] = seg.A * N_arr[sel] ** (-seg.b)
        remaining &= ~sel
    if np.any(remaining):
        raise ValueError("有 N 落在所有分段之外；最后一段的 N_max 应为 None")
    return out if out.shape else float(out)


# --------------------------------------------------------------------------
# 导线卡
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Conductor:
    """导线参数（从 ``configs/conductor_*.yaml`` 读入）。"""

    name: str
    RTS: float
    EI_min: float
    E_al: float
    d_wire_al: float
    ps_x: float
    cigre_segments: list[CigreSegment]
    tension_cases: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def tension(self, fraction_RTS: float) -> float:
        """张力 T = fraction × RTS（N）。"""
        f = float(fraction_RTS)
        if not 0.0 < f < 1.0:
            raise ValueError("fraction_RTS 应在 (0,1) 内，收到 %r" % (fraction_RTS,))
        return f * self.RTS

    def ps_coefficient(self, fraction_RTS: float) -> float:
        """给定张力比例下的 P–S 系数 K（Pa/m）。"""
        return poffenberger_swart_coefficient(
            self.tension(fraction_RTS), self.EI_min, self.E_al, self.d_wire_al,
            self.ps_x,
        )

    def safe_border_stress(self, N):
        return cigre_safe_border_stress(N, self.cigre_segments)


def load_conductor(path: str | Path) -> Conductor:
    """读导线 yaml。

    注意 PyYAML 走 YAML 1.1：``137.02e+3`` 带符号才会被解析成数，
    不带符号的 ``500e6`` 会变成**字符串**（CLAUDE.md 已知陷阱 2），
    故这里所有数值都显式 float() 转换。
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError("%s 不是 mapping" % (path,))

    mech = raw.get("mechanical", {})
    struct = raw.get("structure", {})
    ps = raw.get("poffenberger_swart", {})
    cig = raw.get("cigre_safe_border_line", {})

    which = str(cig.get("applicable", "multilayer"))
    seg_raw = cig.get(which)
    if not seg_raw:
        raise ValueError(
            "导线卡的 cigre_safe_border_line.%s 为空 —— 该系数尚未填写；"
            "PLAN/D-6 标注需先对照 CIGRE TB 332 核对" % which
        )
    segments = [
        CigreSegment(
            N_max=None if s.get("N_max") is None else float(s["N_max"]),
            A=float(s["A"]),
            b=float(s["b"]),
        )
        for s in seg_raw
    ]
    if segments[-1].N_max is not None:
        raise ValueError("CIGRE 分段的最后一段必须以 N_max: null 收尾（覆盖到无穷）")

    return Conductor(
        name=str(raw.get("name", path.stem)),
        RTS=float(mech["RTS"]),
        EI_min=float(mech["EI_min"]),
        E_al=float(mech["E_al"]),
        d_wire_al=float(struct["d_wire_al"]),
        ps_x=float(ps.get("x", 89.0e-3)),
        cigre_segments=segments,
        tension_cases=list(raw.get("tension_cases", [])),
        raw=raw,
    )
