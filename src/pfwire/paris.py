"""Paris 律后处理：从 a(N) 曲线算 da/dN–ΔK 并拟合 C、m（验证 2）。

出处
----
全部按 Carrara et al. 2020 (CMAME 361:112731 / arXiv:1811.02244) Appendix B，
该附录明言遵循 ASTM E 647（CT）与 ASTM E 1820（TPB）的标准做法：

- 裂纹增长率（Eq. B.1）：在**固定裂纹增量** Δa 上取差商

      (da/dN)_{n+1/2} = Δa / (N_{n+1} − N_n),      Δa ≃ 0.25 mm

  取固定 Δa 而不是固定 ΔN，是标准做法：ΔN 固定时早期 Δa 太小、噪声被放大；
- 对应的应力强度因子幅（Eq. B.2）：

      ΔK_{n+1/2} = ΔP / (T √W) · Y(γ_{n+1/2}),     γ = a_{n+1/2}/W

  其中 a_{n+1/2} = (a_n + a_{n+1})/2 是区间**平均**裂纹长度；
- CT 的几何因子（Eq. B.3，即 ASTM E647 的标定式）：

      Y(γ) = (2+γ)/(1−γ)^{3/2} · (0.886 + 4.64γ − 13.32γ² + 14.72γ³ − 5.6γ⁴)

- Paris 参数由**双对数坐标下的线性回归**得到，且原文明确「回归只用整个 ΔK
  跨度的**中间三分之一**内的点」—— 这样自动避开前段的萌生非线性段与末段的
  失稳段，只留稳定扩展段。这个口径必须照抄，否则 m 会系统性偏大（把失稳段
  的陡尾巴混进去）。

注意 a 的口径
-------------
本模块里的 ``a`` 一律是**自载荷线量起**的裂纹长度（与 :class:`pfwire.mesh.CtSpec`
的坐标系一致，尖端 x 坐标就是 a），因为 Y(a/W) 的自变量就是这个口径。

单位：SI（m、N、Pa√m）。C 的单位随 m 变，故 :func:`paris_fit` 同时给出
原文表格所用的 mm/cycle–MPa√m 口径下的 C，便于直接对表。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ct_geometric_factor",
    "tpb_geometric_factor",
    "CrackGrowthRate",
    "ParisFit",
    "crack_growth_rate",
    "paris_fit",
]


def ct_geometric_factor(gamma):
    """CT 试样的无量纲几何因子 Y(γ)，γ = a/W（Carrara Eq. B.3 / ASTM E647）。

    ``gamma`` 可以是标量或数组。γ → 1（裂纹接近背面）时 Y 发散，故要求 γ < 1。
    """
    g = np.asarray(gamma, dtype=float)
    if np.any(g <= 0.0) or np.any(g >= 1.0):
        raise ValueError("gamma = a/W 必须落在 (0, 1) 内，收到 %r" % (gamma,))
    poly = 0.886 + 4.64 * g - 13.32 * g**2 + 14.72 * g**3 - 5.6 * g**4
    return (2.0 + g) / (1.0 - g) ** 1.5 * poly


def tpb_geometric_factor(gamma, *, span_over_W: float):
    """三点弯（TPB）的几何因子 Y(γ)（Carrara Eq. B.4 / ASTM E1820）。

    ``span_over_W`` 是跨距比 S/W。本项目的验证 2 只用 CT，此式仅为完整性与
    交叉核对（原文 Tab. 2 同时给了 TPB 的 C、m）而提供。
    """
    g = np.asarray(gamma, dtype=float)
    if np.any(g <= 0.0) or np.any(g >= 1.0):
        raise ValueError("gamma = a/W 必须落在 (0, 1) 内，收到 %r" % (gamma,))
    bracket = 1.99 - g * (1.0 - g) * (2.15 - 3.93 * g + 2.7 * g**2)
    return (
        3.0
        * span_over_W
        * g
        / (2.0 * (1.0 + 2.0 * g) * (1.0 - g) ** 1.5)
        * bracket
    )


@dataclass(frozen=True)
class CrackGrowthRate:
    """按 Eq. B.1/B.2 采样得到的 da/dN–ΔK 数据点。

    ``a_mid`` 是每个区间的平均裂纹长度，``da_dN`` 与 ``delta_K`` 与之一一对应。
    ``n_intervals`` 为 0 表示 a(N) 的跨度不足一个 Δa，无法定义增长率。
    """

    a_mid: np.ndarray
    da_dN: np.ndarray
    delta_K: np.ndarray

    @property
    def n_intervals(self) -> int:
        return int(self.da_dN.size)


@dataclass(frozen=True)
class ParisFit:
    """Paris 律 da/dN = C (ΔK)^m 的拟合结果。

    ``C`` 与 ``m`` 是 SI 口径（m/cycle、Pa√m）。``C_mm_MPa`` 是把
    da/dN 换成 mm/cycle、ΔK 换成 MPa√m 后的 C —— 原文 Tab. 2 用的就是这个口径，
    对表时**必须**用它，否则会差十几个数量级。
    """

    C: float
    m: float
    C_mm_MPa: float
    n_points_used: int
    delta_K_range: tuple[float, float]
    fit_window: tuple[float, float]
    r_squared: float
    rate: CrackGrowthRate


def crack_growth_rate(
    a,
    N,
    *,
    W: float,
    T: float,
    delta_P: float,
    delta_a: float = 0.25e-3,
    geometric_factor=ct_geometric_factor,
) -> CrackGrowthRate:
    """按 Carrara Eq. B.1/B.2 从 a(N) 采样 da/dN 与 ΔK。

    ``a``、``N`` 是模拟输出的裂纹长度（自载荷线量起，m）与循环数，须单调不减。
    做法是在 a 轴上等距布点（间距 ``delta_a``），把 N 插值到这些点上，
    于是每个区间的 Δa 恒为 ``delta_a``、ΔN 由插值得到 —— 这正是「固定裂纹增量」
    的含义，也让不同算例的采样密度一致、可比。
    """
    a_arr = np.asarray(a, dtype=float)
    N_arr = np.asarray(N, dtype=float)
    if a_arr.shape != N_arr.shape or a_arr.ndim != 1:
        raise ValueError("a 与 N 必须是等长的一维数组")
    if a_arr.size < 2:
        raise ValueError("至少需要两个 (a, N) 点")
    if np.any(np.diff(a_arr) < 0.0):
        raise ValueError("a 必须单调不减（裂纹不可逆）")
    if np.any(np.diff(N_arr) < 0.0):
        raise ValueError("N 必须单调不减")
    if delta_a <= 0.0 or W <= 0.0 or T <= 0.0:
        raise ValueError("delta_a、W、T 必须为正")

    # a 轴等距采样点；至少要跨过一个 delta_a 才有一个区间
    n_edges = int(np.floor((a_arr[-1] - a_arr[0]) / delta_a)) + 1
    if n_edges < 2:
        empty = np.empty(0)
        return CrackGrowthRate(empty, empty, empty)
    edges = a_arr[0] + delta_a * np.arange(n_edges)

    # a 上可能有重复值（裂纹停滞时同一个 a 对应多个 N），而 np.interp 要求 xp 严格
    # 递增，故对重复的 a 只保留一个 N。取该处**最小**的 N，即「首次到达该长度的
    # 循环数」—— N(a) 定义成首次到达，ΔN = N(a+Δa) − N(a) 才正好是裂纹**穿过**
    # 这个区间所花的循环数。取最大是错的：那会把停滞期算进**已经走完**的上一个
    # 区间，而不是正在走的这一个，于是停滞前后两段的速率对调。
    a_uniq, inverse = np.unique(a_arr, return_inverse=True)
    N_uniq = np.full_like(a_uniq, np.inf)
    np.minimum.at(N_uniq, inverse, N_arr)
    N_at_edges = np.interp(edges, a_uniq, N_uniq)

    dN = np.diff(N_at_edges)
    a_mid = 0.5 * (edges[:-1] + edges[1:])
    # ΔN = 0 的区间（裂纹在一个 jump 步里越过整个 delta_a）无法定义速率，剔除
    valid = dN > 0.0
    if not np.any(valid):
        empty = np.empty(0)
        return CrackGrowthRate(empty, empty, empty)

    a_mid = a_mid[valid]
    da_dN = delta_a / dN[valid]
    Y = np.asarray(geometric_factor(a_mid / W), dtype=float)
    delta_K = delta_P / (T * np.sqrt(W)) * Y
    return CrackGrowthRate(a_mid=a_mid, da_dN=da_dN, delta_K=delta_K)


def paris_fit(
    a,
    N,
    *,
    W: float,
    T: float,
    delta_P: float,
    delta_a: float = 0.25e-3,
    geometric_factor=ct_geometric_factor,
    window: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
) -> ParisFit:
    """算 da/dN–ΔK 并在中间三分之一的 ΔK 跨度上拟合 C、m。

    ``window`` 是回归窗口在 ΔK 跨度上的相对位置，默认 (1/3, 2/3) 即原文
    Appendix B 的「central third portion of the ΔK range」。改它就不是原文口径了，
    只在做敏感性分析时才动。
    """
    rate = crack_growth_rate(
        a,
        N,
        W=W,
        T=T,
        delta_P=delta_P,
        delta_a=delta_a,
        geometric_factor=geometric_factor,
    )
    if rate.n_intervals < 2:
        raise ValueError(
            "只得到 %d 个 da/dN 点，无法拟合 —— a(N) 的跨度不足或 delta_a 太大"
            % rate.n_intervals
        )

    lo, hi = float(window[0]), float(window[1])
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError("window 必须满足 0 <= lo < hi <= 1，收到 %r" % (window,))

    k_min, k_max = float(rate.delta_K.min()), float(rate.delta_K.max())
    span = k_max - k_min
    k_lo, k_hi = k_min + lo * span, k_min + hi * span
    mask = (rate.delta_K >= k_lo) & (rate.delta_K <= k_hi)
    if int(mask.sum()) < 2:
        raise ValueError(
            "ΔK 中间三分之一窗口 [%.4g, %.4g] 内只有 %d 个点，无法回归"
            % (k_lo, k_hi, int(mask.sum()))
        )

    x = np.log10(rate.delta_K[mask])
    y = np.log10(rate.da_dN[mask])
    m, log_c = np.polyfit(x, y, 1)
    resid = y - (m * x + log_c)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0.0 else 1.0

    C = float(10.0**log_c)
    # 换到原文 Tab. 2 的口径：da/dN [mm/cycle]、ΔK [MPa√m]
    # da/dN_mm = 1e3 · da/dN_SI = 1e3 · C · (1e6 · ΔK_MPa)^m
    C_mm_MPa = C * 1.0e3 * (1.0e6**m)
    return ParisFit(
        C=C,
        m=float(m),
        C_mm_MPa=float(C_mm_MPa),
        n_points_used=int(mask.sum()),
        delta_K_range=(k_min, k_max),
        fit_window=(k_lo, k_hi),
        r_squared=r2,
        rate=rate,
    )
