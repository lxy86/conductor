"""6.5 基准工况的线夹接触：罚函数法向接触 + Coulomb 摩擦，挂入 PhaseFieldSolver（D-67/6.6.1）。

几何、罚函数与 6.1 的 :class:`pfwire.obstacle.ObstacleContact`（D-57）完全相同，复用其函数：
    g = |X + u − c| − R_clamp，p = ε·max(−g, 0)，ε = κE/h，c = (0, R_clamp − δ)。
区别只在于：这里不持有 u、不建求解器、不带边界条件——接触项是 u 的函数，由相场求解器的 u 方程调用。

摩擦（6.6.1）：``mu=None`` 时**逐位关闭**（= D-67 的无摩擦工况）；给了系数则加
Coulomb 切向牵引 q = ε_t·sign(s)·min(|s|, μp/ε_t)，s = τ·(u − u_prev) 为试探切向滑移。
格式为试探-惩罚（Wriggers 2006 §6.2），基准 u_prev 由 :meth:`set_slip_reference`
在每个载荷步收敛后冻结。

⚠ 6.6.1 单独（δ 恒定）时 q 是**静定**的，不产生周期 ψ⁺ 增量 —— Carrara 的
max(Δψ⁺,0) 会把它滤掉。6.6.1 的价值在**独立验证**（Mindlin 部分滑移），
循环切向驱动要等 6.6.2 的 δ(t)。见 PLAN 6.6.1/6.6.2。
"""

from __future__ import annotations

import ufl
from dolfinx import fem
from mpi4py import MPI

from pfwire.mesh import BLOCK_BOTTOM, BLOCK_TOP
from pfwire.obstacle import (
    clamp_gap,
    clamp_gap_lower,
    clamp_normal,
    clamp_normal_lower,
    penalty_pressure,
    penalty_stiffness,
    penalty_tangential_stiffness,
    plane_gap,
    plane_normal,
    tangential_traction,
)

__all__ = ["ClampContact"]


class ClampContact:
    """线夹的罚函数法向接触，可选 Coulomb 摩擦。

    参数
    ----
    mesh, facet_tags : 网格与面 tag（``BLOCK_TOP`` 为接触面）。
    E                : 弹性模量（SI），用于罚刚度。
    R_clamp          : 线夹轴向曲率半径（m）。
    kappa            : 法向罚系数 κ（无量纲），ε = κE/h_contact。
    h_contact        : 接触面的单元尺寸（m）。D-67：取 clamp_mesh 加密带的 ℓ/4。
    mu               : Coulomb 摩擦系数（无量纲）。**None = 无摩擦**（缺省，
                      逐位回到 D-67 的行为）。
    kappa_t          : 切向罚系数 κ_t（无量纲），ε_t = κ_t E/h。缺省 = κ。
    """

    def __init__(self, mesh, facet_tags, *, E: float, R_clamp: float, kappa: float,
                 h_contact: float, bottom_support: float | None = None,
                 symmetric: bool = False, depth: float | None = None,
                 mu: float | None = None, kappa_t: float | None = None) -> None:
        """``symmetric=True``：单丝被**两个相同线夹对称挤压**（Lévesque 2011）。

        出处：F. Lévesque et al., *Tribology International* 44 (2011) 1014–1023,
        §2："We consider the wire as being squeezed between two similar clamps."
        原模型的边界条件是三个对称面（u_x=0、u_y=0、u_z=0）+ 自由端拉伸 +
        夹持载荷。二维平面应变下只剩轴向锚定的对称面（u_x = 0）。

        为什么必须对称：单侧压紧时，压紧力的反作用力只能由位移约束承担，而
        那个约束（底面固定 / 简支）会同时把弯曲约束掉——实测 σ_a 只剩目标的
        15%，或者 W 只剩 1/7。对称夹持让压紧力**自平衡**，于是只需要一个
        轴向锚定（对称面 u_x = 0），弯曲不被干扰。

        ``bottom_support``（平面支撑）与 ``symmetric``（对称圆弧线夹）二选一：
        前者实测会让接触一发生就发散（支撑面与单丝底面初始重合，罚压力超
        σ_y 八倍），故对称挤压应走 ``symmetric``。
        """
        self.mesh = mesh
        self.facet_tags = facet_tags
        self.R_clamp = float(R_clamp)
        if symmetric and depth is None:
            raise ValueError("symmetric=True 时必须给 depth（单丝厚度），下侧线夹要用它定位")
        self.depth = None if depth is None else float(depth)
        self.eps = penalty_stiffness(kappa, E, h_contact)
        self.h_contact = float(h_contact)
        self.delta = fem.Constant(mesh, 0.0)
        self.bottom_support = bottom_support
        self.symmetric = bool(symmetric)
        if symmetric and bottom_support is not None:
            raise ValueError("symmetric 与 bottom_support 二选一（对称挤压用 symmetric）")
        self.delta_low = fem.Constant(mesh, 0.0)   # 下侧线夹的位移（对称时 = δ/2）
        # 接触项核含 sqrt/max_value（法向、罚压力）。UFL 对残差 F 与对其 u 的导数 J
        # 估计出的多项式阶不同，默认的「按表达式估计求积阶」会让两者落到不同规则，
        # 离散上 F 与 dF/du 不一致（切线校验差到 1.6e-4）。显式固定求积阶 2，
        # 使 F 与 dF/du 用同一规则（线性基函数下 dot(n,v) 的有效阶为 2）。
        self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags,
                              metadata={"quadrature_degree": 2})

        # ---- 摩擦（6.6.1）----
        self.kappa_t = float(kappa_t) if kappa_t is not None else float(kappa)
        self.eps_t = penalty_tangential_stiffness(self.kappa_t, E, h_contact)
        # μ 做成 Constant：**运行时可以切**（set_mu），不必重建 form。
        # 缺省 None 表示「无摩擦」—— 用一个独立的标志而不是 mu=0，
        # 因为 mu=0 时 tangential_residual 仍会被装配（逐位为 0，白耗一次
        # 矩阵装配），而标志能在调用方整个跳过该项。
        self.mu = fem.Constant(mesh, 0.0 if mu is None else float(mu))
        self.has_friction = mu is not None
        #: 构造时是否要求装切向项。**不随 set_mu 变**：切向项装不装是在
        #: form 构造时定死的，而标准加载序列是「先关摩擦压法向、再开摩擦
        #: 推切向」（Mindlin；pfwire/contact.py 的 set_friction_law 同款理由，
        #: 且低压下 Coulomb 的 min() 折点会让 Newton 发散，实测 SNES -6）。
        #: 故调用方判这个标志来决定**装配**，用 mu.value 来控制**大小**。
        self.has_friction_at_build = mu is not None
        if mu is not None and float(mu) <= 0.0:
            raise ValueError("给了 mu 就必须为正（要关摩擦请传 None），收到 %r" % (mu,))
        # 切向滑移基准 u_prev：只在接触面上有意义，但为了能整场插值/存取，
        # 建在与 u 同构的空间上（调用方用 set_slip_reference 冻结）。
        self.u_prev = None       # fem.Function(V)，惰性建（要 V，构造时还没有）

    # ---- 摩擦的运行时接口 ---------------------------------------------
    def bind_slip_space(self, V) -> None:
        """在求解器的位移空间 V 上建切向滑移基准场（只调一次）。

        必须在求解器空间已存在后调用（ClampContact 自己不知道 V）。
        ⚠ 别在外部另建同构空间：那是 CLAUDE.md 陷阱 1 的摩擦版。
        """
        if self.u_prev is None:
            self.u_prev = fem.Function(V)
            self.u_prev.x.array[:] = 0.0

    def set_slip_reference(self, u) -> None:
        """冻结当前切向位移作 s = τ·(u − u_prev) 的基准。

        **每个载荷步收敛后调一次**。这样本步的相对位移才是「新增的滑移」，
        与 Coulomb 的增量语义一致；不冻结的话 s 会从零开始累计，第一次加载
        就把全部历史位移当成滑移，q 立刻饱和（实测 Q/(μP) = 1）。
        """
        if self.u_prev is None:
            raise RuntimeError("先调 bind_slip_space(V) 再 set_slip_reference")
        self.u_prev.x.array[:] = u.x.array
        self.u_prev.x.scatter_forward()

    def set_mu(self, mu: float | None) -> None:
        """运行时切 μ。``None`` 或 0 = 关摩擦（只改 Constant.value，不重建 form）。"""
        self.mu.value = 0.0 if mu is None else float(mu)
        self.has_friction = mu is not None and float(mu) > 0.0

    def _tangents(self, u):
        """(上侧切向 τ, 下侧切向 τ or None)。τ = (n_y, −n_x)（与 pfwire.slip 同口径）。"""
        x = ufl.SpatialCoordinate(self.mesh) + u
        n_up = clamp_normal(x, self.R_clamp, self.delta)
        tau_up = ufl.as_vector([n_up[1], -n_up[0]])
        if self.bottom_support is not None:
            return tau_up, ufl.as_vector(plane_normal())
        if self.symmetric:
            n_lo = clamp_normal_lower(x, self.R_clamp, self.delta_low, -self.depth)
            return tau_up, ufl.as_vector([n_lo[1], -n_lo[0]])
        return tau_up, None

    def tangential_residual(self, u, v) -> ufl.Form:
        """切向牵引的虚功 ∫ q τ·v ds（上侧 + 下侧）。求解器在 u 方程里取负号。

        ``mu=None``（未开摩擦）时返回 **UFL 标量零**（``ufl.as_ufl(0.0)``）。
        ⚠ 判据是 ``has_friction_at_build`` 而非 ``has_friction``：切向项装不装
        在 form 构造时定死，而标准加载序列是「先关摩擦压法向、再开摩擦推切向」
        （Mindlin；低压下 Coulomb 的 min() 折点会让 Newton 发散，实测 -6），
        故 ``set_mu(None)`` 只把 μ 设 0（q ≡ 0）而**不摘除该项**。
        标量零**不是**可装配的 Form（无积分的 form 组装会 IndexError），
        调用方一律先判 ``has_friction_at_build``。
        """
        if not self.has_friction_at_build:
            return ufl.as_ufl(0.0)
        if self.u_prev is None:
            # 静默给 0 比报错更糟：那会变成「摩擦开着但不算」，是问题 19
            # ε_p=0 那类假阳性的同型陷阱。故直接抛。
            raise RuntimeError(
                "开摩擦后必须先 bind_slip_space(V)（切向滑移基准场还没建）")
        p, _n = self._pressure_and_normal(u)
        tau_up, tau_lo = self._tangents(u)
        du = u - self.u_prev
        s_up = ufl.variable(ufl.dot(tau_up, du))
        q_up = tangential_traction(s_up, p, self.mu, self.eps_t)
        form = q_up * ufl.dot(tau_up, v) * self.ds(BLOCK_TOP)
        pb = self._bottom_pressure(u)
        if pb is not None and tau_lo is not None:
            s_lo = ufl.variable(ufl.dot(tau_lo, du))
            q_lo = tangential_traction(s_lo, pb, self.mu, self.eps_t)
            form = form + q_lo * ufl.dot(tau_lo, v) * self.ds(BLOCK_BOTTOM)
        return form

    def tangential_reaction(self, u) -> float:
        """总切向力 Q = ∫ q τ_x ds（N/m，单位厚度），上侧 + 下侧。

        诊断/验证用：判 Coulomb 饱和（Q ≤ μP）、黏着半宽收缩、Q 单调性。
        与 :meth:`tangential_residual` 同判据（``has_friction_at_build``）：
        μ 被设 0 时它算出的是 0，不是「跳过」。
        """
        if not self.has_friction_at_build:
            return 0.0
        if self.u_prev is None:
            raise RuntimeError(
                "开摩擦后必须先 bind_slip_space(V)（切向滑移基准场还没建）")
        p, _n = self._pressure_and_normal(u)
        tau_up, tau_lo = self._tangents(u)
        du = u - self.u_prev
        s_up = ufl.variable(ufl.dot(tau_up, du))
        q_up = tangential_traction(s_up, p, self.mu, self.eps_t)
        local = fem.assemble_scalar(fem.form(q_up * tau_up[0] * self.ds(BLOCK_TOP)))
        pb = self._bottom_pressure(u)
        if pb is not None and tau_lo is not None:
            s_lo = ufl.variable(ufl.dot(tau_lo, du))
            q_lo = tangential_traction(s_lo, pb, self.mu, self.eps_t)
            local = local + fem.assemble_scalar(
                fem.form(q_lo * tau_lo[0] * self.ds(BLOCK_BOTTOM)))
        return float(self.mesh.comm.allreduce(local, op=MPI.SUM))

    def set_delta(self, delta: float) -> None:
        """线夹下压位置（绝对值，m）。由求解器的外层（W' 搜索或延拓）调用。

        delta 是**每个线夹的行程**（对称模式下上下各走 δ，总相对接近量 2δ）。
        这样单侧穿透量 = δ，与单侧模式的 W(δ)、以及 6.1 的结果可比。
        语义来自 Lévesque 2011："a displacement is applied to the middle of the
        wire"——夹持位移就是两个线夹的相对接近量，对称时每侧各分担一半。
        """
        self.delta.value = float(delta)
        if self.symmetric:
            self.delta_low.value = float(delta)   # 上下同行程，相对接近量 2δ

    def _pressure_and_normal(self, u):
        x = ufl.SpatialCoordinate(self.mesh) + u
        g = clamp_gap(x, self.R_clamp, self.delta)
        return penalty_pressure(g, self.eps), clamp_normal(x, self.R_clamp, self.delta)

    def _bottom_pressure(self, u):
        """下侧障碍的压力（无下侧障碍时为 0）。"""
        if self.bottom_support is not None:
            x = ufl.SpatialCoordinate(self.mesh) + u
            return penalty_pressure(plane_gap(x, self.bottom_support), self.eps)
        if self.symmetric:
            x = ufl.SpatialCoordinate(self.mesh) + u
            g = clamp_gap_lower(x, self.R_clamp, self.delta_low, -self.depth)
            return penalty_pressure(g, self.eps)
        return None

    def _bottom_normal(self, u):
        if self.bottom_support is not None:
            return ufl.as_vector(plane_normal())
        if self.symmetric:
            x = ufl.SpatialCoordinate(self.mesh) + u
            return clamp_normal_lower(x, self.R_clamp, self.delta_low, -self.depth)
        return None

    def residual(self, u, v) -> ufl.Form:
        """接触力的虚功 ∫ p n·v ds（上侧 + 下侧）。求解器在 u 方程里取负号。"""
        p, n = self._pressure_and_normal(u)
        form = p * ufl.dot(n, v) * self.ds(BLOCK_TOP)
        pb = self._bottom_pressure(u)
        if pb is not None:
            form = form + pb * ufl.dot(self._bottom_normal(u), v) * self.ds(BLOCK_BOTTOM)
        return form

    def reaction(self, u) -> float:
        """法向反力 W = −∫ p n_y ds（N/m，单位厚度），与 6.1 的 ObstacleContact.reaction 同式。

        只算**上侧线夹**的反力（下侧支撑的反力在静力平衡里与之相等反号，
        不计入 W）——W 的定义是线夹压紧力。
        """
        p, n = self._pressure_and_normal(u)
        local = fem.assemble_scalar(fem.form(-p * n[1] * self.ds(BLOCK_TOP)))
        return float(self.mesh.comm.allreduce(local, op=MPI.SUM))

    def support_reaction(self, u) -> float:
        """下侧支撑的反力（N/m，单位厚度），诊断用：应与 :meth:`reaction` 接近。"""
        pb = self._bottom_pressure(u)
        if pb is None:
            return 0.0
        local = fem.assemble_scalar(fem.form(pb * self.ds(BLOCK_BOTTOM)))
        return float(self.mesh.comm.allreduce(local, op=MPI.SUM))
