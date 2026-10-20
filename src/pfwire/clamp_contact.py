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

import numpy as np
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
        # ---- 切向往复 δ_x(t)（PLAN 6.6.2，D-68 问题 27）----
        # 线夹的**水平行程**。动线夹（刚性障碍的圆心）而不是动接触面自由度：
        # 后者与罚接触同面 prescribed 切向位移是病理组合（黏滑极限环，
        # 6 个测试 SNES -6，D-68 问题 22/25）。
        # 与 delta/mu 同为 fem.Constant：改 .value 即生效，不重建 form（D-16）。
        self.x_clamp = fem.Constant(mesh, 0.0)
        self.x_clamp_low = fem.Constant(mesh, 0.0)
        #: 对称挤压时上下线夹切向运动是否**反向**。默认 True（D-68 问题 27）：
        # 只有相对错动才能在接触斑内产生 Mindlin 部分滑移；同向同幅是刚体平移、
        # 接触面无错动、摩擦不起作用（静默失效），故 False 只作对照工况。
        self.oppose_x = True
        #: 切向行程幅值（m），0 = 关。由配置 contact.tangential_amplitude 给，
        #: 幅值未定（Phase 7.5 扫参对象，D-44「不得预设摩擦学工况」）。
        self.x_amplitude = 0.0
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

        # ---- Uzawa/ALM: 切向牵引乘子 q（DG0 场，断点续跑需存取）----
        # 只有开摩擦时才建（mu is not None），否则为 None 避免白占内存。
        self.q: fem.Function | None = None
        self._uzawa_bound = False  # q 空间是否已绑定

    # ---- 摩擦的运行时接口 ---------------------------------------------
    def bind_slip_space(self, V) -> None:
        """在求解器的位移空间 V 上建切向滑移基准场（只调一次）。

        必须在求解器空间已存在后调用（ClampContact 自己不知道 V）。
        ⚠ 别在外部另建同构空间：那是 CLAUDE.md 陷阱 1 的摩擦版。
        """
        if self.u_prev is None:
            self.u_prev = fem.Function(V)
            self.u_prev.x.array[:] = 0.0

    def bind_uzawa_space(self, V) -> None:
        """在求解器网格上建 Uzawa 乘子场 q（DG0）。

        必须在 **开摩擦**（mu 非 None）且 seek_solver 空间建好后调用。
        q 存入 checkpoint（FIELD_NAMES），续跑时从 .bp 读回。

        ⚠ 名字必须是 ``"q"``：adios4dolfinx 按 ``Function.name`` 存取，
        不给名字时 dolfinx 会塞一个自动名（实测是 ``"f"``），于是写进去的
        键不是 q、读的时候按 "q" 找不到 —— 报的是
        ``"q not found ... Did you mean {'f'}?"``。
        """
        if not self.has_friction_at_build:
            return
        if self.q is not None:
            return
        self.q = fem.Function(fem.functionspace(V.mesh, ("DG", 0)), name="q")
        self.q.x.array[:] = 0.0
        self._uzawa_bound = True

    def has_uzawa(self) -> bool:
        """是否已绑定 Uzawa 乘子场（开摩擦且 bind_uzawa_space 已调用）。"""
        return self._uzawa_bound

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
        n_up = clamp_normal(x, self.R_clamp, self.delta, self.x_clamp)
        tau_up = ufl.as_vector([n_up[1], -n_up[0]])
        if self.bottom_support is not None:
            return tau_up, ufl.as_vector(plane_normal())
        if self.symmetric:
            n_lo = clamp_normal_lower(x, self.R_clamp, self.delta_low, -self.depth,
                                      self.x_clamp_low)
            return tau_up, ufl.as_vector([n_lo[1], -n_lo[0]])
        return tau_up, None

    def _slip(self, u, tau, x_clamp):
        """相对滑移 s = τ·[(u − u_prev) − Δx_clamp]（**唯一口径**，D-68 问题 27）。

        线夹是刚体，其位移矢量是 ``(x_clamp, 0)``，故

            s = τ·(u_wire − u_clamp) = τ·(u − u_prev) − τ_x·x_clamp

        **符号必须是减**：上侧切向 τ = (n_y, −n_x) 在接触顶点附近是 (−1, 0)，
        即 τ 指向 −x。线夹沿 +x 走 x_clamp 时，单丝相对线夹向 −x 错动，
        投影到 τ 上是 **+x_clamp**（正）。写成 ``+τ_x·x_clamp`` 会得到 −x_clamp，
        方向整个反了 —— 实测那样会让摩擦力把单丝往反方向拖，全斑相对滑移
        变大且均匀，``|q|/(μp)`` 在整个接触斑上恒为 0.99（**整体滑移**），
        幅值从 1 µm 到 5 µm 几乎不变，完全看不到 Mindlin 的部分滑移形貌。

        为什么必须相对**线夹**量、而不能只相对单丝自己的历史位移：只取
        τ·(u − u_prev) 时，线夹平移但单丝被摩擦钉住，s ≡ 0 —— 实测线夹走
        40 µm 而表面 u_x 只动 6.8 nm（0.07%），模型完全「看不见」相对滑移。

        ⚠ **残差、反力、Uzawa 更新、诊断必须共用本函数**。此前只有
        ``tangential_residual`` 加了线夹项，``tangential_reaction`` 与测试里的
        剖面辅助没加，于是「装进方程的牵引」与「量出来的牵引」是两个不同的量。
        """
        return ufl.dot(tau, u - self.u_prev) - tau[0] * x_clamp

    def tangential_residual(self, u, v) -> ufl.Form:
        """切向牵引的虚功 ∫ q τ·v ds（上侧 + 下侧）。求解器在 u 方程里取负号。

        - **Uzawa 模式**（`has_uzawa() == True`）：用乘子场 q（DG0）作 Neumann 面力，
          q 由外层 Uzawa 循环更新。此时 form 只含 `q * tau·v`，**不含**切向罚函数。
        - **罚函数模式**（开摩擦但未绑 Uzawa 空间，或兼容旧路径）：
          用正则化试探-惩罚格式 `tangential_traction(s, p, mu, eps_t)`。

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

        tau_up, tau_lo = self._tangents(u)

        # Uzawa 模式：直接用乘子场 q（已绑定在 mesh 上的 DG0 函数）
        if self.has_uzawa():
            assert self.q is not None
            form = self.q * ufl.dot(tau_up, v) * self.ds(BLOCK_TOP)
            pb = self._bottom_pressure(u)
            if pb is not None and tau_lo is not None:
                # 对称挤压：下侧线夹的切向基准 τ_lo ≈ −τ_up，故同一个乘子场
                # 取**相反号**即给出同方向的摩擦力（见 uzawa_update 的推导）。
                # ⚠ 这里**不能**回退到 tangential_traction：那正是墙 3 ——
                # ε_t = κ_t E/h ≈ 3e16 Pa/m 的刚性罚弹簧，Newton 必报 -6。
                form = form + (-self.q) * ufl.dot(tau_lo, v) * self.ds(BLOCK_BOTTOM)
            return form

        # 罚函数模式（旧路径，保留兼容）
        # ⚠ 滑移 s 必须**相对线夹**量（见 _slip 的说明）
        p, _n = self._pressure_and_normal(u)
        s_up = ufl.variable(self._slip(u, tau_up, self.x_clamp))
        q_up = tangential_traction(s_up, p, self.mu, self.eps_t)
        form = q_up * ufl.dot(tau_up, v) * self.ds(BLOCK_TOP)
        pb = self._bottom_pressure(u)
        if pb is not None and tau_lo is not None:
            s_lo = ufl.variable(self._slip(u, tau_lo, self.x_clamp_low))
            q_lo = tangential_traction(s_lo, pb, self.mu, self.eps_t)
            form = form + q_lo * ufl.dot(tau_lo, v) * self.ds(BLOCK_BOTTOM)
        return form

    def tangential_reaction(self, u) -> float:
        """总切向力 Q = ∫ q τ_x ds（N/m，单位厚度），上侧 + 下侧。

        诊断/验证用：判 Coulomb 饱和（Q ≤ μP）、黏着半宽收缩、Q 单调性。
        - **Uzawa 模式**：直接积分乘子场 q。
        - **罚函数模式**：用罚函数公式（兼容旧路径）。
        与 :meth:`tangential_residual` 同判据（``has_friction_at_build``）：
        μ 被设 0 时它算出的是 0，不是「跳过」。
        """
        if not self.has_friction_at_build:
            return 0.0
        if self.u_prev is None:
            raise RuntimeError(
                "开摩擦后必须先 bind_slip_space(V)（切向滑移基准场还没建）")

        tau_up, tau_lo = self._tangents(u)

        # Uzawa 模式：直接积分乘子场 q
        if self.has_uzawa():
            assert self.q is not None
            local = fem.assemble_scalar(fem.form(self.q * tau_up[0] * self.ds(BLOCK_TOP)))
            pb = self._bottom_pressure(u)
            if pb is not None and tau_lo is not None:
                # 对称挤压：下侧用 -q（与 tangential_residual 一致）
                local = local + fem.assemble_scalar(
                    fem.form((-self.q) * tau_lo[0] * self.ds(BLOCK_BOTTOM)))
            return float(self.mesh.comm.allreduce(local, op=MPI.SUM))

        # 罚函数模式（旧路径，保留兼容）
        p, _n = self._pressure_and_normal(u)
        s_up = ufl.variable(self._slip(u, tau_up, self.x_clamp))
        q_up = tangential_traction(s_up, p, self.mu, self.eps_t)
        local = fem.assemble_scalar(fem.form(q_up * tau_up[0] * self.ds(BLOCK_TOP)))
        pb = self._bottom_pressure(u)
        if pb is not None and tau_lo is not None:
            s_lo = ufl.variable(self._slip(u, tau_lo, self.x_clamp_low))
            q_lo = tangential_traction(s_lo, pb, self.mu, self.eps_t)
            local = local + fem.assemble_scalar(
                fem.form(q_lo * tau_lo[0] * self.ds(BLOCK_BOTTOM)))
        return float(self.mesh.comm.allreduce(local, op=MPI.SUM))

    def uzawa_update(self, u, eps_t: float | None = None) -> float:
        """Uzawa/ALM 乘子更新：q ← Proj_{|q| ≤ μp}[q − ε_t · s]。

        ``s = τ·(u − u_prev)`` 是相对上一载荷步（已 commit）的切向滑移。
        ``eps_t`` 为迭代参数（默认 ``self.eps_t``，决定收敛快慢，不影响精度）。
        返回 ``max|Δq| / max(1, max|q|)`` 用作收敛判据。

        ⚠ 仅在 **Uzawa 模式**（``has_uzawa() == True``）下可用；
           罚函数模式直接返回 0.0。

        ⚠⚠ **已知局限（D-68 问题 26 续 5，2026-10-19）**：在**深度黏着**工况
        （接触面切向位移 ≪ μp/E*）下本迭代**不收敛，进二周期极限环**
        （Q/(μP) 在 ±0.97 之间翻符号，rel_change 恒 ~2）。
        根因是黏着区的 q 由体平衡唯一确定，而 Uzawa 用 ε_t·s 迭代它必然过冲；
        ε_t 调小则摩擦被丢掉（Q → 0）。这不是参数问题，是算法与工况不匹配。
        **当前路由**：靠调用方（`PhaseFieldSolver._solve_u_uzawa`）在达到
        max_it 后取**最后一步的 q** 并继续，故不会崩溃、只是切向力停在饱和值。
        要真解黏着区，需改用带 Lipschitz 连续化的对偶原续迄法（Fischer-
        Wriggers 的 Uzawa-CG 或光滑化 Coulomb），PLAN 6.6.2 再评估。
        """
        if not self.has_uzawa():
            return 0.0
        assert self.q is not None
        assert self.u_prev is not None

        if eps_t is None:
            eps_t = self.eps_t

        # 只在接触面（上表面）更新 q；对称/底部支撑用 -q
        tau_up, _ = self._tangents(u)

        # 1) 组装 p（DG0：在接触面单元上取 CG1 节点平均，与 q 同口径）
        if not hasattr(self, "_p_dg_projector"):
            self._build_p_dg_projector()

        p_expr, _n = self._pressure_and_normal(u)
        Vs = fem.functionspace(self.mesh, ("Lagrange", 1))
        p_cg = fem.Function(Vs)
        p_cg.interpolate(fem.Expression(p_expr, Vs.element.interpolation_points))
        p_cg.x.scatter_forward()
        p_dg = self._p_dg_projector.dot(p_cg.x.array[:self._p_n_owned])

        # 2) 滑移 s -> DG0
        # ⚠ 相对滑移必须**相对线夹**量，且与 _slip() 同口径（减号，见其 docstring）。
        # 只取 τ·(u − u_prev) 时线夹平移但单丝被摩擦钉住，s ≡ 0 —— 实测线夹走
        # 40 µm 表面 u_x 只动 6.8 nm（0.07%），模型「看不见」相对滑移。
        s_expr = self._slip(u, tau_up, self.x_clamp)
        s_cg = fem.Function(Vs)
        s_cg.interpolate(fem.Expression(s_expr, Vs.element.interpolation_points))
        s_cg.x.scatter_forward()
        s_dg = self._p_dg_projector.dot(s_cg.x.array[:self._p_n_owned])

        # 3) q 更新：q_tr = q - eps_t * s（负号的物理推导见 docstring），投影到 |q| ≤ μp
        q_old = self.q.x.array.copy()
        q_tr = q_old - eps_t * s_dg
        lim = float(self.mu.value) * np.maximum(p_dg, 0.0)
        q_new = np.where(np.abs(q_tr) <= lim, q_tr, np.sign(q_tr) * lim)

        # 4) 写回并散布
        self.q.x.array[:] = q_new
        self.q.x.scatter_forward()

        # 5) 收敛度量
        dq = float(np.max(np.abs(q_new - q_old)))
        qmax = float(np.max(np.abs(q_new))) if q_new.size else 1.0
        return dq / max(1.0, qmax)

    def _build_p_dg_projector(self) -> None:
        """预构建 CG1 -> DG0 的归约矩阵（仅 BLOCK_TOP 单元的接触节点平均）。

        与 prototype v3 同策略：每个 DG0 单元取它在 BLOCK_TOP 上的 CG1 节点值平均。
        """
        import scipy.sparse as sp
        import numpy as np
        from dolfinx import fem
        from pfwire.mesh import BLOCK_TOP

        Vs = fem.functionspace(self.mesh, ("Lagrange", 1))
        Vq = self.q.function_space  # DG0
        fdim = self.mesh.topology.dim - 1

        # 接触面上的 CG1 节点
        top_nodes = fem.locate_dofs_topological(Vs, fdim,
                                                 self.facet_tags.find(BLOCK_TOP))
        n_owned = Vs.dofmap.index_map.size_local
        top_nodes = top_nodes[top_nodes < n_owned]
        top_set = np.zeros(n_owned, dtype=bool)
        top_set[top_nodes] = True

        n_cell = Vq.dofmap.index_map.size_local
        rows, cols, vals = [], [], []
        for c in range(n_cell):
            nds = np.asarray(Vs.dofmap.cell_dofs(c))
            sel = np.asarray([n for n in nds if n < n_owned and top_set[n]],
                             dtype=np.int32)
            if sel.size:
                for nd in sel:
                    rows.append(c)
                    cols.append(int(nd))
                    vals.append(1.0 / sel.size)

        self._p_dg_projector = sp.csr_matrix((vals, (rows, cols)),
                                              shape=(n_cell, n_owned))
        self._p_n_owned = n_owned

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

    def set_tangential(self, x_amp: float) -> None:
        """设线夹的切向（水平）行程（m）。**只改 Constant.value，不重建 form。**

        上侧线夹移 ``+x_amp``；对称模式下下侧取**反对号**（``oppose_x`` 默认
        True），于是上下线夹的相对错动 = 2|x_amp| —— 这才是微动疲劳的加载方式
        （D-68 问题 27）。同向同幅的相对错动为 0，接触斑内无微滑、摩擦不起作用，
        是必须避免的静默失效，故 ``oppose_x=False`` 只作对照工况。

        ``x_amp=0`` 逐位回到无切向工况（D-67 基准）。

        用途：``clamp_case.set_phase`` 里按相位调它，使切向运动与弯曲同相位::

            clamp.set_tangential(delta_x_amp * sin(2 pi t / T))
        """
        self.x_clamp.value = float(x_amp)
        if self.symmetric:
            sign = -1.0 if self.oppose_x else 1.0
            self.x_clamp_low.value = sign * float(x_amp)

    def _pressure_and_normal(self, u):
        x = ufl.SpatialCoordinate(self.mesh) + u
        g = clamp_gap(x, self.R_clamp, self.delta, self.x_clamp)
        return penalty_pressure(g, self.eps), clamp_normal(
            x, self.R_clamp, self.delta, self.x_clamp)

    def _bottom_pressure(self, u):
        """下侧障碍的压力（无下侧障碍时为 0）。"""
        if self.bottom_support is not None:
            x = ufl.SpatialCoordinate(self.mesh) + u
            return penalty_pressure(plane_gap(x, self.bottom_support), self.eps)
        if self.symmetric:
            x = ufl.SpatialCoordinate(self.mesh) + u
            g = clamp_gap_lower(x, self.R_clamp, self.delta_low, -self.depth,
                                self.x_clamp_low)
            return penalty_pressure(g, self.eps)
        return None

    def _bottom_normal(self, u):
        if self.bottom_support is not None:
            return ufl.as_vector(plane_normal())
        if self.symmetric:
            x = ufl.SpatialCoordinate(self.mesh) + u
            return clamp_normal_lower(x, self.R_clamp, self.delta_low, -self.depth,
                                      self.x_clamp_low)
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
