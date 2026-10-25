"""6.5 基准工况的线夹接触：罚函数法向接触 + Coulomb 摩擦，挂入 PhaseFieldSolver（D-67/6.6.1）。

几何、罚函数与 6.1 的 :class:`pfwire.obstacle.ObstacleContact`（D-57）完全相同，复用其函数：
    g = |X + u − c| − R_clamp，p = ε·max(−g, 0)，ε = κE/h，c = (0, R_clamp − δ)。
区别只在于：这里不持有 u、不建求解器、不带边界条件——接触项是 u 的函数，由相场求解器的 u 方程调用。

摩擦（6.6.1）：``mu=None`` 时**逐位关闭**（= D-67 的无摩擦工况）；给了系数则加
Coulomb 切向牵引 q = ε_t·sign(s)·min(|s|, μp/ε_t)，s = τ·(u − u_prev) 为试探切向滑移。
格式为试探-惩罚（Wriggers 2006 §6.2），基准 u_prev 由 :meth:`set_slip_reference`
在每个载荷步收敛后冻结。

乘子（Uzawa/ALM）路径——6.6.3 起改用**光滑化 Coulomb**（D-68 问题 28 续）：
``q ← smooth_clip(q − θ·ε_t·s; μp)``，θ 为 Fortin–Glowinski (1983) 的阻尼步长。
光滑化投影 :func:`smooth_clip` 用 Fischer–Burmeister 形式（盒内恒等、不抹黏着区），
配阻尼压住 Uzawa 反馈增益 ε_t·K_t ≈ 5e3（否则黏着区进二周期极限环，
问题 26 续 5）。收敛判据同步换成**自然残差** ``q − clip_η(q − ε_t·s)``
（θ 变小会让 Δq 变小而假收敛）。

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

__all__ = ["ClampContact", "smooth_clip", "smooth_clip_ufl"]


def smooth_clip(x, lim, eta):
    """Fischer–Burmeister 光滑化投影：把 ``x`` 折到 ``|·| ≤ lim`` 的盒内（D-68 问题 28 续）。

        clip_η(x; lim) = x − ½·( g + √(g² + η²) )·sign(x)，   g = |x| − lim

    出处：Fischer, *SIAM J. Optim.* 2 (1992) 350–369 的 FB 函数；Kanzow, *Math. Program.*
    74 (1996) 303–320 §3 的 η-光滑化 `φ_η(s,t) = s − t + √(s²+t²)`（t = η/2）用于
    Coulomb 锥投影。**不是** `lim·tanh(x/η)` 之类的外推形式，原因见下。

    三条性质（本函数存在的全部理由）：
      1. **盒内偏差 O(η²/(4·(lim−|x|)))**（η→0 时才严格恒等）：
         令 u = lim − |x| > 0，展开 √(u²+η²) ≈ u + η²/(2u) 得
         result = x − η²/(4u)。偏差随深入盒内按 1/u **快速衰减**，但在
         |x| → lim 处最大（u = η 时偏差 = η/4）。所以 η_rel = 0.05 时
         黏着区深处的相对偏差 ~ η²/(4·lim) ≈ 3e-4，可忽略；**贴近盒面的
         黏着点会有百分之几的偏差**，这是光滑化的代价，也是 η_rel 要取小的原因。
         **关键不在于「严格恒等」，而在于偏差指向盒内、量级 O(η²)、且随 η→0 消失**：
         任何把内部点**整体拖向盒面**的形式（如 μp·tanh(q/η)）在整区给出 O(η)
         量级的偏差，会把黏着牵引整区抹掉 —— 那是 D-68 问题 26 续 5
         「ε_t 调小则 Q→0」的同一个坑，不能用。
      2. **盒外单调、不超限**：|x| > lim 时结果单调趋于 sign(x)·lim，且**始终
         ≤ lim**（Coulomb 锥不被违反）。滑移区 |q| 偏差有界 ≤ η/2。
      3. **η → 0 逐点收敛到精确投影**：η = 0 时 √(g²) = |g|，括号 = max(g, 0)，
         即经典 ``np.clip``。故 η 只决定收敛速率与偏差大小，**不改变不动点集**
         （不动点集恒为盒本身，与 η 无关）。

    ⚠ ``lim = 0``（非接触单元，p = 0）时 g = |x|，结果 = −½(√(x²+η²) − |x|)·sign(x)
       ≈ −η²/(4|x|)·sign(x)：**不恒等于 0**（量级 η²/(4|x|)，比 η 小得多但确实非零），
       会给非接触单元留下小伪切向力。故 :meth:`ClampContact.uzawa_update` 对
       lim=0 的单元走**精确 clip**（得 q ≡ 0），而不是这个光滑形式。

    这是 **numpy 版**，只用于 :meth:`ClampContact.uzawa_update` 的逐单元数组运算。
    需要进 UFL form（单体 Newton 的 Coulomb 投影方程，方向 m）时用
    :func:`smooth_clip_ufl` —— 两个版本必须逐点一致（测试 M4 守着）。
    """
    x = np.asarray(x, dtype=float)
    lim = np.asarray(lim, dtype=float)
    eta = np.asarray(eta, dtype=float)
    g = np.abs(x) - lim
    smooth = g + np.sqrt(g * g + eta * eta)
    return x - 0.5 * smooth * np.sign(x)


def smooth_clip_ufl(x, lim, eta):
    """:func:`smooth_clip` 的 **UFL 版**（D-68 问题 33 续 16，方向 m）。

        clip_η(x; lim) = x − ½·( g + √(g² + η²) )·sign(x)，   g = |x| − lim

    与 numpy 版**同一公式**，只是 x / lim / eta 可为 UFL 表达式
    （``lim = μ·p(u)`` 依赖 u，这正是单体 Newton 需要的）。

    出处同 :func:`smooth_clip`（Fischer 1992 / Kanzow 1996）。

    ⚠ **|x| 写成 ``ufl.sign(x)*x`` 而非 ``ufl.abs(x)``**：UFL 对 ``abs`` 的导数
    在 x=0 处与 ``sign*x`` 不同（前者给出次梯度式的 0 导数），而这里要与 numpy 版
    （``np.abs``）**逐点同值**。``sign*x`` 在 x≠0 时与 ``abs`` 恒等、在 x=0 时为 0
    —— 与 ``np.abs(0) = 0`` 一致。实测两者逐点一致（测试 M4）。

    ⚠ **本函数在 lim=0 时同样不恒等于 0**（与 numpy 版同病，见其 docstring）。
    单体 Newton 的投影方程必须自己处理 lim=0 的单元（走精确 clip 分支）。
    """
    x = ufl.as_ufl(x)
    lim = ufl.as_ufl(lim)
    eta = ufl.as_ufl(eta)
    g = ufl.sign(x) * x - lim
    return x - 0.5 * (g + ufl.sqrt(g * g + eta * eta)) * ufl.sign(x)


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
                 mu: float | None = None, kappa_t: float | None = None,
                 uzawa_eta_rel: float = 0.05) -> None:
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
        # ⚠ **切向行程幅值不是这个对象的状态**：它由 `clamp_case.set_phase` 的
        # 闭包持有（从 config 的 contact.tangential_amplitude 读出），本对象只经
        # `set_tangential(x)` 接收每个相位的**瞬时**偏移。曾有一个
        # `self.x_amplitude` 字段，但没有任何代码读它，改它不产生任何效果
        # （静默失效：实测幅值扫参三档逐位相同）——已删。见 DECISIONS.md
        # D-68 问题 28 续 5「附带发现」。
        # 接触项核含 sqrt/max_value（法向、罚压力）。UFL 对残差 F 与对其 u 的导数 J
        # 估计出的多项式阶不同，默认的「按表达式估计求积阶」会让两者落到不同规则，
        # 离散上 F 与 dF/du 不一致（切线校验差到 1.6e-4）。显式固定求积阶 2，
        # 使 F 与 dF/du 用同一规则（线性基函数下 dot(n,v) 的有效阶为 2）。
        self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags,
                              metadata={"quadrature_degree": 2})

        # ---- 摩擦（6.6.1）----
        self.kappa_t = float(kappa_t) if kappa_t is not None else float(kappa)
        self.eps_t = penalty_tangential_stiffness(self.kappa_t, E, h_contact)
        #: FB 光滑化参数 η = η_rel·μp（D-68 问题 28 续）。盒内偏差 O(η²) → 黏着区
        #: 深处的解几乎不受它影响；只决定盒外（滑移区）|q| 的偏差 ≤ η/2 与 Newton
        #: 的光滑程度。0.05 给盒外 5% 的相对偏差上界，代价是 η 宽的光滑过渡带。
        self.uzawa_eta_rel = float(uzawa_eta_rel)
        # ---- Uzawa 阻尼步长 θ（Fortin–Glowinski 1983；D-68 问题 28 续）----
        # 在 __init__ 就建好（**不是惰性**）：调用方/测试可能先摸 _uzawa_theta 再跑
        # 第一次 uzawa_update，惰性建会留下半初始化态（实测 U15 AttributeError）。
        #   _uzawa_theta     全局阻尼，只用于尚无残差历史的首步
        #   _uzawa_rho_prev  上次的**逐单元**自然残差（判「本单元是否在收敛」用）
        self._uzawa_theta = 1.0
        self._uzawa_theta_init = False
        self._uzawa_rho_prev = None
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
        # ---- 切向 Jacobian 的**运行时**开关（0/1，D-68 问题 33 续 14 修）----
        # u 的 Newton form 在装配时**只编译一次**，而标准加载序列是「先关摩擦
        # 压法向（延拓期）、再开摩擦推切向」：延拓期 μ = 0、q ≡ 0、x_clamp = 0，
        # 切向刚度为 0。若在编译期按 μ 判，项会被永久编译成 0，set_mu 之后再开
        # 摩擦也拿不回来；若在编译期无条件装，延拓期就白扛 ε_t ≈ 1.4e16 的
        # 条件数（实测续 14：延拓期直接 SNES -5，而续 13 延拓期完全正常）。
        # 故用一个 Constant 在**运行时**开关：μ > 0 时置 1，μ = 0 时置 0。
        # form 结构恒定，与 delta / mu 同机制（D-16 的纪律）。
        self._tang_jac_gate = fem.Constant(mesh, 0.0)
        if self.has_friction:
            self._tang_jac_gate.value = 1.0
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

    # ---- 单体 Newton 的混合空间（方向 m，D-68 问题 33 续 16）----
    def bind_monolithic_space(self, V) -> None:
        """建 (u, q) 的**混合空间** W = (V, Q)，并缓存显式 dof 映射。

        方向 m 的载体：把 u 与 q 当一个系统、用单体 Newton 求联立 Jacobian。

        ⚠⚠ **不用 ``ufl.MixedFunctionSpace`` 的隐式重排**（续 16 的设计决定）。
        dolfinx 的混合空间 dof 是 block-interleaved 的，而 `V` 本身已是
        `(Lagrange, 1, (2,))` 向量空间 —— 再混一层 DG0 会得到三层交错的编号。
        那正是 CLAUDE.md 陷阱 1 / 续 9（`end_bcs` 编号空间错配）的同型风险：
        **错位不会报错，只会让约束打在错的点上**（实测得到应力奇异与假损伤）。

        故这里显式建一个 numpy 索引映射，把 W 的 dof 段切成 u 段与 q 段：

            W.x.array[  0 : n_u ]  <->  u.x.array   （V 的全部 dof，含 block 内交错）
            W.x.array[ n_u : n_u+n_q ]  <->  q.x.array[:n_q_owned]

        **必须配守卫测试 M1**（对拍逐位一致），否则这个映射错了没人知道。

        Parameters
        ----------
        V : 求解器的位移空间（``Lagrange 1 (2,)``）。必须与 ``u`` 同空间。
        """
        if not self.has_friction_at_build or not self._uzawa_bound:
            return
        if getattr(self, "_mono_bound", False):
            return
        import numpy as np
        from dolfinx import fem

        n_u = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        n_q = self.q.function_space.dofmap.index_map.size_local
        #: W 的本地规模（不含 ghost）。u 段在前、q 段在后。
        self._mono_n_u = int(n_u)
        self._mono_n_q = int(n_q)
        self._mono_bound = True

    @property
    def mono_n_u(self) -> int:
        """混合空间 W 的 u 段规模（本地，含 block 内交错）。"""
        return getattr(self, "_mono_n_u", 0)

    @property
    def mono_n_q(self) -> int:
        """混合空间 W 的 q 段规模（本地 owned）。"""
        return getattr(self, "_mono_n_q", 0)

    def pack_w(self, w_array, u_array, q_array) -> None:
        """把 (u, q) 写进混合空间 W 的数组（u 段 + q 段）。

        与 :meth:`unpack_w` 互逆。**只切 owned 段**：DG0 的 ``x.array`` 含
        ghost（CLAUDE.md 陷阱 6），CG1 向量空间的 ghost 也在末尾。
        """
        import numpy as np
        n_u, n_q = self._mono_n_u, self._mono_n_q
        w_array[:n_u] = u_array[:n_u]
        w_array[n_u:n_u + n_q] = q_array[:n_q]

    def unpack_w(self, w_array, u_array, q_array) -> None:
        """从混合空间 W 的数组取回 (u, q)。:meth:`pack_w` 的逆。"""
        import numpy as np
        n_u, n_q = self._mono_n_u, self._mono_n_q
        u_array[:n_u] = w_array[:n_u]
        q_array[:n_q] = w_array[n_u:n_u + n_q]

    def contact_cell_bcs(self, u):
        """非接触单元 q 的 DirichletBC 列表（D-68 问题 33 续 17，方向 m 的 B 方案）。

        Coulomb 投影方程只在接触面上有定义（facet 量配 facet 测度，
        见 :meth:`monolithic_residual` 的注释）。若不约束非接触单元的 q，
        全域线性系统有 ~4953 行整行为零 ⇒ 奇异 ⇒ SNES 0 迭代 -3。

        B 方案：把那些 q **钉死为 0**。这在数学上正是 Coulomb 的本意 ——
        非接触面上没有摩擦牵引。与 :meth:`uzawa_update` 对 lim=0 单元走
        ``np.clip(..., 0, 0)`` 得 ``q ≡ 0`` 是**同一件事**，
        只是这里用空间投影（DirichletBC）而非 form 分支。

        ⚠ 这是**每个载荷步都要重建**的：接触斑随 u 变化。
        ⚠ 用空间投影而非 ``ufl.conditional``：后者会吃掉分支导数
        （续 16 坑 2，实测 nnz 从 40536 掉到 0）。
        """
        import numpy as np
        from dolfinx import fem

        if not self.has_friction_at_build:
            return []
        if not self._uzawa_bound:
            return []

        contact, all_dofs = self._contact_cell_dofs(u)
        free_mask = np.ones(all_dofs.size, dtype=bool)
        free_mask[contact] = False
        pinned = all_dofs[free_mask]
        if pinned.size == 0:
            return []
        zero = fem.Function(self.q.function_space)
        return [fem.dirichletbc(zero, pinned)]

    def build_monolithic_problem(self, u, elastic_external_forms, bcs,
                                 q_bcs=None):
        """建 (u, q) 的**单体** ``NonlinearProblem``（D-68 问题 33 续 16，方向 m）。

        Parameters
        ----------
        u : 位移场（CG1 向量）。**它同时是混合系统的第一个未知量**。
        elastic_external_forms : 可调用 ``v -> Form``，返回「弹性 − 外载 − 法向接触」
            那一部分（由 ``phasefield`` 提供，因为弹性/塑性项在那里）。
        bcs : u 的 DirichletBC 列表。
        q_bcs : q 的 DirichletBC 列表。**B 方案必给**
            （:meth:`contact_cell_bcs`：把非接触单元的 q 钉 0）。
            不给则全域系统奇异（SNES 0 迭代 -3，续 16 实测）。

        Returns
        -------
        ``problem`` —— 可直接 ``.solve()`` 的 ``NonlinearProblem``。

        ⚠ **q 的方程没有加权残差的形式**，所以不能用 ``fem.form`` 的常规装配路径
        之外的任何东西 —— 这里两条残差都是普通 form，``NonlinearProblem`` 的
        Sequence 接口原生支持（实测 ``F``/``J`` 都接受 ``Sequence``）。

        ⚠ **单体解出后必须 unpack 回 u 与 q**（M1 的守卫覆盖这个映射）。
        """
        import ufl
        from dolfinx import fem
        from dolfinx.fem.petsc import NonlinearProblem

        if not self._mono_bound:
            raise RuntimeError("先调 bind_monolithic_space(V)")
        if self.u_prev is None:
            raise RuntimeError("开摩擦后必须先 bind_slip_space(V)")
        if q_bcs is None:
            # 不静默：奇异会以 -3 的形式出现，而真因是"没给 q_bcs"
            raise RuntimeError(
                "必须给 q_bcs（contact_cell_bcs 的结果）：Coulomb 投影方程只在"
                "接触面上有定义，不约束非接触单元的 q 会让全域系统奇异")

        v = ufl.TestFunction(u.function_space)
        du = ufl.TrialFunction(u.function_space)
        q_trial = ufl.TestFunction(self.q.function_space)
        q_du = ufl.TrialFunction(self.q.function_space)

        # (1) u 的平衡：调用方给的弹性/外载/法向接触 + 这里的 q 面力
        F_u = elastic_external_forms(v)
        F_u_contact, F_q = self.monolithic_residual(u, v, q_trial, None)
        F_u = F_u + F_u_contact

        blocks = self.monolithic_jacobian(u, du, v, q_trial, q_du)

        # 混合系统的未知量列表：[u, q]（NonlinearProblem 的 Sequence 接口）
        # ⚠⚠ **bcs 必须是扁平列表**（续 17 实测）：dolfinx 0.11 的
        # ``NonlinearProblem`` 对混合系统期望 ``bcs`` 是**一个**平铺列表，
        # 每个 BC 靠自己的 ``function_space`` 归位。
        # 给嵌套的 ``[bcs_u, bcs_q]`` 会让 PETSc 抛 **error code 101**
        # （实测：嵌套 → 101；扁平 → 正常进入迭代）。
        # ⚠ 这与 ``F``/``J`` 的嵌套要求**不一致** —— 那两处要 Sequence。
        flat_bcs = list(bcs) + list(q_bcs)
        problem = NonlinearProblem(
            [F_u, F_q],
            [u, self.q],
            J=blocks,
            bcs=flat_bcs,
            petsc_options_prefix="pfwire_mono_",
            petsc_options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "l2",
                "snes_atol": 1.0e-8,
                "snes_rtol": 1.0e-8,
                "snes_max_it": 40,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        return problem

    # ---- 单体 Newton 的联立残差与 Jacobian（方向 m）----
    def _contact_cell_dofs(self, u):
        """**在接触面上**的 DG0 dof 列表（D-68 问题 33 续 17，方向 m 的 B 方案）。

        为什么需要（续 16/17 的实测结论）：Coulomb 投影方程

            F_q = (q − smooth_clip_ufl(q − ε_t·s, μ·p, η))·q_trial·ds(BLOCK_TOP)

        **只在接触面上有定义**。两种测度选择的实测对比（中心差分判据）：

        ==============  ============  ==========  ==========
        写法            dF_q/du nnz  max|.|      FD 一致
        ==============  ============  ==========  ==========
        facet + dx      26568         1.57e-9    1/4（伪影）
        facet + ds      260           **1.20e11** **4/4（真）**
        ==============  ============  ==========  ==========

        ⇒ **facet 量必须配 facet 测度**。`facet + dx` 是「求值碰巧对、
        求导一定错」（续 16 坑 1 的本质）；`facet + ds` 的 `dF_q/du` 是真的。

        但 `facet + ds` 的代价：5067 个 DG0 单元里只有 114 个有方程，
        其余 4953 行整行为零 ⇒ **全域线性系统奇异**（SNES 0 迭代 -3）。

        **B 方案的处方**：不把方程写到全域，而是把 q 的**自由度**限制在
        接触单元上 —— 给非接触单元的 q 钉 DirichletBC（值 0）。
        这样奇异行对应的未知量本就被约束住，系统恢复非奇异，
        且 `dF_q/du` 保持**真实**（1.20e11，不是 1.57e-9 的伪影）。

        ⚠⚠⚠ **判据必须是「单元是否触及 BLOCK_TOP facet」，不能是 ``p_dg > 0``**
        （续 17 实测的坑）：``p_dg`` 是 **cell 平均投影**（从边界 CG1 节点
        取平均），于是与边界**相邻但不在边界上**的单元也会得到 ``p_dg > 0``。
        实测两种判据差 10 个单元：

            触及 BLOCK_TOP facet : 114 个  ← **正确**（与 ds 的积分域一致）
            p_dg > 0             :  71 个
            交集                 :  61 个
            p_dg>0 但不在边界上   :  10 个  ← 这些既无方程又没被钉 ⇒ 奇异

        用错判据时加完 BC 仍有 **10 个零行**，SuperLU 报
        ``Matrix is exactly singular``、dx 全 NaN。改用 facet 判据后零行 = 0。

        ⚠ 这与续 16 的坑 2（`ufl.conditional` 吃导数）**不矛盾**：
        这里用的是**空间投影**（DirichletBC 消除自由度），
        不是 form 里的 conditional 分支。

        ⚠ 这是**有状态**的：接触斑随载荷步变化，故每次都要重新取 dof。

        Returns
        -------
        ``(contact_dofs, all_dofs)`` —— 两个 numpy int32 数组（owned 段）。
        """
        import numpy as np
        from dolfinx.mesh import compute_incident_entities
        from pfwire.mesh import BLOCK_TOP

        msh = self.mesh
        tdim = msh.topology.dim
        top_facets = self.facet_tags.find(BLOCK_TOP)
        cells = compute_incident_entities(msh.topology, top_facets, tdim - 1, tdim)
        n_owned = msh.topology.index_map(tdim).size_local
        cells = np.sort(cells[cells < n_owned])
        all_dofs = np.arange(self.q.function_space.dofmap.index_map.size_local,
                             dtype=np.int32)
        return cells.astype(np.int32), all_dofs

    def _slip_and_pressure_dg(self, u, tau_up):
        """把滑移 s 与压力 p 采成 **DG0 的 UFL 场**（D-68 问题 33 续 16）。

        为什么需要：:meth:`monolithic_residual` 的 Coulomb 投影方程 F_q 必须
        在**全域**（每个单元）都有定义 —— q 是全域 DG0 场，若 F_q 只写在
        facet 上，5067 个单元里只有 114 个有方程，其余 4953 行整行为零，
        全域线性系统奇异，SNES **0 次迭代就 -3**（续 16 实测）。

        而 ``s`` 与 ``p`` 天生是 facet 上的量（来自 ``clamp_gap`` /
        ``clamp_normal``，在接触面上求值）。所以要先把它们采成 DG0 场，
        与 :meth:`uzawa_update` 的第 1/2 步**完全同口径**（同一个
        ``_p_dg_projector``、同一个 CG1 中转）。

        Returns
        -------
        ``(p_dg, s_dg)`` —— 两个 DG0 ``fem.Function``，可直接进 UFL 表达式。

        ⚠ **这是有状态的**（每次都新建两个 Function 并插值）。单体重用同一
        个 form 时不能这么干，但当前实现是每次调用重建 form（见
        :meth:`build_monolithic_problem`），故无妨。
        若将来要缓存 form，这两个场必须提升为成员并显式更新。
        """
        import numpy as np
        from dolfinx import fem

        if not hasattr(self, "_p_dg_projector"):
            self._build_p_dg_projector()

        Vs = fem.functionspace(self.mesh, ("Lagrange", 1))

        # ⚠⚠⚠ **`_p_dg_projector.dot(...)` 返回的是长度 ``n_cell`` 的 numpy 数组**，
        # 不是 Function（它的 shape 是 ``(n_cell, n_owned_CG1)``，续 16 实测的坑）。
        # 而 ``self._p_n_owned`` 是 **CG1** 的节点数，**不是** DG0 的单元数 ——
        # 拿它去切 DG0 的 ``x.array`` 会留下大片未初始化的槽（实测得 NaN，
        # 且残差与 numpy 版差 1.0e+00）。DG0 的 owned 段长度必须单独取：
        n_cell = self.q.function_space.dofmap.index_map.size_local

        # p：与 uzawa_update 第 1 步同源
        p_expr, _n = self._pressure_and_normal(u)
        p_cg = fem.Function(Vs)
        p_cg.interpolate(fem.Expression(p_expr, Vs.element.interpolation_points))
        p_cg.x.scatter_forward()
        p_arr = self._p_dg_projector.dot(p_cg.x.array[:self._p_n_owned])
        assert p_arr.size == n_cell, "p_dg 长度 %d != DG0 owned %d" % (p_arr.size, n_cell)
        p_dg = fem.Function(self.q.function_space)
        p_dg.x.array[:n_cell] = p_arr
        p_dg.x.scatter_forward()

        # s：与 uzawa_update 第 2 步同源（**必须相对线夹**，见 _slip 的说明）
        s_expr = self._slip(u, tau_up, self.x_clamp)
        s_cg = fem.Function(Vs)
        s_cg.interpolate(fem.Expression(s_expr, Vs.element.interpolation_points))
        s_cg.x.scatter_forward()
        s_arr = self._p_dg_projector.dot(s_cg.x.array[:self._p_n_owned])
        assert s_arr.size == n_cell, "s_dg 长度 %d != DG0 owned %d" % (s_arr.size, n_cell)
        s_dg = fem.Function(self.q.function_space)
        s_dg.x.array[:n_cell] = s_arr
        s_dg.x.scatter_forward()

        return p_dg, s_dg

    def monolithic_residual(self, u, v, q_trial, w) -> tuple:
        """(u, q) 联立系统的**残差**（D-68 问题 33 续 16，方向 m）。

        两条方程：

        (1) u 的平衡 —— 与交错格式**完全同一条**：

            F_u = 弹性 − 外载 − 法向接触 − ∫ q·(τ·v) ds

            q 在这里是 Neumann 面力（DG0 Function），与
            :meth:`tangential_residual` 的 Uzawa 分支逐位相同。
            ⚠ **不含任何 ε_t 项**：续 15 已证 `ε_t·(τ·du)(τ·v)` 不是 F 的导数
            （它在 F 里没有对应项，量级比真实切向刚度大 9.5e7 倍）。

        (2) Coulomb 投影方程（单体系统的第二行，交错格式里它是**外层迭代**）：

            F_q = q − smooth_clip_ufl(q − ε_t·s(u), μ·p(u), η)

        不动点条件正是 `q = Proj_{|q|≤μp}(q − ε_t·s)`，即 Uzawa 的不动点。
        故单体解与交错收敛解**是同一个解**（测试 M3 守着）。

        ⚠ **第二行不是 weighted residual 的形式**，而是「残差 = 0」的点态方程。
        用 `q_trial`（DG0 测试函数）乘起来只是给 dolfinx 一个可装配的 form。

        ⚠⚠⚠ **F_q 必须在全域（`ufl.dx`）上有定义**（续 16 实测的第二个坑）：
        q 是**全域** DG0 场（5067 个单元），而 `p`/`s` 是 **facet** 上的量。
        第一版把 F_q 写在 `ds(BLOCK_TOP)` 上，结果 5067 个单元里只有 114 个
        有方程、**4953 行整行为零** ⇒ 全域线性系统奇异 ⇒ LU 分解失败 ⇒
        SNES **0 次迭代就 -3**（实测：放宽到 200 次仍 -3、q 完全没动）。
        那是比 -5/-9 更难查的失效形态： solver 报告"没收敛"，
        但真正的原因是"方程没写"。

        解法：把 `p` 与 `s` **采成 DG0 场**（`_p_dg_projector`，
        :meth:`uzawa_update` 本来就用这条口径），F_q 写在 `ufl.dx` 上。
        非接触单元自然得到 `p_dg = 0 ⇒ lim = 0 ⇒ η = 0 ⇒ clip ≡ 0 ⇒ F_q = q`，
        即 `q ≡ 0` —— 与 Uzawa 对非接触单元的处理**完全一致**，
        且每个单元都有方程，系统不再奇异。

        Parameters
        ----------
        u       : 位移场（CG1 向量）
        v       : u 的测试函数
        q_trial : q 的测试函数（DG0）
        w       : 混合空间的解（用于读 q 段；交错格式下即 ``self.q``）

        Returns
        -------
        (F_u, F_q) : 两条残差 form
        """
        import ufl
        from dolfinx import fem

        if not self.has_friction_at_build:
            raise RuntimeError("单体路径只在开摩擦时有意义")
        if self.u_prev is None:
            raise RuntimeError("开摩擦后必须先 bind_slip_space(V)")

        tau_up, tau_lo = self._tangents(u)

        # ---- (1) u 的平衡 ----
        # 只保留接触/摩擦部分；弹性与外载由调用方（phasefield）加。
        F_u = self.q * ufl.dot(tau_up, v) * self.ds(BLOCK_TOP)
        pb = self._bottom_pressure(u)
        if pb is not None and tau_lo is not None:
            F_u = F_u + (-self.q) * ufl.dot(tau_lo, v) * self.ds(BLOCK_BOTTOM)

        # ---- (2) Coulomb 投影方程 ----
        # ⚠⚠⚠ **必须用 facet 表达式 + facet 测度**（续 17 实测，B 方案的基础）：
        #   facet + ds : dF_q/du max = 1.20e11、FD 4/4 一致  ⇒ **真导数**
        #   facet + dx : dF_q/du max = 1.57e-9、FD 1/4 一致  ⇒ 伪影
        #   DG0 场+dx  : dF_q/du ≡ 0                      ⇒ 块三角（退化成交错）
        # facet 量（p 来自 clamp_gap、s 来自 τ·(u−u_prev)）**必须配 facet 测度**，
        # 这是 facet 量的定义域决定的，不是风格选择。
        #
        # 代价：F_q 只在接触面上有定义 ⇒ 5067 个 DG0 单元里只有 ~114 个有方程，
        # 其余行整行为零 ⇒ 全域系统奇异。
        # 处方（B）：调用方给**非接触单元的 q 钉 DirichletBC(0)**，
        # 见 :meth:`contact_cell_bcs` 与 :meth:`_contact_cell_dofs`。
        p, _n = self._pressure_and_normal(u)
        s_up = self._slip(u, tau_up, self.x_clamp)
        lim = self.mu * p
        # η = η_rel·μp 逐点自缩放（与 uzawa_update 的 numpy 版同口径）。
        eta = ufl.max_value(self.uzawa_eta_rel * lim, ufl.as_ufl(1.0e-30))
        trial = self.q - self.eps_t * s_up
        # ⚠⚠ **不包 ufl.conditional**（续 16 坑 2）：UFL 对
        # ``conditional(c, a, b)`` 求导时**只给条件本身的导数**，丢掉分支导数。
        # 实测 nnz 从 40536 掉到 0。非接触单元靠 DirichletBC 处理，不用 conditional。
        inner = smooth_clip_ufl(trial, lim, eta)
        F_q = (self.q - inner) * q_trial * self.ds(BLOCK_TOP)

        return F_u, F_q

    def monolithic_jacobian(self, u, du_trial, v, q_trial, q_du_trial) -> list:
        """(u, q) 联立系统的 **2×2 块 Jacobian**（D-68 问题 33 续 16，方向 m）。

        .. math::

            J = \\begin{bmatrix} \\partial F_u/\\partial u & \\partial F_u/\\partial q \\\\
                                    \\partial F_q/\\partial u & \\partial F_q/\\partial q
                \\end{bmatrix}

        四块的来源与量级：

        - ``J[0][0] = dF_u/du``：法向接触项 + q 面力项对 u 的导数。
          **只让 UFL 自动求导**，不手工加任何 ε_t 项（续 15 已证那是错的）。
        - ``J[0][1] = dF_u/dq``：**精确为** ``∫ (τ·v) ds``（q 作 Neumann 面力）。
        - ``J[1][0] = dF_q/du``：**只在黏着区非零**。来自 ``s`` 与 ``p`` 对 u 的依赖。
          第一版只保留 ``−ε_t·(τ·du)·q_trial`` 一项（``∂s/∂u`` 的主项）；
          ``∂p/∂u`` 经 ``lim = μ·p`` 进来的那一项目前**不含**（见下方已知近似）。
        - ``J[1][1] = dF_q/dq``：``1 − ∂smooth_clip/∂(q − ε_t s)``，由 FB 导数给出。
          黏着区 ≈ 1（盒内恒等），滑移区 ≈ 0（投影已饱和，不再随 q 变）。

        ⚠ **已知近似（第一版故意为之，M3 会验证不动点仍不变）**：
        ``lim = μ·p(u)`` 依赖 u，故 ``∂F_q/∂u`` 严格说还有第二项
        ``−ε_t·s·(∂clip/∂lim)·μ·(∂p/∂u)·q_trial``。它被省掉了，理由：
        黏着区 `s → 0` 使该项整体趋 0，而滑移区 `∂clip/∂lim` 很小。
        **这不是 Jacobian 的精确导数**，只影响 Newton 的收敛路径、不影响不动点
        （Newton 用近似 Jacobian 仍收敛到 F = 0，只是可能多几次迭代或线搜索）。

        ⚠ **不用 mask 场，改用 UFL 的自动导数 + conditional**：续 14 附注刚记下
        「编译期恒定 vs 运行期要变」的坑，而 mask 场正是那个坑的温床。
        这里让 UFL 自己从 `smooth_clip_ufl` 的表达式里求导，黏着/滑移的
        区别由 FB 导数自然给出（盒内 ≈1、盒外 ≈0），不需要人工判据。
        """
        import ufl

        if not self.has_friction_at_build:
            raise RuntimeError("单体路径只在开摩擦时有意义")

        F_u, F_q = self.monolithic_residual(u, v, q_trial, None)

        # ---- (1,1) 与 (0,1)：对 q 求导 ----
        # dF_q/dq：UFL 从 smooth_clip_ufl 的表达式自动给出 FB 导数
        J_qq = ufl.derivative(F_q, self.q, q_du_trial)
        # dF_u/dq：q 是 Neumann 面力，导数就是 (τ·v)
        J_uq = ufl.derivative(F_u, self.q, q_du_trial)

        # ---- (0,0) 与 (1,0)：对 u 求导 ----
        J_uu = ufl.derivative(F_u, u, du_trial)
        J_qu = ufl.derivative(F_q, u, du_trial)

        return [[J_uu, J_uq], [J_qu, J_qq]]

    def set_slip_reference(self, u) -> None:
        """冻结当前切向位移作 s = τ·(u − u_prev) 的基准。

        **每个载荷步收敛后调一次**（由 ``PhaseFieldSolver.commit_step`` 调，
        循环工况下每个子步都调）。这样本步的相对位移才是「新增的滑移」，
        与 Coulomb 的增量语义一致；不冻结的话 s 会从零开始累计，第一次加载
        就把全部历史位移当成滑移，q 立刻饱和（实测 Q/(μP) = 1）。

        ⚠ **不碰 q**：循环工况下 q 是跨子步累积的状态（6.6.3 的疲劳驱动正是
        那个累积量），清零会毁掉周期内的摩擦记忆。若要从头重跑一段加载
        （如 U6 那样逐相位独立求解），调用方应显式 :meth:`reset_uzawa`。
        """
        if self.u_prev is None:
            raise RuntimeError("先调 bind_slip_space(V) 再 set_slip_reference")
        self.u_prev.x.array[:] = u.x.array
        self.u_prev.x.scatter_forward()

    def reset_uzawa(self) -> None:
        """清零 q 与 θ/Δq 历史（从头重跑一段加载时用，**不是**每个载荷步都调）。

        用途：一段**独立的**静力求解序列（每个相位各自平衡，q 不该跨相位继承）。
        跨相位继承会让下一段从上一段的饱和值出发，而 Δq 的符号历史又被那个旧值
        带成「单调」→ 逐单元 θ 判成真滑移、直接饱和到另一侧（实测 U6：
        第二个相位起 Q/(μP) 在 ±0.98 之间翻，而第一个相位是正确的 0.038）。
        循环工况**不要**调它：那会毁掉周期内累积的摩擦。
        """
        if self.q is not None:
            self.q.x.array[:] = 0.0
            self.q.x.scatter_forward()
        self._uzawa_theta_init = False
        self._uzawa_rho_prev = None

    def set_mu(self, mu: float | None) -> None:
        """运行时切 μ。``None`` 或 0 = 关摩擦（只改 Constant.value，不重建 form）。"""
        self.mu.value = 0.0 if mu is None else float(mu)
        self.has_friction = mu is not None and float(mu) > 0.0
        # 切向 Jacobian 的运行时开关跟着切（续 14）：μ = 0 时 q ≡ 0，
        # 切向刚度必须是 0，否则延拓期白扛 ε_t ≈ 1.4e16 的条件数（SNES -5）。
        # 用 Constant 而不是 Python bool：form 只编译一次，Python bool 在编译后
        # 不再被读（那正是第一版只判 has_friction_at_build 的原因）。
        self._tang_jac_gate.value = 1.0 if self.has_friction else 0.0
        # 重新开摩擦时让 θ 与残差历史重新起跑（_uzawa_theta0 的解析估计依赖 δ，
        # 而关摩擦期间 δ 可能变过）。q 场也清零：关摩擦期间 q ≡ 0 已被
        # uzawa_update 强制（lim = μp = 0 → 精确 clip），这里显式声明意图。
        self._uzawa_theta_init = False
        self._uzawa_rho_prev = None

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

    def tangential_jacobian(self, u, du_trial, v) -> ufl.Form:
        """Uzawa 模式下切向牵引对 u 的 Jacobian 贡献：ε_t·(τ·du)(τ·v)·ds（方向 f）。

        出处：Uzawa/ALM 的标准做法 —— 乘子项对偶变量的导数要进 Jacobian
        （Fortin & Glowinski 1983, *Augmented Lagrangian Methods*, §3；
         本项目的 ALM 实现见 D-68 问题 26）。

        **为什么需要**（D-68 问题 33 续 13）：切向项 ``q·τ·v`` 里 q 是 DG0
        Function、**不依赖 u**，故 ``ufl.derivative(F, u)`` 自动求导给出的
        切向刚度为 0。后果是 Uzawa 迭代中 q 一旦饱和到 Coulomb 容量
        （实测 it=4 时 843 个接触单元全部 ``|q| = μp``、``q_new|max`` 恰好
        等于 ``lim|max``），下一步 Newton 面对 843 个单元的**巨大面力却毫无
        切向刚度**，必然 SNES -5/-9。

        **加的是什么**：切向的「弹性」刚度 ε_t·(τ·du)(τ·v)。这是 Uzawa 迭代
        参数 ε_t 的本体 —— 它本就是切向罚刚度（``penalty_tangential_stiffness``），
        只是原来的实现把它只用在乘子更新里、没进 Jacobian。

        ⚠⚠ **必须在 μ 实际启用时才非零**（2026-10-25 修）：第一版只判
        ``has_friction_at_build``，而延拓期是「form 里装着切向项但 μ = 0」
        （``set_mu(None)``，见 :meth:`tangential_residual` 的说明）——
        那时 q ≡ 0、**不该有任何切向刚度**。第一版却无条件加了
        ε_t·(τ·du)(τ·v)，结果**延拓期就 SNES -5**（续 13 时延拓期完全正常），
        比不加更糟。

        这里的开关是 **Constant 而非 Python bool**：u 的 Newton form 只编译
        一次（续 14 的接线在 ``_build_u_problem`` 里），而加载序列是「先关摩擦
        延拓、再开摩擦推切向」。用 Python bool 判会永久二选一 ——
        判 ``has_friction_at_build`` 则延拓期白扛 ε_t（-5），
        判运行时 ``has_friction`` 则开摩擦后拿不回来（恒零）。
        故用 ``self._tang_jac_gate``（``set_mu`` 同步置 0/1）乘在整项上：
        form 结构恒定，值随 μ 走，与 ``delta`` / ``mu`` 同机制（D-16）。
        """
        if not self.has_friction_at_build:
            return ufl.as_ufl(0.0)
        if self.u_prev is None:
            raise RuntimeError(
                "开摩擦后必须先 bind_slip_space(V)（切向滑移基准场还没建）")

        tau_up, tau_lo = self._tangents(u)
        # τ·du 即切向滑移的线性化（忽略 dτ/du 的二阶项：τ 是接触面法向的
        # 旋转 90°，dτ/du ~ O(δ/a) ≪ 1，而 ε_t 项本身已是主导）。
        form = self.eps_t * ufl.dot(tau_up, du_trial) * ufl.dot(tau_up, v) \
            * self.ds(BLOCK_TOP)
        if tau_lo is not None and self._bottom_pressure(u) is not None:
            form = form + self.eps_t * ufl.dot(tau_lo, du_trial) \
                * ufl.dot(tau_lo, v) * self.ds(BLOCK_BOTTOM)
        return self._tang_jac_gate * form

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

    def _uzawa_theta0(self) -> float:
        """阻尼步长 θ₀ 的解析估计：``min(1, 2·K_t/ε_t)``（Fortin–Glowinski 1983）。

        Uzawa 定点映射 ``q ← clip(q − θ·ε_t·s)`` 的增益是 ``|1 − θ·ε_t·C|``，
        C 为接触斑的切向柔度（K_t 的倒数）。θ·ε_t·K_t ≈ 1 时一步到位，< 2 时
        几何收敛；θ = 1（原实现）时 ε_t·K_t ≈ **5e3**，每步把扰动放大五千倍，
        这就是黏着区二周期极限环（问题 26 续 5）与 6.6.3 第一个循环步爆炸的
        直接原因。

        K_t 用 Mindlin (1949) 的切向接触刚度 ``E*/(4a)``、``a = √(R·δ)``
        （Hertz 接触半宽，PLAN 6.1 已标定的口径），与阻尼只差一个常数因子无关紧要
        —— θ 自适应会按实际残差修正，这里只要量级对（2e-4 vs 2e-5 差 10 倍也只多
        几次迭代；方向不可能错：K_t 恒为正）。
        """
        if self.eps_t <= 0.0 or self.R_clamp <= 0.0:
            return 1.0
        E_star = 0.5 * self.eps * self.h_contact          # ε = κE/h → E* ≈ E/2（ν→0）
        if E_star <= 0.0:
            return 1.0
        a = float(np.sqrt(max(self.R_clamp * float(self.delta.value), 1.0e-30)))
        k_t = E_star / (4.0 * a)
        return float(min(1.0, 2.0 * k_t / self.eps_t))

    def uzawa_update(self, u, eps_t: float | None = None) -> float:
        """Uzawa/ALM 乘子更新（**光滑化 Coulomb + 阻尼**，D-68 问题 28 续）：

            q_{k+1} = smooth_clip(q_k − θ·ε_t·s ; μp),   θ ∈ (0, 1]

        返回**自然残差** ``max|ρ| / (μ·p_max)``，``ρ = q − smooth_clip(q − ε_t·s; μp)``
        （无阻尼的固定点残差）。黏着区 s→0 使 ρ→0，滑移区 |q| = μp 使 ρ→0，
        两端都正确归零；归一化用 Coulomb 容量 μ·p_max 而不是 max(1, max|q|)，
        因为 θ 变小会让 Δq 同比变小、拿 Δq 当判据会**假收敛**。

        为什么不是原来的无阻尼精确投影（2026-10-20 之前）：
        ``clip(q − ε_t·s)`` 的增益是 ``|1 − ε_t·C| ≈ 5e3``（C = 接触斑切向柔度），
        每步放大五千倍 → 黏着区二周期极限环 → 内圈 Newton 被振荡面力打死
        （实测 Uzawa 第 3 步残差 0.2→1.4e3、第 4 步 5e4，SNES -6/-9）。

        为什么光滑化而不是直接 clip：精确投影的盒边界是 C⁰ 折点，Newton 切线在
        黏着/滑移交界处不连续。FB 光滑化（Fischer 1992 / Kanzow 1996，见
        :func:`smooth_clip`）**盒内恒等**，故黏着区的解完全不受 η 影响 ——
        把内部点也拖向盒面的形式（如 μp·tanh(q/η)）会把黏着牵引整区抹掉，
        那是问题 26 续 5「ε_t 调小则 Q→0」的同一个坑，不能用。

        ``eps_t``：迭代参数（默认 ``self.eps_t``）。
        ⚠ 仅在 **Uzawa 模式**（``has_uzawa() == True``）下可用；罚函数模式返回 0.0。
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

        # 3) 阻尼步长 θ（Fortin–Glowinski 1983 的 Uzawa 步长控制）
        #    θ·ε_t·K_t ≈ 1 时一步收敛、< 2 时几何收敛；θ=1 的 ε_t·K_t ≈ 5e3 是
        #    极限环的根因。这里只用于**尚无符号历史的首步**（之后逐单元按
        #    单调/振荡切换，见第 4 步）。θ 跨载荷步复用：它由 ε_t/K_t
        #    （材料与网格量）决定，与当前载荷步几乎无关。
        if not self._uzawa_theta_init:
            self._uzawa_theta = self._uzawa_theta0()
            self._uzawa_theta_init = True
        theta = self._uzawa_theta

        # 4) q 更新：q_tr = q − θ·ε_t·s（负号的物理推导见 docstring）
        q_old = self.q.x.array.copy()
        lim = float(self.mu.value) * np.maximum(p_dg, 0.0)
        # η = η_rel·μp 逐单元自缩放。**lim = 0 的单元（非接触）走精确 clip**：
        # FB 光滑化在 lim=0 时给 −½(g+√(g²+η²)) ≈ η²/(4g) ≠ 0，会往非接触单元塞
        # 小伪切向力（smooth_clip docstring 的 ⚠）；精确 clip 得 q ≡ 0，正是想要的。
        contact = lim > 0.0
        eta = np.where(contact, self.uzawa_eta_rel * lim, 0.0)

        # 5) 自然残差 ρ_i = |q_i − clip_η(q_i − ε_t·s_i)|（**不含 θ**，故 θ 变小
        #    不会假收敛）。黏着区 s→0、滑移区 |q| = μp 都使 ρ→0。
        #    先按**当前** q 算一次，用作 θ 的判据（上次的 ρ 存在 _uzawa_rho_prev）。
        clip_target = np.where(contact, smooth_clip(q_old - eps_t * s_dg, lim, eta),
                               np.clip(q_old - eps_t * s_dg, -lim, lim))
        rho_here = np.abs(q_old - clip_target)
        rho_prev = self._uzawa_rho_prev

        # 6) θ **逐单元**按「本单元的残差是否在降」定（本设计的关键）：
        #     ρ_i 比上次小（真的在收敛 —— 滑移区的特征）→ θ_i = 1，全速走；
        #     ρ_i 没降（过冲/振荡 —— 黏着区增益 ε_t·C ≫ 2 的特征）→ θ_i 阻尼。
        #     逐单元而非全局：同一接触斑里黏着区与滑移区可以并存（部分滑移），
        #     一个全局 θ 必然牺牲其中一端。
        #   ⚠ 不用「Δq 符号是否单调」判（试过，2026-10-20）：被甩向盒面的过程
        #     Δq 也单调，两种情形分不开 —— 实测 U6 第二个相位起 Q/(μP) 饱和到
        #     ±0.98。残差下降才是收敛的充分特征。
        if rho_prev is None or rho_prev.shape != rho_here.shape:
            th_eff = np.full(q_old.shape, theta)
        else:
            th_eff = np.where(rho_here < rho_prev, 1.0, theta)

        q_tr = q_old - th_eff * eps_t * s_dg
        q_new = np.where(contact, smooth_clip(q_tr, lim, eta),
                         np.clip(q_tr, -lim, lim))

        # 7) 写回并散布
        self.q.x.array[:] = q_new
        self.q.x.scatter_forward()

        # 8) 更新后的残差（下次的判据基准）+ 全局归约
        # ⚠ 归约必须**所有 rank 都做**，只有 rank 0 打印（CLAUDE.md 陷阱 3）。
        rho = np.abs(q_new - clip_target)
        rho_max = float(np.max(rho)) if rho.size else 0.0
        p_max = float(np.max(p_dg)) if p_dg.size else 0.0
        rho_max = float(self.mesh.comm.allreduce(rho_max, op=MPI.MAX))
        p_max = float(self.mesh.comm.allreduce(p_max, op=MPI.MAX))
        scale = max(float(self.mu.value) * p_max, 1.0e-30)

        # 9) 存历史（逐单元的残差数组，供下次判「是否在降」）
        self._uzawa_rho_prev = rho

        return rho_max / scale

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
            # ⚠ 必须去重：二次基函数下 cell_dofs 会**重复列出**边中点（同一节点
            #   在该列表里出现两次），不去重则这个节点从同一单元拿到两份
            #   1/size 权重、平均被它主导（实测 U16 诊断的孤立 |q|/(μp) ≈ 2.01
            #   节点即此 bug，见 tests/test_clamp_uzawa.py 的 _scatter_dg0_to_top_nodes）。
            sel = np.unique(sel)
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
