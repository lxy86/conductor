"""AT1 相场断裂：能量泛函、拉压分解、交错求解器（PLAN.md Phase 1.1）。

模型（PLAN.md 第 2 节）
----------------------
E(u,d) = ∫ [g(d) psi+ + psi-] dV + ∫ f(alpha_bar) (3 Gc/8) [d/ell + ell |grad d|^2] dV
g(d) = (1-d)^2 + k_res

- 退化函数只作用在 psi+ 上（拉伸/剪切驱动开裂），psi- 不退化；
- AT1 的耗散密度对 d 是**线性**的（w(d) = d, c_w = 8/3），因此存在弹性阶段：
  均匀状态下 d 保持 0 直到 sigma 达到 sigma_c = sqrt(3 E Gc/(8 ell))，
  这条解析关系是 test_phasefield.py 的主要物理锚点；
- 不可逆性用 PETSc SNES `vinewtonrsls` 的变量界 [d_old, 1] 施加，不用 history field。

拉压分解
--------
amor（默认，Amor/Marigo/Maurini 2009，d 维体积模量 K_d = lambda + 2 mu/d）：
    psi+ = 0.5 K_d <tr eps>+^2 + mu dev(eps):dev(eps)
    psi- = 0.5 K_d <tr eps>-^2
spectral（Miehe et al. 2010，按主应变分解）：
    psi± = 0.5 lambda <tr eps>±^2 + mu sum_i <eps_i>±^2
两者都满足 psi+ + psi- = psi（无损伤弹性能），由单测钉住。

**u 问题其实是非线性的**：分解在 tr eps = 0（amor）或主应变换号（spectral）处有折点，
故 sigma(eps) 只是分段线性。两条路径：

- ``u_solver="linear"``（默认，PLAN 第 2 节要求的 LinearProblem + MUMPS）：
  把体积项的符号指示函数 chi = H(tr eps(u_prev)) **冻结**，能量重新成为 eps 的二次型，
  变分求两次导即得严格双线性形式。chi 的不动点迭代不额外开循环，直接折进交错循环：
  交错收敛时 chi 与 u 自洽，解满足真正的非线性平衡方程。诊断量 ``sign_flips``
  报告每次交错迭代中 chi 翻转的自由度数，为 0 即表示不动点已收敛。
  只支持 amor（spectral 的冻结需要特征投影，不做半对的实现）。
- ``u_solver="newton"``：对精确残差用 SNES Newton，两种分解都支持。
  用于校验 linear 路径（test_linear_and_newton_u_solvers_agree）。

失败模式
--------
子求解器静默不收敛是最难查的失败模式，因此：u 与 d 的每次求解都独立检查
PETSc 收敛原因，交错循环记录逐次迭代日志，未收敛时 ``StaggeredResult.converged``
为 False 且 ``failure_mode`` 非空——绝不静默返回一个坏解。

单位：SI。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from mpi4py import MPI

from .material import (
    DISSIPATION_MODELS,
    Material,
    dissipation_normalization,
)

__all__ = [
    "SPLITS",
    "DISSIPATION_MODELS",
    "FATIGUE_FORMS",
    "fatigue_degradation",
    "inverse_fatigue_degradation",
    "PhaseFieldModel",
    "SolverConfig",
    "StaggeredResult",
    "PhaseFieldSolver",
    "strain",
    "degradation",
    "energy_split",
    "psi_plus",
    "dissipation_density",
]

logger = logging.getLogger(__name__)

#: 支持的拉压分解
SPLITS: tuple[str, ...] = ("amor", "spectral")

Split = Literal["amor", "spectral"]
USolver = Literal["linear", "newton"]

#: 边界条件既可直接给 DirichletBC 序列，也可给一个「拿到函数空间再构造」的可调用对象。
#: **强烈建议用后者**：求解器自己持有 V_u/V_d，若在外部另建一个结构相同的空间上做 BC，
#: dolfinx 装配时不会把它认作本形式的 BC，矩阵不被约束 -> 奇异 -> MUMPS 仍报「收敛」
#: 却返回垃圾解。这是个静默失败，因此下面 _resolve_bcs 会强制校验空间归属。
BCSpec = (
    "Sequence[fem.DirichletBC] | Callable[[fem.FunctionSpace], Sequence[fem.DirichletBC]]"
)

#: 外力功（Neumann 项）同理：既可直接给 UFL 形式，也可给「拿到 V_u 再构造」的可调用对象。
#: **必须用后者**，除非你能保证形式里的 TestFunction 取自求解器自己的 ``V_u``——
#: 在外部另建一个结构相同的向量空间做 ufl.TestFunction，UFL 组装残差时会看到两个
#: 编号相同却不同源的 Argument，直接报
#: "Found different Arguments with same number and part"。
#: 与 BC 那个陷阱同源（见 BCSpec），只是这个是显式报错、不会静默出垃圾解。
WorkSpec = "ufl.Form | Callable[[fem.FunctionSpace], ufl.Form] | None"

#: 支持的疲劳退化函数形式。两者均为 Carrara, Ambati, Alessi & De Lorenzis 2020,
#: CMAME 361:112731（预印本 arXiv:1811.02244）的 Eq. (37) 与 Eq. (38)。
#: PLAN.md 第 2 节只写了 asymptotic；logarithmic 按 DECISIONS.md D-14 补入，
#: 因为只有它能把 S-N 斜率调到铝的实测值（原文：κ 提供 "a further degree of
#: freedom when simulating real material behaviors"）。
FATIGUE_FORMS: tuple[str, ...] = ("asymptotic", "logarithmic")

#: 2D 主应变闭式解里 sqrt 的正则化项。应变量级 ~1e-3，其平方 ~1e-6，
#: 故 1e-30 远低于任何物理量，只用来避免 sqrt(0) 处导数发散。
EIGENVALUE_REGULARIZATION = 1.0e-30

# --------------------------------------------------------------------------
# UFL 层：应变、退化、能量分解、耗散
# --------------------------------------------------------------------------
def strain(u):
    """小应变 eps = sym(grad u)。"""
    return ufl.sym(ufl.grad(u))

def degradation(d, k_res: float):
    """g(d) = (1-d)^2 + k_res。PLAN.md 第 2 节。"""
    return (1.0 - d) ** 2 + k_res

def _principal_strains_2d(eps):
    """2x2 对称张量的两个特征值（闭式）。

    e_{1,2} = c ± r,  c = tr/2,  r = sqrt(((e11-e22)/2)^2 + e12^2)。
    """
    c = 0.5 * ufl.tr(eps)
    half_diff = 0.5 * (eps[0, 0] - eps[1, 1])
    r = ufl.sqrt(half_diff**2 + eps[0, 1] ** 2 + EIGENVALUE_REGULARIZATION)
    return c + r, c - r

def energy_split(eps, material: Material, split: str, dim: int, *, freeze_from=None):
    """返回 (psi_plus, psi_minus)。

    ``freeze_from`` 为 None 时给出精确（分段二次）表达式；给一个应变张量时，
    体积项的符号按该张量冻结，结果对 ``eps`` 是严格二次型——这是 linear 路径
    生成双线性形式的办法。在不动点处（freeze_from 就是 eps 自身）两者相等，
    由 test_frozen_matches_exact_at_fixed_point 钉住。
    """
    if split not in SPLITS:
        raise ValueError("split 只能取 %r，收到 %r" % (SPLITS, split))
    # 用 ElasticProperties 的方法而不是自己调 lame_parameters(E, nu)，
    # 这样 plane_state（平面应变/平面应力）会自动传导进来
    lmbda, mu = material.elastic.lame
    tr_eps = ufl.tr(eps)

    if split == "amor":
        K = material.elastic.bulk(dim)
        dev = eps - (tr_eps / dim) * ufl.Identity(dim)
        dev_energy = mu * ufl.inner(dev, dev)
        if freeze_from is None:
            tr_plus = ufl.max_value(tr_eps, 0.0)
            tr_minus = ufl.min_value(tr_eps, 0.0)
            psi_plus = 0.5 * K * tr_plus**2 + dev_energy
            psi_minus = 0.5 * K * tr_minus**2
        else:
            # chi = H(tr eps_prev)：冻结后 tr_eps^2 的系数是常数，形式回到二次型
            chi = ufl.conditional(ufl.gt(ufl.tr(freeze_from), 0.0), 1.0, 0.0)
            psi_plus = 0.5 * K * chi * tr_eps**2 + dev_energy
            psi_minus = 0.5 * K * (1.0 - chi) * tr_eps**2
        return psi_plus, psi_minus

    # spectral
    if freeze_from is not None:
        raise NotImplementedError(
            "spectral 分解不提供冻结（双线性）形式：那需要特征投影算子，"
            "本项目按 DECISIONS.md 只把 spectral 作为对比选项，请用 u_solver='newton'。"
        )
    if dim != 2:
        raise NotImplementedError("spectral 分解目前只实现 2D（本项目为平面应变）")
    e1, e2 = _principal_strains_2d(eps)
    tr_plus = ufl.max_value(tr_eps, 0.0)
    tr_minus = ufl.min_value(tr_eps, 0.0)
    psi_plus = 0.5 * lmbda * tr_plus**2 + mu * (
        ufl.max_value(e1, 0.0) ** 2 + ufl.max_value(e2, 0.0) ** 2
    )
    psi_minus = 0.5 * lmbda * tr_minus**2 + mu * (
        ufl.min_value(e1, 0.0) ** 2 + ufl.min_value(e2, 0.0) ** 2
    )
    return psi_plus, psi_minus

def energy_split_3d(eps3, material: Material, split: str):
    """三维嵌入的 (psi_plus, psi_minus)，ε 为 3×3 的 ε_e（含 ε_e,zz，D-65）。

    只实现 spectral。ε 块对角（面内 2×2 块与 ε_zz），其特征值为 (e1, e2, ε_zz)，
    故三维谱分解是精确的，不是近似。ε_zz = 0 时逐式退化为 ``energy_split(…, dim=2)`` 的 spectral。
    amor 的三维形式未定义（2D 的 K = λ + μ 与 2D 偏量不能直接推广到含 ε_zz 的情形），按 NotImplementedError 拒绝。
    """
    if split != "spectral":
        raise NotImplementedError(
            "三维（含 ε_zz）能量分裂只实现 spectral（D-65）；%r 的三维形式未定义" % (split,)
        )
    lmbda, mu = material.elastic.lame
    tr3 = ufl.tr(eps3)
    e1, e2 = _principal_strains_2d(
        ufl.as_tensor([[eps3[0, 0], eps3[0, 1]], [eps3[1, 0], eps3[1, 1]]])
    )
    e3 = eps3[2, 2]
    tr_plus = ufl.max_value(tr3, 0.0)
    tr_minus = ufl.min_value(tr3, 0.0)
    psi_plus = 0.5 * lmbda * tr_plus**2 + mu * (
        ufl.max_value(e1, 0.0) ** 2 + ufl.max_value(e2, 0.0) ** 2 + ufl.max_value(e3, 0.0) ** 2
    )
    psi_minus = 0.5 * lmbda * tr_minus**2 + mu * (
        ufl.min_value(e1, 0.0) ** 2 + ufl.min_value(e2, 0.0) ** 2 + ufl.min_value(e3, 0.0) ** 2
    )
    return psi_plus, psi_minus

def stress_split_3d(eps3, material: Material, split: str, d, k_res: float,
                    *, degrade_stress: bool = True):
    """σ_deg = g(d) ∂ψ⁺/∂ε + ∂ψ⁻/∂ε，ε 为三维 ε_e，返回 3×3 UFL 张量（D-65）。

    对 ε_e 求导，不对 u 求导：返回映射 ε_p(u) 对 u 的依赖由 u 方程的 AD 另外计入。

    ``degrade_stress=False``（b-1′ 口径，D-68 问题 5/6）时**不乘** g(d)，
    即 σ_deg = ∂ψ⁺/∂ε + ∂ψ⁻/∂ε —— 弹性刚度不因损伤退化。
    注意这与**屈服面**的退化是两件事：屈服面的 g 由 ``PlasticLaw.update(u, g=...)``
    的 ``g`` 实参决定（见 ``PhaseFieldSolver._sigma_deg_2d``），不受本开关影响。

    ``d`` 为 None 且 ``degrade_stress=True`` 时等价于退化因子取 1（无历史场时用）。
    """
    var = ufl.variable(eps3)
    psi_plus, psi_minus = energy_split_3d(var, material, split)
    if degrade_stress:
        if d is None:
            raise ValueError(
                "degrade_stress=True 需要 d（D-65 的 σ_deg = g(d)σ⁺ + σ⁻）。"
                "若要不退化，请传 degrade_stress=False。"
            )
        return degradation(d, k_res) * ufl.diff(psi_plus, var) + ufl.diff(psi_minus, var)
    return ufl.diff(psi_plus, var) + ufl.diff(psi_minus, var)

def psi_plus(eps_elastic, split: str, lam: float, mu: float):
    """弹性应变能的拉伸部分 ψ⁺(ε_e)（PLAN §1.5，D-60）。

    ``eps_elastic`` 为 2×2 UFL 张量，取 ε_e = ε − ε_p；无塑性历史时 ε_p = 0，即 ε_e = ε。
    ``lam``、``mu`` 为平面应变 Lamé 常数（``material.elastic.lame``）。

    - amor：ψ⁺ = ½K⟨tr ε_e⟩₊² + μ dev(ε_e):dev(ε_e)，K = λ + μ（2D 体模量）；
    - spectral：ψ⁺ = λ/2 ⟨tr ε_e⟩₊² + μ Σ⟨e_i⟩₊²，与 energy_split 同式。

    与 ``energy_split(...)[0]`` 在 2D 下逐式一致（tests/test_psi_plus_elastic.py 对拍）。
    本函数只定义 ψ⁺ 的输入，求解器尚未把 ε_e 接入 u、d 两个方程（6.5 的耦合）。
    """
    if split not in SPLITS:
        raise ValueError("split 只能取 %r，收到 %r" % (SPLITS, split))
    tr_eps = ufl.tr(eps_elastic)
    tr_plus = ufl.max_value(tr_eps, 0.0)
    if split == "amor":
        dev = eps_elastic - (tr_eps / 2.0) * ufl.Identity(2)
        return 0.5 * (lam + mu) * tr_plus**2 + mu * ufl.inner(dev, dev)
    e1, e2 = _principal_strains_2d(eps_elastic)
    return 0.5 * lam * tr_plus**2 + mu * (
        ufl.max_value(e1, 0.0) ** 2 + ufl.max_value(e2, 0.0) ** 2
    )

def elastic_energy_density(eps, d, material: Material, split, dim, k_res, *,
                           freeze_from=None, degrade_stiffness: bool = True):
    """ψ⁻ + g(d)·ψ⁺（默认）或 ψ⁻ + ψ⁺（``degrade_stiffness=False``，b-1 口径）。

    ``degrade_stiffness=False`` 只改**刚度**这一项：材料不再因损伤软化，于是
    u 的解与 d 的取值无关，交错迭代里 d↑→刚度↓→u↑ 的反馈环被切断
    （D-68 问题 5）。d 方程自身的驱动力与屈服面**仍**保留 g(d)
    （见 PhaseFieldModel.degrade_stiffness 的说明），故不可逆与 B 口径不受影响。

    注意 ``k_res`` 不复用时仍要校验：调用方只有在 degrade_stiffness=True 时
    才会取用 g(d)，但签名保持不变以免各调用点分叉。
    """
    psi_plus, psi_minus = energy_split(
        eps, material, split, dim, freeze_from=freeze_from
    )
    if degrade_stiffness:
        return degradation(d, k_res) * psi_plus + psi_minus
    return psi_plus + psi_minus

def _check_fatigue_args(alpha_bar, alpha_T, form, kappa):
    if form not in FATIGUE_FORMS:
        raise NotImplementedError(
            "疲劳退化形式只支持 %r（Carrara et al. 2020 的 Eq. 37/38），收到 %r。"
            "要加别的形式必须先给出文献出处。" % (FATIGUE_FORMS, form)
        )
    if alpha_T is None:
        raise ValueError("alpha_T 为 None：未标定的 alpha_T 不得进入计算（DECISIONS.md D-3）")
    alpha_T = float(alpha_T)
    if alpha_T <= 0.0:
        raise ValueError("alpha_T 必须为正，收到 %r" % (alpha_T,))
    if form == "logarithmic":
        if kappa is None:
            raise ValueError("form='logarithmic' 必须给 kappa（Carrara 2020 Eq. 38 的材料参数）")
        kappa = float(kappa)
        if kappa <= 0.0:
            raise ValueError("kappa 必须为正，收到 %r" % (kappa,))
    if isinstance(alpha_bar, (int, float)) and alpha_bar < 0.0:
        raise ValueError("alpha_bar 不能为负，收到 %r" % (alpha_bar,))
    return alpha_T, (None if kappa is None else float(kappa))

def _scalar_value(x):
    """取标量值：float 原样返回，``fem.Constant`` 取 ``.value``，None 保持 None。

    只用于**数值校验**与纯标量分支；进 UFL 的仍是原对象（见 fatigue_degradation）。
    """
    if x is None or isinstance(x, (int, float)):
        return x
    value = getattr(x, "value", None)
    if value is None:
        raise TypeError(
            "alpha_T/kappa 只能是 float 或带 .value 的 fem.Constant，收到 %r" % (type(x),)
        )
    return float(value)

def fatigue_degradation(
    alpha_bar, alpha_T: float, form: str = "asymptotic", kappa: float | None = None
):
    """疲劳退化函数 f(alpha_bar)。

    出处：Carrara, Ambati, Alessi & De Lorenzis 2020, CMAME 361:112731，
    预印本 arXiv:1811.02244 的 Eq. (37)（asymptotic，即 PLAN.md 第 2 节所写）与
    Eq. (38)（logarithmic）。Table 1 亦列出这两式。

    asymptotic（Eq. 37）::

        f = 1                                   if a <= a_T
        f = (2 a_T/(a + a_T))^2                 if a >= a_T

    实现写成 ``min(1, (2a_T/(a+a_T))^2)``，与分段定义**严格等价**（因 a >= 0 时
    该平方项 >= 1 当且仅当 a <= a_T），避免 conditional。a = a_T 处 f = 1 连续。

    logarithmic（Eq. 38）::

        f = 1                                   if a <= a_T
        f = [1 - kappa log10(a/a_T)]^2          if a_T <= a <= a_T 10^(1/kappa)
        f = 0                                   if a >= a_T 10^(1/kappa)

    对数取 **10 为底**：这由原文的截止点 a_T·10^(1/kappa) 自洽确定
    （代入得 [1 - kappa·(1/kappa)]^2 = 0）。实现上把 log 的自变量夹到 >= 1
    （``max(a, a_T)/a_T``），于是 a <= a_T 时自动给出 f = 1，既省一层 conditional
    又避免 a = 0 处 log 发散；截止之后由 conditional 强制归零（否则平方项会回升）。

    kappa 只对 logarithmic 有意义，它调的正是 S-N 曲线的斜率——asymptotic 形式的
    斜率被锁死在 1/3，与铝实测的 ~0.13 不符，见 DECISIONS.md D-14。

    ``alpha_bar`` 可以是 float（单测直接算）或 UFL 表达式/Function（进变分形式）。

    ``alpha_T`` 与 ``kappa`` 除了 float，也可以传 ``fem.Constant``。
    **标定时必须传 Constant**：它们若以 Python float 进 UFL，就会被 FFCx 当成
    字面量烘进 form 签名，每换一个试探值都要重新编译一遍（见 DECISIONS.md D-16，
    实测一次编译数秒到数十秒），而 Phase 2.3 的标定要反复改这两个数。
    传 Constant 时只改 ``.value``，form 不变、不触发编译。
    """
    # fem.Constant 取 .value 做数值校验，但**进表达式的仍是 Constant 本身**
    alpha_T_val = _scalar_value(alpha_T)
    kappa_val = _scalar_value(kappa)
    _check_fatigue_args(alpha_bar, alpha_T_val, form, kappa_val)
    is_scalar = isinstance(alpha_bar, (int, float))
    if is_scalar:
        # 纯标量分支：一律用数值，免得把 Constant 混进 math.log10
        alpha_T, kappa = alpha_T_val, kappa_val

    if form == "asymptotic":
        ratio_sq = (2.0 * alpha_T / (alpha_bar + alpha_T)) ** 2
        return min(1.0, ratio_sq) if is_scalar else ufl.min_value(1.0, ratio_sq)

    # logarithmic
    if is_scalar:
        cutoff = alpha_T * 10.0 ** (1.0 / kappa)
        if alpha_bar >= cutoff:
            return 0.0
        return (1.0 - kappa * math.log10(max(alpha_bar, alpha_T) / alpha_T)) ** 2
    # 符号分支：kappa 可能是 Constant，故 10^(1/kappa) 写成 exp(ln10/kappa)
    cutoff = alpha_T * ufl.exp(math.log(10.0) / kappa)
    ratio = ufl.max_value(alpha_bar, alpha_T) / alpha_T
    value = (1.0 - kappa * ufl.ln(ratio) / math.log(10.0)) ** 2
    return ufl.conditional(ufl.ge(alpha_bar, cutoff), 0.0, value)

def inverse_fatigue_degradation(
    f_target: float,
    alpha_T: float,
    form: str = "asymptotic",
    kappa: float | None = None,
) -> float:
    """给定 f 求 alpha_bar（f 单调不增，故解唯一）。解析寿命公式要用。

    asymptotic:   a = a_T (2/sqrt(f) - 1)
    logarithmic:  a = a_T 10^((1 - sqrt(f))/kappa)
    两式由 Eq. (37)/(38) 直接反解，f = 1 时都给 a = a_T。
    """
    alpha_T, kappa = _check_fatigue_args(0.0, alpha_T, form, kappa)
    f_target = float(f_target)
    if not 0.0 < f_target <= 1.0:
        raise ValueError("f_target 必须在 (0,1] 内，收到 %r" % (f_target,))
    root = math.sqrt(f_target)
    if form == "asymptotic":
        return alpha_T * (2.0 / root - 1.0)
    return alpha_T * 10.0 ** ((1.0 - root) / kappa)

def dissipation_density(d, material: Material, f_alpha, model: str = "AT1"):
    """f(alpha_bar) (G_c/c_w) [w(d)/ell + ell |grad d|^2]。

    - AT1（PLAN 第 2 节，生产工况）：w = d、c_w = 8/3 -> 前因子 3 G_c/8；
    - AT2（仅用于复现文献算例）：w = d^2、c_w = 2 -> 前因子 G_c/2，
      即 Miehe/Bourdin 的经典写法 G_c[d^2/(2 ell) + (ell/2)|grad d|^2]。
    """
    if model not in DISSIPATION_MODELS:
        raise ValueError(
            "dissipation 只能取 %r，收到 %r" % (DISSIPATION_MODELS, model)
        )
    Gc = material.fracture.Gc
    ell = material.fracture.ell
    w = d if model == "AT1" else d**2
    prefactor = Gc / dissipation_normalization(model)
    return f_alpha * prefactor * (
        w / ell + ell * ufl.inner(ufl.grad(d), ufl.grad(d))
    )

# --------------------------------------------------------------------------
# 配置与结果
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PhaseFieldModel:
    """相场模型的离散无关部分。"""

    material: Material
    split: Split = "amor"
    k_res: float = 1.0e-6
    dissipation: str = "AT1"
    degrade_stiffness: bool = True
    """g(d) 是否同时退化**弹性刚度**（默认 True，即 PLAN §2 / D-2 的原口径）。

    三个位置原本都乘同一个 g(d) = (1−d)² + k_res：
      1. u 方程的弹性能 g(d)ψ⁺ + ψ⁻（本项 = 刚度）；
      2. d 方程的驱动力 g(d)ψ⁺；
      3. 屈服面 σ_deg = g(d)∂ψ⁺/∂ε + ∂ψ⁻/∂ε（D-62/D-63 的 B 口径）。

    ``degrade_stiffness=False``（b-1 口径，D-68 问题 5）只把第 1 项换成未退化的
    ψ⁺，第 2、3 项的 g(d) **保留**。

    **为什么需要这个开关**：u 与 d 交错时第 1 项构成正反馈环

        d ↑ → g(d) ↓ → 刚度 ↓ → u ↑ → ψ⁺ ↑ → d ↑

    D-68 实测（N=94.2773、峰值相位）：max_d 五轮从 0.084 冲到 1.0，
    ψ⁺ 同时涨 63%（1.63e5 → 2.65e5 Pa），而 max|ε_p| ≡ 0
    —— 即塑性未激活，ψ⁺ 增长纯由 u 增大而来。均匀区驻定分析给出的内部解
    只有 d* ≈ +0.24，是耦合反馈把它推成发散。断开第 1 项即可打断该环。

    ⚠ 这是**新口径**，PLAN §2 未列（PLAN 的公式含 g(d)ψ⁺）。启用时必须在论文中
    说明这一偏离及其依据；默认 False 之外的一切既有算例都仍按原口径。
    """
    damage_region: object = None
    """可选：损伤计算的可信区掩码（D-68 路线 A，A-1 + A-b）。

    ``None``（默认）= 全域参与，既有行为逐位不变。给定时须是一个 DG0 的
    :class:`dolfinx.fem.Function`，逐单元取 0/1：为 0 的单元里 d 方程的驱动力
    ψ⁺ 被置 0，于是那里的 d 恒为 0。

    **为什么需要它（D-68 问题 12/13）**：相场只有在 h ≲ ℓ 处才分辨得出裂纹带。
    本工况 h/ℓ = 42~945 的粗网格区里，ψ⁺ 反推出的等效应力达 451 MPa，
    是 σ_y0 = 167 MPa 的 2.7 倍 —— 超出 J2 屈服面上界 σ_y²/(6E) **7.3 倍**，
    在 J2 模型里根本不可能存在，是离散误差。用它当起裂判据会把
    「一个不该起裂的状态」推进成 d 的棘轮发散（N=94.2773 的失败）。

    **A-1 口径（只屏蔽驱动力，不冻结 ᾱ）**：区外 d 恒为 0，但 ᾱ 照旧累积。
    语义是「这里分辨不了，不判起裂」，而不是「这里没有疲劳」——
    后者会丢掉可能真实的远场疲劳，故不作默认。ᾱ 保留下来也便于日后检验。

    **A-b 掩码标法**：按**实测单元尺寸**判定（h_local ≤ 阈值），而不是按
    refine_boxes 的矩形盒 —— 掩码口径与上面「h/ℓ」的依据一致，
    也避免盒边界与实际尺寸分布不符。见 :func:`pfwire.mesh_h.damage_region_mask`。
    """

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError("split 只能取 %r，收到 %r" % (SPLITS, self.split))
        if not 0.0 < self.k_res < 1.0:
            raise ValueError("k_res 应在 (0,1) 内，收到 %r" % (self.k_res,))
        if self.dissipation not in DISSIPATION_MODELS:
            raise ValueError(
                "dissipation 只能取 %r，收到 %r"
                % (DISSIPATION_MODELS, self.dissipation)
            )
        if self.damage_region is not None:
            # DG0 的 0/1：逐单元掩码，乘进 form 时不需要改求积规则。
            # 用别的空间会静默地插值错（DG2 的 dof 顺序与坐标表不对齐，
            # D-68 探针 29 实测过），故在构造时就拒绝。
            from dolfinx.fem import Function as _Fn
            if not isinstance(self.damage_region, _Fn):
                raise TypeError(
                    "damage_region 必须是 DG0 的 fem.Function（逐单元 0/1 掩码），"
                    "收到 %r。可用 pfwire.mesh_h.damage_region_mask(msh, h_max) 生成。"
                    % (type(self.damage_region).__name__,)
                )
            # dolfinx 0.11 的 basix_element 才有 family/degree/discontinuous；
            # FiniteElement 本身没有 family_name（实测 AttributeError）。
            # 且 DG 与 Lagrange 的 basix family 都是 P，故**必须**看
            # discontinuous 才能区分 DG0 与 P1。
            el = self.damage_region.function_space.element
            bx = el.basix_element
            deg = int(bx.degree)
            disc = bool(bx.discontinuous)
            if deg != 0 or not disc:
                raise TypeError(
                    "damage_region 必须是 DG0（degree=0 且不连续），"
                    "收到 degree=%d discontinuous=%s。" % (deg, disc)
                )
        # 材料卡里的 sigma_c 是按某个耗散模型换算的，与模型不一致就是参数错配，
        # 必须拒绝：AT1 的 sigma_c 与 AT2 的 sigma_y 是两回事（差 (9/16)/sqrt(8/9) 之类的因子）。
        if self.material.fracture.dissipation != self.dissipation:
            raise ValueError(
                "模型用 dissipation=%r，但材料卡的 fracture.dissipation=%r。"
                "两者的 sigma_c <-> (G_c, ell) 换算关系不同，不能混用。"
                % (self.dissipation, self.material.fracture.dissipation)
            )

@dataclass(frozen=True)
class SolverConfig:
    """交错求解与两个子求解器的设置。

    ``tol_d`` / ``max_iter`` 即 PLAN 第 2 节的 ‖Δd‖_inf < 1e-4、最多 50 次。
    """

    tol_d: float = 1.0e-4
    max_iter: int = 50
    u_solver: USolver = "linear"
    degree_u: int = 1
    degree_d: int = 1
    snes_atol: float = 1.0e-8
    snes_rtol: float = 1.0e-6
    snes_max_it: int = 200
    snes_atol_u: float | None = None
    """u 的 Newton 容差。**None 时沿用 snes_atol/snes_rtol**（既有行为不变）。

    为什么需要分开：u 与 d 的容差量级差得很远。d 问题是二次的，残差会掉到
    装配的舍入地板上，故要 atol=1e-8/rtol=1e-6（见 snes_atol 的注释）；
    而 u 的弹塑性 + 罚接触问题残差量级大得多，6.1 标定的是
    atol=1e-6/rtol=1e-10（indent_plastic_obstacle.yaml）。共用一组时，
    取 6.1 的值会让 d 报 SNES -6（残差 8.5e-4 停住而 atol 只有 1e-6，
    差 850 倍），取 d 的值又会让 u 过不了屈服点。
    """
    snes_rtol_u: float | None = None
    """d 的 SNES 容差。**不要再调紧**：d 问题是二次的（残差线性），残差会掉到
    装配的舍入地板上就不动了。实测（h=ell/4、16229 个 d 自由度、AT2、Gc=2700、
    ell=15 um）：某个载荷步的残差历史为 6.27e-4 -> 9.66e-10 -> 6.34e-10 ->
    ... -> 1.04e-10 后卡住约 95 次迭代。原先 atol=1e-10 正压在地板上、
    rtol=1e-9 对应的目标 6.3e-13 更在地板之下，于是必然撞上 max_it。
    现取 atol=1e-8（约为地板的 100 倍）、rtol=1e-6，仍有 5 个数量级的下降余量。
    症状是**越细越早死**：载荷步越细、网格越密 -> ||F_0|| 越小、地板越高
    -> 相对目标越先掉到地板下。"""
    u_linesearch: str = "bt"
    """u 的 Newton 线搜索类型（PETSc ``-snes_linesearch_type``）。

    默认 ``bt``（回溯）即 PETSc 的默认值。**spectral 分解下可能要改成 ``basic``**：
    Miehe 谱分解的 psi+ 只在主应变过零处 C^0，u 的残差因此是分片光滑的，
    而回溯线搜索假设 merit 函数光滑，会在折点处找不到下降方向、
    报 SNESConvergedReason=-6（DIVERGED_LINE_SEARCH），哪怕整步 Newton 本身没问题。
    ``basic`` 直接取整步、不做线搜索。改的是求解算法，不动方程与参数。
    """
    u_linesearch_cyclic: str | None = None
    """进入**循环相位**后可选切换到的线搜索类型（**None = 不切换**，默认）。

    保留的旋钮，**当前生产工况不使用**（2026-10-21 实测无效）：
    延拓/W' 搜索阶段必须 ``bt``（``basic`` 在延拓上直接 SNES -5），而循环相位
    遇到谱分解折点（ψ+ 在过零处 C0，D-31）时 ``bt`` 报 -6。看似该在这里切
    ``basic``，但实测**同点位只把 -6 变成 -9（Jacobian 奇异）** —— 那才是
    6.6.3 的 blocker（法向-切向耦合不稳定，见 DECISIONS.md D-68 问题 28 续 2），
    该修的是接触格式而非线搜索。

    仍保留此字段与方法，因为：切换机制本身已验证可用
    （``snes.getLineSearch().setType``，不重建 Jacobian），若将来某工况确实
    只需要切线搜索即可通过（例如纯谱分解折点、无接触病态），不必重新开发。
    切换由调用方在 ramp→cyclic 边界通过
    :meth:`PhaseFieldSolver.set_u_linesearch` 完成。
    """
    log_iterations: bool = False  # 逐迭代日志（排查静默不收敛）
    relaxation: float = 1.0
    """交错更新中 d 的松弛因子：d <- d_prev + omega (d_solved - d_prev)。
    默认 1.0（不松弛）。极限点附近交错格式可能**振荡**（‖Δd‖ 停在 1e-2 量级不降），
    此时取 omega ~ 0.5 可压住振荡；不动点不变，故不影响解的正确性。
    因 d_prev 与 d_solved 都 >= d_old，凸组合仍满足不可逆下界。"""
    u_relaxation: float = 1.0
    """Uzawa 内循环中 u 的松弛因子：u <- u_iter_prev + omega_u (u_solved - u_iter_prev)。
    默认 1.0（不松弛）。**专治开摩擦后 u 在 Uzawa 正反馈环里漂移**
    （D-68 问题 33 续 3）：s = τ·(u − u_prev_clamp) 完全由 u 决定，
    而 u 的 Newton 解受 q 面力影响，q 又由 s 更新 ——
    s↑ → q↑ → 面力↑ → u↑ → s↑。实测 s 在同轮 Uzawa 的三次迭代里涨
    13 个数量级（3.55e-15 → 3.85e-08 → 1.08e-06），θ₀ 阻尼按 ε_t·C = 2
    设计、只在 s 有界时有效，s 无界增长时无从发挥。
    对 u 欠松弛把每轮 Δu 压 omega_u 倍，s 的增量同比压小，
    正反馈增益从 1 降到 omega_u。**不动点不变**（收敛时
    u_solved = u_iter_prev，凸组合不起作用），故不影响解的正确性。
    只在 u_solver='newton' 且 contact.has_uzawa() 时生效。"""
    monolithic_friction: bool = False
    """开摩擦时用**单体** Newton 解 (u, q) 混合系统，而非交错 Uzawa（默认 False）。

    出处与实测（D-68 问题 33 续 13~18）：
    交错格式在**深度黏着**下进二周期极限环（问题 26 续 5），实测 tiny 网格上
    跑 2000 次迭代、连续快照间 ``du ≈ 1e-15``（u 冻住）但 ``dq = 0.268`` 恒定
    —— q 永远以固定幅度振荡，不收敛。粗网格 6.6.3 的第一个循环子步因此
    SNES -5（blocker B）。

    单体格式把 (u, q) 当一个系统、用联立 Jacobian 一次解出：
    黏着区靠 ``dF_q/du = ε_t·ds/du``（实测 1.20e11，**中心差分验证**）把 u 钉住，
    系统非奇异；实测 13 步收敛到 8.6e-4，且 Coulomb 锥在积分口径下
    **精确满足**（|∫q|/∫μp = 1.0000）。

    ⚠ 实现走 :meth:`pfwire.clamp_contact.ClampContact.solve_monolithic`
    （**手工块 Newton**，见 ``pfwire.monolithic``），不用 dolfinx 的多 form
    ``NonlinearProblem`` —— 后者有两个叠加的 bug（续 17 坑 7）。

    ⚠ 当前局限：**部分滑移形貌尚未达成**。实测接触斑整体滑移
    （68 个接触单元只有 1 个黏着），幅值 8 倍变化 Q 只动 0.06%
    —— 那是续 7 的 blocker A（幅值 vs ``μp/ε_t`` 容量），不是求解器问题。
    """
    monolithic_tol: float = 1.0e-8
    """单体 Newton 的收敛容差（自由 dof 上的残差 2-范数）。

    只在 ``monolithic_friction=True`` 时生效。
    ⚠ **不是越小越好**：d 的残差会掉到装配舍入地板上，u/q 的残差量级差
    10 个量级（实测 u 段 1e2、q 段 1e-5），取一个统一的绝对容差会偏向大的一侧。
    """
    monolithic_max_it: int = 50
    """单体 Newton 的最大迭代数（默认 50）。"""

    direct_solver_options: dict[str, object] = field(
        default_factory=lambda: {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    )

    def __post_init__(self) -> None:
        if self.u_solver not in ("linear", "newton"):
            raise ValueError("u_solver 只能是 'linear' 或 'newton'，收到 %r" % (self.u_solver,))
        if self.tol_d <= 0 or self.max_iter < 1:
            raise ValueError("tol_d 必须为正、max_iter >= 1")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("relaxation 须在 (0,1] 内，收到 %r" % (self.relaxation,))
        if not 0.0 < self.u_relaxation <= 1.0:
            raise ValueError("u_relaxation 须在 (0,1] 内，收到 %r" % (self.u_relaxation,))
        if self.u_linesearch not in ("bt", "basic", "l2", "cp", "nleqerr"):
            raise ValueError(
                "u_linesearch 只能是 PETSc 的 bt/basic/l2/cp/nleqerr，收到 %r"
                % (self.u_linesearch,)
            )
        if self.u_linesearch_cyclic is not None and self.u_linesearch_cyclic not in (
            "bt", "basic", "l2", "cp", "nleqerr"
        ):
            raise ValueError(
                "u_linesearch_cyclic 只能是 PETSc 的 bt/basic/l2/cp/nleqerr，收到 %r"
                % (self.u_linesearch_cyclic,)
            )

@dataclass
class StaggeredResult:
    """一个载荷步的交错求解结果。``converged`` 为 False 时 ``failure_mode`` 必非空。"""

    converged: bool
    iterations: int
    delta_d_inf: float
    sign_flips: int
    max_d: float
    energies: dict[str, float]
    failure_mode: str | None = None
    history: list[tuple[int, float, int]] = field(default_factory=list)

def _space_contains(space: fem.FunctionSpace, other) -> bool:
    """``space`` 是否包含 ``other``（同一空间或其子空间）。

    ``DirichletBC.function_space`` 返回的是 cpp 层对象而非 Python 包装，
    直接调 ``FunctionSpace.contains`` 会去找它不存在的 ``_cpp_object``，故在此归一。
    """
    return space._cpp_object.contains(getattr(other, "_cpp_object", other))

def _resolve_external_work(spec, space: fem.FunctionSpace):
    """把 ``external_work`` 解析成 UFL 形式；可调用对象用求解器自己的 ``V_u`` 调用。

    为什么要这一步：Neumann 项里的 TestFunction 必须取自求解器持有的 ``V_u``。
    在外部另建一个结构相同的向量空间做 ``ufl.TestFunction``，UFL 会看到两个编号
    相同却不同源的 Argument 并报 "Found different Arguments with same number and
    part"（与 BC 那个静默陷阱同源，只是这个会显式炸）。而调用方拿不到 ``V_u``——
    它是构造函数里才建的 —— 所以按本项目 ``bcs_u=lambda V: [...]`` 的既有约定，
    允许传 ``external_work=lambda V: ...``。

    注意判断顺序：``ufl.Form`` 自身也实现了 ``__call__``（用于系数替换），
    故必须**先**判断是不是 Form，再判断 callable，否则会把现成的形式当工厂调用。
    """
    if spec is None:
        return None
    if isinstance(spec, ufl.Form):
        return spec
    if callable(spec):
        return spec(space)
    raise TypeError(
        "external_work 只能是 ufl.Form、接受函数空间的可调用对象或 None，收到 %r"
        % (type(spec),)
    )

def _resolve_bcs(spec, space: fem.FunctionSpace, label: str) -> list[fem.DirichletBC]:
    """把 BCSpec 归一成列表，并校验每个 bc 都属于 ``space``（或其子空间）。"""
    bcs = list(spec(space)) if callable(spec) else list(spec)
    for bc in bcs:
        if not _space_contains(space, bc.function_space):
            raise ValueError(
                "%s 里有边界条件不属于求解器自己的函数空间。"
                "dolfinx 会静默忽略它，导致矩阵缺约束、奇异，而直接解仍报『收敛』并返回垃圾解。"
                "请把 bcs 写成可调用对象：bcs_%s=lambda V: [...]，"
                "或改用 solver.%s 构造 BC。" % (label, label, label.upper())
            )
    return bcs

# --------------------------------------------------------------------------
# 交错求解器
# --------------------------------------------------------------------------
class PhaseFieldSolver:
    """u（弹性）与 d（相场）的交错求解，d 用 SNES 变分不等式保证不可逆。

    典型用法::

        solver = PhaseFieldSolver(mesh, model, bcs_u=bcs, external_work=L)
        for step in load_steps:
            load_constant.value = step
            result = solver.solve_step()
            if not result.converged:
                raise RuntimeError(result.failure_mode)
            solver.commit_step()      # d_old <- d，冻结不可逆下界
    """

    def __init__(
        self,
        mesh: dmesh.Mesh,
        model: PhaseFieldModel,
        *,
        bcs_u: BCSpec = (),
        bcs_d: BCSpec = (),
        external_work=None,
        f_alpha: fem.Function | fem.Constant | None = None,
        config: SolverConfig | None = None,
        plastic=None,
        contact=None,
    ) -> None:
        """``plastic``：可选 :class:`pfwire.plasticity.PlasticLaw`（D-61/D-62）。

        给了则 u、d 两个方程的弹性应变取 ε_e = ε − ε_p(u; g(d))，屈服按退化应力判断（B），
        且 u 只能用 ``u_solver='newton'``（返回映射在残差里，不是冻结形式）。

        ``contact``：可选接触对象，须提供 ``residual(u, v) -> ufl.Form``，表示接触力的虚功
        （见 :class:`pfwire.clamp_contact.ClampContact`，D-67）。它只出现在 u 方程里，
        取负号加入残差，切线由 AD 给出；同样要求 ``u_solver='newton'``。
        """
        self.mesh = mesh
        self.model = model
        self.config = config or SolverConfig()
        if self.config.u_solver == "linear" and model.split != "amor":
            raise ValueError(
                "u_solver='linear' 只支持 split='amor'（spectral 的冻结形式未实现）；"
                "请改用 u_solver='newton'。"
            )
        if plastic is not None and self.config.u_solver == "linear":
            raise ValueError(
                "带塑性（plastic）时 u_solver 必须是 'newton'：返回映射不是冻结形式。"
            )
        if contact is not None and self.config.u_solver == "linear":
            raise ValueError("接触（contact）时 u_solver 必须是 'newton'：接触力依赖 u，不是冻结形式。")
        self.contact = contact
        if plastic is not None and model.split != "spectral":
            raise NotImplementedError(
                "带塑性时只支持 split='spectral'（D-65）：amor 的三维（含 ε_e,zz）形式未定义。"
            )
        if plastic is not None and mesh.geometry.dim != 2:
            raise NotImplementedError("带塑性时只支持二维平面应变（PlasticLaw 为 2D 嵌入）。")
        self.plastic = plastic
        self.dim = mesh.geometry.dim
        self.comm = mesh.comm
        # 先建函数空间，再解析边界条件（BC 必须建在这两个空间上，见 _resolve_bcs）
        self.V_u = fem.functionspace(
            mesh, ("Lagrange", self.config.degree_u, (self.dim,))
        )
        self.V_d = fem.functionspace(mesh, ("Lagrange", self.config.degree_d))
        self.bcs_u = _resolve_bcs(bcs_u, self.V_u, "u")
        self.bcs_d = _resolve_bcs(bcs_d, self.V_d, "d")

        self.u = fem.Function(self.V_u, name="u")
        self.u_prev = fem.Function(self.V_u, name="u_prev")  # 冻结 chi 用
        self.d = fem.Function(self.V_d, name="d")
        self.d_old = fem.Function(self.V_d, name="d_old")  # 上一载荷步，SNES 下界
        self._d_previous_iterate = fem.Function(self.V_d)  # 交错迭代内部用

        # f(alpha_bar)：Phase 1.1 静态算例恒为 1，Phase 1.2 由 cyclejump 传入
        self.f_alpha = f_alpha if f_alpha is not None else fem.Constant(
            mesh, default_scalar_type(1.0)
        )

        self.dx = ufl.Measure("dx", domain=mesh)
        # 有塑性时固定 dx 的求积阶（D-67）：返回映射含 sqrt/conditional，UFL 对残差 F
        # 与其 u 导数估计出的多项式阶不同，默认规则会让 F 与 dF/du 用不同求积——
        # 切线校验差 ~1e-4 且与差分步长无关（系统失配，非折点噪声）。固定为
        # PlasticLaw 的 quad_degree 后 F 与 dF/du 同规则，误差回到 ~1e-9。
        if plastic is not None:
            self.dx = ufl.Measure(
                "dx", domain=mesh,
                metadata={"quadrature_degree": plastic.quad_degree},
            )
        self.external_work = _resolve_external_work(external_work, self.V_u)
        # monolithic（D-68 方案 a）需要「给测试函数就给形式」的工厂。
        # 注意调用方给的是 ``lambda V: Form``（TestFunction 取自 V，见
        # clamp_loads.ClampLoads.traction 的注释），故这里不能直接拿 v 调用它；
        # 改成构造一个只含该 v 的单测函数形式：用 V_u 之外的临时空间不行
        # （会撞到「两个编号相同却不同源的 Argument」），所以对 v 所属空间
        # 直接调用即可 —— v 来自混合空间的 split，其函数空间正是 W.sub(0)。
        self.external_work_factory = (
            None if external_work is None
            else (external_work if callable(external_work)
                  else lambda v, _ew=external_work: _ew)
        )

        # u/d 问题在 setup() 中构建（若有接触需先绑定滑移空间）
        self._u_problem = None
        self._d_problem = None
        self._problems_built = False

    def setup(self) -> None:
        """构建 u/d 子问题（延迟到绑定接触滑移空间之后）。

        必须在 PhaseFieldSolver 创建后、首次求解前调用一次。
        若有接触且开启摩擦，调用方需先 bind 滑移/Uzawa 空间，再调用 setup()。
        """
        if self._problems_built:
            return
        self._build_u_problem()
        self._build_d_problem()
        self._build_energy_forms()
        self._problems_built = True

    def _ensure_setup(self) -> None:
        """内部用：若问题未构建则自动构建（兼容旧调用方式）。"""
        if not self._problems_built:
            self.setup()

    # ---------------- 子问题装配 ----------------
    def _build_u_problem(self) -> None:
        cfg, model = self.config, self.model
        v = ufl.TestFunction(self.V_u)
        if cfg.u_solver == "linear":
            u_trial = ufl.TrialFunction(self.V_u)
            a = self._bilinear_u_form(u_trial, v)
            if self.external_work is not None:
                L = self.external_work
            else:
                # 纯位移边界时没有外力项，给一个零线性形式以保持接口一致
                zero = fem.Constant(
                    self.mesh, np.zeros(self.dim, dtype=default_scalar_type)
                )
                L = ufl.inner(zero, v) * self.dx
            self._u_problem = LinearProblem(
                a,
                L,
                bcs=self.bcs_u,
                u=self.u,
                petsc_options_prefix="pfwire_u_",
                petsc_options=dict(self.config.direct_solver_options),
            )
        else:
            F = self._residual_u_form(v)
            # Jacobian：默认让 UFL 对 F 自动求导。**但切向项要手工补**
            # （方向 f，D-68 问题 33 续 13）：Uzawa 模式下切向牵引 q·τ·v 里
            # q 是 DG0 Function、不依赖 u，自动求导给出的切向刚度为 0 ——
            # q 饱和到 Coulomb 容量后 Newton 面对巨大面力却毫无切向刚度，
            # 必然 -5/-9。补上 ε_t·(τ·du)(τ·v)（Uzawa 迭代参数的本体）。
            u_trial = ufl.TrialFunction(self.V_u)
            J = ufl.derivative(F, self.u, u_trial)
            if (self.contact is not None
                    and getattr(self.contact, "has_friction_at_build", False)
                    and getattr(self.contact, "has_uzawa", lambda: False)()):
                J = J + self.contact.tangential_jacobian(self.u, u_trial, v)
            self._u_problem = NonlinearProblem(
                F,
                self.u,
                J=J,
                bcs=self.bcs_u,
                petsc_options_prefix="pfwire_u_newton_",
                petsc_options={
                    "snes_type": "newtonls",
                    "snes_linesearch_type": cfg.u_linesearch,
                    "snes_atol": cfg.snes_atol if cfg.snes_atol_u is None else cfg.snes_atol_u,
                    "snes_rtol": cfg.snes_rtol if cfg.snes_rtol_u is None else cfg.snes_rtol_u,
                    "snes_max_it": cfg.snes_max_it,
                    **self.config.direct_solver_options,
                },
            )

    def _bilinear_u_form(self, u_trial, v):
        """冻结 chi 后的双线性形式 a(u,v)。

        做法：用 Function ``self.u`` 写出（冻结 chi 的）二次型能量，对 u 求一次导得残差、
        再求一次导得 Jacobian。因为能量在冻结后严格二次，Jacobian 与 ``self.u`` 的取值无关，
        就是刚度矩阵本身；它只通过 g(d) 依赖 d、通过 chi 依赖 u_prev，二者每次迭代更新，
        而 ``LinearProblem.solve()`` 每次都重装配 A，故更新会被正确吃进去。
        """
        model = self.model
        energy = elastic_energy_density(
            strain(self.u),
            self.d,
            model.material,
            model.split,
            self.dim,
            model.k_res,
            freeze_from=strain(self.u_prev),
            degrade_stiffness=model.degrade_stiffness,
        ) * self.dx
        residual = ufl.derivative(energy, self.u, v)
        return ufl.derivative(residual, self.u, u_trial)

    def _residual_u_form(self, v):
        """精确（非线性）残差 F_u(u; v)。

        无塑性：F = dE/du（与标准残差等价，ε_e = ε）。
        有塑性：F = ∫ σ_deg : sym∇v，标准应力残差（D-65）。不取 dE/du：E 不含硬化能，
        dE/du 比标准残差多出 ∫σ:δε_p，且屈服载荷步 Newton 不收敛。
        """
        model = self.model
        if self.plastic is None:
            psi = elastic_energy_density(
                strain(self.u), self.d, model.material, model.split, self.dim,
                model.k_res, degrade_stiffness=model.degrade_stiffness,
            )
            F = ufl.derivative(psi * self.dx, self.u, v)
        else:
            F = ufl.inner(self._sigma_deg_2d(), ufl.sym(ufl.grad(v))) * self.dx
        if self.external_work is not None:
            F = F - self.external_work
        if self.contact is not None:
            F = F - self.contact.residual(self.u, v)
            # 摩擦切向牵引（6.6.1）。与法向项同号进入残差：接触力的虚功
            # 在 u 方程里取负。无摩擦时 tangential_residual 返回 0，
            # 加不加都对结果无影响（但会让 UFL 多带一个零项，故判一下）。
            if getattr(self.contact, "has_friction_at_build", False):
                F = F - self.contact.tangential_residual(self.u, v)
        return F

    def _build_d_problem(self) -> None:
        cfg, model = self.config, self.model
        w = ufl.TestFunction(self.V_d)
        # ⚠ 驱动力用 psi_plus_drive()（含可信区掩码），不是 psi_plus_expr()：
        # 区外 h >> ℓ 处 ψ⁺ 是离散误差，直接当驱动会把不该起裂的状态推成发散
        # （D-68 问题 12/13）。诊断仍看未屏蔽的 psi_plus_expr()。
        psi_plus = self.psi_plus_drive()
        # ⚠ 这里的 g(d) **不随 degrade_stiffness 变**（b-1 口径，D-68 问题 5）：
        # degrade_stiffness=False 只摘掉 u 方程弹性能里的 g(d)（= 材料软化），
        # d 方程的驱动力与屈服面仍保留它 —— 否则不可逆性与 B 口径会被一起改掉。
        # 于是 u 与 d 的 g(d) 用法**故意不对称**：一个管软化、一个管驱动力。
        energy = (
            degradation(self.d, model.k_res) * psi_plus
            + dissipation_density(
                self.d, model.material, self.f_alpha, model.dissipation
            )
        ) * self.dx
        F = ufl.derivative(energy, self.d, w)
        self._d_problem = NonlinearProblem(
            F,
            self.d,
            bcs=self.bcs_d,
            petsc_options_prefix="pfwire_d_",
            petsc_options={
                "snes_type": "vinewtonrsls",  # 变分不等式：界约束 [d_old, 1]
                "snes_atol": cfg.snes_atol,
                "snes_rtol": cfg.snes_rtol,
                "snes_max_it": cfg.snes_max_it,
                **self.config.direct_solver_options,
            },
        )
        # 变量界：下界随载荷步更新为 d_old，上界恒为 1
        self._lb = self._d_problem.x.duplicate()
        self._ub = self._d_problem.x.duplicate()
        self._ub.set(1.0)
        self._d_problem.solver.setVariableBounds(self._lb, self._ub)

    def _build_energy_forms(self) -> None:
        model = self.model
        psi_plus, psi_minus = self._energy_pair()
        # 与 u 方程保持同一口径：degrade_stiffness=False 时 u 最小化的是
        # ψ⁺ + ψ⁻（不含 g(d)），这里也必须照算，否则 energies() 报的不是
        # 求解器真正处置的那个能量（D-68 问题 5 的 b-1）。
        if model.degrade_stiffness:
            elastic_pair = degradation(self.d, model.k_res) * psi_plus + psi_minus
        else:
            elastic_pair = psi_plus + psi_minus
        self._form_elastic = fem.form(elastic_pair * self.dx)
        self._form_dissipated = fem.form(
            dissipation_density(
                self.d, model.material, self.f_alpha, model.dissipation
            )
            * self.dx
        )
        self._form_psi_plus = fem.form(psi_plus * self.dx)

    # ---------------- 求解 ----------------
    def _n_owned_d(self) -> int:
        return self.V_d.dofmap.index_map.size_local

    def _sync_lower_bound(self) -> None:
        self._ensure_setup()
        n = self._n_owned_d()
        self._lb.array[:n] = self.d_old.x.array[:n]

    def set_u_linesearch(self, ls_type: str) -> None:
        """运行时切换 u-SNES 的线搜索类型（不重建 Jacobian/问题）。

        用于 ramp→cyclic 边界的 ``bt``↔``basic`` 切换（见 SolverConfig.u_linesearch_cyclic）：
        SNES 已在首次求解时建好，这里只改其子对象 ``LineSearch`` 的类型，PETSc 下一次
        ``snes.solve`` 即生效。None 或空串视为不切换。
        """
        if not ls_type:
            return
        if self._problems_built:
            snes = self._u_problem.solver
            ls = snes.getLineSearch()
            ls.setType(str(ls_type))
            snes.setLineSearch(ls)

    def solve_displacement(self, fixed_point_iterations: int = 1) -> tuple[int, int]:
        """解 u（d 保持不动），返回 (实际迭代次数, 最后一次 chi 翻转数)。

        ``fixed_point_iterations > 1`` 时对冻结的 chi 做内层不动点迭代直到不再翻转；
        交错循环里用默认值 1 即可（外层已在迭代）。d 冻结、只推进 u 正是 D-4 提速
        路线中「周期内子步」的做法，故这里做成公开接口。
        newton 路径无 chi 可冻结，恒返回 (1, 0)。

        **Uzawa 循环**（Newton + 接触 + 摩擦）：
        当 ``u_solver='newton'`` 且 ``contact.has_uzawa()`` 时，在每次 Newton 求解后
        执行乘子更新 ``q ← Proj(q + eps_t·s)`` 直到收敛。这替代了罚函数的切向项，
        能突破 Q/(μP) ~ 0.1 的数值墙（D-68 问题 22/23）。
        """
        self._ensure_setup()
        if self.config.u_solver == "newton":
            if self.contact and self.contact.has_uzawa():
                if getattr(self.config, "monolithic_friction", False):
                    return self._solve_u_monolithic()
                return self._solve_u_uzawa()
            self._solve_u_once()
            return 1, 0
        sign_state = self._eval_trace_sign()
        flips = 0
        for it in range(1, max(1, fixed_point_iterations) + 1):
            self._solve_u_once()
            flips, sign_state = self._count_sign_flips(sign_state)
            if flips == 0:
                return it, 0
        return max(1, fixed_point_iterations), flips

    def _solve_u_monolithic(self, tol: float = 1.0e-8,
                            max_it: int = 50) -> tuple[int, int]:
        """**单体** Newton 解 (u, q) 混合系统（D-68 问题 33 续 18）。

        与 :meth:`_solve_u_uzawa`（交错格式）的区别：
          - 交错：外层固定点迭代 q ← Proj(q − ε_t·s)，u 子问题的 Jacobian
            不含 q 的反馈。**深度黏着下进二周期极限环**（D-68 问题 26 续 5），
            实测 2000 次迭代 dq 恒 0.268 不收敛。
          - 单体：把 (u, q) 当一个系统，用**联立 Jacobian** 一次求解。
            黏着区靠 ``dF_q/du = ε_t·ds/du``（实测 1.20e11，中心差分验证）
            把 u 钉住，系统非奇异。实测 13 步收敛。

        ⚠ 走 :meth:`pfwire.clamp_contact.ClampContact.solve_monolithic`
        （手工块 Newton），**不用** dolfinx 的多 form ``NonlinearProblem``
        （续 17 坑 7 的两个 bug 见该方法 docstring）。

        Returns ``(迭代数, 0)`` —— chi 翻转在 Newton 路径无意义。
        """
        assert self.contact is not None and self.contact.has_uzawa()

        def _elastic_external(v):
            """弹性 − 外载 − 法向接触（不含切向：切向在 F_q 与 q 面力里）。"""
            F = self._residual_u_form(v)
            # _residual_u_form 已含 elastic − external_work − contact.residual
            # − tangential_residual；单体路径要**换成**自己的切向处理，
            # 故把交错格式的切向项减掉。
            if getattr(self.contact, "has_friction_at_build", False):
                F = F + self.contact.tangential_residual(self.u, v)
            return F

        res = self.contact.solve_monolithic(
            self.u, _elastic_external, self.bcs_u,
            tol=tol, max_it=max_it,
            verbose=getattr(self.config, "log_iterations", False),
        )
        self.uzawa_iters = res.iterations
        self.uzawa_converged = res.converged
        if not res.converged and self.config.log_iterations:
            logger.warning(
                "单体 Newton 未收敛（reason=%s, it=%d, ||F||=%.3e）。"
                "见 DECISIONS.md D-68 问题 33 续 18。",
                res.reason, res.iterations, res.residual)
        return res.iterations, 0

    def _solve_u_once(self) -> None:
        self._ensure_setup()
        self.u_prev.x.array[:] = self.u.x.array
        self._u_problem.solve()
        reason = self._u_problem.solver.getConvergedReason()
        if reason <= 0:
            if self.config.u_solver == "linear":
                raise RuntimeError(
                    "u 的线性求解未收敛：KSPConvergedReason=%d（MUMPS 直接解失败，"
                    "通常意味着刚度矩阵奇异——检查边界条件是否足以约束刚体位移）" % reason
                )
            raise RuntimeError("u 的 Newton 求解未收敛：SNESConvergedReason=%d" % reason)

    def _solve_u_uzawa(self, max_it: int = 80, tol: float = 1.0e-3) -> tuple[int, int]:
        """Uzawa/ALM 循环解带 Coulomb 摩擦的接触位移问题。

        外层循环：
          1. 解 u（带当前 q 的 Neumann 切向牵引）
          2. 更新 q ← Proj(q − ε_t·s)，s = τ·(u - u_prev)
          3. 检查 max|Δq|/max(1, max|q|) < tol

        ``max_it``: Uzawa 最大迭代数（默认 80）
        ``tol``: 相对收敛容差（默认 1e-3）

        ⚠ **深度黏着工况下本迭代进二周期极限环、不会收敛**
        （D-68 问题 26 续 5：黏着区 q 由体平衡唯一确定，ε_t·s 迭代必过冲）。
        此时**不抛异常**（那会让整条 case 停在一个已知局限上），而是：
          - 取最后一步的 q 继续（切向力停在饱和值附近，量级仍合理）；
          - 把实际迭代次数写进 ``self.uzawa_iters`` 供调用方诊断。
        返回 (Uzawa 迭代次数, 0) —— chi 翻转在 Newton 路径无意义。
        """
        assert self.contact is not None and self.contact.has_uzawa()
        eps_t = self.contact.eps_t
        omega = self.config.u_relaxation

        self.uzawa_iters = 0
        for it in range(1, max_it + 1):
            if omega < 1.0:
                u_iter_prev = self.u.x.array.copy()
            self._solve_u_once()
            if omega < 1.0:
                # 欠松弛：u ← u_iter_prev + ω·(u_solved − u_iter_prev)。
                # s = τ·(u − u_prev_clamp) 完全由 u 决定（clamp.u_prev 跨本循环冻结），
                # 故 Δu 被压 ω 倍 ⇒ Δs 同比被压 ⇒ q = clip(q − θε_t·s) 的正反馈
                # 增益从 1 降到 ω。不动点不变（收敛时 u_solved = u_iter_prev）。
                # 见 DECISIONS.md D-68 问题 33 续 4。
                self.u.x.array[:] = u_iter_prev + omega * (self.u.x.array - u_iter_prev)
                self.u.x.scatter_forward()
            rel_change = self.contact.uzawa_update(self.u, eps_t=eps_t)
            self.uzawa_iters = it
            if rel_change <= tol:
                self.uzawa_converged = True
                return it, 0

        self.uzawa_converged = False
        if self.config.log_iterations:
            logger.warning(
                "Uzawa %d 次未收敛（rel=%.3e）：深度黏着工况的已知局限，"
                "取末步 q 继续。见 DECISIONS.md D-68 问题 26 续 5。",
                max_it, rel_change,
            )
        return max_it, 0

    def _snes_diagnostics(self) -> str:
        """d 求解失败时的可诊断信息：最终残差、容差、迭代数。

        只报 reason=-5 是查不出问题的——真正的信息是「残差停在什么量级、
        与 atol 差多少」。见 SolverConfig.snes_atol 的注释。
        """
        self._ensure_setup()
        snes = self._d_problem.solver
        rtol, atol, _, max_it = snes.getTolerances()
        try:
            fnorm = snes.getFunctionNorm()
        except Exception:  # 某些 reason 下取不到
            fnorm = float("nan")
        return (
            "最终残差 ‖F‖=%.3e，atol=%.1e、rtol=%.1e，迭代 %d/%d 次。"
            "若 ‖F‖ 停在 atol 附近不动，说明容差压到了装配的舍入地板上，"
            "应放宽 atol 而不是加迭代次数。"
            % (fnorm, atol, rtol, snes.getIterationNumber(), max_it)
        )

    def solve_damage(self) -> None:
        self._ensure_setup()
        self._sync_lower_bound()
        self._d_problem.solve()
        reason = self._d_problem.solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(
                "d 的 SNES(vinewtonrsls) 未收敛：SNESConvergedReason=%d（界约束 "
                "[d_old,1]）。%s" % (reason, self._snes_diagnostics())
            )

    def _count_sign_flips(self, tr_before: np.ndarray) -> tuple[int, np.ndarray]:
        """统计 tr eps 的符号在本次迭代中翻转的单元数（linear 路径的不动点诊断）。"""
        tr_after = self._eval_trace_sign()
        flips = int(np.count_nonzero(tr_before != tr_after))
        return self.comm.allreduce(flips, op=MPI.SUM), tr_after

    def _eval_trace_sign(self) -> np.ndarray:
        """把 tr eps(u) 的符号投到 DG0 上取值，用于诊断 chi 是否还在翻转。"""
        if not hasattr(self, "_V_dg0"):
            self._V_dg0 = fem.functionspace(self.mesh, ("DG", 0))
            self._sign_fn = fem.Function(self._V_dg0)
            self._sign_expr = fem.Expression(
                ufl.conditional(ufl.gt(ufl.tr(strain(self.u)), 0.0), 1.0, 0.0),
                self._V_dg0.element.interpolation_points,
            )
        self._sign_fn.interpolate(self._sign_expr)
        return self._sign_fn.x.array.copy()

    def solve_step(self) -> StaggeredResult:
        """交错迭代到 ‖Δd‖_inf < tol_d 或达到 max_iter。"""
        self._ensure_setup()
        cfg = self.config
        history: list[tuple[int, float, int]] = []
        delta = np.inf
        flips = 0
        sign_state = self._eval_trace_sign()
        failure: str | None = None
        converged = False
        iteration = 0

        for iteration in range(1, cfg.max_iter + 1):
            self._d_previous_iterate.x.array[:] = self.d.x.array
            try:
                self.solve_displacement()
                self.solve_damage()
                if cfg.relaxation < 1.0:
                    prev = self._d_previous_iterate.x.array
                    self.d.x.array[:] = prev + cfg.relaxation * (
                        self.d.x.array - prev
                    )
                    self.d.x.scatter_forward()
            except Exception as exc:  # 子求解器失败：立刻标记，不静默继续（捕获所有异常，含 PETSc Error）
                failure = str(exc)
                break

            local = np.max(
                np.abs(
                    self.d.x.array[: self._n_owned_d()]
                    - self._d_previous_iterate.x.array[: self._n_owned_d()]
                ),
                initial=0.0,
            )
            delta = self.comm.allreduce(local, op=MPI.MAX)
            # **判据要用未松弛的增量**：上面那个差已经被 omega 缩过一遍
            # （d <- d_prev + omega (d_solved - d_prev)），直接拿去和 tol_d 比，
            # 等于把容差放宽 1/omega 倍 —— omega 越小越容易「收敛」。
            # 实测（微动算例在 N=80012、深 13.5 µm 的状态上冻结 alpha 反复求解）：
            #   omega=0.25 -> 4.24e-4 然后逐位 0，是真不动点；
            #   omega=0.05 -> 8.49e-5 / 8.69e-5 / 8.82e-5，残差**不降反升**，
            #                 却因为压在 tol_d=1e-4 下面而被判「收敛」。
            # 这正是「静默不收敛」那一类最难查的失败：不报错，只是悄悄接受
            # 一个没到不动点的解。除以 omega 还原成真实步长，判据就与 omega 无关。
            delta /= cfg.relaxation
            flips, sign_state = self._count_sign_flips(sign_state)
            history.append((iteration, float(delta), flips))
            if cfg.log_iterations:
                logger.info(
                    "交错迭代 %d: ||Δd||_inf=%.3e, chi 翻转 %d", iteration, delta, flips
                )
            # chi 只在 linear 路径里被冻结；newton 路径直接解精确残差，
            # tr eps 的符号翻转对它毫无意义，把它写进收敛判据会无故阻止收敛。
            chi_ok = flips == 0 or cfg.u_solver != "linear"
            if delta < cfg.tol_d and chi_ok:
                converged = True
                break

        if failure is None and not converged:
            failure = (
                "交错迭代 %d 次未收敛：最后 ‖Δd‖_inf=%.3e（阈值 %.1e）"
                "%s。不要把这个解当结果用。"
                % (
                    iteration,
                    delta,
                    cfg.tol_d,
                    "，chi 翻转 %d 个单元" % flips
                    if cfg.u_solver == "linear"
                    else "（chi 翻转 %d 个单元，仅诊断，newton 路径不参与判据）" % flips,
                )
            )
        return StaggeredResult(
            converged=converged,
            iterations=iteration,
            delta_d_inf=float(delta),
            sign_flips=int(flips),
            max_d=self.max_damage(),
            energies=self.energies(),
            failure_mode=failure,
            history=history,
        )

    def commit_step(self) -> None:
        """把当前 d 冻结为不可逆下界（载荷步推进时调用）；有塑性时同步提交 (ε_p, p)。

        只能在交错迭代**收敛**之后调用：未收敛的步不得提交塑性历史（D-61）。
        """
        self.d_old.x.array[:] = self.d.x.array
        self.d_old.x.scatter_forward()
        if self.plastic is not None:
            self.plastic.commit(self.u, g=self._g())
        # 摩擦的切向滑移基准（6.6.1）：本步收敛后冻结，下一步的 s = τ·(u − u_prev)
        # 就是「本步新增的滑移」。放这里是因为 commit_step 的契约正是
        # 「交错迭代收敛后才调」，与 Coulomb 的增量语义天然对齐。
        if getattr(self.contact, "has_friction_at_build", False):
            self.contact.set_slip_reference(self.u)

    def _g(self):
        """损伤退化因子 g(d) = (1−d)² + k_res（UFL 表达式，随 d 变化）。"""
        return degradation(self.d, self.model.k_res)

    def _energy_pair(self):
        """(ψ⁺, ψ⁻) 的 UFL 表达式：无塑性时用 ε(u)（2D），有塑性时用三维 ε_e（D-65）。"""
        model = self.model
        if self.plastic is None:
            return energy_split(strain(self.u), model.material, model.split, self.dim)
        eps_e = self.plastic.elastic_strain(self.u, g=self._g())
        return energy_split_3d(eps_e, model.material, model.split)

    def psi_plus_expr(self):
        """未退化的弹性能正部 ψ⁺ 的 UFL 表达式。

        驱动力（d 方程）与 α̅ 累积（FatigueState、cycle-jump 的诊断）必须共用这一个表达式，
        否则 Δᾱ 与求解器的驱动力口径会不一致（D-65）。
        """
        return self._energy_pair()[0]

    def set_damage_region(self, mask) -> None:
        """换损伤可信区掩码并**重建 d 问题**（D-68 路线 A）。

        为什么需要这个入口：掩码在装配时烘进 d 方程的 form，直接改
        ``model.damage_region`` 会**静默无效**（form 不重编译）—— 这是
        很容易踩的坑。此方法同时改 model 并重建 ``_d_problem``。

        ``mask=None`` 即恢复全域。重建只涉及 form 装配与 SNES 对象，
        不重建网格/函数空间，代价很小（实测秒级）。
        """
        if mask is not None:
            # __post_init__ 的 DG0 校验不会在这条路径上重跑（frozen dataclass
            # 用 object.__setattr__ 绕过），故这里显式重做同一套检查。
            from dolfinx.fem import Function as _Fn
            if not isinstance(mask, _Fn):
                raise TypeError(
                    "damage_region 必须是 DG0 的 fem.Function，收到 %r。"
                    % (type(mask).__name__,))
            bx = mask.function_space.element.basix_element
            if int(bx.degree) != 0 or not bool(bx.discontinuous):
                raise TypeError(
                    "damage_region 必须是 DG0（degree=0 且不连续），收到 "
                    "degree=%d discontinuous=%s。"
                    % (int(bx.degree), bool(bx.discontinuous)))
            if mask.function_space.mesh is not self.mesh:
                raise ValueError("damage_region 的网格与求解器的网格不是同一个。")
        object.__setattr__(self.model, "damage_region", mask)
        self._build_d_problem()
        # _build_d_problem 把 _lb/_ub 重建成零/一，下界的「不可逆」部分要重新
        # 用 d_old 填（与每步 solve_damage 前的 _sync_lower_bound 同一动作）。
        self._sync_lower_bound()
        self._d_problem.solver.setVariableBounds(self._lb, self._ub)

    def psi_plus_drive(self):
        """d 方程**实际使用**的驱动力（D-68 路线 A）。

        = ``psi_plus_expr()`` × 可信区掩码（DG0 的 0/1）。掩码为 None 时两者相同。
        区外（h >> ℓ）的 ψ⁺ 是离散误差（问题 13：反推 σ_eq 451 MPa，超出 J2
        上界 7.3 倍），若喂给 d 方程会把「不该起裂的状态」推进成棘轮发散。

        ⚠ 只有**驱动力**被屏蔽。``psi_plus_expr()`` 本身保持未屏蔽，
        于是诊断（探针、ᾱ 累积）仍看得到原始值 —— 否则「区外到底有多大」
        就再也测不到了，也不符合 A-1「不冻结 ᾱ」的口径。
        """
        psi = self.psi_plus_expr()
        mask = self.model.damage_region
        if mask is None:
            return psi
        return mask * psi

    def _sigma_deg_2d(self):
        """塑性路径的退化应力 σ_deg 的面内 2×2 分量（D-65）。
        这里同时出现**两个** g(d)，各司其职，别混为一谈：

        1. ``elastic_strain(u, g=self._g())`` 的 ``g`` —— **屈服面**的退化
           （返回映射按退化应力判断屈服，D-62/D-63 的 B 口径）；
        2. ``stress_split_3d`` 的 ``degrade_stress`` —— **应力/刚度**的退化，
           即 u 方程里 σ = g(d)∂ψ⁺/∂ε + ∂ψ⁻/∂ε。

        b-1′（D-68 问题 6）把第 2 项摘掉、保留第 1 项，于是材料刚度不再随损伤软化，
        d↑→g↓→刚度↓→u↑→ψ⁺↑→d↑ 的正反馈环被切断（D-68 实测：max_d 五轮从 0.084
        冲到 1.0，而 ε_p ≡ 0 —— 反馈纯由 u 一侧造成）；屈服面退化不受影响。
        """
        model = self.model
        s3 = stress_split_3d(
            self.plastic.elastic_strain(self.u, g=self._g()),
            model.material, model.split, self.d, model.k_res,
            degrade_stress=model.degrade_stiffness,
        )
        return ufl.as_tensor([[s3[0, 0], s3[0, 1]], [s3[1, 0], s3[1, 1]]])

    def residual_u_form_at(self, u, v, d=None, V_u=None, with_external=True):

        """u 残差写成 ``(u, v)`` 的显式函数（monolithic 用，D-68 方案 a）。

        与 ``_residual_u_form`` 同式，但位移与测试函数都由调用方给出，

        于是可以把同一个式子写进混合空间的单一残差里。

        ``d`` 可选：monolithic 下退化因子应取自同一个混合解变量，

        否则 AD 得不到 ∂F_u/∂d 这一块（那正是交错丢掉的东西）。

        """

        model = self.model

        dd = self.d if d is None else d

        if self.plastic is None:

            psi = elastic_energy_density(

                strain(u), dd, model.material, model.split, self.dim,

                model.k_res, degrade_stiffness=model.degrade_stiffness,

            )

            F = ufl.derivative(psi * self.dx, u, v)

        else:

            s3 = stress_split_3d(

                self.plastic.elastic_strain(u, g=degradation(dd, model.k_res)),

                model.material, model.split, dd, model.k_res,

                degrade_stress=model.degrade_stiffness,

            )

            sigma = ufl.as_tensor([[s3[0, 0], s3[0, 1]], [s3[1, 0], s3[1, 1]]])

            F = ufl.inner(sigma, ufl.sym(ufl.grad(v))) * self.dx

        if self.external_work_factory is not None and with_external:

            # 工厂签名是 ``lambda V: Form``（TestFunction 取自 V），故传 v 的

            # 函数空间而不是 v 本身。

            # 工厂签名是 ``lambda V: Form``（TestFunction 取自 V）。
            # v 由 ufl.split 得到（ListTensor，没有 function_space），
            # 故由调用方显式给 V_u。
            F = F - self.external_work_factory(V_u)

        if self.contact is not None:

            F = F - self.contact.residual(u, v)

            # 摩擦切向牵引（6.6.1）：与法向项同号。无摩擦时跳过。

            if getattr(self.contact, "has_friction_at_build", False):

                F = F - self.contact.tangential_residual(u, v)

        return F

    def residual_d_form_at(self, u, d, w):
        """d 残差写成 ``(u, d, w)`` 的显式函数（monolithic 用，D-68 方案 a）。

        与 ``_build_d_problem`` 的能量同式：g(d)psi+ + f(alpha_bar)(3Gc/8)(d/ell
        + ell|grad d|^2)，对 d 求导。``d`` 由调用方给出以便 AD 得到交叉块。
        """
        model = self.model
        if self.plastic is None:
            psi_plus, _psi_minus = energy_split(
                strain(u), model.material, model.split, self.dim)
        else:
            eps_e = self.plastic.elastic_strain(u, g=degradation(d, model.k_res))
            psi_plus, _psi_minus = energy_split_3d(eps_e, model.material, model.split)
        energy = (
            degradation(d, model.k_res) * psi_plus
            + dissipation_density(d, model.material, self.f_alpha, model.dissipation)
        ) * self.dx
        return ufl.derivative(energy, d, w)
    # ---------------- 诊断量 ----------------
    def _assemble(self, form) -> float:
        return self.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)

    def energies(self) -> dict[str, float]:
        self._ensure_setup()
        elastic = self._assemble(self._form_elastic)
        dissipated = self._assemble(self._form_dissipated)
        return {
            "elastic": elastic,
            "dissipated": dissipated,
            "total": elastic + dissipated,
            "psi_plus": self._assemble(self._form_psi_plus),
        }

    def reaction(self, probe: fem.Function) -> float:
        """沿 ``probe`` 方向的支反力：R = dPi_el/du . probe = ∫ sigma : eps(probe) dV。

        ``probe`` 取「在受约束边界上等于单位方向、其余处为零」的场，
        于是该标量就是该边界上的总反力分量。耗散项不含 u，求导自动落掉，
        故这里对**精确**（非冻结）弹性能求导即可，两种分解都适用。
        比直接积分边界面力稳妥：不需要显式写出 sigma，避免对分解求导写错。
        """
        self._ensure_setup()
        model = self.model
        if self.plastic is None:
            psi = elastic_energy_density(
                strain(self.u), self.d, model.material, model.split, self.dim,
                model.k_res, degrade_stiffness=model.degrade_stiffness,
            )
            form = fem.form(ufl.derivative(psi * self.dx, self.u, probe))
        else:
            # 与残差同式：∫ σ_deg : sym∇probe（D-65）
            form = fem.form(ufl.inner(self._sigma_deg_2d(), ufl.sym(ufl.grad(probe))) * self.dx)
        return self.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)

    def make_probe(self, bcs_probe) -> fem.Function:
        """按给定的 DirichletBC（值为单位方向）构造 :meth:`reaction` 用的探针场。"""
        self._ensure_setup()
        probe = fem.Function(self.V_u, name="probe")
        for bc in _resolve_bcs(bcs_probe, self.V_u, "u"):
            bc.set(probe.x.array)
        probe.x.scatter_forward()
        return probe

    def max_damage(self) -> float:
        n = self._n_owned_d()
        local = float(np.max(self.d.x.array[:n], initial=0.0))
        return self.comm.allreduce(local, op=MPI.MAX)

    def min_damage(self) -> float:
        n = self._n_owned_d()
        local = float(np.min(self.d.x.array[:n], initial=1.0))
        return self.comm.allreduce(local, op=MPI.MIN)
