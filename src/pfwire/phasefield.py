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


def elastic_energy_density(eps, d, material: Material, split, dim, k_res, *, freeze_from=None):
    """g(d) psi+ + psi-。"""
    psi_plus, psi_minus = energy_split(
        eps, material, split, dim, freeze_from=freeze_from
    )
    return degradation(d, k_res) * psi_plus + psi_minus


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
    log_iterations: bool = False  # 逐迭代日志（排查静默不收敛）
    relaxation: float = 1.0
    """交错更新中 d 的松弛因子：d <- d_prev + omega (d_solved - d_prev)。
    默认 1.0（不松弛）。极限点附近交错格式可能**振荡**（‖Δd‖ 停在 1e-2 量级不降），
    此时取 omega ~ 0.5 可压住振荡；不动点不变，故不影响解的正确性。
    因 d_prev 与 d_solved 都 >= d_old，凸组合仍满足不可逆下界。"""

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
        if self.u_linesearch not in ("bt", "basic", "l2", "cp", "nleqerr"):
            raise ValueError(
                "u_linesearch 只能是 PETSc 的 bt/basic/l2/cp/nleqerr，收到 %r"
                % (self.u_linesearch,)
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
    ) -> None:
        self.mesh = mesh
        self.model = model
        self.config = config or SolverConfig()
        if self.config.u_solver == "linear" and model.split != "amor":
            raise ValueError(
                "u_solver='linear' 只支持 split='amor'（spectral 的冻结形式未实现）；"
                "请改用 u_solver='newton'。"
            )
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
        self.external_work = _resolve_external_work(external_work, self.V_u)

        self._build_u_problem()
        self._build_d_problem()
        self._build_energy_forms()

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
            self._u_problem = NonlinearProblem(
                F,
                self.u,
                bcs=self.bcs_u,
                petsc_options_prefix="pfwire_u_newton_",
                petsc_options={
                    "snes_type": "newtonls",
                    "snes_linesearch_type": cfg.u_linesearch,
                    "snes_atol": cfg.snes_atol,
                    "snes_rtol": cfg.snes_rtol,
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
        ) * self.dx
        residual = ufl.derivative(energy, self.u, v)
        return ufl.derivative(residual, self.u, u_trial)

    def _residual_u_form(self, v):
        """精确（非线性）残差 F_u(u; v)。"""
        model = self.model
        psi = elastic_energy_density(
            strain(self.u),
            self.d,
            model.material,
            model.split,
            self.dim,
            model.k_res,
        )
        F = ufl.derivative(psi * self.dx, self.u, v)
        if self.external_work is not None:
            F = F - self.external_work
        return F

    def _build_d_problem(self) -> None:
        cfg, model = self.config, self.model
        w = ufl.TestFunction(self.V_d)
        psi_plus, _ = energy_split(
            strain(self.u), model.material, model.split, self.dim
        )
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
        psi_plus, psi_minus = energy_split(
            strain(self.u), model.material, model.split, self.dim
        )
        self._form_elastic = fem.form(
            (degradation(self.d, model.k_res) * psi_plus + psi_minus) * self.dx
        )
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
        n = self._n_owned_d()
        self._lb.array[:n] = self.d_old.x.array[:n]

    def solve_displacement(self, fixed_point_iterations: int = 1) -> tuple[int, int]:
        """解 u（d 保持不动），返回 (实际迭代次数, 最后一次 chi 翻转数)。

        ``fixed_point_iterations > 1`` 时对冻结的 chi 做内层不动点迭代直到不再翻转；
        交错循环里用默认值 1 即可（外层已在迭代）。d 冻结、只推进 u 正是 D-4 提速
        路线中「周期内子步」的做法，故这里做成公开接口。
        newton 路径无 chi 可冻结，恒返回 (1, 0)。
        """
        if self.config.u_solver == "newton":
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

    def _solve_u_once(self) -> None:
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

    def _snes_diagnostics(self) -> str:
        """d 求解失败时的可诊断信息：最终残差、容差、迭代数。

        只报 reason=-5 是查不出问题的——真正的信息是「残差停在什么量级、
        与 atol 差多少」。见 SolverConfig.snes_atol 的注释。
        """
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
            except RuntimeError as exc:  # 子求解器失败：立刻标记，不静默继续
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
        """把当前 d 冻结为不可逆下界（载荷步推进时调用）。"""
        self.d_old.x.array[:] = self.d.x.array
        self.d_old.x.scatter_forward()

    # ---------------- 诊断量 ----------------
    def _assemble(self, form) -> float:
        return self.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)

    def energies(self) -> dict[str, float]:
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
        model = self.model
        psi = elastic_energy_density(
            strain(self.u),
            self.d,
            model.material,
            model.split,
            self.dim,
            model.k_res,
        )
        form = fem.form(ufl.derivative(psi * self.dx, self.u, probe))
        return self.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)

    def make_probe(self, bcs_probe) -> fem.Function:
        """按给定的 DirichletBC（值为单位方向）构造 :meth:`reaction` 用的探针场。"""
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
