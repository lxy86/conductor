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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from mpi4py import MPI

from .material import Material, bulk_modulus, lame_parameters

__all__ = [
    "SPLITS",
    "FATIGUE_FORMS",
    "fatigue_degradation",
    "PhaseFieldModel",
    "SolverConfig",
    "StaggeredResult",
    "PhaseFieldSolver",
    "strain",
    "degradation",
    "energy_split",
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

#: 支持的疲劳退化函数形式。PLAN.md 第 2 节只给了 asymptotic 一种，
#: 其余形式不凭记忆补（硬约束 2），需要时再按文献添加。
FATIGUE_FORMS: tuple[str, ...] = ("asymptotic",)

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
    E, nu = material.elastic.E, material.elastic.nu
    lmbda, mu = lame_parameters(E, nu)
    tr_eps = ufl.tr(eps)

    if split == "amor":
        K = bulk_modulus(E, nu, dim)
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


def elastic_energy_density(eps, d, material: Material, split, dim, k_res, *, freeze_from=None):
    """g(d) psi+ + psi-。"""
    psi_plus, psi_minus = energy_split(
        eps, material, split, dim, freeze_from=freeze_from
    )
    return degradation(d, k_res) * psi_plus + psi_minus


def fatigue_degradation(alpha_bar, alpha_T: float, form: str = "asymptotic"):
    """Carrara et al. 2020 的疲劳退化函数 f(alpha_bar)。PLAN.md 第 2 节：

        f = 1                          (alpha_bar <= alpha_T)
        f = (2 alpha_T/(alpha_bar+alpha_T))^2   (alpha_bar >  alpha_T)

    实现上写成 ``f = min(1, (2 a_T/(a+a_T))^2)``，与分段定义**严格等价**：
    因 alpha_bar >= 0（按构造只累加非负增量），(2a_T/(a+a_T))^2 >= 1 当且仅当
    a <= a_T。这样避免 conditional，UFL 侧更平滑。在 a = a_T 处 f = 1 连续。

    ``alpha_bar`` 可以是 float（单测直接算）或 UFL 表达式/Function（进变分形式）。
    """
    if form not in FATIGUE_FORMS:
        raise NotImplementedError(
            "疲劳退化形式只实现了 %r（PLAN.md 第 2 节给的 asymptotic），收到 %r。"
            "要加别的形式必须先给出文献出处。" % (FATIGUE_FORMS, form)
        )
    if alpha_T is None:
        raise ValueError("alpha_T 为 None：未标定的 alpha_T 不得进入计算（DECISIONS.md D-3）")
    alpha_T = float(alpha_T)
    if alpha_T <= 0.0:
        raise ValueError("alpha_T 必须为正，收到 %r" % (alpha_T,))
    ratio_sq = (2.0 * alpha_T / (alpha_bar + alpha_T)) ** 2
    if isinstance(alpha_bar, (int, float)):
        if alpha_bar < 0.0:
            raise ValueError("alpha_bar 不能为负，收到 %r" % (alpha_bar,))
        return min(1.0, ratio_sq)
    return ufl.min_value(1.0, ratio_sq)


def dissipation_density(d, material: Material, f_alpha):
    """f(alpha_bar) (3 Gc/8) [d/ell + ell |grad d|^2]。PLAN.md 第 2 节。"""
    Gc = material.fracture.Gc
    ell = material.fracture.ell
    return f_alpha * (3.0 * Gc / 8.0) * (d / ell + ell * ufl.inner(ufl.grad(d), ufl.grad(d)))


# --------------------------------------------------------------------------
# 配置与结果
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PhaseFieldModel:
    """相场模型的离散无关部分。"""

    material: Material
    split: Split = "amor"
    k_res: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError("split 只能取 %r，收到 %r" % (SPLITS, self.split))
        if not 0.0 < self.k_res < 1.0:
            raise ValueError("k_res 应在 (0,1) 内，收到 %r" % (self.k_res,))


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
    snes_atol: float = 1.0e-10
    snes_rtol: float = 1.0e-9
    snes_max_it: int = 100
    log_iterations: bool = False  # 逐迭代日志（排查静默不收敛）

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
        self.external_work = external_work

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
            + dissipation_density(self.d, model.material, self.f_alpha)
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
            dissipation_density(self.d, model.material, self.f_alpha) * self.dx
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

    def solve_damage(self) -> None:
        self._sync_lower_bound()
        self._d_problem.solve()
        reason = self._d_problem.solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(
                "d 的 SNES(vinewtonrsls) 未收敛：SNESConvergedReason=%d，"
                "迭代 %d 次（界约束 [d_old,1]）"
                % (reason, self._d_problem.solver.getIterationNumber())
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
            flips, sign_state = self._count_sign_flips(sign_state)
            history.append((iteration, float(delta), flips))
            if cfg.log_iterations:
                logger.info(
                    "交错迭代 %d: ||Δd||_inf=%.3e, chi 翻转 %d", iteration, delta, flips
                )
            if delta < cfg.tol_d and flips == 0:
                converged = True
                break

        if failure is None and not converged:
            failure = (
                "交错迭代 %d 次未收敛：最后 ‖Δd‖_inf=%.3e（阈值 %.1e），"
                "chi 翻转 %d 个单元。不要把这个解当结果用。"
                % (iteration, delta, cfg.tol_d, flips)
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

    def max_damage(self) -> float:
        n = self._n_owned_d()
        local = float(np.max(self.d.x.array[:n], initial=0.0))
        return self.comm.allreduce(local, op=MPI.MAX)

    def min_damage(self) -> float:
        n = self._n_owned_d()
        local = float(np.min(self.d.x.array[:n], initial=1.0))
        return self.comm.allreduce(local, op=MPI.MIN)
