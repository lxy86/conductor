"""疲劳历史变量 alpha_bar 的累积与 cycle-jump（PLAN.md Phase 1.2 / 1.3）。

Phase 1.2（本文件当前内容）
--------------------------
alpha_bar 的累积规则（PLAN.md 第 2 节，Carrara et al. 2020）：

    alpha_bar_{n+1} = alpha_bar_n + max(psi+_{n+1} - psi+_n, 0)

即只累加 psi+ 的**上升**部分。alpha_bar 与 psi+_prev 存 **DG-2**（不是 quadrature 元）：
quadrature 元非插值型，adios4dolfinx 无法可靠 checkpoint，与硬约束 5（断点续跑）冲突。
DG-2 自由度数与 degree-2 quadrature 同阶，可安全存取、可直接可视化。见 DECISIONS.md D-7。

cycle-jump（Phase 1.3，PLAN.md 第 2 节）
--------------------------------------
- 显式循环：n_sub = 8 子步/周期，得逐点 Δᾱ_cyc；
- 外推 ΔN：ᾱ += (ΔN-1)·Δᾱ_cyc（显式那一圈本身已算 1 个循环），随后重解 (u,d)；
- ΔN 自适应：目标 max Δd <= 0.02，ΔN ∈ [1, 1e5]，乘/除 2 调节。

按 DECISIONS.md D-4，周期内的子步**只解 u、d 冻结**（单个周期内损伤演化不显著），
只在外推之后才做完整的 (u,d) 交错求解——这是把单工况从 ~20 h 压回可行区间的关键。
ΔN 被拒绝（Δd 超限或交错不收敛）时会回滚到外推前的状态并减半重试，
Δᾱ_cyc 与 ΔN 无关故不需重跑 8 个子步。ΔN 已到下界 1 仍不收敛则硬失败，不静默继续。

均匀状态的解析寿命
-----------------
下面 :func:`homogeneous_cycles_to_damage` 给出的不是新公式，而是 PLAN 第 2 节那几条式子
在**均匀单轴**状态下的精确推论，推导见该函数 docstring。它有三个用处：
单测的物理锚点、alpha_T 标定的初值、以及暴露一个重要事实——
本模型均匀极限下的 S-N 斜率固定为 sigma_a ~ N^(-1/3)，与铝的实测 b ≈ 0.13 不符
（见 DECISIONS.md D-14）。

单位：SI。alpha_bar 与 psi+ 的量纲都是 J/m^3。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dolfinx import default_scalar_type, fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from .material import Material
from .phasefield import (
    PhaseFieldModel,
    PhaseFieldSolver,
    StaggeredResult,
    degradation,
    energy_split,
    fatigue_degradation,
    inverse_fatigue_degradation,
    strain,
)

#: ᾱ 累积的是哪个 psi+。Carrara et al. 2020 的 Eq. (13) 定义
#: psi_el = g(d) psi+_el,0 + psi-_el,0 = psi+_el(eps,d) + psi-_el(eps)，
#: 而 Table 1 的累积式写的是 psi+_el —— 即**退化后**的那个（= g(d) psi+_0）。
#: PLAN.md 第 2 节只写 psi+ 未加限定。两者在 d = 0（起裂前）完全相同，
#: 故对 S-N 曲线无影响，只影响起裂之后的演化速度。默认按原文取 degraded。
#: 见 DECISIONS.md D-15。
ALPHA_SOURCES: tuple[str, ...] = ("degraded", "undegraded")

__all__ = [
    "ALPHA_SOURCES",
    "FatigueState",
    "AccumulationReport",
    "CycleJumpConfig",
    "JumpReport",
    "CycleJumpDriver",
    "homogeneous_cycles_to_damage",
    "homogeneous_damage_after_cycles",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccumulationReport:
    """一次 alpha_bar 累积的诊断量（全局归约后的值）。"""

    max_increment: float
    max_alpha_bar: float
    min_f_alpha: float


class FatigueState:
    """持有 alpha_bar 与 psi+_prev（均为 DG-2），并提供 f(alpha_bar) 表达式。

    用法（注意构造顺序：先建状态，再建求解器，最后绑定位移）::

        state = FatigueState(mesh, model, alpha_T=alpha_T)
        solver = PhaseFieldSolver(mesh, model, bcs_u=..., f_alpha=state.f_alpha)
        state.bind_displacement(solver.u)
        ...
        solver.solve_displacement()
        state.accumulate()        # alpha_bar += max(Δpsi+, 0)
        solver.solve_damage()     # 用更新后的 f(alpha_bar) 解 d
    """

    def __init__(
        self,
        mesh: dmesh.Mesh,
        model: PhaseFieldModel,
        *,
        alpha_T: float,
        form: str = "asymptotic",
        kappa: float | None = None,
        degree: int = 2,
        alpha_source: str = "degraded",
    ) -> None:
        self.mesh = mesh
        self.model = model
        self.comm = mesh.comm
        self.degree = int(degree)
        self.form = form
        self.kappa = kappa
        if alpha_source not in ALPHA_SOURCES:
            raise ValueError(
                "alpha_source 只能取 %r，收到 %r" % (ALPHA_SOURCES, alpha_source)
            )
        self.alpha_source = alpha_source
        if alpha_T is None:
            raise ValueError(
                "alpha_T 为 None：未标定的 alpha_T 不得进入计算（DECISIONS.md D-3）。"
                "请先做 Phase 2.3 标定，或在工况 yaml 里显式给值。"
            )
        if float(alpha_T) <= 0.0:
            raise ValueError("alpha_T 必须为正，收到 %r" % (alpha_T,))
        self.alpha_T = float(alpha_T)
        # 提前校验 form/kappa 组合，别等到装配时才炸
        fatigue_degradation(0.0, self.alpha_T, form, kappa)
        # alpha_T / kappa 以 Constant 进 UFL：标定要反复改它们，
        # 若以 float 进去每次都会触发 FFCx 重编译（DECISIONS.md D-16）。
        self._alpha_T_const = fem.Constant(mesh, default_scalar_type(self.alpha_T))
        self._kappa_const = (
            None if kappa is None
            else fem.Constant(mesh, default_scalar_type(float(kappa)))
        )
        self.V = fem.functionspace(mesh, ("DG", self.degree))
        self.alpha_bar = fem.Function(self.V, name="alpha_bar")
        self.psi_plus_prev = fem.Function(self.V, name="psi_plus_prev")
        self._psi_plus_work = fem.Function(self.V, name="psi_plus")
        self._psi_expression: fem.Expression | None = None

    # ---------------- 绑定与采样 ----------------
    def bind_displacement(
        self, u: fem.Function, d: fem.Function | None = None, psi_plus=None
    ) -> None:
        """绑定位移场（与 alpha_source="degraded" 时所需的 d），之后才能采样 psi+。

        psi_plus 为 ψ⁺ 的 UFL 表达式，缺省由 ε(u) 算出；有塑性时须传求解器的
        ``psi_plus_expr()``，与驱动力共用同一表达式（D-65）。
        """
        if psi_plus is None:
            psi_plus, _ = energy_split(
                strain(u), self.model.material, self.model.split, self.mesh.geometry.dim
            )
        if self.alpha_source == "degraded":
            if d is None:
                raise ValueError(
                    "alpha_source='degraded' 需要同时绑定 d（Carrara 2020 Eq. 13 的 "
                    "psi+_el = g(d) psi+_el,0）。请用 bind_solver(solver) 或传入 d。"
                )
            psi_plus = degradation(d, self.model.k_res) * psi_plus
        self._psi_expression = fem.Expression(
            psi_plus, self.V.element.interpolation_points
        )

    def bind_solver(self, solver: PhaseFieldSolver) -> None:
        """便捷绑定：从求解器取 u 与 d。"""
        self.bind_displacement(solver.u, solver.d, psi_plus=solver.psi_plus_expr())

    def _require_bound(self) -> fem.Expression:
        if self._psi_expression is None:
            raise RuntimeError(
                "FatigueState 尚未绑定位移场：请先调用 bind_displacement(solver.u)。"
            )
        return self._psi_expression

    def sample_psi_plus(self) -> fem.Function:
        """把当前 psi+ 插值到 DG-2 并返回（内部工作函数，勿长期持有）。"""
        self._psi_plus_work.interpolate(self._require_bound())
        return self._psi_plus_work

    # ---------------- 累积 ----------------
    def accumulate(self) -> AccumulationReport:
        """alpha_bar += max(psi+ - psi+_prev, 0)，随后把 psi+_prev 推进到当前值。"""
        psi_now = self.sample_psi_plus().x.array
        increment = np.maximum(psi_now - self.psi_plus_prev.x.array, 0.0)
        self.alpha_bar.x.array[:] += increment
        self.psi_plus_prev.x.array[:] = psi_now
        self.alpha_bar.x.scatter_forward()
        self.psi_plus_prev.x.scatter_forward()
        return self._report(increment)

    def prime(self) -> None:
        """把 psi+_prev 设为当前 psi+ 而**不**累积（用于初始化或跳步后重设基准）。"""
        self.psi_plus_prev.x.array[:] = self.sample_psi_plus().x.array
        self.psi_plus_prev.x.scatter_forward()

    def extrapolate(self, delta_N: float, increment_per_cycle: np.ndarray) -> None:
        """cycle-jump 外推：alpha_bar += ΔN * Δalpha_cyc（Phase 1.3 用）。"""
        if delta_N < 0.0:
            raise ValueError("ΔN 不能为负，收到 %r" % (delta_N,))
        self.alpha_bar.x.array[:] += delta_N * increment_per_cycle
        self.alpha_bar.x.scatter_forward()

    def cycle_increment(self) -> np.ndarray:
        """返回「本次累积量」的本地数组副本，供 cycle-jump 外推使用。"""
        return np.maximum(
            self.sample_psi_plus().x.array - self.psi_plus_prev.x.array, 0.0
        )

    # ---------------- 诊断 ----------------
    @property
    def n_owned(self) -> int:
        return self.V.dofmap.index_map.size_local

    def _report(self, increment: np.ndarray) -> AccumulationReport:
        n = self.n_owned
        max_inc = self.comm.allreduce(
            float(np.max(increment[:n], initial=0.0)), op=MPI.MAX
        )
        max_alpha = self.max_alpha_bar()
        return AccumulationReport(
            max_increment=max_inc,
            max_alpha_bar=max_alpha,
            min_f_alpha=fatigue_degradation(
                max_alpha, self.alpha_T, self.form, self.kappa
            ),
        )

    def max_alpha_bar(self) -> float:
        n = self.n_owned
        return self.comm.allreduce(
            float(np.max(self.alpha_bar.x.array[:n], initial=0.0)), op=MPI.MAX
        )

    @property
    def f_alpha(self):
        """f(alpha_bar) 的 UFL 表达式，直接传给 PhaseFieldSolver。

        用的是 Constant 而非 float，故 :meth:`set_fatigue_params` 改参数后
        **无需重建 form、也不会触发重编译**。
        """
        return fatigue_degradation(
            self.alpha_bar, self._alpha_T_const, self.form, self._kappa_const
        )

    def set_fatigue_params(
        self, alpha_T: float | None = None, kappa: float | None = None
    ) -> None:
        """就地改 alpha_T / kappa（标定用）。

        只写 Constant 的 ``.value``，变分形式原封不动 —— 这正是把它们做成
        Constant 的目的（D-16）。改完下一次求解即生效。
        """
        if alpha_T is not None:
            value = float(alpha_T)
            if value <= 0.0:
                raise ValueError("alpha_T 必须为正，收到 %r" % (alpha_T,))
            self.alpha_T = value
            self._alpha_T_const.value = value
        if kappa is not None:
            if self._kappa_const is None:
                raise ValueError(
                    "当前 form=%r 不使用 kappa，不能设置" % (self.form,)
                )
            value = float(kappa)
            if value <= 0.0:
                raise ValueError("kappa 必须为正，收到 %r" % (kappa,))
            self.kappa = value
            self._kappa_const.value = value

    # ---------------- checkpoint（硬约束 5） ----------------
    def save(self, path: str | Path) -> None:
        """把 alpha_bar 与 psi+_prev 写入 .bp（含网格），供断点续跑。"""
        import adios4dolfinx

        path = Path(path)
        adios4dolfinx.write_mesh(path, self.mesh)
        adios4dolfinx.write_function(path, self.alpha_bar)
        adios4dolfinx.write_function(path, self.psi_plus_prev)

    def load(self, path: str | Path) -> None:
        """从 .bp 读回 alpha_bar 与 psi+_prev（网格须与当前网格一致）。"""
        import adios4dolfinx

        path = Path(path)
        adios4dolfinx.read_function(path, self.alpha_bar)
        adios4dolfinx.read_function(path, self.psi_plus_prev)
        self.alpha_bar.x.scatter_forward()
        self.psi_plus_prev.x.scatter_forward()


# --------------------------------------------------------------------------
# cycle-jump 驱动器（PLAN.md Phase 1.3）
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CycleJumpConfig:
    """cycle-jump 设置。默认值取自 PLAN.md 第 5 节的 ``cycle`` 段。"""

    n_sub: int = 8              # 每周期的显式子步数
    dN_init: float = 100.0
    dN_min: float = 1.0         # ΔN = 1 即退化为纯显式循环（无外推）
    dN_max: float = 1.0e5
    dd_max: float = 0.02        # 单次 jump 允许的 max Δd
    grow_below: float = 0.5     # Δd < grow_below*dd_max 时把 ΔN 翻倍
    factor: float = 2.0         # 乘/除的倍数（PLAN：乘/除 2）
    N_max: float = 1.0e8        # 超过即 runout
    peak_phase: float | None = None
    """重解 (u,d) 所用的相位。None 表示自动取显式循环中 ∫psi+ dx 最大的子步——
    比硬编码 1/4 稳健：接触工况下驱动力最大的相位不一定在拉伸峰值。"""
    shakedown: "ShakedownConfig | None" = None
    """循环稳定性判据（PLAN §2.4）。带塑性的求解器必须给出，否则 CycleJumpDriver 报 ValueError。"""

    def __post_init__(self) -> None:
        if self.n_sub < 1:
            raise ValueError("n_sub 必须 >= 1，收到 %r" % (self.n_sub,))
        if not 0.0 < self.dN_min <= self.dN_init <= self.dN_max:
            raise ValueError(
                "ΔN 的取值须满足 0 < dN_min <= dN_init <= dN_max，"
                "收到 %r/%r/%r" % (self.dN_min, self.dN_init, self.dN_max)
            )
        if self.dd_max <= 0.0:
            raise ValueError("dd_max 必须为正")
        if self.factor <= 1.0:
            raise ValueError("factor 必须 > 1")
        if self.peak_phase is not None and not 0.0 <= self.peak_phase <= 1.0:
            raise ValueError("peak_phase 须在 [0,1] 内（以周期为单位）")


@dataclass(frozen=True)
class ShakedownConfig:
    """循环稳定性判据的阈值（PLAN §2.4，D-66）。三个值都必须由配置给出，代码里不设默认。

    zero_tol：净增量视为零的阈值（无量纲应变）；
    decay_ratio：衰减判据 e_k / e_{k−1} 的上限，须在 (0,1) 内；
    n_check：判定所需的连续周期增量个数（≥ 2）。
    """

    zero_tol: float
    decay_ratio: float
    n_check: int
    decay_jitter: float = 0.0
    """对 decay_ratio 的**相对容差**（默认 0，严格判据）。

    为什么需要：周期塑性应变增量 e_k 是 FEM 在离散网格上量出来的，
    会带数值抖动。若序列本意是逐周期精确减半（e_k ≈ 0.5·e_{k-1}），
    某个 e_k 可能读出 0.502 而不是 0.499 —— 严格判据
    `cur <= decay_ratio * prev` 会因这 0.4% 的超标把整条衰减序列误判成
    not_decaying（D-68 问题 17 实测：[..., 1.263e-08, 6.325e-09]，第三项
    超出 0.5 阈值 0.16%，被误判为棘轮停机）。

    这不是调参让跑通：jitter 只放宽「衰减」的**边界判定**，不放宽
    「不衰减」的定义 —— 真正的棘轮（e_k ≈ e_{k-1} 或更大）依然过不了。
    取 0.01（1%）即覆盖测量噪声，又远小于区分「衰减 vs 持平」所需的间隔。
    """

    def __post_init__(self) -> None:
        if not self.zero_tol >= 0.0:
            raise ValueError("zero_tol 必须 >= 0，收到 %r" % (self.zero_tol,))
        if not 0.0 < self.decay_ratio < 1.0:
            raise ValueError("decay_ratio 须在 (0,1) 内，收到 %r" % (self.decay_ratio,))
        if self.n_check < 2:
            raise ValueError("n_check 须 >= 2，收到 %r" % (self.n_check,))
        if not 0.0 <= self.decay_jitter < 0.5:
            raise ValueError("decay_jitter 须在 [0, 0.5) 内，收到 %r" % (self.decay_jitter,))


#: 允许外推的判定结果；其余（not_decaying、undetermined）一律不外推
SHAKEDOWN_ALLOWED = ("closed", "decaying")


def classify_plastic_history(increments, cfg: ShakedownConfig) -> str:
    """由逐周期的净塑性应变增量 e_k（取范数）判定循环稳定性（PLAN §2.4）。

    返回：
      ``undetermined``：周期数不足 n_check，继续显式推进；
      ``closed``：末两个增量均 ≤ zero_tol（弹性安定，或滞回环闭合）；
      ``decaying``：最近 n_check 个增量单调衰减，每步 e_k ≤ decay_ratio · e_{k−1}，且 e_{k−1} > zero_tol；
      ``not_decaying``：其余情形，包括慢衰减与重新出现增量——视为棘轮，禁止外推。
    """
    e = np.asarray(increments, dtype=float)
    if e.size < cfg.n_check:
        return "undetermined"
    if e[-1] <= cfg.zero_tol and e[-2] <= cfg.zero_tol:
        return "closed"
    recent = e[-cfg.n_check:]
    prev, cur = recent[:-1], recent[1:]
    # decay_jitter：允许 e_k 超出 decay_ratio·e_{k-1} 一个小的相对量
    # （D-68 问题 17：FEM 噪声会让本该 0.499 的项读出 0.502，
    #  严格判据把整条 halving 序列误判成 not_decaying）。
    ratio_limit = cfg.decay_ratio * (1.0 + cfg.decay_jitter)
    if np.all(prev > cfg.zero_tol) and np.all(cur <= ratio_limit * prev):
        return "decaying"
    return "not_decaying"


class RatchetDetected(RuntimeError):
    """棘轮：塑性应变净增量不衰减，外推被禁止（PLAN §2.4，硬约束 6）。"""

    def __init__(self, N: float, increments) -> None:
        self.N = float(N)
        self.increments = [float(x) for x in increments]
        super().__init__(
            "N=%g 处判定为棘轮：最近的周期塑性应变增量 %s 不衰减，停止外推；"
            "该工况的寿命结果不可用。"
            % (self.N, ["%.3e" % x for x in self.increments[-4:]])
        )


@dataclass
class JumpReport:
    """一次 cycle-jump 的结果。``accepted`` 为 False 时 ``failure_mode`` 必非空。

    ``delta_N`` 是最终被接受（失败时为最后尝试）的 ΔN；``delta_N_attempted`` 是本步
    **第一次**尝试的 ΔN。自适应增长是否生效要看后者——被拒绝的大 ΔN 不会出现在前者里。
    """

    N: float
    delta_N: float
    delta_N_attempted: float
    max_dd: float
    max_d: float
    max_alpha_bar: float
    delta_alpha_cyc_max: float
    peak_phase: float
    rejections: int
    accepted: bool
    runout: bool = False
    staggered: StaggeredResult | None = None
    failure_mode: str | None = None
    shakedown: str | None = None
    """塑性求解器的循环稳定性判定（见 classify_plastic_history）；弹性时为 None。"""


class CycleJumpDriver:
    """把「显式一周期 -> 外推 ΔN -> 重解 -> 自适应调 ΔN」串起来。

    ``set_phase(t_over_T)`` 由调用方提供：给定周期内相位（0..1）设置全部载荷
    （面力幅值、本体应力等）。驱动器不知道载荷长什么样，只负责推进循环数。

    用法::

        driver = CycleJumpDriver(solver, state, set_phase, config)
        while driver.N < config.N_max:
            report = driver.step()
            if not report.accepted:
                raise RuntimeError(report.failure_mode)
            if failure_criterion(solver.d):
                break
    """

    def __init__(
        self,
        solver: PhaseFieldSolver,
        state: FatigueState,
        set_phase: Callable[[float], None],
        config: CycleJumpConfig | None = None,
    ) -> None:
        self.solver = solver
        self.state = state
        self.set_phase = set_phase
        self.config = config or CycleJumpConfig()
        if getattr(solver, "plastic", None) is not None and self.config.shakedown is None:
            # 外推只作用于 alpha_bar，塑性历史 (ε_p, p) 不能按 ΔN 外推；必须先判定循环稳定（PLAN §2.4）
            raise ValueError("带塑性的 cycle-jump 必须给出 CycleJumpConfig.shakedown（PLAN §2.4）")
        self.N = 0.0
        self.delta_N = float(self.config.dN_init)
        self.reports: list[JumpReport] = []
        #: 逐周期的净塑性应变增量 e_k（范数），仅塑性求解器有
        self.plastic_increments: list[float] = []
        self._ep_cycle_end: np.ndarray | None = None

        self._psi_integral = fem.form(solver.psi_plus_expr() * solver.dx)

    # ---------------- 内部工具 ----------------
    def _log(self, level: str, msg: str, *args) -> None:
        """只在 rank 0 打日志。

        并行下不加守卫的话每条消息会打 comm.size 遍。这不只是刷屏：连续几条
        **一模一样**的「拒绝 ΔN=12.5」会让人以为自适应的减半根本没生效
        （实际是不同 rank 各打了一遍），排查时反倒被误导。
        """
        if self.solver.comm.rank == 0:
            getattr(logger, level)(msg, *args)

    def _psi_plus_total(self) -> float:
        return self.solver.comm.allreduce(
            fem.assemble_scalar(self._psi_integral), op=MPI.SUM
        )

    def _snapshot(self) -> dict[str, np.ndarray]:
        snap = {
            "u": self.solver.u.x.array.copy(),
            "d": self.solver.d.x.array.copy(),
            "d_old": self.solver.d_old.x.array.copy(),
            "alpha_bar": self.state.alpha_bar.x.array.copy(),
            "psi_prev": self.state.psi_plus_prev.x.array.copy(),
        }
        plastic = self.solver.plastic
        if plastic is not None:
            snap["ep"] = plastic.ep_n.x.array.copy()
            snap["p"] = plastic.p_n.x.array.copy()
        return snap

    def _restore(self, snap: dict[str, np.ndarray]) -> None:
        self.solver.u.x.array[:] = snap["u"]
        self.solver.d.x.array[:] = snap["d"]
        self.solver.d_old.x.array[:] = snap["d_old"]
        self.state.alpha_bar.x.array[:] = snap["alpha_bar"]
        self.state.psi_plus_prev.x.array[:] = snap["psi_prev"]
        fns = [self.solver.u, self.solver.d, self.solver.d_old]
        plastic = self.solver.plastic
        if plastic is not None:
            plastic.ep_n.x.array[:] = snap["ep"]
            plastic.p_n.x.array[:] = snap["p"]
            fns += [plastic.ep_n, plastic.p_n]
        for fn in fns:
            fn.x.scatter_forward()
        self.state.alpha_bar.x.scatter_forward()
        self.state.psi_plus_prev.x.scatter_forward()

    def _max_over_owned(self, array: np.ndarray, n_owned: int) -> float:
        return self.solver.comm.allreduce(
            float(np.max(array[:n_owned], initial=0.0)), op=MPI.MAX
        )

    # ---------------- 显式循环 ----------------
    def explicit_cycle(self) -> tuple[np.ndarray, float]:
        """跑一个完整周期的 n_sub 个子步（d 冻结），返回 (逐点 Δᾱ_cyc, 重解相位)。

        子步取相位 1/n_sub, 2/n_sub, ..., 1，起点在相位 0 处 prime，
        这样最后一个子步回到相位 1 == 0，周期是闭合的；否则会漏掉
        最后一段上升，Δᾱ_cyc 偏小。
        """
        cfg = self.config
        self.set_phase(0.0)
        self.solver.solve_displacement()
        self._commit_plastic_substep()
        self.state.prime()
        alpha_before = self.state.alpha_bar.x.array.copy()

        best_value, best_phase = -math.inf, 0.0
        for k in range(1, cfg.n_sub + 1):
            phase = k / cfg.n_sub
            self.set_phase(phase)
            self.solver.solve_displacement()
            self._commit_plastic_substep()
            self.state.accumulate()
            value = self._psi_plus_total()
            if value > best_value:
                best_value, best_phase = value, phase

        self._record_cycle_increment()
        delta_alpha = self.state.alpha_bar.x.array - alpha_before
        phase = cfg.peak_phase if cfg.peak_phase is not None else best_phase
        return delta_alpha, phase

    def _commit_plastic_substep(self) -> None:
        """塑性时每个子步都提交 (ε_p, p)。

        塑性历史是路径相关的：只在周期末提交会丢掉周期内的屈服（R = −1 时周期末卸到零应力，
        单步返回映射不改变 ε_p，净增量被误判为零）。故逐子步提交。d 在显式周期内不变，
        commit_step 对 d_old 的写入是空操作。
        """
        if self.solver.plastic is not None:
            self.solver.commit_step()

    def _record_cycle_increment(self) -> None:
        """记录本周期的净塑性应变增量 e_k = max|ε_p(周期末) − ε_p(上一周期末)|（全进程取最大）。"""
        plastic = self.solver.plastic
        if plastic is None:
            return
        ep = plastic.ep_n.x.array
        if self._ep_cycle_end is not None:
            local = float(np.max(np.abs(ep - self._ep_cycle_end), initial=0.0))
            self.plastic_increments.append(self.solver.comm.allreduce(local, op=MPI.MAX))
        self._ep_cycle_end = ep.copy()

    # ---------------- 一次 jump ----------------
    def step(self) -> JumpReport:
        cfg = self.config
        n_d = self.solver.V_d.dofmap.index_map.size_local

        if self.N >= cfg.N_max:
            report = JumpReport(
                N=self.N,
                delta_N=0.0,
                delta_N_attempted=0.0,
                max_dd=0.0,
                max_d=self.solver.max_damage(),
                max_alpha_bar=self.state.max_alpha_bar(),
                delta_alpha_cyc_max=0.0,
                peak_phase=0.0,
                rejections=0,
                accepted=True,
                runout=True,
            )
            self.reports.append(report)
            return report

        delta_alpha, peak_phase = self.explicit_cycle()
        delta_alpha_max = self._max_over_owned(delta_alpha, self.state.n_owned)
        # 塑性：先判定循环稳定性（PLAN §2.4）。棘轮直接停；未定或不允许时本步只做显式周期
        shake, allow_jump = None, True
        if self.solver.plastic is not None:
            shake = classify_plastic_history(self.plastic_increments, cfg.shakedown)
            if shake == "not_decaying":
                raise RatchetDetected(self.N, self.plastic_increments)
            allow_jump = shake in SHAKEDOWN_ALLOWED
        snap = self._snapshot()
        d_before = snap["d"]

        rejections = 0
        delta_N_attempted = 0.0
        while True:
            # 不要跨过 N_max；但至少推进 dN_min
            delta_N = max(cfg.dN_min, min(self.delta_N, cfg.N_max - self.N))
            if not allow_jump:
                delta_N = 1.0   # 本步只做一个显式周期，不外推（ΔN = 1 即无外推）
            if rejections == 0:
                delta_N_attempted = delta_N
            self.state.extrapolate(delta_N - 1.0, delta_alpha)
            self.set_phase(peak_phase)
            result = self.solver.solve_step()

            if result.converged:
                max_dd = self._max_over_owned(
                    np.abs(self.solver.d.x.array - d_before), n_d
                )
                if max_dd <= cfg.dd_max or delta_N <= cfg.dN_min:
                    if max_dd > cfg.dd_max:
                        self._log(
                            "warning",
                            "ΔN 已到下界 %g 但 max Δd = %.4f 仍超过 %.4f："
                            "裂纹扩展太快，考虑减小 dd_max 或加密网格",
                            cfg.dN_min,
                            max_dd,
                            cfg.dd_max,
                        )
                    break
                reason = "max Δd = %.4f > dd_max = %.4f" % (max_dd, cfg.dd_max)
            else:
                max_dd = math.nan
                reason = "交错求解未收敛：%s" % (result.failure_mode,)

            # 拒绝：回滚到外推前，ΔN 减半重试（Δᾱ_cyc 与 ΔN 无关，不必重跑子步）
            self._restore(snap)
            rejections += 1
            if delta_N <= cfg.dN_min:
                failure = (
                    "ΔN 已降到下界 %g 仍失败（%s）。这不是可以忽略的告警："
                    "结果不可用，请检查网格/载荷或降低 dd_max。" % (cfg.dN_min, reason)
                )
                self._log("error", "%s", failure)
                report = JumpReport(
                    N=self.N,
                    delta_N=delta_N,
                    delta_N_attempted=delta_N_attempted,
                    max_dd=max_dd,
                    max_d=self.solver.max_damage(),
                    max_alpha_bar=self.state.max_alpha_bar(),
                    delta_alpha_cyc_max=delta_alpha_max,
                    peak_phase=peak_phase,
                    rejections=rejections,
                    accepted=False,
                    staggered=result,
                    failure_mode=failure,
                    shakedown=shake,
                )
                self.reports.append(report)
                return report
            self.delta_N = max(cfg.dN_min, self.delta_N / cfg.factor)
            self._log("info", "拒绝 ΔN=%g（%s），减半重试", delta_N, reason)

        # 接受
        self.N += delta_N
        self.solver.commit_step()
        if allow_jump and max_dd == max_dd and max_dd < cfg.grow_below * cfg.dd_max:  # 非 NaN 且够小
            self.delta_N = min(self.delta_N * cfg.factor, cfg.dN_max)

        report = JumpReport(
            N=self.N,
            delta_N=delta_N,
            delta_N_attempted=delta_N_attempted,
            max_dd=max_dd,
            max_d=self.solver.max_damage(),
            max_alpha_bar=self.state.max_alpha_bar(),
            delta_alpha_cyc_max=delta_alpha_max,
            peak_phase=peak_phase,
            rejections=rejections,
            accepted=True,
            runout=self.N >= cfg.N_max,
            staggered=result,
            shakedown=shake,
        )
        self.reports.append(report)
        return report

    # ---------------- 批量推进 ----------------
    def run(
        self,
        stop: Callable[[JumpReport], bool] | None = None,
        max_steps: int | None = None,
    ) -> list[JumpReport]:
        """连续 jump 直到 runout、``stop`` 返回 True、失败或达到 max_steps。"""
        out: list[JumpReport] = []
        steps = 0
        while True:
            report = self.step()
            out.append(report)
            steps += 1
            if not report.accepted or report.runout:
                # **runout 的那一步也要交给 stop 记一笔**再退出。
                # 原先是先 break，于是最后一次推进不进 history，
                # N_final 取 history[-1]["N"] 就少报整整一个 jump
                # （实测：N_max=600 的算例真跑到了 600，却报 N=300）。
                # 这对 Phase 2.5 的 runout 工况是实打实的错报。
                # 失败的那一步同样记录：它是「死在哪里」的现场。
                if stop is not None:
                    stop(report)
                break
            if stop is not None and stop(report):
                break
            if max_steps is not None and steps >= max_steps:
                break
        return out


# --------------------------------------------------------------------------
# 均匀状态的解析寿命（PLAN 第 2 节公式的精确推论）
# --------------------------------------------------------------------------
def homogeneous_cycles_to_damage(
    stress_ratio_to_sigma_c: float,
    alpha_T: float,
    material: Material,
    d_target: float = 0.0,
    delta_alpha_per_cycle: float | None = None,
    form: str = "asymptotic",
    kappa: float | None = None,
) -> float:
    """均匀单轴状态下，达到 d = d_target 所需的循环数（闭式）。

    推导（全部来自 PLAN.md 第 2 节的式子，无新公式）：

    1. AT1 关系给 3 Gc/(8 ell) = sigma_c^2/E；
    2. 均匀单轴状态下 2 psi+ = sigma^2/E，记 s = sigma/sigma_c；
    3. d 的驻定条件 -2(1-d) psi+ + f(alpha_bar) 3Gc/(8 ell) = 0
       => 1 - d = f/s^2，故 d = d_target 对应 f = s^2 (1 - d_target)；
       d 从 0 起动即 f <= s^2；
    4. f = (2 a_T/(a+a_T))^2 反解得 alpha_bar = a_T [2/(s sqrt(1-d_target)) - 1]；
    5. 每周期累积量 Δalpha。``delta_alpha_per_cycle`` 为 None 时取默认的
       psi+_max = sigma^2/(2E)，对应 0 -> 峰 -> 0（R = 0）的加载；
    6. 于是 N = a_T [2/(s sqrt(1-d_target)) - 1] / Δalpha。

    **注意加载波形**：正弦 R = -1 载荷下每周期累积量**不是** psi+_max。
    Amor 分解中压缩半周期的体积项被划入 psi-，但**偏量项仍留在 psi+**，
    于是压缩峰值处 psi+ = psi+_max/2（nu = 0 的单轴情形），一个整周期累积
    psi+_max + psi+_max/2 = **1.5 psi+_max**。即 R = -1 的寿命只有 R = 0 的 2/3。
    这一点由 test_cyclejump.py 的 explicit_cycle 测试定量钉住，
    也是 DECISIONS.md D-5（R = -1、零平均应力）的一个实际后果。

    s >= 1 表示静载已超过 sigma_c，返回 0（静态起裂，不属疲劳区）。

    **重要推论**：s << 1 时 N ~ 2 a_T (2E/sigma_c^2) / s^3，即 sigma_a ~ N^(-1/3)。
    本模型均匀极限的 S-N 斜率固定为 1/3，与铝单丝实测的 b ≈ 0.13 差很多，
    且 sigma_c（或 Gc）只平移曲线、不改斜率。见 DECISIONS.md D-14。
    """
    s = float(stress_ratio_to_sigma_c)
    if s <= 0.0:
        raise ValueError("s = sigma/sigma_c 必须为正，收到 %r" % (s,))
    if not 0.0 <= d_target < 1.0:
        raise ValueError("d_target 必须在 [0,1) 内，收到 %r" % (d_target,))
    if s >= 1.0:
        return 0.0
    alpha_T = float(alpha_T)
    E = material.elastic.E
    sigma_c = material.fracture.sigma_c
    # f = s^2 (1 - d_target) 时达到 d_target，反解所需的 alpha_bar
    alpha_needed = inverse_fatigue_degradation(
        s**2 * (1.0 - d_target), alpha_T, form, kappa
    )
    increment = (
        0.5 * (s * sigma_c) ** 2 / E
        if delta_alpha_per_cycle is None
        else float(delta_alpha_per_cycle)
    )
    if increment <= 0.0:
        raise ValueError("每周期累积量必须为正，收到 %r" % (increment,))
    return alpha_needed / increment


def homogeneous_damage_after_cycles(
    stress_ratio_to_sigma_c: float,
    alpha_T: float,
    material: Material,
    n_cycles: float,
    delta_alpha_per_cycle: float | None = None,
    form: str = "asymptotic",
    kappa: float | None = None,
) -> float:
    """:func:`homogeneous_cycles_to_damage` 的反函数：给定循环数返回 d。

    d = max(0, 1 - f(N Δalpha)/s^2)。``delta_alpha_per_cycle`` 的含义同上。
    """
    s = float(stress_ratio_to_sigma_c)
    if s <= 0.0:
        raise ValueError("s 必须为正")
    E = material.elastic.E
    sigma_c = material.fracture.sigma_c
    increment = (
        0.5 * (s * sigma_c) ** 2 / E
        if delta_alpha_per_cycle is None
        else float(delta_alpha_per_cycle)
    )
    alpha_bar = float(n_cycles) * increment
    f = fatigue_degradation(alpha_bar, alpha_T, form, kappa)
    return max(0.0, 1.0 - f / s**2)
