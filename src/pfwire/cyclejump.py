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
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from .material import Material
from .phasefield import (
    PhaseFieldModel,
    PhaseFieldSolver,
    StaggeredResult,
    energy_split,
    fatigue_degradation,
    strain,
)

__all__ = [
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
        degree: int = 2,
    ) -> None:
        self.mesh = mesh
        self.model = model
        self.comm = mesh.comm
        self.degree = int(degree)
        if alpha_T is None:
            raise ValueError(
                "alpha_T 为 None：未标定的 alpha_T 不得进入计算（DECISIONS.md D-3）。"
                "请先做 Phase 2.3 标定，或在工况 yaml 里显式给值。"
            )
        if float(alpha_T) <= 0.0:
            raise ValueError("alpha_T 必须为正，收到 %r" % (alpha_T,))
        self.alpha_T = float(alpha_T)
        self.V = fem.functionspace(mesh, ("DG", self.degree))
        self.alpha_bar = fem.Function(self.V, name="alpha_bar")
        self.psi_plus_prev = fem.Function(self.V, name="psi_plus_prev")
        self._psi_plus_work = fem.Function(self.V, name="psi_plus")
        self._psi_expression: fem.Expression | None = None

    # ---------------- 绑定与采样 ----------------
    def bind_displacement(self, u: fem.Function) -> None:
        """绑定位移场，之后才能采样 psi+。"""
        psi_plus, _ = energy_split(
            strain(u), self.model.material, self.model.split, self.mesh.geometry.dim
        )
        self._psi_expression = fem.Expression(
            psi_plus, self.V.element.interpolation_points
        )

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
            min_f_alpha=fatigue_degradation(max_alpha, self.alpha_T),
        )

    def max_alpha_bar(self) -> float:
        n = self.n_owned
        return self.comm.allreduce(
            float(np.max(self.alpha_bar.x.array[:n], initial=0.0)), op=MPI.MAX
        )

    @property
    def f_alpha(self):
        """f(alpha_bar) 的 UFL 表达式，直接传给 PhaseFieldSolver。"""
        return fatigue_degradation(self.alpha_bar, self.alpha_T)

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
        self.N = 0.0
        self.delta_N = float(self.config.dN_init)
        self.reports: list[JumpReport] = []

        psi_plus, _ = energy_split(
            strain(solver.u),
            solver.model.material,
            solver.model.split,
            solver.dim,
        )
        self._psi_integral = fem.form(psi_plus * solver.dx)

    # ---------------- 内部工具 ----------------
    def _psi_plus_total(self) -> float:
        return self.solver.comm.allreduce(
            fem.assemble_scalar(self._psi_integral), op=MPI.SUM
        )

    def _snapshot(self) -> dict[str, np.ndarray]:
        return {
            "u": self.solver.u.x.array.copy(),
            "d": self.solver.d.x.array.copy(),
            "d_old": self.solver.d_old.x.array.copy(),
            "alpha_bar": self.state.alpha_bar.x.array.copy(),
            "psi_prev": self.state.psi_plus_prev.x.array.copy(),
        }

    def _restore(self, snap: dict[str, np.ndarray]) -> None:
        self.solver.u.x.array[:] = snap["u"]
        self.solver.d.x.array[:] = snap["d"]
        self.solver.d_old.x.array[:] = snap["d_old"]
        self.state.alpha_bar.x.array[:] = snap["alpha_bar"]
        self.state.psi_plus_prev.x.array[:] = snap["psi_prev"]
        for fn in (self.solver.u, self.solver.d, self.solver.d_old):
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
        self.state.prime()
        alpha_before = self.state.alpha_bar.x.array.copy()

        best_value, best_phase = -math.inf, 0.0
        for k in range(1, cfg.n_sub + 1):
            phase = k / cfg.n_sub
            self.set_phase(phase)
            self.solver.solve_displacement()
            self.state.accumulate()
            value = self._psi_plus_total()
            if value > best_value:
                best_value, best_phase = value, phase

        delta_alpha = self.state.alpha_bar.x.array - alpha_before
        phase = cfg.peak_phase if cfg.peak_phase is not None else best_phase
        return delta_alpha, phase

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
        snap = self._snapshot()
        d_before = snap["d"]

        rejections = 0
        delta_N_attempted = 0.0
        while True:
            # 不要跨过 N_max；但至少推进 dN_min
            delta_N = max(cfg.dN_min, min(self.delta_N, cfg.N_max - self.N))
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
                        logger.warning(
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
                logger.error(failure)
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
                )
                self.reports.append(report)
                return report
            self.delta_N = max(cfg.dN_min, self.delta_N / cfg.factor)
            logger.info("拒绝 ΔN=%g（%s），减半重试", delta_N, reason)

        # 接受
        self.N += delta_N
        self.solver.commit_step()
        if max_dd == max_dd and max_dd < cfg.grow_below * cfg.dd_max:  # 非 NaN 且够小
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
    alpha_needed = alpha_T * (2.0 / (s * math.sqrt(1.0 - d_target)) - 1.0)
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
    f = fatigue_degradation(alpha_bar, alpha_T)
    return max(0.0, 1.0 - f / s**2)
