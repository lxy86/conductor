"""疲劳历史变量 alpha_bar 的累积与 cycle-jump（PLAN.md Phase 1.2 / 1.3）。

Phase 1.2（本文件当前内容）
--------------------------
alpha_bar 的累积规则（PLAN.md 第 2 节，Carrara et al. 2020）：

    alpha_bar_{n+1} = alpha_bar_n + max(psi+_{n+1} - psi+_n, 0)

即只累加 psi+ 的**上升**部分。alpha_bar 与 psi+_prev 存 **DG-2**（不是 quadrature 元）：
quadrature 元非插值型，adios4dolfinx 无法可靠 checkpoint，与硬约束 5（断点续跑）冲突。
DG-2 自由度数与 degree-2 quadrature 同阶，可安全存取、可直接可视化。见 DECISIONS.md D-7。

Phase 1.3 将在此文件加 ΔN 自适应外推。

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

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from .material import Material
from .phasefield import PhaseFieldModel, energy_split, fatigue_degradation, strain

__all__ = [
    "FatigueState",
    "AccumulationReport",
    "homogeneous_cycles_to_damage",
    "homogeneous_damage_after_cycles",
]


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
# 均匀状态的解析寿命（PLAN 第 2 节公式的精确推论）
# --------------------------------------------------------------------------
def homogeneous_cycles_to_damage(
    stress_ratio_to_sigma_c: float,
    alpha_T: float,
    material: Material,
    d_target: float = 0.0,
) -> float:
    """均匀单轴状态下，达到 d = d_target 所需的循环数（闭式）。

    推导（全部来自 PLAN.md 第 2 节的式子，无新公式）：

    1. AT1 关系给 3 Gc/(8 ell) = sigma_c^2/E；
    2. 均匀单轴状态下 2 psi+ = sigma^2/E，记 s = sigma/sigma_c；
    3. d 的驻定条件 -2(1-d) psi+ + f(alpha_bar) 3Gc/(8 ell) = 0
       => 1 - d = f/s^2，故 d = d_target 对应 f = s^2 (1 - d_target)；
       d 从 0 起动即 f <= s^2；
    4. f = (2 a_T/(a+a_T))^2 反解得 alpha_bar = a_T [2/(s sqrt(1-d_target)) - 1]；
    5. 每个循环（谷值 0 -> 峰值 -> 谷值 0）的累积量 Δalpha = psi+_max = sigma^2/(2E)；
    6. 于是 N = 2 E a_T [2/(s sqrt(1-d_target)) - 1] / (s^2 sigma_c^2)。

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
    psi_plus_max = 0.5 * (s * sigma_c) ** 2 / E
    return alpha_needed / psi_plus_max


def homogeneous_damage_after_cycles(
    stress_ratio_to_sigma_c: float,
    alpha_T: float,
    material: Material,
    n_cycles: float,
) -> float:
    """:func:`homogeneous_cycles_to_damage` 的反函数：给定循环数返回 d。

    d = max(0, 1 - f(N psi+_max)/s^2)。
    """
    s = float(stress_ratio_to_sigma_c)
    if s <= 0.0:
        raise ValueError("s 必须为正")
    E = material.elastic.E
    sigma_c = material.fracture.sigma_c
    psi_plus_max = 0.5 * (s * sigma_c) ** 2 / E
    alpha_bar = float(n_cycles) * psi_plus_max
    f = fatigue_degradation(alpha_bar, alpha_T)
    return max(0.0, 1.0 - f / s**2)
