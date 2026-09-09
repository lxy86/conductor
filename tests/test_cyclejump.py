"""Phase 1.2 单测：f(alpha_bar)、alpha_bar 累积、疲劳寿命解析锚点、checkpoint。

本文件的解析锚点建立在「ᾱ 累积**退化前** psi+」的规则上（PLAN.md 第 2 节的写法），
故一律显式传 alpha_source="undegraded"。默认值已按 Carrara 2020 Eq. (13) 改为
degraded（= g(d) psi+_0），两者在 d = 0 时相同，其差异由
test_fatigue_forms.py::test_alpha_source_agrees_while_undamaged_and_differs_after 覆盖。

测试意图
--------
1. f(alpha_bar) 的性质：alpha_bar <= alpha_T 时恒为 1、在 alpha_T 处连续、
   严格单调下降、alpha_bar -> inf 时趋 0，并有手算锚点 f(3 a_T) = 1/4；
   未标定（None）或非正的 alpha_T 必须报错而不是取默认值；
2. 累积规则只吃 psi+ 的上升部分：升 -> 累加、降 -> 不变、再升 -> 再累加；
   一个完整循环（0 -> 峰 -> 0）恰好累加 psi+_max，N 个循环累加 N psi+_max；
3. **主物理锚点**：均匀单轴循环下，起裂循环数与闭式解
   N = 2 E a_T (2/s - 1)/(s^2 sigma_c^2) 一致，且起裂后各循环的 d 与
   d = 1 - f(N psi+_max)/s^2 一致——这条同时验证 f、累积规则和与 d 方程的耦合；
4. alpha_bar / psi+_prev 存 DG-2 且能 checkpoint 往返（硬约束 5、DECISIONS.md D-7）;
5. 未绑定位移场就累积必须报错，而不是静默累加 0。
"""

from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from pfwire import cyclejump as cj
from pfwire import phasefield as pf
from test_phasefield import SIGMA_C, UniaxialStrainCase, make_material


# --------------------------------------------------------------------------
# 1. f(alpha_bar)
# --------------------------------------------------------------------------
def test_fatigue_degradation_is_one_below_threshold_and_continuous():
    a_T = 1.0e6
    for a in (0.0, 0.1 * a_T, 0.999 * a_T, a_T):
        assert pf.fatigue_degradation(a, a_T) == pytest.approx(1.0)
    # a_T 右侧邻域仍连续
    assert pf.fatigue_degradation(1.0000001 * a_T, a_T) == pytest.approx(1.0, rel=1e-6)


def test_fatigue_degradation_numeric_anchors_and_monotonicity():
    a_T = 1.0e6
    # 手算：a = 3 a_T -> (2/(3+1))^2 = 1/4
    assert pf.fatigue_degradation(3.0 * a_T, a_T) == pytest.approx(0.25)
    # a = 9 a_T -> (2/10)^2 = 0.04
    assert pf.fatigue_degradation(9.0 * a_T, a_T) == pytest.approx(0.04)
    values = [pf.fatigue_degradation(x * a_T, a_T) for x in np.linspace(1.0, 50.0, 60)]
    assert all(b < a for a, b in zip(values, values[1:]))
    assert values[-1] < 0.01  # a -> inf 时趋 0


def test_fatigue_degradation_rejects_uncalibrated_or_bad_input():
    with pytest.raises(ValueError, match="alpha_T"):
        pf.fatigue_degradation(1.0, None)
    with pytest.raises(ValueError, match="alpha_T"):
        pf.fatigue_degradation(1.0, 0.0)
    with pytest.raises(ValueError, match="alpha_bar"):
        pf.fatigue_degradation(-1.0, 1.0e6)
    with pytest.raises(NotImplementedError):
        pf.fatigue_degradation(1.0, 1.0e6, form="parabolic")  # 无文献出处的形式


# --------------------------------------------------------------------------
# 2. alpha_bar 累积规则
# --------------------------------------------------------------------------
def test_accumulation_only_counts_rising_psi_plus():
    """升 -> 累加、降 -> 不变、再升 -> 再累加。"""
    a_T = 1.0e6
    case = UniaxialStrainCase()
    state = cj.FatigueState(
        case.msh, case.solver.model, alpha_source="undegraded", alpha_T=a_T)
    state.bind_displacement(case.solver.u)

    def hold(s: float):
        case.u_applied.value = s * case.strain_c * case.L
        case.solver.solve_displacement()
        return state.accumulate()

    r1 = hold(0.5)
    assert r1.max_increment > 0.0
    a1 = state.max_alpha_bar()

    r2 = hold(0.2)  # 卸载：psi+ 下降
    assert r2.max_increment == pytest.approx(0.0)
    assert state.max_alpha_bar() == pytest.approx(a1)

    r3 = hold(0.5)  # 再加载回原位：psi+ 从谷值升回峰值
    assert r3.max_increment > 0.0
    assert state.max_alpha_bar() > a1


def test_full_cycle_accumulates_exactly_psi_plus_max():
    """N 个 0->峰->0 循环后 alpha_bar = N psi+_max（均匀场，可解析核对）。"""
    a_T = 1.0e12  # 取大值使 f = 1、d 全程为 0，纯看累积
    s = 0.6
    case = UniaxialStrainCase()
    state = cj.FatigueState(
        case.msh, case.solver.model, alpha_source="undegraded", alpha_T=a_T)
    state.bind_displacement(case.solver.u)

    psi_max = 0.5 * (s * SIGMA_C) ** 2 / case.material.elastic.E
    n_cycles = 5
    for _ in range(n_cycles):
        case.u_applied.value = s * case.strain_c * case.L
        case.solver.solve_displacement()
        state.accumulate()
        case.u_applied.value = 0.0
        case.solver.solve_displacement()
        state.accumulate()
    assert state.max_alpha_bar() == pytest.approx(n_cycles * psi_max, rel=1e-9)
    assert case.solver.max_damage() == pytest.approx(0.0, abs=1e-12)


def test_accumulate_without_binding_raises():
    case = UniaxialStrainCase()
    state = cj.FatigueState(
        case.msh, case.solver.model, alpha_source="undegraded", alpha_T=1.0e6)
    with pytest.raises(RuntimeError, match="bind_displacement"):
        state.accumulate()


def test_fatigue_state_rejects_uncalibrated_alpha_T():
    case = UniaxialStrainCase()
    with pytest.raises(ValueError, match="alpha_T"):
        cj.FatigueState(
        case.msh, case.solver.model, alpha_source="undegraded", alpha_T=None)


def test_alpha_bar_lives_on_dg2():
    case = UniaxialStrainCase()
    state = cj.FatigueState(
        case.msh, case.solver.model, alpha_source="undegraded", alpha_T=1.0e6)
    # DG：自由度不跨单元共享 -> 全局自由度数 = 单元数 * 每单元自由度数（三角形 P2 为 6）
    n_cells = case.msh.topology.index_map(case.msh.topology.dim).size_global
    assert state.V.dofmap.index_map.size_global == n_cells * 6


# --------------------------------------------------------------------------
# 3. 主物理锚点：解析寿命
# --------------------------------------------------------------------------
def test_homogeneous_life_formula_self_consistency():
    """解析寿命与其反函数互为逆（纯公式层自检）。"""
    material = make_material(nu=0.0)
    a_T, s = 5.0e5, 0.7
    for d_target in (0.0, 0.3, 0.9):
        n = cj.homogeneous_cycles_to_damage(s, a_T, material, d_target=d_target)
        d_back = cj.homogeneous_damage_after_cycles(s, a_T, material, n)
        assert d_back == pytest.approx(d_target, abs=1e-12)
    # s >= 1：静载已超 sigma_c，不属疲劳区
    assert cj.homogeneous_cycles_to_damage(1.0, a_T, material) == 0.0
    # 精确比值：N ∝ (2/s - 1)/s^2，s 从 0.4 减半到 0.2 -> (9/0.04)/(4/0.16) = 9.0
    n1 = cj.homogeneous_cycles_to_damage(0.4, a_T, material)
    n2 = cj.homogeneous_cycles_to_damage(0.2, a_T, material)
    assert n2 / n1 == pytest.approx(9.0, rel=1e-12)
    # 只有 s << 1 才退化成 N ~ s^-3（比值 8）；这正是「本模型 S-N 斜率趋于 1/3」的来源
    n3 = cj.homogeneous_cycles_to_damage(0.02, a_T, material)
    n4 = cj.homogeneous_cycles_to_damage(0.01, a_T, material)
    assert n4 / n3 == pytest.approx(8.0, rel=0.01)


def test_cyclic_nucleation_matches_analytic_life():
    """**主锚点**：循环加载下起裂循环数与闭式解一致，起裂后 d(N) 也一致。

    取 s = 0.8、alpha_T 使解析起裂寿命 = 6.4 个循环，则第 1..6 个循环后 d 必须仍为 0，
    第 7 个循环后 d > 0；第 10 个循环后的 d 与 1 - f(10 psi+_max)/s^2 吻合。
    """
    s = 0.8
    material = make_material(nu=0.0)
    psi_max = 0.5 * (s * SIGMA_C) ** 2 / material.elastic.E
    n_target = 6.4
    # 由 N = a_T (2/s - 1)/psi_max 反解 a_T
    a_T = n_target * psi_max / (2.0 / s - 1.0)
    assert cj.homogeneous_cycles_to_damage(s, a_T, material) == pytest.approx(
        n_target, rel=1e-12
    )

    case = UniaxialStrainCase()
    state = cj.FatigueState(
        case.msh, case.solver.model, alpha_source="undegraded", alpha_T=a_T)
    # 把疲劳退化接进求解器：重建一个带 f(alpha_bar) 的求解器
    case.solver = pf.PhaseFieldSolver(
        case.msh,
        case.solver.model,
        bcs_u=case.make_bcs,
        f_alpha=state.f_alpha,
        config=case.solver.config,
    )
    state.bind_displacement(case.solver.u)

    damage_history = []
    for cycle in range(1, 11):
        case.u_applied.value = s * case.strain_c * case.L
        case.solver.solve_displacement()
        state.accumulate()
        case.solver.solve_damage()
        case.solver.commit_step()
        damage_history.append(case.solver.max_damage())
        case.u_applied.value = 0.0
        case.solver.solve_displacement()
        state.accumulate()

    for cycle in range(1, 7):  # N = 1..6 < 6.4
        assert damage_history[cycle - 1] == pytest.approx(0.0, abs=1e-12), (
            f"第 {cycle} 个循环就起裂了，早于解析寿命 {n_target}"
        )
    assert damage_history[6] > 0.0, "第 7 个循环仍未起裂，晚于解析寿命"

    for cycle in (7, 8, 9, 10):
        expected = cj.homogeneous_damage_after_cycles(s, a_T, material, cycle)
        assert damage_history[cycle - 1] == pytest.approx(expected, rel=5e-3), (
            f"第 {cycle} 个循环的 d 与闭式解不符"
        )
    assert state.max_alpha_bar() == pytest.approx(10 * psi_max, rel=1e-9)


# --------------------------------------------------------------------------
# 4. checkpoint 往返
# --------------------------------------------------------------------------
def test_fatigue_state_checkpoint_round_trip(tmp_path):
    """alpha_bar 与 psi+_prev 必须能存取往返，这是断点续跑的地基（硬约束 5）。

    注意路径必须从 rank 0 广播：pytest 的 tmp_path 是每进程各自生成的，
    并行下各 rank 会拿到不同目录，而 adios4dolfinx 的写入是按 COMM_WORLD 的
    集合操作 —— 路径不一致会直接死锁（本测试初版就是这么挂住的）。
    """
    case = UniaxialStrainCase()
    state = cj.FatigueState(
        case.msh, case.solver.model, alpha_source="undegraded", alpha_T=1.0e6)
    state.bind_displacement(case.solver.u)
    case.u_applied.value = 0.7 * case.strain_c * case.L
    case.solver.solve_displacement()
    state.accumulate()

    alpha_ref = state.alpha_bar.x.array.copy()
    psi_ref = state.psi_plus_prev.x.array.copy()
    assert np.max(alpha_ref) > 0.0

    path = MPI.COMM_WORLD.bcast(tmp_path / "fatigue.bp", root=0)
    state.save(path)

    state.alpha_bar.x.array[:] = 0.0
    state.psi_plus_prev.x.array[:] = 0.0
    state.load(path)

    assert np.allclose(state.alpha_bar.x.array, alpha_ref, rtol=1e-12)
    assert np.allclose(state.psi_plus_prev.x.array, psi_ref, rtol=1e-12)


def test_extrapolate_rejects_negative_dN():
    case = UniaxialStrainCase()
    state = cj.FatigueState(
        case.msh, case.solver.model, alpha_source="undegraded", alpha_T=1.0e6)
    with pytest.raises(ValueError, match="ΔN"):
        state.extrapolate(-1.0, np.zeros_like(state.alpha_bar.x.array))
