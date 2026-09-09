"""Phase 1.3 单测：cycle-jump（显式循环 + ΔN 外推 + ΔN 自适应）。

测试意图
--------
1. **显式循环的解析锚点**：正弦 R = -1 载荷、nu = 0 的均匀单轴算例下，
   一个周期的逐点累积量有闭式值 Δᾱ_cyc = 0.75 E e_a^2。
   推导：amor 分解中拉伸峰 psi+ = 0.5 E e_a^2、压缩峰 psi+ = 0.25 E e_a^2
   （体积项归 psi-、偏量项仍在 psi+），一周期两次上升合计 0.75 E e_a^2。
   n_sub = 8 与 4 都恰好取到两个峰值故**精确**；n_sub = 3 取不到，只得一半 ——
   这就是 PLAN 取 n_sub = 8 的理由，也定量说明了 R = -1 比 R = 0 多累积 50%。
2. **端到端锚点**：本算例的 Δᾱ_cyc 与 d 无关，故外推是精确的，
   cycle-jump 给出的 d(N) 必须与闭式解 d = 1 - f(N Δᾱ_cyc)/s^2 逐点吻合。
3. ΔN 自适应：Δd 超限则回滚 + 减半重试，被接受的步必满足 max Δd <= dd_max；
   Δd 很小则翻倍，且不超过 dN_max、不低于 dN_min。
4. 不越过 N_max，到达即报 runout。
5. 交错求解一直不收敛时，ΔN 降到下界后必须**硬失败**（accepted=False + failure_mode），
   不得静默返回坏解。
6. 回滚正确性：被拒绝后 ᾱ/d/u 必须恢复原状。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pfwire import cyclejump as cj
from pfwire import phasefield as pf
from test_phasefield import SIGMA_C, UniaxialStrainCase, make_material


def make_sinusoidal_case(s: float, alpha_T: float, n_sub: int = 8, **cfg_kwargs):
    """均匀单轴算例 + 正弦 R = -1 载荷 + 疲劳状态 + cycle-jump 驱动器。"""
    case = UniaxialStrainCase()
    state = cj.FatigueState(case.msh, case.solver.model, alpha_T=alpha_T)
    case.solver = pf.PhaseFieldSolver(
        case.msh,
        case.solver.model,
        bcs_u=case.make_bcs,
        f_alpha=state.f_alpha,
        config=case.solver.config,
    )
    state.bind_displacement(case.solver.u)

    amplitude = s * case.strain_c * case.L

    def set_phase(t: float) -> None:
        case.u_applied.value = amplitude * math.sin(2.0 * math.pi * t)

    config = cj.CycleJumpConfig(n_sub=n_sub, **cfg_kwargs)
    driver = cj.CycleJumpDriver(case.solver, state, set_phase, config)
    return case, state, driver


def analytic_cycle_increment(s: float, material) -> float:
    """0.75 E e_a^2，e_a = s sigma_c/E  ->  = 1.5 * psi+_peak。"""
    e_a = s * SIGMA_C / material.elastic.E
    return 0.75 * material.elastic.E * e_a**2


# --------------------------------------------------------------------------
# 1. 显式循环的解析锚点
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_sub,expected_factor", [(8, 1.0), (4, 1.0), (3, 0.5)])
def test_explicit_cycle_increment_matches_closed_form(
    n_sub: int, expected_factor: float
):
    s = 0.6
    material = make_material(nu=0.0)
    case, state, driver = make_sinusoidal_case(s, alpha_T=1.0e12, n_sub=n_sub)
    delta_alpha, peak_phase = driver.explicit_cycle()

    expected = expected_factor * analytic_cycle_increment(s, material)
    got = float(np.max(delta_alpha[: state.n_owned]))
    assert got == pytest.approx(expected, rel=1e-9), (
        f"n_sub={n_sub}: Δᾱ_cyc = {got:.6e}，期望 {expected:.6e}"
    )
    # 重解相位应自动落在拉伸峰（psi+ 最大处），n_sub=3 时最接近的是 1/3
    assert 0.2 <= peak_phase <= 0.35 or n_sub == 3


def test_compression_half_cycle_contributes_half_of_tension():
    """R = -1 的每周期累积是 1.5 倍 psi+_peak：压缩半周期的偏量项也算损伤。

    这不是实现细节，而是 amor 分解 + 零平均应力（D-5）的直接后果，
    意味着 R = -1 的寿命只有 R = 0 的 2/3。
    """
    s = 0.6
    material = make_material(nu=0.0)
    psi_peak = 0.5 * (s * SIGMA_C) ** 2 / material.elastic.E
    assert analytic_cycle_increment(s, material) == pytest.approx(
        1.5 * psi_peak, rel=1e-12
    )
    _, state, driver = make_sinusoidal_case(s, alpha_T=1.0e12, n_sub=8)
    delta_alpha, _ = driver.explicit_cycle()
    got = float(np.max(delta_alpha[: state.n_owned]))
    assert got == pytest.approx(1.5 * psi_peak, rel=1e-9)


# --------------------------------------------------------------------------
# 2. 端到端：d(N) 对拍闭式解
# --------------------------------------------------------------------------
def test_cycle_jump_damage_history_matches_closed_form():
    """**主锚点**：本算例外推精确，故 d(N) 必须与闭式解逐点吻合。"""
    s = 0.6
    material = make_material(nu=0.0)
    increment = analytic_cycle_increment(s, material)
    # 取 alpha_T 使起裂约在 200 个循环
    alpha_T = 200.0 * increment / (2.0 / s - 1.0)
    n_nucleation = cj.homogeneous_cycles_to_damage(
        s, alpha_T, material, delta_alpha_per_cycle=increment
    )
    assert n_nucleation == pytest.approx(200.0, rel=1e-12)

    case, state, driver = make_sinusoidal_case(
        s, alpha_T=alpha_T, n_sub=8, dN_init=50.0, dN_max=400.0, dd_max=0.02
    )
    reports = driver.run(stop=lambda r: r.max_d > 0.6, max_steps=80)
    assert reports[-1].accepted, reports[-1].failure_mode

    for r in reports:
        expected = cj.homogeneous_damage_after_cycles(
            s, alpha_T, material, r.N, delta_alpha_per_cycle=increment
        )
        if expected == 0.0:
            assert r.max_d == pytest.approx(0.0, abs=1e-12)
        else:
            assert r.max_d == pytest.approx(expected, rel=2e-3), (
                f"N = {r.N:g}: d = {r.max_d:.6f}，闭式解 {expected:.6f}"
            )
    # ᾱ 也必须精确等于 N * Δᾱ_cyc（说明多次拒绝-回滚没造成重复累积）
    assert state.max_alpha_bar() == pytest.approx(
        reports[-1].N * increment, rel=1e-9
    )
    assert reports[-1].max_d > 0.6, f"80 步只跑到 d = {reports[-1].max_d:.3f}"
    # 起裂前后都要有取样点，否则对拍没覆盖到关键段
    assert any(r.max_d == 0.0 for r in reports)
    assert any(r.max_d > 0.3 for r in reports)


# --------------------------------------------------------------------------
# 3. ΔN 自适应
# --------------------------------------------------------------------------
def test_dN_grows_and_caps_when_damage_is_inactive():
    """无损伤演化时 Δd = 0，ΔN 应当持续翻倍并停在 dN_max。"""
    case, state, driver = make_sinusoidal_case(
        0.3, alpha_T=1.0e12, n_sub=4, dN_init=100.0, dN_max=1600.0, N_max=1.0e9
    )
    reports = driver.run(max_steps=8)
    used = [r.delta_N for r in reports]
    assert used[0] == pytest.approx(100.0)
    assert used[1] == pytest.approx(200.0)  # Δd = 0 -> 翻倍
    assert max(used) == pytest.approx(1600.0), f"ΔN 没长到 dN_max：{used}"
    assert all(u <= 1600.0 + 1e-9 for u in used), f"ΔN 越过了 dN_max：{used}"
    assert all(r.rejections == 0 for r in reports)
    assert driver.solver.max_damage() == pytest.approx(0.0, abs=1e-12)


def test_dN_shrinks_when_dd_exceeds_limit():
    """损伤活跃后 Δd 变大，必须触发拒绝-减半，且被接受的步都满足 dd_max。"""
    s = 0.6
    material = make_material(nu=0.0)
    increment = analytic_cycle_increment(s, material)
    alpha_T = 200.0 * increment / (2.0 / s - 1.0)
    case, state, driver = make_sinusoidal_case(
        s, alpha_T=alpha_T, n_sub=8, dN_init=100.0, dN_max=1600.0, dd_max=0.02
    )
    reports = driver.run(stop=lambda r: r.max_d > 0.5, max_steps=80)
    assert reports[-1].accepted, reports[-1].failure_mode

    for r in reports:
        if r.runout:
            continue
        # 被接受的步：要么满足 dd_max，要么 ΔN 已在下界（此时代码会告警）
        assert r.max_dd <= driver.config.dd_max + 1e-12 or r.delta_N <= 1.0
        assert driver.config.dN_min <= r.delta_N <= driver.config.dN_max
    # 自适应增长要通过「尝试值」看：被拒绝的大 ΔN 不会进 delta_N
    assert max(r.delta_N_attempted for r in reports) > 100.0
    assert any(r.rejections > 0 for r in reports), "从未触发 ΔN 减半，自适应没被检验到"
    # 减半后的 ΔN 必须小于尝试值
    shrunk = [r for r in reports if r.rejections > 0]
    assert all(r.delta_N < r.delta_N_attempted for r in shrunk)


def test_config_rejects_inconsistent_dN_bounds():
    with pytest.raises(ValueError, match="dN_min"):
        cj.CycleJumpConfig(dN_init=1.0e6, dN_max=1.0e5)
    with pytest.raises(ValueError, match="n_sub"):
        cj.CycleJumpConfig(n_sub=0)
    with pytest.raises(ValueError, match="dd_max"):
        cj.CycleJumpConfig(dd_max=0.0)
    with pytest.raises(ValueError, match="peak_phase"):
        cj.CycleJumpConfig(peak_phase=1.5)


# --------------------------------------------------------------------------
# 4. runout
# --------------------------------------------------------------------------
def test_driver_stops_at_N_max_without_overshoot():
    s = 0.3  # 低应力，不会起裂
    case, state, driver = make_sinusoidal_case(
        s, alpha_T=1.0e12, n_sub=4, dN_init=100.0, dN_max=100.0, N_max=250.0
    )
    reports = driver.run(max_steps=10)
    assert reports[-1].runout
    assert driver.N == pytest.approx(250.0), f"N = {driver.N}，不应越过 N_max"
    assert all(r.accepted for r in reports)
    assert driver.solver.max_damage() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# 5-6. 硬失败与回滚
# --------------------------------------------------------------------------
def test_persistent_nonconvergence_fails_hard_not_silently():
    """交错求解永远不收敛时，ΔN 降到下界后必须显式失败。"""
    s = 0.6
    material = make_material(nu=0.0)
    increment = analytic_cycle_increment(s, material)
    alpha_T = 10.0 * increment / (2.0 / s - 1.0)
    case = UniaxialStrainCase(max_iter=1)  # 交错只允许 1 次迭代 -> 必不收敛
    state = cj.FatigueState(case.msh, case.solver.model, alpha_T=alpha_T)
    case.solver = pf.PhaseFieldSolver(
        case.msh,
        case.solver.model,
        bcs_u=case.make_bcs,
        f_alpha=state.f_alpha,
        config=case.solver.config,
    )
    state.bind_displacement(case.solver.u)
    amplitude = s * case.strain_c * case.L

    def set_phase(t: float) -> None:
        case.u_applied.value = amplitude * math.sin(2.0 * math.pi * t)

    driver = cj.CycleJumpDriver(
        case.solver,
        state,
        set_phase,
        cj.CycleJumpConfig(n_sub=4, dN_init=8.0, dN_max=8.0, dd_max=0.02),
    )
    report = driver.step()
    assert report.accepted is False
    assert report.failure_mode and "下界" in report.failure_mode
    assert report.rejections >= 3  # 8 -> 4 -> 2 -> 1 至少四次尝试
    assert driver.N == 0.0  # 失败不推进循环数


def test_rejected_jump_restores_state():
    """人为把 dd_max 设得极小以强制拒绝，检查 ᾱ/d 被回滚。"""
    s = 0.8
    material = make_material(nu=0.0)
    increment = analytic_cycle_increment(s, material)
    alpha_T = 5.0 * increment / (2.0 / s - 1.0)
    case, state, driver = make_sinusoidal_case(
        s, alpha_T=alpha_T, n_sub=4, dN_init=64.0, dN_max=64.0, dd_max=1.0e-6
    )
    report = driver.step()
    assert report.accepted
    assert report.rejections > 0, "dd_max = 1e-6 竟然一次通过，说明没触发拒绝"
    assert report.delta_N < report.delta_N_attempted
    assert driver.N == pytest.approx(report.delta_N)
    # **回滚正确性的关键断言**：多次拒绝后 ᾱ 必须正好等于 ΔN 个周期的量，
    # 若回滚漏了 ᾱ，这里会累积成好几倍。
    assert state.max_alpha_bar() == pytest.approx(
        report.delta_N * increment, rel=1e-9
    )
    # 未起裂（ΔN 小于起裂寿命 5）故 d 仍为 0
    assert report.max_d == pytest.approx(0.0, abs=1e-12)
