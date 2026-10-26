# -*- coding: utf-8 -*-
"""D-68 问题 33 续 19：求解器失败**必须显式失败**，不许静默继续。

背景（实测）
------------
粗网格 6.6.3 首跑（单体摩擦开启）：相位 0.05 的单体 Newton 未收敛
（50 次迭代、``max|q|=5.8e13``），而 ``_solve_u_monolithic`` **只 warn**，
于是 ``explicit_cycle`` 拿着未收敛的 u 继续累积 ``alpha_bar``：

    max|u|        = 6.139e7 m     （位移尺度 2.7e-5 m 的 2.3e12 倍）
    max|alpha_bar|= 3.29e31
    max|d|        = 0.0           （d 一步没动）

最后表现成「d 的 SNES -6」，**病因完全指错地方**。交错路径
``_solve_u_once`` 一直是抛的 —— 两条路的失败处理必须一致。

测试意图
--------
F1 ``MonolithicNotConverged`` 携带 reason/iterations/residual/tol，
   且消息里能读到关键量（不必解字符串也能诊断）。
F2 ``explicit_cycle`` 把子步失败转成 ``ExplicitCycleFailed``，**带相位**。
F3 ``step()`` 把 ``ExplicitCycleFailed`` 转成 ``accepted=False`` 的
   ``JumpReport``：不推进 N、failure_mode 含相位、不重试（rejections=0）。
F4 相位 0（prime 起点）失败同样被转成 report，且 N 不推进。
F5 成功的周期**不受影响**（回归保护：新加的 try 不能改变正常路径）。

F1–F5 全用假 solver（不建网格、不装配），秒级。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pfwire import cyclejump as cj
from pfwire.phasefield import MonolithicNotConverged


# --------------------------------------------------------------------------
# 假 solver / state：只实现 CycleJumpDriver 用到的那几个属性
# --------------------------------------------------------------------------
class _FakeState:
    """``explicit_cycle`` 只用到 prime / accumulate / alpha_bar。"""

    def __init__(self, msh, n: int = 4) -> None:
        import ufl
        from dolfinx import fem
        V = fem.functionspace(msh, ("DG", 0))
        self._ab = fem.Function(V)
        self.psi_plus_prev = fem.Function(V)
        self.n_owned = self._ab.function_space.dofmap.index_map.size_local
        # CycleJumpDriver 只用 .x.array，包一层保持接口一致
        self.alpha_bar = self._ab

    def prime(self) -> None:
        pass

    def accumulate(self) -> None:
        self.alpha_bar.x.array[:] += 1.0

    def extrapolate(self, delta_N: float, increment) -> None:
        """ΔN - 1 个额外周期的外推（F5b 走接受路径时需要）。"""
        self.alpha_bar.x.array[:] += delta_N * np.asarray(increment)

    def max_alpha_bar(self) -> float:
        return float(np.max(self.alpha_bar.x.array))


class _FakeSolver:
    """``solve_displacement`` 在指定相位抛异常，其余正常。"""

    def __init__(self, msh, fail_at: float | None = None,
                 exc: BaseException | None = None):
        import ufl
        from dolfinx import fem
        self._msh = msh
        self._dx = ufl.dx(domain=msh)
        self.fail_at = fail_at
        self.exc = exc or MonolithicNotConverged("max_it", 50, 3.39e12, 1.0e-3)
        self.calls: list[float] = []
        self._phase = 0.0
        self.comm = type("C", (), {"rank": 0})()
        self.plastic = None
        V = fem.functionspace(msh, ("Lagrange", 1))
        self.u = fem.Function(V)
        self.d = fem.Function(V)
        self.d_old = fem.Function(V)
        # step() 开头读 V_d.dofmap.index_map.size_local
        self.V_d = V

    def set_phase(self, p: float) -> None:
        self._phase = p

    def solve_displacement(self):
        self.calls.append(self._phase)
        if self.fail_at is not None and abs(self._phase - self.fail_at) < 1e-12:
            raise self.exc
        return 1, 0

    def max_damage(self) -> float:
        return 0.0

    def commit_step(self) -> None:
        pass

    # CycleJumpDriver.__init__ 会建 fem.form(psi_plus_expr() * dx)
    def psi_plus_expr(self):
        import ufl
        return ufl.as_ufl(1.0)

    @property
    def dx(self):
        return self._dx


def _tiny_mesh():
    """1x1 单位方块的 2 个三角形：只为了让 fem.form 有积分域。"""
    from dolfinx import mesh as dmesh
    from mpi4py import MPI
    return dmesh.create_unit_square(MPI.COMM_WORLD, 1, 1)


def _make_driver(fail_at, exc=None, n_sub=4):
    msh = _tiny_mesh()
    solver = _FakeSolver(msh, fail_at=fail_at, exc=exc)
    state = _FakeState(msh)
    cfg = cj.CycleJumpConfig(n_sub=n_sub, dN_init=1.0, dN_max=1.0)
    driver = cj.CycleJumpDriver(solver, state, solver.set_phase, config=cfg)
    # _psi_plus_total 走 assemble_scalar，假 solver 没有真 form —— 直接替换
    driver._psi_plus_total = lambda: 1.0
    driver._max_over_owned = lambda arr, n: float(np.max(np.abs(arr[:n]), initial=0.0))
    return driver, solver, state


# --------------------------------------------------------------------------
# F1：异常类型携带诊断字段
# --------------------------------------------------------------------------
def test_F1_monolithic_not_converged_carries_diagnostics():
    """F1：字段齐全，且消息里能读到 reason/迭代数/残差/tol。"""
    exc = MonolithicNotConverged("max_it", 50, 3.39e12, 1.0e-3)
    assert exc.reason == "max_it"
    assert exc.iterations == 50
    assert exc.residual == pytest.approx(3.39e12)
    assert exc.tol == pytest.approx(1.0e-3)
    msg = str(exc)
    for needle in ("max_it", "50", "3.39", "1.0e-03"):
        assert needle in msg, "消息里应能读到 %r：%s" % (needle, msg)
    assert isinstance(exc, RuntimeError)


# --------------------------------------------------------------------------
# F2：explicit_cycle 转成带相位的异常
# --------------------------------------------------------------------------
def test_F2_explicit_cycle_raises_with_phase():
    """F2：子步 0.50 失败 ⇒ ExplicitCycleFailed 且 phase == 0.50。"""
    driver, _solver, _state = _make_driver(fail_at=0.50)
    with pytest.raises(cj.ExplicitCycleFailed) as ei:
        driver.explicit_cycle()
    exc = ei.value
    assert exc.phase == pytest.approx(0.50)
    assert isinstance(exc.cause, MonolithicNotConverged)
    assert "0.500" in str(exc)
    # 原因链没丢：调用方还能拿到原始诊断
    assert exc.cause.iterations == 50


def test_F2b_phase_zero_failure_is_reported_as_phase_zero():
    """F2b：prime 起点（相位 0）失败也标成相位 0，不能标成别的。"""
    driver, _solver, _state = _make_driver(fail_at=0.0)
    with pytest.raises(cj.ExplicitCycleFailed) as ei:
        driver.explicit_cycle()
    assert ei.value.phase == pytest.approx(0.0)


# --------------------------------------------------------------------------
# F3：step() 转成 accepted=False 的 report
# --------------------------------------------------------------------------
def test_F3_step_returns_rejected_report_not_retry():
    """F3：不重试（ΔN 减半对「解不出来」没用），N 不推进，failure_mode 含相位。"""
    driver, solver, _state = _make_driver(fail_at=0.50)
    report = driver.step()

    assert report.accepted is False
    assert report.failure_mode, "失败必须带 failure_mode"
    assert "0.500" in report.failure_mode, report.failure_mode
    assert "单体" in report.failure_mode or "max_it" in report.failure_mode
    # 不重试：显式周期都跑不完，ΔN 减半毫无意义
    assert report.rejections == 0
    assert report.delta_N == 0.0
    # N 不推进
    assert driver.N == 0.0
    assert math.isnan(report.max_dd)
    # 相位被记下来，便于定位
    assert report.peak_phase == pytest.approx(0.50)
    # report 进历史，run() 才能交给 stop()
    assert driver.reports[-1] is report
    # 失败发生在相位 0.50，之前的子步确实跑过（不是没开始就报错）
    assert 0.50 in solver.calls


def test_F4_phase_zero_failure_also_yields_report():
    """F4：相位 0 失败 ⇒ 同样是 accepted=False 的 report，N 不推进。"""
    driver, _solver, _state = _make_driver(fail_at=0.0)
    report = driver.step()
    assert report.accepted is False
    assert report.peak_phase == pytest.approx(0.0)
    assert driver.N == 0.0


def test_F4b_run_stops_on_the_failed_report():
    """F4b：``run()`` 遇到失败必须停下来，且把失败那一步交给 stop()。"""
    driver, _solver, _state = _make_driver(fail_at=0.50)
    seen = []
    reports = driver.run(stop=lambda r: seen.append(r) or False)
    assert len(reports) == 1
    assert reports[0].accepted is False
    assert seen and seen[-1] is reports[0], "失败那一步必须进 stop()"


# --------------------------------------------------------------------------
# F5：正常路径不受影响（回归保护）
# --------------------------------------------------------------------------
def test_F5_successful_cycle_unchanged():
    """F5：没有失败时 explicit_cycle 照常返回增量，try 不改变正常路径。"""
    driver, _solver, state = _make_driver(fail_at=None, n_sub=4)
    delta_alpha, peak = driver.explicit_cycle()
    # accumulate 每子步 +1，共 n_sub=4 次；prime 不清零（假 state 的 prime 是空操作）
    assert np.allclose(delta_alpha, 4.0)
    assert 0.0 <= peak <= 1.0
    assert state.max_alpha_bar() == pytest.approx(4.0)


# 注：step() 的**成功**路径不在这里用假 solver 测 ——
# tests/test_cyclejump_driver.py 已用真求解器端到端覆盖（解析锚点 + 外推 + 自适应），
# 比再堆一个假 solver 更可靠。本文件只守「失败必须显式」这一条。
