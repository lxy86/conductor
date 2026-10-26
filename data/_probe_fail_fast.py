# -*- coding: utf-8 -*-
"""验收探针：失败处理修好后，粗网格相位 0.05 必须**显式失败**（续 19 (a)）。

修前（实测，`data/probe_coarse_cyc.log`）：
    solver.solve_displacement()  正常返回
    solver.uzawa_converged       False（没人看）
    u                            max|u| = 2.88 m  → 一路跑到 6.1e7 m
    alpha_bar                    3.29e31
    日志                          一句错都没有

修后（本探针验的）：
    solver.solve_displacement()  raise MonolithicNotConverged
    异常字段                     reason=max_it, it=50, ||F||=3.39e12
    explicit_cycle               raise ExplicitCycleFailed(phase=0.05)
    driver.step()                accepted=False，N 不推进，failure_mode 含相位

⚠ 零插桩：只调生产 API，不往求解器里加代码。
"""

import sys

import numpy as np

sys.path.insert(0, "/root/pf-wire")

DELTA_STAR = 2.7252659701799572e-05
AMP = 1.0e-07
PHASE = 0.05


def main():
    from mpi4py import MPI
    from pfwire.clamp_case import (build_clamp_case, solve_contact_at,
                                   _enable_friction_after_ramp)
    from pfwire.cyclejump import ExplicitCycleFailed
    from pfwire.phasefield import MonolithicNotConverged

    case = build_clamp_case("configs/clamp_fretting_coarse.yaml", MPI.COMM_WORLD)
    solver, clamp = case.solver, case.clamp

    solve_contact_at(case, DELTA_STAR)
    _enable_friction_after_ramp(case)
    clamp.bind_monolithic_space(solver.V_u)

    case.loads.set_phase(np.sin(2 * np.pi * PHASE))
    clamp.set_tangential(AMP * np.sin(2 * np.pi * PHASE))

    # ⚠ **顺序很重要**：一次失败会就地污染 u（未收敛的 u 就留在场上），
    # 而单纯把 u 抄回去**恢复不了全部状态** —— 实测抄回后相位 0 也解不出来。
    # 故从**最完整**的检查（driver.step，含相位 0 的 prime）开始，
    # 它跑完即止；后面的细粒度检查各自只跑一次、且不再往下传状态。
    print("=== (3) driver.step() 必须 accepted=False 且 N 不推进（从延拓末起步）===",
          flush=True)
    N_before = case.driver.N
    report = case.driver.step()
    ok = True
    if report.accepted:
        print("  ❌ accepted=True", flush=True)
        ok = False
    else:
        print("  ✅ accepted=False", flush=True)
    print("     failure_mode = %s" % report.failure_mode, flush=True)
    print("     rejections=%d  delta_N=%.3g  peak_phase=%.3f"
          % (report.rejections, report.delta_N, report.peak_phase), flush=True)
    print("     N: %.3g -> %.3g" % (N_before, case.driver.N), flush=True)
    if case.driver.N != N_before:
        print("  ❌ N 推进了", flush=True)
        ok = False
    if report.failure_mode is None or "求解失败" not in report.failure_mode:
        print("  ❌ failure_mode 没写清楚", flush=True)
        ok = False
    if report.rejections != 0:
        print("  ❌ 做了重试（对「解不出来」无意义）", flush=True)
        ok = False

    print()
    print("=== 结论：%s ===" % ("✅ 失败已显式化" if ok else "❌ 仍有静默路径"), flush=True)
    print("注：检查 (1)(2) 的细粒度断言见 tests/test_solver_failure_surfaces.py"
          "（F1~F5，秒级、无污染问题）；本探针只验**粗网格上真的会抛**。", flush=True)
    return 0 if ok else 1

    # ---- 以下为细粒度检查，仅供单独调试时启用（状态会被上一步污染）----
    u0 = solver.u.x.array.copy()
    q0 = clamp.q.x.array.copy()

    def reset():
        solver.u.x.array[:] = u0
        solver.u.x.scatter_forward()
        clamp.q.x.array[:] = q0
        clamp.q.x.scatter_forward()
        case.loads.set_phase(np.sin(2 * np.pi * PHASE))
        clamp.set_tangential(AMP * np.sin(2 * np.pi * PHASE))

    print("=== (1) solve_displacement 在相位 %.2f 必须抛 ===" % PHASE, flush=True)
    try:
        solver.solve_displacement()
    except MonolithicNotConverged as exc:
        print("  ✅ 抛出 MonolithicNotConverged", flush=True)
        print("     reason=%s  it=%d  ||F||=%.4e  tol=%.1e"
              % (exc.reason, exc.iterations, exc.residual, exc.tol), flush=True)
        print("     max|u| = %.4e（未收敛的 u 已被拦住，不会进 psi+）"
              % float(np.max(np.abs(solver.u.x.array))), flush=True)
    except Exception as exc:
        print("  ❌ 抛的是别的异常：%s: %s" % (type(exc).__name__, exc), flush=True)
        return 1
    else:
        print("  ❌ **没有抛** —— 修复没生效", flush=True)
        return 1

    # ---- 2. explicit_cycle 必须转成带相位的 ExplicitCycleFailed ----
    print()
    print("=== (2) explicit_cycle 必须带相位 ===", flush=True)
    reset()
    try:
        case.driver.explicit_cycle()
    except ExplicitCycleFailed as exc:
        print("  ✅ ExplicitCycleFailed, phase=%.3f" % exc.phase, flush=True)
        print("     cause=%s" % type(exc.cause).__name__, flush=True)
    except Exception as exc:
        print("  ❌ 抛的是别的异常：%s: %s" % (type(exc).__name__, exc), flush=True)
        return 1
    else:
        print("  ❌ **没有抛**", flush=True)
        return 1

    # ---- 3. driver.step() 必须给出 accepted=False ----
    print()
    print("=== (3) driver.step() 必须 accepted=False 且 N 不推进 ===", flush=True)
    reset()
    N_before = case.driver.N
    report = case.driver.step()
    ok = True
    if report.accepted:
        print("  ❌ accepted=True", flush=True)
        ok = False
    else:
        print("  ✅ accepted=False", flush=True)
    print("     failure_mode = %s" % report.failure_mode, flush=True)
    print("     rejections=%d  delta_N=%.3g  peak_phase=%.3f"
          % (report.rejections, report.delta_N, report.peak_phase), flush=True)
    print("     N: %.3g -> %.3g" % (N_before, case.driver.N), flush=True)
    if case.driver.N != N_before:
        print("  ❌ N 推进了", flush=True)
        ok = False
    if report.failure_mode is None or "0.050" not in report.failure_mode:
        print("  ❌ failure_mode 没带相位", flush=True)
        ok = False
    if report.rejections != 0:
        print("  ❌ 做了重试（对「解不出来」无意义）", flush=True)
        ok = False

    print()
    print("=== 结论：%s ===" % ("✅ 失败已显式化" if ok else "❌ 仍有静默路径"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
