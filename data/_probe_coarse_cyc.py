# -*- coding: utf-8 -*-
"""零插桩探针：粗网格上逐相位看单体 Newton 的收敛标志（续 19 续 7）。

tiny 网格（5257 单元）20 个子步**全部 converged**、max|u| ~1.5e-5 —— 复现不了。
故爆炸是**粗网格特有**的。

本探针用生产入口装配同一个工况（configs/clamp_fretting_coarse.yaml），
延拓到 result.json 记录的 δ*，开摩擦，然后逐相位调
``solver.solve_displacement()`` —— 与 explicit_cycle 完全同一入口 ——
读**公开属性** ``solver.uzawa_converged`` / ``solver.uzawa_iters``
（``_solve_u_monolithic`` 会把 BlockNewtonResult 写进去）。

⚠ 零插桩：不往 solve_displacement / solve_monolithic 里加任何代码。
  若某相位 converged=False 而循环照常推进 ⇒ 失败被静默吞掉（H1）。

一旦发现未收敛的相位，就地再用 ``clamp.solve_monolithic(..., verbose=True)``
重跑该相位拿残差历史（也是公开 API 的公开参数）。
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, "/root/pf-wire")

import ufl

DELTA_STAR = 2.7252659701799572e-05   # result.json 里 W' 搜索的结果
AMP = 1.0e-07
N_SUB = 20


def _elastic_external(solver):
    """复刻 PhaseFieldSolver._solve_u_monolithic 内部的 _elastic_external。"""
    def _f(v):
        F = solver._residual_u_form(v)
        if getattr(solver.contact, "has_friction_at_build", False):
            F = F + solver.contact.tangential_residual(solver.u, v)
        return F
    return _f


def main():
    from mpi4py import MPI
    from pfwire.clamp_case import (build_clamp_case, solve_contact_at,
                                   _enable_friction_after_ramp)
    from pfwire.mesh import BLOCK_TOP
    from dolfinx import fem

    comm = MPI.COMM_WORLD
    print("装配粗网格工况…", flush=True)
    case = build_clamp_case("configs/clamp_fretting_coarse.yaml", comm)
    solver, clamp = case.solver, case.clamp
    u, V = solver.u, solver.V_u
    print("  单元数 %d" % case.mesh.topology.index_map(2).size_global, flush=True)

    print("延拓到 δ* = %.6e …" % DELTA_STAR, flush=True)
    W = solve_contact_at(case, DELTA_STAR)
    print("  W = %.6e（目标 1.75e6）" % W, flush=True)
    _enable_friction_after_ramp(case)

    def state():
        n_q = clamp.q.function_space.dofmap.index_map.size_local
        ab = case.state.alpha_bar.x.array
        return (float(np.max(np.abs(u.x.array))),
                float(np.max(np.abs(clamp.q.x.array[:n_q]))),
                float(np.max(np.abs(ab))))

    print()
    print("  phase  conv  it  max|u|      max|q|      max|alpha_bar|")
    first_bad = None
    for k in range(0, N_SUB + 1):
        phase = k / N_SUB
        s = np.sin(2 * np.pi * phase)
        case.loads.set_phase(s)
        clamp.set_tangential(AMP * s)
        solver.solve_displacement()
        conv = getattr(solver, "uzawa_converged", None)
        nit = getattr(solver, "uzawa_iters", None)
        mu, q, ab = state()
        print("  %.2f   %-5s %2s  %.3e  %.3e  %.3e"
              % (phase, conv, nit, mu, q, ab), flush=True)
        if conv is False and first_bad is None:
            first_bad = phase
            print("  ⚠⚠ 该相位**未收敛**但 solve_displacement 正常返回 —— H1 成立",
                  flush=True)
            break

    if first_bad is not None:
        print()
        print("重跑相位 %.2f 拿残差历史（verbose=True）…" % first_bad, flush=True)
        s = np.sin(2 * np.pi * first_bad)
        case.loads.set_phase(s)
        clamp.set_tangential(AMP * s)
        # 先把 u 退回到延拓结束时的量级，否则从 1e7 起步看不出是不是初值问题
        res = clamp.solve_monolithic(
            u, _elastic_external(solver), solver.bcs_u,
            tol=1.0e-3, max_it=50, verbose=True)
        print("  ⇒ converged=%s reason=%s it=%d ||F||=%.4e"
              % (res.converged, res.reason, res.iterations, res.residual))
    else:
        print()
        print("  所有相位都 converged —— 与粗网格 checkpoint 的 6.1e7 不符，")
        print("  说明爆炸不在 solve_displacement，需另找来源。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
