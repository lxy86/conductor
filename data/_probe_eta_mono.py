# -*- coding: utf-8 -*-
"""零插桩：把 uzawa_eta_rel 降到 1e-3，单体 Newton 能否在相位 0.05 收敛？

D-68 问题 33 续 21 的结论：单体 F_q 的残差地板恒为 eta/2（归一化 eta_rel/2）。
默认 eta_rel=0.05 ⇒ 地板 0.025 = tol(1e-3) 的 25 倍 ⇒ 永远压不下去。
本探针**只改 config**（不改源码），看把 eta 降到地板 < tol 后是否收敛。

用法：bash data/_run_eta_mono.sh <eta_rel>
      bash data/_run_eta_mono.sh 0.001
      bash data/_run_eta_mono.sh 0.0005
"""
import sys

import numpy as np

DELTA_STAR = 2.7252659701799572e-05   # 与 _probe_fail_fast.py 同口径
AMP = 1.0e-07
PHASE = 0.05


def main() -> None:
    eta_rel = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0e-3

    from mpi4py import MPI

    import yaml
    import pathlib

    from pfwire.clamp_case import (build_clamp_case, solve_contact_at,
                                   _enable_friction_after_ramp)
    from pfwire.phasefield import MonolithicNotConverged

    comm = MPI.COMM_WORLD
    rank = comm.rank

    # 读 config → 覆盖 eta → 写临时文件（build_clamp_case 只收路径）
    cfg = pathlib.Path(__file__).resolve().parents[1] / "configs" / "clamp_fretting_coarse.yaml"
    raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    raw = yaml.safe_load(yaml.safe_dump(raw))
    raw.setdefault("contact", {})["uzawa_eta_rel"] = eta_rel
    # 输出目录也改掉，避免覆盖既有结果
    raw.setdefault("output", {})["dir"] = "data/_eta_mono_probe"
    tmp_cfg = pathlib.Path("data/_eta_mono_cfg.yaml")
    tmp_cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")

    if rank == 0:
        print("=== eta_rel = %.4e（地板/ tol = %.3f）===" % (eta_rel, (eta_rel / 2) / 1e-3),
              flush=True)

    case = build_clamp_case(str(tmp_cfg), comm)
    solver, clamp = case.solver, case.clamp

    if rank == 0:
        print("延拓 + W' 搜索 ...", flush=True)
    solve_contact_at(case, DELTA_STAR)
    _enable_friction_after_ramp(case)
    clamp.bind_monolithic_space(solver.V_u)

    # 相位 0.05 —— 就是续 19/20 卡住的那个点
    case.loads.set_phase(np.sin(2 * np.pi * PHASE))
    clamp.set_tangential(AMP * np.sin(2 * np.pi * PHASE))

    if rank == 0:
        print("求解相位 %.2f ..." % PHASE, flush=True)
    try:
        solver.solve_displacement()
        conv = getattr(solver, "uzawa_converged", None)
        u_max = float(np.max(np.abs(solver.u.x.array)))
        q = clamp.q
        q_max = float(np.max(np.abs(q.x.array))) if q is not None else 0.0
        if rank == 0:
            print("  ✅ 收敛: uzawa_converged=%s  max|u|=%.4e  max|q|=%.4e"
                  % (conv, u_max, q_max), flush=True)
            print("  u 的物理尺度参考：delta = %.3e" % DELTA_STAR, flush=True)
    except MonolithicNotConverged as exc:
        if rank == 0:
            print("  ❌ MonolithicNotConverged: reason=%s it=%d ||F||=%.4e tol=%.1e"
                  % (exc.reason, exc.iterations, exc.residual, exc.tol), flush=True)
    except Exception as exc:      # noqa: BLE001
        if rank == 0:
            print("  ❌ %s: %s" % (type(exc).__name__, str(exc)[:200]), flush=True)


if __name__ == "__main__":
    main()
