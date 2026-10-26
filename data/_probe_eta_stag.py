# -*- coding: utf-8 -*-
"""零插桩对照：同一相位 0.05、同一 eta_rel 下，**交错 Uzawa** 能否收敛？

与 _probe_eta_mono.py 唯一差别是 config 里 monolithic_friction=false。
目的：判断路线 3（回交错）到底是「零成本回退」还是「同样卡住」。

交错法的收敛判据是 max|Δq| / max(1,max|q|)（比的是**更新量**，不是残差），
所以它不受 F_q 的 eta/2 残差地板影响 —— 这是它能在 6.6.3 前几个相位跑通的原因。
但续 13 记录过 blocker B：深度黏着下交错法进二周期极限环（dq 恒 0.268）。

用法：bash data/_run_eta_stag.sh [eta_rel] [phase]
"""
import sys

import numpy as np

DELTA_STAR = 2.7252659701799572e-05
AMP = 1.0e-07


def main() -> None:
    eta_rel = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    phase = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

    from mpi4py import MPI
    import pathlib
    import yaml

    from pfwire.clamp_case import (build_clamp_case, solve_contact_at,
                                   _enable_friction_after_ramp)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    cfg = pathlib.Path(__file__).resolve().parents[1] / "configs" / "clamp_fretting_coarse.yaml"
    raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    raw = yaml.safe_load(yaml.safe_dump(raw))
    raw.setdefault("contact", {})["uzawa_eta_rel"] = eta_rel
    raw.setdefault("solver", {})["monolithic_friction"] = False
    raw.setdefault("output", {})["dir"] = "data/_eta_stag_probe"
    tmp_cfg = pathlib.Path("data/_eta_stag_cfg.yaml")
    tmp_cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")

    if rank == 0:
        print("=== 交错 Uzawa, eta_rel=%.4e, phase=%.2f ===" % (eta_rel, phase), flush=True)

    case = build_clamp_case(str(tmp_cfg), comm)
    solver, clamp = case.solver, case.clamp

    if rank == 0:
        print("延拓 + W' 搜索 ...", flush=True)
    solve_contact_at(case, DELTA_STAR)
    _enable_friction_after_ramp(case)

    # ⚠ 必须复现真实流程的顺序：explicit_cycle 先在相位 0 处 prime，
    # 再逐步推进。直接跳到 0.05 会让 u_prev/滑移基准停在延拓末的状态，
    # 那不是 6.6.3 实际走过的路径（2026-10-26 实测：跳过 prime 会假失败）。
    if rank == 0:
        print("相位 0.00（prime 起点）...", flush=True)
    case.loads.set_phase(0.0)
    clamp.set_tangential(0.0)
    try:
        solver.solve_displacement()
    except Exception as exc:                                  # noqa: BLE001
        if rank == 0:
            print("  ❌ 相位 0 就失败：%s: %s"
                  % (type(exc).__name__, str(exc)[:300]), flush=True)
        return
    if rank == 0:
        print("  ✅ 相位 0 ok", flush=True)

    case.loads.set_phase(np.sin(2 * np.pi * phase))
    clamp.set_tangential(AMP * np.sin(2 * np.pi * phase))

    if rank == 0:
        print("求解相位 %.2f ..." % phase, flush=True)
    try:
        solver.solve_displacement()
    except Exception as exc:                                  # noqa: BLE001
        if rank == 0:
            print("  ❌ %s: %s" % (type(exc).__name__, str(exc)[:300]), flush=True)
        return

    if rank == 0:
        conv = getattr(solver, "uzawa_converged", None)
        u_max = float(np.max(np.abs(solver.u.x.array)))
        q = clamp.q
        q_max = float(np.max(np.abs(q.x.array))) if q is not None else 0.0
        print("  ✅ 跑完: uzawa_converged=%s  max|u|=%.4e  max|q|=%.4e"
              % (conv, u_max, q_max), flush=True)


if __name__ == "__main__":
    main()
