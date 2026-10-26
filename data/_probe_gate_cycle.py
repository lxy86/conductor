# -*- coding: utf-8 -*-
"""零插桩：gate=0 下，完整 explicit_cycle（交错 Uzawa）能否跑完一个周期？

_probe_gate.py 已证明：相位 0.05 的 u 单独求解，gate=1.0 → -5 跑满 200 次；
gate=0.0 → 4 次收敛。但那只验了**第一次** u 求解（q 仍为 0）。
本探针验真正的 6.6.3 流程：相位 0 prime → 各子步 → q 逐步更新。

判读：
  · 跑完 ⇒ 方向 f 的假弹簧就是 6.6.3 的病因，三条改 F_q 的路线都不需要。
  · 仍失败 ⇒ 假弹簧只是**第一层**，后面还有 blocker B。
"""
import sys

import numpy as np

DELTA_STAR = 2.7252659701799572e-05


def main() -> None:
    gate_value = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

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
    raw.setdefault("solver", {})["monolithic_friction"] = False
    raw.setdefault("output", {})["dir"] = "data/_gate_cycle_probe"
    tmp_cfg = pathlib.Path("data/_gate_cycle_cfg.yaml")
    tmp_cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")

    case = build_clamp_case(str(tmp_cfg), comm)
    solver, clamp = case.solver, case.clamp

    solve_contact_at(case, DELTA_STAR)
    _enable_friction_after_ramp(case)
    clamp._tang_jac_gate.value = gate_value

    if rank == 0:
        print("=== gate=%.1f，完整 explicit_cycle（交错 Uzawa）===" % gate_value,
              flush=True)

    try:
        delta_alpha, peak_phase = case.driver.explicit_cycle()
    except Exception as exc:                                  # noqa: BLE001
        if rank == 0:
            cause = getattr(exc, "cause", None)
            phase = getattr(exc, "phase", None)
            print("  ❌ %s（phase=%s）" % (type(exc).__name__, phase), flush=True)
            if cause is not None:
                print("     cause: %s: %s"
                      % (type(cause).__name__, str(cause)[:200]), flush=True)
            it = getattr(solver, "uzawa_iters", None)
            print("     uzawa_iters=%s  uzawa_converged=%s"
                  % (it, getattr(solver, "uzawa_converged", None)), flush=True)
        return

    if rank == 0:
        da = float(np.max(np.abs(delta_alpha)))
        q = clamp.q
        q_max = float(np.max(np.abs(q.x.array))) if q is not None else 0.0
        print("  ✅ 周期跑完: peak_phase=%.3f  max|Δᾱ|=%.4e  max|q|=%.4e"
              % (peak_phase, da, q_max), flush=True)
        print("     uzawa_iters=%s  uzawa_converged=%s"
              % (getattr(solver, "uzawa_iters", None),
                 getattr(solver, "uzawa_converged", None)), flush=True)


if __name__ == "__main__":
    main()
