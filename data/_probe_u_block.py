# -*- coding: utf-8 -*-
"""零插桩：相位 0.05 下，u 块本身是否可解？（q 冻结在延拓末值）

动机（2026-10-26）：三条路线全部实测失败 ——
  · 单体 eta_rel=1e-3（地板降到 tol 以下）仍 max_it、||F||=2.81e13
  · 交错 Uzawa 在同相位报 u 的 SNESConvergedReason=-5（DIVERGED_FUNCTION_DOMAIN）
  · 单体 eta_rel=0.05 报 max_it、||F||≈5e14
两条路都**卡在同一相位**，且交错法在**第一次 q 更新之前**就失败。
⇒ 怀疑真因不在 q 块（三条路线都在改 q 块），而在 u 块。

本探针只调生产 API：
  1. 相位 0 解一次（对照，已知 ok）
  2. 相位 0.05 只调 solver._solve_u_once()（= 交错循环的第一步，q 尚未更新）
     捕获异常后读 SNES 公开状态：reason / function norm / 迭代数
  3. 若 function norm 是 nan/inf ⇒ 起点残差非有限（不是"没有根"，是求值就崩）
"""
import sys

import numpy as np

DELTA_STAR = 2.7252659701799572e-05
AMP = 1.0e-07


def _snes_state(solver, tag: str) -> None:
    """读 u 的 SNES 公开状态（不插桩）。"""
    snes = solver._u_problem.solver
    try:
        fnorm = snes.getFunctionNorm()
    except Exception as exc:                                  # noqa: BLE001
        fnorm = "unavailable(%s)" % type(exc).__name__
    try:
        rtol, atol, dtol, max_it = snes.getTolerances()
        tol = "rtol=%.1e atol=%.1e max_it=%d" % (rtol, atol, max_it)
    except Exception:                                          # noqa: BLE001
        tol = "tolerances unavailable"
    print("    [%s] reason=%d  it=%s  fnorm=%s  %s"
          % (tag, snes.getConvergedReason(), snes.getIterationNumber(), fnorm, tol),
          flush=True)


def main() -> None:
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
    # 用交错路径（u 单独求解、q 不更新）—— 正是我们要隔离的东西
    raw.setdefault("solver", {})["monolithic_friction"] = False
    raw.setdefault("output", {})["dir"] = "data/_u_block_probe"
    tmp_cfg = pathlib.Path("data/_u_block_cfg.yaml")
    tmp_cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")

    case = build_clamp_case(str(tmp_cfg), comm)
    solver, clamp = case.solver, case.clamp

    solve_contact_at(case, DELTA_STAR)
    _enable_friction_after_ramp(case)

    if rank == 0:
        print("=== u 块可解性（q 冻结，不更新）===", flush=True)

    # ---- 对照：相位 0 ----
    case.loads.set_phase(0.0)
    clamp.set_tangential(0.0)
    if rank == 0:
        print("相位 0.00 ...", flush=True)
    try:
        solver._solve_u_once()
        if rank == 0:
            _snes_state(solver, "phase 0.00")
            print("  ✅ 相位 0 的 u 可解", flush=True)
    except Exception as exc:                                  # noqa: BLE001
        if rank == 0:
            print("  ❌ 相位 0 就失败：%s" % exc, flush=True)
            _snes_state(solver, "phase 0.00")
        return

    # ---- 目标：相位 0.05 ----
    phase = 0.05
    case.loads.set_phase(np.sin(2 * np.pi * phase))
    clamp.set_tangential(AMP * np.sin(2 * np.pi * phase))
    if rank == 0:
        print("相位 0.05（只解 u，q 不动）...", flush=True)
    try:
        solver._solve_u_once()
        if rank == 0:
            _snes_state(solver, "phase 0.05")
            print("  ✅ 相位 0.05 的 u **单独**可解 ⇒ 问题在 q 块/耦合", flush=True)
    except Exception as exc:                                  # noqa: BLE001
        if rank == 0:
            print("  ❌ 相位 0.05 的 u **单独**就失败：%s" % exc, flush=True)
            _snes_state(solver, "phase 0.05")
            print("  ⇒ u 块自身有问题（不是 q 格式）；三条改 F_q 的路线都不对症", flush=True)


if __name__ == "__main__":
    main()
