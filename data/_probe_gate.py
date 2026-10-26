# -*- coding: utf-8 -*-
"""零插桩：关掉方向 f 的 `_tang_jac_gate`，u 在相位 0.05 能否解出？

### 为什么要做这个实验

D-68 问题 33 续 15 已用**形式判据**证伪方向 f：
Uzawa 模式 F 里的切向项是 `-q·(τ·v)·ds`，它的 u-导数只有 `-q·(dτ/du[du])·v`；
而 `tangential_jacobian` 往 J 里加的 `ε_t·(τ·du)(τ·v)` 在 F 里**没有对应项**，
量级是真导数的 9.5e7 倍（续 15 实测）。Newton 拿这种 J，搜索方向不是下降方向。

但那份代码**一直留在 `_build_u_problem` 里**，只由 `_tang_jac_gate` 控制：
`set_mu(μ>0)` 时门 = 1.0 ⇒ 开摩擦后每次 u 的 Newton 都带着这根一亿倍的假弹簧。

`data/_probe_u_block.py` 已实测：相位 0.05 时 u **单独**就 `reason=-5`
（跑满 200 次，fnorm=4763 不降），而相位 0 只需 1 次迭代。
⇒ 假弹簧只在「残差非零、Newton 真的要走」时才发作 —— 与续 15 的判据吻合。

### 为什么只做一次求解、门值从命令行给

CLAUDE.md 陷阱 9：一次失败会就地改写 u，抄回去恢复不了全部状态。
故每次调用**只跑一个门值**，由 _run_gate.sh 起独立进程分别跑。

`_tang_jac_gate` 是 `fem.Constant` —— 改它就是改一个数据值，
不在任何求解器方法体内、不触发 import/IO，不算插桩（陷阱 8 的反面）。

用法：bash data/_run_gate.sh        # 跑 1.0 与 0.0 两组
      bash data/_run_gate.sh 0.0    # 单跑一组
"""
import sys

import numpy as np

DELTA_STAR = 2.7252659701799572e-05
AMP = 1.0e-07
PHASE = 0.05


def _snes_state(solver, tag: str) -> None:
    snes = solver._u_problem.solver
    try:
        fnorm = snes.getFunctionNorm()
    except Exception as exc:                                  # noqa: BLE001
        fnorm = "unavailable(%s)" % type(exc).__name__
    try:
        rtol, atol, _dtol, max_it = snes.getTolerances()
        tol = "rtol=%.1e atol=%.1e max_it=%d" % (rtol, atol, max_it)
    except Exception:                                          # noqa: BLE001
        tol = "tolerances unavailable"
    print("    [%s] reason=%d  it=%s  fnorm=%s  %s"
          % (tag, snes.getConvergedReason(), snes.getIterationNumber(), fnorm, tol),
          flush=True)


def main() -> None:
    gate_value = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

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
    raw.setdefault("output", {})["dir"] = "data/_gate_probe"
    tmp_cfg = pathlib.Path("data/_gate_cfg.yaml")
    tmp_cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")

    case = build_clamp_case(str(tmp_cfg), comm)
    solver, clamp = case.solver, case.clamp

    solve_contact_at(case, DELTA_STAR)
    _enable_friction_after_ramp(case)

    gate = clamp._tang_jac_gate
    gate.value = gate_value

    if rank == 0:
        print("=== gate=%.1f（1.0 = 方向 f 的假弹簧进 J；0.0 = 关掉）===" % gate_value,
              flush=True)

    # 相位 0 先解一次：这是 explicit_cycle 的真实起点，也保证 q 仍为 0
    case.loads.set_phase(0.0)
    clamp.set_tangential(0.0)
    try:
        solver._solve_u_once()
    except Exception as exc:                                  # noqa: BLE001
        if rank == 0:
            print("  ❌ 相位 0 就失败：%s" % str(exc)[:150], flush=True)
            _snes_state(solver, "phase0.00")
        return

    # 相位 0.05，u 单独求解（q 尚未更新，恒为 0）
    case.loads.set_phase(np.sin(2 * np.pi * PHASE))
    clamp.set_tangential(AMP * np.sin(2 * np.pi * PHASE))
    if rank == 0:
        print("相位 %.2f（u 单独求解，q 恒为 0）..." % PHASE, flush=True)
    try:
        solver._solve_u_once()
        if rank == 0:
            _snes_state(solver, "phase0.05")
            print("  ✅ u 可解", flush=True)
    except Exception as exc:                                  # noqa: BLE001
        if rank == 0:
            _snes_state(solver, "phase0.05")
            print("  ❌ u 失败：%s" % str(exc)[:150], flush=True)


if __name__ == "__main__":
    main()
