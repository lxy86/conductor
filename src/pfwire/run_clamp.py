"""Phase 6.5 基准线夹工况的命令行入口（D-67）。

用法::

    mpirun -n 6 python -m pfwire.run_clamp configs/clamp_base.yaml
    mpirun -n 6 python -m pfwire.run_clamp configs/clamp_base.yaml --resume

流程（D-67 §3–§5）：
  1. 装配（网格布局 C、无摩擦罚接触、端部 T' + 往复弯矩、plastic + contact）；
  2. 割线法搜 δ 使接触反力 = W'（T4），之后保持 δ 不变；
  3. cycle-jump 推进到 θ_later 可测（d ≥ 0.95 达 100 µm）或 N_max；
  4. 按 D-67 §5 判验收（起裂位置、倾斜、转向）；**任一条不满足就停下报告**。
  5. κ 敏感性（configs 的 contact.kappa_sens）：重算接触反力，报告相对变化。

κ 敏感性默认**只比接触反力**（6.1 的口径，阈值 3%），不跑第二遍完整疲劳——
那要付与本体相同的算例代价。
"""

from __future__ import annotations

import argparse
import logging
import sys

from mpi4py import MPI

logger = logging.getLogger("pfwire.clamp")

__all__ = ["main"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 6.5 基准线夹工况")
    ap.add_argument("config", nargs="?", default="configs/clamp_base.yaml")
    ap.add_argument("--resume", action="store_true", help="从 output.dir 的断点续跑")
    ap.add_argument("--no-search", action="store_true",
                    help="跳过 W' 的 δ 搜索，直接用 loading.delta0（调试用）")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    comm = MPI.COMM_WORLD
    from pfwire.clamp_case import (build_clamp_case, find_working_delta,
                                   kappa_sensitivity, run_clamp_case)

    case = build_clamp_case(args.config, comm, resume=args.resume)
    raw = case.config
    W_target = float(raw["loading"]["W_target"])

    if args.no_search:
        delta = float(raw["loading"]["delta0"])
        case.set_delta(delta)
        if comm.rank == 0:
            print("跳过 W' 搜索：δ = %.4g（--no-search）" % delta, flush=True)
    else:
        delta = find_working_delta(
            case, W_target, delta0=float(raw["loading"]["delta0"]),
            rtol=float(raw["loading"]["search"]["rtol"]),
            max_iter=int(raw["loading"]["search"]["max_iter"]))
        W = case.clamp.reaction(case.solver.u)
        if comm.rank == 0:
            print("W' 搜索：δ* = %.6g m，W = %.6g（目标 %.6g，偏差 %.3f%%）"
                  % (delta, W, W_target, 100 * abs(W - W_target) / W_target), flush=True)

    # κ 敏感性：只比接触反力（6.1 的口径）
    #
    # ⚠ 不能「换掉 solver.contact 再重解」：NonlinearProblem 的残差 form 在构造时
    # 就已编译，换属性不会改变已装好的 form，那样重解出的还是 κ=10 的结果
    # （静默给出 rel = 0）。这里另建一个**独立的**弹性接触求解器，与本体同一网格、
    # 同一 δ、同一材料，只比接触反力。
    sens = None
    k_sens = raw["contact"].get("kappa_sens")
    if k_sens is not None:
        sens = kappa_sensitivity(case, float(k_sens))
        if comm.rank == 0:
            print("κ 敏感性：κ = %g → %g，接触反力相对变化 %.3f%%（阈值 3%%，仅报告）"
                  % (sens["kappa"], sens["kappa_sens"], 100 * sens["rel"]), flush=True)

    result = run_clamp_case(case, log=logger)
    if sens is not None:
        result["kappa_sensitivity"] = sens
        if comm.rank == 0:
            import json
            tmp = raw["_outdir"] / "result.json.tmp"
            tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(raw["_outdir"] / "result.json")

    if comm.rank == 0:
        acc = result["acceptance"]
        print("")
        print("验收（D-67 §5）：%s" % ("通过" if acc["passed"] else "**未通过**"))
        for name, ok in acc["checks"].items():
            print("  %-24s %s" % (name, "通过" if ok else "**未通过**"))
        print("  %s" % acc["details"])
        print("  shakedown 标签：%s" % result["shakedown_tags"])
        if not acc["passed"]:
            print("")
            print("⚠ 硬约束 6：判据未满足即停下报告，不调参。")
    return 0 if result["acceptance"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
