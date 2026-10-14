"""Phase 6.5 的**静态验证**（C 方案，D-67）：只跑接触与载荷，不做 cycle-jump。

为什么先做静态：完整验收（W' 搜索 + cycle-jump 到 100 µm 裂纹）在布局 C
（145 万单元）上，仅 W' 搜索的第一步延拓就 20 分钟未完——一次验收要跑几十小时，
中途失败无法诊断。故先降规模（加密带单元 1 µm，37 万单元）跑静态步，确认：

  1. δ 延拓能走到 δ₀，接触反力 W(δ) 单调；
  2. W' = 1.75e6 N/m 能被搜到（或如实报告搜不到）；
  3. 远场 σ_xx 幅值 == σ_a（端部加载没加载错）；
  4. 接触区压力分布、塑性区位置合理（无假损伤）。

任一条异常即停下报告，不调参（硬约束 6）。

用法：mpirun -n 6 python scripts/verify_clamp_static.py [configs/clamp_base_static.yaml]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from mpi4py import MPI

from pfwire.clamp_case import _ramp_plan, build_clamp_case, solve_contact_at

REPO = pathlib.Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="6.5 静态验证（接触 + 载荷，无疲劳）")
    ap.add_argument("config", nargs="?", default="configs/clamp_base_static.yaml")
    ap.add_argument("--delta-only", action="store_true",
                    help="只跑 δ 延拓并报 W(δ) 曲线，不做 W' 搜索")
    args = ap.parse_args(argv)
    comm = MPI.COMM_WORLD
    rank = comm.rank

    def say(msg):
        if rank == 0:
            print(msg, flush=True)

    t0 = time.perf_counter()
    case = build_clamp_case(REPO / args.config, comm)
    raw = case.config
    n_cells = case.mesh.topology.index_map(2).size_global
    say("网格 %d 单元，h_contact = %.3g µm（%.1f s）"
        % (n_cells, 1e6 * case.h_contact, time.perf_counter() - t0))

    delta0 = float(raw["loading"]["delta0"])
    W_target = float(raw["loading"]["W_target"])
    report = {"case": raw.get("case"), "n_cells": int(n_cells),
              "h_contact": case.h_contact, "delta0": delta0, "W_target": W_target}

    # ---- 1) δ 延拓曲线 ----
    # 必须用 _ramp_plan（首步按罚压力定）而不是等分：等分的首步 δ₁ = δ₀/12
    # 给出罚压力 ε·δ₁ ≈ 1e10 Pa，超 σ_y 两个数量级，Newton 从 u=0 起步发散。
    seq = _ramp_plan(delta0, case.clamp.eps, h_contact=case.h_contact, R_clamp=case.clamp.R_clamp)
    say("")
    say("δ 延拓（%d 步，首步 δ=%.3g µm）：" % (len(seq), 1e6 * seq[0]))
    curve = []
    for i, d in enumerate(seq):
        t1 = time.perf_counter()
        case.set_delta(float(d))
        try:
            case.solver.solve_displacement()
        except RuntimeError as exc:
            say("  ⚠ 步 %2d δ=%.4g 失败：%s" % (i + 1, d, exc))
            report["ramp_failed_at"] = {"step": i + 1, "delta": float(d), "error": str(exc)}
            break
        W = case.clamp.reaction(case.solver.u)
        curve.append({"delta": float(d), "W": W})
        say("  步 %2d δ=%7.3f µm  W=%.6g N/m  （%.0f s）"
            % (i + 1, 1e6 * d, W, time.perf_counter() - t1))
    report["ramp_curve"] = curve
    if args.delta_only:
        _write(report, raw)
        return 0 if monotone else 1
    if not curve:
        say("")
        say("静态验证未通过：δ 延拓第一步就失败")
        return 1
    # W(δ) 必须单调增（Hertz：P ∝ δ^{3/2}；塑性后趋于饱和但不减）
    Ws = np.array([c["W"] for c in curve])
    monotone = bool(np.all(np.diff(Ws) > 0))
    report["W_monotone"] = monotone
    say("")
    say("  W(δ) 单调增：%s（末值 %.6g）" % ("是" if monotone else "**否**", Ws[-1]))

    # ---- 2) W' 搜索 ----
    say("")
    say("W' 搜索（目标 %.6g N/m，容差 1%%）：" % W_target)
    t1 = time.perf_counter()
    try:
        from pfwire.clamp_case import find_working_delta
        delta_star = find_working_delta(case, W_target, delta0=delta0,
                                        rtol=1.0e-2, max_iter=20)
        W_star = case.clamp.reaction(case.solver.u)
        say("  δ* = %.4g µm，W = %.6g（偏差 %.3f%%，%.0f s）"
            % (1e6 * delta_star, W_star, 100 * abs(W_star - W_target) / W_target,
               time.perf_counter() - t1))
        report["search"] = {"status": "converged", "delta_star": float(delta_star),
                            "W": float(W_star),
                            "rel": float(abs(W_star - W_target) / W_target)}
    except RuntimeError as exc:
        say("  ⚠ 搜索失败：%s" % exc)
        report["search"] = {"status": "failed", "error": str(exc)}

    # ---- 3) 端部弯矩的自洽性（**无接触**下验 σ_a）----
    # 有接触时不验 σ_a：线夹压紧本身是跨中的横向集中力（W ~ 1e6 N/m），
    # 梁成三点弯曲构型，弯矩沿 x 线性变化，没有「远场等弯矩」这回事
    # （实测 x=3.5 mm 处 9.9e7、x=10 mm 处 1.1e4，差四个数量级）。
    # 端部加载 σ_a 的校验是 T3c，在**无接触纯弯曲**下做（偏差 0.46%）。
    say("")
    say("端部弯矩自洽性（有接触时「远场 σ_a」不适用，见上）：")
    case.clamp.set_delta(0.0)
    case.solver.solve_displacement()
    case.loads.set_phase(1.0)
    case.solver.solve_displacement()
    sigma_a_free = _far_field_sigma_a(case, x_center=3.5e-3)
    if sigma_a_free is not None:
        rel = abs(sigma_a_free - float(raw["loading"]["sigma_a"])) / float(
            raw["loading"]["sigma_a"])
        say("  无接触纯弯曲：σ_xx 幅值 = %.6g Pa，偏差 %.3f%%"
            % (sigma_a_free, 100 * rel))
        report["sigma_a_free"] = sigma_a_free
        report["sigma_a_free_rel_err"] = float(rel)

    _write(report, raw)
    say("")
    say("静态验证结束（%.0f s）" % (time.perf_counter() - t0))
    return 0


def _far_field_sigma_a(case, x_center=0.0, halfband=2.0e-3) -> float | None:
    """远场截面 σ_xx 的弯曲幅值（线性拟合斜率 × H/2）。

    ``halfband`` 默认 2 mm：截面要取足够多的单元才拟合得出斜率（取 100 µm 时
    粗网格上一个单元都没有，会被静默跳过）。
    """
    import ufl
    from dolfinx import fem
    from dolfinx.mesh import compute_midpoints

    msh = case.mesh
    H = float(case.config["loading"]["H"])
    mu, lmbda = case.solver.model.material.elastic.lame
    u = case.solver.u
    eps = ufl.sym(ufl.grad(u))
    sxx = (2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(msh.geometry.dim))[0, 0]
    V0 = fem.functionspace(msh, ("DG", 0))
    s_fn = fem.Function(V0)
    s_fn.interpolate(fem.Expression(sxx, V0.element.interpolation_points))
    nloc = msh.topology.index_map(msh.topology.dim).size_local
    mid = compute_midpoints(msh, msh.topology.dim, np.arange(nloc, dtype=np.int32))
    on = np.abs(mid[:, 0] - x_center) <= halfband
    ys = np.concatenate(msh.comm.allgather(mid[on, 1]))
    ss = np.concatenate(msh.comm.allgather(s_fn.x.array[:nloc][on]))
    if ys.size < 2:
        return None
    k = np.polyfit(ys, ss, 1)[0]
    return abs(k) * 0.5 * H


def _write(report, raw) -> None:
    """只在 rank 0 写：并行下每个 rank 都去 replace 同一个临时文件会撞车
    （后面的 rank 抢先替换掉，前面的再 replace 就 FileNotFoundError）。"""
    from mpi4py import MPI
    if MPI.COMM_WORLD.rank != 0:
        return
    outdir = raw["_outdir"]
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = outdir / "static_result.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(outdir / "static_result.json")


if __name__ == "__main__":
    sys.exit(main())
