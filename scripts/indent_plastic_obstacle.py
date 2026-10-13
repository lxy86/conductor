"""D-57 S3：Phase 6.1 v2 —— 弹塑性单丝被刚体圆弧线夹（罚函数）压，无相场、无摩擦。

替代 scripts/indent_plastic_clamp.py（dolfinx_contact 版，D-56 已证其接触核与塑性不一致）。
接触本身已由 scripts/verify_obstacle_hertz.py（S2，厚块 vs Hertz）独立对拍通过。

判据（configs/indent_plastic_obstacle.yaml）：
  1. 每个线夹下压量 δ 的 Newton 都收敛（硬约束 6）；
  2. 工作点（P 最接近 W'）的 p0/σ_y 落在允许带内（区间检查，不是饱和证据；饱和指数 m 只报告，D-57）；
  3. 工作点 P 与 W' 相对差 < w_rel；
  4. 塑性接触半宽 a_pl 远离域边界（< apl_edge_frac × Lw/2）；
  5. 罚刚度 κ 从 10 改到 1，末步合力相对变化 < kappa_sens_rel。

不做「弹性段 vs Hertz」检查：3.75 mm 单丝不满足半空间前提（D-51/D-56）。

用法：python scripts/indent_plastic_obstacle.py [configs/indent_plastic_obstacle.yaml]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from mpi4py import MPI

from pfwire.indent import saturation_exponent
from pfwire.obstacle import build_block_mesh
from pfwire.obstacle_contact import ObstacleContact
from pfwire.plasticity import J2Params, PlasticLaw
from pfwire.verify_contact_common import fnum, load_config

REPO = pathlib.Path(__file__).resolve().parents[1]

#: 判「塑性区」用的等效塑性应变阈值（相对 σ_y/E，与 6.1 旧脚本一致）。
P_EQ_FLOOR = 1.0e-3


def a_sub(x, p, floor: float = 0.01) -> float:
    """接触半宽：子单元线性插值（D-49 口径，与 verify_obstacle_hertz.py 一致）。"""
    thr = floor * p.max()
    inside = p > thr
    if not inside.any():
        return 0.0
    i = int(np.argmax(inside))
    a_left = x[0]
    if i > 0:
        t = (thr - p[i - 1]) / (p[i] - p[i - 1])
        a_left = x[i - 1] + t * (x[i] - x[i - 1])
    j = len(p) - 1 - int(np.argmax(inside[::-1]))
    a_right = x[-1]
    if j < len(p) - 1:
        t = (thr - p[j]) / (p[j + 1] - p[j])
        a_right = x[j] + t * (x[j + 1] - x[j])
    return float(0.5 * (a_right - a_left))


def plastic_zone(law: PlasticLaw, comm, thr: float):
    """塑性区的横向半宽 a_pl 与深度（由 DG-2 的等效塑性应变历史量取得，全局合并）。"""
    coords = law.V_p.tabulate_dof_coordinates()
    n_own = law.V_p.dofmap.index_map.size_local
    p = law.p_n.x.array[:n_own]
    pl = p > thr
    a_pl = float(np.abs(coords[:n_own, 0][pl]).max()) if pl.any() else 0.0
    depth = float(-coords[:n_own, 1][pl].min()) if pl.any() else 0.0
    return comm.allreduce(a_pl, op=MPI.MAX), comm.allreduce(depth, op=MPI.MAX)


def run(cfg, comm, say, *, kappa=None):
    """跑完整的压下序列。返回 (rows, model, law, ok)。"""
    g, mat, m, sol, pen = (cfg["geometry"], cfg["material"], cfg["mesh"],
                           cfg["solver"], cfg["penalty"])
    E, nu, sy = fnum(mat, "E"), fnum(mat, "nu"), fnum(mat, "sigma_y")
    prm = J2Params(E=E, nu=nu, sigma_y0=sy, H=fnum(mat, "H"))
    outdir = REPO / cfg["output"]["dir"]
    h = fnum(m, "h_fine")
    mesh, _ct, ft = build_block_mesh(
        outdir, width=fnum(g, "width"), depth=fnum(g, "depth"),
        h_fine=h, h_coarse=fnum(m, "h_coarse"), fine_halfwidth=fnum(m, "fine_halfwidth"),
        fine_depth=fnum(m, "fine_depth"), transition=fnum(m, "transition"),
        order=int(m["order"]), quads=bool(m["quads"]), comm=comm)
    law = PlasticLaw(mesh, prm)
    solver = {k: str(v) for k, v in sol.items()}
    k_use = fnum(pen, "kappa") if kappa is None else float(kappa)
    model = ObstacleContact(mesh, ft, E=E, nu=nu, R_clamp=fnum(g, "R_clamp"), kappa=k_use,
                            h_contact=h, solver=solver, plastic=law)
    thr = P_EQ_FLOOR * sy / E
    rows, ok = [], True
    for d in (float(x) for x in cfg["loading"]["deltas"]):
        try:
            n_it = model.step(d)
        except RuntimeError as exc:                     # 硬约束 6：不收敛即停
            say("  ⚠ δ = %.2f µm 未收敛：%s" % (1e6 * d, exc))
            ok = False
            break
        P = model.reaction()
        xs, ps = model.pressure_profile()
        a_pl, depth = plastic_zone(law, comm, thr)
        rows.append({"delta": d, "P": P, "p0": float(ps.max()), "a": a_fe_from(xs, ps),
                     "p_eq_max": float(comm.allreduce(float(law.p_n.x.array.max()), op=MPI.MAX)),
                     "a_pl": a_pl, "depth": depth, "newton_it": n_it, "kappa": k_use})
        say("  δ = %5.2f µm  P = %11.5g N/m  p0 = %7.1f MPa (%.3f σy)  a = %.4f mm  "
            "a_pl = %.4f mm  塑性深 = %.3f mm  p_eq,max = %.4f  Newton = %d"
            % (1e6 * d, P, rows[-1]["p0"] / 1e6, rows[-1]["p0"] / sy, 1e3 * rows[-1]["a"],
               1e3 * a_pl, 1e3 * depth, rows[-1]["p_eq_max"], n_it))
        law.commit(model.u)
    return rows, model, law, ok


def a_fe_from(xs, ps) -> float:
    return a_sub(xs, ps) if xs.size else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D-57 S3：弹塑性单丝 + 刚体圆弧线夹（罚函数）")
    ap.add_argument("config", nargs="?", default="configs/indent_plastic_obstacle.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    comm = MPI.COMM_WORLD
    outdir = REPO / cfg["output"]["dir"]
    sy = fnum(cfg["material"], "sigma_y")
    Lw = fnum(cfg["geometry"], "width")
    tol = cfg["tolerance"]

    def say(msg):
        if comm.rank == 0:
            print(msg, flush=True)

    say("D-57 S3：弹塑性单丝（Lw = %.0f mm，D = %.2f mm，σ_y = %.0f MPa）+ 刚体圆弧线夹（罚函数）"
        % (1e3 * Lw, 1e3 * fnum(cfg["geometry"], "depth"), sy / 1e6))
    rows, model, law, ok = run(cfg, comm, say)
    if not rows:
        say("没有任何收敛步，停止")
        return 1

    w_target = fnum(cfg["loading"], "w_target")
    op = min(rows, key=lambda r: abs(r["P"] - w_target))
    # 饱和指数 m = d ln p0 / d ln P（末 4 点）。Hertz 线接触 m = 0.5，完全饱和 m -> 0。
    # 只报告，不设通过阈值：阈值须由 PLAN 定，不在这里自拟（硬约束 2）。
    sat = saturation_exponent([r["P"] for r in rows[2:]], [r["p0"] for r in rows[2:]],
                              window=min(4, len(rows) - 2))
    say("  饱和指数 m = d ln p0 / d ln P（末 %d 点）= %.3f（Hertz 0.5，完全饱和 → 0）"
        % (min(4, len(rows) - 2), sat.exponent))
    band = tol["p0_band_over_sy"]
    lo, hi = float(band[0]), float(band[1])
    checks = [
        ("1 全部 δ 的 Newton 收敛", ok and len(rows) == len(cfg["loading"]["deltas"])),
        ("2 工作点 p0/σy = %.3f ∈ [%.2f, %.2f]（区间检查；饱和指数 m 仅报告，见 D-57）"
         % (op["p0"] / sy, lo, hi), lo <= op["p0"] / sy <= hi),
        ("3 工作点 P 与 W' = %.3g 相对差 < %.0f%%（P = %.3g）"
         % (w_target, 100 * fnum(tol, "w_rel"), op["P"]),
         abs(op["P"] - w_target) / w_target < fnum(tol, "w_rel")),
        ("4 a_pl = %.3f mm < %.2f × Lw/2（远离域边界）"
         % (1e3 * op["a_pl"], fnum(tol, "apl_edge_frac")),
         op["a_pl"] < fnum(tol, "apl_edge_frac") * Lw / 2),
    ]

    say("")
    say("  κ 敏感性：末步在 κ = %g 下重算 ..." % fnum(cfg["penalty"], "kappa_sens"))
    rows2, _m2, _l2, ok2 = run(cfg, comm, lambda *_: None, kappa=fnum(cfg["penalty"], "kappa_sens"))
    if ok2 and rows2:
        dP = abs(rows2[-1]["P"] - rows[-1]["P"]) / rows[-1]["P"]
        checks.append(("5 κ 敏感性：末步合力相对变化 %.2f%% < %.0f%%"
                       % (100 * dP, 100 * fnum(tol, "kappa_sens_rel")),
                       dP < fnum(tol, "kappa_sens_rel")))
    else:
        checks.append(("5 κ 敏感性（第二个 κ 也须收敛）", False))

    say("")
    all_ok = True
    for name, passed in checks:
        all_ok = all_ok and passed
        say("  %-68s %s" % (name, "通过" if passed else "**未通过**"))

    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "result.json").write_text(json.dumps(
            {"passed": all_ok, "sigma_y": sy, "w_target": w_target,
             "operating_point": op, "sat_exponent": sat.exponent, "steps": rows,
             "checks": [{"name": n, "passed": bool(p)} for n, p in checks]},
            ensure_ascii=False, indent=2), encoding="utf-8")
    say("")
    say("Phase 6.1 v2 %s" % ("通过" if all_ok else "未通过"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
