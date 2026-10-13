"""D-57 S2：刚体圆弧线夹（罚函数）压厚块，与 Hertz 解析解对拍（线弹性，无塑性）。

判据（configs/verify_obstacle_hertz.yaml）：
  1. 每个线夹下压量 δ 的 Newton 都收敛（硬约束 6）；
  2. 每个 δ 下，接触半宽 a 与峰值压力 p0 相对 Hertz（以精确合力 P 反算）偏差 < hertz_rel；
  3. 罚刚度 κ 从 10 改到 1，合力相对变化 < kappa_sens_rel（罚函数偏差可忽略）。

用法：python scripts/verify_obstacle_hertz.py [configs/verify_obstacle_hertz.yaml]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from mpi4py import MPI

from pfwire.loads import hertz_half_width, hertz_peak_pressure
from pfwire.material import contact_effective_modulus
from pfwire.obstacle import build_block_mesh
from pfwire.obstacle_contact import ObstacleContact
from pfwire.verify_contact_common import fnum, load_config

REPO = pathlib.Path(__file__).resolve().parents[1]


def a_sub(x, p, floor: float = 0.01) -> float:
    """接触半宽：子单元线性插值找 p = floor·p_max 的穿越点（比取节点最大 |x| 少一层采样误差）。

    与 `scripts/_thick.py` 诊断中用的一致，D-49 的口径。
    """
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D-57 S2：罚函数障碍 vs Hertz")
    ap.add_argument("config", nargs="?", default="configs/verify_obstacle_hertz.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    comm = MPI.COMM_WORLD
    outdir = REPO / cfg["output"]["dir"]
    g, mat, m, pen = cfg["geometry"], cfg["material"], cfg["mesh"], cfg["penalty"]
    E, nu, R = fnum(mat, "E"), fnum(mat, "nu"), fnum(g, "R_clamp")
    h = fnum(m, "h_fine")
    solver = {k: str(v) for k, v in cfg["solver"].items()}
    deltas = [float(d) for d in cfg["loading"]["deltas"]]
    tol = cfg["tolerance"]

    def say(msg):
        if comm.rank == 0:
            print(msg, flush=True)

    say("D-57 S2：罚函数障碍 vs Hertz（线弹性，E = %.4g GPa，ν = %.2f，R = %.0f mm）"
        % (E / 1e9, nu, 1e3 * R))
    mesh, _ct, ft = build_block_mesh(
        outdir, width=fnum(g, "width"), depth=fnum(g, "depth"),
        h_fine=h, h_coarse=fnum(m, "h_coarse"), fine_halfwidth=fnum(m, "fine_halfwidth"),
        fine_depth=fnum(m, "fine_depth"), transition=fnum(m, "transition"),
        order=int(m["order"]), quads=bool(m["quads"]), comm=comm)
    say("  网格：%d 单元" % mesh.topology.index_map(mesh.topology.dim).size_global)

    e_star = contact_effective_modulus(E, nu, 1e30, nu)      # 刚性线夹一侧 -> E/(1−ν²)
    model = ObstacleContact(mesh, ft, E=E, nu=nu, R_clamp=R, kappa=fnum(pen, "kappa"),
                            h_contact=h, solver=solver)

    say("")
    say("  %10s %12s %10s %10s %9s %9s %s"
        % ("δ (µm)", "P (N/m)", "a_FE (mm)", "a_Hz (mm)", "a 偏差", "p0 偏差", "判定"))
    rows, ok = [], True
    for d in deltas:
        try:
            n_it = model.step(d)
        except RuntimeError as exc:
            say("  ⚠ %s" % exc)
            ok = False
            break
        P = model.reaction()
        xs, ps = model.pressure_profile()
        a_fe = a_sub(xs, ps)
        p0_fe = float(ps.max())
        a_hz = hertz_half_width(P=P, R=R, E_star=e_star)
        p0_hz = hertz_peak_pressure(P=P, a=a_hz)
        ea, ep = abs(a_fe - a_hz) / a_hz, abs(p0_fe - p0_hz) / p0_hz
        passed = ea < fnum(tol, "hertz_rel") and ep < fnum(tol, "hertz_rel")
        ok = ok and passed
        rows.append({"delta": d, "P": P, "a_fe": a_fe, "a_hz": a_hz, "p0_fe": p0_fe,
                     "p0_hz": p0_hz, "rel_a": ea, "rel_p0": ep, "newton_it": n_it})
        say("  %10.2f %12.5g %10.4f %10.4f %8.2f%% %8.2f%% %s"
            % (1e6 * d, P, 1e3 * a_fe, 1e3 * a_hz, 100 * ea, 100 * ep,
               "通过" if passed else "**未通过**"))
        model.commit()

    # 罚刚度敏感性：在最大 δ 处换 κ 重算合力
    sens_ok = False
    if ok and rows:
        k2 = fnum(pen, "kappa_sens")
        model2 = ObstacleContact(mesh, ft, E=E, nu=nu, R_clamp=R, kappa=k2,
                                 h_contact=h, solver=solver)
        model2.step(deltas[-1])
        P2 = model2.reaction()
        dP = abs(P2 - rows[-1]["P"]) / rows[-1]["P"]
        sens_ok = dP < fnum(tol, "kappa_sens_rel")
        say("")
        say("  κ 敏感性：κ = %g → %g，合力相对变化 %.3f%%（阈值 %.1f%%）"
            % (fnum(pen, "kappa"), k2, 100 * dP, 100 * fnum(tol, "kappa_sens_rel")))
    else:
        dP = float("nan")

    checks = [("1 全部 δ 的 Newton 收敛", ok and len(rows) == len(deltas)),
              ("2 a、p0 与 Hertz 偏差 < %.0f%%" % (100 * fnum(tol, "hertz_rel")), ok),
              ("3 κ 敏感性（合力 < %.1f%%）" % (100 * fnum(tol, "kappa_sens_rel")), sens_ok)]
    say("")
    all_ok = True
    for name, passed in checks:
        all_ok = all_ok and passed
        say("  %-40s %s" % (name, "通过" if passed else "**未通过**"))

    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "result.json").write_text(json.dumps(
            {"passed": all_ok, "E_star": e_star, "kappa": fnum(pen, "kappa"),
             "kappa_sens_rel_dP": dP, "steps": rows,
             "checks": [{"name": n, "passed": bool(p)} for n, p in checks]},
            ensure_ascii=False, indent=2), encoding="utf-8")
    say("")
    say("S2 %s" % ("通过" if all_ok else "未通过"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
