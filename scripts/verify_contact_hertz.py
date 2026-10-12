"""Phase 4.3：FE 摩擦接触解与 Hertz 解析解对拍（PLAN §4 Phase 4.3）。

装配与求解走 `pfwire.contact`（对 dolfinx_contact 的薄封装），
**接触压力的提取与对拍是本项目自己做的** —— 上游的 `ContactWriter`
在 dolfinx 0.11 / UFL 2026.1 下已失效（DECISIONS.md D-48）。

对拍口径（PLAN Phase 4.3）：接触半宽 a、峰值压力 p0 相对解析解偏差 < 3%；
另加一条更严的**合力校核**：接触面法向合力与施加的 P 之差 < 2%。
合力用**精确面积分**而不是对节点值做梯形 —— 后者会把接触区外被 CG1
抹出来的数值残尾算进去，实测差 3.10% vs 0.0014%（D-49）。

用法：python scripts/verify_contact_hertz.py [configs/verify_hertz_contact.yaml]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from dolfinx import default_scalar_type
from dolfinx.fem import Constant, dirichletbc, locate_dofs_topological
from dolfinx.geometry import bb_tree, compute_closest_entity
from mpi4py import MPI

from pfwire.contact import ContactModel, build_halfdisk_plane, fix_block_bottom
from pfwire.loads import hertz_half_width, hertz_peak_pressure
from pfwire.material import contact_effective_modulus
from pfwire.verify_contact_common import fnum, load_config, report

REPO = pathlib.Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 4.3 Hertz 接触对拍")
    ap.add_argument("config", nargs="?", default="configs/verify_hertz_contact.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    comm = MPI.COMM_WORLD
    outdir = REPO / cfg["output"]["dir"]

    def say(msg):
        if comm.rank == 0:
            print(msg, flush=True)

    g, mat, ld, sol, m = (cfg["geometry"], cfg["material"], cfg["loading"],
                          cfg["solver"], cfg["mesh"])
    R, H, P, steps = fnum(g, "R"), fnum(g, "H"), fnum(ld, "P"), int(ld["steps"])
    E, nu = fnum(mat, "E"), fnum(mat, "nu")

    say("Phase 4.3：FE 摩擦接触 vs Hertz 解析解")
    mesh, ct, ft = build_halfdisk_plane(
        outdir, R=R, H=H, L=fnum(g, "L"), gap=fnum(g, "gap"),
        res=fnum(m, "res"), order=int(m["order"]), quads=bool(m["quads"]), comm=comm)
    say("  网格：%d 单元" % mesh.topology.index_map(mesh.topology.dim).size_global)

    model = ContactModel(mesh, ct, ft, E=E, nu=nu, order=int(m["order"]))

    # 块体底面全固定；另钉圆柱内两个节点的 x 位移，消掉水平刚体位移。
    # 无摩擦算例里这样做是安全的：切向无载荷，钉点不承力。
    tree = bb_tree(mesh, 0)
    nodes = np.hstack([compute_closest_entity(tree, tree, mesh,
                                              np.array([0.0, y, 0.0], dtype=np.float64))
                       for y in (-R / 2.5, -R / 5.0)])
    bc_x = dirichletbc(Constant(mesh, default_scalar_type(0.0)),
                       locate_dofs_topological(model.V.sub(0), 0, nodes), model.V.sub(0))
    traction = Constant(mesh, default_scalar_type((0.0, 0.0)))
    ksp_tol = fnum(sol, "ksp_tol")
    model.setup(
        bcs=[bc_x, fix_block_bottom(model)], traction=traction, friction=None,
        gamma_factor=fnum(sol, "nitsche_gamma_factor"), theta=fnum(sol, "nitsche_theta"),
        newton_options={"relaxation_parameter": 1.0,
                        "atol": fnum(sol, "newton_tol"), "rtol": fnum(sol, "newton_tol"),
                        "convergence_criterion": "residual",
                        "max_it": int(sol["max_it"]), "error_on_nonconvergence": True},
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "ksp_rtol": ksp_tol, "ksp_atol": ksp_tol})

    # 初值：让圆柱先落下一点，否则第一步接触搜索找不到对面
    model.du.interpolate(
        lambda x: np.vstack([np.zeros(x.shape[1]),
                             np.full(x.shape[1], -H / 100.0 - fnum(g, "gap"))]),
        ct.find(model.tags.body_cells[0]))

    for i in range(steps):
        traction.value[-1] = -P * (i + 1) / (steps * 2.0 * R)
        n_it = model.step("载荷步 %d/%d" % (i + 1, steps))
        say("  载荷步 %d/%d：Newton %d 次迭代收敛" % (i + 1, steps, n_it))
        if i < steps - 1:
            model.prepare_next()

    xs, ps, _ = model.surface_tractions()
    load_exact = model.resultant(component=1)

    e_star = contact_effective_modulus(E, nu, E, nu)
    a_ref = hertz_half_width(P=P, R=R, E_star=e_star)
    p0_ref = hertz_peak_pressure(P=P, a=a_ref)
    p0_fe = float(ps.max())
    # 接触半宽用阈值而不是「首个零」：CG1 插值会在边缘留下小的数值残余
    inside = ps > 0.01 * p0_fe
    a_fe = float(np.max(np.abs(xs[inside]))) if inside.any() else 0.0
    load_trapz = float(np.trapezoid(np.clip(ps, 0.0, None), xs))

    tol = cfg["tolerance"]
    checks = [("接触半宽 a", a_fe, a_ref, fnum(tol, "a_rel")),
              ("峰值压力 p0", p0_fe, p0_ref, fnum(tol, "p0_rel")),
              ("合力（面积分）", load_exact, P, fnum(tol, "load_rel"))]
    say("")
    say("  E* = %.6g，a/R = %.4f（Hertz 要求接触区远小于曲率半径）" % (e_star, a_ref / R))
    say("  接触区内取到 %d 个采样点" % int(inside.sum()))
    say("  （对照：对 CG1 节点值做梯形积分得 %.6g，偏差 %.2f%% —— 含接触区外的"
        "数值残尾，仅供诊断，见 D-49）" % (load_trapz, 100 * abs(load_trapz - P) / P))
    ok, rows = report(say, checks)

    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "result.json").write_text(json.dumps(
            {"passed": ok, "E_star": e_star, "a_over_R": a_ref / R,
             "n_points_in_contact": int(inside.sum()),
             "P_trapz": load_trapz, "checks": rows},
            ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez_compressed(outdir / "pressure.npz", x=xs, p=ps)
    say("")
    say("Phase 4.3 %s" % ("通过" if ok else "未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
