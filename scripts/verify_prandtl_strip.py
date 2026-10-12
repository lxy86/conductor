"""Phase 6.1 的独立验证：刚性平冲头压理想塑性块体 vs Prandtl 极限载荷（PLAN §6.1）。

目标：块体（J2，H = 0）顶面 |x| <= w 被刚性平冲头按下，平均接触压力趋于
    p_lim = (2 + π) k,  k = σ_y/√3  ->  p_lim = 2.9708 σ_y
出处：L. Prandtl, "Über die Härte plastischer Körper", Nachr. Ges. Wiss. Göttingen (1920)；
      亦见 K. L. Johnson, *Contact Mechanics*, 1985, ch. 6（⚠️ 具体节号/式号待查书核对）。

**口径**（D-54）：Prandtl 的前提是「刚性平底冲头 + 半无限体」。这里不建冲头、不用接触，
而是把冲头的刚性直接写成边界条件：顶面 |x| <= w 上 u_y = -δ、u_x = 0（刚性平底、界面不滑），
其余顶面自由。力学问题与 Prandtl 同一，且避开了平-平接触在 dolfinx_contact 里的投影退化。
「接触 + 塑性」的耦合另由 6.1 本体的自洽量（力平衡、穿透、γ 敏感性）把关。

用法：python scripts/verify_prandtl_strip.py [configs/verify_prandtl_strip.yaml]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

from pfwire.plasticity import J2Params, PlasticLaw
from pfwire.verify_contact_common import fnum, load_config

REPO = pathlib.Path(__file__).resolve().parents[1]
#: Prandtl 平面应变极限平均压力系数：(2 + π)/√3。
PRANDTL_FACTOR = (2.0 + math.pi) / math.sqrt(3.0)


def run_level(cfg, h_over_w, comm, say):
    g, mat, ld, sol = cfg["geometry"], cfg["material"], cfg["loading"], cfg["solver"]
    w, L, D = fnum(g, "half_width"), fnum(g, "block_half_width"), fnum(g, "block_depth")
    E, nu, sy, H = fnum(mat, "E"), fnum(mat, "nu"), fnum(mat, "sigma_y"), fnum(mat, "H")
    prm = J2Params(E=E, nu=nu, sigma_y0=sy, H=H)
    # x = ±w 必须落在网格线上：半宽 L 用 w/h_over_w 的整数倍划分
    n_x = int(round(2.0 * L / (h_over_w * w)))
    n_y = int(round(D / (h_over_w * w)))
    mesh = dmesh.create_rectangle(comm, [np.array([-L, -D]), np.array([L, 0.0])],
                                  [n_x, n_y], dmesh.CellType.quadrilateral)
    law = PlasticLaw(mesh, prm)
    V = fem.functionspace(mesh, ("Lagrange", int(ld["degree"]), (2,)))
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    F = ufl.inner(law.sigma(u), law.strain(v)) * law.dx

    fdim = mesh.topology.dim - 1
    fixed = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], -D, atol=1e-12 * D))
    strip = dmesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[1], 0.0, atol=1e-12 * D) & (np.abs(x[0]) <= w + 1e-12 * w))
    g_uy = fem.Constant(mesh, np.float64(0.0))
    zero = fem.Constant(mesh, np.float64(0.0))
    bcs = [fem.dirichletbc(zero, fem.locate_dofs_topological(V.sub(1), fdim, fixed), V.sub(1)),
           fem.dirichletbc(g_uy, fem.locate_dofs_topological(V.sub(1), fdim, strip), V.sub(1)),
           fem.dirichletbc(zero, fem.locate_dofs_topological(V.sub(0), fdim, strip), V.sub(0))]
    n_strip = comm.allreduce(len(fem.locate_dofs_topological(V.sub(1), fdim, strip)), op=MPI.SUM)
    if n_strip == 0:
        raise RuntimeError("顶面条带上没有自由度 —— 检查 x = ±w 是否落在网格线上（n_x = %d）" % n_x)

    prob = NonlinearProblem(
        F, u, bcs=bcs, petsc_options_prefix="prandtl_",
        petsc_options={"snes_type": "newtonls", "snes_linesearch_type": "bt",
                       "snes_atol": fnum(sol, "newton_atol"), "snes_rtol": fnum(sol, "newton_rtol"),
                       "snes_max_it": int(sol["max_it"]),
                       "ksp_type": "preonly", "pc_type": "lu",
                       "ksp_rtol": fnum(sol, "ksp_tol"), "ksp_atol": fnum(sol, "ksp_tol")})

    # 顶面法向合力（精确面积分）与塑性区深度
    top_all = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 0.0, atol=1e-12 * D))
    ft = dmesh.meshtags(mesh, fdim, np.sort(top_all), np.full(len(top_all), 1, dtype=np.int32))
    ds_top = ufl.Measure("ds", domain=mesh, subdomain_data=ft, metadata={"quadrature_degree": 4})
    P_form = fem.form(-law.sigma(u)[1, 1] * ds_top(1))
    Wp = law.V_p
    coords = Wp.tabulate_dof_coordinates()
    n_own = Wp.dofmap.index_map.size_local

    def load_and_depth():
        P = comm.allreduce(fem.assemble_scalar(P_form), op=MPI.SUM)
        p = law.p_n.x.array[:n_own]
        y = coords[:n_own, 1]
        # 顶面 y=0、底面 y=-D：深度 = 0 - y_min（从顶面往下量）。
        # 给两个阈值："有任何塑性的最深点" 与 "实质塑性（p > 0.01 σ_y/E）的最深点"——
        # 前者会被塑性区外缘的弥散边界拖得很深，判"受限"应当用后者。
        out = []
        for thr in (1e-4, 1e-2):
            pl = p > thr * sy / E
            out.append(comm.allreduce(float(-y[pl].min()) if pl.any() else 0.0, op=MPI.MAX))
        return float(P), out[0], out[1]

    steps = int(ld["steps"])
    rows = []
    for i in range(steps):
        g_uy.value = fnum(ld, "u_y") * (i + 1) / steps
        prob.solve()
        if prob.solver.getConvergedReason() <= 0:
            raise RuntimeError("Newton 未收敛：δ = %.4g（h/w = %g），停止，不使用未收敛解"
                               % (g_uy.value, h_over_w))
        law.commit(u)
        P, depth, depth_sub = load_and_depth()
        rows.append({"step": i + 1, "u_y": float(g_uy.value), "P": P, "p_mean": P / (2.0 * w),
                     "p_mean_over_sy": P / (2.0 * w) / sy, "ratio": P / (2.0 * w) / sy / PRANDTL_FACTOR,
                     "depth": depth, "depth_sub": depth_sub, "depth_over_D": depth / D,
                     "depth_sub_over_D": depth_sub / D,
                     "n_it": int(prob.solver.getIterationNumber())})
    return dict(rows=rows, n_x=n_x, n_y=n_y, n_cells=int(mesh.topology.index_map(2).size_global),
                w=w, D=D, sy=sy)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 6.1 独立验证：Prandtl 平冲头极限载荷（条带加载）")
    ap.add_argument("config", nargs="?", default="configs/verify_prandtl_strip.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    comm = MPI.COMM_WORLD
    outdir = REPO / cfg["output"]["dir"]

    def say(msg):
        if comm.rank == 0:
            print(msg, flush=True)

    sy = fnum(cfg["material"], "sigma_y")
    say("Phase 6.1 独立验证：刚性平冲头（条带位移加载，无接触）压理想塑性块体")
    say("  w = %.4g m，σ_y = %.4g Pa，Prandtl 极限平均压力 = %.4f σ_y = %.4g Pa"
        % (fnum(cfg["geometry"], "half_width"), sy, PRANDTL_FACTOR, PRANDTL_FACTOR * sy))

    results = []
    for hw in cfg["tolerance"]["refine"]:
        res = run_level(cfg, float(hw), comm, say)
        results.append(res)
        rows = res["rows"]
        say("")
        say("  h/w = %.3f：%d x %d = %d 单元；Newton 每步迭代数 %s"
            % (hw, res["n_x"], res["n_y"], res["n_cells"], [r["n_it"] for r in rows]))
        say("  %-4s %11s %11s %11s %11s %10s %10s"
            % ("步", "δ/w", "P", "p_mean/σy", "/Prandtl", "塑性深/w", "实质塑性深/w"))
        for r in rows[:: max(1, len(rows) // 8)] + [rows[-1]]:
            say("  %-4d %11.3f %11.4g %11.4f %11.4f %10.2f %10.2f"
                % (r["step"], -r["u_y"] / res["w"], r["P"], r["p_mean_over_sy"], r["ratio"],
                   r["depth"] / res["w"], r["depth_sub"] / res["w"]))

    tol = cfg["tolerance"]
    fine, coarse = results[-1]["rows"], results[0]["rows"]
    checks = []
    # 口径（D-54）：平冲头的经典极限载荷是**渐近**达到的 —— 冲头有限宽、底面固定，
    # FE 只能从下方逼近。所以判据是「逼近得够近 + 单调逼近 + 网格无关」，不是等式。
    min_ratio = fnum(tol, "min_ratio")
    spray = fnum(tol, "monotone_slack")
    checks.append(("1 细网格末值 ≥ %.2f × Prandtl（%.4f σ_y）" % (min_ratio, min_ratio * PRANDTL_FACTOR),
                   fine[-1]["ratio"] >= min_ratio))
    checks.append(("2 比值随 δ 单调增（末 %d 步，容差 %.3f）" % (tol["monotone_steps"], spray),
                   bool(np.all(np.diff([r["ratio"] for r in fine[-int(tol["monotone_steps"]):]])
                               >= -spray))))
    checks.append(("3 末值高于半程值（δ/w = %.3f 处）" % (-fine[-1]["u_y"] / results[-1]["w"] / 2),
                   fine[-1]["ratio"] > fine[len(fine) // 2]["ratio"]))
    # 4：网格敏感性。极限载荷从下方逼近 Prandtl，所以细化应当**抬高**比值；
    # 只看 |差| 会放过方向错误的实现，故同时判方向。实测 0.15 -> 0.1：0.9501 -> 0.9756（差 0.0255）。
    d_ref = fine[-1]["ratio"] - coarse[-1]["ratio"]
    checks.append(("4 细化抬高比值且 |Δ| < %.3f（收敛方向朝 Prandtl）" % fnum(tol, "refine_slack"),
                   0.0 < d_ref < fnum(tol, "refine_slack")))
    checks.append(("5 实质塑性区受限（深度 < %.0f%% 块深）" % (100 * fnum(tol, "contained")),
                   fine[-1]["depth_sub_over_D"] < fnum(tol, "contained")))

    say("")
    ok = True
    for name, passed in checks:
        ok = ok and passed
        say("  %-52s %s" % (name, "通过" if passed else "**未通过**"))
    say("")
    say("  网格：h/w = %.2f -> %.2f 时比值 %.4f -> %.4f（Δ = %+.4f，朝 Prandtl 收敛）"
        % (tol["refine"][0], tol["refine"][1], coarse[-1]["ratio"], fine[-1]["ratio"],
           fine[-1]["ratio"] - coarse[-1]["ratio"]))
    say("  细网格最终 p_mean = %.4g Pa = %.4f σ_y = **%.4f × Prandtl**（%.4f σ_y）"
        % (fine[-1]["p_mean"], fine[-1]["p_mean_over_sy"], fine[-1]["ratio"], PRANDTL_FACTOR))
    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "result.json").write_text(json.dumps(
            {"passed": ok, "prandtl_factor": PRANDTL_FACTOR, "sigma_y": sy,
             "checks": [{"name": n, "passed": bool(p)} for n, p in checks],
             "levels": [{"h_over_w": hw, "n_cells": r["n_cells"], "steps": r["rows"]}
                        for hw, r in zip(tol["refine"], results)]},
            ensure_ascii=False, indent=2), encoding="utf-8")
    say("")
    say("Phase 6.1（Prandtl 独立验证）%s" % ("通过" if ok else "未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
