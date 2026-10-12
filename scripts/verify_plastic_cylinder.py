"""Phase 5.4：J2 塑性 FE 解与厚壁圆筒内压弹塑性解析解对拍（PLAN §5 Phase 5.4）。

基准：Mises、平面应变、弹性不可压、理想塑性（H = 0）的闭式解，见 `pfwire.thick_cylinder`。
FE：1/4 圆环，P2 位移元，`PlasticLaw`（历史存 DG-2），nu = 0.499（解析取 nu = 0.5 极限）。

对拍项（阈值在 configs/verify_plastic_cylinder.yaml）：
  1. 起屈压力 p_y：由最后一个弹性步按弹性比例外推（σ_eq 与 p 成正比），与解析比；
  2. 弹性段内壁位移 u_r(a)：与**实际 nu** 的 Lamé 解比（精确解，应到 1e-3）；
  3. 每个弹塑性压力水平：弹塑性边界 c、内壁位移 u_r(a)、塑性区 σ_r/σ_θ 的 L2 误差；
  4. 加密网格后上述误差不变差（检验 DG-2 存历史带来的近似是收敛的，D-7）。
Newton 不收敛即停（硬约束 6）。

用法：python scripts/verify_plastic_cylinder.py [configs/verify_plastic_cylinder.yaml]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

from pfwire.plasticity import J2Params, PlasticLaw
from pfwire.thick_cylinder import (
    c_from_pressure, lame_u_inner, plastic_stress, plastic_u_inner, yield_onset_pressure)
from pfwire.verify_contact_common import fnum, load_config

REPO = pathlib.Path(__file__).resolve().parents[1]


def build_mesh(a, b, nr, nth, comm):
    """(r, theta) 矩形映射成 1/4 圆环（四边形，几何一阶）。"""
    m = dmesh.create_rectangle(comm, [np.array([a, 0.0]), np.array([b, np.pi / 2])],
                               [nr, nth], dmesh.CellType.quadrilateral)
    x = m.geometry.x
    r, th = x[:, 0].copy(), x[:, 1].copy()
    x[:, 0], x[:, 1] = r * np.cos(th), r * np.sin(th)
    return m


def run_level(cfg, nr, nth, comm, say, nu=None):
    g, mat, ld, sol = cfg["geometry"], cfg["material"], cfg["loading"], cfg["solver"]
    a, b = fnum(g, "a"), fnum(g, "b")
    E, sy, H = fnum(mat, "E"), fnum(mat, "sigma_y"), fnum(mat, "H")
    nu = fnum(mat, "nu") if nu is None else float(nu)
    prm = J2Params(E=E, nu=nu, sigma_y0=sy, H=H)
    mesh = build_mesh(a, b, nr, nth, comm)
    law = PlasticLaw(mesh, prm)
    V = fem.functionspace(mesh, ("Lagrange", 2, (2,)))
    u = fem.Function(V)
    v = ufl.TestFunction(V)

    fdim = mesh.topology.dim - 1
    inner_f = dmesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.hypot(x[0], x[1]) < a * (1.0 + 1e-9))
    ft = dmesh.meshtags(mesh, fdim, np.sort(inner_f), np.full(len(inner_f), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft, metadata={"quadrature_degree": 4})
    X = ufl.SpatialCoordinate(mesh)
    rhat = X / ufl.sqrt(ufl.dot(X, X))
    p_load = fem.Constant(mesh, np.float64(0.0))
    # 内壁受压：作用在物体上的面力 = p * r_hat（物体外法向为 -r_hat，t = -p n）
    F = ufl.inner(law.sigma(u), law.strain(v)) * law.dx - p_load * ufl.dot(rhat, v) * ds(1)

    zero = fem.Constant(mesh, np.float64(0.0))
    left = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 0.0, atol=1e-12 * b))
    bottom = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 0.0, atol=1e-12 * b))
    bcs = [fem.dirichletbc(zero, fem.locate_dofs_topological(V.sub(0), fdim, left), V.sub(0)),
           fem.dirichletbc(zero, fem.locate_dofs_topological(V.sub(1), fdim, bottom), V.sub(1))]
    prob = NonlinearProblem(
        F, u, bcs=bcs, petsc_options_prefix="pl_cyl_",
        petsc_options={"snes_type": "newtonls", "snes_linesearch_type": sol["linesearch"],
                       "snes_atol": fnum(sol, "snes_atol"), "snes_rtol": fnum(sol, "snes_rtol"),
                       "snes_max_it": int(sol["snes_max_it"]),
                       "ksp_type": "preonly", "pc_type": "lu"})

    # 采样量（DG-2 向量）：[sigma_xx, sigma_yy, p_eq, sigma_eq, u_x]
    upd = law.update(u)
    s3 = upd.sigma3
    dev = s3 - ufl.tr(s3) / 3.0 * ufl.Identity(3)
    expr = ufl.as_vector([upd.sigma[0, 0], upd.sigma[1, 1], law.p_n,
                          ufl.sqrt(1.5 * ufl.inner(dev, dev)), u[0]])
    W = fem.functionspace(mesh, ("DG", 2, (5,)))
    fW = fem.Function(W)
    cexpr = fem.Expression(expr, W.element.interpolation_points)
    coords = W.tabulate_dof_coordinates()
    n_owned = W.dofmap.index_map.size_local
    on_line = np.where(np.isclose(coords[:n_owned, 1], 0.0, atol=1e-12 * b))[0]

    def sample():
        fW.interpolate(cexpr)
        vals = fW.x.array.reshape(-1, 5)[on_line]
        r = coords[on_line, 0]
        r = np.concatenate(comm.allgather(r))
        vals = np.concatenate(comm.allgather(vals))
        o = np.argsort(r, kind="stable")
        return r[o], vals[o]

    step, p_max = fnum(ld, "step"), fnum(ld, "p_max")
    n_steps = int(round(p_max / step))
    levels = [round(x, 10) for x in cfg["loading"]["report_levels"]]
    snaps, last_elastic, p_y_fe, iters = {}, None, None, []
    for k in range(1, n_steps + 1):
        frac = round(k * step, 10)
        p_load.value = frac * sy
        prob.solve()
        if prob.solver.getConvergedReason() <= 0:
            raise RuntimeError("Newton 未收敛：压力 %.3f sigma_y（网格 %dx%d），停止，不使用未收敛解"
                               % (frac, nr, nth))
        iters.append(int(prob.solver.getIterationNumber()))
        law.commit(u)
        r, vals = sample()
        pmax_loc = comm.allreduce(float(law.p_n.x.array.max()), op=MPI.MAX)
        if pmax_loc <= 1e-12:
            last_elastic = (frac * sy, float(vals[:, 3].max()))
        elif p_y_fe is None:
            p_el, seq = last_elastic
            p_y_fe = p_el * sy / seq                  # 弹性段 σ_eq ∝ p
        if any(abs(frac - L) < 1e-9 for L in levels):
            snaps[frac] = (r.copy(), vals.copy())
    return dict(a=a, b=b, E=E, nu=nu, sy=sy, p_y_fe=p_y_fe, snaps=snaps, iters=iters,
                n_cells=int(mesh.topology.index_map(mesh.topology.dim).size_global))


def _split(v, sr, st, mask, sy):
    """应力误差拆成偏量部分（σθ − σr）与平均部分（(σr + σθ)/2），都除以 sigma_y 取 L2。

    近不可压（nu -> 1/2）的位移元里压力场会振荡，它同时抬高 σr 和 σθ、不进 σθ − σr；
    拆开才能区分「屈服条件/返回映射的误差」与「近不可压离散的压力误差」。
    """
    d_fe, d_ref = v[mask, 1] - v[mask, 0], st[mask] - sr[mask]
    m_fe, m_ref = 0.5 * (v[mask, 1] + v[mask, 0]), 0.5 * (st[mask] + sr[mask])
    return {"dev_l2": float(np.sqrt(np.mean((d_fe - d_ref) ** 2)) / sy),
            "mean_l2": float(np.sqrt(np.mean((m_fe - m_ref) ** 2)) / sy)}


def errors_at(level, snap, a, b, E, nu, sy):
    """一个压力水平上的各项相对误差（与解析比）。"""
    r, v = snap
    p = level * sy
    out = {"level": level}
    u_a = float(np.mean(v[np.abs(r - a) < 1e-9 * b, 4]))
    if p <= yield_onset_pressure(sy, a, b):
        ref = lame_u_inner(p, E, nu, a, b)
        out.update(kind="elastic", u_a=u_a, u_ref=ref, u_err=abs(u_a - ref) / ref)
        sr, st = plastic_stress(r, p, sy, a, b)
        out["stress_l2"] = float(np.sqrt(np.mean(((v[:, 0] - sr) ** 2 + (v[:, 1] - st) ** 2) / 2)) / sy)
        out.update(_split(v, sr, st, np.ones(len(r), bool), sy))
        return out
    c_ref = c_from_pressure(p, sy, a, b)
    pl = v[:, 2]
    thr = 1e-3 * pl.max()
    c_fe = float(r[pl > thr].max())
    G_inf = E / 3.0                                  # nu = 1/2 极限的剪切模量
    u_ref = plastic_u_inner(p, sy, G_inf, a, b)
    sr, st = plastic_stress(r, p, sy, a, b)
    zone = r <= 0.95 * c_ref
    l2 = float(np.sqrt(np.mean(((v[zone, 0] - sr[zone]) ** 2 + (v[zone, 1] - st[zone]) ** 2) / 2)) / sy)
    out.update(_split(v, sr, st, zone, sy))
    out.update(kind="plastic", c_fe=c_fe, c_ref=c_ref, c_err=abs(c_fe - c_ref) / c_ref,
               u_a=u_a, u_ref=u_ref, u_err=abs(u_a - u_ref) / u_ref, stress_l2=l2,
               n_zone=int(zone.sum()))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 5.4 厚壁圆筒内压弹塑性对拍")
    ap.add_argument("config", nargs="?", default="configs/verify_plastic_cylinder.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    comm = MPI.COMM_WORLD
    outdir = REPO / cfg["output"]["dir"]

    def say(msg):
        if comm.rank == 0:
            print(msg, flush=True)

    tol = cfg["tolerance"]
    say("Phase 5.4：J2 塑性 FE vs 厚壁圆筒内压弹塑性解析解（H = %g，P2 位移元）" % fnum(cfg["material"], "H"))
    checks, case_out = [], []

    def add(name, passed, detail):
        checks.append((name, bool(passed), detail))

    slack = fnum(tol, "refine_slack")
    for case in cfg["cases"]:
        nu, gate = float(case["nu"]), case["stress_gate"]
        gkey = {"dev": "dev_l2", "full": "stress_l2"}[gate]
        gname = {"dev": "偏量 σθ−σr", "full": "σr、σθ 全量"}[gate]
        say("")
        say("== 工况 %s：nu = %g，应力判据取 %s ==" % (case["name"], nu, gname))
        results = []
        for nr, nth in case["meshes"]:
            say("  网格 %d x %d ..." % (nr, nth))
            res = run_level(cfg, int(nr), int(nth), comm, say, nu=nu)
            errs = [errors_at(L, s_, res["a"], res["b"], res["E"], res["nu"], res["sy"])
                    for L, s_ in sorted(res["snaps"].items())]
            py_ref = yield_onset_pressure(res["sy"], res["a"], res["b"])
            res_out = dict(nr=nr, nth=nth, n_cells=res["n_cells"], p_y_fe=res["p_y_fe"],
                           p_y_ref=py_ref, p_y_err=abs(res["p_y_fe"] - py_ref) / py_ref,
                           newton_iters=res["iters"], errors=errs, _snaps=res["snaps"])
            results.append(res_out)
            say("    %d 单元；Newton 每步迭代数 %s；起屈压力 FE %.4g / 解析 %.4g（差 %.2f%%）"
                % (res["n_cells"], res["iters"], res["p_y_fe"], py_ref, 100 * res_out["p_y_err"]))
            for e in errs:
                if e["kind"] == "elastic":
                    say("    p=%.2f sy 弹性：u_r(a) 差 %.3e（Lamé，实际 nu）；应力 L2 %.3e（偏量 %.3e / 平均 %.3e）"
                        % (e["level"], e["u_err"], e["stress_l2"], e["dev_l2"], e["mean_l2"]))
                else:
                    say("    p=%.2f sy 弹塑性：c FE %.4g / 解析 %.4g（差 %.2f%%）；u_r(a) 差 %.2f%%；"
                        "应力 L2 %.2f%%（偏量 %.2f%% / 平均 %.2f%%；%d 点）"
                        % (e["level"], e["c_fe"], e["c_ref"], 100 * e["c_err"], 100 * e["u_err"],
                           100 * e["stress_l2"], 100 * e["dev_l2"], 100 * e["mean_l2"], e["n_zone"]))
        fine, coarse = results[-1], results[0]
        pre = "[%s] " % case["name"]
        add(pre + "1 起屈压力 p_y", fine["p_y_err"] < fnum(tol, "p_yield_rel"),
            "%.2f%% < %.1f%%" % (100 * fine["p_y_err"], 100 * fnum(tol, "p_yield_rel")))
        for e in fine["errors"]:
            tag = "p=%.2f" % e["level"]
            if e["kind"] == "elastic":
                add(pre + "2 弹性 %s u_r(a) vs Lamé" % tag, e["u_err"] < fnum(tol, "elastic_u_rel"),
                    "%.2e < %.0e" % (e["u_err"], fnum(tol, "elastic_u_rel")))
                add(pre + "2 弹性 %s 应力（%s）" % (tag, gname), e[gkey] < fnum(tol, "stress_l2"),
                    "%.2f%% < %.1f%%" % (100 * e[gkey], 100 * fnum(tol, "stress_l2")))
            else:
                add(pre + "3 %s 边界 c" % tag, e["c_err"] < fnum(tol, "c_rel"),
                    "%.2f%% < %.1f%%" % (100 * e["c_err"], 100 * fnum(tol, "c_rel")))
                add(pre + "3 %s u_r(a)" % tag, e["u_err"] < fnum(tol, "u_rel"),
                    "%.2f%% < %.1f%%" % (100 * e["u_err"], 100 * fnum(tol, "u_rel")))
                add(pre + "3 %s 应力（%s）" % (tag, gname), e[gkey] < fnum(tol, "stress_l2"),
                    "%.2f%% < %.1f%%" % (100 * e[gkey], 100 * fnum(tol, "stress_l2")))
        for ec, ef in zip(coarse["errors"], fine["errors"]):
            keys = ["u_err", gkey, "mean_l2"] + (["c_err"] if ef["kind"] == "plastic" else [])
            for key in keys:
                add(pre + "4 加密不变差 p=%.2f %s" % (ef["level"], key), ef[key] <= ec[key] + slack,
                    "粗 %.3e -> 细 %.3e" % (ec[key], ef[key]))
        case_out.append((case["name"], results))

    say("")
    ok = True
    for name, passed, detail in checks:
        ok = ok and passed
        say("  %-44s %-26s %s" % (name, detail, "通过" if passed else "**未通过**"))
    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        clean = {n: [{k: v for k, v in r.items() if k != "_snaps"} for r in rs] for n, rs in case_out}
        (outdir / "result.json").write_text(json.dumps(
            {"passed": ok, "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
             "cases": clean}, ensure_ascii=False, indent=2), encoding="utf-8")
        for n, rs in case_out:
            f = rs[-1]["_snaps"]
            np.savez_compressed(outdir / ("profiles_fine_%s.npz" % n),
                                **{"r_%.2f" % L: s_[0] for L, s_ in f.items()},
                                **{"v_%.2f" % L: s_[1] for L, s_ in f.items()})
    say("")
    say("Phase 5.4 %s" % ("通过" if ok else "未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
