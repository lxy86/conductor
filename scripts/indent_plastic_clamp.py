"""Phase 6.1 本体：弹塑性线夹压单丝（PLAN §6.1，无摩擦、无相场）。

几何与目的见 configs/indent_plastic_clamp.yaml 的头注释。此脚本回答两个问题：
1. **接触压力是否饱和**：p_mean/σ_y 随载荷升高而趋于一个平台（D-50：完全塑性 ~2.15 σ_y；
   平面应变 Prandtl 上限 2.97 σ_y）。饱和用末段的 d ln p0/d ln P 判定（Hertz 线接触 = 0.5）。
2. **塑性接触半宽 a_pl 是多少**：单丝顶面上等效塑性应变超阈值的横向范围（PLAN 6.2 要用）。

顺带做 D-54 记的三个自洽量：力平衡、穿透量、γ 敏感性（「接触 + 塑性」耦合没有闭式解可对，
只能靠这些内部一致性检查把关；塑性的正确性另由 Phase 5.4 的厚壁圆筒与 6.1 的 Prandtl
条带验证保证）。

用法：python scripts/indent_plastic_clamp.py [configs/indent_plastic_clamp.yaml]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from dolfinx import default_scalar_type
from dolfinx.fem import Constant, dirichletbc, locate_dofs_topological
from mpi4py import MPI

from pfwire.contact import CLAMP_WIRE, ContactModel, build_clamp_wire, fix_surface
from pfwire.indent import contact_extent, mean_pressure, saturation_exponent
from pfwire.loads import hertz_half_width, hertz_peak_pressure
from pfwire.material import contact_effective_modulus
from pfwire.plasticity import J2Params
from pfwire.verify_contact_common import fnum, load_config

REPO = pathlib.Path(__file__).resolve().parents[1]
#: 判「塑性区」用的等效塑性应变阈值（相对 σ_y/E）。
P_EQ_FLOOR = 1.0e-3


def build(cfg, comm, *, gamma_factor=None):
    g, mat, ld, sol, m = (cfg["geometry"], cfg["material"], cfg["loading"],
                          cfg["solver"], cfg["mesh"])
    D, Rc = fnum(g, "D"), fnum(g, "R_clamp")
    mesh, ct, ft = build_clamp_wire(
        REPO / cfg["output"]["dir"], R_clamp=Rc, Lc=fnum(g, "Lc"), Hc=fnum(g, "Hc"),
        Lw=fnum(g, "Lw"), D=D, gap=fnum(g, "gap"),
        h_fine=fnum(m, "h_fine"), h_coarse=fnum(m, "h_coarse"),
        fine_halfwidth=fnum(m, "fine_halfwidth"), order=int(m["order"]),
        quads=True, comm=comm)
    E_w, nu_w = fnum(mat, "E_wire"), fnum(mat, "nu_wire")
    prm = J2Params(E=E_w, nu=nu_w, sigma_y0=fnum(mat, "sigma_y"), H=fnum(mat, "H"))
    model = ContactModel(
        mesh, ct, ft, E=(fnum(mat, "E_clamp"), E_w), nu=(fnum(mat, "nu_clamp"), nu_w),
        order=int(m["degree"]) if "degree" in m else int(m["order"]),
        tags=CLAMP_WIRE, plastic={1: prm})

    # 位移驱动：线夹顶面 u_y = -δ、u_x = 0（消平面内刚体模）；单丝底面全固定。
    g_uy = Constant(mesh, default_scalar_type(0.0))
    g_ux = Constant(mesh, default_scalar_type(0.0))
    fdim = mesh.topology.dim - 1
    top = ft.find(CLAMP_WIRE.load_facet)
    bcs = [dirichletbc(g_uy, locate_dofs_topological(model.V.sub(1), fdim, top), model.V.sub(1)),
           dirichletbc(g_ux, locate_dofs_topological(model.V.sub(0), fdim, top), model.V.sub(0)),
           fix_surface(model)]
    ksp = fnum(sol, "ksp_tol")
    model.setup(bcs=bcs, traction=None, friction=None,
                gamma_factor=fnum(sol, "nitsche_gamma_factor") if gamma_factor is None else gamma_factor,
                theta=fnum(sol, "nitsche_theta"),
                newton_options={"relaxation_parameter": 1.0,
                                "atol": fnum(sol, "newton_atol"), "rtol": fnum(sol, "newton_rtol"),
                                "convergence_criterion": "residual",
                                "max_it": int(sol["max_it"]), "error_on_nonconvergence": True},
                petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                               "ksp_rtol": ksp, "ksp_atol": ksp})
    return model, g_uy, mesh, E_w, nu_w, Rc, D


def measure(model, E_c, nu_c, E_w, nu_w, Rc, D):
    """一个状态的诊断量。p0、a 用采样剖面；P 用精确面积分（force_facet）。

    Hertz 参考的 E* 必须与**模型实际刚度**一致（线夹是 E_clamp 的弹性体，不是刚体）。
    早先用刚体线夹的 E* 会引入 ~5% 的系统性参考误差（见 D-55）。
    """
    xs, ps, _ = model.surface_tractions()
    a, p0, _ = contact_extent(xs, ps)
    P = model.resultant(component=1)
    xg, gn = model.penetration()
    e_star = contact_effective_modulus(E_w, nu_w, E_c, nu_c)
    a_hz = hertz_half_width(P=max(P, 1e-30), R=Rc, E_star=e_star)
    p0_hz = hertz_peak_pressure(P=max(P, 1e-30), a=a_hz)
    # 塑性区横向范围与深度（DG-2 历史）
    law = model.laws[1]
    thrc = P_EQ_FLOOR * fnum_ratio(law)
    coords = law.V_p.tabulate_dof_coordinates()
    n_own = law.V_p.dofmap.index_map.size_local
    p = law.p_n.x.array[:n_own]
    pl = p > thrc
    a_pl = float(np.abs(coords[:n_own, 0][pl]).max()) if pl.any() else 0.0
    depth = float(-coords[:n_own, 1][pl].min()) if pl.any() else 0.0
    comm = model.mesh.comm
    # 线夹顶面反力（精确面积分）。**力平衡要拿它与接触面合力比**，不是与 ∫p dx 比 ——
    # 后者是对 CG1 节点值做梯形，含接触区外的数值残尾（D-49：实测差 3.1%）。
    P_clamp = model.resultant(component=1, tag=model.tags.load_facet, body=0)
    return dict(a=float(a), p0=float(p0), P=float(P), P_clamp=float(P_clamp),
                p_mean=float(mean_pressure(P, a)),
                a_hz=float(a_hz), p0_hz=float(p0_hz),
                a_pl=comm.allreduce(a_pl, op=MPI.MAX), depth=comm.allreduce(depth, op=MPI.MAX),
                p_eq_max=comm.allreduce(float(law.p_n.x.array.max()), op=MPI.MAX),
                pen=float(np.min(gn)) if gn.size else float("nan"),
                xs=xs, ps=ps, n_pl=int(comm.allreduce(int(pl.sum()), op=MPI.SUM)))


def fnum_ratio(law) -> float:
    """塑性阈值 = P_EQ_FLOOR * σ_y/E（用 J2Params 推）。"""
    return law.prm.sigma_y0 / law.prm.E


def run(cfg, comm, say, *, gamma_factor=None):
    model, g_uy, mesh, E_w, nu_w, Rc, D = build(cfg, comm, gamma_factor=gamma_factor)
    E_c, nu_c = fnum(cfg["material"], "E_clamp"), fnum(cfg["material"], "nu_clamp")
    steps = int(cfg["loading"]["steps"])
    u_y = fnum(cfg["loading"], "u_y")
    rows, ok = [], True
    for i in range(steps):
        g_uy.value = u_y * (i + 1) / steps
        try:
            n_it = model.step("压下 %d/%d" % (i + 1, steps))
        except RuntimeError as exc:                     # 硬约束 6：不收敛即停
            say("  ⚠ 压下步 %d/%d 未收敛：%s" % (i + 1, steps, exc))
            ok = False
            break
        rows.append({**measure(model, E_c, nu_c, E_w, nu_w, Rc, D), "step": i + 1,
                     "u_y": float(g_uy.value), "n_it": n_it})
        if i < steps - 1:
            model.prepare_next()
    return rows, ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 6.1：弹塑性线夹压单丝")
    ap.add_argument("config", nargs="?", default="configs/indent_plastic_clamp.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    comm = MPI.COMM_WORLD
    outdir = REPO / cfg["output"]["dir"]
    sy = fnum(cfg["material"], "sigma_y")

    def say(msg):
        if comm.rank == 0:
            print(msg, flush=True)

    say("Phase 6.1：弹塑性线夹压单丝（R_clamp = %.0f mm，D = %.2f mm，σ_y = %.0f MPa）"
        % (1e3 * fnum(cfg["geometry"], "R_clamp"), 1e3 * fnum(cfg["geometry"], "D"), sy / 1e6))
    rows, ok = run(cfg, comm, say)
    if not rows:
        say("没有任何收敛步，停止")
        return 1
    say("")
    say("  %-4s %11s %11s %10s %10s %10s %10s %9s %9s"
        % ("步", "δ (µm)", "P (N/m)", "p0 (MPa)", "a (mm)", "p_mean/σy", "a_pl (mm)", "塑性深", "max p_eq"))
    for r in rows:
        say("  %-4d %11.2f %11.4g %10.1f %10.4f %10.4f %10.4f %9.3f %9.4f"
            % (r["step"], -1e6 * r["u_y"], r["P"], r["p0"] / 1e6, 1e3 * r["a"],
               r["p_mean"] / sy, 1e3 * r["a_pl"], r["depth"] / 1e-3, r["p_eq_max"]))

    tol = cfg["tolerance"]
    # 工作点：最接近 D-50 的 W' = 1.75e6 N/m 的那一步
    w_target = fnum(cfg["loading"], "w_target")
    op = min(rows, key=lambda r: abs(r["P"] - w_target))
    say("")
    say("  工作点（最接近 W' = %.3g N/m 的一步：δ = %.2f µm，P = %.4g N/m）："
        % (w_target, -1e6 * op["u_y"], op["P"]))
    say("    p0 = %.1f MPa = %.3f σ_y；a = %.4f mm；p_mean = %.1f MPa = %.3f σ_y；"
        "a_pl = %.4f mm；塑性区深 %.3f mm"
        % (op["p0"] / 1e6, op["p0"] / sy, 1e3 * op["a"], op["p_mean"] / 1e6,
           op["p_mean"] / sy, 1e3 * op["a_pl"], op["depth"] / 1e-3))

    # 弹性段：还没有塑性的最早几步与 Hertz 比
    el = [r for r in rows if r["p_eq_max"] == 0.0]
    checks = [("1 全部压下步 Newton 收敛", ok and len(rows) == int(cfg["loading"]["steps"]))]
    if el:
        r = el[-1]
        e_a = abs(r["a"] - r["a_hz"]) / r["a_hz"]
        e_p = abs(r["p0"] - r["p0_hz"]) / r["p0_hz"]
        checks.append(("2 弹性段（δ = %.2f µm）a、p0 vs Hertz < %.0f%%"
                       % (-1e6 * r["u_y"], 100 * fnum(tol, "elastic_rel")),
                       e_a < fnum(tol, "elastic_rel") and e_p < fnum(tol, "elastic_rel")))
        say("  弹性段最后一步（δ = %.2f µm）：a %.4f vs Hertz %.4f（差 %.2f%%）；"
            "p0 %.1f vs %.1f MPa（差 %.2f%%）"
            % (-1e6 * r["u_y"], 1e3 * r["a"], 1e3 * r["a_hz"], 100 * e_a,
               r["p0"] / 1e6, r["p0_hz"] / 1e6, 100 * e_p))
    else:
        checks.append(("2 弹性段存在（有未屈服的步）", False))

    band = tol["p0_band_over_sy"]
    lo, hi = float(band[0]), float(band[1])
    dx_int = abs(rows[-1]["P"] - np.trapezoid(np.clip(rows[-1]["ps"], 0, None), rows[-1]["xs"]))
    sat = saturation_exponent([r["P"] for r in rows[2:]], [r["p0"] for r in rows[2:]],
                              window=min(4, len(rows) - 2))
    checks += [
        ("3 工作点 p0/σy = %.2f ∈ [%.2f, %.2f]（D-50 的 2.15 ~ Prandtl 2.97）"
         % (op["p0"] / sy, lo, hi), lo <= op["p0"] / sy <= hi),
        ("4 工作点 Q 量级合理：P 在工作点的量级 %.3g vs W' %.3g（差 < 50%%）"
         % (op["P"], w_target), abs(op["P"] - w_target) / w_target < 0.5),
        ("5a 力平衡：线夹顶面反力 vs 接触面合力 < %.0f%%（**精确面积分对精确面积分**）"
         % (100 * fnum(tol, "force_balance")),
         abs(rows[-1]["P_clamp"] - rows[-1]["P"]) / abs(rows[-1]["P"]) < fnum(tol, "force_balance")),
        ("5b 穿透 max|g_n| / a < %.0f%%" % (100 * fnum(tol, "penetration_over_a")),
         abs(rows[-1]["pen"]) / rows[-1]["a"] < fnum(tol, "penetration_over_a")),
    ]
    gfac = fnum(cfg["solver"], "nitsche_gamma_factor")
    say("  参考：∫p dx（CG1 梯形）与接触面合力差 %.2f%%（诊断量，走 D-49 的口径）"
        % (100 * dx_int / abs(rows[-1]["P"])))
    say("  重跑 γ = %.0f -> %.0f 检查敏感性 ..." % (gfac, 2 * gfac))
    rows2, ok2 = run(cfg, comm, say, gamma_factor=2.0 * gfac)
    if ok2 and rows2:
        d_p0 = abs(rows2[-1]["p0"] - rows[-1]["p0"]) / rows[-1]["p0"]
        d_a = abs(rows2[-1]["a"] - rows[-1]["a"]) / rows[-1]["a"]
        lim = fnum(tol, "gamma_sensitivity")
        checks.append(("5c γ 加倍后 p0、a 变化 < %.0f%%" % (100 * lim), d_p0 < lim and d_a < lim))
        say("    Δp0 = %.2f%%，Δa = %.2f%%" % (100 * d_p0, 100 * d_a))
    else:
        checks.append(("5c γ 加倍后收敛", False))

    say("")
    ok_all = True
    for name, passed in checks:
        ok_all = ok_all and passed
        say("  %-62s %s" % (name, "通过" if passed else "**未通过**"))
    say("")
    say("  **a_pl = %.4f mm**（PLAN 6.2 的加密区尺寸按它定，不得沿用旧模型的 11.62 µm）"
        % (1e3 * op["a_pl"]))
    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        clean = [{k: v for k, v in r.items() if k not in ("xs", "ps")} for r in rows]
        (outdir / "result.json").write_text(json.dumps(
            {"passed": ok_all, "sigma_y": sy, "w_target": w_target,
             "operating_point": {k: v for k, v in op.items() if k not in ("xs", "ps")},
             "sat_exponent": sat.exponent,
             "checks": [{"name": n, "passed": bool(p)} for n, p in checks], "steps": clean},
            ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez_compressed(outdir / "profiles.npz",
                            **{"x_%d" % r["step"]: r["xs"] for r in rows},
                            **{"p_%d" % r["step"]: r["ps"] for r in rows})
    say("")
    say("Phase 6.1 %s" % ("通过" if ok_all else "未通过"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
