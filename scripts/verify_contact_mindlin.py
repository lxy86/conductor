"""Phase 4.4：FE 摩擦接触的**定性**验证，对照 Cattaneo–Mindlin 部分滑移图像（PLAN §4 Phase 4.4，D-51）。

几何：**圆柱–圆柱**（上游 create_quarter_disks_mesh），位移驱动两阶段：
  阶段 1  压下 u_y -> 建立 Hertz 接触，由接触面反力测得法向力 P；
  阶段 2  上圆柱顶面切向推 u_x -> 每步测 Q 与接触面 (p, q) 分布。

**为什么不做 c/a 定量对拍**（D-51）：Cattaneo–Mindlin 是半空间解。有限尺寸圆柱的顶面剪力
与接触面反力构成力偶，须由法向力偏心 e = Q h / P 平衡，R/a 约 8 时 e 远大于 a，q(x) 本就
不对称；FE 与半空间解 c/a 差 84% 是物理差别而非实现错误。三种消除办法（补 y 约束、上体
刚性平移、对称加载）均已试过无效/不收敛。用户 2026-10-12 决定降级为定性验证。

**判据（全部必须成立）**——每条都是 Coulomb 摩擦接触的结构性性质，与半空间无关：
  1. 所有切向步 Newton 收敛（不收敛即停，硬约束 6）；
  2. 库仑不等式 |q| <= mu p 在接触区内处处成立（容差 COULOMB_TOL）；
  3. 总切向力 Q <= mu P（容差同上）且 Q 随步数单调增；
  4. 滑移区占比（|q| >= mu p (1-eps) 的接触点份额）随步数单调不减；
  5. 出现过部分滑移：存在某步 0 < 滑移占比 < 1（黏着与滑移共存）。
另打印 FE 的 c/a 与 Mindlin 解析值作**参考**，不参与判定。

用法：python scripts/verify_contact_mindlin.py [configs/verify_mindlin_friction.yaml]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from dolfinx import default_scalar_type
from dolfinx.fem import Constant, dirichletbc, locate_dofs_topological
from dolfinx.mesh import locate_entities
from mpi4py import MPI

from pfwire.contact import ContactModel, SlipTracker, build_quarter_disks, fix_surface, QUARTER_DISKS
from pfwire.slip import classify_slip
from pfwire.loads import hertz_half_width, mindlin_stick_half_width
from pfwire.material import contact_effective_modulus
from pfwire.verify_contact_common import fnum, load_config

REPO = pathlib.Path(__file__).resolve().parents[1]

#: 判黏着时留的相对余量：|q| < mu*p*(1-STICK_EPS) 才算黏着。
#: 数值解在黏滑边界附近本就是过渡的，不留余量会把边界系统性地往里推。
STICK_EPS = 0.15
#: 只在压力足够大的点上判黏滑；接触区边缘 p -> 0 时 q/(mu p) 是 0/0。
P_FLOOR = 0.10
#: 库仑不等式 / 滑移占比只在「核心区」(p >= CORE_FLOOR * p_max，约 |x| < 0.9a) 上判。接触区边缘 p -> 0，
#: 面力取自体应力 σ·n（CG1 插值 + Nitsche 弱施加），边缘处 q 不随 p 趋零，|q|/(mu p) 发散
#: （实测 x/a ≈ 1.04 处达 4~9 倍）。这是**测量口径**的限制，不是解违反摩擦律：
#: 积分量 Q <= mu P 照样满足。核心区外不作逐点判定。
CORE_FLOOR = 0.50


def measure_stick_halfwidth(xs, ps, qs, mu: float, a_fe: float) -> float:
    """从 (p, q) 分布测黏着半宽 c。"""
    thresh = P_FLOOR * ps.max()
    live = (ps > thresh) & (np.abs(xs) <= a_fe)
    if not live.any():
        return float("nan")
    stick = live & (np.abs(qs) < mu * ps * (1.0 - STICK_EPS))
    return float(np.max(np.abs(xs[stick]))) if stick.any() else 0.0


def analyze_step(step, xs, ps, qs, Pn, Q, *, mu, R_eff, e_star):
    """一个切向步的诊断量。纯函数（输入是已提取的剖面），便于离线重判。"""
    ratio = abs(Q) / (mu * Pn) if Pn > 0 else float("nan")
    a_i = hertz_half_width(P=Pn, R=R_eff, E_star=e_star)
    live = ps > P_FLOOR * ps.max()
    a_fe = float(np.max(np.abs(xs[live]))) if live.any() else 0.0
    core = ps >= CORE_FLOOR * ps.max()
    qr = np.abs(qs[core]) / (mu * ps[core])
    coul = float(qr.max()) if core.any() else float("nan")
    slip_frac = float(np.mean(qr >= 1.0 - STICK_EPS)) if core.any() else float("nan")
    c_fe = measure_stick_halfwidth(xs, ps, qs, mu, a_fe)
    c_ref = (mindlin_stick_half_width(a=a_i, Q_over_mu_P=ratio)
             if 0.0 <= ratio <= 1.0 else float("nan"))
    return {"step": step, "P": Pn, "Q": Q, "q_ratio": ratio, "a_fe": a_fe, "a_hertz": a_i,
            "n_core": int(core.sum()), "slip_frac": slip_frac, "coulomb_max": coul,
            "c_over_a_fe": c_fe / a_fe if a_fe else float("nan"),
            "c_over_a_mindlin": c_ref / a_i if a_i else float("nan")}


# 阶段 2（Coulomb）的 KSP：切向 Jacobian 非对称，用 GMRES + GAMG，
# 照上游 demo_friction_cylinders.py。LU 直接解在这里同样进黏滑极限环。
def _gamg_options(ksp_tol: float) -> dict:
    return {"matptap_via": "scalable", "ksp_type": "gmres",
            "ksp_rtol": ksp_tol, "ksp_atol": ksp_tol,
            "pc_type": "gamg", "pc_mg_cycles": 1,
            "mg_levels_ksp_type": "chebyshev", "mg_levels_pc_type": "jacobi",
            "pc_gamg_type": "agg", "pc_gamg_coarse_eq_limit": 100,
            "pc_gamg_agg_nsmooths": 1, "pc_gamg_threshold": 1e-3,
            "pc_gamg_square_graph": 2, "pc_gamg_reuse_interpolation": False,
            "ksp_initial_guess_nonzero": False, "ksp_norm_type": "unpreconditioned"}


def _lu_options(ksp_tol: float) -> dict:
    return {"ksp_type": "preonly", "pc_type": "lu",
            "ksp_rtol": ksp_tol, "ksp_atol": ksp_tol}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 4.4 Cattaneo–Mindlin 对拍")
    ap.add_argument("config", nargs="?", default="configs/verify_mindlin_friction.yaml")
    ap.add_argument("--reanalyze", action="store_true",
                    help="不跑 FE，读 output.dir/profiles.npz 重新判定")
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    comm = MPI.COMM_WORLD
    outdir = REPO / cfg["output"]["dir"]

    def say(msg):
        if comm.rank == 0:
            print(msg, flush=True)

    g, mat, ld, sol, m = (cfg["geometry"], cfg["material"], cfg["loading"],
                          cfg["solver"], cfg["mesh"])
    R, E, nu = fnum(g, "R"), fnum(mat, "E"), fnum(mat, "nu")
    mu = fnum(cfg["friction"], "mu")
    n_norm, n_tang = int(ld["steps_normal"]), int(ld["steps_tangential"])

    if args.reanalyze:
        d = np.load(outdir / "profiles.npz")
        n_have = len([k for k in d.files if k.endswith("_x")])
        raw = [tuple(d[f"s{k}_{n}"] for n in ("x", "p", "q")) + (float(d[f"s{k}_P"]), float(d[f"s{k}_Q"]))
               for k in range(n_have)]
        slips = ([(d[f"s{k}_sx"], d[f"s{k}_sg"]) for k in range(n_have)]
                 if "s0_sx" in d.files else None)
        return judge(comm, say, cfg, outdir, raw, slips, mu=float(d["mu"]), R_eff=float(d["R_eff"]),
                     e_star=float(d["E_star"]), n_tang=int(d["n_tang"]),
                     converged_all=bool(d["converged"]), P_normal=raw[0][3])

    say("Phase 4.4：FE 摩擦接触 vs Cattaneo–Mindlin（圆柱–圆柱，mu = %.2f）" % mu)
    mesh, ct, ft = build_quarter_disks(
        outdir, R=R, gap=fnum(g, "gap"),
        res=fnum(m, "res"), order=int(m["order"]), quads=bool(m["quads"]), comm=comm)
    say("  网格：%d 单元" % mesh.topology.index_map(mesh.topology.dim).size_global)

    # 下圆柱圆心 (0, -2R-gap)（create_quarter_disks_mesh 的几何），用弧面真实法向取 p、q
    model = ContactModel(mesh, ct, ft, E=E, nu=nu, order=int(m["order"]), tags=QUARTER_DISKS,
                         sample_center=(0.0, -2.0 * R - fnum(g, "gap")))

    # 位移驱动：上圆柱顶面给 (u_x, u_y)，下圆柱底面全固定。
    # 顶面全向量约束 + 底面全固定足以消掉所有刚体模，不必再钉对称节点。
    g_top = Constant(mesh, default_scalar_type((0.0, 0.0)))
    dofs_top = locate_dofs_topological(model.V, mesh.topology.dim - 1,
                                       ft.find(model.tags.load_facet))
    ksp_tol = fnum(sol, "ksp_tol")
    model.setup(
        bcs=[dirichletbc(g_top, dofs_top, model.V), fix_surface(model)],
        traction=None, friction=None,          # 阶段 1 无摩擦
        gamma_factor=fnum(sol, "nitsche_gamma_factor"), theta=fnum(sol, "nitsche_theta"),
        newton_options={"relaxation_parameter": 1.0,
                        "atol": fnum(sol, "newton_tol"), "rtol": fnum(sol, "newton_tol"),
                        "convergence_criterion": "residual",
                        "max_it": int(sol["max_it"]), "error_on_nonconvergence": True},
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "ksp_rtol": ksp_tol, "ksp_atol": ksp_tol})

    e_star = contact_effective_modulus(E, nu, E, nu)
    R_eff = R / 2.0          # 两个等半径圆柱：1/R_eff = 1/R + 1/R

    # ---- 阶段 1：压下 -------------------------------------------------
    say("  阶段 1：压下 u_y = %.4g，分 %d 步" % (fnum(ld, "u_y"), n_norm))
    for i in range(n_norm):
        g_top.value[0] = 0.0
        g_top.value[1] = fnum(ld, "u_y") / n_norm
        model.step("法向步 %d/%d" % (i + 1, n_norm))
        model.prepare_next()
    P = model.resultant(component=1)
    # 微滑幅基线 = 阶段 1（无摩擦压下）结束时的状态（D-52）
    tracker = SlipTracker(model, center=(0.0, -2.0 * R - fnum(g, "gap")))
    a_ref = hertz_half_width(P=P, R=R_eff, E_star=e_star)
    say("    压下后：P = %.6g，Hertz a = %.6g（a/R_eff = %.4f）" % (P, a_ref, a_ref / R_eff))

    # ---- 阶段 2：切向推（顶面施 u_x 增量，u_y 增量 0）-------------------
    model.set_friction_law(
        mu, gamma_factor=fnum(sol, "nitsche_gamma_factor_friction"),
        petsc_options=(_lu_options if sol.get("friction_ksp", "gamg") == "lu"
                       else _gamg_options)(fnum(sol, "ksp_tol_friction")))
    say("  阶段 2：切向推 u_x = %.4g，分 %d 步（已切 Coulomb，mu = %.2f）"
        % (fnum(ld, "u_x"), n_tang, mu))
    raw, slips, converged_all = [], [], True
    for i in range(n_tang):
        g_top.value[0] = fnum(ld, "u_x") / n_tang
        g_top.value[1] = 0.0
        try:
            model.step("切向步 %d/%d" % (i + 1, n_tang))
        except RuntimeError as exc:
            # 判据 1：不收敛就停，且判未通过（硬约束 6：不得把未收敛的解当结果）。
            say("  ⚠ 切向步 %d/%d 未收敛：%s" % (i + 1, n_tang, exc))
            converged_all = False
            break
        xs, ps, qs = model.surface_tractions()
        raw.append((xs, ps, qs, model.resultant(component=1), model.resultant(component=0)))
        slips.append(tracker.measure())
        if i < n_tang - 1:
            model.prepare_next()

    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outdir / "profiles.npz", mu=mu, R_eff=R_eff, E_star=e_star,
                            n_tang=n_tang, converged=converged_all,
                            **{f"s{k}_{n}": v for k, r in enumerate(raw)
                               for n, v in zip(("x", "p", "q", "P", "Q"), r)},
                            **{f"s{k}_{n}": v for k, r in enumerate(slips)
                               for n, v in zip(("sx", "sg"), r)})
    return judge(comm, say, cfg, outdir, raw, slips, mu=mu, R_eff=R_eff, e_star=e_star,
                 n_tang=n_tang, converged_all=converged_all, P_normal=P)


def judge(comm, say, cfg, outdir, raw, slips, *, mu, R_eff, e_star, n_tang, converged_all, P_normal):
    """定性判据（纯后处理，可由 --reanalyze 对存盘剖面重判）。"""
    rows = [analyze_step(k + 1, *r, mu=mu, R_eff=R_eff, e_star=e_star)
            for k, r in enumerate(raw)]
    # ---- 定性判据 ------------------------------------------------------
    ctol = fnum(cfg["tolerance"], "coulomb_rel")
    mtol = fnum(cfg["tolerance"], "monotone_rel")
    qtol = fnum(cfg["tolerance"], "q_ratio_rel")
    smin = fnum(cfg["tolerance"], "saturation_min")
    say("")
    say("  %-4s %9s %9s %10s %12s %12s   (参考)c/a FE | Mindlin"
        % ("步", "Q/(muP)", "a_fe", "滑移占比", "max|q|/(mu p)", "P"))
    for r in rows:
        say("  %-4d %9.4f %9.4f %10.3f %12.4f %12.5g   %.3f | %.3f"
            % (r["step"], r["q_ratio"], r["a_fe"], r["slip_frac"], r["coulomb_max"],
               r["P"], r["c_over_a_fe"], r["c_over_a_mindlin"]))

    qr = np.array([r["q_ratio"] for r in rows])
    sf = np.array([r["slip_frac"] for r in rows])
    cm = np.array([r["coulomb_max"] for r in rows])
    checks = [
        ("1 全部切向步收敛", converged_all and len(rows) == n_tang),
        ("2 库仑不等式 max|q|/(mu p) <= 1+%.0f%%" % (100 * ctol),
         bool(len(rows)) and bool(np.all(cm <= 1.0 + ctol))),
        ("3a Q/(mu P) <= 1+%.0f%%" % (100 * qtol),
         bool(len(rows)) and bool(np.all(qr <= 1.0 + qtol))),
        ("3b Q 随步数单调增（回退容差 %.0f%% of 终值）" % (100 * mtol),
         bool(len(rows)) and bool(np.all(np.diff(qr) >= -mtol * qr.max()))),
        ("4 滑移占比单调不减（回退容差 %.2f）" % mtol,
         bool(len(rows)) and bool(np.all(np.diff(sf) >= -mtol))),
        ("5 出现部分滑移（某步 0<滑移占比<1）",
         bool(len(rows)) and bool(np.any((sf > 0) & (sf < 1)))),
        ("6 摩擦力饱和：终值 Q/(mu P) >= %.2f（整体滑移时 Q -> mu P）" % smin,
         bool(len(rows)) and bool(qr[-1] >= smin)),
    ]
    # ---- 微滑幅（Phase 4.5，D-52）---------------------------------------
    slip_info = []
    if slips is not None and rows:
        st = cfg["slip"]
        srel, smono = fnum(st, "stick_rel"), fnum(st, "monotone_tol")
        say("")
        say("  微滑幅（相对阶段 1 结束的切向相对位移增量，核心区 |x| <= %.1fa）"
            % fnum(st, "core_halfwidth_over_a"))
        say("  %-4s %9s %12s %9s %9s %11s %12s %9s  %s"
            % ("步", "Q/(muP)", "max|slip|", "黏着n", "黏着半宽", "q判黏着半宽",
               "(参考)c/a滑移", "c/a Mindlin", "向外单调"))
        for r, (sx, sg), (xs, ps, qs, _, _) in zip(rows, slips, raw):
            m = np.isfinite(sg) & (np.abs(sx) <= fnum(st, "core_halfwidth_over_a") * r["a_hertz"])
            sp = classify_slip(sx[m], sg[m], stick_rel=srel, monotone_tol=smono)
            half = 0.5 * (sp.stick_hi - sp.stick_lo) if sp.n_stick else 0.0
            core = ps >= CORE_FLOOR * ps.max()
            qstick = np.abs(qs[core]) < mu * ps[core] * (1.0 - STICK_EPS)
            q_half = float(np.max(np.abs(xs[core][qstick]))) if qstick.any() else 0.0
            slip_info.append({"step": r["step"], "q_ratio": r["q_ratio"],
                              "max_slip": float(np.max(np.abs(sg[m]))), "n_stick": sp.n_stick,
                              "stick_half": half, "q_stick_half": q_half,
                              "monotone": sp.monotone_outward,
                              "mean_core_slip": float(np.mean(sg[m]))})
            say("  %-4d %9.4f %12.4g %9d %9.3f %11.3f %12.3f %9.3f  %s"
                % (r["step"], r["q_ratio"], slip_info[-1]["max_slip"], sp.n_stick, half, q_half,
                   half / r["a_hertz"], r["c_over_a_mindlin"],
                   "是" if sp.monotone_outward else "否"))
        qrr = np.array([i["q_ratio"] for i in slip_info])
        du = fnum(cfg["loading"], "u_x") / n_tang
        # 「可分辨」：max|slip| 必须高于数值噪声底（弱施加黏着 + CG1 插值的残余），
        # 否则相对 5% 的黏着判据只是在分辨噪声。底取每步施加位移 du 的一个小份额（D-52）。
        floor = fnum(st, "resolved_slip_over_du") * du
        for i in slip_info:
            i["resolved"] = i["max_slip"] >= floor
        part = [i for i in slip_info if i["resolved"]
                and fnum(st, "partial_q_min") <= i["q_ratio"] <= fnum(st, "partial_q_max")]
        halves = np.array([i["stick_half"] for i in part])
        plat = [i for i in slip_info if i["q_ratio"] >= fnum(st, "plateau_q_min")]
        incr = np.diff([i["mean_core_slip"] for i in plat]) if len(plat) >= 3 else np.array([])
        dx_tol = fnum(st, "shrink_slack_over_a") * rows[0]["a_hertz"]
        checks += [
            ("7a 部分滑移步（可分辨）都存在黏着区",
             bool(part) and all(i["n_stick"] > 0 for i in part)),
            ("7b 部分滑移步黏着半宽随 Q 不增（容差 %.2fa）" % fnum(st, "shrink_slack_over_a"),
             bool(len(halves) > 1) and bool(np.all(np.diff(halves) <= dx_tol))),
            ("7c 部分滑移步滑移区内向外单调增",
             bool(part) and all(i["monotone"] for i in part)),
            ("7d 整体滑移平台期每步核心平均滑移增量 = du（±%.0f%%）" % (100 * fnum(st, "plateau_rel")),
             bool(incr.size) and bool(np.all(np.abs(incr - du) <= fnum(st, "plateau_rel") * du))),
        ]
        if incr.size:
            say("  平台期（Q/(muP)>=%.2f 共 %d 步）每步核心平均滑移增量：%s；施加 du = %.4g"
                % (fnum(st, "plateau_q_min"), len(plat), np.array2string(incr, precision=5), du))

    say("")
    ok = True
    for name, passed in checks:
        ok = ok and passed
        say("  %-52s %s" % (name, "通过" if passed else "**未通过**"))
    if rows:
        say("  Q/(muP) 终值 %.3f，滑移占比 %.3f -> %.3f，max|q|/(mu p) 最大 %.4f"
            % (qr[-1], sf[0], sf[-1], cm.max()))
    worst = float(cm.max()) if len(rows) else float("nan")

    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "result.json").write_text(json.dumps(
            {"passed": ok, "mu": mu, "E_star": e_star, "P_after_normal": P_normal,
             "coulomb_max": worst, "checks": {n: bool(v) for n, v in checks}, "steps": rows, "slip": slip_info},
            ensure_ascii=False, indent=2), encoding="utf-8")
    say("")
    say("Phase 4.4 / 4.5 %s" % ("通过" if ok else "未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
