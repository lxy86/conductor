"""Phase 4.4：FE 摩擦接触解与 Cattaneo–Mindlin 部分滑移解对拍（PLAN §4 Phase 4.4）。

几何：**圆柱–圆柱**（上游 create_quarter_disks_mesh）。这是 Cattaneo–Mindlin
的原生场景，也是上游 demo_friction_cylinders.py 实测切向加载能收敛的那套。
半圆柱压块体（Phase 4.3 Hertz 用的那套）在切向步会进黏滑极限环，压不住。

两阶段、**位移驱动**：
  阶段 1  压下 u_y  -> 建立 Hertz 接触，由接触面反力测得法向力 P；
  阶段 2  切向推 u_x -> 每步测 Q，算 Q/(mu P)，与解析的
          c/a = sqrt(1 - Q/(mu P))（Cattaneo–Mindlin）对拍。

**黏滑边界怎么从 FE 解里测**：滑移区的定义就是切向面力达到摩擦上限，
即 |q(x)| = mu * p(x)。于是
    黏着  <=>  |q| < mu * p * (1 - eps)
取仍处于黏着的最大 |x| 即为 c。这是判据本身，不是拟合。

对拍口径（PLAN Phase 4.4）：c/a 偏差 < 5%，且取**区间内所有点的最大偏差**。

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

from pfwire.contact import ContactModel, build_quarter_disks, fix_surface, QUARTER_DISKS
from pfwire.loads import hertz_half_width, mindlin_stick_half_width
from pfwire.material import contact_effective_modulus
from pfwire.verify_contact_common import fnum, load_config

REPO = pathlib.Path(__file__).resolve().parents[1]

#: 判黏着时留的相对余量：|q| < mu*p*(1-STICK_EPS) 才算黏着。
#: 数值解在黏滑边界附近本就是过渡的，不留余量会把边界系统性地往里推。
STICK_EPS = 0.02
#: 只在压力足够大的点上判黏滑；接触区边缘 p -> 0 时 q/(mu p) 是 0/0。
P_FLOOR = 0.05


def measure_stick_halfwidth(xs, ps, qs, mu: float, a_fe: float) -> float:
    """从 (p, q) 分布测黏着半宽 c。"""
    thresh = P_FLOOR * ps.max()
    live = (ps > thresh) & (np.abs(xs) <= a_fe)
    if not live.any():
        return float("nan")
    stick = live & (np.abs(qs) < mu * ps * (1.0 - STICK_EPS))
    return float(np.max(np.abs(xs[stick]))) if stick.any() else 0.0


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

    say("Phase 4.4：FE 摩擦接触 vs Cattaneo–Mindlin（圆柱–圆柱，mu = %.2f）" % mu)
    mesh, ct, ft = build_quarter_disks(
        outdir, R=R, gap=fnum(g, "gap"),
        res=fnum(m, "res"), order=int(m["order"]), quads=bool(m["quads"]), comm=comm)
    say("  网格：%d 单元" % mesh.topology.index_map(mesh.topology.dim).size_global)

    model = ContactModel(mesh, ct, ft, E=E, nu=nu, order=int(m["order"]), tags=QUARTER_DISKS)

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

    # ---- 阶段 1：压下 -------------------------------------------------
    say("  阶段 1：压下 u_y = %.4g，分 %d 步" % (fnum(ld, "u_y"), n_norm))
    for i in range(n_norm):
        g_top.value[0] = 0.0
        g_top.value[1] = fnum(ld, "u_y") / n_norm
        model.step("法向步 %d/%d" % (i + 1, n_norm))
        model.prepare_next()
    P = model.resultant(component=1)
    a_ref = hertz_half_width(P=P, R=R, E_star=e_star)
    say("    压下后：P = %.6g，Hertz a = %.6g（a/R = %.4f）" % (P, a_ref, a_ref / R))

    # ---- 阶段 2：切向推（**上体整体刚性平移**）-----------------------
    # 为什么改刚性平移：顶面施刚性 ux、底面固定会把上圆柱剪成悬臂，诱发
    # 弯曲型剪应力叠加在接触面上，使 q(x) 左半黏着、右半滑移（非对称），
    # Cattaneo–Mindlin 的对称黏滑分区就测不出来（本次 Phase 4.4 诊断，
    # 左侧 y-约束无效排除了转动）。改成**对上体所有 dof 的 x 分量施同一增量**
    # —— 上体在 x 向只能整体平移、不能弯，这正是 Mindlin 的「刚性切向位移 δ」
    # 标准设定。法向由顶面 y 增量 = 0 维持（上体竖直位置冻结，接触保持）。
    tdim = mesh.topology.dim
    upper_cells = ct.find(model.tags.body_cells[0])          # 上圆柱
    dofs_body_x = locate_dofs_topological(model.V.sub(0), tdim, upper_cells)
    dofs_top_y = locate_dofs_topological(model.V.sub(1), tdim - 1,
                                         ft.find(model.tags.load_facet))
    g_body_x = Constant(mesh, default_scalar_type(0.0))      # 每步的 x 增量
    g_top_y = Constant(mesh, default_scalar_type(0.0))       # 顶面 y 冻结（增量 0）
    bc_body_x = dirichletbc(g_body_x, dofs_body_x, model.V.sub(0))
    bc_top_y = dirichletbc(g_top_y, dofs_top_y, model.V.sub(1))
    # 换掉阶段 1 的 BC 组：上体 x 刚性平移 + 顶面 y 冻结 + 下体底面固定。
    model.bcs[:] = [bc_body_x, bc_top_y, fix_surface(model)]
    n_body = comm.allreduce(len(dofs_body_x), op=MPI.SUM)
    say("  （上体整体刚性平移：%d 个 x-dof 同步，消除悬臂弯曲）" % n_body)

    # 切到 Coulomb，换 GMRES+GAMG，并把 Nitsche 罚参数降下来（上游做法）。
    model.set_friction_law(
        mu, gamma_factor=fnum(sol, "nitsche_gamma_factor_friction"),
        petsc_options=(_lu_options if sol.get("friction_ksp", "gamg") == "lu"
                       else _gamg_options)(fnum(sol, "ksp_tol_friction")))
    say("  阶段 2：切向推 u_x = %.4g，分 %d 步（已切 Coulomb，mu = %.2f）"
        % (fnum(ld, "u_x"), n_tang, mu))
    lo, hi = fnum(ld, "q_ratio_min"), fnum(ld, "q_ratio_max")
    rows = []
    for i in range(n_tang):
        g_body_x.value = fnum(ld, "u_x") / n_tang   # 上体整体 x 增量
        g_top_y.value = 0.0                          # 顶面 y 冻结
        try:
            model.step("切向步 %d/%d" % (i + 1, n_tang))
        except RuntimeError as exc:
            # 切向步一旦不收敛就停止扫描，**用已收敛的点判定**，不硬撑。
            # 硬约束 6：不得把未收敛的解当结果用。
            say("  ⚠ 切向步 %d/%d 未收敛，停止扫描：%s" % (i + 1, n_tang, exc))
            break
        xs, ps, qs = model.surface_tractions()
        Pn = model.resultant(component=1)
        Q = abs(model.resultant(component=0))
        ratio = Q / (mu * Pn) if Pn > 0 else float("nan")
        a_i = hertz_half_width(P=Pn, R=R, E_star=e_star)
        inside = ps > 0.01 * ps.max()
        a_fe = float(np.max(np.abs(xs[inside]))) if inside.any() else 0.0
        c_fe = measure_stick_halfwidth(xs, ps, qs, mu, a_fe)
        if 0.0 <= ratio <= 1.0:
            c_ref = mindlin_stick_half_width(a=a_i, Q_over_mu_P=ratio)
        else:
            c_ref = float("nan")
        rows.append({"step": i + 1, "P": Pn, "Q": Q, "q_ratio": ratio,
                     "a_ref": a_i, "a_fe": a_fe,
                     "c_ref": c_ref, "c_fe": c_fe,
                     "c_over_a_ref": c_ref / a_i if a_i else float("nan"),
                     "c_over_a_fe": c_fe / a_fe if a_fe else float("nan")})
        if i < n_tang - 1:
            model.prepare_next()

    # ---- 对拍 ----------------------------------------------------------
    tol = fnum(cfg["tolerance"], "c_over_a_rel")
    say("")
    say("  %-5s %9s %9s %10s %10s %9s %s"
        % ("步", "Q/(muP)", "a_fe", "c/a 解析", "c/a FE", "相对差", "判定"))
    used, worst, ok = [], 0.0, True
    for r in rows:
        if not (lo <= r["q_ratio"] <= hi) or not np.isfinite(r["c_over_a_ref"]):
            continue
        rel = abs(r["c_over_a_fe"] - r["c_over_a_ref"]) / r["c_over_a_ref"]
        r["rel"] = rel
        passed = rel < tol
        ok = ok and passed
        worst = max(worst, rel)
        used.append(r)
        say("  %-5d %9.4f %9.4f %10.4f %10.4f %8.2f%% %s"
            % (r["step"], r["q_ratio"], r["a_fe"], r["c_over_a_ref"],
               r["c_over_a_fe"], 100 * rel, "通过" if passed else "**未通过**"))
    if not used:
        say("  ⚠ 没有落在 Q/(muP) ∈ [%.2f, %.2f] 的对拍点 —— 调 loading.u_x" % (lo, hi))
        ok = False
    else:
        say("")
        say("  收敛步 %d/%d，其中落在对拍区间的 %d 个" % (len(rows), n_tang, len(used)))
        say("  **最大**相对差 %.2f%%（阈值 %.1f%%）"
            % (100 * worst, 100 * tol))

    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "result.json").write_text(json.dumps(
            {"passed": ok, "mu": mu, "E_star": e_star, "P_after_normal": P,
             "worst_rel": worst, "n_compared": len(used), "steps": rows},
            ensure_ascii=False, indent=2), encoding="utf-8")
    say("")
    say("Phase 4.4 %s" % ("通过" if ok else "未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
