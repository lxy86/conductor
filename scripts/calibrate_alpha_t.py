"""Phase 2.3：用无接触纯拉伸均匀试样标定 alpha_T 与 kappa。

做法（三步，每一步都能独立验证）
--------------------------------
1. **实测每周期累积系数**。均匀试样跑一个显式循环，量出 Δᾱ_cyc，再除以
   ψ⁺_peak 得无量纲系数 ``r``。解析预期：R = −1 且 Amor 分解下 r = 1.5
   （压缩半周期的体积项归 psi−，但**偏量项仍在 psi+**，故半周期再贡献 0.5 倍；
   见 cyclejump.homogeneous_cycles_to_damage 的 docstring）。
   同时量出 ψ⁺_peak 与 σ²/(2E) 的比值 —— 平面应变下应为 (1−ν²) = 0.891，
   而不是一维杆的 1。D-14 的解析预拟合用的是一维式子，这一步就是来纠正它的。

2. **闭式最小二乘重拟合**。对数型 f 的起裂寿命闭式为

       log10 N = log10(α_T/Δᾱ_per_cycle) + (1 − s)/κ − 2 log10 s

   对 (log10 α_T, 1/κ) 是**线性**的，直接最小二乘，不需要迭代跑有限元。
   Δᾱ_per_cycle 用第 1 步实测的系数，于是拟合出来的参数天然带着平面应变修正。

3. **有限元复核**。用拟合值在 4 个应力水平上真跑 cycle-jump，
   比对目标寿命，报 mean |Δlog10 N|（D-3 验收阈值 0.30）。

**为什么不直接对有限元结果做迭代优化**：每次试参数都要跑 4 个算例，
而闭式解已经把 (α_T, κ) 的依赖写成线性形式，先解析拟合再有限元复核既快又可查。
若第 3 步与第 2 步对不上，说明闭式解与有限元有系统偏差 —— 那本身就是要报告的发现，
而不是该靠调参掩盖的东西（硬约束 6）。

用法：python scripts/calibrate_alpha_t.py [configs/calib_tension.yaml]
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import numpy as np
import ufl
import yaml
from dolfinx import default_scalar_type, fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pfwire.cyclejump import (  # noqa: E402
    CycleJumpConfig,
    CycleJumpDriver,
    FatigueState,
)
from pfwire.material import load_material  # noqa: E402
from pfwire.phasefield import (  # noqa: E402
    PhaseFieldModel,
    PhaseFieldSolver,
    SolverConfig,
    energy_split,
    strain,
)


def build_case(cfg: dict, material, sigma_a: float, alpha_T: float, kappa: float):
    """建均匀试样：左边 u_x = 0、下边 u_y = 0，右边施加 σ(t)。

    这样 σ_yy = 0（上边自由），得到**平面应变下的单轴拉伸**应力状态 ——
    与生产工况里本体应力 σ_b 的加法一致。
    """
    comm = MPI.COMM_WORLD
    mcfg = cfg["mesh"]
    W, H, h = float(mcfg["W"]), float(mcfg["H"]), float(mcfg["h"])
    nx, ny = max(1, round(W / h)), max(1, round(H / h))
    msh = dmesh.create_rectangle(
        comm, [np.array([0.0, 0.0]), np.array([W, H])], [nx, ny],
        dmesh.CellType.triangle,
    )
    mo = cfg["model"]
    model = PhaseFieldModel(
        material=material,
        split=str(mo["split"]),
        k_res=float(mo["k_res"]),
        dissipation=str(mo["dissipation"]),
    )
    scfg = cfg["solver"]
    solver_cfg = SolverConfig(
        u_solver=str(scfg["u_solver"]),
        tol_d=float(scfg["tol_d"]),
        max_iter=int(scfg["max_iter"]),
    )

    fdim = msh.topology.dim - 1
    traction = fem.Constant(msh, default_scalar_type(0.0))
    zero = fem.Constant(msh, default_scalar_type(0.0))
    ds = ufl.Measure("ds", domain=msh)

    def left(x):
        return np.isclose(x[0], 0.0)

    def bottom(x):
        return np.isclose(x[1], 0.0)

    def right(x):
        return np.isclose(x[0], W)

    left_facets = dmesh.locate_entities_boundary(msh, fdim, left)
    bottom_facets = dmesh.locate_entities_boundary(msh, fdim, bottom)

    def make_bcs(V):
        # 用**拓扑**定位：locate_dofs_geometrical 不接受子空间
        # （报 "Cannot tabulate coordinates for a FunctionSpace that is a subspace"），
        # 而本项目其它地方（run_case）也一律用 locate_dofs_topological。
        Vx, Vy = V.sub(0), V.sub(1)
        return [
            fem.dirichletbc(
                default_scalar_type(0.0),
                fem.locate_dofs_topological(Vx, fdim, left_facets),
                Vx,
            ),
            fem.dirichletbc(
                default_scalar_type(0.0),
                fem.locate_dofs_topological(Vy, fdim, bottom_facets),
                Vy,
            ),
        ]

    right_facets = dmesh.locate_entities_boundary(msh, fdim, right)
    marker = dmesh.meshtags(
        msh, fdim, np.sort(right_facets),
        np.full(right_facets.size, 1, dtype=np.int32),
    )
    ds_right = ufl.Measure("ds", domain=msh, subdomain_data=marker)(1)

    def make_work(V):
        v = ufl.TestFunction(V)
        return ufl.inner(ufl.as_vector([traction, zero]), v) * ds_right

    # **不从材料卡读 alpha_T**：卡里现在是 null（未标定），而 D-3 规定未标定的
    # alpha_T 不得进入计算。标定脚本本身就是来产生这个值的，故由调用方传入
    # （初值取自 S-N 参考卡的 *_initial_guess）。
    form = material.fatigue_form
    state = FatigueState(msh, model, alpha_T=alpha_T, form=form, kappa=kappa)
    solver = PhaseFieldSolver(
        msh, model, bcs_u=make_bcs, external_work=make_work,
        f_alpha=state.f_alpha, config=solver_cfg,
    )
    state.bind_solver(solver)

    R = float(cfg["loading"]["stress_ratio_R"])

    def set_phase(phase: float) -> None:
        # R = -1：σ(t) = σ_a sin(2π t/T)，一个整周期覆盖拉与压
        traction.value = sigma_a * math.sin(2.0 * math.pi * float(phase))

    psi_plus, _ = energy_split(strain(solver.u), material, model.split, 2)
    psi_form = fem.form(psi_plus * ufl.dx)
    area = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ufl.dx)), op=MPI.SUM
    )
    return dict(
        mesh=msh, solver=solver, state=state, set_phase=set_phase,
        psi_form=psi_form, area=area, comm=comm, R=R, traction=traction,
        sigma_a=sigma_a,
    )


def measure_accumulation(case, cfg) -> dict:
    """第 1 步：实测每周期累积系数 r = Δᾱ_cyc / ψ⁺_peak，以及 ψ⁺_peak/(σ²/2E)。"""
    comm, solver, state = case["comm"], case["solver"], case["state"]
    n_sub = int(cfg["loading"]["cycle"]["n_sub"])

    # 峰值相位（sin 在 1/4 周期取最大）
    case["set_phase"](0.25)
    solver.solve_displacement()
    psi_peak = comm.allreduce(
        fem.assemble_scalar(case["psi_form"]), op=MPI.SUM
    ) / case["area"]

    # 一个完整循环的 ᾱ 累积
    case["set_phase"](0.0)
    solver.solve_displacement()
    state.prime()
    before = state.alpha_bar.x.array.copy()
    for k in range(1, n_sub + 1):
        case["set_phase"](k / n_sub)
        solver.solve_displacement()
        state.accumulate()
    n_owned = state.V.dofmap.index_map.size_local
    d_alpha = comm.allreduce(
        float(np.max((state.alpha_bar.x.array - before)[:n_owned], initial=0.0)),
        op=MPI.MAX,
    )
    # 复位
    state.alpha_bar.x.array[:] = before
    state.alpha_bar.x.scatter_forward()

    E = case["solver"].model.material.elastic.E
    nu = case["solver"].model.material.elastic.nu
    sigma = case["sigma_a"]
    psi_1d = sigma**2 / (2.0 * E)
    return {
        "psi_peak": psi_peak,
        "psi_1d": psi_1d,
        "psi_peak_over_1d": psi_peak / psi_1d,
        "plane_strain_factor": 1.0 - nu**2,
        "delta_alpha_cycle": d_alpha,
        "r": d_alpha / psi_peak if psi_peak > 0 else float("nan"),
    }


def fit_parameters(targets, sigma_c, E, coeff, gamma) -> tuple[float, float, dict]:
    """第 2 步：闭式最小二乘拟合 (alpha_T, kappa)。**平面应变口径**。

    推导（与 homogeneous_cycles_to_damage 同一套，但不假设一维）：
    AT1 的 d 驻定条件 2(1-d)psi+ = f*(3Gc/8l)，而 3Gc/(8l) = sigma_c^2/E，故

        1 - d = f sigma_c^2 / (2 E psi+)

    平面应变单轴拉伸下 psi+_peak = (1-nu^2) sigma^2/(2E)，于是 2E psi+ =
    (1-nu^2) sigma^2，起裂（d 从 0 起动）条件为 **f = (1-nu^2) s^2 = (gamma s)^2**，
    gamma = sqrt(1-nu^2)。一维杆才是 f = s^2。
    对数型 f = [1 - kappa log10(a/a_T)]^2 反解得

        a_needed = a_T * 10^((1 - gamma s)/kappa)

    每周期累积 Delta_a = coeff * sigma^2/(2E)（coeff 由第 1 步实测，已含 r 与
    (1-nu^2)），故 **log10 N = log10 a_T + (1 - gamma s)/kappa - log10(Delta_a)**，
    对 (X = log10 a_T, Y = 1/kappa) 线性，基为 [1, 1 - gamma s]。

    两个曾经写错的地方，留作警示：
    1. D-14 的式子写成 "... + (1-s)/kappa - 2 log10 s"，那里的 Delta_a_coeff
       **不含** sigma^2，sigma^2 的依赖由 -2 log10 s 承担。本函数的 ``d_alpha``
       已经含 sigma^2，**再加 2 log10 s 就是把 sigma^2 算了两遍**；
    2. 阈值必须用 gamma*s（平面应变）而不是 s（一维）。只改累积不改阈值不自洽。
    这两处各自都会让闭式与有限元系统性偏离，且偏离随应力下降而增大 ——
    正是第一次运行时实测到的 -0.5 ~ -1.2 的 Delta log10。
    """
    A, rhs = [], []
    for t in targets:
        sigma = float(t["sigma_a"])
        N = float(t["N_target"])
        s = sigma / sigma_c
        d_alpha = coeff * sigma**2 / (2.0 * E)
        A.append([1.0, 1.0 - gamma * s])
        rhs.append(math.log10(N) + math.log10(d_alpha))
    A = np.array(A)
    rhs = np.array(rhs)
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    log_alpha_T, inv_kappa = sol
    alpha_T = 10.0**log_alpha_T
    kappa = 1.0 / inv_kappa
    # 拟合残差
    pred = A @ sol
    resid = pred - rhs
    return alpha_T, kappa, {
        "residual_log10": resid.tolist(),
        "mean_abs_log10": float(np.mean(np.abs(resid))),
        "max_abs_log10": float(np.max(np.abs(resid))),
    }


def analytic_life(sigma, sigma_c, E, coeff, gamma, alpha_T, kappa) -> float:
    """平面应变口径的起裂寿命闭式（推导见 fit_parameters）。"""
    s = sigma / sigma_c
    d_alpha = coeff * sigma**2 / (2.0 * E)
    log_N = math.log10(alpha_T) - math.log10(d_alpha) + (1.0 - gamma * s) / kappa
    return 10.0**log_N


def run_fe_life(cfg, material, sigma_a, alpha_T, kappa) -> dict:
    """第 3 步：有限元跑到起裂，返回 N_起裂。"""
    case = build_case(cfg, material, sigma_a, alpha_T, kappa)
    lcfg = cfg["loading"]
    c = lcfg["cycle"]
    jump_cfg = CycleJumpConfig(
        n_sub=int(c["n_sub"]), dN_init=float(c["dN_init"]),
        dN_min=float(c["dN_min"]), dN_max=float(c["dN_max"]),
        dd_max=float(c["dd_max"]), grow_below=float(c["grow_below"]),
        factor=float(c["factor"]), N_max=float(c["N_max"]),
        peak_phase=None if c.get("peak_phase") is None else float(c["peak_phase"]),
    )
    driver = CycleJumpDriver(
        case["solver"], case["state"], case["set_phase"], jump_cfg
    )
    d_nuc = float(lcfg["nucleation_d"])
    t0 = time.perf_counter()

    def stop(report):
        return report.max_d >= d_nuc

    reports = driver.run(stop=stop)
    last = reports[-1] if reports else None
    ok = last is not None and last.accepted and last.max_d >= d_nuc
    return {
        "sigma_a": sigma_a,
        "N_nucleation": float(last.N) if last is not None else float("nan"),
        "max_d": float(last.max_d) if last is not None else float("nan"),
        "n_jumps": len(reports),
        "runout": bool(last.runout) if last is not None else False,
        "converged": bool(last.accepted) if last is not None else False,
        "reached_nucleation": bool(ok),
        "seconds": time.perf_counter() - t0,
    }


def main(argv: list[str]) -> int:
    cfg_path = ROOT / (argv[1] if len(argv) > 1 else "configs/calib_tension.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    material = load_material(ROOT / cfg["material"])
    sn = yaml.safe_load((ROOT / cfg["sn_reference"]).read_text(encoding="utf-8"))
    targets = sn["calibration_targets"]
    tol = float(sn["acceptance"]["tolerance"])
    comm = MPI.COMM_WORLD
    root = comm.rank == 0

    E = material.elastic.E
    sigma_c = material.fracture.sigma_c
    if root:
        print("材料 %s：E = %.4g Pa，sigma_c = %.6g Pa，nu = %.3g"
              % (material.name, E, sigma_c, material.elastic.nu))
        print("靶曲线 %s（is_surrogate = %s）" % (sn["name"], sn.get("is_surrogate")))
        print()

    # ---- 第 1 步 ----
    probe_sigma = float(targets[0]["sigma_a"])
    # 第 1 步只量弹性累积量，与 alpha_T/kappa 无关，用初值即可
    guess_aT = float(sn["acceptance"]["alpha_T_initial_guess"])
    guess_k = float(sn["acceptance"]["kappa_initial_guess"])
    case = build_case(cfg, material, probe_sigma, guess_aT, guess_k)
    acc = measure_accumulation(case, cfg)
    if root:
        print("=== 第 1 步：实测每周期累积（σ_a = %.4g MPa）===" % (probe_sigma / 1e6))
        print("  ψ⁺_peak = %.6e J/m³；σ²/(2E) = %.6e；比值 = %.6f"
              % (acc["psi_peak"], acc["psi_1d"], acc["psi_peak_over_1d"]))
        print("  平面应变预期 (1−ν²) = %.6f  ->  差 %.2e"
              % (acc["plane_strain_factor"],
                 abs(acc["psi_peak_over_1d"] - acc["plane_strain_factor"])))
        print("  Δᾱ_cyc = %.6e J/m³；r = Δᾱ_cyc/ψ⁺_peak = %.6f（R=−1 解析预期 1.5）"
              % (acc["delta_alpha_cycle"], acc["r"]))

    # 每周期累积量相对 σ²/(2E) 的总系数
    coeff = acc["r"] * acc["psi_peak_over_1d"]
    if root:
        print("  -> 每周期 Δᾱ = %.6f · σ²/(2E)；起裂阈值 f = (γs)²，γ = %.6f"
              % (coeff, math.sqrt(acc["psi_peak_over_1d"])))
        print()

    # ---- 第 2 步 ----
    gamma = math.sqrt(acc["psi_peak_over_1d"])
    alpha_T, kappa, fit = fit_parameters(targets, sigma_c, E, coeff, gamma)
    if root:
        print("=== 第 2 步：闭式最小二乘 ===")
        print("  alpha_T = %.6e J/m³   kappa = %.6f" % (alpha_T, kappa))
        print("  D-14 一维预拟合值：alpha_T = 3.407e+06, kappa = 0.14888")
        print("  拟合残差 mean|Δlog10 N| = %.4f，max = %.4f"
              % (fit["mean_abs_log10"], fit["max_abs_log10"]))
        print()
        print("  σ_a(MPa)   N_target     N_闭式      Δlog10")
        for t in targets:
            s_a = float(t["sigma_a"])
            N_t = float(t["N_target"])
            N_a = analytic_life(s_a, sigma_c, E, coeff, gamma, alpha_T, kappa)
            print("   %6.1f   %9.3e  %9.3e   %+.4f"
                  % (s_a / 1e6, N_t, N_a, math.log10(N_a / N_t)))
        print()

    # ---- 第 3 步 ----
    if root:
        print("=== 第 3 步：有限元复核 ===")
    results = []
    for t in targets:
        s_a = float(t["sigma_a"])
        res = run_fe_life(cfg, material, s_a, alpha_T, kappa)
        res["N_target"] = float(t["N_target"])
        res["N_analytic"] = analytic_life(s_a, sigma_c, E, coeff, gamma, alpha_T, kappa)
        res["dlog10_vs_target"] = (
            math.log10(res["N_nucleation"] / res["N_target"])
            if res["N_nucleation"] > 0 else float("nan")
        )
        res["dlog10_vs_analytic"] = (
            math.log10(res["N_nucleation"] / res["N_analytic"])
            if res["N_nucleation"] > 0 else float("nan")
        )
        results.append(res)
        if root:
            print("  σ_a = %5.1f MPa: N_FE = %9.3e（目标 %9.3e，Δlog10 %+.4f；"
                  "闭式 %9.3e，Δlog10 %+.4f）%s  %d 步 %.0f s"
                  % (s_a / 1e6, res["N_nucleation"], res["N_target"],
                     res["dlog10_vs_target"], res["N_analytic"],
                     res["dlog10_vs_analytic"],
                     "" if res["reached_nucleation"] else " **未起裂**",
                     res["n_jumps"], res["seconds"]))

    errs = [abs(r["dlog10_vs_target"]) for r in results
            if np.isfinite(r["dlog10_vs_target"])]
    mean_err = float(np.mean(errs)) if errs else float("nan")
    passed = bool(errs) and mean_err <= tol and all(
        r["reached_nucleation"] for r in results
    )
    if root:
        print()
        print("mean |Δlog10 N| = %.4f（D-3 验收阈值 %.2f）-> %s"
              % (mean_err, tol, "通过" if passed else "**未通过**"))

        out_dir = ROOT / cfg["output"]["dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "calibration.json").write_text(
            json.dumps(
                {
                    "material": material.name,
                    "sn_reference": sn["name"],
                    "is_surrogate": bool(sn.get("is_surrogate")),
                    "accumulation": acc,
                    "coeff_delta_alpha_over_psi_1d": coeff,
                    "alpha_T": alpha_T,
                    "kappa": kappa,
                    "closed_form_fit": fit,
                    "fe_results": results,
                    "mean_abs_log10_error": mean_err,
                    "tolerance": tol,
                    "passed": passed,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print("已写 %s" % (out_dir / "calibration.json").relative_to(ROOT))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
