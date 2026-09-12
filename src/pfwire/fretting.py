"""微动疲劳算例驱动（PLAN.md Phase 2.4）。

载荷（PLAN 第 2 节，Nowell–Hills 的常规做法：用解析面力代替接触求解）::

    上表面  |x| <= a:  t = ( q(x)·sin(2πt/T),  −p(x) )     法向压力**不随时间变**
    右边界:            t = ( σ_a·sin(2πt/T),   0       )
    底边 u_y = 0；左边 u_x = 0

p(x)、q(x) 由 :mod:`pfwire.loads` 给出。**接触半宽 a 由 loads 从 (p0, R, E*) 算出
再传进网格**，不在配置里写死 —— 见 DECISIONS.md D-25：网格用的 a 与面力用的 a
必须同源，一旦不一致，「裂纹在 x ≈ ±a 萌生」这条判据就失去意义。

Phase 2.4 的硬关卡
------------------
PLAN 要求**必须复现**三件事：

1. 裂纹在接触边缘 x ≈ ±a 附近（**滑移区** c <= |x| <= a）萌生；
2. 初始倾斜（不是一开始就垂直向下）；
3. 随后转向垂直于本体应力方向（即趋于竖直）。

本模块只**如实测量并报告**这三项（``gate`` 字段），不做任何"凑"的事情：
不满足就把 ``gate.passed`` 记为 False 并原样输出，由人判断（硬约束 6）。

单位：SI。坐标系与 :class:`pfwire.mesh.FrettingSpec` 一致 ——
上表面在 y = 0，深度向下为负；接触中心 x = 0。
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import ufl
import yaml
from dolfinx import default_scalar_type, fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from . import loads as pfloads
from . import mesh as pfmesh
from .cyclejump import CycleJumpConfig, CycleJumpDriver, FatigueState
from .failure import DamageGrid, FailureCriterion, GridSpec, extract_crack
from .material import contact_pair_properties
from .phasefield import PhaseFieldSolver

__all__ = ["run_fretting"]

logger = logging.getLogger(__name__)


def _traction_forms(msh, contact: pfloads.ContactLoads, phase, sigma_b, tags):
    """上表面的 (q·phase, −p) 与右边界的 (σ_b, 0)。

    p、q 写成 x 的 UFL 表达式。``max_value(..., 0)`` 让接触区外自动归零，
    同时兼作 sqrt 的保护 —— 接触区外 1−(x/a)² 是负数，不夹住会出 nan。
    黏着项同理：|x| > c 时自动为 0，于是分段定义不必写 conditional。
    """
    x = ufl.SpatialCoordinate(msh)
    a, c, p0, mu = contact.a, contact.c, contact.p0, contact.mu

    outer = ufl.sqrt(ufl.max_value(1.0 - (x[0] / a) ** 2, 0.0))
    p_expr = p0 * outer
    if c > 0.0:
        inner = ufl.sqrt(ufl.max_value(1.0 - (x[0] / c) ** 2, 0.0))
        q_expr = mu * p0 * (outer - (c / a) * inner)
    else:
        q_expr = mu * p0 * outer

    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)

    def make_work(V):
        v = ufl.TestFunction(V)
        # 法向压力指向体内 -> t_y = −p；切向随相位往复
        top = ufl.inner(
            ufl.as_vector([q_expr * phase, -p_expr]), v
        ) * ds(pfmesh.FRETTING_TAG_TOP)
        right = ufl.inner(
            ufl.as_vector([sigma_b, fem.Constant(msh, default_scalar_type(0.0))]), v
        ) * ds(pfmesh.FRETTING_TAG_RIGHT)
        return top + right

    return make_work


def _make_bcs(msh, tags, fdim):
    """底边 u_y = 0、左边 u_x = 0（PLAN 第 2 节）。

    这两条既提供反力又消掉刚体位移：左边约束 u_x 阻止 x 平移与转动，
    底边约束 u_y 阻止 y 平移。
    """
    bottom = tags.find(pfmesh.FRETTING_TAG_BOTTOM)
    left = tags.find(pfmesh.FRETTING_TAG_LEFT)

    def make(V):
        Vx, Vy = V.sub(0), V.sub(1)
        return [
            fem.dirichletbc(
                default_scalar_type(0.0),
                fem.locate_dofs_topological(Vy, fdim, bottom),
                Vy,
            ),
            fem.dirichletbc(
                default_scalar_type(0.0),
                fem.locate_dofs_topological(Vx, fdim, left),
                Vx,
            ),
        ]

    return make


def _gate_report(
    history, contact: pfloads.ContactLoads, *, judge_depth_over_a: float = 2.0
) -> dict[str, Any]:
    """Phase 2.4 的三条硬关卡：萌生位置、初始倾角、是否转向。

    **只测量与报告，不做判断以外的事**。任何一条不满足都如实标 False。

    ``judge_depth_over_a``：判「转向」所需的最小深度（单位 a）。裂纹没长到这个
    深度就结束的话，verdict 记 ``inconclusive`` 而不是 ``fail``。
    """
    nucleation = next(
        (h for h in history if h["x_surface"] is not None), None
    )
    if nucleation is None:
        return {
            "passed": False,
            "reason": "整个算例没有出现与表面连通的裂纹（无从判断萌生位置）",
        }
    a, c = contact.a, contact.c
    x_init = float(nucleation["x_surface"])
    x_over_a = abs(x_init) / a
    in_slip = (c / a) <= x_over_a <= 1.25  # 滑移区 c<=|x|<=a，留 25% 余量给离散

    # 倾角要用**第一个量得出角度的点**，不是第一个出现表面裂纹的点：
    # 刚萌生时裂纹只有一两行深，extract_crack 因 min_rows_for_angle 返回 None。
    # 早先按后者取，于是明明 −14° 的倾斜被记成「未倾斜」。
    with_angle = [h for h in history if h["inclination_deg"] is not None]
    incl0 = with_angle[0]["inclination_deg"] if with_angle else None
    depth0 = with_angle[0]["depth"] if with_angle else None
    tilted = incl0 is not None and abs(incl0) > 5.0

    # 判据 3 必须用**尖端局部方向**，不能用上面那个表层窗口的量：
    # inclination_deg 的窗口固定在 min(inclination_depth, depth)，裂纹深过窗口后
    # 它就一直在量最初那一段，按定义反映不了当前扩展方向（DECISIONS.md OPEN-2）。
    with_tip = [h for h in history if h.get("tip_direction_deg") is not None]
    tip0 = with_tip[0]["tip_direction_deg"] if with_tip else None
    tip_last = with_tip[-1]["tip_direction_deg"] if with_tip else None
    turned = (
        tip0 is not None and tip_last is not None
        and abs(tip_last) < abs(tip0) - 2.0   # 明显更接近竖直
    )
    # 表层口径同时保留，便于对照（它**不**参与 passed）
    later = [h for h in with_angle if depth0 is not None and h["depth"] > depth0]
    incl_last = later[-1]["inclination_deg"] if later else None

    max_depth = max((h["depth"] for h in history), default=0.0)
    deep_enough = max_depth >= judge_depth_over_a * a
    if in_slip and tilted and turned:
        verdict = "pass"
    elif not deep_enough:
        # 还没长到该看转向的深度就结束了 —— 这是「没看到」，不是「证伪」。
        # stage I 斜向段本就延伸到接触半宽量级，之后才在本体应力主导下转 mode I。
        verdict = "inconclusive"
    else:
        verdict = "fail"

    return {
        "passed": verdict == "pass",
        "verdict": verdict,
        "x_init": x_init,
        "x_init_over_a": x_over_a,
        "slip_zone_over_a": [c / a, 1.0],
        "nucleates_in_slip_zone": bool(in_slip),
        "inclination_initial_deg": incl0,
        "inclination_initial_depth": depth0,
        "inclination_final_deg": incl_last,
        "tip_direction_initial_deg": tip0,
        "tip_direction_final_deg": tip_last,
        "n_points_with_angle": len(with_angle),
        "initially_tilted": bool(tilted),
        "turns_toward_normal": bool(turned),
        "max_depth_over_a": max_depth / a,
        "judge_depth_over_a": judge_depth_over_a,
        "deep_enough_to_judge": bool(deep_enough),
        "note": (
            "三条判据：(1) 萌生于滑移区 c<=|x|<=a；(2) 初始倾斜 |角|>5°（表层窗口）；"
            "(3) 尖端方向转向竖直（|角|至少减小 2°）。三条全中才 passed=True。"
            "若裂纹没长到 judge_depth_over_a×a 就结束，verdict 记 inconclusive "
            "——「没看到」不等于「证伪」。按硬约束 6 停下报告，不得调参凑。"
        ),
    }


def run_fretting(cfg, comm: MPI.Comm = MPI.COMM_WORLD) -> dict[str, Any]:
    """跑一个微动疲劳工况，返回 N_f、裂纹几何与 Phase 2.4 的关卡报告。"""
    raw = cfg.raw
    ccfg = dict(raw.get("contact", {}))
    fcfg = dict(raw.get("failure", {}))
    lcfg = dict(cfg.loading)
    is_root = comm.rank == 0

    def log(level, msg, *args):
        if is_root:
            getattr(logger, level)(msg, *args)

    material = cfg.material
    E_star, R = contact_pair_properties(
        material.elastic.E, material.elastic.nu,
        float(ccfg["R_wire"]), str(ccfg["counterpart"]),
    )
    contact = pfloads.ContactLoads.from_peak_pressure(
        p0=float(ccfg["p0"]), mu=float(ccfg["mu"]),
        Q_over_mu_P=float(ccfg["Q_over_muP"]), E_star=E_star, R=R,
    )

    # --- 网格：a 由 loads 给，**不用 FrettingSpec 的默认值**（D-25）---
    mcfg = dict(cfg.mesh_cfg)
    mcfg.pop("type", None)
    spec = pfmesh.FrettingSpec(
        contact_halfwidth=contact.a,
        **{k: float(v) for k, v in mcfg.items()},
    )
    t0 = time.perf_counter()
    fr = pfmesh.fretting_mesh(spec, comm)
    msh, tags = fr.mesh, fr.facet_tags
    fdim = msh.topology.dim - 1
    n_cells = msh.topology.index_map(msh.topology.dim).size_global
    t_mesh = time.perf_counter() - t0
    log("info",
        "微动基准：p0=%.4g Pa, μ=%.3g, Q/μP=%.3g -> a=%.4f µm, c/a=%.4f；"
        "σ_a=%.4g Pa；网格 %d 单元（%.1f s）",
        contact.p0, contact.mu, float(ccfg["Q_over_muP"]), contact.a * 1e6,
        contact.c / contact.a, float(lcfg["sigma_a"]), n_cells, t_mesh)

    # --- 载荷 ---
    phase = fem.Constant(msh, default_scalar_type(0.0))
    sigma_b = fem.Constant(msh, default_scalar_type(0.0))
    sigma_a = float(lcfg["sigma_a"])
    make_work = _traction_forms(msh, contact, phase, sigma_b, tags)

    alpha_T, form, kappa = material.require_fatigue_params()
    state = FatigueState(msh, cfg.model, alpha_T=alpha_T, form=form, kappa=kappa)
    solver = PhaseFieldSolver(
        msh, cfg.model, bcs_u=_make_bcs(msh, tags, fdim),
        external_work=make_work, f_alpha=state.f_alpha, config=cfg.solver,
    )
    state.bind_solver(solver)

    def set_phase(t_over_T: float) -> None:
        value = math.sin(2.0 * math.pi * float(t_over_T))
        phase.value = value
        sigma_b.value = sigma_a * value

    # --- 裂纹采样网格（表面 y=0，深度向下）---
    half = float(fcfg["sample_halfwidth_factor"]) * contact.a
    grid_spec = GridSpec(
        x_min=-half, x_max=half,
        depth_max=float(fcfg["sample_depth"]),
        spacing=float(fcfg["sample_spacing"]),
        y_surface=0.0,
    )
    grid = DamageGrid(msh, grid_spec, h_fine=spec.h_fine)
    criterion = FailureCriterion(
        d_threshold=float(fcfg["d_threshold"]), depth=float(fcfg["depth"])
    )
    incl_depth = float(fcfg["inclination_depth"])
    tip_win = float(fcfg.get("tip_window", incl_depth))
    judge_depth = float(fcfg.get("judge_depth_over_a", 2.0))

    # --- cycle-jump ---
    c = dict(lcfg["cycle"])
    jump_cfg = CycleJumpConfig(
        n_sub=int(c["n_sub"]), dN_init=float(c["dN_init"]),
        dN_min=float(c["dN_min"]), dN_max=float(c["dN_max"]),
        dd_max=float(c["dd_max"]), grow_below=float(c["grow_below"]),
        factor=float(c["factor"]), N_max=float(c["N_max"]),
        peak_phase=None if c.get("peak_phase") is None else float(c["peak_phase"]),
    )
    driver = CycleJumpDriver(solver, state, set_phase, jump_cfg)
    dump_every = int(lcfg.get("dump_every", 25))

    history: list[dict[str, Any]] = []
    timers = {"crack": 0.0}
    t_solve = time.perf_counter()

    def make_result(status: str, n_jumps: int, seconds: float) -> dict[str, Any]:
        out = {
            "case": cfg.path.name,
            "case_type": "fretting",
            "status": status,
            "material": material.name,
            "model": {
                "split": cfg.model.split, "dissipation": cfg.model.dissipation,
                "k_res": cfg.model.k_res, "ell": material.fracture.ell,
                "Gc": material.fracture.Gc, "sigma_c": material.fracture.sigma_c,
                "alpha_T": alpha_T, "fatigue_form": form, "kappa": kappa,
            },
            "contact": {
                "p0": contact.p0, "mu": contact.mu,
                "Q_over_muP": float(ccfg["Q_over_muP"]),
                "a": contact.a, "c": contact.c, "c_over_a": contact.c / contact.a,
                "E_star": E_star, "R": R,
                "normal_load": contact.normal_load,
                "tangential_load": contact.tangential_load,
            },
            "loading": {"sigma_a": sigma_a},
            "mesh": {"type": "fretting", "n_cells": int(n_cells), **cfg.mesh_cfg},
            "failure_criterion": criterion.describe(),
            "n_jumps": n_jumps,
            "N_final": float(history[-1]["N"]) if history else 0.0,
            "timing_s": {"mesh": t_mesh, "solve": seconds,
                         "crack_extraction": timers["crack"]},
            "history": history,
        }
        if status == "running":
            out["note"] = "运行中的快照，不是最终结果"
        out["gate"] = _gate_report(
            history, contact, judge_depth_over_a=judge_depth
        )
        return out

    def write_result(out) -> None:
        if comm.rank != 0:
            return
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        tmp = cfg.output_dir / "result.json.tmp"
        tmp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cfg.output_dir / "result.json")
        (cfg.output_dir / "config.yaml").write_text(
            yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    def record(report):
        t_rec = time.perf_counter()
        values = grid.sample(solver.d)
        metrics = extract_crack(
            values, grid.x_coords, grid.depth_coords,
            threshold=criterion.d_threshold, inclination_depth=incl_depth,
            tip_window=tip_win,
        )
        timers["crack"] += time.perf_counter() - t_rec
        history.append({
            "N": report.N, "delta_N": report.delta_N,
            "max_d": report.max_d, "max_alpha_bar": report.max_alpha_bar,
            "max_dd": report.max_dd, "rejections": report.rejections,
            "depth": metrics.depth,
            "depth_over_a": metrics.depth / contact.a,
            "x_surface": metrics.x_surface,
            "x_surface_over_a": (
                None if metrics.x_surface is None else metrics.x_surface / contact.a
            ),
            "x_tip": metrics.x_tip,
            "inclination_deg": metrics.inclination_deg,
            "tip_direction_deg": metrics.tip_direction_deg,
            "n_surface_components": metrics.n_surface_components,
            "max_damage_sampled": metrics.max_damage,
        })
        return metrics

    def stop(report) -> bool:
        metrics = record(report)
        last = history[-1]
        if len(history) <= 5 or len(history) % 10 == 0:
            log("info",
                "N=%.5g ΔN=%.4g 深度=%.2f µm(%.2f a) x_init=%s max_d=%.4f "
                "ᾱ=%.4g 已用 %.0f s（取裂纹 %.0f s）",
                report.N, report.delta_N, last["depth"] * 1e6,
                last["depth_over_a"],
                "-" if last["x_surface"] is None
                else "%.2f a" % last["x_surface_over_a"],
                report.max_d, report.max_alpha_bar,
                time.perf_counter() - t_solve, timers["crack"])
        if dump_every > 0 and len(history) % dump_every == 0:
            write_result(make_result("running", len(driver.reports),
                                     time.perf_counter() - t_solve))
        return criterion.is_failed(metrics)

    reports = driver.run(stop=stop)
    t_solve = time.perf_counter() - t_solve

    final = reports[-1] if reports else None
    if final is not None and not final.accepted:
        status = "not_converged"
        log("error", "cycle-jump 在 N=%.5g 失败：%s", final.N, final.failure_mode)
    elif final is not None and final.runout:
        status = "runout"
    elif history and history[-1]["depth"] >= criterion.depth:
        status = "failed"      # 达到失效判据 -> N_f 有效
    else:
        status = "completed"

    result = make_result(status, len(reports), t_solve)
    result["N_f"] = result["N_final"] if status == "failed" else None
    write_result(result)
    return result
