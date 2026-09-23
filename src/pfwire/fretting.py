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

from . import checkpoint
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


def _json_number(value: float) -> float | None:
    """NaN -> None。裸 NaN 不是合法 JSON。"""
    return None if value != value else float(value)


def _gate_report(
    history, contact: pfloads.ContactLoads, *, judge_depth_over_a: float = 2.0,
    initial_tilt_depth_over_a: float = 0.5,
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
    depth0 = with_angle[0]["depth"] if with_angle else None

    # 判据 3 必须用**尖端局部方向**，不能用表层窗口的量：
    # inclination_deg 的窗口固定在 min(inclination_depth, depth)，裂纹深过窗口后
    # 它就一直在量最初那一段，按定义反映不了当前扩展方向（DECISIONS.md OPEN-2）。
    with_tip = [h for h in history if h.get("tip_direction_deg") is not None]

    # **基准角取在固定深度，不取「第一个能算出角度的点」**（D-37）。
    # 后者取决于哪个点碰巧先够 3 个采样行：σ_a=35 的第一点在 2 µm 给出 −16.7°，
    # σ_a=27 的第一点在 1 µm（裂纹才 2 行深）拟合出 −8e−14 ≈ 0 —— 于是
    # 「初始倾斜」被判 False，「转向」又拿 0° 当基准，两条判据同时失真。
    #
    # 深度取 0.5a 的依据是实测而非拍脑袋：三个工况在相同深度处的尖端角逐点相同
    # （0.25a → −6.1°、0.5a → −20.4°、0.75a → −25.3°、1.0a → −25.1°），
    # 说明稳定段的裂纹路径由接触场决定、与 σ_a 无关。0.25a 仍在上升段，
    # 0.75a 起进入 −25° 平台但 σ_a=45 那档（最深 0.56a）够不着 —— 0.5a 是
    # 「已明确倾斜、且所有出现裂纹的工况都覆盖得到」的位置。
    base = next((h for h in with_tip
                 if h["depth"] >= initial_tilt_depth_over_a * a), None)
    tip0 = base["tip_direction_deg"] if base else None
    tip_last = with_tip[-1]["tip_direction_deg"] if with_tip else None
    tilted = tip0 is not None and abs(tip0) > 5.0
    turned = (
        tip0 is not None and tip_last is not None
        and abs(tip_last) < abs(tip0) - 2.0   # 明显更接近竖直
    )
    # 最斜处只作参考，**不参与 passed** —— 「自最大值回落」天然容易满足，
    # 拿它当判据会把判据调松。
    tip_max = max(with_tip, key=lambda h: abs(h["tip_direction_deg"]), default=None)
    incl0 = with_angle[0]["inclination_deg"] if with_angle else None
    # 表层口径同时保留，便于对照（它**不**参与 passed）
    later = [h for h in with_angle if depth0 is not None and h["depth"] > depth0]
    incl_last = later[-1]["inclination_deg"] if later else None

    max_depth = max((h["depth"] for h in history), default=0.0)
    deep_enough = max_depth >= judge_depth_over_a * a
    if not deep_enough:
        # 还没长到该看转向的深度就结束了 —— 这是「没看到」，不是「证伪」，
        # **也不能算通过**。stage I 斜向段本就延伸到接触半宽量级，
        # 之后才在本体应力主导下转 mode I。
        #
        # 深度检查必须排在 pass 之前。曾经把 pass 放在最前面，于是一条只有
        # 2.5 µm（0.215a）的裂纹被判「关卡通过」—— 那个深度比 tip_window（5 µm）
        # 还浅，尖端窗口覆盖了整条裂纹，所谓「尖端方向」其实就是表层方向，
        # 「转向」二字无从谈起。
        verdict = "inconclusive"
    elif in_slip and tilted and turned:
        verdict = "pass"
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
        "tip_direction_initial_depth_over_a": initial_tilt_depth_over_a,
        "tip_direction_max_deg": (
            None if tip_max is None else tip_max["tip_direction_deg"]),
        "tip_direction_max_depth_over_a": (
            None if tip_max is None else tip_max["depth"] / a),
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


#: 失稳判别时最多推多少次交错迭代。墙上实测约 2700 次走完 13.5 -> 81.5 µm，
#: 每次约 0.7 s，故 4000 次约 45 min —— 相对动辄数小时的算例可以接受。
RUNAWAY_MAX_ITER = 4000


class SnapshotWriter:
    """把采样网格上的 d 场按间隔存成 .npz，供论文图 4（d 场演化）使用。

    **不存 dolfinx 场**：`DamageGrid.sample()` 给出的是规则网格上的
    (n_depth, n_x) 数组，正好覆盖裂纹区，直接能画等值图，体积也小
    （141×241 的 float32 约 136 KB/帧）。存整个 .bp 既大又要另写读取端。

    注意与 checkpoint 的区别：checkpoint 是**续跑用的状态**、两槽轮换只留最后两个；
    快照是**可视化用的时间序列**、只增不删。两者当初共用 `save_fields_every`
    是个误接（D-31），这里分开成 `output.snapshot_every`。
    """

    def __init__(self, path, x_coords, depth_coords, every: int):
        self.path = Path(path)
        self.x = np.asarray(x_coords)
        self.depth = np.asarray(depth_coords)
        self.every = int(every)
        self.frames: list[np.ndarray] = []
        self.meta: list[dict[str, float]] = []

    @property
    def enabled(self) -> bool:
        return self.every > 0

    def maybe_capture(self, index: int, values, **meta) -> bool:
        """``index`` 是第几次采样；每 ``every`` 次存一帧。"""
        if not self.enabled or index % self.every != 0:
            return False
        self.capture(values, **meta)
        return True

    def capture(self, values, **meta) -> None:
        self.frames.append(np.asarray(values, dtype=np.float32))
        self.meta.append({k: float(v) for k, v in meta.items()})

    def save(self, comm) -> None:
        if comm.rank != 0 or not self.frames:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for m in self.meta for k in m})
        np.savez_compressed(
            self.path,
            # 坐标叫 x_coords/depth_coords，不叫 x/depth —— 每帧元数据里也有个
            # depth（当时的裂纹深度），同名会让 savez 报「多个同名关键字」。
            d=np.stack(self.frames), x_coords=self.x, depth_coords=self.depth,
            **{k: np.array([m.get(k, np.nan) for m in self.meta]) for k in keys},
        )


def probe_runaway(
    driver, solver, state, set_phase, measure, *,
    failure_depth: float, judge_depth: float,
    max_iter: int = RUNAWAY_MAX_ITER, log=lambda *a: None,
    snapshots: SnapshotWriter | None = None, snapshot_every: int = 100,
    sample_values=None,
) -> dict[str, Any]:
    """交错求解硬失败后，判别这是**失稳扩展**还是求解器问题。

    做法（与 data/_does_it_run.py 手工诊断一致）：再做一个周期的 ᾱ 累积，
    然后**只推交错迭代、不判收敛**，跟踪裂纹深度。

    - 深度越过 ``judge_depth`` 而残差始终不降 -> ``unstable``：
      起点与终点之间不存在平衡解，裂纹在此处失稳扩展，N_f 就落在这一步；
    - 迭代自己收敛了 -> ``converged``：之前只是爬得慢，不是失稳；
    - 子求解器崩了但深度已越过 ``judge_depth`` -> 仍记 ``unstable``
      （韧带贯穿后刚度趋于奇异，这是最终断裂的数值表现）；
    - 其余 -> ``inconclusive``，按硬约束 6 如实报告，不猜。

    ⚠️ **会改变 solver / state**，只能在算例已经结束、结果已记录之后调用。
    """
    delta_alpha, peak_phase = driver.explicit_cycle()
    state.extrapolate(0.0, delta_alpha)      # ΔN = 1，不外推
    set_phase(peak_phase)

    omega = float(solver.config.relaxation)
    tol = float(solver.config.tol_d)
    n_owned = solver.V_d.dofmap.index_map.size_local
    prev = solver.d.x.array.copy()
    trace: list[dict[str, float]] = []
    start = measure().depth
    verdict, note = "inconclusive", ""

    for k in range(1, max_iter + 1):
        try:
            solver.solve_displacement()
            solver.solve_damage()
        except RuntimeError as exc:
            m = measure()
            trace.append({"iter": k, "residual": None, "depth": m.depth})
            if m.depth >= judge_depth:
                verdict = "unstable"
                note = ("子求解器在深度 %.2f µm 处崩溃（%s）；此前深度已单调越过 "
                        "%.2f µm，判为失稳扩展" % (m.depth * 1e6, str(exc)[:60],
                                                  judge_depth * 1e6))
            else:
                note = "子求解器在深度 %.2f µm 处崩溃，尚未越过判定深度：%s" % (
                    m.depth * 1e6, str(exc)[:80])
            break
        if omega < 1.0:
            solver.d.x.array[:] = prev + omega * (solver.d.x.array - prev)
            solver.d.x.scatter_forward()
        local = float(np.max(np.abs(
            solver.d.x.array[:n_owned] - prev[:n_owned]), initial=0.0))
        residual = solver.comm.allreduce(local, op=MPI.MAX) / omega
        prev = solver.d.x.array.copy()

        if residual < tol:
            m = measure()
            trace.append({"iter": k, "residual": residual, "depth": m.depth})
            verdict = "converged"
            note = "推到第 %d 次迭代自己收敛了，深度 %.2f µm —— 不是失稳，是爬得慢" % (
                k, m.depth * 1e6)
            break
        if k % 50 == 0 or k <= 3:
            m = measure()
            trace.append({"iter": k, "residual": residual, "depth": m.depth})
            # 图 4 的「转向」「贯穿」两帧只存在于失控段 —— 收敛解里没有，
            # 所以快照必须也在判别器里取。
            if (snapshots is not None and snapshots.enabled
                    and sample_values is not None and k % snapshot_every == 0):
                snapshots.capture(sample_values(), N=float("nan"), depth=m.depth,
                                  phase=2.0, iteration=float(k))
            if m.depth >= failure_depth:
                verdict = "unstable"
                note = "深度在无平衡解的情况下越过失效判据 %.0f µm" % (
                    failure_depth * 1e6)
                break
            if m.depth >= judge_depth and residual > tol:
                verdict = "unstable"
                note = ("深度从 %.2f µm 单调推进到 %.2f µm（越过 %.2f µm）"
                        "而残差始终不降，其间不存在平衡解"
                        % (start * 1e6, m.depth * 1e6, judge_depth * 1e6))
                break
    else:
        m = measure()
        note = "推满 %d 次迭代仍未定论，深度 %.2f µm" % (max_iter, m.depth * 1e6)

    log("info", "失稳判别：%s —— %s", verdict, note)
    return {"verdict": verdict, "note": note, "depth_start": start,
            "depth_end": trace[-1]["depth"] if trace else start,
            "n_iterations": trace[-1]["iter"] if trace else 0,
            "trace": trace}


def run_fretting(
    cfg, comm: MPI.Comm = MPI.COMM_WORLD, *, resume: bool = False
) -> dict[str, Any]:
    """跑一个微动疲劳工况，返回 N_f、裂纹几何与 Phase 2.4 的关卡报告。

    ``resume=True`` 时从 ``output_dir`` 下的断点接着跑（硬约束 5）。
    断点里存的是**网格本身**，不是重新生成 —— adios4dolfinx 按 input 编号存值，
    重建的网格顺序不同会静默读出乱序的值（见 pfwire/checkpoint.py）。
    """
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
    resumed = None
    if resume:
        resumed = checkpoint.resume(cfg.output_dir, comm)
        msh = resumed.mesh
        tags = resumed.read_facet_tags()
        log("info", "从断点续跑：%s（N=%.6g，已走 %d 步）",
            resumed.path.name, resumed.meta["N"], resumed.meta["n_jumps"])
    else:
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
    #: 本体应力的拉伸平均值。默认 0（D-5 的简化：R = −1 全反复）。
    #: 加它之前必须先验起裂门槛裕度 —— psi+ 由**峰值总应力** sigma_m + sigma_a
    #: 决定，而门槛在峰值约 69-72 MPa 处就被跨过（D-43 实测）。
    sigma_m = float(lcfg.get("sigma_m", 0.0))
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
        # sigma_m 只加在**本体应力**上，切向接触面力仍是全反复（∝ phase）。
        # 这对应真实工况：导线在 EDS 张力下，铝丝带恒定拉伸平均应力，
        # 而微风振动在交叉点产生往复的切向力。D-5 原本把 sigma_m 简化掉了，
        # 这里把它做成可选项，用于与 CIGRE 口径（试验在 EDS 20% RTS 下做）对照。
        sigma_b.value = sigma_m + sigma_a * value

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
    initial_tilt_depth = float(fcfg.get("initial_tilt_depth_over_a", 0.5))

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
    save_every = int(cfg.save_fields_every)

    history: list[dict[str, Any]] = []
    n_jumps_before = 0
    if resumed is not None:
        resumed.read_fields(solver, state)
        driver.N = float(resumed.meta["N"])
        driver.delta_N = float(resumed.meta["delta_N"])
        history.extend(resumed.meta.get("history", []))
        n_jumps_before = int(resumed.meta["n_jumps"])
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
            history, contact, judge_depth_over_a=judge_depth,
            initial_tilt_depth_over_a=initial_tilt_depth
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

    snap_every = int(raw.get("output", {}).get("snapshot_every", 0))
    snapshots = SnapshotWriter(
        cfg.output_dir / "snapshots.npz", grid.x_coords, grid.depth_coords,
        snap_every)

    def sample_values():
        return grid.sample(solver.d)

    def measure():
        """当前 d 场的裂纹几何。record 与失稳判别共用同一套口径。"""
        return extract_crack(
            sample_values(), grid.x_coords, grid.depth_coords,
            threshold=criterion.d_threshold, inclination_depth=incl_depth,
            tip_window=tip_win,
        )

    def record(report):
        t_rec = time.perf_counter()
        values = sample_values()
        metrics = extract_crack(
            values, grid.x_coords, grid.depth_coords,
            threshold=criterion.d_threshold, inclination_depth=incl_depth,
            tip_window=tip_win,
        )
        timers["crack"] += time.perf_counter() - t_rec
        snapshots.maybe_capture(len(history), values, N=report.N,
                                depth=metrics.depth, phase=1.0, iteration=0.0)
        history.append({
            "N": report.N, "delta_N": report.delta_N,
            "max_d": report.max_d, "max_alpha_bar": report.max_alpha_bar,
            # 失败的那一步 max_dd 是 NaN；裸 NaN 不是合法 JSON
            # （Python 读得回来，JSON.parse 之类会炸），统一写成 null。
            "max_dd": _json_number(report.max_dd),
            "rejections": report.rejections,
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
            write_result(make_result("running", n_jumps_before + len(driver.reports),
                                     time.perf_counter() - t_solve))
        if save_every > 0 and len(history) % save_every == 0:
            t_ck = time.perf_counter()
            checkpoint.save(
                cfg.output_dir, solver, state,
                {"N": driver.N, "delta_N": driver.delta_N,
                 "n_jumps": n_jumps_before + len(driver.reports),
                 "history": history},
                comm, facet_tags=tags,
            )
            log("info", "存断点：N=%.6g（%.1f s）", driver.N,
                time.perf_counter() - t_ck)
        return criterion.is_failed(metrics)

    try:
        reports = driver.run(stop=stop)
    finally:
        # **退出时一定存一次断点**，无论是跑完、runout 还是求解失败。
        # 只按「每 save_every 步」存的话，算例死在两个倍数之间就白跑
        # —— 而失败恰恰是最需要断点的时刻（D-30 那次 6 小时就是这么丢的）。
        if save_every > 0:
            try:
                checkpoint.save(
                    cfg.output_dir, solver, state,
                    {"N": driver.N, "delta_N": driver.delta_N,
                     "n_jumps": n_jumps_before + len(driver.reports),
                     "history": history},
                    comm, facet_tags=tags,
                )
                log("info", "退出前存断点：N=%.6g", driver.N)
            except Exception as exc:          # 存断点失败不该盖掉真正的错误
                log("error", "退出前存断点失败：%s", exc)
    t_solve = time.perf_counter() - t_solve

    final = reports[-1] if reports else None
    runaway: dict[str, Any] | None = None
    if final is not None and not final.accepted:
        log("error", "cycle-jump 在 N=%.5g 失败：%s", final.N, final.failure_mode)
        # 硬失败不等于「算例废了」：也可能是裂纹在此处**失稳扩展**，
        # 即起点与终点之间根本没有平衡解可收敛（D-32）。判别一次再下结论。
        log("info", "开始失稳判别（最多 %d 次迭代）……", RUNAWAY_MAX_ITER)
        runaway = probe_runaway(
            driver, solver, state, set_phase, measure,
            failure_depth=criterion.depth,
            judge_depth=judge_depth * contact.a,
            log=log, snapshots=snapshots, sample_values=sample_values,
        )
        status = "failed_unstable" if runaway["verdict"] == "unstable" else "not_converged"
    elif final is not None and final.runout:
        status = "runout"
    elif history and history[-1]["depth"] >= criterion.depth:
        status = "failed"      # 达到失效判据 -> N_f 有效
    else:
        status = "completed"

    snapshots.save(comm)
    if snapshots.enabled:
        log("info", "快照 %d 帧 -> %s", len(snapshots.frames), snapshots.path.name)
    result = make_result(status, n_jumps_before + len(reports), t_solve)
    if runaway is not None:
        result["runaway"] = runaway
    # failed = 稳定扩展到 100 µm；failed_unstable = 在此处失稳扩展，
    # 两者的 N_f 都有效（后者的裂纹在同一个周期内穿过失效深度，见 D-32）
    result["N_f"] = (result["N_final"]
                     if status in ("failed", "failed_unstable") else None)
    write_result(result)
    return result
