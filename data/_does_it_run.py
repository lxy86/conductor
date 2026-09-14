"""墙上那一步的真解是什么：裂纹只是长一点，还是一路跑到底？

背景：在 N=80012（深 13.5 µm ≈ 1.16a）处，一个周期的疲劳增量之后交错迭代不收敛。
残差历史显示它不是振荡，而是**匀速漂移**（每轮把 d 推进几乎恒定的量，
300 轮从 1.704e-3 微升到 2.020e-3 后平住）。d 上限是 1，所以只能是损伤带
在空间上一个单元一个单元地铺开 —— 也就是裂纹在跑，交错迭代只是爬得太慢。

本脚本不判收敛，只把迭代一直推下去，看裂纹深度march 到哪里：
  - 停在某个深度不动  -> 真解就在附近，纯粹是交错格式爬得慢，加速格式即可；
  - 一路到 100 µm（失效判据）-> 裂纹在此处失稳扩展，N_f 就在这一步，
    失效判据反而是达到了。

用法：mpirun -n 6 python data/_does_it_run.py <工况 yaml> [最大迭代数]
"""

import dataclasses
import sys

import numpy as np
from mpi4py import MPI
from dolfinx import default_scalar_type, fem

from pfwire import checkpoint
from pfwire import loads as pfloads
from pfwire.cyclejump import CycleJumpConfig, CycleJumpDriver, FatigueState
from pfwire.failure import DamageGrid, GridSpec, extract_crack
from pfwire.fretting import _make_bcs, _traction_forms
from pfwire.material import contact_pair_properties
from pfwire.phasefield import PhaseFieldSolver
from pfwire.run_case import load_case

comm = MPI.COMM_WORLD
cfg = load_case(sys.argv[1])
max_it = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
raw = cfg.raw
ccfg, fcfg, lcfg = dict(raw["contact"]), dict(raw["failure"]), dict(cfg.loading)


def say(m, *a):
    if comm.rank == 0:
        print(m % a if a else m, flush=True)


material = cfg.material
E_star, R = contact_pair_properties(
    material.elastic.E, material.elastic.nu,
    float(ccfg["R_wire"]), str(ccfg["counterpart"]))
contact = pfloads.ContactLoads.from_peak_pressure(
    p0=float(ccfg["p0"]), mu=float(ccfg["mu"]),
    Q_over_mu_P=float(ccfg["Q_over_muP"]), E_star=E_star, R=R)

r = checkpoint.resume(cfg.output_dir, comm)
msh, tags = r.mesh, r.read_facet_tags()
fdim = msh.topology.dim - 1

phase = fem.Constant(msh, default_scalar_type(0.0))
sigma_b = fem.Constant(msh, default_scalar_type(0.0))
sigma_a = float(lcfg["sigma_a"])
make_work = _traction_forms(msh, contact, phase, sigma_b, tags)
alpha_T, form, kappa = material.require_fatigue_params()
state = FatigueState(msh, cfg.model, alpha_T=alpha_T, form=form, kappa=kappa)
# u 的 Newton 迭代上限放开：失控段停在 81.5 µm 是 u 求解器先撑不住，
# 不是裂纹停了。这里只为确认失控能否越过 100 µm 判据。
solver_cfg = dataclasses.replace(cfg.solver, snes_max_it=int(sys.argv[3])
                                 if len(sys.argv) > 3 else cfg.solver.snes_max_it)
solver = PhaseFieldSolver(
    msh, cfg.model, bcs_u=_make_bcs(msh, tags, fdim),
    external_work=make_work, f_alpha=state.f_alpha, config=solver_cfg)
state.bind_solver(solver)
r.read_fields(solver, state)


def set_phase(t_over_T: float) -> None:
    import math
    v = math.sin(2.0 * math.pi * float(t_over_T))
    phase.value = v
    sigma_b.value = sigma_a * v


c = dict(lcfg["cycle"])
driver = CycleJumpDriver(solver, state, set_phase, CycleJumpConfig(
    n_sub=int(c["n_sub"]), dN_init=1.0, dN_min=1.0, dN_max=1.0,
    dd_max=float(c["dd_max"]), grow_below=0.0, factor=2.0,
    N_max=float(c["N_max"]), peak_phase=None))

half = float(fcfg["sample_halfwidth_factor"]) * contact.a
grid = DamageGrid(msh, GridSpec(
    x_min=-half, x_max=half, depth_max=float(fcfg["sample_depth"]),
    spacing=float(fcfg["sample_spacing"]), y_surface=0.0))
thr = float(fcfg["d_threshold"])
fail_depth = float(fcfg["depth"])


def metrics_now():
    return extract_crack(grid.sample(solver.d), grid.x_coords,
                         grid.depth_coords, threshold=thr,
                         inclination_depth=float(fcfg["inclination_depth"]),
                         tip_window=float(fcfg.get("tip_window",
                                                   fcfg["inclination_depth"])))


def depth_now():
    return metrics_now().depth


say("u 的 SNES 迭代上限 = %d（默认 %d）", solver_cfg.snes_max_it, cfg.solver.snes_max_it)
say("断点 N=%.6g，深度 %.2f µm (%.3f a)", r.meta["N"], depth_now() * 1e6,
    depth_now() / contact.a)
say("失效判据深度 %.0f µm；采样上限 %.0f µm",
    fail_depth * 1e6, float(fcfg["sample_depth"]) * 1e6)

# 一个真实周期的疲劳累积（与生产步一致），然后**只推迭代、不判收敛**
delta_alpha, peak_phase = driver.explicit_cycle()
state.extrapolate(0.0, delta_alpha)      # ΔN=1 -> 不外推
set_phase(peak_phase)
say("")
say("做完一个周期的 ᾱ 累积，现在只推交错迭代（不判收敛），看裂纹走到哪：")
say("  迭代    ‖Δd‖_inf    深度(µm)  深度/a   尖端角°  x_tip(µm)")

omega = cfg.solver.relaxation
prev = solver.d.x.array.copy()
n = solver.V_d.dofmap.index_map.size_local
for k in range(1, max_it + 1):
    try:
        solver.solve_displacement()
        solver.solve_damage()
    except RuntimeError as exc:
        # 子求解器失败也要先把「走到哪了」记下来 —— 那正是本脚本要的数据
        m = metrics_now()
        say("  %5d  %10s  %8.2f  %6.3f  %7s  %8s  <- 子求解器失败：%s",
            k, "-", m.depth * 1e6, m.depth / contact.a,
            "-" if m.tip_direction_deg is None else "%.1f" % m.tip_direction_deg,
            "-" if m.x_tip is None else "%.2f" % (m.x_tip * 1e6), str(exc)[:60])
        break
    if omega < 1.0:
        solver.d.x.array[:] = prev + omega * (solver.d.x.array - prev)
        solver.d.x.scatter_forward()
    dd = comm.allreduce(float(np.max(np.abs(
        solver.d.x.array[:n] - prev[:n]), initial=0.0)), op=MPI.MAX) / omega
    prev = solver.d.x.array.copy()
    if k <= 3 or k % 50 == 0 or dd < float(cfg.solver.tol_d):
        m = metrics_now()
        dep = m.depth
        say("  %5d  %10.3e  %8.2f  %6.3f  %7s  %8s", k, dd, dep * 1e6,
            dep / contact.a,
            "-" if m.tip_direction_deg is None else "%.1f" % m.tip_direction_deg,
            "-" if m.x_tip is None else "%.2f" % (m.x_tip * 1e6))
        if dd < float(cfg.solver.tol_d):
            say("  -> 收敛了：真解就在附近，纯粹是爬得慢")
            break
        if dep >= fail_depth:
            say("  -> 已达失效判据深度，裂纹在此处失稳扩展")
            break
else:
    say("  -> 推到 %d 轮仍未停，深度 %.2f µm", max_it, depth_now() * 1e6)
