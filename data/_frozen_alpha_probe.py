"""从断点出发，**冻结疲劳累积**，只解静态问题 —— 判断那堵墙是物理失稳还是数值问题。

背景：三次生产算例里有两次死在同一个裂纹深度（13.50 µm ≈ 1.16a），
松弛因子与线搜索都不同，N 只差 0.05%。这更像耦合问题在该处失稳，
而不是求解器参数没调好。

判据：把 alpha_bar 钉死（不再 accumulate），在峰值相位反复解「位移 + 损伤」。
  - 若 d 不再增长  -> 该状态是静态稳定的，墙是**数值**问题（交错格式跟不上），
                     可以换 omega / 换格式去攻；
  - 若 d 继续长大  -> 该状态**静态失稳**：不再加任何疲劳驱动力裂纹也自己走，
                     说明模型预测裂纹在此处失稳扩展，D-1 的 100 µm 判据在本载荷下
                     根本到不了，N_f 的定义要重谈。

用法：mpirun -n 6 python data/_frozen_alpha_probe.py <工况 yaml> [静态迭代次数]
"""

import sys

import numpy as np
from mpi4py import MPI

from pfwire import checkpoint
from pfwire import loads as pfloads
from pfwire.cyclejump import FatigueState
from pfwire.failure import DamageGrid, GridSpec, extract_crack
from pfwire.fretting import _make_bcs, _traction_forms
from pfwire.material import contact_pair_properties
from pfwire.phasefield import PhaseFieldSolver
from pfwire.run_case import load_case
from dolfinx import default_scalar_type, fem

comm = MPI.COMM_WORLD
cfg = load_case(sys.argv[1])
n_static = int(sys.argv[2]) if len(sys.argv) > 2 else 12
raw = cfg.raw
ccfg, fcfg, lcfg = dict(raw["contact"]), dict(raw["failure"]), dict(cfg.loading)


def say(msg, *a):
    if comm.rank == 0:
        print(msg % a if a else msg, flush=True)


material = cfg.material
E_star, R = contact_pair_properties(
    material.elastic.E, material.elastic.nu,
    float(ccfg["R_wire"]), str(ccfg["counterpart"]))
contact = pfloads.ContactLoads.from_peak_pressure(
    p0=float(ccfg["p0"]), mu=float(ccfg["mu"]),
    Q_over_mu_P=float(ccfg["Q_over_muP"]), E_star=E_star, R=R)

# **网格只能从断点读**（D-31）：重建的网格 input 编号顺序不同，
# read_function 会静默读出乱序的值。
r = checkpoint.resume(cfg.output_dir, comm)
msh = r.mesh
tags = r.read_facet_tags()
fdim = msh.topology.dim - 1
say("断点：%s  N=%.6g  已走 %d 步", r.path.name, r.meta["N"], r.meta["n_jumps"])

phase = fem.Constant(msh, default_scalar_type(0.0))
sigma_b = fem.Constant(msh, default_scalar_type(0.0))
sigma_a = float(lcfg["sigma_a"])
make_work = _traction_forms(msh, contact, phase, sigma_b, tags)

alpha_T, form, kappa = material.require_fatigue_params()
state = FatigueState(msh, cfg.model, alpha_T=alpha_T, form=form, kappa=kappa)
solver = PhaseFieldSolver(
    msh, cfg.model, bcs_u=_make_bcs(msh, tags, fdim),
    external_work=make_work, f_alpha=state.f_alpha, config=cfg.solver)
state.bind_solver(solver)
r.read_fields(solver, state)

half = float(fcfg["sample_halfwidth_factor"]) * contact.a
grid = DamageGrid(msh, GridSpec(
    x_min=-half, x_max=half, depth_max=float(fcfg["sample_depth"]),
    spacing=float(fcfg["sample_spacing"]), y_surface=0.0))


def depth_now():
    m = extract_crack(grid.sample(solver.d), grid.x_coords, grid.depth_coords,
                      threshold=float(fcfg["d_threshold"]))
    return m.depth


# 峰值相位：取 |sin| = 1，与 cycle-jump 里 peak_phase 的取法一致
phase.value = 1.0
sigma_b.value = sigma_a

say("起点：max_d=%.6f  深度=%.2f µm(%.3f a)  ᾱ_max=%.6g",
    solver.max_damage(), depth_now() * 1e6, depth_now() / contact.a,
    state.max_alpha_bar())
say("")
say("现在**不再累积 ᾱ**，只反复解静态问题 %d 次：", n_static)
say("  轮次   max_d      深度(µm)   深度/a    ‖Δd‖_inf   交错迭代  收敛")

d_prev = solver.d.x.array.copy()
n_owned = solver.V_d.dofmap.index_map.size_local
for k in range(1, n_static + 1):
    res = solver.solve_step()
    solver.commit_step()
    dd = comm.allreduce(
        float(np.max(np.abs(solver.d.x.array[:n_owned] - d_prev[:n_owned],
                            ), initial=0.0)), op=MPI.MAX)
    d_prev = solver.d.x.array.copy()
    dep = depth_now()
    say("  %4d  %.6f  %8.2f  %7.3f  %10.3e  %6d   %s",
        k, solver.max_damage(), dep * 1e6, dep / contact.a, dd,
        len(res.history), "是" if res.converged else "否：%s" % res.failure_mode)
    if not res.converged:
        say("  -> 静态问题本身就解不动，说明这个状态已经在失稳附近")
        break

say("")
say("判读：Δd 逐轮衰减到 0 => 静态稳定，墙是数值问题；"
    "Δd 不衰减或深度持续增加 => 静态失稳，墙是物理的。")
