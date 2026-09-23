"""从本模型自己算出接触区的相对滑移幅，供 Archard 磨损估算用（PLAN Phase 3 讨论 a）。

**为什么不直接套解析式**：本模型用解析面力代替接触对偶（PLAN 第 2 节的做法），
域内只有一个物体，相对滑移不是求解量。而硬约束 2 不许自创公式，
Cattaneo–Mindlin 的滑移位移解析式我手头没有可引的出处。

**可以绕开**：对偶是同材料同几何的铝丝（``counterpart: aluminum``），
接触对称，故第二个体的表面切向位移与本体等大反向：

    相对滑移 δ(x) = u_x^(1)(x) − u_x^(2)(x) = 2 u_x(x)

而 u_x 是本模型的求解量。于是滑移幅 = 2 × [u_x(峰值相位) − u_x(谷值相位)] / 2
                              = u_x(+1) − u_x(−1)，取表面 y = 0 上的值。

弹性解即可（d ≡ 0）：磨损发生在裂纹萌生之前，那时材料还没损伤。

用法：mpirun -n 2 python data/_slip_amplitude.py [工况 yaml]
"""

from __future__ import annotations

import sys

import numpy as np
from dolfinx import default_scalar_type, fem
from mpi4py import MPI

from pfwire import loads as pfloads
from pfwire import mesh as pfmesh
from pfwire.fretting import _make_bcs, _traction_forms
from pfwire.material import contact_pair_properties
from pfwire.phasefield import PhaseFieldSolver
from pfwire.run_case import load_case

comm = MPI.COMM_WORLD
cfg = load_case(sys.argv[1] if len(sys.argv) > 1 else "configs/fretting_base.yaml")
raw = cfg.raw
ccfg, lcfg = dict(raw["contact"]), dict(cfg.loading)


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

mcfg = dict(cfg.mesh_cfg)
mcfg.pop("type", None)
spec = pfmesh.FrettingSpec(contact_halfwidth=contact.a,
                           **{k: float(v) for k, v in mcfg.items()})
fr = pfmesh.fretting_mesh(spec, comm)
msh, tags = fr.mesh, fr.facet_tags
fdim = msh.topology.dim - 1

phase = fem.Constant(msh, default_scalar_type(0.0))
sigma_b = fem.Constant(msh, default_scalar_type(0.0))
sigma_a = float(lcfg["sigma_a"])
sigma_m = float(lcfg.get("sigma_m", 0.0))
make_work = _traction_forms(msh, contact, phase, sigma_b, tags)

# d ≡ 0 的弹性求解器；不接 f_alpha（磨损发生在损伤之前）
solver = PhaseFieldSolver(
    msh, cfg.model, bcs_u=_make_bcs(msh, tags, fdim),
    external_work=make_work, config=cfg.solver)


def surface_ux(value: float) -> tuple[np.ndarray, np.ndarray]:
    """在给定相位解一次弹性问题，返回表面 y=0 上的 (x, u_x)。"""
    phase.value = value
    sigma_b.value = sigma_m + sigma_a * value
    solver.solve_displacement()
    V = solver.u.function_space
    coords = V.tabulate_dof_coordinates()
    n_owned = V.dofmap.index_map.size_local
    on_top = np.nonzero(np.abs(coords[:n_owned, 1]) < 1.0e-12)[0]
    xs = coords[on_top, 0]
    ux = solver.u.x.array.reshape(-1, 2)[on_top, 0]
    xs = np.concatenate(comm.allgather(xs))
    ux = np.concatenate(comm.allgather(ux))
    order = np.argsort(xs)
    return xs[order], ux[order]


x_p, ux_p = surface_ux(+1.0)
x_m, ux_m = surface_ux(-1.0)
assert np.allclose(x_p, x_m)

# 相对滑移幅（峰谷之差）：对称接触下 δ = 2 u_x，故幅值 = u_x(+1) − u_x(−1)
delta = ux_p - ux_m
a, c = contact.a, contact.c
in_slip = (np.abs(x_p) >= c) & (np.abs(x_p) <= a)
in_stick = np.abs(x_p) < c

say("工况 %s：p0=%.4g Pa, Q/muP=%.3g, a=%.3f um, c/a=%.4f",
    cfg.path.name, contact.p0, float(ccfg["Q_over_muP"]), a * 1e6, c / a)
say("sigma_a=%.4g Pa, sigma_m=%.4g Pa", sigma_a, sigma_m)
say("")
say("表面 y=0 上取到 %d 个点（|x|<=a 内 %d 个）",
    x_p.size, int((np.abs(x_p) <= a).sum()))
say("")
say("**绝对切向位移幅不是微滑**：本模型用规定面力代替接触，黏着区并没有被")
say("任何约束强制「不滑」，所以 u_x 里含整个表面的切向变形（刚体式平动）。")
say("绝对值 delta = u_x(+1) - u_x(-1)：")
say("  黏着区最大 %.4g nm / 滑移区均值 %.4g nm  —— 两者几乎相同，正说明它不是微滑",
    np.max(np.abs(delta[in_stick])) * 1e9 if in_stick.any() else float("nan"),
    np.mean(np.abs(delta[in_slip])) * 1e9 if in_slip.any() else float("nan"))

# 扣掉接触中心的平动，剩下的才是接触区内的**相对**切向错动
i0 = int(np.argmin(np.abs(x_p)))
rel = delta - delta[i0]
say("")
say("扣掉接触中心（x=0）的平动之后，接触区内的相对错动 |delta(x) - delta(0)|:")
for frac in (0.25, 0.5, 0.632, 0.75, 0.9, 1.0):
    j = int(np.argmin(np.abs(x_p - frac * a)))
    tag = "  <- c/a（黏滑边界）" if abs(frac - c / a) < 0.02 else ""
    say("  |x|/a = %.3f : %8.3f nm%s", frac, abs(rel[j]) * 1e9, tag)
say("  滑移区内最大 %.3f nm", np.max(np.abs(rel[in_slip])) * 1e9 if in_slip.any() else float("nan"))
say("")
say("⚠ 这个「相对错动」仍不等于真实微滑：真实微滑由接触约束决定，")
say("  本模型没有接触对偶，只能给出量级参考。Archard 估算须据此标注不确定性。")
say("")
say("滑移区宽度 a - c = %.4f um（占接触宽度 2a 的 %.1f%%）",
    (a - c) * 1e6, 100.0 * (a - c) / (2 * a))
