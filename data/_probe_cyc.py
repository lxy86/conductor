# -*- coding: utf-8 -*-
"""零插桩探针：一个完整显式周期里，单体 Newton 是否**静默不收敛**（续 19 续 6）。

粗网格现场（ckpt_1.bp，已读回）：
  max|u|          = 6.139e7 m      （位移尺度 δ = 2.725e-5 ⇒ 大 2.3e12 倍）
  max|q|          = 3.430e9 Pa     （= 0.7 × 4.90e9 = mu*p ⇒ **投影本身正常**）
  max|d|          = 0              （d 一步没动）
  alpha_bar = psi_prev = 3.29e31   （≈ 0.5*E*(u/L)^2，u 的下游量）

u 的分布：**全部 30955 个顶点**都 >1e-3（中位数 3.78e7）⇒ 整体跑飞，
不是局部伪影、不是刚体模态（刚体模态会是常数场，这里从 3.8e7 到 7.2e7 有梯度）。

⇒ 假设 H1：`_solve_u_monolithic` 在 `max_it` 时不抛异常，只 warn（且需
   log_iterations，本配置为 false），于是 explicit_cycle 拿着一个**未收敛的 u**
   继续往下跑，psi+ 从它算出来 ⇒ alpha_bar 爆 ⇒ d 的 SNES -6。

本探针在 tiny 网格上跑一个完整周期（20 子步，与生产 n_sub 同），
**逐子步报告 block Newton 的 converged / reason / ||F|| / 迭代数**，
以及 max|u|。若某个相位出现 converged=False 而循环照常推进，H1 成立。
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, "/root/pf-wire")
sys.path.insert(0, "/root/pf-wire/tests")

import ufl
from dolfinx import fem
from pfwire.obstacle_contact import plane_strain_lame
from test_clamp_uzawa import (_setup, _free_support, _uzawa, MU, DELTA,
                              E_BULK, NU_BULK)


def _ramp(ctx):
    """与生产 find_working_delta 同序：关摩擦延拓 → 冻结滑移基准 → 开摩擦。"""
    clamp, u, prob = ctx["clamp"], ctx["u"], ctx["problem"]
    clamp.set_mu(None)
    for d in (1.0e-7, 1.0e-6, 1.0e-5, DELTA):
        clamp.set_delta(d)
        prob.solve()
        r = prob.solver.getConvergedReason()
        if r <= 0:
            raise RuntimeError("延拓不收敛 delta=%g reason=%d" % (d, r))
    clamp.set_slip_reference(u)
    clamp.set_mu(MU)


def _mono_elastic(ctx):
    """弹性 − 外载 − 法向接触（切向由 F_q 与 q 面力承担，与生产 _elastic_external 同）。"""
    clamp, u, V = ctx["clamp"], ctx["u"], ctx["V"]
    loads = ctx["loads"]
    mu_l, lmbda = plane_strain_lame(E_BULK, NU_BULK)

    def _f(v):
        eps = ufl.sym(ufl.grad(u))
        sig = 2.0 * mu_l * eps + lmbda * ufl.tr(eps) * ufl.Identity(2)
        F = ufl.inner(sig, ufl.sym(ufl.grad(v))) * ufl.dx
        F = F - loads.traction()(V)
        F = F - clamp.residual(u, v)
        return F
    return _f


def _psi_total(ctx):
    """总弹性能 ∫0.5 σ:ε（用来判「有没有爆」，不是 psi+）。"""
    clamp, u, V = ctx["clamp"], ctx["u"], ctx["V"]
    mu_l, lmbda = plane_strain_lame(E_BULK, NU_BULK)
    eps = ufl.sym(ufl.grad(u))
    sig = 2.0 * mu_l * eps + lmbda * ufl.tr(eps) * ufl.Identity(2)
    return float(fem.assemble_scalar(fem.form(0.5 * ufl.inner(sig, eps) * ufl.dx)))


def run_cycle(ctx, mode, amp, n_sub=20):
    clamp, u, V = ctx["clamp"], ctx["u"], ctx["V"]
    loads = ctx["loads"]
    clamp.bind_monolithic_space(V)
    _ramp(ctx)
    elastic = _mono_elastic(ctx)
    bcs = list(_free_support(ctx["ft"], ctx["msh"])(V))

    rows = []
    for k in range(1, n_sub + 1):
        phase = k / n_sub
        s = np.sin(2 * np.pi * phase)
        loads.set_phase(s)
        clamp.set_tangential(amp * s)
        if mode == "mono":
            res = clamp.solve_monolithic(u, elastic, bcs, tol=1e-3, max_it=50)
            conv, reason, nit, rf = res.converged, res.reason, res.iterations, res.residual
        else:
            nit = _uzawa(ctx["problem"], clamp, u)
            conv, reason, rf = True, "uzawa", float("nan")
        rows.append((phase, float(np.max(np.abs(u.x.array))),
                     float(np.max(np.abs(clamp.q.x.array))),
                     _psi_total(ctx), conv, reason, nit, rf))
    return rows


def main():
    tmp = "/tmp/probe_cyc2"
    pathlib.Path(tmp).mkdir(parents=True, exist_ok=True)
    amp = 1.0e-7          # 与 configs/clamp_fretting_coarse.yaml 同

    for mode in ("mono", "stag"):
        label = "单体" if mode == "mono" else "交错"
        print("=== %s（amp=%.0e）===" % (label, amp))
        ctx = _setup(tmp + "_" + mode)
        rows = run_cycle(ctx, mode, amp)
        print("   phase  max|u|      max|q|      psi_total   conv  reason      it")
        for (ph, mu, q, psi, conv, reason, nit, rf) in rows:
            print("   %.2f  %.3e  %.3e  %.3e  %-5s %-11s %2d"
                  % (ph, mu, q, psi, conv, reason, nit))
        us = [r[1] for r in rows]
        ps = [r[3] for r in rows]
        nbad = sum(1 for r in rows if not r[4])
        print("   ⇒ max|u| ∈ [%.3e, %.3e]，psi ∈ [%.3e, %.3e]，未收敛子步 %d/%d"
              % (min(us), max(us), min(ps), max(ps), nbad, len(rows)))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
