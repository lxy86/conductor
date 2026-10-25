# -*- coding: utf-8 -*-
"""零插桩探针：地板的来源 —— q 与 s 的符号结构（续 19 续 13）。

已排除：
  续 9  Jacobian 错（FD rel 6.2e-6 ∝ eps）           —— 否证
  续 12 η 光滑化地板（η 从 0.05 扫到 0.0005，残差 4.52e4→4.50e4，几乎不动）
                                                       —— 否证

现场：Armijo 走到 max|q|≈8e7（mu p 量级）、max|u|=7.2e-5（物理量级），
但 ||F_free|| 卡在 4.5e4 不再降，且 u 段与 q 段都有。

F_q 的不动点条件 q = clip(q − ε_t·s) 的**解析解**：
    |q| < lim（黏着） : 要求 s = 0
    q = +lim（滑移）  : 要求 s < 0
    q = −lim（滑移）  : 要求 s > 0
⇒ **摩擦必须反抗滑移：sign(q) = −sign(s)**。
  若实测大量单元的 sign(q) = +sign(s)，那就是**符号错**（摩擦助推滑移），
  既解释爆炸（正反馈无耗散），也解释「没有根」（该方向无解）。

本探针在 Armijo 卡住的状态上，逐接触单元报 q、s、lim 的符号结构与
残差归因。**只看，不改**。
"""

import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, "/root/pf-wire")

import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector

DELTA_STAR = 2.7252659701799572e-05
AMP = 1.0e-07
PHASE = 0.05


def _to_csr(mat):
    mat.assemble()
    r, c = mat.getLocalSize()
    indptr, indices, data = mat.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=(r, c))


def main():
    from mpi4py import MPI
    from pfwire.clamp_case import (build_clamp_case, solve_contact_at,
                                   _enable_friction_after_ramp)

    case = build_clamp_case("configs/clamp_fretting_coarse.yaml", MPI.COMM_WORLD)
    solver, clamp = case.solver, case.clamp
    u, q, V, Q = solver.u, clamp.q, solver.V_u, clamp.q.function_space

    solve_contact_at(case, DELTA_STAR)
    _enable_friction_after_ramp(case)
    clamp.bind_monolithic_space(V)

    def elastic(v):
        F = solver._residual_u_form(v)
        if getattr(clamp, "has_friction_at_build", False):
            F = F + clamp.tangential_residual(solver.u, v)
        return F

    n_u, n_q = clamp.mono_n_u, clamp.mono_n_q
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    q_t = ufl.TestFunction(Q)
    q_du = ufl.TrialFunction(Q)

    u_bc = np.unique(np.concatenate(
        [bc.dof_indices()[0] for bc in solver.bcs_u])).astype(np.int64)
    contact, all_dofs = clamp._contact_cell_dofs(u)
    q_bc = np.setdiff1d(all_dofs, contact).astype(np.int64) + n_u
    g_bc = np.concatenate([u_bc, q_bc])
    n = n_u + n_q
    free = np.setdiff1d(np.arange(n, dtype=np.int64), g_bc)

    def residual():
        F_u = elastic(v)
        F_uc, F_q = clamp.monolithic_residual(u, v, q_t, None)
        F_u = F_u + F_uc
        return np.concatenate([assemble_vector(fem.form(F_u)).array[:n_u],
                               assemble_vector(fem.form(F_q)).array[:n_q]])

    def blocks():
        F_u = elastic(v)
        F_uc, F_q = clamp.monolithic_residual(u, v, q_t, None)
        F_u = F_u + F_uc
        b = clamp.monolithic_jacobian(u, du, v, q_t, q_du, fu_form=F_u)
        return tuple(_to_csr(assemble_matrix(fem.form(b[i][j]), bcs=[]))
                     for i in (0, 1) for j in (0, 1))

    case.loads.set_phase(np.sin(2 * np.pi * PHASE))
    clamp.set_tangential(AMP * np.sin(2 * np.pi * PHASE))

    # ---- 先走到卡住的状态（Armijo）----
    print("走到卡住状态（Armijo，40 步上限）…", flush=True)
    for it in range(40):
        r = residual()
        rn = float(np.linalg.norm(r[free]))
        if it % 5 == 0:
            print("   it %2d ||F||=%.4e" % (it, rn), flush=True)
        if rn < 1e-3:
            break
        Juu, Juq, Jqu, Jqq = blocks()
        J = sp.bmat([[Juu, Juq], [Jqu, Jqq]], format="csr")
        keep = np.ones(n, dtype=bool)
        keep[g_bc] = False
        ro = np.repeat(np.arange(n), np.diff(J.indptr))
        s2 = keep[ro]
        c2 = np.diff(J.indptr).copy()
        c2[g_bc] = 0
        ip2 = np.concatenate([[0], np.cumsum(c2)])
        J = sp.csr_matrix((J.data[s2], J.indices[s2], ip2), shape=(n, n))
        J = (J + sp.csr_matrix((np.ones(g_bc.size), (g_bc, g_bc)),
                               shape=(n, n))).tocsr()
        dkf = spla.spsolve(J[free][:, free], -r[free])
        dk = np.zeros(n)
        dk[free] = dkf
        us, qs = u.x.array.copy(), q.x.array.copy()
        th = 1.0
        for _ in range(40):
            u.x.array[:] = us
            u.x.scatter_forward()
            q.x.array[:] = qs
            q.x.scatter_forward()
            u.x.array[:n_u] += th * dk[:n_u]
            u.x.scatter_forward()
            q.x.array[:n_q] += th * dk[n_u:]
            q.x.scatter_forward()
            if float(np.linalg.norm(residual()[free])) < (1 - 1e-4 * th) * rn:
                break
            th *= 0.5

    # ---- 逐接触单元看符号结构 ----
    print()
    print("=== 卡住状态的逐单元符号结构 ===", flush=True)
    from dolfinx import fem as _fem
    Vs = _fem.functionspace(case.mesh, ("Lagrange", 1))
    tau_up, _ = clamp._tangents(u)
    s_expr = clamp._slip(u, tau_up, clamp.x_clamp)
    s_cg = _fem.Function(Vs)
    s_cg.interpolate(_fem.Expression(s_expr, Vs.element.interpolation_points))
    s_cg.x.scatter_forward()
    if not hasattr(clamp, "_p_dg_projector"):
        clamp._build_p_dg_projector()
    s_dg = clamp._p_dg_projector.dot(s_cg.x.array[:clamp._p_n_owned])

    p_expr, _nrm = clamp._pressure_and_normal(u)
    p_cg = _fem.Function(Vs)
    p_cg.interpolate(_fem.Expression(p_expr, Vs.element.interpolation_points))
    p_cg.x.scatter_forward()
    p_dg = clamp._p_dg_projector.dot(p_cg.x.array[:clamp._p_n_owned])

    qa = q.x.array[:n_q]
    lim = float(clamp.mu.value) * np.maximum(p_dg, 0.0)
    ct = lim > 0
    print("  接触单元 %d 个（p_dg>0）" % int(ct.sum()), flush=True)
    qq, ss, ll = qa[ct], s_dg[ct], lim[ct]
    print("    max|q|=%.4e  max|s|=%.4e  max(lim)=%.4e"
          % (np.max(np.abs(qq)), np.max(np.abs(ss)), np.max(ll)), flush=True)
    same = np.sign(qq) * np.sign(ss) > 0
    print("    sign(q)==sign(s) 的单元：%d / %d（**应为 0：摩擦必须反抗滑移**）"
          % (int(same.sum()), same.size), flush=True)
    print("    sign(q)!=sign(s) 的单元：%d" % int((np.sign(qq) * np.sign(ss) < 0).sum()),
          flush=True)
    print("    s==0 的单元：%d" % int((ss == 0).sum()), flush=True)

    eta = clamp.uzawa_eta_rel * ll
    from pfwire.clamp_contact import smooth_clip
    tgt = smooth_clip(qq - clamp.eps_t * ss, ll, eta)
    rho = np.abs(qq - tgt)
    print("    逐单元残差 rho：max=%.4e  中位=%.4e  和=%.4e"
          % (np.max(rho), np.median(rho), np.sum(rho)), flush=True)
    order = np.argsort(rho)[::-1][:8]
    print("    残差最大的 8 个单元：", flush=True)
    print("      %-12s %-12s %-12s %-12s %-12s %s"
          % ("q", "s", "lim", "mu p", "rho", "sign(q)*sign(s)"), flush=True)
    for i in order:
        print("      %+.4e %+.4e %+.4e %+.4e %+.4e %+d"
              % (qq[i], ss[i], ll[i], ll[i], rho[i],
                 int(np.sign(qq[i]) * np.sign(ss[i]))), flush=True)
    print()
    print("  eps_t = %.4e, 1/eps_t = %.4e（s 每 1/eps_t 就把 q 推到盒外）"
          % (clamp.eps_t, 1.0 / clamp.eps_t), flush=True)
    print("  典型 s 使 eps_t*s = %.4e vs lim = %.4e（倍数 %.3e）"
          % (clamp.eps_t * np.median(np.abs(ss)), np.median(ll),
             clamp.eps_t * np.median(np.abs(ss)) / max(np.median(ll), 1e-300)),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
