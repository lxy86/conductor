# -*- coding: utf-8 -*-
"""零插桩探针：粗网格相位 0.05 的 Jacobian 是否是真导数（续 19 续 9）。

已有证据（data/probe_damp.log）：
  θ=1    : 5.0e4 → 1.8e11（一步过冲 6.5 个量级）
  θ=0.5  : 5.0e4 → 8.6e10，随后 4.3e10 → 2.2e10（严格按 (1-θ) 减半）
  θ=0.1  : 5.0e4 → 1.0e10，随后 15 步**严格按 0.9 衰减**，第 17 步才爆

「严格按 (1−θ) 衰减」= 该区间内系统**线性且 J 精确**（牛顿方向对、
只是只走了 θ 份）。故不是「Jacobian 整体错」，而是：
  (a) 第一步就把状态推过某个**非光滑事件**（接触集变化 / 滑移-黏着切换），或
  (b) 某个块的导数在**特定子集**上是错的。

本轮做**整块有限差分**：取自由 dof 上的随机方向 δ，
比较 r(x+εδ) − r(x) 与 J·δ，并分块报告（u 段 / q 段）。
判据：
  相对误差 < 1e-3  ⇒ J 是真导数 ⇒ 问题在全局化/非光滑事件（H2a）
  相对误差 ≫ 1     ⇒ J 有实质错误（H2b）

⚠ 零插桩：全部调公开方法（monolithic_residual / monolithic_jacobian /
  _contact_cell_dofs），在探针内复刻 solve_block_newton 的 BC 掩码。
"""

import sys

import numpy as np
import scipy.sparse as sp

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

    case.loads.set_phase(np.sin(2 * np.pi * PHASE))
    clamp.set_tangential(AMP * np.sin(2 * np.pi * PHASE))

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
    free = np.setdiff1d(np.arange(n_u + n_q, dtype=np.int64), g_bc)
    print("n_u=%d n_q=%d  BC(u)=%d  BC(q)=%d  自由=%d"
          % (n_u, n_q, u_bc.size, q_bc.size, free.size), flush=True)

    def residual():
        F_u = elastic(v)
        F_uc, F_q = clamp.monolithic_residual(u, v, q_t, None)
        F_u = F_u + F_uc
        return np.concatenate([assemble_vector(fem.form(F_u)).array[:n_u],
                               assemble_vector(fem.form(F_q)).array[:n_q]])

    def jacobian():
        F_u = elastic(v)
        F_uc, F_q = clamp.monolithic_residual(u, v, q_t, None)
        F_u = F_u + F_uc
        b = clamp.monolithic_jacobian(u, du, v, q_t, q_du, fu_form=F_u)
        Juu = _to_csr(assemble_matrix(fem.form(b[0][0]), bcs=[]))
        Juq = _to_csr(assemble_matrix(fem.form(b[0][1]), bcs=[]))
        Jqu = _to_csr(assemble_matrix(fem.form(b[1][0]), bcs=[]))
        Jqq = _to_csr(assemble_matrix(fem.form(b[1][1]), bcs=[]))
        return Juu, Juq, Jqu, Jqq

    r0 = residual()
    print("||F_free|| = %.4e（u 段 %.3e，q 段 %.3e）"
          % (np.linalg.norm(r0[free]), np.linalg.norm(r0[:n_u][free[free < n_u]]),
             np.linalg.norm(r0[n_u:])), flush=True)

    Juu, Juq, Jqu, Jqq = jacobian()
    J = sp.bmat([[Juu, Juq], [Jqu, Jqq]], format="csr")

    rng = np.random.default_rng(0)
    # 方向幅值按 dof 的实际量级取：u ~1e-5 m、q ~1e8 Pa
    scale = np.ones(n_u + n_q)
    scale[:n_u] = 1.0e-5
    scale[n_u:] = 1.0e8
    for eps in (1.0e-4, 1.0e-6):
        d = rng.standard_normal(n_u + n_q) * scale * eps
        d[g_bc] = 0.0
        u.x.array[:n_u] += d[:n_u]
        u.x.scatter_forward()
        q.x.array[:n_q] += d[n_u:]
        q.x.scatter_forward()
        r1 = residual()
        u.x.array[:n_u] -= d[:n_u]
        u.x.scatter_forward()
        q.x.array[:n_q] -= d[n_u:]
        q.x.scatter_forward()

        fd = r1 - r0
        lin = J @ d
        for name, sl in (("全部", slice(None)),
                         ("u 段", slice(0, n_u)),
                         ("q 段", slice(n_u, None))):
            a, b_ = fd[sl], lin[sl]
            na, nb = np.linalg.norm(a), np.linalg.norm(b_)
            rel = np.linalg.norm(a - b_) / max(na, nb, 1e-300)
            print("  eps=%.0e %-4s : |FD|=%.4e |J·d|=%.4e  rel=%.3e"
                  % (eps, name, na, nb, rel), flush=True)
        print(flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
