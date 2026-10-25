# -*- coding: utf-8 -*-
"""零插桩探针：第一个 Newton 步为什么把残差放大 6.5 个量级（续 19 续 10）。

已确认（data/probe_fd_coarse.log）：
  Jacobian 是**真导数**：eps=1e-6 时 rel=6.2e-6，且 rel ∝ eps（纯截断误差）。
  ⇒ H2b（Jacobian 错）**否证**。

已确认（data/probe_damp.log）：
  θ=0.1 时残差**严格按 0.9^k 衰减 15 步** —— 这正是「线性问题 + 精确 J」的
  解析解 r_{k+1} = (1-θ) r_k。⇒ 局部线性、J 精确，只是**全步离开线性区**。
  但第 1 步就把 5.0e4 抬到 1.0e10（6 个量级）—— 这才是病根。

本轮量三件事（都在探针内，不改源码）：
  1. **Newton 步本身多大**：||dx_u||、||dx_q||，与 u、q 的量级比
  2. **全步后的实际残差 vs Newton 预测的 0**：比值 = 非线性程度
  3. **Jf 的条件数**（用 svds 取最小几个奇异值，粗网格上不能全分解）

判据：
  若 ||dx_u|| ~ 1 m（位移尺度 1e-5 的 1e5 倍）⇒ J 在某方向近乎奇异
     ⇒ 不是「步太长」，是**约束缺失**（与 BC 只有 12 个 u dof 呼应）
  若 ||dx_u|| ~ 1e-5 而残差仍爆 ⇒ 强非线性，需要延拓/线搜索
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
    free = np.setdiff1d(np.arange(n_u + n_q, dtype=np.int64), g_bc)

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
        return (_to_csr(assemble_matrix(fem.form(b[0][0]), bcs=[])),
                _to_csr(assemble_matrix(fem.form(b[0][1]), bcs=[])),
                _to_csr(assemble_matrix(fem.form(b[1][0]), bcs=[])),
                _to_csr(assemble_matrix(fem.form(b[1][1]), bcs=[])))

    print("=== 延拓末（相位 0）===", flush=True)
    print("  max|u| = %.4e  max|q| = %.4e  u_prev 已冻结"
          % (np.max(np.abs(u.x.array)), np.max(np.abs(q.x.array))), flush=True)

    case.loads.set_phase(np.sin(2 * np.pi * PHASE))
    clamp.set_tangential(AMP * np.sin(2 * np.pi * PHASE))
    print("=== 相位 %.2f：M/M_a = %.4f, x_clamp = %.4e m ==="
          % (PHASE, np.sin(2 * np.pi * PHASE),
             AMP * np.sin(2 * np.pi * PHASE)), flush=True)

    r0 = residual()
    rn0 = float(np.linalg.norm(r0[free]))
    print("  ||F_free|| = %.4e" % rn0, flush=True)

    Juu, Juq, Jqu, Jqq = jacobian()
    J = sp.bmat([[Juu, Juq], [Jqu, Jqq]], format="csr")
    n = n_u + n_q
    keep = np.ones(n, dtype=bool)
    keep[g_bc] = False
    row_of = np.repeat(np.arange(n), np.diff(J.indptr))
    sel = keep[row_of]
    counts = np.diff(J.indptr).copy()
    counts[g_bc] = 0
    indptr = np.concatenate([[0], np.cumsum(counts)])
    J = sp.csr_matrix((J.data[sel], J.indices[sel], indptr), shape=(n, n))
    J = (J + sp.csr_matrix((np.ones(g_bc.size), (g_bc, g_bc)), shape=(n, n))).tocsr()
    Jf = J[free][:, free]

    # ---- 1. 直接解的对角诊断（svds 在 62718 阶上跑不动，已弃用）----
    # 用 splu 的 U 对角：若有零主元，spsolve 会给出 inf/nan 或巨大 dx。
    # 这里只报对角范围，判「有没有几乎为零的主元」。
    print("  Jf 对角范围：[%.4e, %.4e]"
          % (np.min(np.abs(Jf.diagonal())), np.max(np.abs(Jf.diagonal()))), flush=True)

    # ---- 2. 一步 Newton ----
    dxf = spla.spsolve(Jf, -r0[free])
    print("  ||dx_free|| = %.4e" % np.linalg.norm(dxf), flush=True)
    dx = np.zeros(n)
    dx[free] = dxf
    print("    ||dx_u|| = %.4e  （u 量级 %.4e ⇒ 倍数 %.3e）"
          % (np.linalg.norm(dx[:n_u]), np.linalg.norm(u.x.array[:n_u]),
             np.linalg.norm(dx[:n_u]) / max(np.linalg.norm(u.x.array[:n_u]), 1e-300)),
          flush=True)
    print("    max|dx_u| = %.4e  max|dx_q| = %.4e"
          % (np.max(np.abs(dx[:n_u])), np.max(np.abs(dx[n_u:]))), flush=True)

    u_save = u.x.array.copy()
    q_save = q.x.array.copy()
    u.x.array[:n_u] += dx[:n_u]
    u.x.scatter_forward()
    q.x.array[:n_q] += dx[n_u:]
    q.x.scatter_forward()
    r1 = residual()
    rn1 = float(np.linalg.norm(r1[free]))
    print("  全步后 ||F_free|| = %.4e（Newton 预测 0，放大 %.3e 倍）"
          % (rn1, rn1 / max(rn0, 1e-300)), flush=True)
    print("  全步后 max|u| = %.4e  max|q| = %.4e"
          % (np.max(np.abs(u.x.array)), np.max(np.abs(q.x.array))), flush=True)

    # ---- 3. 沿同方向扫 θ，看残差 vs 步长（找线性区的边界）----
    print()
    print("  沿 Newton 方向扫步长（判线性区半径）:", flush=True)
    for th in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0):
        u.x.array[:] = u_save
        u.x.scatter_forward()
        q.x.array[:] = q_save
        q.x.scatter_forward()
        u.x.array[:n_u] += th * dx[:n_u]
        u.x.scatter_forward()
        q.x.array[:n_q] += th * dx[n_u:]
        q.x.scatter_forward()
        rr = residual()
        rn = float(np.linalg.norm(rr[free]))
        print("    θ=%.0e : ||F||/||F0|| = %.4e  %s"
              % (th, rn / max(rn0, 1e-300),
                 "（线性区，应为 1-θ）" if abs(rn / max(rn0, 1e-300) - (1 - th)) < 0.05
                 else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
