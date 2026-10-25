# -*- coding: utf-8 -*-
"""零插桩探针：粗网格单体 Newton 为什么发散（续 19 续 8）。

已确认（续 19 续 7，data/probe_coarse_cyc.log）：
  粗网格 30 480 单元，相位 0.05（第一个开摩擦的相位）：
    conv=False  it=50  max|u|=2.88  max|q|=5.81e13（= mu*p 的 1.2e4 倍）
    ||F_free|| 历史：2.9e12 → 1.8e12 → 2.3e12 → 1.2e13 → 1.5e14 → 反复跳
  而 `solve_displacement()` **正常返回**，`uzawa_converged=False` 没人看
  ⇒ explicit_cycle 拿着未收敛的 u 继续 ⇒ psi+ ~1e31 ⇒ alpha_bar 爆。

tiny 网格同工况 20 个子步全收敛 ⇒ 粗网格特有。

本轮区分两个假设（都不改源码）：
  H2a **全局化失效**：Newton 方向是对的，但整步太大，u 冲到 2.88 m
      把罚接触的非线性拉爆。加阻尼（θ=0.5 / 0.25）应当能救。
  H2b **Jacobian 错**：方向本身就错，阻尼只会让它慢慢错。
      特征是加了阻尼残差仍不下降、或需要极小 θ。

判据：若 θ=0.25 能让 ||F|| 单调下降并收敛 ⇒ H2a（可修，加线搜索/阻尼）；
      若 θ 小到 1e-3 仍不降 ⇒ H2b（Jacobian 有实质错误）。

⚠ 零插桩：本探针**自己复刻** solve_monolithic 的装配步骤（调的全是公开方法
  monolithic_residual / monolithic_jacobian / _contact_cell_dofs /
  _to_csr / solve_block_newton 的同类逻辑），只在**探针内**加阻尼。
  绝不改 src/。
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


def _to_csr(mat):
    mat.assemble()
    r, c = mat.getLocalSize()
    indptr, indices, data = mat.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=(r, c))


def _mask_bc(J, g_bc, n):
    keep = np.ones(n, dtype=bool)
    keep[g_bc] = False
    row_of = np.repeat(np.arange(n), np.diff(J.indptr))
    sel = keep[row_of]
    counts = np.diff(J.indptr).copy()
    counts[g_bc] = 0
    indptr = np.concatenate([[0], np.cumsum(counts)])
    J = sp.csr_matrix((J.data[sel], J.indices[sel], indptr), shape=(n, n))
    I = sp.csr_matrix((np.ones(g_bc.size), (g_bc, g_bc)), shape=(n, n))
    return (J + I).tocsr()


def damped_newton(u, q, clamp, elastic, u_bcs, *, theta, max_it=50, tol=1e-3,
                  label=""):
    """与 solve_block_newton 同构，但步长乘 theta（探针内实现）。"""
    V, Q = u.function_space, q.function_space
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    q_t = ufl.TestFunction(Q)
    q_du = ufl.TrialFunction(Q)

    n_u = clamp.mono_n_u
    n_q = clamp.mono_n_q

    u_bc = np.unique(np.concatenate([bc.dof_indices()[0] for bc in u_bcs])) \
        if u_bcs else np.array([], dtype=np.int32)
    contact, all_dofs = clamp._contact_cell_dofs(u)
    q_bc = np.setdiff1d(all_dofs, contact) + n_u
    g_bc = np.concatenate([u_bc.astype(np.int64), q_bc.astype(np.int64)])
    free = np.setdiff1d(np.arange(n_u + n_q, dtype=np.int64), g_bc)

    hist = []
    for it in range(max_it):
        F_u = elastic(v)
        F_uc, F_q = clamp.monolithic_residual(u, v, q_t, None)
        F_u = F_u + F_uc
        blocks = clamp.monolithic_jacobian(u, du, v, q_t, q_du, fu_form=F_u)

        r = np.concatenate([assemble_vector(fem.form(F_u)).array[:n_u],
                            assemble_vector(fem.form(F_q)).array[:n_q]])
        rn = float(np.linalg.norm(r[free]))
        hist.append(rn)
        print("   [%s θ=%.3g] it %2d : ||F_free|| = %.4e" % (label, theta, it, rn),
              flush=True)
        if rn < tol:
            return True, it, rn, hist

        Juu = _to_csr(assemble_matrix(fem.form(blocks[0][0]), bcs=[]))
        Juq = _to_csr(assemble_matrix(fem.form(blocks[0][1]), bcs=[]))
        Jqu = _to_csr(assemble_matrix(fem.form(blocks[1][0]), bcs=[]))
        Jqq = _to_csr(assemble_matrix(fem.form(blocks[1][1]), bcs=[]))
        J = sp.bmat([[Juu, Juq], [Jqu, Jqq]], format="csr")
        J = _mask_bc(J, g_bc, n_u + n_q)
        Jf = J[free][:, free]
        dxf = spla.spsolve(Jf, -r[free])
        if not np.all(np.isfinite(dxf)):
            return False, it, rn, hist
        dx = np.zeros(n_u + n_q)
        dx[free] = theta * dxf
        u.x.array[:n_u] += dx[:n_u]
        u.x.scatter_forward()
        q.x.array[:n_q] += dx[n_u:]
        q.x.scatter_forward()
    return False, max_it, hist[-1], hist


def main():
    from mpi4py import MPI
    from pfwire.clamp_case import (build_clamp_case, solve_contact_at,
                                   _enable_friction_after_ramp)

    case = build_clamp_case("configs/clamp_fretting_coarse.yaml", MPI.COMM_WORLD)
    solver, clamp = case.solver, case.clamp
    u = solver.u

    solve_contact_at(case, DELTA_STAR)
    _enable_friction_after_ramp(case)
    clamp.bind_monolithic_space(solver.V_u)

    def elastic(v):
        F = solver._residual_u_form(v)
        if getattr(clamp, "has_friction_at_build", False):
            F = F + clamp.tangential_residual(solver.u, v)
        return F

    # 存延拓结束时的状态，每个 theta 都从这里起步（同一初值，可比）
    u0 = u.x.array.copy()
    q0 = clamp.q.x.array.copy()

    print("=== 相位 0.05，从延拓末状态起步 ===")
    for theta in (1.0, 0.5, 0.25, 0.1):
        u.x.array[:] = u0
        u.x.scatter_forward()
        clamp.q.x.array[:] = q0
        clamp.q.x.scatter_forward()
        case.loads.set_phase(np.sin(2 * np.pi * 0.05))
        clamp.set_tangential(AMP * np.sin(2 * np.pi * 0.05))
        ok, it, rn, h = damped_newton(
            u, clamp.q, clamp, elastic, solver.bcs_u,
            theta=theta, max_it=30, tol=1e-3, label="p0.05")
        print("   ⇒ θ=%.3g : %s it=%d ||F||=%.4e  max|u|=%.3e"
              % (theta, "CONVERGED" if ok else "FAILED", it, rn,
                 float(np.max(np.abs(u.x.array)))), flush=True)
        print(flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
