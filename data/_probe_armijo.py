# -*- coding: utf-8 -*-
"""零插桩探针：1e-5 的零主元在哪一块 + Armijo 回溯能否全局化（续 19 续 11）。

已有实测链（全部零插桩，调公开 API）：
  续 7  data/probe_coarse_cyc.log  相位 0.05 首个开摩擦相位即发散：
        conv=False it=50 max|u|=2.88 max|q|=5.81e13，而 solve_displacement
        **正常返回**（uzawa_converged=False 没人看）
  续 9  data/probe_fd_coarse.log    Jacobian 是**真导数**（rel 6.2e-6 ∝ eps）
  续 10 data/probe_step.log         Newton 步：max|dx_u|=2.97e-4（全场 max|u|=7.1e-5，
        即一步 4 倍）、max|dx_q|=3.65e9（= mu*p 的 24 倍）；
        Jf 对角最小 9.9863e-06 vs 最大 3.5224e+12（跨 18 个量级）；
        线性区只到 θ≈1e-3（θ=1e-2 已放大到 1.27 倍）。

⇒ 不是 Jacobian 错，是**系统在黏-滑切换处极度非光滑 + 尺度悬殊**，
  无阻尼 Newton 必然过冲。本探针补最后两块拼图：

  (1) **1e-5 的主元在哪一块**（J_uu / J_qq），以及各块对角的量级分布。
      若在 J_qq ⇒ 是 smooth_clip 在 η→0 处导数塌陷（lim=0 的单元）；
      若在 J_uu ⇒ 是弹性刚度在某个 dof 上塌陷。
  (2) **Armijo 回溯能否全局化**：沿 Newton 方向做标准回溯线搜索
      （不是固定 θ），看能不能把残差真正降下去。
      能 ⇒ 加线搜索即可修（求解器问题）；
      不能 ⇒ 路径本身不可达，需要延拓（相位/切向幅值的增量加载）。
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


def _stats(name, v):
    a = np.abs(v)
    nz = a[a > 0]
    if nz.size == 0:
        print("    %-4s : 全零" % name)
        return
    q = np.percentile(nz, [0, 1, 50, 99, 100])
    print("    %-4s : min=%.3e p1=%.3e 中位=%.3e p99=%.3e max=%.3e  零元=%d/%d"
          % (name, q[0], q[1], q[2], q[3], q[4],
             int((a == 0).sum()), a.size))


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

    r0 = residual()
    rn0 = float(np.linalg.norm(r0[free]))
    print("||F_free|| = %.4e" % rn0, flush=True)

    Juu, Juq, Jqu, Jqq = blocks()
    print("=== (1) 各块对角量级（1e-5 的主元在哪一块）===", flush=True)
    _stats("Juu", Juu.diagonal())
    _stats("Juq", Juq.diagonal())
    _stats("Jqu", Jqu.diagonal())
    _stats("Jqq", Jqq.diagonal())
    print("    Jqq 对角里 <1e-3 的个数：%d / %d"
          % (int((np.abs(Jqq.diagonal()) < 1e-3).sum()), n_q), flush=True)
    print("    （接触单元 %d 个、非接触 %d 个；非接触的 q 被 BC 钉 0）"
          % (contact.size, all_dofs.size - contact.size), flush=True)

    J = sp.bmat([[Juu, Juq], [Jqu, Jqq]], format="csr")
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

    dxf = spla.spsolve(Jf, -r0[free])
    dx = np.zeros(n)
    dx[free] = dxf

    u_save = u.x.array.copy()
    q_save = q.x.array.copy()

    def trial(theta):
        u.x.array[:] = u_save
        u.x.scatter_forward()
        q.x.array[:] = q_save
        q.x.scatter_forward()
        u.x.array[:n_u] += theta * dx[:n_u]
        u.x.scatter_forward()
        q.x.array[:n_q] += theta * dx[n_u:]
        q.x.scatter_forward()
        return float(np.linalg.norm(residual()[free]))

    print()
    print("=== (2) Armijo 回溯线搜索（标准全局化，c1=1e-4）===", flush=True)
    theta = 1.0
    prev = rn0
    ok = False
    for ls in range(40):
        rn = trial(theta)
        print("    回溯 %2d : θ=%.3e  ||F||/||F0|| = %.4e  %s"
              % (ls, theta, rn / rn0, "← 接受" if rn < (1 - 1e-4 * theta) * rn0 else ""),
              flush=True)
        if rn < (1 - 1e-4 * theta) * rn0:
            ok = True
            break
        theta *= 0.5
    print("    ⇒ Armijo %s（θ=%.3e, ||F||=%.4e）"
          % ("成功" if ok else "**40 次回溯仍找不到下降步**", theta, rn), flush=True)

    if ok:
        print()
        print("  用该 θ 连走 20 步（固定步长，看能否一路降下去）：", flush=True)
        u.x.array[:] = u_save
        u.x.scatter_forward()
        q.x.array[:] = q_save
        q.x.scatter_forward()
        for k in range(20):
            rk = residual()
            rnk = float(np.linalg.norm(rk[free]))
            print("    it %2d : ||F|| = %.4e  max|u|=%.3e  max|q|=%.3e"
                  % (k, rnk, np.max(np.abs(u.x.array)),
                     np.max(np.abs(q.x.array))), flush=True)
            if rnk < 1e-3:
                print("    ⇒ 收敛", flush=True)
                break
            bk = blocks()
            Jk = sp.bmat([[bk[0], bk[1]], [bk[2], bk[3]]], format="csr")
            keep2 = np.ones(n, dtype=bool)
            keep2[g_bc] = False
            ro = np.repeat(np.arange(n), np.diff(Jk.indptr))
            s2 = keep2[ro]
            c2 = np.diff(Jk.indptr).copy()
            c2[g_bc] = 0
            ip2 = np.concatenate([[0], np.cumsum(c2)])
            Jk = sp.csr_matrix((Jk.data[s2], Jk.indices[s2], ip2), shape=(n, n))
            Jk = (Jk + sp.csr_matrix((np.ones(g_bc.size), (g_bc, g_bc)),
                                     shape=(n, n))).tocsr()
            dkf = spla.spsolve(Jk[free][:, free], -rk[free])
            dk = np.zeros(n)
            dk[free] = dkf
            # 同一条 Armijo 回溯
            th = 1.0
            us, qs = u.x.array.copy(), q.x.array.copy()
            for _ in range(40):
                u.x.array[:] = us
                u.x.scatter_forward()
                q.x.array[:] = qs
                q.x.scatter_forward()
                u.x.array[:n_u] += th * dk[:n_u]
                u.x.scatter_forward()
                q.x.array[:n_q] += th * dk[n_u:]
                q.x.scatter_forward()
                if float(np.linalg.norm(residual()[free])) < (1 - 1e-4 * th) * rnk:
                    break
                th *= 0.5
    return 0


if __name__ == "__main__":
    sys.exit(main())
