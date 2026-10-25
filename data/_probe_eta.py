# -*- coding: utf-8 -*-
"""零插桩探针：残差地板是不是 smooth_clip 的 η 造成的（续 19 续 12）。

已有实测链（全部零插桩）：
  续 7  data/probe_coarse_cyc.log  相位 0.05 首个开摩擦相位发散，且静默
  续 9  data/probe_fd_coarse.log   Jacobian 是真导数（rel 6.2e-6 ∝ eps）
  续 10 data/probe_step.log        Newton 步 max|dx_q|=3.65e9 = 24 mu p；线性区 θ~1e-3
  续 11 data/probe_armijo.log      Jqq 对角 ≈ 1e-5（滑移区，dclip/dtrial≈1）；
        Armijo 回溯能接受（θ=2e-3），但残差**卡在 4.83e4 不动**，
        而同时 max|q| 稳定在 1.358e8 ≈ mu p、max|u|=7.16e-5 —— **物理上合理**。

⇒ 新假设 H3：F_q 的残差地板是 **smooth_clip 的 η 光滑化偏移**，不是数值噪声。
  解析：F_q = q − clip_η(q − ε_t s; μp, η) = 0 要求
        g + √(g² + η²) = 0  ⇒  √(g²+η²) = −g  ⇒  g ≤ 0 且 g²+η² = g²
   ⇒ **只有 η = 0 才有精确根**。η = 0.05·μp 时最小残差 ≈ η/2 = 0.025·μp，
     乘上接触面积后就是一个**不可能消掉的常数**。

  这不是数值病，是**方程本身没有根**：交错格式不受影响（q 由显式投影更新，
  不去驱动 F_q→0），单体格式把「残差 = 0」当收敛判据，于是永远到不了 tol。

本轮验证三件事：
  (1) 解析：在 η 与实测 μp 下，预测的地板量级是多少
  (2) 实测：Armijo 找到的状态是不是**交错 Uzawa 的不动点**
      （若 uzawa_update 在该状态几乎不改 q ⇒ 单体找到的就是真解）
  (3) 实测：把 η 调小（uzawa_eta_rel）残差地板是否同比缩小
      —— 这是 H3 的直接判据，且 η 只是光滑化参数、不动点与 η 无关
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
        return (np.concatenate([assemble_vector(fem.form(F_u)).array[:n_u],
                                assemble_vector(fem.form(F_q)).array[:n_q]]),
                assemble_vector(fem.form(F_q)).array[:n_q])

    def blocks():
        F_u = elastic(v)
        F_uc, F_q = clamp.monolithic_residual(u, v, q_t, None)
        F_u = F_u + F_uc
        b = clamp.monolithic_jacobian(u, du, v, q_t, q_du, fu_form=F_u)
        return tuple(_to_csr(assemble_matrix(fem.form(b[i][j]), bcs=[]))
                     for i in (0, 1) for j in (0, 1))

    def solve_armijo(max_it=40, tol=1e-3):
        """带 Armijo 回溯的块 Newton（探针内实现，与 solve_block_newton 同构）。"""
        hist = []
        for it in range(max_it):
            r, rq = residual()
            rn = float(np.linalg.norm(r[free]))
            hist.append(rn)
            if rn < tol:
                return True, it, rn, hist
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
                if float(np.linalg.norm(residual()[0][free])) < (1 - 1e-4 * th) * rn:
                    break
                th *= 0.5
        return False, max_it, hist[-1], hist

    case.loads.set_phase(np.sin(2 * np.pi * PHASE))
    clamp.set_tangential(AMP * np.sin(2 * np.pi * PHASE))

    # ---- (1) 解析预测 ----
    print("=== (1) η 光滑化地板的解析预测 ===", flush=True)
    print("    uzawa_eta_rel = %.4f（η = %.4f·μp）" % (clamp.uzawa_eta_rel, clamp.uzawa_eta_rel))
    print("    F_q = 0 要求 g + √(g²+η²) = 0，只有 η=0 才有解", flush=True)
    print("    ⇒ 单点残差下界 ≈ η/2 = %.4f·μp" % (clamp.uzawa_eta_rel / 2), flush=True)

    # ---- (2) Armijo 走到地板 ----
    print()
    print("=== (2) Armijo 块 Newton 走到地板 ===", flush=True)
    ok, it, rn, hist = solve_armijo(max_it=40, tol=1e-3)
    r, rq = residual()
    print("    converged=%s it=%d ||F_free||=%.4e" % (ok, it, rn), flush=True)
    print("    u 段 %.4e / q 段 %.4e" % (np.linalg.norm(r[:n_u][free[free < n_u]]),
                                        np.linalg.norm(rq)), flush=True)
    print("    max|u|=%.4e  max|q|=%.4e" % (np.max(np.abs(u.x.array)),
                                            np.max(np.abs(q.x.array))), flush=True)
    print("    残差历史：%s" % " ".join("%.2e" % x for x in hist[:8]), flush=True)

    # ---- (3) 该状态是不是交错 Uzawa 的不动点 ----
    print()
    print("=== (3) 该状态是不是 Uzawa 的不动点（q 改了多少）===", flush=True)
    q_before = q.x.array[:n_q].copy()
    rel = clamp.uzawa_update(u)
    q_after = q.x.array[:n_q].copy()
    dq = np.max(np.abs(q_after - q_before))
    print("    uzawa_update 的 rel_change = %.4e" % rel, flush=True)
    print("    max|Δq| = %.4e（相对 max|q|=%.3e 是 %.3e）"
          % (dq, np.max(np.abs(q_before)), dq / max(np.max(np.abs(q_before)), 1e-300)),
          flush=True)
    print("    ⇒ %s" % ("**是不动点**：单体找到的就是交错要的那个解" if rel < 1e-3
                        else "不是不动点，两者解不同（需进一步查）"), flush=True)

    # ---- (4) 缩小 η，看地板是否同比缩 ----
    print()
    print("=== (4) 改 η（只改光滑化，不动点与 η 无关）看地板 ===", flush=True)
    u.x.array[:] = u.x.array   # 保持当前状态作初值
    for eta_rel in (0.05, 0.005, 0.0005):
        clamp.uzawa_eta_rel = eta_rel
        ok2, it2, rn2, h2 = solve_armijo(max_it=25, tol=1e-3)
        print("    eta_rel=%.4f : converged=%-5s it=%2d ||F||=%.4e  max|q|=%.4e"
              % (eta_rel, ok2, it2, rn2, np.max(np.abs(q.x.array))), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
