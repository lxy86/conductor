# -*- coding: utf-8 -*-
"""新增 src/pfwire/monolithic.py：手工块 Newton 的生产实现（D-68 问题 33 续 18）。

为什么不用 dolfinx 的多 form ``NonlinearProblem``（续 17 坑 7，两个叠加的 bug）：
  (a) ``bcs_by_block`` 里 ``V.contains(bc.function_space)`` ——
      ``bc.function_space`` 是 **cpp 对象**，Python 的 ``contains`` 去取
      ``._cpp_object`` 抛 AttributeError。该异常在 PETSc 的 SNES 回调里
      **被吞掉** ⇒ SNES 什么都不做就返回 -3、**it=0**、u/q 精确不变。
      这是最难查的失效形态：solver 说「没收敛」，实际是「根本没开始」。
  (b) 即使 patch 掉 (a)，BC 的 dof 编号仍是各空间自己的编号，
      而 ``set_bc`` 当成拼接向量的全局编号 ⇒ q 的 BC 全部落到 u 段。

而四个块**各自都能正常装配**（实测 size 全对、中心差分一致），
故这里自己拼块矩阵、自己施加 BC、自己调线性求解器。

设计要点
--------
1. **块三角/满块都支持**：本实现对任意 2x2 块系统通用，不假设哪块为零。
2. **BC 在各自的编号空间里施加**（续 9/续 17 坑 6 的教训）：
   u 的 BC 用 u 的编号、q 的 BC 用 q 的编号，拼进全局 [u|q] 时**加偏移**。
3. **残差只在自由 dof 上量**：BC dof 上是**支座反力**，不是残差
   （续 17 实测：BC 段 5.25e3、自由段 1.02e-4，混在一起会误判不收敛）。
4. **不解非接触单元的 q 方程**（B 方案）：非接触单元直接钉 0，
   不进自由集、不进 Jacobian —— 这是子空间限定，不是罚正则。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector

__all__ = ["BlockNewtonConfig", "BlockNewtonResult", "solve_block_newton"]


@dataclass
class BlockNewtonConfig:
    """手工块 Newton 的配置。

    Attributes
    ----------
    tol : 收敛容差（自由 dof 上的残差 2-范数）。
    max_it : 最大迭代数。
    linear_solver : "spsolve"（串行直接解）或 "gmres"。
    """

    tol: float = 1.0e-8
    max_it: int = 50
    linear_solver: str = "spsolve"
    #: 每步打印残差（默认 False，避免刷屏）
    verbose: bool = False


@dataclass
class BlockNewtonResult:
    """块 Newton 的结果。

    Attributes
    ----------
    converged : 是否收敛。
    iterations : 实际迭代数。
    residual : 最终残差（自由 dof 上）。
    history : 每步残差列表。
    reason : "converged" / "max_it" / "linear_failed" / "nan"。
    """

    converged: bool
    iterations: int
    residual: float
    history: list = field(default_factory=list)
    reason: str = "converged"


def _to_csr(petsc_mat) -> sp.csr_matrix:
    """PETSc Mat -> scipy csr，**用 ``getValuesCSR`` 永不稠密化**。

    ⚠⚠⚠ **不要用 ``getValues(range(r), range(c))``**（续 18 实测的坑）：
    petsc4py 会为它**一次性分配 r×c 的稠密数组**。粗网格 6.6.3 上
    ``J_uu`` 是 61910×61910 ⇒ **30.66 GB**，分配失败后 numpy 报的却是

        ValueError: negative dimensions are not allowed

    （整数溢出成负维数）—— 完全指不到病因。而 tiny 网格 10322×10322
    只要 0.85 GB，所以**tiny 绿、粗网格红**，且「四块尺寸全对」
    （尺寸是 PETSc 报的，与取数据无关）。这个坑花了两轮才定位。

    ``getValuesCSR`` 直接给 (indptr, indices, data) 三元组，
    实测与 ``getValues`` 的结果**逐位一致**（最大差 0.000e+00）。
    """
    petsc_mat.assemble()
    r, c = petsc_mat.getLocalSize()
    if r <= 0 or c <= 0:
        raise ValueError(
            "Jacobian 块是空的（本地尺寸 %d x %d）：几乎肯定是在接触斑建立前"
            "调了单体路径。延拓期 μ=0（set_mu(None)）时 q ≡ 0、没有 Coulomb "
            "投影要解，应走交错格式。判据用 has_friction（运行时）而非 "
            "has_friction_at_build（编译期）。" % (r, c))
    indptr, indices, data = petsc_mat.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=(r, c))


def solve_block_newton(u, q, *, fu_form, fq_form, juu, juq, jqu, jqq,
                       u_bc_dofs, q_bc_dofs, n_u, n_q,
                       config: BlockNewtonConfig | None = None) -> BlockNewtonResult:
    """解 (u, q) 混合非线性系统的**手工块 Newton**。

    Parameters
    ----------
    u, q : 两个未知量场（CG1 向量 / DG0）。**就地更新**。
    fu_form, fq_form : 两条残差 form（已编译的 ``fem.form``）。
    juu, juq, jqu, jqq : 四块 Jacobian form（**未编译**的 UFL 表达式即可，
        本函数内部编译）。
    u_bc_dofs, q_bc_dofs : 各自的 DirichletBC dof（**在各自空间的编号里**）。
    n_u, n_q : 两个空间的本地规模（不含 ghost）。
    config : 见 :class:`BlockNewtonConfig`。

    Notes
    -----
    **不假设任何一块为零**：满块与块三角都支持。

    ⚠ **BC dof 的编号空间**（续 9 / 续 17 坑 6）：
    ``u_bc_dofs`` 必须是 u 空间的编号、``q_bc_dofs`` 必须是 q 空间的编号。
    拼进全局 ``[u | q]`` 时 q 段要 **+n_u 偏移**。
    混用会让 BC 钉到另一个块上 —— 那不会报错，只会让系统奇异或解错。
    """
    cfg = config or BlockNewtonConfig()
    u_bc = np.asarray(u_bc_dofs, dtype=np.int64)
    q_bc = np.asarray(q_bc_dofs, dtype=np.int64) + n_u   # 全局偏移
    g_bc = np.concatenate([u_bc, q_bc]) if u_bc.size + q_bc.size else np.array([], dtype=np.int64)
    free = np.setdiff1d(np.arange(n_u + n_q, dtype=np.int64), g_bc)

    fu_form = fem.form(fu_form) if not hasattr(fu_form, "_cpp_object") else fu_form
    fq_form = fem.form(fq_form) if not hasattr(fq_form, "_cpp_object") else fq_form
    j_forms = [[fem.form(juu), fem.form(juq)], [fem.form(jqu), fem.form(jqq)]]

    history: list[float] = []
    for it in range(cfg.max_it):
        # ---- 残差（只量自由 dof）----
        r = np.concatenate([assemble_vector(fu_form).array[:n_u],
                            assemble_vector(fq_form).array[:n_q]])
        rn = float(np.linalg.norm(r[free]))
        history.append(rn)
        if cfg.verbose:
            print("  block-newton it %2d : ||F_free|| = %.6e" % (it, rn))
        if rn < cfg.tol:
            return BlockNewtonResult(True, it, rn, history, "converged")

        # ---- 拼块 Jacobian（**全程 CSR**，续 18 实测）----
        # ⚠ 不要用 format="lil" 再逐行赋值：粗网格是 92390 x 92390，
        # lil 的逐行 Python 循环 + 转置会慢到不可用（tiny 上看不出来）。
        # 改成「先 CSR 拼块，再用稀疏方式把 BC 行替换成单位行」。
        Juu = _to_csr(assemble_matrix(j_forms[0][0], bcs=[]))
        Juq = _to_csr(assemble_matrix(j_forms[0][1], bcs=[]))
        Jqu = _to_csr(assemble_matrix(j_forms[1][0], bcs=[]))
        Jqq = _to_csr(assemble_matrix(j_forms[1][1], bcs=[]))
        J = sp.bmat([[Juu, Juq], [Jqu, Jqq]], format="csr")

        # ---- 施加 BC：把 BC 行替换成单位行（全程稀疏）----
        # ⚠ 不用 lil 逐行赋值（92390 阶上慢到不可用），也不用
        # ``J.multiply(keep[:,None])``（会生成一个全新的同规模稀疏阵）。
        # 直接用 CSR 三元组：保留**自由行**的非零元，再叠加 BC 位置的单位元。
        if g_bc.size:
            n = n_u + n_q
            keep_row = np.ones(n, dtype=bool)
            keep_row[g_bc] = False
            # 逐元过滤：row_of_entry < n 时保留
            row_of = np.repeat(np.arange(n), np.diff(J.indptr))
            sel = keep_row[row_of]
            data = J.data[sel]
            indices = J.indices[sel]
            # 重建 indptr（自由行保留原行元数，BC 行 0 个）
            counts = np.diff(J.indptr).copy()
            counts[g_bc] = 0
            indptr = np.concatenate([[0], np.cumsum(counts)])
            J = sp.csr_matrix((data, indices, indptr), shape=(n, n))
            # 叠加 BC 位置的单位元
            I_bc = sp.csr_matrix(
                (np.ones(g_bc.size), (g_bc, g_bc)), shape=(n, n))
            J = (J + I_bc).tocsr()
        Jf = J[free][:, free]

        # ---- 线性求解（只在自由子块上）----
        try:
            if cfg.linear_solver == "gmres":
                dxf, info = spla.gmres(Jf, -r[free], atol=1e-12, rtol=1e-10)
            else:
                dxf = spla.spsolve(Jf, -r[free])
        except Exception:
            return BlockNewtonResult(False, it, rn, history, "linear_failed")

        if not np.all(np.isfinite(dxf)):
            return BlockNewtonResult(False, it, rn, history, "nan")

        dx = np.zeros(n_u + n_q)
        dx[free] = dxf
        u.x.array[:n_u] += dx[:n_u]
        u.x.scatter_forward()
        q.x.array[:n_q] += dx[n_u:]
        q.x.scatter_forward()

    return BlockNewtonResult(False, cfg.max_it, history[-1] if history else float("nan"),
                             history, "max_it")
