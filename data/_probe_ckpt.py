# -*- coding: utf-8 -*-
"""零插桩探针：从 ckpt_1.bp 读回爆掉的现场（续 19 续 4）。

粗网格 6.6.3 首跑（单体开）：
  - 延拓通过（W 偏差 0.089%、kappa 2.624%）
  - explicit_cycle 第一个周期后 max_alpha_bar = 3.29e31
  - 随后 d 的 SNES -6，‖F‖=2.9e19
  - **max_d = 0.0** ⇒ d 完全没动过，3.29e31 是在 d 还没参与时产生的

alpha_bar 只由 psi_plus 累积：
    alpha_bar += max(psi_now - psi_prev, 0)
⇒ psi_plus 本身到过 ~1e31。

psi_plus = 0.5*lmbda*<tr e>+^2 + mu*(<e1>+^2 + <e2>+^2)，是 u 的二次型。
铝 E=7e10、sigma~1e8 ⇒ 物理量级 ~1e5 J/m^3。
1e31 = 物理值的 1e26 倍 ⇒ u 被解飞了，不是物理。

**要查的假设（H1）**：
  `_solve_u_monolithic` 在 max_it 时**不抛异常**（只 warn，且需 log_iterations），
  于是显式循环拿到一个**未收敛的 u** 继续跑。
  交错路径 `_solve_u_once` 在 SNES<=0 时是 raise RuntimeError 的 —— 两条路径
  的失败处理不一致，单体这条会**静默吞掉**失败。

本探针直接读 checkpoint 里的场，不插桩、不重跑：
  max|u|、max|psi_plus_prev|、max|alpha_bar|、max|q|
若 max|u| ~ 1e10 而位移尺度是 1e-5 m，H1 成立。
"""

import sys

sys.path.insert(0, "/root/pf-wire")

import numpy as np


def main():
    import adios4dolfinx
    from mpi4py import MPI
    from dolfinx import fem, mesh

    comm = MPI.COMM_WORLD
    bp = "/root/pf-wire/data/clamp_fretting_coarse/ckpt_1.bp"

    msh = adios4dolfinx.read_mesh(bp, comm, ghost_mode=mesh.GhostMode.shared_facet)
    print("网格：%d 单元、%d 顶点" % (msh.topology.index_map(2).size_global,
                                     msh.topology.index_map(0).size_global))

    def _read(name, V):
        f = fem.Function(V, name=name)
        adios4dolfinx.read_function(bp, f, name=name)
        f.x.scatter_forward()
        a = f.x.array
        return float(np.max(np.abs(a))), int(np.count_nonzero(a))

    for name, V in (("u", fem.functionspace(msh, ("CG", 1, (2,)))),
                    ("d", fem.functionspace(msh, ("CG", 1))),
                    ("d_old", fem.functionspace(msh, ("CG", 1))),
                    ("alpha_bar", fem.functionspace(msh, ("DG", 2))),
                    ("psi_plus_prev", fem.functionspace(msh, ("DG", 2))),
                    ("q", fem.functionspace(msh, ("DG", 0)))):
        try:
            mx, nz = _read(name, V)
            print("  %-14s max|.| = %.6e   非零 %d" % (name, mx, nz))
        except Exception as e:
            print("  %-14s 读取失败：%s" % (name, type(e).__name__))

    print()
    print("参考：位移尺度 δ = 2.725e-05 m；psi+ 物理量级 ~1e5 J/m^3")
    print("若 max|u| 远大于 1e-3 m，则 u 被解飞 —— 对应「单体 Newton 未收敛被吞」")
    return 0


if __name__ == "__main__":
    sys.exit(main())
