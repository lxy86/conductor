# -*- coding: utf-8 -*-
"""零插桩探针：u 的 6.1e7 长在哪里（续 19 续 5）。

现场（ckpt_1.bp，粗网格串行 6.6.3 首跑）：
  max|u| = 6.139e7 m      （位移尺度 δ = 2.725e-5 m ⇒ 大 2.3e12 倍）
  max|q| = 3.430e9 Pa     （= 0.7 × 4.9e9 = mu*p ⇒ **Coulomb 投影本身正常**）
  max|d| = 0              （d 一步没动）
  alpha_bar = psi_prev = 3.29e31

q 与 p 的关系是对的 ⇒ 不是摩擦投影坏了。
问题只在 u。要判：那 6.1e7 是**局部一个点**（罚接触的伪影/自由 dof 跑飞）
还是**整体尺度**（边界条件丢了 ⇒ 刚体模态）。

判据：
  - 若只有少数 dof 大 ⇒ 局部伪影，看坐标是否落在接触斑/自由边界
  - 若大量 dof 同量级 ⇒ 刚体模态，BC 没起作用
同时报 x 方向与 y 方向的分量，看是法向（y）还是切向（x）跑飞。
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
    V = fem.functionspace(msh, ("CG", 1, (2,)))
    u = fem.Function(V, name="u")
    adios4dolfinx.read_function(bp, u, name="u")
    u.x.scatter_forward()

    arr = u.x.array.reshape(-1, 2)
    mag = np.linalg.norm(arr, axis=1)

    print("=== u 的分布 ===")
    qs = np.percentile(mag, [50, 90, 99, 99.9, 99.99, 100])
    for p, v in zip((50, 90, 99, 99.9, 99.99, 100), qs):
        print("   分位 %6.2f%% : %.4e" % (p, v))
    print("   max|ux| = %.4e, max|uy| = %.4e"
          % (np.max(np.abs(arr[:, 0])), np.max(np.abs(arr[:, 1]))))

    # 大于 1e-3 m（位移尺度的 37 倍）的 dof 有几个、在哪
    bad = np.where(mag > 1.0e-3)[0]
    print()
    print("   |u| > 1e-3 m 的顶点数：%d / %d" % (bad.size, mag.size))
    if bad.size:
        coords = V.tabulate_dof_coordinates()
        # CG1 向量：每个顶点 2 个 dof，坐标重复
        dof_xyz = coords[bad]
        print("   这些点的 x 范围 [%.4e, %.4e]"
              % (dof_xyz[:, 0].min(), dof_xyz[:, 0].max()))
        print("   这些点的 y 范围 [%.4e, %.4e]"
              % (dof_xyz[:, 1].min(), dof_xyz[:, 1].max()))
        print("   （线夹中心 x=0、接触面 y=0；丝半径/高度见 mesh）")
        print()
        print("   最大的 10 个：")
        order = np.argsort(mag)[::-1][:10]
        for i in order:
            c = coords[i]
            print("      x=%+.4e y=%+.4e  ux=%+.4e uy=%+.4e"
                  % (c[0], c[1], arr[i, 0], arr[i, 1]))

    # 参考：整体几何范围
    X = msh.geometry.x
    print()
    print("=== 网格范围 ===")
    print("   x [%.4e, %.4e], y [%.4e, %.4e]"
          % (X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
