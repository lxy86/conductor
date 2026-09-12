"""决定 checkpoint 设计：read_function 能否配**重新生成**的网格？

这关系到续跑时是 (A) 用 gmsh 重建网格再读函数，还是 (B) 直接从 .bp 读网格。
(B) 一定对但要多存网格；(A) 省事但依赖「重建的网格与原网格 input 顺序一致」。
不猜，跑一遍。
"""
import sys, pathlib
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh
import adios4dolfinx

comm = MPI.COMM_WORLD
P = pathlib.Path("/root/pf-wire/data/_ckpt_probe.bp")
mode = sys.argv[1]


def build():
    return dmesh.create_rectangle(
        comm, [np.array([-1.0e-3, -1.0e-3]), np.array([1.0e-3, 0.0])],
        [12, 9], cell_type=dmesh.CellType.triangle)


def f_expr(x):
    return np.sin(3.0e3 * x[0]) * np.cos(2.0e3 * x[1]) + 2.0


if mode == "write":
    msh = build()
    V = fem.functionspace(msh, ("Lagrange", 1))
    g = fem.Function(V, name="d")
    g.interpolate(f_expr)
    adios4dolfinx.write_mesh(P, msh)
    adios4dolfinx.write_function(P, g)
    if comm.rank == 0:
        print("写入完成")
else:
    if mode == "read_regen":
        msh = build()                      # 方案 A：重新生成
    else:
        msh = adios4dolfinx.read_mesh(P, comm)   # 方案 B：从 .bp 读
    V = fem.functionspace(msh, ("Lagrange", 1))
    g = fem.Function(V, name="d")
    try:
        adios4dolfinx.read_function(P, g)
        g.x.scatter_forward()
    except Exception as e:
        if comm.rank == 0:
            print("%s: 读取抛异常 -> %s: %s" % (mode, type(e).__name__, str(e)[:120]))
        raise SystemExit
    ref = fem.Function(V)
    ref.interpolate(f_expr)
    n = V.dofmap.index_map.size_local
    err = comm.allreduce(
        float(np.max(np.abs(g.x.array[:n] - ref.x.array[:n]))), op=MPI.MAX)
    if comm.rank == 0:
        print("%s: 最大逐点误差 = %.3e  -> %s" % (mode, err, "一致" if err < 1e-12 else "**不一致**"))
