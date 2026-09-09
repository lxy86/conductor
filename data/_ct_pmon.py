"""CT 单调力控加载，估 P_mon —— 验证 2 开跑前的**独立锚点**。

为什么这个锚点可靠：Carrara Tab. 2 给的是 Pmax/Pmon 百分比，而 Pmax = ΔP
（参考算例 Pmin = 0），故可反解 P_mon：
    ΔP = 20  N -> 13.5% -> P_mon = 148.1 N
    ΔP = 140 N -> 94.2% -> P_mon = 148.6 N
两个独立行给出同一个值，说明这是原文自洽的量，可以当靶子。
（原文 Fig. 12b 的纵轴标 "P [kN]" 与此差 1000 倍，按 Tab. 2 应为 N。）

做法：力控单调加载。力控下越过极限点就没有平衡解，故**最后一个收敛的 P**
即 P_mon 的下界估计；步长取够细时它就逼近 P_mon。

这不是 PLAN 的验收项，只是开长算例前的自检：几何、边界条件、面力换算、
材料卡这四件里任何一件错了，P_mon 都不会落在 148 N 附近。
"""

import sys
import time

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem
from mpi4py import MPI

from pfwire import mesh as pfmesh
from pfwire.material import load_material
from pfwire.phasefield import PhaseFieldModel, PhaseFieldSolver, SolverConfig

P_TARGET = 148.1  # N，由 Tab. 2 反解
THICKNESS = 1.0e-3

comm = MPI.COMM_WORLD
is_root = comm.rank == 0

h_fine = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0e-4
# 锥收长度 L 与缺口半高 nh = W/32 决定尖端半角 atan(nh/L)：
#   L = 1.5625 mm -> 45°（默认，钝）    L = 6.25 mm -> 14°（细长，尖）
# 即**锥收越长尖端越尖**。缺口越钝、应力集中越弱、峰值载荷越高。
taper = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5625e-3
dP = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
spec = pfmesh.CtSpec(
    h_fine=h_fine,
    h_coarse=1.5e-3,
    refine_halfheight=1.0e-3,
    refine_x_margin=2.0e-3,
    notch_taper_length=taper,
)
t0 = time.perf_counter()
ct = pfmesh.ct_mesh(spec, comm)
msh, tags = ct.mesh, ct.facet_tags
fdim = msh.topology.dim - 1
n_cells = msh.topology.index_map(msh.topology.dim).size_global
if is_root:
    print("网格 %d 单元，h_fine = %.4g m = ell/%.2f，耗时 %.1f s"
          % (n_cells, h_fine, 2.0e-4 / h_fine, time.perf_counter() - t0))
    print("锥收 L = %.4g m -> 尖端半角 %.1f°，载荷步 %.4g N"
          % (taper, np.degrees(np.arctan(spec.notch_half_height / taper)), dP))

material = load_material("configs/material_carrara.yaml")
model = PhaseFieldModel(material=material, split="spectral", k_res=1.0e-6,
                        dissipation="AT2")

ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
hole_length = comm.allreduce(
    fem.assemble_scalar(
        fem.form(fem.Constant(msh, default_scalar_type(1.0))
                 * ds(pfmesh.CT_TAG_HOLE_TOP))),
    op=MPI.SUM,
)
traction = fem.Constant(msh, default_scalar_type(0.0))
zero = fem.Constant(msh, default_scalar_type(0.0))


def make_bcs(V):
    lower = tags.find(pfmesh.CT_TAG_HOLE_BOTTOM)
    zero_vec = np.zeros(msh.geometry.dim, dtype=default_scalar_type)
    return [fem.dirichletbc(zero_vec, fem.locate_dofs_topological(V, fdim, lower), V)]


def make_work(V):
    v = ufl.TestFunction(V)
    return ufl.inner(ufl.as_vector([zero, traction]), v) * ds(pfmesh.CT_TAG_HOLE_TOP)


solver = PhaseFieldSolver(
    msh, model, bcs_u=make_bcs, external_work=make_work,
    config=SolverConfig(u_solver="newton", tol_d=1.0e-4, max_iter=300),
)


def set_P(P):
    traction.value = P / (THICKNESS * hole_length)


# 先粗扫到失败，再在最后一个成功点附近细扫
last_ok = 0.0
t0 = time.perf_counter()
for P in np.arange(dP, 320.0, dP):
    set_P(float(P))
    res = solver.solve_step()
    if not res.converged:
        if is_root:
            print("P = %6.1f N 未收敛（%s）" % (P, (res.failure_mode or "")[:60]))
        break
    solver.commit_step()
    last_ok = float(P)
    if is_root:
        print("P = %6.1f N  max_d = %.4f  交错 %d 次  已用 %.0f s"
              % (P, res.max_d, res.iterations, time.perf_counter() - t0))

if is_root:
    print()
    print("最后收敛的 P = %.1f N   靶值 P_mon ≈ %.1f N   偏差 %+.1f%%"
          % (last_ok, P_TARGET, 100.0 * (last_ok - P_TARGET) / P_TARGET))
    print("（力控下越过极限点即无解，故 P_mon ∈ (%.1f, %.1f]）"
          % (last_ok, last_ok + dP))
