"""判定 DamageGrid.sample 是否会外插出 >1 的值（result.json 里 max_damage_sampled=1.074）。

做法：在真实微动网格上造一个 P1 函数，把节点值**硬钳位到 [0,1]**，
再用生产同款采样网格取值。P1 单元内是节点值的凸组合，
所以任何 >1 的采样值只能来自「点被指到了不含它的单元」（外插），即采样 bug。
"""
import numpy as np
from mpi4py import MPI
from dolfinx import fem

from pfwire.mesh import FrettingSpec, fretting_mesh
from pfwire.failure import DamageGrid, GridSpec

comm = MPI.COMM_WORLD
E, nu = 69.0e9, 0.33
a = 1.162304347826087e-05   # 与 data/fretting_base/result.json 的 contact.a 同源
spec = FrettingSpec(W=6.4e-3, H=3.2e-3, refine_halfwidth_factor=3.0,
                    refine_depth=130.0e-6, h_fine=0.5e-6, h_coarse=2.0e-4,
                    contact_halfwidth=a, transition_factor=8.0)
mesh = fretting_mesh(spec, comm=comm).mesh
V = fem.functionspace(mesh, ("Lagrange", 1))
d = fem.Function(V)

# 一条过 (x0, 0) 、倾角 -20° 的陡裂纹带，宽度 ~ ell = 2 µm（和真实解同量级）
x0 = -1.1869e-05
ell = 2.0e-6


def f(x):
    t = np.tan(np.deg2rad(-20.0))
    s = (x[0] - x0 - t * x[1]) / ell          # 注意 x[1] 是 y（向下为负）
    return np.exp(-(s ** 2))


d.interpolate(f)
d.x.array[:] = np.clip(d.x.array, 0.0, 1.0)
d.x.scatter_forward()
n_own = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
own_max = comm.allreduce(float(np.max(d.x.array[:n_own])), op=MPI.MAX)
all_max = comm.allreduce(float(np.max(d.x.array)), op=MPI.MAX)

grid = DamageGrid(mesh, GridSpec(x_min=-3 * a, x_max=3 * a,
                                 depth_max=120.0e-6, spacing=0.5e-6))
vals = grid.sample(d)
if comm.rank == 0:
    print("a = %.4f um, 单元数(全局) = %d" % (a * 1e6, mesh.topology.index_map(2).size_global))
    print("节点最大值 owned = %.9f, owned+ghost = %.9f" % (own_max, all_max))
    print("采样最大值           = %.9f" % float(vals.max()))
    over = float(vals.max()) - 1.0 if all_max <= 1.0 + 1e-12 else np.nan
    print("超出量 = %.3e  ->  %s" % (over, "采样外插（BUG）" if over > 1e-9 else "采样正常"))
    n_over = int(np.count_nonzero(vals > all_max + 1e-12))
    print("采样点中超过节点最大值的个数 = %d / %d" % (n_over, vals.size))
