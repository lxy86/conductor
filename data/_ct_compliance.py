"""CT 弹性柔度对拍 ASTM E647 —— 把「弹性模型错」与「断裂模型错」分开。

P_mon 偏高 28~42% 时，病可能在两处：
  (a) 弹性侧：几何、边界条件、E/nu、面力换算 —— 会同时让柔度偏小（偏刚）；
  (b) 断裂侧：G_c/ℓ/AT2 或网格分辨率 —— 只影响峰值，不影响弹性柔度。
本脚本只解**线弹性**（d ≡ 0），量载荷线柔度 v/P 并与 ASTM E647 的标定式比：

    v E B / P = [(1+a)/(1-a)]^2 (2.163 + 12.219a - 20.065a^2
                                 - 0.9925a^3 + 20.609a^4 - 9.9314a^5),  a = a/W

（Saxena & Hudak 的 C(T) 载荷线柔度式，ASTM E647 用它做 compliance 反推裂纹长度。）
若柔度对得上而 P_mon 不对，病在 (b)；若柔度就偏刚，病在 (a)。

v 的口径：ASTM 的 v 是两个销孔的**相对**位移。本脚本下销孔全约束，
故 v = 上销孔边界上 u_y 的平均值。注意这本身就与 ASTM 有差异 ——
全约束抑制了下孔的局部椭圆化，会让我们偏刚一点；本脚本正是要量这个「一点」有多大。
"""

import sys

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

from pfwire import mesh as pfmesh
from pfwire.material import lame_parameters, load_material

THICKNESS = 1.0e-3
P_APPLIED = 148.6  # N，Tab. 2 反解的 P_mon，取它便于同时看位移量级


def astm_compliance_factor(alpha: float) -> float:
    """ASTM E647 C(T) 载荷线柔度的无量纲因子 v E B / P。"""
    r = (1.0 + alpha) / (1.0 - alpha)
    poly = (
        2.163
        + 12.219 * alpha
        - 20.065 * alpha**2
        - 0.9925 * alpha**3
        + 20.609 * alpha**4
        - 9.9314 * alpha**5
    )
    return r**2 * poly


comm = MPI.COMM_WORLD
is_root = comm.rank == 0
h_fine = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0e-4

spec = pfmesh.CtSpec(
    h_fine=h_fine, h_coarse=1.5e-3, refine_halfheight=1.0e-3,
    refine_x_margin=2.0e-3, notch_taper_length=1.5625e-3,
)
ct = pfmesh.ct_mesh(spec, comm)
msh, tags = ct.mesh, ct.facet_tags
fdim = msh.topology.dim - 1

material = load_material("configs/material_carrara.yaml")
E, nu = material.elastic.E, material.elastic.nu
lmbda, mu = lame_parameters(E, nu, material.elastic.plane_state)

ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
one = fem.Constant(msh, default_scalar_type(1.0))
L_top = comm.allreduce(
    fem.assemble_scalar(fem.form(one * ds(pfmesh.CT_TAG_HOLE_TOP))), op=MPI.SUM
)

V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)


def eps(w):
    return ufl.sym(ufl.grad(w))


sigma = 2.0 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(2)
a_form = ufl.inner(sigma, eps(v)) * ufl.dx
traction = P_APPLIED / (THICKNESS * L_top)
t_vec = ufl.as_vector([fem.Constant(msh, default_scalar_type(0.0)),
                       fem.Constant(msh, default_scalar_type(traction))])
L_form = ufl.inner(t_vec, v) * ds(pfmesh.CT_TAG_HOLE_TOP)

lower = tags.find(pfmesh.CT_TAG_HOLE_BOTTOM)
bcs = [
    fem.dirichletbc(
        np.zeros(msh.geometry.dim, dtype=default_scalar_type),
        fem.locate_dofs_topological(V, fdim, lower),
        V,
    )
]
problem = LinearProblem(
    a_form, L_form, bcs=bcs, petsc_options_prefix="ct_comp_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                   "pc_factor_mat_solver_type": "mumps"},
)
uh = problem.solve()
assert problem.solver.getConvergedReason() > 0

# 上孔边界上 u_y 的平均值 = 载荷线位移
uy_int = comm.allreduce(
    fem.assemble_scalar(fem.form(uh[1] * ds(pfmesh.CT_TAG_HOLE_TOP))), op=MPI.SUM
)
v_fe = uy_int / L_top

factor_fe = v_fe * E * THICKNESS / P_APPLIED
factor_astm = astm_compliance_factor(spec.a0_over_W)
v_astm = factor_astm * P_APPLIED / (E * THICKNESS)

if is_root:
    print("网格 %d 单元  h_fine = %.4g m = ell/%.2f"
          % (msh.topology.index_map(msh.topology.dim).size_global, h_fine,
             material.fracture.ell / h_fine))
    print("上孔周长（实测）= %.6f m" % L_top)
    print("施加 P = %.1f N -> 面力 %.4g Pa" % (P_APPLIED, traction))
    print()
    print("载荷线位移  FE   v = %.6e m" % v_fe)
    print("载荷线位移  ASTM v = %.6e m" % v_astm)
    print("无量纲柔度  FE   vEB/P = %.4f" % factor_fe)
    print("无量纲柔度  ASTM vEB/P = %.4f" % factor_astm)
    print()
    print("FE/ASTM = %.4f  ->  FE 比 ASTM %s %.1f%%"
          % (factor_fe / factor_astm,
             "偏刚" if factor_fe < factor_astm else "偏柔",
             100.0 * abs(factor_fe / factor_astm - 1.0)))
