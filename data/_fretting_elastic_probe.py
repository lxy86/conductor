"""只解**线弹性**，看接触载荷本身是否已越过起裂门槛。

AT1 起裂条件：d 从 0 起动当 2ψ⁺ >= f·(3Gc/8ℓ) = f·σ_c²/E。
未疲劳时 f = 1，故门槛是 **ψ⁺ > σ_c²/(2E)**。
若弹性解就越过它，损伤是**静态**产生的、与疲劳无关 —— 那时「裂纹在滑移区萌生」
根本无从谈起，属 DECISIONS.md D-8 的情形。

**为什么可以直接解线弹性**：d ≡ 0 时 g(0) = 1 + k_res ≈ 1，能量退化为
g·ψ⁺ + ψ⁻ ≈ ψ⁺ + ψ⁻ = ψ（完整弹性能），于是 u 的问题**与分解无关、就是线弹性**。
分解只影响事后如何把 ψ 拆成 ψ⁺/ψ⁻。这样既避开了非线性 Newton
（实测在这个载荷下线搜索直接崩，SNES=-6），结果也更干净。
"""

import sys

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

sys.path.insert(0, "src")
from pfwire import loads as pfloads  # noqa: E402
from pfwire import mesh as pfmesh  # noqa: E402
from pfwire.fretting import _make_bcs, _traction_forms  # noqa: E402
from pfwire.material import contact_pair_properties, load_material  # noqa: E402
from pfwire.phasefield import energy_split, strain  # noqa: E402

comm = MPI.COMM_WORLD
material = load_material("configs/material_al1350.yaml")
E, nu = material.elastic.E, material.elastic.nu
sigma_c = material.fracture.sigma_c
threshold = sigma_c**2 / (2.0 * E)
if comm.rank == 0:
    print("σ_c = %.4g Pa；起裂门槛 ψ⁺ > σ_c²/(2E) = %.6e J/m³" % (sigma_c, threshold))

p0 = float(sys.argv[1]) if len(sys.argv) > 1 else 500.0e6
sigma_a = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0e6
E_star, R = contact_pair_properties(E, nu, 1.875e-3, "aluminum")
contact = pfloads.ContactLoads.from_peak_pressure(
    p0=p0, mu=0.7, Q_over_mu_P=0.6, E_star=E_star, R=R)
if comm.rank == 0:
    print("p0 = %.4g Pa（= %.2f σ_c）  σ_a = %.4g Pa  a = %.4f µm  c/a = %.4f"
          % (p0, p0 / sigma_c, sigma_a, contact.a * 1e6, contact.c / contact.a))

spec = pfmesh.FrettingSpec(
    contact_halfwidth=contact.a, W=6.4e-3, H=3.2e-3,
    refine_halfwidth_factor=3.0, refine_depth=150.0e-6,
    h_fine=2.0e-6, h_coarse=2.0e-4, transition_factor=8.0)
fr = pfmesh.fretting_mesh(spec, comm)
msh, tags = fr.mesh, fr.facet_tags
fdim = msh.topology.dim - 1
if comm.rank == 0:
    print("网格 %d 单元" % msh.topology.index_map(msh.topology.dim).size_global)

V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
phase = fem.Constant(msh, default_scalar_type(1.0))       # 峰值相位
sigma_b = fem.Constant(msh, default_scalar_type(sigma_a))
work = _traction_forms(msh, contact, phase, sigma_b, tags)

lmbda, mu_lame = material.elastic.lame
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
eps = ufl.sym(ufl.grad(u))
sig = 2.0 * mu_lame * eps + lmbda * ufl.tr(eps) * ufl.Identity(2)
a_form = ufl.inner(sig, ufl.sym(ufl.grad(v))) * ufl.dx
problem = LinearProblem(
    a_form, work(V), bcs=_make_bcs(msh, tags, fdim)(V),
    petsc_options_prefix="probe_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                   "pc_factor_mat_solver_type": "mumps"})
uh = problem.solve()
assert problem.solver.getConvergedReason() > 0

V0 = fem.functionspace(msh, ("DG", 0))
xc = fem.Function(V0)
xc.interpolate(fem.Expression(ufl.SpatialCoordinate(msh)[0],
                              V0.element.interpolation_points))
yc = fem.Function(V0)
yc.interpolate(fem.Expression(ufl.SpatialCoordinate(msh)[1],
                              V0.element.interpolation_points))
n = V0.dofmap.index_map.size_local
x, y = xc.x.array[:n], yc.x.array[:n]

for split in ("amor", "spectral"):
    psi_plus, _ = energy_split(strain(uh), material, split, 2)
    pf_fun = fem.Function(V0)
    pf_fun.interpolate(fem.Expression(psi_plus, V0.element.interpolation_points))
    psi = pf_fun.x.array[:n]

    local_max = float(psi.max()) if n else -1.0
    gmax = comm.allreduce(local_max, op=MPI.MAX)
    loc = ""
    if n and local_max == gmax:
        i = int(np.argmax(psi))
        loc = "x = %+.3f a, y = %.2f µm" % (x[i] / contact.a, y[i] * 1e6)
    loc = comm.allreduce(loc, op=MPI.MAX)
    over = comm.allreduce(int((psi > threshold).sum()), op=MPI.SUM)
    total = comm.allreduce(n, op=MPI.SUM)

    # 表面（最上一层单元）上 ψ⁺ 沿 x 的分布，看峰在中心还是边缘
    surf = y > -2.5e-6
    prof = []
    for lo in np.arange(-2.0, 2.01, 0.25):
        sel = surf & (x >= lo * contact.a) & (x < (lo + 0.25) * contact.a)
        val = comm.allreduce(
            float(psi[sel].max()) if sel.any() else -1.0, op=MPI.MAX)
        prof.append((lo + 0.125, val))

    if comm.rank == 0:
        print()
        print("--- split = %s ---" % split)
        print("  max ψ⁺ = %.6e = %.3f × 门槛   位置 %s" % (gmax, gmax / threshold, loc))
        print("  越门槛单元 %d / %d（%.2f%%）" % (over, total, 100.0 * over / max(total, 1)))
        print("  %s" % ("**弹性解已越门槛 -> 损伤是静态的，与疲劳无关**"
                        if gmax > threshold else "弹性解未越门槛（正常，起裂需靠 f(ᾱ) 下降）"))
        print("  表面 ψ⁺/门槛 沿 x/a 的分布：")
        line = "   "
        for xx, val in prof:
            if val < 0:
                continue
            line += "%+.2f:%.2f  " % (xx, val / threshold)
        print(line)
