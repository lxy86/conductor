"""诊断 d 的 SNES 为什么撞上迭代上限：打印残差历史，看它停在哪个量级。"""
import numpy as np
from mpi4py import MPI
from dolfinx import default_scalar_type, fem

from pfwire import mesh as pfmesh, material as mat, phasefield as pf

E, NU, GC, ELL = 210.0e9, 0.3, 2700.0, 15.0e-6
material = mat.Material(
    name="m", elastic=mat.ElasticProperties(E=E, nu=NU, plane_state="plane_strain"),
    fracture=mat.resolve_fracture_properties(E, NU, mode="from_gc_and_ell",
                                             Gc=GC, ell=ELL, dissipation="AT2"),
    sigma_u=2.0e9, alpha_T=None, sources={})
model = pf.PhaseFieldModel(material, split="spectral", dissipation="AT2")
spec = pfmesh.SentSpec(h_fine=ELL/4, h_coarse=3.0e-5, refine_halfheight=6.0e-5)
sent = pfmesh.sent_mesh(spec, MPI.COMM_WORLD)
msh, tags = sent.mesh, sent.facet_tags
fdim = msh.topology.dim - 1
n = msh.topology.index_map(msh.topology.dim).size_global
u_applied = fem.Constant(msh, default_scalar_type(0.0))

def make_bcs(V):
    return [fem.dirichletbc(np.zeros(2, dtype=default_scalar_type),
                fem.locate_dofs_topological(V, fdim, tags.find(pfmesh.SENT_TAG_BOTTOM)), V),
            fem.dirichletbc(u_applied,
                fem.locate_dofs_topological(V.sub(1), fdim, tags.find(pfmesh.SENT_TAG_TOP)), V.sub(1))]

solver = pf.PhaseFieldSolver(msh, model, bcs_u=make_bcs,
                             config=pf.SolverConfig(u_solver="newton", max_iter=500))
snes = solver._d_problem.solver
hist = []
snes.setMonitor(lambda s, it, norm: hist.append((it, norm)))
if MPI.COMM_WORLD.rank == 0:
    print("单元数 =", n, " d 自由度 =", solver.V_d.dofmap.index_map.size_global)
    print("atol=%g rtol=%g stol=%g max_it=%d" % (*snes.getTolerances()[:3], snes.getTolerances()[3]))

# 按 benchmark_sent_h4 的加载方式推进，直到 SNES 出问题
stages = [(2.0e-7, 26), (2.0e-8, 15), (1.0e-9, 60)]
u_val, step = 0.0, 0
for du, ns in stages:
    for _ in range(ns):
        u_val += du; step += 1
        u_applied.value = u_val
        hist.clear()
        try:
            r = solver.solve_step()
        except Exception as exc:
            if MPI.COMM_WORLD.rank == 0:
                print("步 %d 抛异常: %s" % (step, exc))
            r = None
        reason = snes.getConvergedReason()
        if MPI.COMM_WORLD.rank == 0 and (step % 10 == 0 or reason <= 0 or (r and not r.converged)):
            print("步 %3d u=%.4e reason=%d  最后一次 SNES 残差历史(%d 条): %s"
                  % (step, u_val, reason, len(hist),
                     " ".join("%.2e" % v for _, v in hist[:3]) + " ... " +
                     " ".join("%.2e" % v for _, v in hist[-4:])))
        if r is not None and not r.converged:
            if MPI.COMM_WORLD.rank == 0:
                print("  -> 交错未收敛：%s" % r.failure_mode)
            raise SystemExit(0)
        solver.commit_step()
