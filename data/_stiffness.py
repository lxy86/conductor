"""单步弹性刚度收敛：h_coarse 变细时 k 是否下降并收敛？

P1 三角形偏刚，体区网格过粗会高估刚度、进而高估峰值载荷。
只需一个很小的位移步（d 几乎为 0），故极便宜。
"""
import numpy as np
from mpi4py import MPI
from dolfinx import default_scalar_type, fem

from pfwire import mesh as pfmesh
from pfwire import material as mat
from pfwire import phasefield as pf

E, NU, GC, ELL = 210.0e9, 0.3, 2700.0, 7.5e-6
THICK = 1.0e-3
U_STEP = 2.0e-7  # m，很小，d 可忽略

material = mat.Material(
    name="miehe",
    elastic=mat.ElasticProperties(E=E, nu=NU, plane_state="plane_strain"),
    fracture=mat.resolve_fracture_properties(
        E, NU, mode="from_gc_and_ell", Gc=GC, ell=ELL, dissipation="AT2"),
    sigma_u=2.824e9, alpha_T=None, sources={},
)
model = pf.PhaseFieldModel(material, split="spectral", dissipation="AT2")

if MPI.COMM_WORLD.rank == 0:
    print("%10s %10s %8s %14s %10s" % ("h_coarse", "h_fine", "cells", "k[N/m]", "F(5.6um)"))
for h_coarse, h_fine in ((3.0e-5, 3.75e-6), (1.5e-5, 3.75e-6), (7.5e-6, 3.75e-6),
                         (3.0e-5, 1.875e-6)):
    spec = pfmesh.SentSpec(h_fine=h_fine, h_coarse=h_coarse, refine_halfheight=6.0e-5)
    sent = pfmesh.sent_mesh(spec, MPI.COMM_WORLD)
    msh, tags = sent.mesh, sent.facet_tags
    fdim = msh.topology.dim - 1
    u_applied = fem.Constant(msh, default_scalar_type(U_STEP))

    def make_bcs(V, tags=tags, u_applied=u_applied):
        Vy = V.sub(1)
        return [
            fem.dirichletbc(np.zeros(2, dtype=default_scalar_type),
                            fem.locate_dofs_topological(V, fdim, tags.find(pfmesh.SENT_TAG_BOTTOM)), V),
            fem.dirichletbc(u_applied,
                            fem.locate_dofs_topological(Vy, fdim, tags.find(pfmesh.SENT_TAG_TOP)), Vy),
        ]

    def make_probe(V, tags=tags):
        one = fem.Constant(msh, default_scalar_type(1.0))
        return [fem.dirichletbc(one,
                fem.locate_dofs_topological(V.sub(1), fdim, tags.find(pfmesh.SENT_TAG_TOP)), V.sub(1))]

    solver = pf.PhaseFieldSolver(msh, model, bcs_u=make_bcs,
                                 config=pf.SolverConfig(u_solver="newton", max_iter=200))
    probe = solver.make_probe(make_probe)
    res = solver.solve_step()
    F = solver.reaction(probe) * THICK
    k = F / U_STEP
    n = msh.topology.index_map(msh.topology.dim).size_global
    if MPI.COMM_WORLD.rank == 0:
        print("%10.3g %10.4g %8d %14.5e %10.1f  (max_d=%.1e, conv=%s)"
              % (h_coarse, h_fine, n, k, k * 5.6e-6, res.max_d, res.converged))
