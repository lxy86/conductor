"""探针：跑圆柱–圆柱两阶段加载，dump 几个切向步的 (x, p, q, q/mu p) 剖面，
定位 c_fe 恒定（0.813a）的原因 —— 是黏滑测量口径问题还是 FE 解本身。"""
import pathlib, numpy as np
from dolfinx import default_scalar_type
from dolfinx.fem import Constant, dirichletbc, locate_dofs_topological
from mpi4py import MPI
from pfwire.contact import ContactModel, build_quarter_disks, fix_surface, QUARTER_DISKS
from pfwire.loads import hertz_half_width, mindlin_stick_half_width
from pfwire.material import contact_effective_modulus

comm = MPI.COMM_WORLD
R, E, nu, mu = 8.0, 200.0, 0.3, 0.3
out = pathlib.Path("/tmp/vm_profile")
mesh, ct, ft = build_quarter_disks(out, R=R, gap=0.0, res=0.3, order=1, quads=True, comm=comm)
model = ContactModel(mesh, ct, ft, E=E, nu=nu, order=1, tags=QUARTER_DISKS)
gtop = Constant(mesh, default_scalar_type((0.0, 0.0)))
dofs = locate_dofs_topological(model.V, mesh.topology.dim-1, ft.find(model.tags.load_facet))
tol = 1e-10
model.setup(bcs=[dirichletbc(gtop, dofs, model.V), fix_surface(model)],
    traction=None, friction=None, gamma_factor=100.0, theta=1.0,
    newton_options={"relaxation_parameter":1.0,"atol":1e-7,"rtol":1e-7,
        "convergence_criterion":"residual","max_it":200,"error_on_nonconvergence":True},
    petsc_options={"ksp_type":"preonly","pc_type":"lu","ksp_rtol":tol,"ksp_atol":tol})
estar = contact_effective_modulus(E, nu, E, nu)
for i in range(4):
    gtop.value[0]=0.0; gtop.value[1]=-0.20/4
    model.step(); model.prepare_next()
P = model.resultant(1)
a = hertz_half_width(P=P, R=R, E_star=estar)
if comm.rank==0: print("after normal: P=%.4g a=%.4g a/R=%.3f"%(P,a,a/R), flush=True)
gamg = {"matptap_via":"scalable","ksp_type":"gmres","ksp_rtol":1e-12,"ksp_atol":1e-12,
    "pc_type":"gamg","pc_mg_cycles":1,"mg_levels_ksp_type":"chebyshev","mg_levels_pc_type":"jacobi",
    "pc_gamg_type":"agg","pc_gamg_coarse_eq_limit":100,"pc_gamg_agg_nsmooths":1,
    "pc_gamg_threshold":1e-3,"pc_gamg_square_graph":2,"pc_gamg_reuse_interpolation":False,
    "ksp_initial_guess_nonzero":False,"ksp_norm_type":"unpreconditioned"}
model.set_friction_law(mu, gamma_factor=10.0, petsc_options=gamg)
dump_steps = {6, 10, 14}
for i in range(16):
    gtop.value[0]=0.14/28; gtop.value[1]=0.0
    model.step()
    xs, ps, qs = model.surface_tractions()
    Pn=model.resultant(1); Q=abs(model.resultant(0)); ratio=Q/(mu*Pn)
    a_i=hertz_half_width(P=Pn,R=R,E_star=estar)
    if comm.rank==0 and (i+1) in dump_steps:
        c_ref=mindlin_stick_half_width(a=a_i,Q_over_mu_P=ratio)
        print("\n=== step %d: Q/(muP)=%.4f  a_i=%.4g c_ref/a=%.4f ==="%(i+1,ratio,a_i,c_ref/a_i),flush=True)
        m=(ps>0.01*ps.max())&(np.abs(xs)<=a_i*1.3)
        xx,pp,qq=xs[m],ps[m],qs[m]
        o=np.argsort(xx); xx,pp,qq=xx[o],pp[o],qq[o]
        print("  pmax=%.4g  n_in=%d"%(ps.max(),m.sum()),flush=True)
        print("   x/a     p       q     q/(mu*p)",flush=True)
        for j in range(0,len(xx),max(1,len(xx)//24)):
            r=qq[j]/(mu*pp[j]) if pp[j]>0 else float('nan')
            print("  %+.3f  %7.3f %7.3f  %6.3f"%(xx[j]/a_i,pp[j],qq[j],r),flush=True)
    if i<15: model.prepare_next()
