"""Phase 5.3：塑性内变量（DG-2）经 adios4dolfinx 的往返与续跑连续性（硬约束 5）。

比较用**加权积分**而不是逐 dof：重建的网格 dof 编号与原网格不同，逐 dof 比对无意义；
而 ∫ w(x,y) f dx（w 非对称）对读乱序的场会立刻变样，并行下也成立。

续跑做法同 checkpoint.py：存 mesh + 场；读回的网格上**重新建**问题与塑性律。
"""
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

from pfwire.plasticity import J2Params, PlasticLaw

adios4dolfinx = pytest.importorskip("adios4dolfinx")

PRM = J2Params(E=69.0e9, nu=0.33, sigma_y0=167.0e6, H=5.0e9)
EPS_Y = PRM.sigma_y0 / PRM.E
# 加载-卸载-反向，左端全固定使塑性场非均匀（剪切约束）
HISTORY = [k * EPS_Y for k in (1, 2, 3, 4, 5, 6, 4, 2, 0, -2, -4, -6)]


def _setup(msh):
    """单位正方形，P2；左边**全固定**、右边 ux=δ（上下自由）-> 非均匀塑性场。"""
    V = fem.functionspace(msh, ("Lagrange", 2, (2,)))
    u = fem.Function(V)
    u.name = "u"
    law = PlasticLaw(msh, PRM)
    v = ufl.TestFunction(V)
    F = ufl.inner(law.sigma(u), law.strain(v)) * law.dx
    fdim = msh.topology.dim - 1
    left = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    right = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    g = fem.Constant(msh, np.float64(0.0))
    zero2 = fem.Constant(msh, np.array([0.0, 0.0]))
    bcs = [
        fem.dirichletbc(zero2, fem.locate_dofs_topological(V, fdim, left), V),
        fem.dirichletbc(g, fem.locate_dofs_topological(V.sub(0), fdim, right), V.sub(0)),
    ]
    prob = NonlinearProblem(F, u, bcs=bcs, petsc_options_prefix="pl_ckpt_",
                            petsc_options={"snes_type": "newtonls", "snes_linesearch_type": "bt",   # basic 在卸载步（弹/塑转换）会 Newton 不收敛，bt 稳
                                           "snes_atol": 1e-6, "snes_rtol": 1e-14, "snes_max_it": 30,
                                           "ksp_type": "preonly", "pc_type": "lu"})
    return u, law, g, prob


def _run(u, law, g, prob, deltas):
    for d in deltas:
        g.value = d
        prob.solve()
        assert prob.solver.getConvergedReason() > 0
        law.commit(u)


def _signature(msh, u, law):
    """场的加权积分指纹：与 dof 编号、分区无关。"""
    x = ufl.SpatialCoordinate(msh)
    w = [1.0, 1.0 + 3.0 * x[0] + 7.0 * x[1] * x[1]]
    dx = law.dx
    comm = msh.comm
    out = []
    for f in (law.p_n, law.ep_n[0], law.ep_n[1], law.ep_n[2], u[0], u[1]):
        for wi in w:
            out.append(comm.allreduce(fem.assemble_scalar(fem.form(wi * f * dx)), op=MPI.SUM))
    return np.array(out)


def _mesh0(comm=MPI.COMM_WORLD):
    return dmesh.create_unit_square(comm, 6, 6, dmesh.CellType.quadrilateral)


def test_run_reaches_plasticity_and_is_nonuniform():
    msh = _mesh0()
    u, law, g, prob = _setup(msh)
    _run(u, law, g, prob, HISTORY)
    p = msh.comm.allreduce(float(law.p_n.x.array.max()), op=MPI.MAX)
    pmin = msh.comm.allreduce(float(law.p_n.x.array.min()), op=MPI.MIN)
    assert p > 0.5 * EPS_Y and p > 3.0 * max(pmin, 1e-12)      # 进塑性且空间非均匀


def test_roundtrip_dg2_internal_variables(tmp_path):
    comm = MPI.COMM_WORLD
    path = comm.bcast(str(tmp_path / "ckpt.bp"), root=0)
    msh = _mesh0(comm)
    u, law, g, prob = _setup(msh)
    _run(u, law, g, prob, HISTORY[:8])
    ref = _signature(msh, u, law)
    assert np.abs(ref).max() > 0

    adios4dolfinx.write_mesh(path, msh)
    for f in (u, law.ep_n, law.p_n):
        adios4dolfinx.write_function(path, f)
    comm.Barrier()

    msh2 = adios4dolfinx.read_mesh(path, comm)
    u2, law2, _, _ = _setup(msh2)
    for f in (u2, law2.ep_n, law2.p_n):
        adios4dolfinx.read_function(path, f)
        f.x.scatter_forward()
    got = _signature(msh2, u2, law2)
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def test_restart_continuity_matches_uninterrupted_run(tmp_path):
    """连续跑 12 步  ==  跑 8 步 -> 存盘 -> 重建 -> 读回 -> 续跑 4 步（逐指纹 <= 1e-9）。"""
    comm = MPI.COMM_WORLD
    d = comm.bcast(str(tmp_path), root=0)
    k = 8

    msh = _mesh0(comm)
    u, law, g, prob = _setup(msh)
    _run(u, law, g, prob, HISTORY[:k])
    ck = d + "/ckpt.bp"
    adios4dolfinx.write_mesh(ck, msh)
    for f in (u, law.ep_n, law.p_n):
        adios4dolfinx.write_function(ck, f)
    _run(u, law, g, prob, HISTORY[k:])                       # 同一进程里接着跑完 -> 连续解
    ref = _signature(msh, u, law)
    comm.Barrier()

    msh2 = adios4dolfinx.read_mesh(ck, comm)
    u2, law2, g2, prob2 = _setup(msh2)
    for f in (u2, law2.ep_n, law2.p_n):
        adios4dolfinx.read_function(ck, f)
        f.x.scatter_forward()
    g2.value = HISTORY[k - 1]                                # 恢复到第 k 步的边界值，再走余下步
    _run(u2, law2, g2, prob2, HISTORY[k:])
    got = _signature(msh2, u2, law2)
    tol = 1e-9 * np.abs(ref) + 1e-12 * np.abs(ref).max()     # 近零的积分（对称抵消）只给绝对容差
    assert np.all(np.abs(got - ref) <= tol), np.abs(got - ref) / tol


@pytest.mark.parametrize("nproc", [2])
def test_roundtrip_and_restart_in_parallel(nproc, tmp_path):
    """同样的往返 + 续跑在 mpirun -n 2 下也通过（分区不同，指纹应一致）。"""
    import shutil
    import subprocess
    import sys

    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun is None:
        pytest.skip("无 mpirun")
    from mpi4py import MPI as _MPI
    if _MPI.COMM_WORLD.size > 1:
        pytest.skip("已在并行下运行")
    me = __file__
    cmd = [mpirun, "-n", str(nproc), sys.executable, "-m", "pytest", "-q", "-x", me,
           "-k", "roundtrip_dg2 or restart_continuity", "-p", "no:cacheprovider"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       env={**__import__("os").environ, "OMP_NUM_THREADS": "1"})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
