"""Phase 0 环境验收（CLAUDE.md 验收命令：pytest tests/test_env.py）。

测试意图：
1. 版本与标量类型：dolfinx 0.11.x、PETSc 为 real 变体（不是 complex）；
2. 全部必需包可 import，且 API 名字与本项目代码假定一致
   （0.11 把 dolfinx.io.gmshio 改成了 dolfinx.io.gmsh，这类改名必须被测住）；
3. gmsh -> dolfinx 链路能在**含中文的 Windows 路径**（/mnt/c/...）上工作；
4. adios4dolfinx 0.11.0（PyPI 版）与 dolfinx 0.11.0 真能读写 checkpoint，
   这是硬约束 5「断点续跑」的地基；
5. 线弹性悬臂梁冒烟：MUMPS 直接解 + 端部挠度与梁理论误差 < 1%（PLAN.md Phase 0）。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

REPO = Path(__file__).resolve().parents[1]
TMP = REPO / "data" / "_tmp_env_test"  # 故意放在含中文的仓库路径下（见测试意图 3）


@pytest.fixture(scope="module")
def tmp_in_repo() -> Path:
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


# --------------------------------------------------------------------------
# 1-2. 版本、标量类型、依赖
# --------------------------------------------------------------------------
def test_dolfinx_version_is_0_11():
    import dolfinx

    assert dolfinx.__version__.startswith("0.11"), (
        "本项目按 dolfinx 0.11.0 的 API 写（DECISIONS.md D-0），"
        f"当前是 {dolfinx.__version__}"
    )


def test_petsc_is_real_not_complex():
    assert np.issubdtype(PETSc.ScalarType, np.floating), (
        f"需要 real PETSc，当前 ScalarType={PETSc.ScalarType}"
    )


def test_required_packages_importable():
    import adios4dolfinx  # noqa: F401
    import basix  # noqa: F401
    import gmsh  # noqa: F401
    import pyvista  # noqa: F401
    import scipy  # noqa: F401
    import ufl  # noqa: F401
    import yaml  # noqa: F401

    assert adios4dolfinx.__version__.startswith("0.11")


def test_dolfinx_api_names_this_project_relies_on():
    """把 0.9 -> 0.11 的关键改名/签名变化测住，避免凭记忆写出 legacy 代码。"""
    import inspect

    from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
    from dolfinx.io import gmsh as gmshio  # 0.9 里叫 dolfinx.io.gmshio

    assert hasattr(gmshio, "model_to_mesh")
    # 0.10+ 起 petsc_options_prefix 是必需关键字参数
    for cls in (LinearProblem, NonlinearProblem):
        params = inspect.signature(cls.__init__).parameters
        assert "petsc_options_prefix" in params
        assert params["petsc_options_prefix"].default is inspect.Parameter.empty


def test_snes_vi_solver_available():
    """Phase 1.1 的 d 场不可逆要用 SNES vinewtonrsls + 变量上下界。"""
    snes = PETSc.SNES().create(MPI.COMM_SELF)
    snes.setType("vinewtonrsls")
    assert snes.getType() == "vinewtonrsls"
    assert hasattr(snes, "setVariableBounds")
    snes.destroy()


# --------------------------------------------------------------------------
# 3. gmsh -> dolfinx，中文路径
# --------------------------------------------------------------------------
def test_gmsh_to_dolfinx_on_unicode_path(tmp_in_repo: Path):
    import gmsh
    from dolfinx.io import gmsh as gmshio

    msh_file = tmp_in_repo / "unit_square.msh"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("unit_square")
        gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 1.0, 1.0, tag=1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [1], tag=10)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.25)
        gmsh.model.mesh.generate(2)

        data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        msh = data.mesh
        msh.topology.create_connectivity(2, 0)
        n_cells = msh.topology.index_map(2).size_global
        assert n_cells > 0

        # 写到中文路径再读回，确认 gmsh 的文件 IO 不被路径编码卡住
        gmsh.write(str(msh_file))
    finally:
        gmsh.finalize()

    assert msh_file.exists() and msh_file.stat().st_size > 0

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(msh_file))
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        assert node_tags.size > 0
    finally:
        gmsh.finalize()


# --------------------------------------------------------------------------
# 4. adios4dolfinx checkpoint 往返
# --------------------------------------------------------------------------
def test_adios4dolfinx_checkpoint_round_trip(tmp_in_repo: Path):
    """硬约束 5 的地基：写出的场必须能被逐点读回（DG-2，见 DECISIONS.md D-7）。"""
    import adios4dolfinx
    from dolfinx import fem
    from dolfinx import mesh as dmesh

    path = tmp_in_repo / "checkpoint.bp"
    comm = MPI.COMM_WORLD
    msh = dmesh.create_unit_square(comm, 8, 8, dmesh.CellType.triangle)
    V = fem.functionspace(msh, ("DG", 2))
    u = fem.Function(V, name="alpha_bar")
    u.interpolate(lambda x: 1.0 + x[0] ** 2 + 2.0 * x[1] ** 2)
    reference = u.x.array.copy()

    adios4dolfinx.write_mesh(path, msh)
    adios4dolfinx.write_function(path, u)

    msh_in = adios4dolfinx.read_mesh(path, comm)
    V_in = fem.functionspace(msh_in, ("DG", 2))
    u_in = fem.Function(V_in, name="alpha_bar")
    adios4dolfinx.read_function(path, u_in)

    # 读回后网格分区/自由度编号可能不同，故比全局范数与积分而非逐 dof 数组
    import ufl

    def integral(f, m):
        return m.comm.allreduce(
            fem.assemble_scalar(fem.form(f * ufl.dx)), op=MPI.SUM
        )

    assert integral(u_in, msh_in) == pytest.approx(integral(u, msh), rel=1e-10)
    assert integral(u_in * u_in, msh_in) == pytest.approx(
        integral(u * u, msh), rel=1e-10
    )
    # 自由度总数用 index_map.size_global 比：x.array 含 ghost，其长度依赖分区，
    # 连各 rank 求和也依赖分区（DG 的 ghost cell 数随分区变），不能用来比较。
    def n_dofs_global(space):
        return space.dofmap.index_map.size_global * space.dofmap.index_map_bs

    assert n_dofs_global(V_in) == n_dofs_global(V)
    assert reference.size > 0


# --------------------------------------------------------------------------
# 5. 线弹性悬臂梁冒烟（MUMPS + 挠度 < 1%）
# --------------------------------------------------------------------------
def test_cantilever_tip_deflection_within_1pct(tmp_in_repo: Path):
    """平面应变悬臂梁，端部剪力载荷，端部挠度与 Timoshenko 梁理论对比。

    参考解（单位厚度）：
        delta = F L^3/(3 E' I) + F L/(kappa G A),
        E' = E/(1-nu^2)（平面应变）, I = H^3/12, A = H, kappa = 5/6。
    有限元侧取端截面 u_y 的平均值 (1/H)∫ u_y ds，避免点求值、且天然并行安全。
    """
    import ufl
    from dolfinx import default_scalar_type, fem
    from dolfinx import mesh as dmesh
    from dolfinx.fem.petsc import LinearProblem

    from pfwire.material import lame_parameters, plane_strain_modulus

    E, nu = 69.0e9, 0.33
    L, H = 0.2, 0.01  # L/H = 20，剪切修正项 ~0.2%，足够让 EB/Timoshenko 都在 1% 内
    F = 1.0e3  # N/m，单位厚度上的端部总剪力

    comm = MPI.COMM_WORLD
    msh = dmesh.create_rectangle(
        comm,
        [np.array([0.0, -0.5 * H]), np.array([L, 0.5 * H])],
        [100, 10],
        cell_type=dmesh.CellType.triangle,
    )
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", 2, (gdim,)))

    fdim = msh.topology.dim - 1
    left = dmesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0, atol=1e-12)
    )
    right = dmesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], L, atol=1e-12)
    )
    # 全局判断：并行时某个 rank 可能不拥有左/右端面，局部 size 为 0 是正常的
    assert comm.allreduce(left.size, op=MPI.SUM) > 0
    assert comm.allreduce(right.size, op=MPI.SUM) > 0

    bc = fem.dirichletbc(
        np.zeros(gdim, dtype=default_scalar_type),
        fem.locate_dofs_topological(V, fdim, left),
        V,
    )

    facet_tags = dmesh.meshtags(
        msh, fdim, np.sort(right), np.full(right.size, 1, dtype=np.int32)
    )
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    lmbda, mu = lame_parameters(E, nu)
    lmbda_c = fem.Constant(msh, default_scalar_type(lmbda))
    mu_c = fem.Constant(msh, default_scalar_type(mu))

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu_c * eps(w) + lmbda_c * ufl.tr(eps(w)) * ufl.Identity(gdim)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    traction = fem.Constant(msh, np.array([0.0, -F / H], dtype=default_scalar_type))
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    Lf = ufl.inner(traction, v) * ds(1)

    problem = LinearProblem(
        a,
        Lf,
        bcs=[bc],
        petsc_options_prefix="cantilever_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    uh = problem.solve()
    assert problem.solver.getConvergedReason() > 0

    tip_area = comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ds(1))), op=MPI.SUM
    )
    tip_uy = comm.allreduce(
        fem.assemble_scalar(fem.form(uh[1] * ds(1))), op=MPI.SUM
    )
    delta_fe = abs(tip_uy / tip_area)
    assert tip_area == pytest.approx(H, rel=1e-12)

    E_prime = plane_strain_modulus(E, nu)
    I = H**3 / 12.0
    G = E / (2.0 * (1.0 + nu))
    kappa = 5.0 / 6.0
    delta_eb = F * L**3 / (3.0 * E_prime * I)
    delta_ref = delta_eb + F * L / (kappa * G * H)

    rel_err = abs(delta_fe - delta_ref) / delta_ref
    print(  # pytest -s 时可见，用于汇报实测偏差
        f"\n[cantilever] FE={delta_fe:.6e} m  Timoshenko={delta_ref:.6e} m  "
        f"EB={delta_eb:.6e} m  rel_err={rel_err:.4%}"
    )
    assert math.isfinite(delta_fe)
    assert rel_err < 0.01, (
        f"端部挠度误差 {rel_err:.4%} 超 1%："
        f"FE={delta_fe:.6e} m, Timoshenko={delta_ref:.6e} m, "
        f"Euler-Bernoulli={delta_eb:.6e} m"
    )
