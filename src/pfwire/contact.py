"""FE 摩擦接触的薄封装（PLAN §3；D-46 的方向性改写）。

包的是 `dolfinx_contact`（Wells-Group/asimov-contact，Nitsche 法）。
**只封装我们真正要用的那一小块**：建两体网格、装配、Newton 求解、
以及从解里取接触面上的法向/切向面力。

刻意**不用**上游的 `ContactWriter`：它在 dolfinx 0.11 / UFL 2026.1 下已失效
（`create_submesh` 改返回 `EntityMap`，且仍在用已删除的 legacy `ufl.FiniteElement`），
见 DECISIONS.md D-48。本模块取面力走的是另一条路，见 `surface_tractions`。

环境要求：`dolfinx_contact` 必须是 **stable-ABI 构建**（`cpp.abi3.so`），
否则它与 dolfinx 的 nanobind 类型不互通，`ContactProblem` 构造会报
"incompatible function arguments"。`scripts/setup_env.sh` 已自动打补丁（D-48）。
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import ufl
from dolfinx import default_scalar_type
from dolfinx.fem import (
    Constant,
    Expression,
    Function,
    assemble_scalar,
    dirichletbc,
    extract_function_spaces,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_vector,
    set_bc,
)
from dolfinx.graph import adjacencylist
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py.PETSc import InsertMode, ScatterMode  # type: ignore

from dolfinx_contact.cpp import ContactMode
from dolfinx_contact.general_contact.contact_problem import ContactProblem, FrictionLaw
from dolfinx_contact.helpers import (
    epsilon,
    lame_parameters,
    rigid_motions_nullspace_subdomains,
    sigma_func,
)
from dolfinx_contact.meshing import convert_mesh, create_halfdisk_plane_mesh
from dolfinx_contact.newton_solver import NewtonSolver

__all__ = [
    "HalfdiskPlaneTags",
    "HALFDISK_PLANE",
    "ContactModel",
    "build_halfdisk_plane",
]


@dataclass(frozen=True)
class HalfdiskPlaneTags:
    """`create_halfdisk_plane_mesh` 写死的物理组编号。

    上游没有把它们导出成常量，只能照抄。集中放这里，**上游一改就会在
    `assert_tags_present` 里炸出来**，而不是变成一个查不出来的错误结果。
    """

    disk_contact: int = 7      # 半圆柱弧面
    block_contact: int = 6     # 块体上表面（平面，外法向 (0, 1)）
    block_bottom: int = 4      # 块体底面
    disk_top: int = 8          # 半圆柱顶面（直边）
    cell_disk: int = 1
    cell_block: int = 2


HALFDISK_PLANE = HalfdiskPlaneTags()


def build_halfdisk_plane(
    workdir: pathlib.Path, *, R: float, H: float, L: float, gap: float,
    res: float, order: int, quads: bool, comm=MPI.COMM_WORLD,
):
    """生成「半圆柱压块体」网格并读回 dolfinx，返回 (mesh, cell_tags, facet_tags)。"""
    workdir.mkdir(parents=True, exist_ok=True)
    stem = str(workdir / "halfdisk_plane")
    if comm.rank == 0:
        create_halfdisk_plane_mesh(filename=f"{stem}.msh", res=res, order=order,
                                   quads=quads, r=R, height=H, length=L, gap=gap)
        convert_mesh(stem, f"{stem}.xdmf", gdim=2)
    comm.barrier()
    with XDMFFile(comm, f"{stem}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()
        cell_tags = xdmf.read_meshtags(mesh, name="cell_marker")
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        facet_tags = xdmf.read_meshtags(mesh, name="facet_marker")
    assert_tags_present(facet_tags, cell_tags, comm=comm)
    return mesh, cell_tags, facet_tags


def assert_tags_present(facet_tags, cell_tags, *, comm,
                        tags: HalfdiskPlaneTags = HALFDISK_PLANE) -> None:
    """上游若改了物理组编号，在这里立刻失败，而不是给出一个错误的结果。"""
    for name, tag, store in (
        ("disk_contact", tags.disk_contact, facet_tags),
        ("block_contact", tags.block_contact, facet_tags),
        ("block_bottom", tags.block_bottom, facet_tags),
        ("disk_top", tags.disk_top, facet_tags),
        ("cell_disk", tags.cell_disk, cell_tags),
        ("cell_block", tags.cell_block, cell_tags),
    ):
        n = comm.allreduce(int(store.find(tag).size), op=MPI.SUM)
        if n == 0:
            raise RuntimeError(
                "网格里找不到物理组 %s=%d —— 上游 create_halfdisk_plane_mesh 的编号"
                "可能已变动，请核对后更新 HalfdiskPlaneTags（D-48）" % (name, tag))


class ContactModel:
    """一个两体摩擦接触问题：装配、加载、求解、取面力。"""

    def __init__(self, mesh, cell_tags, facet_tags, *, E: float, nu: float,
                 order: int = 1, quadrature_degree: int = 5,
                 tags: HalfdiskPlaneTags = HALFDISK_PLANE):
        self.mesh, self.cell_tags, self.facet_tags = mesh, cell_tags, facet_tags
        self.tags, self.order, self.E, self.nu = tags, order, E, nu
        gdim = mesh.geometry.dim
        self.V = functionspace(mesh, ("CG", order, (gdim,)))

        mu_f, lmbda_f = lame_parameters(True)      # True = 平面应变
        V0 = functionspace(mesh, ("DG", 0))
        self.mu_c, self.lmbda_c = Function(V0), Function(V0)
        for cells in (cell_tags.find(tags.cell_disk), cell_tags.find(tags.cell_block)):
            self.mu_c.interpolate(lambda x: np.full((1, x.shape[1]), mu_f(E, nu)), cells)
            self.lmbda_c.interpolate(lambda x: np.full((1, x.shape[1]), lmbda_f(E, nu)), cells)
        self.sigma = sigma_func(self.mu_c, self.lmbda_c)

        self.u, self.du = Function(self.V), Function(self.V)
        self.dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
        self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
        self.problem = ContactProblem(
            [facet_tags],
            adjacencylist(np.array([tags.disk_contact, tags.block_contact], dtype=np.int32),
                          np.array([0, 2], dtype=np.int32)),
            [(0, 1), (1, 0)], mesh, quadrature_degree,
            [ContactMode.ClosestPoint, ContactMode.ClosestPoint],
        )
        self._newton = None

    # ---- 装配 -----------------------------------------------------------
    def setup(self, *, bcs: Sequence, traction: Constant | None = None,
              friction: float | None = None, gamma_factor: float = 100.0,
              theta: float = 1.0, newton_options: dict, petsc_options: dict) -> None:
        """建残差/雅可比、接触数据与 Newton 求解器。

        `friction=None` -> 无摩擦；给了系数 -> Coulomb 摩擦。
        """
        v, w = ufl.TestFunction(self.V), ufl.TrialFunction(self.V)
        F = ufl.inner(self.sigma(self.u), epsilon(v)) * self.dx
        if traction is not None:
            F -= ufl.inner(traction, v) * self.ds(self.tags.disk_top)
        F = ufl.replace(F, {self.u: self.u + self.du})
        self.F_c = form(F)
        self.J_c = form(ufl.derivative(F, self.du, w))
        self.bcs = list(bcs)

        coeffs = {"u": self.u, "du": self.du, "mu": self.mu_c, "lambda": self.lmbda_c}
        if friction is None:
            law = FrictionLaw.Frictionless
        else:
            law = FrictionLaw.Coulomb
            fric = Function(functionspace(self.mesh, ("DG", 0)))
            fric.interpolate(lambda x: np.full((1, x.shape[1]), float(friction)))
            coeffs["fric"] = fric
            self._fric = fric
        self.problem.generate_contact_data(
            law, self.V, coeffs, self.E * gamma_factor * self.order**2, theta)

        A = self.problem.create_matrix(self.J_c)
        b = create_vector(extract_function_spaces(self.F_c))
        self._newton = NewtonSolver(self.mesh.comm, A, b, self.problem.coeffs)
        self._newton.set_residual(self._residual)
        self._newton.set_jacobian(self._jacobian)
        self._newton.set_coefficients(self._coefficients)
        self._set_nullspace(A)
        self._newton.set_newton_options(newton_options)
        self._newton.set_krylov_options(petsc_options)
        self._petsc_options = petsc_options

    def _set_nullspace(self, A) -> None:
        A.setNearNullSpace(rigid_motions_nullspace_subdomains(
            self.V, self.cell_tags, np.unique(self.cell_tags.values)))

    def _coefficients(self, x, coeffs) -> None:
        self.du.x.scatter_forward()
        self.problem.update_contact_data(self.du)

    def _residual(self, x, b, coeffs) -> None:
        b.zeroEntries()
        b.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        self.problem.assemble_vector(b, self.V)
        assemble_vector(b, self.F_c)
        apply_lifting(b, [self.J_c], bcs=[self.bcs], x0=[x], alpha=-1.0)
        b.ghostUpdate(addv=InsertMode.ADD, mode=ScatterMode.REVERSE)
        set_bc(b, self.bcs, x, -1.0)

    def _jacobian(self, x, A, coeffs) -> None:
        A.zeroEntries()
        self.problem.assemble_matrix(A, self.V)
        assemble_matrix(A, self.J_c, bcs=self.bcs)
        A.assemble()

    # ---- 求解 -----------------------------------------------------------
    def step(self, label: str = "") -> int:
        """解一个增量步；**不收敛就抛异常**（硬约束 6：不得静默继续）。"""
        if self.bcs:
            set_bc(self.du.x.petsc_vec, self.bcs)
        n_it, converged = self._newton.solve(self.du, write_solution=True)
        if not converged:
            raise RuntimeError("接触 Newton 未收敛%s" % (f"（{label}）" if label else ""))
        self.du.x.scatter_forward()
        self.u.x.array[:] += self.du.x.array[:]
        return int(n_it)

    def prepare_next(self) -> None:
        """为下一增量步更新接触搜索与矩阵（上游算例的标准做法）。"""
        self.problem.update_contact_detection(self.u)
        A = self.problem.create_matrix(self.J_c)
        self._set_nullspace(A)
        self._newton.set_petsc_matrix(A)
        # 取上一步的一小部分作初值：全无 Dirichlet 约束时可避免奇异矩阵
        self.du.x.array[:] = 0.1 * self.du.x.array[:]
        self.problem.update_contact_data(self.du)

    # ---- 取面力 ---------------------------------------------------------
    def resultant(self, component: int, tag: int | None = None) -> float:
        """接触面上的合力分量，用**精确面积分**。

        比对节点值做梯形积分可靠得多：后者会把接触区外被 CG1 抹出来的
        数值残尾也算进去（D-49，Phase 4.3 实测差 3.1% vs 0.0014%）。
        """
        tag = self.tags.block_contact if tag is None else tag
        local = assemble_scalar(form(-self.sigma(self.u)[1, component] * self.ds(tag)))
        return float(self.mesh.comm.allreduce(local, op=MPI.SUM))

    def surface_tractions(self):
        """块体上表面的 (x, p, q)：法向压力 p = -sigma_yy、切向面力 q = -sigma_xy。

        该面是平面、外法向 (0, 1)，所以法向/切向分量就是 sigma 的两个分量，
        **不需要 FacetNormal，也不需要 submesh** —— 正好绕开 D-48 的两个坑。
        """
        Vs = functionspace(self.mesh, ("CG", 1))
        block = self.cell_tags.find(self.tags.cell_block)
        out = []
        for comp in (1, 0):                       # sigma_yy, sigma_xy
            f = Function(Vs)
            f.interpolate(Expression(self.sigma(self.u)[1, comp],
                                     Vs.element.interpolation_points), cells0=block)
            out.append(f)
        dofs = locate_dofs_topological(Vs, self.mesh.topology.dim - 1,
                                       self.facet_tags.find(self.tags.block_contact))
        n_owned = Vs.dofmap.index_map.size_local
        keep = dofs[dofs < n_owned]               # 并行下只留本进程拥有的
        xs = Vs.tabulate_dof_coordinates()[keep, 0]
        comm = self.mesh.comm
        xs = np.concatenate(comm.allgather(xs))
        p = np.concatenate(comm.allgather(-out[0].x.array[keep]))
        q = np.concatenate(comm.allgather(-out[1].x.array[keep]))
        o = np.argsort(xs)
        return xs[o], p[o], q[o]


def fix_block_bottom(model: ContactModel):
    """块体底面全固定 —— 两个验证算例共用的边界条件。"""
    dofs = locate_dofs_topological(model.V, model.mesh.topology.dim - 1,
                                   model.facet_tags.find(model.tags.block_bottom))
    return dirichletbc(Constant(model.mesh, default_scalar_type((0.0, 0.0))), dofs, model.V)
