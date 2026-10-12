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
from dolfinx.mesh import meshtags
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
from dolfinx_contact.meshing import (
    convert_mesh,
    create_halfdisk_plane_mesh,
    create_quarter_disks_mesh,
)
from dolfinx_contact.newton_solver import NewtonSolver

__all__ = [
    "HalfdiskPlaneTags",
    "HALFDISK_PLANE",
    "QuarterDisksTags",
    "QUARTER_DISKS",
    "ContactModel",
    "build_halfdisk_plane",
    "build_quarter_disks",
    "fix_block_bottom",
    "fix_surface",
]


# --------------------------------------------------------------------------
# 几何的物理组编号
#
# 两种验证几何各有一套上游写死的 tag。两个 dataclass 都暴露同一组**语义访问器**
# （body_cells / contact_facets / load_facet / sample_cell / sample_facet /
# fixed_facet），`ContactModel` 只认这组语义名，不认具体编号 —— 换几何只换 geom。
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class HalfdiskPlaneTags:
    """`create_halfdisk_plane_mesh` 写死的物理组编号（Phase 4.3 Hertz 用）。

    上游没有把它们导出成常量，只能照抄。集中放这里，**上游一改就会在
    `assert_tags_present` 里炸出来**，而不是变成一个查不出来的错误结果。
    """

    disk_contact: int = 7      # 半圆柱弧面
    block_contact: int = 6     # 块体上表面（平面，外法向 (0, 1)）
    block_bottom: int = 4      # 块体底面
    disk_top: int = 8          # 半圆柱顶面（直边）
    cell_disk: int = 1
    cell_block: int = 2

    # ---- 语义访问器（ContactModel 只用这一组）----
    @property
    def body_cells(self) -> tuple[int, int]:
        return (self.cell_disk, self.cell_block)

    @property
    def contact_facets(self) -> tuple[int, int]:
        return (self.disk_contact, self.block_contact)

    @property
    def load_facet(self) -> int:
        return self.disk_top

    @property
    def sample_cell(self) -> int:
        return self.cell_block

    @property
    def sample_facet(self) -> int:
        return self.block_contact

    @property
    def force_facet(self) -> int:
        # 读合力的平面（外法向 (0,1)）：块体上表面即接触面本身，就是平面。
        return self.block_contact

    @property
    def fixed_facet(self) -> int:
        return self.block_bottom


HALFDISK_PLANE = HalfdiskPlaneTags()


@dataclass(frozen=True)
class QuarterDisksTags:
    """`create_quarter_disks_mesh` 的物理组编号（Phase 4.4 Cattaneo–Mindlin 用）。

    上游 `demo_friction_cylinders.py` 读回后把成对的弧/平面 tag 合并
    （5→4、8→7、11→10、14→13）。`build_quarter_disks` 照做，故这里记的是
    **合并后**的编号。几何：上半圆柱（body 0，cell 1）压在下半圆柱
    （body 1，cell 2）上，两者在 (0, -R) 处相切。
    """

    disk_upper_contact: int = 4    # 上圆柱下弧（接触）
    disk_lower_contact: int = 10   # 下圆柱上弧（接触）
    disk_upper_top: int = 7        # 上圆柱顶面（平面，外法向 (0, 1)）：加载面
    disk_lower_bottom: int = 13    # 下圆柱底面（平面）：全固定
    cell_upper: int = 1
    cell_lower: int = 2

    @property
    def body_cells(self) -> tuple[int, int]:
        return (self.cell_upper, self.cell_lower)

    @property
    def contact_facets(self) -> tuple[int, int]:
        return (self.disk_upper_contact, self.disk_lower_contact)

    @property
    def load_facet(self) -> int:
        return self.disk_upper_top

    @property
    def sample_cell(self) -> int:
        # 取**下**圆柱上弧的面力：接触区 (x≈0, y≈-R) 处外法向 ≈ (0, 1)，
        # 故 p = -sigma_yy、q = -sigma_xy 成立（与半平面近似同口径）。
        return self.cell_lower

    @property
    def sample_facet(self) -> int:
        return self.disk_lower_contact

    @property
    def force_facet(self) -> int:
        # 读合力用**平面**：取上圆柱顶面（外法向 (0,1)），力控位移就加在这。
        # 接触弧是曲面、法向只在 x≈0 处才 ≈(0,1)，不能直接 -sigma_yy 读力；
        # 而顶面是上圆柱唯一的其他外载面，按平衡 ∫顶面 = 透过接触的合力。
        return self.disk_upper_top

    @property
    def fixed_facet(self) -> int:
        return self.disk_lower_bottom


QUARTER_DISKS = QuarterDisksTags()


def build_halfdisk_plane(
    workdir: pathlib.Path, *, R: float, H: float, L: float, gap: float,
    res: float, order: int, quads: bool, comm=MPI.COMM_WORLD,
):
    """生成「半圆柱压块体」网格并读回 dolfinx，返回 (mesh, cell_tags, facet_tags)。"""
    workdir.mkdir(parents=True, exist_ok=True)
    stem = str(workdir / "halfdisk_plane")
    # 上游两个函数内部自己只让 rank 0 干活、并在结尾对 COMM_WORLD 做 Barrier，
    # 所以必须**所有 rank 都调用**；包在 `if rank == 0` 里会让 rank 0 多等一个
    # 没人配对的 Barrier，其余 rank 卡在下面的 XDMF 读取 -> 并行死锁。
    create_halfdisk_plane_mesh(filename=f"{stem}.msh", res=res, order=order,
                               quads=quads, r=R, height=H, length=L, gap=gap)
    convert_mesh(stem, f"{stem}.xdmf", gdim=2)
    with XDMFFile(comm, f"{stem}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()
        cell_tags = xdmf.read_meshtags(mesh, name="cell_marker")
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        facet_tags = xdmf.read_meshtags(mesh, name="facet_marker")
    assert_tags_present(facet_tags, cell_tags, HALFDISK_PLANE, comm=comm)
    return mesh, cell_tags, facet_tags


def build_quarter_disks(
    workdir: pathlib.Path, *, R: float, gap: float,
    res: float, order: int, quads: bool, comm=MPI.COMM_WORLD,
):
    """生成「圆柱–圆柱」网格并读回 dolfinx，返回 (mesh, cell_tags, facet_tags)。

    这是 Cattaneo–Mindlin 的原生几何（两个相切圆柱），也是上游
    `demo_friction_cylinders.py` 实测切向加载能收敛的那套 —— 半圆柱压块体
    在切向步会进黏滑极限环（见本次 Phase 4.4 的诊断）。

    读回后要做两件 dolfinx 0.11 下的收尾（上游 demo 的做法照搬）：
    1. **合并成对 tag**：上游给每个弧/平面分了两段（对称面两侧），
       5→4、8→7、11→10、14→13 合并，使接触面是单一 tag；
       0.11 下 `meshtags.values` 是只读的，不能原地改，要重建一个 meshtags。
    2. **建顶点→单元连通**：`locate_dofs_topological(..., 0, ...)` 需要它，
       0.11 不再自动建（旧版本会）。
    """
    workdir.mkdir(parents=True, exist_ok=True)
    stem = str(workdir / "quarter_disks")
    # 同上：上游函数内部自带 rank-0 守卫与 Barrier，须全体 rank 调用。
    create_quarter_disks_mesh(f"{stem}.msh", res, order, quads, R, gap)
    convert_mesh(stem, f"{stem}.xdmf", gdim=2)
    with XDMFFile(comm, f"{stem}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()
        cell_tags = xdmf.read_meshtags(mesh, name="cell_marker")
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facet_tags = xdmf.read_meshtags(mesh, name="facet_marker")

    # (1) 合并成对 tag（meshtags.values 只读 -> 重建）
    vals = facet_tags.values.copy()
    for src, dst in ((5, 4), (8, 7), (11, 10), (14, 13)):
        vals[vals == src] = dst
    facet_tags = meshtags(mesh, tdim - 1, facet_tags.indices.copy(), vals)
    # (2) 顶点 -> 单元连通（约束节点定位要用）
    mesh.topology.create_connectivity(0, tdim)

    assert_tags_present(facet_tags, cell_tags, QUARTER_DISKS, comm=comm)
    return mesh, cell_tags, facet_tags


def assert_tags_present(facet_tags, cell_tags, tags, *, comm) -> None:
    """上游若改了物理组编号，在这里立刻失败，而不是给出一个错误的结果。

    只核对 `ContactModel` 真正用到的那组语义 tag（两个接触面、加载面、
    固定面、两种单元），所以对两种几何通用。
    """
    checks = [
        ("contact_facet[0]", tags.contact_facets[0], facet_tags),
        ("contact_facet[1]", tags.contact_facets[1], facet_tags),
        ("load_facet", tags.load_facet, facet_tags),
        ("fixed_facet", tags.fixed_facet, facet_tags),
        ("body_cell[0]", tags.body_cells[0], cell_tags),
        ("body_cell[1]", tags.body_cells[1], cell_tags),
    ]
    for name, tag, store in checks:
        n = comm.allreduce(int(store.find(tag).size), op=MPI.SUM)
        if n == 0:
            raise RuntimeError(
                "网格里找不到物理组 %s=%d —— 上游网格生成器的编号可能已变动，"
                "请核对后更新对应的 Tags dataclass（D-48）" % (name, tag))


class ContactModel:
    """一个两体摩擦接触问题：装配、加载、求解、取面力。"""

    def __init__(self, mesh, cell_tags, facet_tags, *, E: float, nu: float,
                 order: int = 1, quadrature_degree: int = 5,
                 tags=HALFDISK_PLANE, sample_center=None):
        # sample_center：采样接触弧的圆心 (cx, cy)。给了就用弧面真实法向 n 取
        # p = -n·σn、q = -τ·σn；不给则按平面近似 n = (0, 1)（Hertz 算例的块体上表面）。
        # D-51 的诊断里发现：曲面上直接用 σ_yy/σ_xy 会把 σ_xx·(x/R) 混进 q，
        # 接触区边缘 p 很小时能把 |q|/(μp) 抬到 3~4 倍，使库仑不等式测不准。
        self.sample_center = sample_center
        self.mesh, self.cell_tags, self.facet_tags = mesh, cell_tags, facet_tags
        self.tags, self.order, self.E, self.nu = tags, order, E, nu
        gdim = mesh.geometry.dim
        self.V = functionspace(mesh, ("CG", order, (gdim,)))

        mu_f, lmbda_f = lame_parameters(True)      # True = 平面应变
        V0 = functionspace(mesh, ("DG", 0))
        self.mu_c, self.lmbda_c = Function(V0), Function(V0)
        for cell_tag in tags.body_cells:
            cells = cell_tags.find(cell_tag)
            self.mu_c.interpolate(lambda x: np.full((1, x.shape[1]), mu_f(E, nu)), cells)
            self.lmbda_c.interpolate(lambda x: np.full((1, x.shape[1]), lmbda_f(E, nu)), cells)
        self.sigma = sigma_func(self.mu_c, self.lmbda_c)

        self.u, self.du = Function(self.V), Function(self.V)
        self.dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
        self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
        self.problem = ContactProblem(
            [facet_tags],
            adjacencylist(np.array(tags.contact_facets, dtype=np.int32),
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
            F -= ufl.inner(traction, v) * self.ds(self.tags.load_facet)
        F = ufl.replace(F, {self.u: self.u + self.du})
        self.F_c = form(F)
        self.J_c = form(ufl.derivative(F, self.du, w))
        self.bcs = list(bcs)

        self._theta = theta
        self._generate(friction, gamma_factor, theta)

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

    def _generate(self, friction, gamma_factor: float, theta: float) -> None:
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

    def set_friction_law(self, friction: float | None, *, gamma_factor: float,
                         petsc_options: dict | None = None) -> None:
        """中途切换摩擦律（无摩擦 <-> Coulomb），并重建矩阵。

        为什么要分两段：带摩擦直接做第一次法向加载，Newton 在 200 次迭代内
        不收敛（实测）。上游 friction 算例也是先无摩擦压下、再切 Coulomb 推，
        且切换时把 Nitsche 罚参数从 E*100 降到 **E*10** —— 摩擦项本身已经
        提供了切向刚度，罚参数再取大就病态。

        ``petsc_options``：切换摩擦后可换一套 KSP。Coulomb 的切向 Jacobian
        **非对称**，上游算例因此从压下阶段的 CG 换成 **GMRES + GAMG**；
        不传则沿用 setup 时的那套。
        """
        self._generate(friction, gamma_factor, self._theta)
        A = self.problem.create_matrix(self.J_c)
        self._set_nullspace(A)
        self._newton.set_petsc_matrix(A)
        if petsc_options is not None:
            self._petsc_options = petsc_options
        self._newton.update_krylov_solver(self._petsc_options)

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
        """某**平面**上的合力分量，用**精确面积分** ∫ -sigma[1, component] ds。

        默认取 `tags.force_facet`：对半圆柱+块是接触面本身（平面），
        对圆柱–圆柱是上圆柱顶面（曲面接触弧不能这样读力，见 force_facet
        的注释）。两者外法向都是 (0, 1)，故 component=1 得法向力、=0 得切向力。

        比对节点值做梯形积分可靠得多：后者会把接触区外被 CG1 抹出来的
        数值残尾也算进去（D-49，Phase 4.3 实测差 3.1% vs 0.0014%）。
        """
        tag = self.tags.force_facet if tag is None else tag
        local = assemble_scalar(form(-self.sigma(self.u)[1, component] * self.ds(tag)))
        return float(self.mesh.comm.allreduce(local, op=MPI.SUM))

    def surface_tractions(self):
        """接触采样面的 (x, p, q)：法向压力 p = -sigma_yy、切向面力 q = -sigma_xy。

        采样面取 `tags.sample_facet`（半圆柱算例是块体上表面，圆柱–圆柱算例
        是下圆柱上弧）。接触区位于 x≈0、y≈-R 附近，那里外法向 ≈ (0, 1)，
        故法向/切向分量就是 sigma 的两个分量 —— **不需要 FacetNormal，
        也不需要 submesh**，正好绕开 D-48 的两个坑。对圆柱弧这是半平面近似，
        在接触区（|x| << R）内误差可忽略，恰是 Cattaneo–Mindlin 本身的前提。
        """
        Vs = functionspace(self.mesh, ("CG", 1))
        body = self.cell_tags.find(self.tags.sample_cell)
        sig = self.sigma(self.u)
        if self.sample_center is None:
            p_expr, q_expr = sig[1, 1], sig[1, 0]          # sigma_yy, sigma_xy
        else:
            xc = ufl.SpatialCoordinate(self.mesh)
            cx, cy = self.sample_center
            n = ufl.as_vector([xc[0] - cx, xc[1] - cy])
            n = n / ufl.sqrt(ufl.dot(n, n))                # 圆弧外法向
            tau = ufl.as_vector([n[1], -n[0]])             # x≈0 处 = (1, 0)
            t = ufl.dot(sig, n)
            p_expr, q_expr = ufl.dot(n, t), ufl.dot(tau, t)
        out = []
        for expr in (p_expr, q_expr):
            f = Function(Vs)
            f.interpolate(Expression(expr, Vs.element.interpolation_points), cells0=body)
            out.append(f)
        dofs = locate_dofs_topological(Vs, self.mesh.topology.dim - 1,
                                       self.facet_tags.find(self.tags.sample_facet))
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
    """固定面全约束 —— 两个验证算例共用（半圆柱算例的块体底面、
    圆柱–圆柱算例的下圆柱底面）。"""
    dofs = locate_dofs_topological(model.V, model.mesh.topology.dim - 1,
                                   model.facet_tags.find(model.tags.fixed_facet))
    return dirichletbc(Constant(model.mesh, default_scalar_type((0.0, 0.0))), dofs, model.V)


#: 兼容别名：语义上就是「固定住 fixed_facet」。
fix_surface = fix_block_bottom
