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

from pfwire import slip as _slip
from pfwire.plasticity import PlasticLaw

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
    "PunchBlockTags",
    "PUNCH_BLOCK",
    "build_punch_block",
    "ClampWireTags",
    "CLAMP_WIRE",
    "build_clamp_wire",
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
    def force_body(self) -> int:
        return 1           # 块体（body_cells 的第 2 个）

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
    def force_body(self) -> int:
        return 0           # 上圆柱

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
        ("force_facet", tags.force_facet, facet_tags),
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


@dataclass(frozen=True)
class PunchBlockTags:
    """平冲头压块体（Phase 6.1 的 Prandtl 基准）的物理组编号。

    几何：冲头（body 0）在**上**，块体（body 1）在**下**，两者接触面都是平面，
    高度方向外力加在冲头顶面、块体底面固定 —— 与 HalfdiskPlaneTags 的朝向一致，
    所以语义访问器可以直接照搬。**平-平接触**是本几何与半圆柱算例的唯一区别；
    投影在冲头角点附近是奇异的（平冲头的经典特征），对拍时把角点邻域排除。
    """

    punch_contact: int = 3     # 冲头底面（接触，外法向 (0, -1)）
    block_contact: int = 4     # 块体顶面（接触，外法向 (0, +1)）
    punch_top: int = 5         # 冲头顶面（加载面，外法向 (0, +1)）
    block_bottom: int = 6      # 块体底面（全固定）
    cell_punch: int = 1
    cell_block: int = 2

    @property
    def body_cells(self) -> tuple[int, int]:
        return (self.cell_punch, self.cell_block)

    @property
    def contact_facets(self) -> tuple[int, int]:
        return (self.punch_contact, self.block_contact)

    @property
    def load_facet(self) -> int:
        return self.punch_top

    @property
    def sample_cell(self) -> int:
        return self.cell_block

    @property
    def sample_facet(self) -> int:
        return self.block_contact

    @property
    def force_facet(self) -> int:
        # 冲头顶面是平面、外法向 (0,1)，按平衡即透过接触的合力
        return self.punch_top

    @property
    def force_body(self) -> int:
        return 0

    @property
    def fixed_facet(self) -> int:
        return self.block_bottom


PUNCH_BLOCK = PunchBlockTags()


def build_punch_block(
    workdir: pathlib.Path, *, half_width: float, block_half_width: float,
    block_depth: float, punch_height: float, h_fine: float, h_coarse: float, gap: float = 0.0,
    order: int, quads: bool, comm=MPI.COMM_WORLD,
):
    """生成「平冲头 + 块体」网格并读回 dolfinx，返回 (mesh, cell_tags, facet_tags)。

    冲头底面 y = 0、块体顶面 y = -gap（gap 只允许 0 或极小值，见下），接触面在 |x| <= half_width 内加密到
    h_fine，远处粗化到 h_coarse。物理组编号见 PunchBlockTags；面组显式指定，
    不像上游那样依赖 gmsh 的自动编号。
    """
    import gmsh

    workdir.mkdir(parents=True, exist_ok=True)
    stem = str(workdir / "punch_block")
    if comm.rank == 0:
        gmsh.initialize()
        gmsh.clear()                 # 同一进程里多次调用时，清掉上一次的模型
        try:
            _punch_block_gmsh(gmsh, str(stem), half_width=half_width, block_half_width=block_half_width,
                              block_depth=block_depth, punch_height=punch_height, gap=gap,
                              h_fine=h_fine, h_coarse=h_coarse, order=order, quads=quads)
        finally:
            gmsh.finalize()          # 中途抛异常也必须收尾，否则下一次 initialize 会失败
    comm.barrier()
    convert_mesh(stem, f"{stem}.xdmf", gdim=2)
    with XDMFFile(comm, f"{stem}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()
        cell_tags = xdmf.read_meshtags(mesh, name="cell_marker")
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        facet_tags = xdmf.read_meshtags(mesh, name="facet_marker")
    mesh.topology.create_connectivity(0, mesh.topology.dim)
    assert_tags_present(facet_tags, cell_tags, PUNCH_BLOCK, comm=comm)
    return mesh, cell_tags, facet_tags


def _punch_block_gmsh(gmsh, stem, *, half_width, block_half_width, block_depth,
                      punch_height, gap, h_fine, h_coarse, order, quads):
    """gmsh 建体：平冲头 + 块体。物理组编号见 PunchBlockTags（面组显式指定）。"""
    if True:
        w, W, D, hp = half_width, block_half_width, block_depth, punch_height
        punch = gmsh.model.occ.addRectangle(-w, 0.0, 0.0, 2 * w, hp)
        block = gmsh.model.occ.addRectangle(-W, -gap - D, 0.0, 2 * W, D)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [punch], tag=PUNCH_BLOCK.cell_punch)
        gmsh.model.addPhysicalGroup(2, [block], tag=PUNCH_BLOCK.cell_block)
        # 面组按「所属面的边界曲线里哪条是水平线、且 y 等于目标值」挑。
        # gmsh 的包围盒带 ~1e-7 绝对容差（相对模型尺寸 1e-4），所以容差取特征长度的 1e-3，
        # 既远大于该误差（1e-7），又远小于几何间距（half_width）。
        # gmsh 的包围盒带 ~1e-4 相对尺寸的绝对误差（模型 1e-3 时约 1e-7），
        # 所以挑水平边界曲线的容差取特征长度的 1e-3；gap 必须明显大于它才能与 0 区分。
        tol = 1e-3 * min(w, D, hp)
        if gap != 0.0 and gap < 3.0 * tol:
            # gap = 0 允许（两体初始相切，如 4.4 的 quarter_disks）：此时冲头底面与块体顶面
            # 包围盒重合，但两者是**分别**在各自曲线上找，不会混淆。
            raise ValueError("gap 必须为 0 或 >= %.3g（gmsh 包围盒容差的 3 倍），得到 %r"
                             % (3.0 * tol, gap))

        def _flat_face(ents, y_target):
            out = []
            for d, e in ents:
                ymin, ymax = gmsh.model.getBoundingBox(d, e)[1], gmsh.model.getBoundingBox(d, e)[4]
                if abs(ymin - y_target) < tol and abs(ymax - y_target) < tol:
                    out.append(e)
            if len(out) != 1:
                raise RuntimeError("在 y = %g 处找到 %d 条水平边界曲线，应当恰好 1 条" % (y_target, len(out)))
            return out[0]

        gmsh.model.addPhysicalGroup(1, [_flat_face(gmsh.model.getBoundary([(2, punch)]), 0.0)],
                                    tag=PUNCH_BLOCK.punch_contact)
        gmsh.model.addPhysicalGroup(1, [_flat_face(gmsh.model.getBoundary([(2, block)]), -gap)],
                                    tag=PUNCH_BLOCK.block_contact)
        gmsh.model.addPhysicalGroup(1, [_flat_face(gmsh.model.getBoundary([(2, punch)]), hp)],
                                    tag=PUNCH_BLOCK.punch_top)
        gmsh.model.addPhysicalGroup(1, [_flat_face(gmsh.model.getBoundary([(2, block)]), -gap - D)],
                                    tag=PUNCH_BLOCK.block_bottom)
        gmsh.model.mesh.field.add("Box", 1)
        gmsh.model.mesh.field.setNumber(1, "VIn", h_fine)
        gmsh.model.mesh.field.setNumber(1, "VOut", h_coarse)
        gmsh.model.mesh.field.setNumber(1, "XMin", -1.5 * w)
        gmsh.model.mesh.field.setNumber(1, "XMax", 1.5 * w)
        gmsh.model.mesh.field.setNumber(1, "YMin", -gap - 2.0 * w)
        gmsh.model.mesh.field.setNumber(1, "YMax", hp)
        gmsh.model.mesh.field.setNumber(1, "Thickness", 3.0 * w)
        gmsh.model.mesh.field.setAsBackgroundMesh(1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        if quads:
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
            gmsh.option.setNumber("Mesh.RecombineAll", 2)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)
        gmsh.write(f"{stem}.msh")


@dataclass(frozen=True)
class ClampWireTags:
    """线夹压单丝（Phase 6.1 本体）的物理组编号。

    几何（PLAN §1.1）：2D 平面应变，**导线轴向–径向平面**。单丝截面在此平面里是矩形
    （厚度 = 单丝直径 D = 3.75 mm），线夹是刚体、其轴向曲率半径 R_clamp = 330 mm（D-50）。
    故把线夹建成「底边是半径 R_clamp 的**向下凸**圆弧的块体」，压在下方的平板（单丝）上
    —— 弧-平接触，与 4.3 Hertz 同型（4.3/4.4 证明能收敛的正是这种形态）。
    线夹在上（body 0）、单丝在下（body 1），朝向与 HalfdiskPlaneTags 一致。
    """

    clamp_contact: int = 3     # 线夹底面弧（接触）
    wire_contact: int = 4      # 单丝顶面（接触，平面，外法向 (0, +1)）
    clamp_top: int = 5         # 线夹顶面（加载面，外法向 (0, +1)）
    wire_bottom: int = 6       # 单丝底面（全固定）
    cell_clamp: int = 1
    cell_wire: int = 2

    @property
    def body_cells(self) -> tuple[int, int]:
        return (self.cell_clamp, self.cell_wire)

    @property
    def contact_facets(self) -> tuple[int, int]:
        return (self.clamp_contact, self.wire_contact)

    @property
    def load_facet(self) -> int:
        return self.clamp_top

    @property
    def sample_cell(self) -> int:
        return self.cell_wire

    @property
    def sample_facet(self) -> int:
        return self.wire_contact

    @property
    def force_facet(self) -> int:
        # 单丝顶面是平面、外法向 (0,1)：按平衡即透过接触的合力（4.3 的做法，精度 0.0014%）
        return self.wire_contact

    @property
    def force_body(self) -> int:
        return 1

    @property
    def fixed_facet(self) -> int:
        return self.wire_bottom


CLAMP_WIRE = ClampWireTags()


def build_clamp_wire(
    workdir: pathlib.Path, *, R_clamp: float, Lc: float, Hc: float, Lw: float, D: float,
    h_fine: float, h_coarse: float, fine_halfwidth: float, order: int, quads: bool,
    gap: float = 0.0, comm=MPI.COMM_WORLD,
):
    """线夹（大半径圆弧底）压单丝（平板）网格，返回 (mesh, cell_tags, facet_tags)。

    R_clamp = 330 mm 时弧的矢高在 ±Lc/2 上只有 Lc²/(8R_clamp) ≈ 54 µm，不能沿用上游
    `create_halfdisk_plane_mesh`（它按半径 R 自动加密，R = 330 mm 会把 165 mm 半径全加密）。
    这里自己建：|x| <= fine_halfwidth 内加密到 h_fine，远处 h_coarse。
    """
    import gmsh

    workdir.mkdir(parents=True, exist_ok=True)
    stem = str(workdir / "clamp_wire")
    if comm.rank == 0:
        gmsh.initialize()
        gmsh.clear()
        try:
            _clamp_wire_gmsh(gmsh, stem, R_clamp=R_clamp, Lc=Lc, Hc=Hc, Lw=Lw, D=D,
                             h_fine=h_fine, h_coarse=h_coarse, fine_halfwidth=fine_halfwidth,
                             order=order, quads=quads, gap=gap)
        finally:
            gmsh.finalize()
    comm.barrier()
    convert_mesh(stem, f"{stem}.xdmf", gdim=2)
    with XDMFFile(comm, f"{stem}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()
        cell_tags = xdmf.read_meshtags(mesh, name="cell_marker")
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        facet_tags = xdmf.read_meshtags(mesh, name="facet_marker")
    mesh.topology.create_connectivity(0, mesh.topology.dim)
    assert_tags_present(facet_tags, cell_tags, CLAMP_WIRE, comm=comm)
    return mesh, cell_tags, facet_tags


def _clamp_wire_gmsh(gmsh, stem, *, R_clamp, Lc, Hc, Lw, D, h_fine, h_coarse,
                     fine_halfwidth, order, quads, gap):
    """gmsh 建体：线夹 = 矩形但底边换成半径 R_clamp 的**向下凸**圆弧（圆心在 (0, R)）；
    单丝 = 矩形（顶面平，位于 y = -gap）。"""
    hc = Lc / 2.0
    y_arc = R_clamp - np.sqrt(R_clamp**2 - hc * hc)          # 端点相对弧顶的抬高
    p_bl = gmsh.model.occ.addPoint(-hc, y_arc, 0.0)
    p_br = gmsh.model.occ.addPoint(hc, y_arc, 0.0)
    center = gmsh.model.occ.addPoint(0.0, R_clamp, 0.0)
    p_tl = gmsh.model.occ.addPoint(-hc, Hc, 0.0)
    p_tr = gmsh.model.occ.addPoint(hc, Hc, 0.0)
    arc = gmsh.model.occ.addCircleArc(p_bl, center, p_br)
    l_right = gmsh.model.occ.addLine(p_br, p_tr)
    l_top = gmsh.model.occ.addLine(p_tr, p_tl)
    l_left = gmsh.model.occ.addLine(p_tl, p_bl)
    clamp = gmsh.model.occ.addPlaneSurface(
        [gmsh.model.occ.addCurveLoop([arc, l_right, -l_top, -l_left])])
    wire = gmsh.model.occ.addRectangle(-Lw / 2.0, -gap - D, 0.0, Lw, D)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(2, [clamp], tag=CLAMP_WIRE.cell_clamp)
    gmsh.model.addPhysicalGroup(2, [wire], tag=CLAMP_WIRE.cell_wire)
    gmsh.model.addPhysicalGroup(1, [arc], tag=CLAMP_WIRE.clamp_contact)
    gmsh.model.addPhysicalGroup(1, [l_top], tag=CLAMP_WIRE.clamp_top)

    tol = 1e-3 * min(D, Hc, Lc)
    top_wire, bot_wire = [], []
    for d, e in gmsh.model.getBoundary([(2, wire)]):
        ymin = gmsh.model.getBoundingBox(d, e)[1]
        ymax = gmsh.model.getBoundingBox(d, e)[4]
        if abs(ymin + gap) < tol and abs(ymax + gap) < tol:
            top_wire.append(e)
        elif abs(ymin + gap + D) < tol and abs(ymax + gap + D) < tol:
            bot_wire.append(e)
    if len(top_wire) != 1 or len(bot_wire) != 1:
        raise RuntimeError("单丝顶/底边不唯一：%d / %d" % (len(top_wire), len(bot_wire)))
    gmsh.model.addPhysicalGroup(1, top_wire, tag=CLAMP_WIRE.wire_contact)
    gmsh.model.addPhysicalGroup(1, bot_wire, tag=CLAMP_WIRE.wire_bottom)

    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", h_fine)
    gmsh.model.mesh.field.setNumber(1, "VOut", h_coarse)
    gmsh.model.mesh.field.setNumber(1, "XMin", -fine_halfwidth)
    gmsh.model.mesh.field.setNumber(1, "XMax", fine_halfwidth)
    gmsh.model.mesh.field.setNumber(1, "YMin", -0.75 * D)
    gmsh.model.mesh.field.setNumber(1, "YMax", 0.25 * Hc)
    gmsh.model.mesh.field.setNumber(1, "Thickness", fine_halfwidth)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    if quads:
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 2)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    gmsh.write(f"{stem}.msh")


class ContactModel:
    """一个两体摩擦接触问题：装配、加载、求解、取面力。"""

    def __init__(self, mesh, cell_tags, facet_tags, *, E, nu,
                 order: int = 1, quadrature_degree: int = 5,
                 tags=HALFDISK_PLANE, sample_center=None, plastic=None):
        """E、nu：标量（两体同材料）或长度 2 的序列（按 tags.body_cells 顺序，分体取值）。

        plastic：{物体序号: J2Params}。给了的物体用 PlasticLaw（J2 各向同性硬化，历史存 DG-2，
        Phase 5）作体本构，其余物体仍线弹性。其 E、nu 须与上面一致。

        **已知近似（D-54）**：上游 Nitsche 接触核在 C++ 里写死线弹性应力（只接收 mu、lambda），
        所以接触项里的 sigma·n 是弹性试探应力，体积分里才是塑性应力。接触面上的间隙因此约为
        (sigma_el - sigma_true)·n/gamma（gamma ~ gamma_factor·E/h，量级 eps_p·h/gamma_factor）；
        是否小到可忽略由 penetration() / 力平衡 / gamma 敏感性实测，不凭论证。

        sample_center：采样接触弧的圆心 (cx, cy)。给了就用弧面真实法向 n 取
        p = -n·sigma·n、q = -tau·sigma·n；不给则按平面近似 n = (0, 1)（Hertz 算例的块体上表面）。
        D-51 的诊断里发现：曲面上直接用 sigma_yy/sigma_xy 会把 sigma_xx·(x/R) 混进 q，
        接触区边缘 p 很小时能把 |q|/(mu p) 抬到 3~4 倍，使库仑不等式测不准。
        """
        self.sample_center = sample_center
        self.mesh, self.cell_tags, self.facet_tags = mesh, cell_tags, facet_tags
        n_body = len(tags.body_cells)
        Es = list(E) if np.ndim(E) else [float(E)] * n_body
        nus = list(nu) if np.ndim(nu) else [float(nu)] * n_body
        self.E_body, self.nu_body = [float(x) for x in Es], [float(x) for x in nus]
        self.tags, self.order = tags, order
        # Nitsche 罚参数 γ ∝ E 取**较软**一侧：接触刚度由可变形的那一侧决定
        # （等效模量 1/E* = Σ(1-ν²)/E，刚体一侧贡献可忽略）。取较硬一侧会使 γ 过大，
        # 实测把 Newton 残差底从 1e-9 抬到 1e-7 并卡死不收敛（D-54）。
        self.E = min(self.E_body)
        self.nu = self.nu_body[int(np.argmin(self.E_body))]
        gdim = mesh.geometry.dim
        self.V = functionspace(mesh, ("CG", order, (gdim,)))

        mu_f, lmbda_f = lame_parameters(True)      # True = 平面应变
        V0 = functionspace(mesh, ("DG", 0))
        self.mu_c, self.lmbda_c = Function(V0), Function(V0)
        for k, cell_tag in enumerate(tags.body_cells):
            cells = cell_tags.find(cell_tag)
            Ek, nuk = self.E_body[k], self.nu_body[k]
            self.mu_c.interpolate(lambda x, Ek=Ek, nuk=nuk: np.full((1, x.shape[1]), mu_f(Ek, nuk)), cells)
            self.lmbda_c.interpolate(lambda x, Ek=Ek, nuk=nuk: np.full((1, x.shape[1]), lmbda_f(Ek, nuk)), cells)
        self.sigma = sigma_func(self.mu_c, self.lmbda_c)      # 线弹性（弹性体用）

        self.plastic = dict(plastic or {})
        self.laws = {}
        for k, prm in self.plastic.items():
            if abs(prm.E - self.E_body[k]) > 1e-9 * prm.E or abs(prm.nu - self.nu_body[k]) > 1e-12:
                raise ValueError("物体 %d 的 J2Params(E,nu) 与 ContactModel 的 E,nu 不一致" % k)
            self.laws[k] = PlasticLaw(mesh, prm)

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
        F = 0
        for k, cell_tag in enumerate(self.tags.body_cells):
            if k in self.laws:                      # 塑性体：PlasticLaw，固定求积阶数
                dxk = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.cell_tags,
                                  metadata={"quadrature_degree": 4})
                F += ufl.inner(self.laws[k].sigma(self.u), epsilon(v)) * dxk(cell_tag)
            else:
                F += ufl.inner(self.sigma(self.u), epsilon(v)) * self.dx(cell_tag)
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
        self.commit()
        return int(n_it)

    def commit(self) -> None:
        """塑性体：把收敛后的 (eps_p, p) 写回历史（只写该体的单元）。弹性体无操作。"""
        for k, law in self.laws.items():
            law.commit(self.u, cells=self.cell_tags.find(self.tags.body_cells[k]))

    def stress(self, u, body: int):
        """物体 body 的 UFL 应力：塑性体 -> PlasticLaw，弹性体 -> 线弹性。"""
        return self.laws[body].sigma(u) if body in self.laws else self.sigma(u)

    def plastic_fields(self) -> dict:
        """所有塑性体的历史场（给 checkpoint），键为 eps_p_<体> / p_eq_<体>。"""
        out = {}
        for k, law in self.laws.items():
            out["eps_p_%d" % k], out["p_eq_%d" % k] = law.ep_n, law.p_n
        return out

    def prepare_next(self) -> None:
        """为下一增量步更新接触搜索与矩阵（上游算例的标准做法）。"""
        self.problem.update_contact_detection(self.u)
        A = self.problem.create_matrix(self.J_c)
        self._set_nullspace(A)
        self._newton.set_petsc_matrix(A)
        # 取上一步的一小部分作初值：全无 Dirichlet 约束时可避免奇异矩阵
        self.du.x.array[:] = 0.1 * self.du.x.array[:]
        self.problem.update_contact_data(self.du)

    # ---- 接触弧节点位移（微滑幅，Phase 4.5）-------------------------------
    def arc_nodes(self, body: int):
        """第 `body` 个物体（0 上 / 1 下）接触弧上的 (X_ref (N,2), u_total (N,2))。

        节点取自位移空间 V 在 `tags.contact_facets[body]` 上的自由度，参考坐标即网格初始坐标
        （网格不动），u 为累计位移 `self.u`。按参考 x 排序，allgather 后各 rank 一致。
        """
        comm = self.mesh.comm
        fdim = self.mesh.topology.dim - 1
        dofs = locate_dofs_topological(
            self.V, fdim, self.facet_tags.find(self.tags.contact_facets[body]))
        n_owned = self.V.dofmap.index_map.size_local
        dofs = dofs[dofs < n_owned]                   # 并行下只留本进程拥有的
        gdim = self.mesh.geometry.dim
        X = self.V.tabulate_dof_coordinates()[dofs, :gdim]
        u = self.u.x.array.reshape(-1, gdim)[dofs]
        X = np.concatenate(comm.allgather(X))
        u = np.concatenate(comm.allgather(u))
        o = np.argsort(X[:, 0], kind="stable")
        return X[o], u[o]

    # ---- 取面力 ---------------------------------------------------------
    def resultant(self, component: int, tag: int | None = None,
                  body: int | None = None) -> float:
        """某**平面**上的合力分量，用**精确面积分** ∫ -sigma[1, component] ds。

        默认取 `tags.force_facet`：对半圆柱+块是接触面本身（平面），
        对圆柱–圆柱是上圆柱顶面（曲面接触弧不能这样读力，见 force_facet
        的注释）。两者外法向都是 (0, 1)，故 component=1 得法向力、=0 得切向力。

        比对节点值做梯形积分可靠得多：后者会把接触区外被 CG1 抹出来的
        数值残尾也算进去（D-49，Phase 4.3 实测差 3.1% vs 0.0014%）。
        """
        tag = self.tags.force_facet if tag is None else tag
        if body is None:
            body = self.tags.force_body if tag in (None, self.tags.force_facet) else None
        if body is None:
            raise ValueError("resultant(tag=%r)：请显式给 body（物体序号）" % (tag,))
        local = assemble_scalar(form(-self.stress(self.u, body)[1, component] * self.ds(tag)))
        return float(self.mesh.comm.allreduce(local, op=MPI.SUM))

    def _body_of_facet_tag(self, tag: int) -> int:
        """读合力只支持 force_facet（其所属物体由 tags.force_body 给定）。"""
        raise ValueError("resultant(tag=%r)：只支持 force_facet=%r；其他面请显式扩展"
                         % (tag, self.tags.force_facet))

    def penetration(self):
        """接触弧两侧按 x 配对的法向间隙 (x, g_n)，g_n < 0 表示穿透（Nitsche 诊断量）。

        法向取下体（body 1）接触弧的外法向：有 sample_center 用弧面法向，否则 (0, 1)。
        配对沿 x（半平面近似）：对下弧每个节点，在上弧当前构形里插值同 x 处的点。
        """
        X_up, u_up = self.arc_nodes(0)
        X_lo, u_lo = self.arc_nodes(1)
        P_up, P_lo = X_up + u_up, X_lo + u_lo
        o = np.argsort(P_up[:, 0])
        yu = np.interp(P_lo[:, 0], P_up[o, 0], P_up[o, 1])
        inside = (P_lo[:, 0] >= P_up[o, 0][0]) & (P_lo[:, 0] <= P_up[o, 0][-1])
        if self.sample_center is None:
            ny = np.ones(len(P_lo))
        else:
            d = X_lo - np.asarray(self.sample_center, float)
            ny = d[:, 1] / np.linalg.norm(d, axis=1)
        g = (yu - P_lo[:, 1]) * ny                      # 沿 x 配对：只有 y 向分量参与
        o2 = np.argsort(P_lo[:, 0])
        return P_lo[o2, 0][inside[o2]], g[o2][inside[o2]]

    def surface_tractions(self):
        """接触采样面的 (x, p, q)：法向压力 p = -sigma_yy、切向面力 q = -sigma_xy。

        采样面取 `tags.sample_facet`（半圆柱算例是块体上表面，圆柱–圆柱算例
        是下圆柱上弧）。接触区位于 x≈0、y≈-R 附近，那里外法向 ≈ (0, 1)，
        故法向/切向分量就是 sigma 的两个分量 —— **不需要 FacetNormal，
        也不需要 submesh**，正好绕开 D-48 的两个坑。对圆柱弧这是半平面近似，
        在接触区（|x| << R）内误差可忽略，恰是 Cattaneo–Mindlin 本身的前提。
        """
        Vs = functionspace(self.mesh, ("CG", self.order))
        body = self.cell_tags.find(self.tags.sample_cell)
        sig = self.stress(self.u, list(self.tags.body_cells).index(self.tags.sample_cell))
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


class SlipTracker:
    """微滑幅追踪（口径见 `pfwire.slip`，D-52）。

    在**阶段 1（无摩擦压下）结束后**构造：冻结配对 s、记下基线相对位移 g_base。
    之后每步 `measure()` 返回相对基线的切向滑移增量。
    """

    def __init__(self, model: ContactModel, center: tuple[float, float]):
        self.model, self.center = model, center
        self.X_up, u_up = model.arc_nodes(0)
        self.X_lo, u_lo = model.arc_nodes(1)
        self.tau = _slip.arc_tangent(self.X_lo, center)
        self.s = _slip.pair_material_points(self.X_up, u_up, self.X_lo, u_lo)
        self.g_base = _slip.relative_tangential_displacement(
            self.X_up, u_up, self.X_lo, u_lo, self.s, self.tau)

    def measure(self):
        """返回 (x_lower_ref, slip)：当前累计位移下相对基线的切向滑移增量（配不上的点为 nan）。"""
        # 上弧节点的参考坐标不变；位移取最新的 u（节点顺序与构造时一致）
        _, u_up = self.model.arc_nodes(0)
        _, u_lo = self.model.arc_nodes(1)
        g = _slip.relative_tangential_displacement(
            self.X_up, u_up, self.X_lo, u_lo, self.s, self.tau)
        return self.X_lo[:, 0].copy(), _slip.slip_increment(g, self.g_base)
