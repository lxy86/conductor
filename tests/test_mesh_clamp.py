"""矩形单丝网格（pfwire.mesh.block_mesh / clamp_mesh，Phase 6.2）的测试。

测试意图：
  - 两个边界面 tag 存在、总面积正确（并行下用 allreduce，不写按 rank 的断言）；
  - 加密盒内单元面积接近 h_fine²，盒外接近 h_coarse²（加密真的生效）；
  - clamp_mesh 的加密半宽等于 halfwidth_factor · a_pl（PLAN §6.2：按实测 a_pl 定加密区）；
  - 非法输入被拒绝。
"""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.mesh import compute_midpoints
from mpi4py import MPI

from pfwire.mesh import BLOCK_BOTTOM, BLOCK_TOP, block_mesh, clamp_mesh

WIDTH, DEPTH = 20e-3, 3e-3
H_FINE, H_COARSE, TRANSITION = 0.1e-3, 1.0e-3, 0.5e-3


def _cell_areas(mesh):
    """每个本地单元的面积（DG0 测试函数的装配向量）。"""
    V0 = fem.functionspace(mesh, ("DG", 0))
    v = ufl.TestFunction(V0)
    vec = fem.assemble_vector(fem.form(v * ufl.dx(domain=mesh)))
    n = mesh.topology.index_map(mesh.topology.dim).size_local
    return np.asarray(vec.array[:n]).copy(), compute_midpoints(
        mesh, mesh.topology.dim, np.arange(n, dtype=np.int32))


def test_block_mesh_boundary_tags_and_area(tmp_path):
    mesh, _ct, ft = block_mesh(
        tmp_path / "b", width=WIDTH, depth=DEPTH,
        refine_boxes=[(-2e-3, 2e-3, -DEPTH, 0.0)],
        h_fine=H_FINE, h_coarse=H_COARSE, transition=TRANSITION, comm=MPI.COMM_WORLD)
    comm = mesh.comm
    assert comm.allreduce(int(ft.find(BLOCK_TOP).size), op=MPI.SUM) > 0
    assert comm.allreduce(int(ft.find(BLOCK_BOTTOM).size), op=MPI.SUM) > 0
    areas, _ = _cell_areas(mesh)
    assert comm.allreduce(float(areas.sum()), op=MPI.SUM) == pytest.approx(WIDTH * DEPTH, rel=1e-9)


def test_refinement_inside_box_and_coarsening_outside(tmp_path):
    """盒内外的单元尺度比应接近 (h_coarse / h_fine)²。

    注意：gmsh 的四边形单元面积 ≈ 0.15–0.2 · h²（实测，D-58），所以不能拿面积直接等于 h²；
    这里只比**比值**，并允许一个数量级的余量（gmsh 的 Box 场是逐点近似）。
    """
    half = 2e-3
    mesh, _ct, _ft = block_mesh(
        tmp_path / "r", width=WIDTH, depth=DEPTH,
        refine_boxes=[(-half, half, -DEPTH, 0.0)],
        h_fine=H_FINE, h_coarse=H_COARSE, transition=TRANSITION, comm=MPI.COMM_WORLD)
    areas, mid = _cell_areas(mesh)
    inside = (np.abs(mid[:, 0]) < half - 2 * H_FINE) & (mid[:, 1] > -DEPTH + 2 * H_FINE)
    far = np.abs(mid[:, 0]) > half + 3 * TRANSITION
    assert inside.any() and far.any()
    ratio = np.median(areas[far]) / np.median(areas[inside])
    ideal = (H_COARSE / H_FINE) ** 2
    assert ideal / 3 < ratio < ideal * 3, "单元面积比 %.1f 偏离 (h_c/h_f)² = %.0f" % (ratio, ideal)


def test_clamp_mesh_halfwidth_follows_a_pl(tmp_path):
    """加密半宽 = halfwidth_factor · a_pl：盒内单元应比盒外细，分界在 factor·a_pl 附近。"""
    a_pl, factor = 1.0e-3, 2.0
    mesh, _ct, _ft = clamp_mesh(
        tmp_path / "c", width=WIDTH, depth=DEPTH, a_pl=a_pl, halfwidth_factor=factor,
        fine_depth=DEPTH, h_fine=H_FINE, h_coarse=H_COARSE, transition=TRANSITION,
        comm=MPI.COMM_WORLD)
    areas, mid = _cell_areas(mesh)
    hw = factor * a_pl
    near = np.abs(mid[:, 0]) < hw - 2 * H_FINE
    beyond = np.abs(mid[:, 0]) > hw + 3 * TRANSITION
    assert near.any() and beyond.any()
    assert np.median(areas[beyond]) / np.median(areas[near]) > 30


def test_clamp_mesh_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        clamp_mesh(tmp_path / "x", width=WIDTH, depth=DEPTH, a_pl=0.0, halfwidth_factor=3.0,
                   fine_depth=DEPTH, h_fine=H_FINE, h_coarse=H_COARSE, transition=TRANSITION)
    with pytest.raises(ValueError):
        block_mesh(tmp_path / "y", width=WIDTH, depth=DEPTH, refine_boxes=[],
                   h_fine=H_FINE, h_coarse=H_COARSE, transition=TRANSITION)
