"""刚体圆弧障碍（pfwire.obstacle，D-57 S1）的几何与罚函数测试。

测试意图：
  - 间隙、法向、罚压力的 UFL 表达式与解析式逐点一致（u = 0 时 g = sqrt(x² + R²) − R）；
  - 线夹下压 δ 后，单丝表面中心的间隙为 −δ（穿透判据）；
  - 罚压力只在穿透时非零，且与穿透量线性相关；
  - 网格两个边界面 tag 存在，面积正确（并行下用 allreduce 汇总，不写按 rank 的断言）。
"""

import pathlib

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.mesh import CellType, create_rectangle, create_unit_square
from mpi4py import MPI

from pfwire.obstacle import (
    BLOCK_BOTTOM,
    BLOCK_TOP,
    build_block_mesh,
    clamp_gap,
    clamp_normal,
    penalty_pressure,
    penalty_stiffness,
)

R = 0.33  # 线夹轴向曲率半径，m（PLAN §1.1，D-50）


def _integrate(mesh, expr):
    """在 mesh 上对 UFL 标量表达式做面积分（单进程）。"""
    return fem.assemble_scalar(fem.form(expr * ufl.dx(domain=mesh)))


def test_gap_on_top_surface_matches_closed_form():
    """δ = 0：表面点 (x, 0) 的间隙应为 sqrt(x² + R²) − R。

    用一个沿 y = 0 的网格（顶点恰在表面上），在顶点处插值后与解析式对比。
    """
    xmax = 6e-3
    mesh = create_rectangle(MPI.COMM_SELF, [[-xmax, 0.0], [xmax, 1e-3]], [12, 2],
                            CellType.triangle)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    x = ufl.SpatialCoordinate(mesh)
    g = clamp_gap(x, R, 0.0)
    f = fem.Function(V)
    f.interpolate(fem.Expression(g, V.element.interpolation_points))
    X = V.tabulate_dof_coordinates()
    top = np.isclose(X[:, 1], 0.0, atol=1e-15)
    assert top.any()
    expected = np.sqrt(X[top, 0] ** 2 + R**2) - R
    np.testing.assert_allclose(f.x.array[top], expected, rtol=1e-10, atol=1e-18)


def test_gap_at_surface_centre_equals_minus_delta():
    """线夹下压 δ：x = 0 处表面间隙为 −δ（圆心下移 δ，表面点落进圆内 δ）。"""
    delta = 2e-6
    mesh = create_unit_square(MPI.COMM_SELF, 1, 1)
    g = clamp_gap(ufl.as_vector([0.0, 0.0]), R, delta)
    # 被积函数为常数，单位面积上的积分即其值
    assert _integrate(mesh, g) == pytest.approx(-delta, rel=1e-9)


def test_normal_is_unit_and_points_down_from_centre():
    mesh = create_unit_square(MPI.COMM_SELF, 1, 1)
    p = ufl.as_vector([3e-3, 0.0])
    n = clamp_normal(p, R, 0.0)
    assert _integrate(mesh, ufl.sqrt(ufl.inner(n, n))) == pytest.approx(1.0, rel=1e-12)
    # 圆心在 (0, R)，表面点在其下方：法向朝下，且 |n_x| << 1
    assert _integrate(mesh, n[1]) < 0.0
    assert abs(_integrate(mesh, n[0])) < 1e-2


def test_penalty_is_zero_when_separated_and_linear_when_penetrated():
    eps = 1.0e12
    mesh = create_unit_square(MPI.COMM_SELF, 1, 1)
    gap_sep = fem.Constant(mesh, 1e-6)
    gap_pen = fem.Constant(mesh, -1e-6)
    assert _integrate(mesh, penalty_pressure(gap_sep, eps)) == 0.0
    assert _integrate(mesh, penalty_pressure(gap_pen, eps)) == pytest.approx(eps * 1e-6, rel=1e-12)


def test_penalty_stiffness_scaling_and_validation():
    assert penalty_stiffness(10.0, 69e9, 50e-6) == pytest.approx(10.0 * 69e9 / 50e-6)
    with pytest.raises(ValueError):
        penalty_stiffness(0.0, 69e9, 50e-6)
    with pytest.raises(ValueError):
        penalty_stiffness(10.0, 69e9, -1.0)


def test_block_mesh_has_boundary_tags_and_correct_area(tmp_path: pathlib.Path):
    width, depth = 20e-3, 10e-3
    mesh, _ct, ft = build_block_mesh(
        tmp_path / "block", width=width, depth=depth,
        h_fine=1e-3, h_coarse=4e-3, fine_halfwidth=3e-3, fine_depth=3e-3,
        transition=2e-3, comm=MPI.COMM_WORLD)
    comm = mesh.comm
    n_top = comm.allreduce(int(ft.find(BLOCK_TOP).size), op=MPI.SUM)
    n_bot = comm.allreduce(int(ft.find(BLOCK_BOTTOM).size), op=MPI.SUM)
    assert n_top > 0 and n_bot > 0
    area_local = fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * ufl.dx(domain=mesh)))
    assert comm.allreduce(area_local, op=MPI.SUM) == pytest.approx(width * depth, rel=1e-9)
