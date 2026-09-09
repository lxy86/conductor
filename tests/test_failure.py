"""Phase 1.4 单测：失效判据与裂纹几何提取。

测试意图
--------
把两件事分开测，各自都有精确答案：

A. ``extract_crack`` 的几何逻辑 —— 直接喂**合成网格数组**，不经有限元，
   故期望值是手算的精确几何：
   1. 竖直裂纹：深度、萌生位置、倾角 ≈ 0；
   2. 斜裂纹：倾角与构造角度吻合（这条同时验证必须用 8 连通，
      4 连通会把斜裂纹判成一串孤立点）；
   3. **内部孤立损伤斑不算裂纹**（不与表面连通就必须被忽略）；
   4. 两条表面裂纹：报 n_surface_components = 2，深度取更深的那条；
   5. 无损伤 / 全部低于阈值：depth = 0、位置量为 None、未失效。
B. ``DamageGrid.sample`` 的采样正确性 —— 用**线性**场，P1 插值对线性函数是精确的，
   故每个采样点的值都能手算；顺带验证并行下所有 rank 拿到同一个数组、
   且没有采样点落在域外。
C. ``FailureCriterion`` 的配置解析：depth 与 depth_fraction 只能给一个，
   两个都给或都不给必须报错（不许静默选一个）。
D. ``CrackPathGrid``（验证 2 的 CT 用，生长方向任意）：
   1. **与 DamageGrid 对拍** —— 取 growth_dir = (0,−1) 时两者语义等价，
      同一个 d 场上必须给出同一个深度。这是最强的一条：它保证新类没有
      引入方向/索引错误，而不是只自证其说；
   2. 沿 +x 的水平裂纹（CT 的真实方向）长度正确 —— DamageGrid 表达不了这个；
   3. 45° 方向的裂纹用旋转后的 growth_dir 也能正确量出，验证方向抽象没写反；
   4. 参数校验：spacing 过粗、零向量、伸出计算域。
"""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pfwire import failure as fl

SPACING = 1.0e-6
NX, ND = 41, 31  # x: 40 µm 宽, depth: 30 µm 深


def make_coords():
    x = SPACING * (np.arange(NX) - (NX - 1) // 2)  # 以 x = 0 为中心
    depth = SPACING * np.arange(ND)
    return x, depth


def blank_field(x, depth):
    return np.zeros((depth.size, x.size))


def paint_column(values, x, depth, *, x_center: float, depth_max: float, half_width: float = 1.5e-6):
    """在给定 x 处画一条竖直裂纹（d = 1），深度到 depth_max。"""
    cols = np.abs(x - x_center) <= half_width + 1e-15
    rows = depth <= depth_max + 1e-15
    values[np.ix_(rows, cols)] = 1.0


def paint_inclined(values, x, depth, *, x_start: float, angle_deg: float, depth_max: float):
    """画一条自表面起、与竖直方向成 angle_deg 的斜裂纹（朝 +x 为正）。"""
    slope = np.tan(np.radians(angle_deg))
    for r, dep in enumerate(depth):
        if dep > depth_max + 1e-15:
            break
        xc = x_start + slope * dep
        col = int(np.argmin(np.abs(x - xc)))
        values[r, col] = 1.0


# --------------------------------------------------------------------------
# A. 几何提取逻辑
# --------------------------------------------------------------------------
def test_vertical_crack_depth_and_position():
    x, depth = make_coords()
    v = blank_field(x, depth)
    paint_column(v, x, depth, x_center=5.0e-6, depth_max=12.0e-6)
    m = fl.extract_crack(v, x, depth)
    assert m.has_surface_crack
    assert m.n_surface_components == 1
    assert m.depth == pytest.approx(12.0e-6, abs=1e-15)
    assert m.x_surface == pytest.approx(5.0e-6, abs=1e-15)
    assert m.x_tip == pytest.approx(5.0e-6, abs=1e-15)
    assert abs(m.inclination_deg) < 1.0e-9
    assert m.max_damage == pytest.approx(1.0)
    # 面积 = 像素数 * spacing^2：宽 3 列 × 深 13 行
    assert m.area == pytest.approx(3 * 13 * SPACING**2, rel=1e-12)


@pytest.mark.parametrize("angle", [0.0, 15.0, 30.0, 45.0, -30.0])
def test_inclined_crack_angle(angle: float):
    """斜裂纹的倾角必须被还原；这条也验证了 8 连通的必要性。"""
    x, depth = make_coords()
    v = blank_field(x, depth)
    paint_inclined(v, x, depth, x_start=0.0, angle_deg=angle, depth_max=20.0e-6)
    m = fl.extract_crack(v, x, depth, inclination_depth=20.0e-6)
    assert m.has_surface_crack, f"angle={angle}: 斜裂纹没被判为与表面连通"
    assert m.depth == pytest.approx(20.0e-6, abs=1e-15)
    assert m.inclination_deg == pytest.approx(angle, abs=2.0), (
        f"angle={angle}: 提取到 {m.inclination_deg}"
    )


def test_four_connectivity_would_break_inclined_crack():
    """反证：45° 斜裂纹只有对角相邻，4 连通会把它切成 21 段。"""
    from scipy import ndimage

    x, depth = make_coords()
    v = blank_field(x, depth)
    paint_inclined(v, x, depth, x_start=0.0, angle_deg=45.0, depth_max=20.0e-6)
    damaged = v >= 0.95
    _, n8 = ndimage.label(damaged, structure=np.ones((3, 3), dtype=bool))
    _, n4 = ndimage.label(damaged)  # 默认 4 连通
    assert n8 == 1
    assert n4 > 1, "4 连通居然没把斜裂纹切断，测试前提不成立"


def test_interior_blob_not_connected_to_surface_is_ignored():
    x, depth = make_coords()
    v = blank_field(x, depth)
    # 深度 10~20 µm 的内部损伤斑，不触及表面
    rows = (depth >= 10.0e-6) & (depth <= 20.0e-6)
    cols = np.abs(x) <= 3.0e-6
    v[np.ix_(rows, cols)] = 1.0
    m = fl.extract_crack(v, x, depth)
    assert not m.has_surface_crack
    assert m.depth == 0.0
    assert m.x_surface is None and m.x_tip is None
    assert m.max_damage == pytest.approx(1.0)  # 仍报告最大损伤，便于诊断
    assert not fl.FailureCriterion(depth=5.0e-6).is_failed(m)


def test_deepest_of_two_surface_cracks_wins():
    x, depth = make_coords()
    v = blank_field(x, depth)
    paint_column(v, x, depth, x_center=-10.0e-6, depth_max=8.0e-6)
    paint_column(v, x, depth, x_center=10.0e-6, depth_max=18.0e-6)
    m = fl.extract_crack(v, x, depth)
    assert m.n_surface_components == 2  # 两侧接触边缘各一条，本身就是诊断信息
    assert m.depth == pytest.approx(18.0e-6, abs=1e-15)
    assert m.x_surface == pytest.approx(10.0e-6, abs=1e-15)


def test_no_damage_and_below_threshold():
    x, depth = make_coords()
    v = blank_field(x, depth)
    m = fl.extract_crack(v, x, depth)
    assert not m.has_surface_crack and m.depth == 0.0 and m.max_damage == 0.0

    v[:, :] = 0.94  # 全场略低于阈值
    m = fl.extract_crack(v, x, depth, threshold=0.95)
    assert not m.has_surface_crack
    assert m.max_damage == pytest.approx(0.94)
    # 阈值下调后同一个场就成了「全场贯穿」
    m2 = fl.extract_crack(v, x, depth, threshold=0.9)
    assert m2.has_surface_crack
    assert m2.depth == pytest.approx(depth[-1], abs=1e-15)


def test_extract_crack_rejects_bad_input():
    x, depth = make_coords()
    v = blank_field(x, depth)
    with pytest.raises(ValueError, match="形状"):
        fl.extract_crack(v[:-1], x, depth)
    with pytest.raises(ValueError, match="threshold"):
        fl.extract_crack(v, x, depth, threshold=1.0)


def test_angle_needs_enough_rows():
    """裂纹太浅（窗口内行数不足）时倾角必须返回 None，而不是硬凑一个数。"""
    x, depth = make_coords()
    v = blank_field(x, depth)
    paint_column(v, x, depth, x_center=0.0, depth_max=1.0e-6)  # 只有 2 行
    m = fl.extract_crack(v, x, depth, min_rows_for_angle=3)
    assert m.has_surface_crack and m.depth == pytest.approx(1.0e-6)
    assert m.inclination_deg is None


# --------------------------------------------------------------------------
# B. 采样正确性
# --------------------------------------------------------------------------
def test_grid_sample_is_exact_for_linear_field():
    """P1 插值对线性函数精确，故每个采样点的值可手算。"""
    L, H = 40.0e-6, 30.0e-6
    msh = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([-0.5 * L, -H]), np.array([0.5 * L, 0.0])],
        [20, 15],
        cell_type=dmesh.CellType.triangle,
    )
    V = fem.functionspace(msh, ("Lagrange", 1))
    f = fem.Function(V)
    # d(x,y) = 3 + 1e5 x - 2e5 y  （线性）
    f.interpolate(lambda p: 3.0 + 1.0e5 * p[0] - 2.0e5 * p[1])

    spec = fl.GridSpec(
        x_min=-0.4 * L, x_max=0.4 * L, depth_max=0.8 * H, spacing=2.0e-6
    )
    grid = fl.DamageGrid(msh, spec)
    got = grid.sample(f)
    xx, dd = np.meshgrid(grid.x_coords, grid.depth_coords)
    expected = 3.0 + 1.0e5 * xx - 2.0e5 * (spec.y_surface - dd)
    assert got.shape == expected.shape
    assert np.allclose(got, expected, rtol=1e-10, atol=1e-10)
    # 并行一致性：所有 rank 必须拿到同一个数组
    spread = msh.comm.allreduce(float(np.max(np.abs(got))), op=MPI.MAX) - msh.comm.allreduce(
        float(np.max(np.abs(got))), op=MPI.MIN
    )
    assert spread == pytest.approx(0.0)


def test_grid_outside_mesh_is_rejected():
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    spec = fl.GridSpec(x_min=0.1, x_max=0.9, depth_max=2.0, spacing=0.1, y_surface=1.0)
    with pytest.raises(ValueError, match="计算域之外"):
        fl.DamageGrid(msh, spec)


def test_grid_spacing_must_resolve_mesh():
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    spec = fl.GridSpec(x_min=0.1, x_max=0.9, depth_max=0.5, spacing=0.1, y_surface=1.0)
    with pytest.raises(ValueError, match="低估裂纹深度"):
        fl.DamageGrid(msh, spec, h_fine=0.05)


def test_grid_spec_validation():
    with pytest.raises(ValueError, match="x_max"):
        fl.GridSpec(x_min=1.0, x_max=0.0, depth_max=1.0, spacing=0.1)
    with pytest.raises(ValueError, match="必须为正"):
        fl.GridSpec(x_min=0.0, x_max=1.0, depth_max=-1.0, spacing=0.1)
    with pytest.raises(ValueError, match="太粗"):
        fl.GridSpec(x_min=0.0, x_max=1.0, depth_max=1.0, spacing=0.9)


# --------------------------------------------------------------------------
# C. 失效判据与配置解析
# --------------------------------------------------------------------------
def test_failure_criterion_threshold_behaviour():
    x, depth = make_coords()
    v = blank_field(x, depth)
    paint_column(v, x, depth, x_center=0.0, depth_max=10.0e-6)
    m = fl.extract_crack(v, x, depth)
    assert fl.FailureCriterion(depth=10.0e-6).is_failed(m)
    assert fl.FailureCriterion(depth=9.9e-6).is_failed(m)
    assert not fl.FailureCriterion(depth=10.1e-6).is_failed(m)


def test_failure_criterion_from_config():
    # D-1 的绝对深度写法
    c = fl.FailureCriterion.from_config({"d_threshold": 0.95, "depth": 1.0e-4})
    assert c.depth == pytest.approx(1.0e-4) and c.d_threshold == pytest.approx(0.95)
    # PLAN 第 5 节的 depth_fraction 写法
    c2 = fl.FailureCriterion.from_config(
        {"d_threshold": 0.95, "depth_fraction": 0.5}, R_wire=1.875e-3
    )
    assert c2.depth == pytest.approx(0.5 * 1.875e-3)
    # 两个都给 -> 报错（不许静默选一个）
    with pytest.raises(ValueError, match="同时给了"):
        fl.FailureCriterion.from_config({"depth": 1e-4, "depth_fraction": 0.5})
    # 都不给 -> 报错
    with pytest.raises(ValueError, match="必须给"):
        fl.FailureCriterion.from_config({"d_threshold": 0.95})
    # depth_fraction 缺 R_wire
    with pytest.raises(ValueError, match="R_wire"):
        fl.FailureCriterion.from_config({"depth_fraction": 0.5})
    # 拼错的键
    with pytest.raises(KeyError, match="failure"):
        fl.FailureCriterion.from_config({"dept": 1e-4})


def test_failure_criterion_validation():
    with pytest.raises(ValueError, match="d_threshold"):
        fl.FailureCriterion(d_threshold=1.0, depth=1e-4)
    with pytest.raises(ValueError, match="depth"):
        fl.FailureCriterion(depth=0.0)
    assert "0.95" in fl.FailureCriterion().describe()


# --------------------------------------------------------------------------
# D. CrackPathGrid（任意生长方向）
# --------------------------------------------------------------------------
PG_L = 40.0e-6  # 测试域边长


@pytest.fixture(scope="module")
def square_mesh():
    """[-L/2, L/2]^2 的方域，用于 CrackPathGrid 的采样测试。"""
    return dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([-0.5 * PG_L, -0.5 * PG_L]), np.array([0.5 * PG_L, 0.5 * PG_L])],
        [40, 40],
        dmesh.CellType.triangle,
    )


def strip_damage(msh, *, along: str, length: float, halfwidth: float):
    """造一个「条带损伤」场：沿指定方向从原点延伸 length、半宽 halfwidth 内 d = 1。

    用 P1 插值一个分段常数会被平滑，故这里直接按节点坐标赋值 —— 采样点落在
    单元内时取的是 P1 插值，条带边界附近会有过渡，测试的容差按 spacing 给。
    """
    V = fem.functionspace(msh, ("Lagrange", 1))
    d = fem.Function(V)
    coords = V.tabulate_dof_coordinates()
    x, y = coords[:, 0], coords[:, 1]
    if along == "+x":
        inside = (x >= 0.0) & (x <= length) & (np.abs(y) <= halfwidth)
    elif along == "-y":
        inside = (y <= 0.0) & (y >= -length) & (np.abs(x) <= halfwidth)
    elif along == "45":
        # 沿 (1,1)/sqrt(2)：投影 s 在 [0,length]，横向 |t| <= halfwidth
        s = (x + y) / np.sqrt(2.0)
        t = (-x + y) / np.sqrt(2.0)
        inside = (s >= 0.0) & (s <= length) & (np.abs(t) <= halfwidth)
    else:
        raise ValueError(along)
    d.x.array[:] = np.where(inside, 1.0, 0.0)
    d.x.scatter_forward()
    return d


def test_crack_path_grid_agrees_with_damage_grid(square_mesh):
    """**对拍**：growth_dir = (0,−1) 时与 DamageGrid 语义等价，深度必须一致。"""
    msh = square_mesh
    spacing = PG_L / 40.0
    length, halfwidth = 12.0e-6, 3.0e-6
    d = strip_damage(msh, along="-y", length=length, halfwidth=halfwidth)

    # DamageGrid：表面在 y = 0，向下取深度
    gs = fl.GridSpec(
        x_min=-10.0e-6,
        x_max=10.0e-6,
        depth_max=18.0e-6,
        spacing=spacing,
        y_surface=0.0,
    )
    dg = fl.DamageGrid(msh, gs)
    m_dg = fl.extract_crack(dg.sample(d), dg.x_coords, dg.depth_coords, threshold=0.95)

    # CrackPathGrid：同一起点、生长方向 −y
    pg = fl.CrackPathGrid(
        msh,
        origin=(0.0, 0.0),
        growth_length=18.0e-6,
        halfwidth=10.0e-6,
        spacing=spacing,
        growth_dir=(0.0, -1.0),
    )
    m_pg = pg.crack_extension(d, threshold=0.95)

    assert m_pg.depth == pytest.approx(m_dg.depth, abs=1e-15)
    assert m_pg.has_surface_crack == m_dg.has_surface_crack


def test_crack_path_grid_measures_horizontal_crack(square_mesh):
    """沿 +x 的裂纹（CT 的真实方向）——DamageGrid 的 −y 约定表达不了。"""
    msh = square_mesh
    spacing = PG_L / 40.0
    length, halfwidth = 12.0e-6, 3.0e-6
    d = strip_damage(msh, along="+x", length=length, halfwidth=halfwidth)
    pg = fl.CrackPathGrid(
        msh,
        origin=(0.0, 0.0),
        growth_length=18.0e-6,
        halfwidth=10.0e-6,
        spacing=spacing,
        growth_dir=(1.0, 0.0),
    )
    m = pg.crack_extension(d, threshold=0.95)
    assert m.has_surface_crack
    # 分辨率是 spacing，故容差给一个间距
    assert m.depth == pytest.approx(length, abs=spacing)


def test_crack_path_grid_handles_rotated_direction(square_mesh):
    """45° 方向：用旋转后的 growth_dir 必须量出同样的长度。"""
    msh = square_mesh
    spacing = PG_L / 40.0
    length, halfwidth = 12.0e-6, 3.0e-6
    d = strip_damage(msh, along="45", length=length, halfwidth=halfwidth)
    pg = fl.CrackPathGrid(
        msh,
        origin=(0.0, 0.0),
        growth_length=16.0e-6,
        halfwidth=6.0e-6,
        spacing=spacing,
        growth_dir=(1.0, 1.0),  # 不必是单位向量，内部归一化
    )
    m = pg.crack_extension(d, threshold=0.95)
    assert m.has_surface_crack
    assert m.depth == pytest.approx(length, abs=2.0 * spacing)


def test_crack_path_grid_no_damage_gives_zero(square_mesh):
    msh = square_mesh
    V = fem.functionspace(msh, ("Lagrange", 1))
    d = fem.Function(V)  # 全 0
    pg = fl.CrackPathGrid(
        msh,
        origin=(0.0, 0.0),
        growth_length=15.0e-6,
        halfwidth=6.0e-6,
        spacing=PG_L / 40.0,
    )
    m = pg.crack_extension(d)
    assert m.depth == 0.0
    assert not m.has_surface_crack


def test_crack_path_grid_validation(square_mesh):
    msh = square_mesh
    spacing = PG_L / 40.0
    with pytest.raises(ValueError, match="太粗"):
        fl.CrackPathGrid(
            msh, origin=(0.0, 0.0), growth_length=spacing, halfwidth=5e-6, spacing=spacing
        )
    with pytest.raises(ValueError, match="零向量"):
        fl.CrackPathGrid(
            msh,
            origin=(0.0, 0.0),
            growth_length=15e-6,
            halfwidth=5e-6,
            spacing=spacing,
            growth_dir=(0.0, 0.0),
        )
    with pytest.raises(ValueError, match="必须为正"):
        fl.CrackPathGrid(
            msh, origin=(0.0, 0.0), growth_length=-1.0, halfwidth=5e-6, spacing=spacing
        )
    with pytest.raises(ValueError, match="h_fine"):
        fl.CrackPathGrid(
            msh,
            origin=(0.0, 0.0),
            growth_length=15e-6,
            halfwidth=5e-6,
            spacing=spacing,
            h_fine=0.1 * spacing,
        )
    # 伸出计算域必须报错，而不是静默把域外点当无损伤
    with pytest.raises(ValueError, match="计算域之外"):
        fl.CrackPathGrid(
            msh,
            origin=(0.0, 0.0),
            growth_length=5.0 * PG_L,
            halfwidth=5e-6,
            spacing=spacing,
        )
