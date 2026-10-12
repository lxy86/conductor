"""Phase 6.1 基础件：ContactModel 的塑性体路径（分体材料、commit、penetration）+ Prandtl 系数。

完整的两网格 Prandtl 对拍在 scripts/verify_prandtl_strip.py（约 20 分钟），不进 pytest；
这里跑**粗网格**版本，只验证路径正确与走向合理。
"""
import importlib.util
import math
import pathlib

import numpy as np
import pytest

pytest.importorskip("dolfinx")
import ufl  # noqa: E402
from dolfinx import fem, mesh as dmesh  # noqa: E402
from dolfinx.fem.petsc import NonlinearProblem  # noqa: E402
from mpi4py import MPI  # noqa: E402

from pfwire.contact import PUNCH_BLOCK, ContactModel, build_punch_block, fix_surface  # noqa: E402
from pfwire.plasticity import J2Params  # noqa: E402

W = 0.5e-3
E_WIRE, NU_WIRE, SY = 69.0e9, 0.33, 167.0e6
E_CLAMP = 20.0 * E_WIRE
PRM = J2Params(E=E_WIRE, nu=NU_WIRE, sigma_y0=SY, H=0.0)


def _prandtl_factor() -> float:
    return (2.0 + math.pi) / math.sqrt(3.0)


def _build(tmp_path, *, plastic, sigma_y=SY, n=1):
    """粗网格平冲头压块体；plastic=True 时块体（体 1）用 J2。"""
    mesh, ct, ft = build_punch_block(
        tmp_path, half_width=W, block_half_width=3 * W, block_depth=4 * W, punch_height=W,
        gap=0.0, h_fine=W / 4, h_coarse=W, order=1, quads=True, comm=MPI.COMM_WORLD)
    prm = J2Params(E=E_WIRE, nu=NU_WIRE, sigma_y0=sigma_y, H=0.0)
    model = ContactModel(mesh, ct, ft, E=(E_CLAMP, E_WIRE), nu=(0.3, NU_WIRE),
                         order=1, tags=PUNCH_BLOCK, plastic={1: prm} if plastic else None)
    from dolfinx import default_scalar_type
    from dolfinx.fem import Constant, dirichletbc, locate_dofs_topological
    g_uy = Constant(mesh, default_scalar_type(0.0))
    g_ux = Constant(mesh, default_scalar_type(0.0))
    fdim = mesh.topology.dim - 1
    top = ft.find(PUNCH_BLOCK.load_facet)
    bcs = [dirichletbc(g_uy, locate_dofs_topological(model.V.sub(1), fdim, top), model.V.sub(1)),
           dirichletbc(g_ux, locate_dofs_topological(model.V.sub(0), fdim, top), model.V.sub(0)),
           fix_surface(model)]
    model.setup(bcs=bcs, traction=None, friction=None, gamma_factor=100.0, theta=1.0,
                newton_options={"relaxation_parameter": 1.0, "atol": 1e-5, "rtol": 1e-3,
                                "convergence_criterion": "residual", "max_it": 40,
                                "error_on_nonconvergence": True},
                petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                               "ksp_rtol": 1e-11, "ksp_atol": 1e-11})
    return model, g_uy, mesh


def test_prandtl_factor_value():
    assert _prandtl_factor() == pytest.approx(2.9685, abs=1e-4)


def test_elastic_limit_regression_matches_purely_elastic_model(tmp_path):
    """σ_y -> ∞ 时，带 plastic 参数的模型必须与纯弹性模型逐位一致（证明塑性路径在弹性区不扰动解）。"""
    m_pl, g_pl, mesh = _build(tmp_path / "a", plastic=True, sigma_y=1.0e30)
    m_el, g_el, _ = _build(tmp_path / "b", plastic=False)
    u_y = -6.0e-6
    for _ in range(3):
        g_pl.value = u_y
        g_el.value = u_y
        m_pl.step()
        m_el.step()
        m_pl.prepare_next()
        m_el.prepare_next()
    du = np.abs(m_pl.u.x.array - m_el.u.x.array).max()
    scale = np.abs(m_el.u.x.array).max()
    assert du < 1e-10 * scale, du
    assert m_pl.laws[1].p_n.x.array.max() == 0.0          # 没进塑性


def test_plastic_history_only_grows_in_plastic_body(tmp_path):
    """塑性历史只在塑性体的单元上非零：冲头（体 0）对应单元必须保持 0。"""
    model, g_uy, mesh = _build(tmp_path, plastic=True)
    for i in range(6):
        g_uy.value = -1.2e-5 * (i + 1) / 6
        model.step()
        if i < 5:
            model.prepare_next()
    p = model.laws[1].p_n
    dm = p.function_space.dofmap
    n_own = dm.index_map.size_local

    def dofs_of(tag):
        cells = model.cell_tags.find(tag)
        if cells.size == 0:
            return np.zeros(0, dtype=np.int32)
        d = np.unique(np.concatenate([dm.cell_dofs(int(c)) for c in cells]))
        return d[d < n_own]

    d_block, d_punch = dofs_of(model.tags.body_cells[1]), dofs_of(model.tags.body_cells[0])
    n_block = MPI.COMM_WORLD.allreduce(d_block.size, op=MPI.SUM)
    assert n_block > 0
    if d_block.size:
        assert p.x.array[d_block].max() > 0.0            # 块体进了塑性
    if d_punch.size:
        assert np.allclose(p.x.array[d_punch], 0.0)      # 冲头严格为 0


def test_penetration_is_finite_and_signs_are_sane(tmp_path):
    model, g_uy, mesh = _build(tmp_path, plastic=False)
    g_uy.value = 0.0
    model.step()                                          # 零载荷：间隙应处处 >= 0
    x, g = model.penetration()
    assert x.size > 0 and np.all(np.isfinite(g))
    assert g.max() < 1e-9                                 # 初始相切
    model.prepare_next()
    g_uy.value = -4.0e-6
    model.step()                                          # 压下：接触区内间隙转负（穿透）
    x, g = model.penetration()
    inside = np.abs(x) <= W
    assert inside.any() and g[inside].min() < 0.0
    assert abs(g[inside].min()) < 0.05 * W                # 穿透远小于接触宽度


def test_strip_prandtl_coarse_mesh_converges_and_approaches_limit(tmp_path):
    """Prandtl 条带算例的**很粗**网格版：只验证路径可跑、走向正确。
    准确值对网格很敏感（h/w = 0.5 时比值可到 1.3），所以这里只查单调性与量级。"""
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "verify_prandtl_strip.py"
    spec = importlib.util.spec_from_file_location("verify_prandtl_strip", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from pfwire.verify_contact_common import load_config
    cfg = load_config(pathlib.Path(mod.__file__).resolve().parents[1]
                      / "configs" / "verify_prandtl_strip.yaml")
    cfg["loading"]["steps"] = 6
    cfg["loading"]["u_y"] = -8.0e-5
    res = mod.run_level(cfg, 0.25, MPI.COMM_WORLD, lambda *_: None)   # 1152 单元
    ratios = [r["ratio"] for r in res["rows"]]
    assert all(r["n_it"] >= 1 for r in res["rows"])
    assert np.all(np.diff(ratios) > -0.02)               # 单调逼近
    assert 0.5 < ratios[-1] < 1.45                       # 量级合理（不进塑性或爆掉都会出界）
