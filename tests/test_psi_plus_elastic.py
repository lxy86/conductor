"""Phase 6.3.2：ψ⁺ 在弹性应变上的定义（pfwire.phasefield.psi_plus，PLAN §1.5，D-60）。

测试意图：
  1. 与 energy_split 的 ψ⁺ 一致（ε_p = 0 时定义不变）：随机应变、两种分解，逐点对拍；
  2. 弹性应变为零时 ψ⁺ = 0；
  3. 平面应变单轴 R = −1 的解析值（D-59/D-60）：spectral 下拉伸峰 ψ⁺ 与压缩峰 μ ε_yy² 各自成立，
     amor 下拉伸峰为全能量 (1−ν²)σ²/(2E)；
  4. 弹性步（ε_p 历史为零、应力低于屈服）时 J2 的 ε_e 等于 ε，即 ψ⁺(ε_e) 退化为 ψ⁺(ε)。
"""

import pathlib

import numpy as np
import pytest
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pfwire import phasefield as pf
from pfwire import plasticity as pl
from pfwire.material import load_material

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def material():
    return load_material(ROOT / "configs/material_al1350.yaml")


@pytest.fixture(scope="module")
def msh():
    return dmesh.create_unit_square(MPI.COMM_WORLD, 1, 1)


def _const_tensor(msh, eps):
    c = [[fem.Constant(msh, default_scalar_type(float(eps[i, j]))) for j in range(2)] for i in range(2)]
    return ufl.as_matrix(c)


def _mean(msh, expr):
    """常值场上的积分平均（单位面积的积分，并行安全）。"""
    comm = msh.comm
    area = comm.allreduce(fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ufl.dx)), op=MPI.SUM)
    return comm.allreduce(fem.assemble_scalar(fem.form(expr * ufl.dx)), op=MPI.SUM) / area


@pytest.mark.parametrize("split", ["amor", "spectral"])
def test_psi_plus_matches_energy_split(split, material, msh):
    lam, mu = material.elastic.lame
    rng = np.random.default_rng(20261013)
    for _ in range(6):
        a = rng.normal(size=(2, 2)) * 1e-3
        eps = 0.5 * (a + a.T)
        E_c = _const_tensor(msh, eps)
        exact = _mean(msh, pf.energy_split(E_c, material, split, 2)[0])
        ours = _mean(msh, pf.psi_plus(E_c, split, lam, mu))
        assert ours == pytest.approx(exact, rel=1e-13, abs=1e-30)


@pytest.mark.parametrize("split", ["amor", "spectral"])
def test_psi_plus_vanishes_at_zero_elastic_strain(split, material, msh):
    """零应变处 ψ⁺ = 0。spectral 分支的 sqrt 带 EIGENVALUE_REGULARIZATION（1e-30），
    给出 μ·1e-30 ≈ 3e-20 J/m³ 的底噪（energy_split 同样），容差取 1e-15 J/m³，远小于典型能量 1e4 量级。"""
    lam, mu = material.elastic.lame
    value = _mean(msh, pf.psi_plus(_const_tensor(msh, np.zeros((2, 2))), split, lam, mu))
    assert value == pytest.approx(0.0, abs=1e-15)


def test_uniaxial_plane_strain_closed_forms(material, msh):
    """σ_xx = σ（平面应变，ε_zz = 0）：ε_xx = (1−ν²)σ/E，ε_yy = −ν(1+ν)σ/E。"""
    lam, mu = material.elastic.lame
    E, nu = material.elastic.E, material.elastic.nu
    full = lambda s: 0.5 * s**2 / E  # σ²/(2E) 的单位

    # 拉伸 σ > 0：spectral 只计 ε_xx 与 tr 的拉伸部分；amor 中 tr > 0 故 ψ⁺ = 全能量
    s = 1.0e8
    eps_t = np.array([[(1 - nu**2) * s / E, 0.0], [0.0, -nu * (1 + nu) * s / E]])
    spec_t = _mean(msh, pf.psi_plus(_const_tensor(msh, eps_t), "spectral", lam, mu))
    amor_t = _mean(msh, pf.psi_plus(_const_tensor(msh, eps_t), "amor", lam, mu))
    assert spec_t == pytest.approx(full(s) * (nu * (1 + nu) * (1 - 2 * nu) + (1 - nu) ** 2 * (1 + nu)), rel=1e-10)
    assert amor_t == pytest.approx(full(s) * (1 - nu**2), rel=1e-10)

    # 压缩 σ < 0：spectral 中 ε_yy > 0 仍计入，ψ⁺ = μ ε_yy² = ν²(1+ν)/2 · σ²/E（D-60）；amor 为偏量能
    sc = -1.0e8
    eps_c = np.array([[(1 - nu**2) * sc / E, 0.0], [0.0, -nu * (1 + nu) * sc / E]])
    spec_c = _mean(msh, pf.psi_plus(_const_tensor(msh, eps_c), "spectral", lam, mu))
    assert spec_c == pytest.approx(mu * (nu * (1 + nu) * sc / E) ** 2, rel=1e-10)
    assert spec_c == pytest.approx(full(sc) * nu**2 * (1 + nu), rel=1e-10)


def test_elastic_step_has_zero_plastic_strain(material):
    """ε_p 历史为零的弹性步：ε_p = 0，故 ε_e = ε − ε_p = ε（ψ⁺ 的输入不变）。"""
    prm = pl.J2Params(E=material.elastic.E, nu=material.elastic.nu, sigma_y0=167.0e6, H=3.0e9)
    eps = np.diag([1.0e-4, -3.0e-5, 0.0])
    ret = pl.return_map(eps, np.zeros((3, 3)), 0.0, prm)
    assert not ret.plastic
    assert np.allclose(ret.eps_p, 0.0)
