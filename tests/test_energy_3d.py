"""三维嵌入的能量分裂与退化应力（D-65，pfwire.phasefield.energy_split_3d / stress_split_3d）。

测试意图：
  1. ε_zz = 0 时，energy_split_3d 与二维 energy_split(…, dim=2) 的 spectral 逐点一致（rel 1e-10）：
     三维谱分解在无 ε_zz 时退化为已有的二维式，无塑性路径的能量不因此改变；
  2. ψ⁺ + ψ⁻ = ½λ tr₃²ε + μ ε:ε（三维总能量，逐点，含 ε_zz ≠ 0）：ε_zz 不被遗漏；
  3. stress_split_3d：纯拉伸特征值全正时 σ = g(d)·C:ε，纯压缩时 σ = C:ε（分裂的两个极限）；
  4. amor 的三维形式被拒绝（NotImplementedError），不静默给出错误结果。
"""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pfwire.material import load_material
from pfwire.phasefield import energy_split, energy_split_3d, stress_split_3d

MATERIAL = "configs/material_al1350.yaml"
K_RES = 1.0e-6


def _eval(msh, expr, n_comp, degree=2):
    """在 DG-degree 节点上逐点求值（UFL 表达式 -> (n_points, n_comp)）。"""
    W = fem.functionspace(msh, ("DG", degree, (n_comp,)) if n_comp > 1 else ("DG", degree))
    f = fem.Function(W)
    f.interpolate(fem.Expression(expr, W.element.interpolation_points))
    return f.x.array.reshape(-1, n_comp)


def _flat(t):
    """3×3 张量 -> 9 分量向量，便于用 _eval 逐点取值。"""
    return ufl.as_vector([t[i, j] for i in range(3) for j in range(3)])


def _strain_fields(msh, a, b, c):
    """u = (a x, b y)：ε_2D = diag(a, b)（常数）；ε_zz 取常数 c（塑性嵌入的 ε_e,zz）。"""
    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    u = fem.Function(V)
    u.interpolate(lambda x: np.vstack([a * x[0], b * x[1]]))
    eps2 = ufl.sym(ufl.grad(u))
    cz = fem.Constant(msh, float(c))
    eps3 = ufl.as_tensor(
        [[eps2[0, 0], eps2[0, 1], 0.0], [eps2[1, 0], eps2[1, 1], 0.0], [0.0, 0.0, cz]]
    )
    return u, eps2, eps3


def _mesh():
    return dmesh.create_unit_square(MPI.COMM_WORLD, 2, 2, dmesh.CellType.triangle)


def test_energy_split_3d_reduces_to_2d_when_ezz_is_zero():
    material = load_material(MATERIAL)
    msh = _mesh()
    rng = np.random.default_rng(3)
    # 用一般的（非对角）面内应变：在 P1 上给 u 随机值
    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    u = fem.Function(V)
    u.x.array[:] = rng.normal(size=u.x.array.size) * 2e-3
    eps2 = ufl.sym(ufl.grad(u))
    eps3 = ufl.as_tensor(
        [[eps2[0, 0], eps2[0, 1], 0.0], [eps2[1, 0], eps2[1, 1], 0.0], [0.0, 0.0, 0.0]]
    )
    p3, m3 = energy_split_3d(eps3, material, "spectral")
    p2, m2 = energy_split(eps2, material, "spectral", 2)
    scale = material.elastic.E * 1.0e-6
    assert np.allclose(_eval(msh, p3, 1)[:, 0], _eval(msh, p2, 1)[:, 0], rtol=1e-10, atol=1e-10 * scale)
    assert np.allclose(_eval(msh, m3, 1)[:, 0], _eval(msh, m2, 1)[:, 0], rtol=1e-10, atol=1e-10 * scale)


def test_energy_split_3d_total_is_full_3d_energy_with_ezz():
    material = load_material(MATERIAL)
    lam, mu = material.elastic.lame
    msh = _mesh()
    rng = np.random.default_rng(5)
    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    u = fem.Function(V)
    u.x.array[:] = rng.normal(size=u.x.array.size) * 2e-3
    eps2 = ufl.sym(ufl.grad(u))
    cz = fem.Constant(msh, 3.0e-4)
    eps3 = ufl.as_tensor(
        [[eps2[0, 0], eps2[0, 1], 0.0], [eps2[1, 0], eps2[1, 1], 0.0], [0.0, 0.0, cz]]
    )
    p3, m3 = energy_split_3d(eps3, material, "spectral")
    total = _eval(msh, p3 + m3, 1)[:, 0]
    tr3 = ufl.tr(eps3)
    expected = _eval(msh, 0.5 * lam * tr3**2 + mu * ufl.inner(eps3, eps3), 1)[:, 0]
    assert np.allclose(total, expected, rtol=1e-10, atol=1e-10 * material.elastic.E * 1e-6)


@pytest.mark.parametrize("sign", [+1.0, -1.0])
def test_stress_split_3d_limits(sign):
    """纯拉伸（全特征值 > 0）：σ = g C:ε；纯压缩（全特征值 < 0）：σ = C:ε，不退化。"""
    material = load_material(MATERIAL)
    lam, mu = material.elastic.lame
    msh = _mesh()
    a, b, c = sign * 2.0e-3, sign * 1.0e-3, sign * 5.0e-4
    _, _, eps3 = _strain_fields(msh, a, b, c)
    d_val = 0.3
    d = fem.Constant(msh, d_val)
    sig = stress_split_3d(eps3, material, "spectral", d, K_RES)
    g = (1.0 - d_val) ** 2 + K_RES
    if sign > 0:
        factor = g
    else:
        factor = 1.0
    C_eps = lam * ufl.tr(eps3) * ufl.Identity(3) + 2.0 * mu * eps3
    got = _eval(msh, _flat(sig), 9)
    want = _eval(msh, _flat(factor * C_eps), 9)
    assert np.allclose(got, want, rtol=1e-10, atol=1e-10 * material.elastic.E * 1e-3)


def test_amor_three_dimensional_form_is_refused():
    material = load_material(MATERIAL)
    msh = _mesh()
    _, _, eps3 = _strain_fields(msh, 1.0e-3, 0.0, 0.0)
    with pytest.raises(NotImplementedError):
        energy_split_3d(eps3, material, "amor")
