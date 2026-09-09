"""AT2 耗散模型与平面应力开关（DECISIONS.md D-10 的选项 A）。

这两个开关**只为复现文献算例**（验证 1 的 Miehe 2010、验证 2 的 Carrara CT），
生产工况仍是 AT1 + 平面应变（PLAN 第 1/2 节）。

测试意图
--------
1. c_w 约定与耗散前因子：AT1 -> 3G_c/8（PLAN 第 2 节）、AT2 -> G_c/2
   （Miehe/Bourdin 经典写法 G_c[d²/(2ℓ) + (ℓ/2)|∇d|²]），
   在均匀 d 场上积分核对，防止把两套约定搞混；
2. **AT2 的均匀响应闭式解**：d(ε) = ε²/(ε²+k)，k = G_c/(ℓE)；
   峰值应力 σ_y = (9/16)√(EG_c/(3ℓ)) 出现在 ε_y = √(k/3) 处、此时 d = 1/4。
   这条同时验证 AT2 的 w(d)=d² 与前因子 G_c/2；
3. AT2 **没有弹性阈值**：任意 ε>0 都有 d>0（与 AT1 的 d≡0 直到 σ_c 形成对照）；
4. 平面应力：λ* = Eν/(1−ν²)，单轴时 σ_xx = Eε_xx 且横向自由收缩 ε_yy = −νε_xx；
5. AT1/AT2 各自的 (G_c, ℓ, σ_c) 三角在三种 mode 下互为逆；
6. 模型与材料卡的 dissipation 不一致必须报错（σ_c 的换算关系不同，混用就是参数错配）；
7. Miehe 2010 的 λ = 121.15、μ = 80.77 kN/mm² 必须由 E=210 GPa、ν=0.3 的
   **平面应变**关系精确还原 —— 既是换算的回归锚点，也确认了该基准是平面应变。
"""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pfwire import material as mat
from pfwire import phasefield as pf

# Miehe et al. 2010 (CMAME 199:2765-2778) 第 5.1 节 SENT 基准的材料参数
MIEHE_E, MIEHE_NU = 210.0e9, 0.3
MIEHE_GC = 2700.0  # J/m^2 = 2.7e-3 kN/mm
MIEHE_ELL = 7.5e-6  # m = 0.0075 mm


def make_material(
    *,
    dissipation: str,
    Gc: float,
    ell: float,
    E: float = MIEHE_E,
    nu: float = 0.0,
    plane_state: str = "plane_strain",
) -> mat.Material:
    fracture = mat.resolve_fracture_properties(
        E, nu, mode="from_gc_and_ell", Gc=Gc, ell=ell, dissipation=dissipation
    )
    return mat.Material(
        name="bench-%s" % dissipation,
        elastic=mat.ElasticProperties(E=E, nu=nu, plane_state=plane_state),
        fracture=fracture,
        sigma_u=fracture.sigma_c,
        alpha_T=None,
        sources={},
    )


# --------------------------------------------------------------------------
# 1. 前因子
# --------------------------------------------------------------------------
def test_cw_convention():
    assert mat.dissipation_normalization("AT1") == pytest.approx(8.0 / 3.0)
    assert mat.dissipation_normalization("AT2") == pytest.approx(2.0)
    with pytest.raises(ValueError, match="dissipation"):
        mat.dissipation_normalization("AT3")


@pytest.mark.parametrize(
    "model,expected_prefactor", [("AT1", 3.0 / 8.0), ("AT2", 1.0 / 2.0)]
)
def test_dissipation_prefactor_on_uniform_field(model: str, expected_prefactor: float):
    """均匀 d 场（∇d = 0）下耗散 = 前因子·G_c·w(d)/ℓ·面积，可逐项手算。"""
    material = make_material(dissipation=model, Gc=MIEHE_GC, ell=MIEHE_ELL)
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = fem.functionspace(msh, ("Lagrange", 1))
    d = fem.Function(V)
    d_value = 0.4
    d.x.array[:] = d_value
    d.x.scatter_forward()
    one = fem.Constant(msh, default_scalar_type(1.0))
    total = msh.comm.allreduce(
        fem.assemble_scalar(
            fem.form(pf.dissipation_density(d, material, one, model) * ufl.dx)
        ),
        op=MPI.SUM,
    )
    w = d_value if model == "AT1" else d_value**2
    expected = expected_prefactor * MIEHE_GC * w / MIEHE_ELL  # 单位面积
    assert total == pytest.approx(expected, rel=1e-12)


def test_dissipation_density_rejects_unknown_model():
    material = make_material(dissipation="AT1", Gc=MIEHE_GC, ell=MIEHE_ELL)
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    d = fem.Function(fem.functionspace(msh, ("Lagrange", 1)))
    with pytest.raises(ValueError, match="dissipation"):
        pf.dissipation_density(d, material, 1.0, "AT3")


# --------------------------------------------------------------------------
# 2-3. AT2 的均匀响应闭式解
# --------------------------------------------------------------------------
class Uniaxial:
    """nu = 0 的方块，约束 eps_yy = 0，施加均匀 eps_xx。amor 分解下 psi+ = ½Eε²。"""

    def __init__(self, material: mat.Material, dissipation: str, n: int = 4):
        self.material = material
        self.L = 10.0 * material.fracture.ell
        self.msh = dmesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([0.0, 0.0]), np.array([self.L, self.L])],
            [n, n],
            cell_type=dmesh.CellType.triangle,
        )
        model = pf.PhaseFieldModel(material, split="amor", dissipation=dissipation)
        self.u_applied = fem.Constant(self.msh, default_scalar_type(0.0))

        def make_bcs(V):
            fdim = self.msh.topology.dim - 1
            left = dmesh.locate_entities_boundary(
                self.msh, fdim, lambda x: np.isclose(x[0], 0.0)
            )
            right = dmesh.locate_entities_boundary(
                self.msh, fdim, lambda x: np.isclose(x[0], self.L)
            )
            horiz = dmesh.locate_entities_boundary(
                self.msh,
                fdim,
                lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], self.L),
            )
            Vx, Vy = V.sub(0), V.sub(1)
            zero = fem.Constant(self.msh, default_scalar_type(0.0))
            return [
                fem.dirichletbc(zero, fem.locate_dofs_topological(Vx, fdim, left), Vx),
                fem.dirichletbc(
                    self.u_applied, fem.locate_dofs_topological(Vx, fdim, right), Vx
                ),
                fem.dirichletbc(
                    zero, fem.locate_dofs_topological(Vy, fdim, horiz), Vy
                ),
            ]

        self.solver = pf.PhaseFieldSolver(self.msh, model, bcs_u=make_bcs)

    def apply_strain(self, eps: float):
        self.u_applied.value = eps * self.L
        return self.solver.solve_step()


@pytest.mark.parametrize("factor", [1.0 / 3.0, 1.0, 3.0, 9.0])
def test_at2_homogeneous_damage_closed_form(factor: float):
    """d(ε) = ε²/(ε²+k)，k = G_c/(ℓE)。factor = ε²/k。"""
    material = make_material(dissipation="AT2", Gc=MIEHE_GC, ell=MIEHE_ELL)
    k = MIEHE_GC / (MIEHE_ELL * MIEHE_E)
    eps = np.sqrt(factor * k)
    case = Uniaxial(material, "AT2")
    result = case.apply_strain(eps)
    assert result.converged, result.failure_mode
    expected = factor / (factor + 1.0)
    assert case.solver.max_damage() == pytest.approx(expected, rel=2e-3)
    assert case.solver.min_damage() == pytest.approx(expected, rel=2e-3)


def test_at2_peak_stress_matches_closed_form():
    """峰值 σ_y = (9/16)√(EG_c/(3ℓ)) 出现在 ε_y = √(k/3)，此时 d = 1/4。"""
    material = make_material(dissipation="AT2", Gc=MIEHE_GC, ell=MIEHE_ELL)
    k = MIEHE_GC / (MIEHE_ELL * MIEHE_E)
    eps_y = np.sqrt(k / 3.0)
    sigma_y = mat.at2_sigma_c_from_gc_ell(MIEHE_E, MIEHE_GC, MIEHE_ELL)
    # 材料卡在 mode=from_gc_and_ell + AT2 下导出的 sigma_c 就是这个 sigma_y
    assert material.fracture.sigma_c == pytest.approx(sigma_y, rel=1e-12)

    case = Uniaxial(material, "AT2")
    peak, peak_eps = -np.inf, None
    for eps in eps_y * np.linspace(0.4, 2.5, 22):
        result = case.apply_strain(float(eps))
        assert result.converged, result.failure_mode
        d = case.solver.max_damage()
        sigma = pf.degradation(d, case.solver.model.k_res) * MIEHE_E * eps
        if sigma > peak:
            peak, peak_eps = sigma, float(eps)
    assert peak == pytest.approx(sigma_y, rel=5e-3)
    assert peak_eps == pytest.approx(eps_y, rel=0.1)
    # 峰值处 d = 1/4（闭式）
    case2 = Uniaxial(material, "AT2")
    case2.apply_strain(float(eps_y))
    assert case2.solver.max_damage() == pytest.approx(0.25, rel=2e-3)


def test_at2_has_no_elastic_threshold_unlike_at1():
    """AT2 在任意 ε>0 都有 d>0；AT1 在 σ<σ_c 时 d 严格为 0。"""
    k = MIEHE_GC / (MIEHE_ELL * MIEHE_E)
    small_eps = 0.05 * np.sqrt(k / 3.0)  # 远低于 AT2 的峰值应变

    at2 = Uniaxial(make_material(dissipation="AT2", Gc=MIEHE_GC, ell=MIEHE_ELL), "AT2")
    assert at2.apply_strain(small_eps).converged
    assert at2.solver.max_damage() > 0.0
    assert at2.solver.max_damage() == pytest.approx(
        small_eps**2 / (small_eps**2 + k), rel=1e-2
    )

    at1_mat = make_material(dissipation="AT1", Gc=MIEHE_GC, ell=MIEHE_ELL)
    at1 = Uniaxial(at1_mat, "AT1")
    eps_c = at1_mat.fracture.sigma_c / MIEHE_E
    assert at1.apply_strain(0.9 * eps_c).converged
    assert at1.solver.max_damage() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# 4. 平面应力
# --------------------------------------------------------------------------
def test_plane_stress_lame_and_uniaxial_response():
    """λ* = Eν/(1−ν²)；单轴（横向自由）时 σ_xx = Eε_xx、ε_yy = −νε_xx。"""
    E, nu = 70.0e9, 0.33
    lmbda_ps, mu = mat.lame_parameters(E, nu, "plane_stress")
    assert lmbda_ps == pytest.approx(E * nu / (1.0 - nu**2), rel=1e-14)
    assert mu == pytest.approx(E / (2.0 * (1.0 + nu)), rel=1e-14)
    # 二维体积模量随之变化：K_2D = λ* + μ = E/(2(1−ν))
    assert mat.bulk_modulus(E, nu, 2, "plane_stress") == pytest.approx(
        E / (2.0 * (1.0 - nu)), rel=1e-14
    )
    with pytest.raises(ValueError, match="plane_state"):
        mat.lame_parameters(E, nu, "axisymmetric")

    # 有限元核对：横向自由的单轴拉伸
    material = make_material(
        dissipation="AT1",
        Gc=MIEHE_GC,
        ell=MIEHE_ELL,
        E=E,
        nu=nu,
        plane_state="plane_stress",
    )
    L = 10.0 * MIEHE_ELL
    msh = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([L, L])],
        [6, 6],
        cell_type=dmesh.CellType.triangle,
    )
    eps_applied = 1.0e-4
    model = pf.PhaseFieldModel(material, split="amor")

    def make_bcs(V):
        fdim = msh.topology.dim - 1
        left = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
        right = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], L))
        bottom = dmesh.locate_entities_boundary(
            msh, fdim, lambda x: np.isclose(x[1], 0.0)
        )
        Vx, Vy = V.sub(0), V.sub(1)
        zero = fem.Constant(msh, default_scalar_type(0.0))
        applied = fem.Constant(msh, default_scalar_type(eps_applied * L))
        return [
            fem.dirichletbc(zero, fem.locate_dofs_topological(Vx, fdim, left), Vx),
            fem.dirichletbc(applied, fem.locate_dofs_topological(Vx, fdim, right), Vx),
            fem.dirichletbc(zero, fem.locate_dofs_topological(Vy, fdim, bottom), Vy),
        ]

    solver = pf.PhaseFieldSolver(msh, model, bcs_u=make_bcs)
    solver.solve_displacement()

    eps = pf.strain(solver.u)
    lmbda_eff, mu_eff = material.elastic.lame
    sigma = 2.0 * mu_eff * eps + lmbda_eff * ufl.tr(eps) * ufl.Identity(2)
    area = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ufl.dx)), op=MPI.SUM
    )

    def mean(expr):
        return (
            msh.comm.allreduce(
                fem.assemble_scalar(fem.form(expr * ufl.dx)), op=MPI.SUM
            )
            / area
        )

    assert mean(eps[0, 0]) == pytest.approx(eps_applied, rel=1e-9)
    # 容差取 1e-5 而不是更紧：残余刚度 k_res = 1e-6 只乘在 psi+ 上、不乘 psi-，
    # 把 lambda/mu 的配比扰动了 O(k_res)，实测横向应变偏差正是 1.4e-6 相对量。
    assert mean(eps[1, 1]) == pytest.approx(-nu * eps_applied, rel=1e-5)
    assert mean(sigma[0, 0]) == pytest.approx(E * eps_applied, rel=1e-5)
    assert mean(sigma[1, 1]) == pytest.approx(0.0, abs=1e-5 * E * eps_applied)


# --------------------------------------------------------------------------
# 5-7. 参数换算与一致性
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dissipation", mat.DISSIPATION_MODELS)
def test_fracture_triangle_round_trip_both_models(dissipation: str):
    E, nu = MIEHE_E, 0.3
    sigma_c = mat.sigma_c_from_gc_ell(E, MIEHE_GC, MIEHE_ELL, dissipation)
    assert mat.gc_from_sigma_c_ell(E, sigma_c, MIEHE_ELL, dissipation) == pytest.approx(
        MIEHE_GC, rel=1e-12
    )
    assert mat.ell_from_gc_sigma_c(E, MIEHE_GC, sigma_c, dissipation) == pytest.approx(
        MIEHE_ELL, rel=1e-12
    )
    triples = []
    for mode in mat.FRACTURE_MODES:
        fp = mat.resolve_fracture_properties(
            E,
            nu,
            mode=mode,
            Gc=MIEHE_GC,
            ell=MIEHE_ELL,
            sigma_c=sigma_c,
            dissipation=dissipation,
        )
        assert fp.dissipation == dissipation
        triples.append((fp.Gc, fp.ell, fp.sigma_c))
    for other in triples[1:]:
        for got, ref in zip(other, triples[0]):
            assert got == pytest.approx(ref, rel=1e-10)


def test_at1_and_at2_sigma_c_differ():
    """两套换算给出的 sigma_c 不同，这正是不许混用的原因。"""
    s1 = mat.sigma_c_from_gc_ell(MIEHE_E, MIEHE_GC, MIEHE_ELL, "AT1")
    s2 = mat.sigma_c_from_gc_ell(MIEHE_E, MIEHE_GC, MIEHE_ELL, "AT2")
    assert s1 / s2 > 1.5


def test_model_material_dissipation_mismatch_is_rejected():
    material = make_material(dissipation="AT1", Gc=MIEHE_GC, ell=MIEHE_ELL)
    with pytest.raises(ValueError, match="dissipation"):
        pf.PhaseFieldModel(material, dissipation="AT2")
    with pytest.raises(ValueError, match="dissipation"):
        pf.PhaseFieldModel(material, dissipation="AT3")
    # 一致时正常
    assert pf.PhaseFieldModel(material, dissipation="AT1").dissipation == "AT1"


def test_miehe_lame_constants_reproduced():
    """Miehe 2010 第 5.1 节给的 λ = 121.15、μ = 80.77 kN/mm² 必须由
    E = 210 GPa、ν = 0.3 的**平面应变**关系还原（kN/mm² = GPa）。

    这既是换算的回归锚点，也确认了该 SENT 基准用的是平面应变
    （他们直接给三维 Lamé 常数）。
    """
    lmbda, mu = mat.lame_parameters(MIEHE_E, MIEHE_NU)  # 默认 plane_strain
    assert lmbda / 1.0e9 == pytest.approx(121.15, abs=0.01)
    assert mu / 1.0e9 == pytest.approx(80.77, abs=0.01)
    # 平面应力会给出完全不同的 λ，不能与文献值对上
    lmbda_ps, _ = mat.lame_parameters(MIEHE_E, MIEHE_NU, "plane_stress")
    assert abs(lmbda_ps / 1.0e9 - 121.15) > 50.0


def test_material_yaml_accepts_plane_state_and_dissipation(tmp_path):
    y = tmp_path / "m.yaml"
    y.write_text(
        "name: t\n"
        "elastic: {E: 210e9, nu: 0.3, plane_state: plane_stress}\n"
        "strength: {sigma_u: 1e9}\n"
        "fracture: {mode: from_gc_and_ell, Gc: 2700.0, ell: 7.5e-6, dissipation: AT2}\n"
        "fatigue: {alpha_T: null}\n",
        encoding="utf-8",
    )
    m = mat.load_material(y)
    assert m.elastic.plane_state == "plane_stress"
    assert m.fracture.dissipation == "AT2"
    assert m.fracture.sigma_c == pytest.approx(
        mat.at2_sigma_c_from_gc_ell(210.0e9, 2700.0, 7.5e-6), rel=1e-12
    )
    # 拼错的键必须报错
    y.write_text(
        "name: t\n"
        "elastic: {E: 210e9, nu: 0.3, plain_state: plane_stress}\n"
        "strength: {sigma_u: 1e9}\n"
        "fracture: {mode: from_gc_and_ell, Gc: 2700.0, ell: 7.5e-6}\n"
        "fatigue: {alpha_T: null}\n",
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="elastic"):
        mat.load_material(y)
