"""material.py 的参数换算单测（PLAN.md Phase 0）。

测试意图：
1. 每个换算式与独立推导的闭式表达对拍（不是把实现抄一遍）；
2. 三角关系 (G_c, ell, sigma_c) 在三种 mode 下互为逆运算、且过定不自洽必须报错；
3. 极限情形有意义（刚性对偶体、nu -> 0）；
4. configs/material_al1350.yaml 能载入且导出量落在预期数量级（回归锚点）。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pfwire import material as mat

REPO = Path(__file__).resolve().parents[1]
MATERIAL_YAML = REPO / "configs" / "material_al1350.yaml"

E_AL = 69.0e9
NU_AL = 0.33


# --------------------------------------------------------------------------
# 弹性常数
# --------------------------------------------------------------------------
def test_lame_matches_closed_form():
    lmbda, mu = mat.lame_parameters(E_AL, NU_AL)
    assert mu == pytest.approx(E_AL / (2.0 * (1.0 + NU_AL)), rel=1e-14)
    assert lmbda == pytest.approx(
        E_AL * NU_AL / ((1.0 + NU_AL) * (1.0 - 2.0 * NU_AL)), rel=1e-14
    )


def test_lame_reproduces_uniaxial_stress_state():
    """用 Lamé 常数做单轴应力状态反演，应还原 E 与 nu。

    单轴应力 (sigma_11 = s, 其余为 0) 下 eps_11 = s/E, eps_22 = eps_33 = -nu s/E；
    代回 sigma = 2 mu eps + lambda tr(eps) I 必须还原 (s, 0, 0)。
    """
    lmbda, mu = mat.lame_parameters(E_AL, NU_AL)
    s = 1.0
    e11 = s / E_AL
    e22 = e33 = -NU_AL * s / E_AL
    tr = e11 + e22 + e33
    assert 2.0 * mu * e11 + lmbda * tr == pytest.approx(s, rel=1e-12)
    assert 2.0 * mu * e22 + lmbda * tr == pytest.approx(0.0, abs=1e-12 * s)


def test_plane_strain_modulus_and_bulk_identity():
    assert mat.plane_strain_modulus(E_AL, NU_AL) == pytest.approx(
        E_AL / (1.0 - NU_AL**2), rel=1e-14
    )
    # K_2D = lambda + mu = E/(2(1+nu)(1-2nu))，独立推导
    assert mat.bulk_modulus(E_AL, NU_AL, dim=2) == pytest.approx(
        E_AL / (2.0 * (1.0 + NU_AL) * (1.0 - 2.0 * NU_AL)), rel=1e-14
    )
    # K_3D = lambda + 2mu/3 = E/(3(1-2nu))
    assert mat.bulk_modulus(E_AL, NU_AL, dim=3) == pytest.approx(
        E_AL / (3.0 * (1.0 - 2.0 * NU_AL)), rel=1e-14
    )


def test_bad_elastic_inputs_raise():
    with pytest.raises(ValueError):
        mat.lame_parameters(-1.0, 0.3)
    with pytest.raises(ValueError):
        mat.lame_parameters(E_AL, 0.5)
    with pytest.raises(ValueError):
        mat.bulk_modulus(E_AL, NU_AL, dim=1)


# --------------------------------------------------------------------------
# 接触参数
# --------------------------------------------------------------------------
def test_effective_modulus_identical_pair_and_rigid_limit():
    e_star_pair = mat.contact_effective_modulus(E_AL, NU_AL, E_AL, NU_AL)
    assert e_star_pair == pytest.approx(E_AL / (2.0 * (1.0 - NU_AL**2)), rel=1e-14)

    e_star_rigid = mat.contact_effective_modulus(E_AL, NU_AL)
    assert e_star_rigid == pytest.approx(E_AL / (1.0 - NU_AL**2), rel=1e-14)
    # 刚性极限：E2 -> inf 应连续趋近显式的刚性分支
    e_star_stiff = mat.contact_effective_modulus(E_AL, NU_AL, 1.0e30, 0.3)
    assert e_star_stiff == pytest.approx(e_star_rigid, rel=1e-12)
    assert e_star_pair == pytest.approx(0.5 * e_star_rigid, rel=1e-14)


def test_effective_radius_and_pair_properties():
    R_wire = 1.6e-3
    assert mat.contact_effective_radius(R_wire, "aluminum") == pytest.approx(0.8e-3)
    assert mat.contact_effective_radius(R_wire, "rigid") == pytest.approx(R_wire)
    with pytest.raises(ValueError):
        mat.contact_effective_radius(R_wire, "steel")

    # a = 2 R p0/E*：铝-铝时 R 减半、E* 也减半，故两种对偶给出同一个 a。
    # 这是 DECISIONS.md D-1 尺度论证的关键（该恒等式本体在 Phase 2.2 loads.py 实现）。
    p0 = 500.0e6
    a_pair = None
    for counterpart in ("aluminum", "rigid"):
        e_star, r_eff = mat.contact_pair_properties(E_AL, NU_AL, R_wire, counterpart)
        a = 2.0 * r_eff * p0 / e_star
        if a_pair is None:
            a_pair = a
        else:
            assert a == pytest.approx(a_pair, rel=1e-12)
    assert a_pair == pytest.approx(20.7e-6, rel=2e-2)  # ≈20.7 um，见 D-1


def test_effective_modulus_requires_nu2_when_e2_given():
    with pytest.raises(ValueError):
        mat.contact_effective_modulus(E_AL, NU_AL, E_AL, None)


# --------------------------------------------------------------------------
# Irwin 与 AT1
# --------------------------------------------------------------------------
def test_irwin_round_trip():
    K_IC = 25.0e6  # Pa*m^0.5，仅作换算测试用，非 Al 1350 的取值
    Gc = mat.gc_from_kic(K_IC, E_AL, NU_AL)
    assert Gc == pytest.approx(K_IC**2 * (1.0 - NU_AL**2) / E_AL, rel=1e-14)
    assert mat.kic_from_gc(Gc, E_AL, NU_AL) == pytest.approx(K_IC, rel=1e-12)


def test_at1_triangle_is_mutually_consistent():
    sigma_c, ell = 170.0e6, 4.0e-6
    Gc = mat.at1_gc_from_sigma_c_ell(E_AL, sigma_c, ell)
    assert Gc == pytest.approx(8.0 * sigma_c**2 * ell / (3.0 * E_AL), rel=1e-14)
    assert mat.at1_ell_from_gc_sigma_c(E_AL, Gc, sigma_c) == pytest.approx(
        ell, rel=1e-12
    )
    assert mat.at1_sigma_c_from_gc_ell(E_AL, Gc, ell) == pytest.approx(
        sigma_c, rel=1e-12
    )


def test_at1_numeric_anchor():
    """独立手算锚点：sigma_c=170 MPa, ell=4 um, E=69 GPa -> G_c ≈ 4.468 J/m^2。"""
    Gc = mat.at1_gc_from_sigma_c_ell(E_AL, 170.0e6, 4.0e-6)
    assert Gc == pytest.approx(4.468, rel=1e-3)
    assert mat.kic_from_gc(Gc, E_AL, NU_AL) == pytest.approx(0.588e6, rel=1e-2)


# --------------------------------------------------------------------------
# 三角关系求解器
# --------------------------------------------------------------------------
def test_resolve_all_modes_agree_on_consistent_triple():
    sigma_c, ell = 170.0e6, 4.0e-6
    Gc = mat.at1_gc_from_sigma_c_ell(E_AL, sigma_c, ell)
    triples = []
    for mode in mat.FRACTURE_MODES:
        fp = mat.resolve_fracture_properties(
            E_AL, NU_AL, mode=mode, Gc=Gc, ell=ell, sigma_c=sigma_c
        )
        triples.append((fp.Gc, fp.ell, fp.sigma_c))
        assert fp.derived in ("Gc", "ell", "sigma_c")
    for other in triples[1:]:
        for got, ref in zip(other, triples[0]):
            assert got == pytest.approx(ref, rel=1e-10)


def test_resolve_derives_missing_value():
    fp = mat.resolve_fracture_properties(
        E_AL, NU_AL, mode="from_sigma_c_and_ell", sigma_c=170.0e6, ell=4.0e-6
    )
    assert fp.derived == "Gc"
    assert fp.Gc == pytest.approx(4.468, rel=1e-3)
    assert fp.kic_equivalent == pytest.approx(
        mat.kic_from_gc(fp.Gc, E_AL, NU_AL), rel=1e-14
    )


def test_resolve_rejects_inconsistent_overspecification():
    with pytest.raises(ValueError, match="过定"):
        mat.resolve_fracture_properties(
            E_AL,
            NU_AL,
            mode="from_sigma_c_and_ell",
            sigma_c=170.0e6,
            ell=4.0e-6,
            Gc=100.0,  # 与 AT1 导出的 4.468 相差两个量级
        )


def test_resolve_rejects_missing_inputs_and_bad_mode():
    with pytest.raises(ValueError, match="缺少"):
        mat.resolve_fracture_properties(
            E_AL, NU_AL, mode="from_gc_and_ell", ell=4.0e-6
        )
    with pytest.raises(ValueError, match="mode"):
        mat.resolve_fracture_properties(E_AL, NU_AL, mode="magic", Gc=1.0, ell=1.0)


# --------------------------------------------------------------------------
# yaml 载入
# --------------------------------------------------------------------------
def test_load_material_al1350():
    m = mat.load_material(MATERIAL_YAML)
    assert m.name == "Al 1350-H19"
    assert m.elastic.E == pytest.approx(69.0e9)
    assert m.elastic.nu == pytest.approx(0.33)
    assert m.sigma_u == pytest.approx(185.0e6)
    # yaml 选的是 from_sigma_c_and_ell，故 G_c 是导出量
    assert m.fracture.derived == "Gc"
    # ell 于 2026-09-11 由 4 µm 改为 2 µm（D-28：p0 降到 240 MPa 后，
    # D-1 的 a >= 5 ell 与 D-27 的静态门槛在 ell = 4 µm 下无解）
    assert m.fracture.ell == pytest.approx(2.0e-6)
    assert m.fracture.sigma_c == pytest.approx(185.0e6)
    # 手算锚点：Gc = 8 sigma_c^2 ell/(3E) = 8*(185e6)^2*2e-6/(3*69e9) = 2.645 J/m^2
    assert m.fracture.Gc == pytest.approx(2.645, rel=1e-3)
    assert m.fracture.kic_equivalent == pytest.approx(0.4526e6, rel=1e-2)
    # AT1 关系必须在载入后仍严格成立
    assert m.fracture.sigma_c == pytest.approx(
        mat.at1_sigma_c_from_gc_ell(m.elastic.E, m.fracture.Gc, m.fracture.ell),
        rel=1e-12,
    )
    assert all(math.isfinite(v) for v in (m.fracture.Gc, m.fracture.ell))
    # 关键参数必须有出处记录（硬约束 2/3）
    for key in ("E", "nu", "sigma_u", "sigma_c", "ell", "Gc", "alpha_T"):
        assert key in m.sources and m.sources[key]


def test_alpha_T_uncalibrated_raises_instead_of_defaulting(tmp_path):
    """未标定的 alpha_T 必须报错，不得静默取默认值（D-3）。

    这里用**临时材料卡**而不是生产卡：生产卡的 alpha_T 在 Phase 2.3 标定后
    已经有值了，若继续断言它是 None，这条守卫就会随标定进度一起失效 ——
    而它要守的是「未标定时不许蒙一个默认值」这个行为本身，与标定进度无关。
    """
    y = tmp_path / "uncalibrated.yaml"
    y.write_text(
        "name: t\n"
        "elastic: {E: 69.0e+9, nu: 0.33}\n"
        "strength: {sigma_u: 185.0e+6}\n"
        "fracture: {mode: from_sigma_c_and_ell, sigma_c: 185.0e+6, ell: 4.0e-6}\n"
        "fatigue: {alpha_T: null}\n",
        encoding="utf-8",
    )
    m = mat.load_material(y)
    assert m.alpha_T is None
    with pytest.raises(ValueError, match="alpha_T"):
        m.require_alpha_T()


def test_production_material_is_calibrated():
    """生产材料卡在 Phase 2.3 之后必须带标定值，且能被 require_* 取出。"""
    m = mat.load_material(MATERIAL_YAML)
    alpha_T, form, kappa = m.require_fatigue_params()
    assert alpha_T > 0.0
    assert form == "logarithmic"
    assert kappa is not None and kappa > 0.0


def test_load_material_tolerates_yaml11_unsigned_exponent(tmp_path):
    """PLAN.md 第 5 节的写法（500e6 / 1.0e8，指数不带符号）必须能安全载入。

    PyYAML 走 YAML 1.1：指数不带显式符号时会被解析成 **字符串**。
    若不 coerce，E 会以 str 存进数据类，后续算式静默出错或崩在很远的地方。
    """
    y = tmp_path / "m.yaml"
    y.write_text(
        "name: unsigned-exponent\n"
        "elastic: {E: 69e9, nu: 0.33}\n"
        "strength: {sigma_u: 170e6}\n"
        "fracture: {mode: from_sigma_c_and_ell, sigma_c: 170e6, ell: 4e-6}\n"
        "fatigue: {alpha_T: 1e6}\n",
        encoding="utf-8",
    )
    m = mat.load_material(y)
    for value in (
        m.elastic.E,
        m.elastic.nu,
        m.sigma_u,
        m.alpha_T,
        m.fracture.Gc,
        m.fracture.ell,
        m.fracture.sigma_c,
        m.fracture.kic_equivalent,
    ):
        assert isinstance(value, float)
    assert m.elastic.E == pytest.approx(69.0e9)
    assert m.fracture.Gc == pytest.approx(4.468, rel=1e-3)
    assert m.require_alpha_T() == pytest.approx(1.0e6)


def test_load_material_missing_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nelastic: {E: 1.0e9, nu: 0.3}\n", encoding="utf-8")
    with pytest.raises(KeyError):
        mat.load_material(bad)
