"""对数型疲劳退化函数与 ᾱ 来源（DECISIONS.md D-14 / D-15）。

出处：Carrara, Ambati, Alessi & De Lorenzis 2020, CMAME 361:112731
（预印本 arXiv:1811.02244）的 Eq. (37)/(38)、Eq. (13) 与 Table 1。

测试意图
--------
1. Eq. (38) 的手算锚点：f(a_T) = 1、f(10 a_T) = (1-κ)²、f(100 a_T) = (1-2κ)²、
   f(截止点) = 0，且**截止之后保持 0**（平方项本身会回升，靠 conditional 压住）；
2. 单调性，以及 κ 越大衰减越快；κ 缺失/非正必须报错；
3. inverse_fatigue_degradation 与 f 互为逆（两种形式都测）；
4. **f 的 float 分支与 UFL 分支是两段独立代码，必须逐点给出同一个值**；
5. **D-14 的核心**：κ 能把 S–N 局部斜率调到铝的 b ≈ 0.13 附近，
   并与闭式斜率 b = 1/(2 + s ln10/κ) 逐点吻合；
6. D-15：ᾱ 累积退化前还是退化后的 ψ⁺，在 d = 0 时完全相同（故不影响 S–N），
   d > 0 后 degraded 的累积恰好少 g(d) 倍。
"""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from pfwire import cyclejump as cj
from pfwire import phasefield as pf
from test_phasefield import SIGMA_C, UniaxialStrainCase, make_material, mean_value, unit_mesh

KAPPA = 0.3
A_T = 1.0e6


# --------------------------------------------------------------------------
# 1-2. Eq. (38) 的性质
# --------------------------------------------------------------------------
def test_logarithmic_form_anchors():
    log = dict(form="logarithmic", kappa=KAPPA)
    assert pf.fatigue_degradation(0.0, A_T, **log) == pytest.approx(1.0)
    assert pf.fatigue_degradation(A_T, A_T, **log) == pytest.approx(1.0)
    # a = 10 a_T -> log10 = 1 -> f = (1-kappa)^2
    assert pf.fatigue_degradation(10.0 * A_T, A_T, **log) == pytest.approx(
        (1.0 - KAPPA) ** 2
    )
    # a = 100 a_T -> log10 = 2 -> f = (1-2 kappa)^2
    assert pf.fatigue_degradation(100.0 * A_T, A_T, **log) == pytest.approx(
        (1.0 - 2.0 * KAPPA) ** 2
    )
    cutoff = A_T * 10.0 ** (1.0 / KAPPA)
    assert pf.fatigue_degradation(cutoff, A_T, **log) == pytest.approx(0.0, abs=1e-12)
    assert pf.fatigue_degradation(10.0 * cutoff, A_T, **log) == 0.0


def test_logarithmic_form_is_monotone_and_kappa_tunes_decay():
    for kappa in (0.15, 0.3, 0.5):
        cutoff = A_T * 10.0 ** (1.0 / kappa)
        xs = np.geomspace(A_T, cutoff * 0.999, 40)
        vals = [
            pf.fatigue_degradation(x, A_T, form="logarithmic", kappa=kappa) for x in xs
        ]
        assert all(b <= a + 1e-15 for a, b in zip(vals, vals[1:]))
        assert vals[0] == pytest.approx(1.0)
        assert vals[-1] < 1.0e-4
    a = 20.0 * A_T
    assert pf.fatigue_degradation(
        a, A_T, form="logarithmic", kappa=0.5
    ) < pf.fatigue_degradation(a, A_T, form="logarithmic", kappa=0.15)


def test_logarithmic_form_requires_positive_kappa():
    with pytest.raises(ValueError, match="kappa"):
        pf.fatigue_degradation(1.0, A_T, form="logarithmic")
    with pytest.raises(ValueError, match="kappa"):
        pf.fatigue_degradation(1.0, A_T, form="logarithmic", kappa=0.0)


# --------------------------------------------------------------------------
# 3. 反函数
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "form,kwargs", [("asymptotic", {}), ("logarithmic", {"kappa": KAPPA})]
)
def test_inverse_fatigue_degradation_round_trip(form: str, kwargs: dict):
    for f_target in (1.0, 0.8, 0.5, 0.25, 0.05, 0.01):
        a = pf.inverse_fatigue_degradation(f_target, A_T, form, **kwargs)
        assert a >= A_T - 1e-9
        assert pf.fatigue_degradation(a, A_T, form, **kwargs) == pytest.approx(
            f_target, rel=1e-10
        )
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="f_target"):
            pf.inverse_fatigue_degradation(bad, A_T, form, **kwargs)


# --------------------------------------------------------------------------
# 4. float 分支与 UFL 分支必须一致
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "form,kwargs", [("asymptotic", {}), ("logarithmic", {"kappa": KAPPA})]
)
def test_ufl_and_scalar_branches_agree(form: str, kwargs: dict):
    msh = unit_mesh(2)
    V = fem.functionspace(msh, ("DG", 2))
    alpha = fem.Function(V)
    cutoff = A_T * 10.0 ** (1.0 / KAPPA)
    for value in (0.0, 0.5 * A_T, A_T, 3.0 * A_T, 30.0 * A_T, 2.0 * cutoff):
        alpha.x.array[:] = value
        alpha.x.scatter_forward()
        ufl_value = mean_value(pf.fatigue_degradation(alpha, A_T, form, **kwargs), msh)
        scalar_value = pf.fatigue_degradation(value, A_T, form, **kwargs)
        assert ufl_value == pytest.approx(scalar_value, rel=1e-12, abs=1e-14), (
            f"form={form}, alpha_bar={value:.3e}: UFL={ufl_value}, float={scalar_value}"
        )


# --------------------------------------------------------------------------
# 5. D-14 的核心：kappa 调 S–N 斜率
# --------------------------------------------------------------------------
def test_logarithmic_life_slope_brackets_aluminium_exponent():
    """κ ≈ 0.149 时局部斜率跨过铝的 b ≈ 0.1336，并与闭式斜率吻合。

    闭式：dlog10N/dlog10s = -(2 + s ln10/κ)，故 b = 1/(2 + s ln10/κ)。
    对照 asymptotic 形式：b 被锁死在 ~1/3，无论参数怎么取都到不了 0.13。
    """
    material = make_material(nu=0.0)
    a_T, kappa = 3.407e6, 0.14888
    inc_coeff = 1.5 * 0.5 * SIGMA_C**2 / material.elastic.E  # 每周期 = inc_coeff·s²

    def life(s: float, **kw) -> float:
        return cj.homogeneous_cycles_to_damage(
            s, a_T, material, delta_alpha_per_cycle=inc_coeff * s**2, **kw
        )

    for s in (0.243, 0.297, 0.378, 0.486):
        h = 1.0e-4 * s
        kw = dict(form="logarithmic", kappa=kappa)
        slope = (np.log10(life(s + h, **kw)) - np.log10(life(s - h, **kw))) / (
            np.log10(s + h) - np.log10(s - h)
        )
        b_numeric = -1.0 / slope
        b_closed = 1.0 / (2.0 + s * np.log(10.0) / kappa)
        assert b_numeric == pytest.approx(b_closed, rel=1e-4)
        assert 0.10 < b_numeric < 0.18

    b_lo = 1.0 / (2.0 + 0.486 * np.log(10.0) / kappa)
    b_hi = 1.0 / (2.0 + 0.243 * np.log(10.0) / kappa)
    assert b_lo < 0.1336 < b_hi, "kappa 没能把目标斜率夹住"

    # asymptotic 形式在同一区间的斜率远大于 0.18（够不到铝的值）
    for s in (0.243, 0.486):
        h = 1.0e-4 * s
        slope = (np.log10(life(s + h)) - np.log10(life(s - h))) / (
            np.log10(s + h) - np.log10(s - h)
        )
        assert -1.0 / slope > 0.28


def test_logarithmic_prefit_reproduces_four_sn_targets():
    """回归：预拟合参数必须让 4 个标定点的寿命偏差都在验收阈值内。

    数值来自 DECISIONS.md D-14 的解析预拟合（α_T = 3.407e6, κ = 0.14888）。
    """
    material = make_material(nu=0.0)
    a_T, kappa = 3.407e6, 0.14888
    inc_coeff = 1.5 * 0.5 * SIGMA_C**2 / material.elastic.E
    targets = [(90.0e6, 1.23e5), (70.0e6, 8.02e5), (55.0e6, 4.90e6), (45.0e6, 2.20e7)]
    errors = []
    for sigma_a, n_target in targets:
        s = sigma_a / SIGMA_C
        n_model = cj.homogeneous_cycles_to_damage(
            s,
            a_T,
            material,
            delta_alpha_per_cycle=inc_coeff * s**2,
            form="logarithmic",
            kappa=kappa,
        )
        errors.append(abs(np.log10(n_model / n_target)))
    assert max(errors) < 0.10, f"最大偏差 {max(errors):.3f} 超出预期"
    assert float(np.mean(errors)) < 0.08


# --------------------------------------------------------------------------
# 6. D-15：ᾱ 累积退化前还是退化后的 psi+
# --------------------------------------------------------------------------
def test_alpha_source_agrees_while_undamaged_and_differs_after():
    case = UniaxialStrainCase()
    states = {
        source: cj.FatigueState(
            case.msh, case.solver.model, alpha_T=1.0e12, alpha_source=source
        )
        for source in cj.ALPHA_SOURCES
    }
    for st in states.values():
        st.bind_solver(case.solver)

    case.u_applied.value = 0.5 * case.strain_c * case.L
    case.solver.solve_displacement()
    for st in states.values():
        st.accumulate()
    # d = 0 时两者只差 g(0) = 1 + k_res 这个残余刚度因子（1e-6 量级），
    # 故对 S–N 曲线无实质影响。这里按 g(0) 精确比，而不是笼统地要求「相等」。
    k_res = case.solver.model.k_res
    assert states["degraded"].max_alpha_bar() / states[
        "undegraded"
    ].max_alpha_bar() == pytest.approx(pf.degradation(0.0, k_res), rel=1e-12)

    # 人为置 d = 0.5 再加载：degraded 只吃 g(0.5) 倍
    case.solver.d.x.array[:] = 0.5
    case.solver.d.x.scatter_forward()
    for st in states.values():
        st.prime()
    case.u_applied.value = 0.9 * case.strain_c * case.L
    case.solver.solve_displacement()
    increments = {}
    for source, st in states.items():
        before = st.max_alpha_bar()
        st.accumulate()
        increments[source] = st.max_alpha_bar() - before
    ratio = increments["degraded"] / increments["undegraded"]
    assert ratio == pytest.approx(
        pf.degradation(0.5, case.solver.model.k_res), rel=1e-6
    )


def test_degraded_alpha_source_is_default_and_requires_d():
    case = UniaxialStrainCase()
    st = cj.FatigueState(case.msh, case.solver.model, alpha_T=1.0e6)
    assert st.alpha_source == "degraded"  # 按 Carrara Eq. 13
    with pytest.raises(ValueError, match="degraded"):
        st.bind_displacement(case.solver.u)


def test_bad_alpha_source_rejected():
    case = UniaxialStrainCase()
    with pytest.raises(ValueError, match="alpha_source"):
        cj.FatigueState(
            case.msh, case.solver.model, alpha_T=1.0e6, alpha_source="magic"
        )


# --------------------------------------------------------------------------
# 材料卡侧：form/kappa 的载入与校验
# --------------------------------------------------------------------------
def test_material_requires_kappa_when_logarithmic(tmp_path):
    from pfwire import material as mat

    y = tmp_path / "m.yaml"
    base = (
        "name: t\n"
        "elastic: {E: 69e9, nu: 0.33}\n"
        "strength: {sigma_u: 185e6}\n"
        "fracture: {mode: from_sigma_c_and_ell, sigma_c: 185e6, ell: 4e-6}\n"
    )
    y.write_text(base + "fatigue: {form: logarithmic, alpha_T: 3.4e6}\n", encoding="utf-8")
    m = mat.load_material(y)
    assert m.fatigue_form == "logarithmic" and m.kappa is None
    with pytest.raises(ValueError, match="kappa"):
        m.require_fatigue_params()

    y.write_text(
        base + "fatigue: {form: logarithmic, alpha_T: 3.4e6, kappa: 0.149}\n",
        encoding="utf-8",
    )
    m = mat.load_material(y)
    assert m.require_fatigue_params() == (3.4e6, "logarithmic", 0.149)

    y.write_text(base + "fatigue: {alpha_T: 1e10}\n", encoding="utf-8")
    m = mat.load_material(y)
    assert m.require_fatigue_params() == (1.0e10, "asymptotic", None)

    y.write_text(base + "fatigue: {alpha_T: 1e10, kapa: 0.1}\n", encoding="utf-8")
    with pytest.raises(KeyError, match="fatigue"):
        mat.load_material(y)  # 拼错的键必须报错，不能静默忽略


# --------------------------------------------------------------------------
# alpha_T / kappa 以 fem.Constant 进 UFL（DECISIONS.md D-16 / Phase 2.3 前置）
# --------------------------------------------------------------------------
def _unit_state(form: str, alpha_T: float, kappa=None):
    from dolfinx import mesh as dmesh
    from mpi4py import MPI

    from pfwire.cyclejump import FatigueState
    from pfwire.material import load_material
    from pfwire.phasefield import PhaseFieldModel

    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    material = load_material("configs/material_al1350.yaml")
    model = PhaseFieldModel(
        material=material, split="amor", k_res=1.0e-6, dissipation="AT1"
    )
    return msh, FatigueState(
        msh, model, alpha_T=alpha_T, form=form, kappa=kappa
    )


def _integrate_f(msh, state):
    """在单位面积域上积分 f(alpha_bar)，即得该常值处的 f。"""
    return msh.comm.allreduce(
        fem.assemble_scalar(fem.form(state.f_alpha * ufl.dx)), op=MPI.SUM
    )


@pytest.mark.parametrize(
    "form,alpha_T,kappa",
    [("asymptotic", 3.407e6, None), ("logarithmic", 3.407e6, 0.14888)],
)
def test_constant_backed_f_alpha_matches_float_formula(form, alpha_T, kappa):
    """Constant 进 UFL 后，装配出来的 f 必须与 float 公式逐点一致。"""
    msh, state = _unit_state(form, alpha_T, kappa)
    alpha_value = 5.0e7
    state.alpha_bar.x.array[:] = alpha_value
    state.alpha_bar.x.scatter_forward()
    got = _integrate_f(msh, state)
    expected = pf.fatigue_degradation(alpha_value, alpha_T, form, kappa)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_set_fatigue_params_changes_result_without_rebuilding_form():
    """**D-16 的要点**：改 alpha_T/kappa 只写 Constant.value，form 不重建。

    这里用同一个 `fem.form` 对象连续装配三组参数 —— 若参数是以 float 烘进
    form 的，改值根本不会生效（结果不变）；能变且与 float 公式一致，
    就证明它们确实是活的 Constant。标定要反复改这两个数，没有这一条
    每试一个值都要重新编译一遍（实测数秒到数十秒）。
    """
    msh, state = _unit_state("logarithmic", 3.407e6, 0.14888)
    alpha_value = 5.0e7
    state.alpha_bar.x.array[:] = alpha_value
    state.alpha_bar.x.scatter_forward()
    compiled = fem.form(state.f_alpha * ufl.dx)  # 只编译这一次

    seen = []
    for aT, kp in ((3.407e6, 0.14888), (1.0e7, 0.14888), (3.407e6, 0.30)):
        state.set_fatigue_params(alpha_T=aT, kappa=kp)
        got = msh.comm.allreduce(fem.assemble_scalar(compiled), op=MPI.SUM)
        expected = pf.fatigue_degradation(alpha_value, aT, "logarithmic", kp)
        assert got == pytest.approx(expected, rel=1e-12)
        seen.append(got)
    # 三组参数必须给出三个不同的值，否则说明根本没生效
    assert len(set(round(v, 12) for v in seen)) == 3


def test_set_fatigue_params_validates():
    msh, state = _unit_state("logarithmic", 3.407e6, 0.14888)
    with pytest.raises(ValueError, match="alpha_T"):
        state.set_fatigue_params(alpha_T=-1.0)
    with pytest.raises(ValueError, match="kappa"):
        state.set_fatigue_params(kappa=0.0)
    _, asym = _unit_state("asymptotic", 3.407e6)
    with pytest.raises(ValueError, match="不使用 kappa"):
        asym.set_fatigue_params(kappa=0.2)
