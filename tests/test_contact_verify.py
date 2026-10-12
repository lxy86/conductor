"""接触验证脚本里的纯后处理（黏滑判定、库仑不等式、滑移占比）的单元测试。

只测不依赖 FE 的部分：输入手造的 Hertz / Cattaneo–Mindlin 剖面，核对判定函数。
FE 本体的验证是 `scripts/verify_contact_*.py`（串行约 15 分钟，不进 pytest）。
"""
import importlib.util
import pathlib

import numpy as np
import pytest

pytest.importorskip("dolfinx")

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "verify_contact_mindlin.py"
_spec = importlib.util.spec_from_file_location("verify_contact_mindlin", _SCRIPT)
vm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vm)

MU, A, P0 = 0.3, 1.0, 10.0
E_STAR = 100.0


def _hertz(n=201):
    x = np.linspace(-1.3 * A, 1.3 * A, n)
    p = np.where(np.abs(x) < A, P0 * np.sqrt(np.clip(1 - (x / A) ** 2, 0, None)), 0.0)
    return x, p


def _mindlin_q(x, p, c):
    """Cattaneo–Mindlin：q = mu p0 sqrt(1-x²/a²) - mu p0 (c/a) sqrt(1-x²/c²)（Johnson 1985, §7.2）。"""
    q = MU * p.copy()
    inner = np.abs(x) < c
    q[inner] -= MU * P0 * (c / A) * np.sqrt(1 - (x[inner] / c) ** 2)
    return q


@pytest.mark.parametrize("c_over_a", [0.3, 0.6, 0.9])
def test_stick_halfwidth_recovers_mindlin_c(c_over_a):
    x, p = _hertz()
    q = _mindlin_q(x, p, c_over_a * A)
    c = vm.measure_stick_halfwidth(x, p, q, MU, A)
    assert c == pytest.approx(c_over_a * A, abs=0.06)


def test_analyze_gross_sliding():
    """整体滑移：q = mu p 处处 -> 滑移占比 1、库仑比 1。"""
    x, p = _hertz()
    r = vm.analyze_step(1, x, p, MU * p, Pn=1.0, Q=MU * 1.0, mu=MU, R_eff=4.0, e_star=E_STAR)
    assert r["slip_frac"] == pytest.approx(1.0)
    assert r["coulomb_max"] == pytest.approx(1.0)
    assert r["q_ratio"] == pytest.approx(1.0)


def test_analyze_pure_stick():
    """q 很小 -> 无滑移，库仑比远小于 1。"""
    x, p = _hertz()
    r = vm.analyze_step(1, x, p, 0.05 * MU * p, Pn=1.0, Q=0.05 * MU, mu=MU, R_eff=4.0, e_star=E_STAR)
    assert r["slip_frac"] == 0.0
    assert r["coulomb_max"] == pytest.approx(0.05)


def test_analyze_ignores_edge_blowup():
    """接触区边缘 p -> 0 而 q 不为 0 -> |q|/(mu p) 发散；核心区判定不应受其影响。"""
    x, p = _hertz()
    q = 0.5 * MU * p
    edge = (np.abs(x) > 0.95 * A) & (np.abs(x) < A)
    q[edge] = 3.0 * MU * P0       # 边缘处人为放一个巨大的 q
    r = vm.analyze_step(1, x, p, q, Pn=1.0, Q=0.5 * MU, mu=MU, R_eff=4.0, e_star=E_STAR)
    assert r["coulomb_max"] == pytest.approx(0.5)


def test_analyze_partial_slip_has_both_zones():
    x, p = _hertz()
    q = _mindlin_q(x, p, 0.6 * A)
    r = vm.analyze_step(1, x, p, q, Pn=1.0, Q=0.5, mu=MU, R_eff=4.0, e_star=E_STAR)
    assert 0.0 < r["slip_frac"] < 1.0
