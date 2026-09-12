"""Phase 2.4 硬关卡 `fretting._gate_report` 的单测。

关卡本身不算数值，只是对 history 的判读 —— 所以可以（也应该）用合成 history
把每一条判据的边界都盯住，不必跑 6 小时的算例。

背景见 DECISIONS.md OPEN-2：判据 3「转向垂直」原先错用了**表层窗口**的倾角，
那个量在裂纹深过窗口后就不再变化，按定义测不出转向。
"""

from __future__ import annotations

import numpy as np
import pytest

from pfwire import loads as pfloads
from pfwire.fretting import _gate_report

A = 11.62e-6          # 接触半宽，与 configs/fretting_base.yaml 同量级
C_OVER_A = 0.6325     # c/a = sqrt(1 - Q/(mu P))，Q/(mu P) = 0.6


@pytest.fixture
def contact():
    return pfloads.ContactLoads.from_peak_pressure(
        p0=240.0e6, mu=0.7, Q_over_mu_P=0.6, E_star=38.716e9, R=0.9375e-3
    )


def step(N, depth, *, x_surface=-1.02 * A, incl=None, tip=None):
    return {
        "N": N, "depth": depth, "x_surface": x_surface,
        "inclination_deg": incl, "tip_direction_deg": tip,
    }


def history(depths, incls, tips, *, x_surface=-1.02 * A):
    return [
        step(100.0 * (i + 1), d, x_surface=x_surface, incl=a, tip=t)
        for i, (d, a, t) in enumerate(zip(depths, incls, tips))
    ]


def test_gate_passes_when_crack_turns_toward_normal(contact):
    """典型的微动裂纹：滑移区萌生、初始 −17°、长到 2a 以上时尖端转到 −3°。"""
    deps = np.linspace(2.0e-6, 2.5 * A, 12)
    g = _gate_report(
        history(deps, [-17.0] * 12, np.linspace(-17.0, -3.0, 12)), contact
    )
    assert g["verdict"] == "pass" and g["passed"]
    assert g["nucleates_in_slip_zone"] and g["initially_tilted"]
    assert g["turns_toward_normal"]
    assert g["deep_enough_to_judge"]


def test_gate_is_inconclusive_when_crack_too_shallow(contact):
    """只长到 1.16a 就结束（第一次生产算例的真实情形）：是「没看到」，不是「证伪」。"""
    deps = np.linspace(2.0e-6, 1.16 * A, 10)
    g = _gate_report(history(deps, [-17.0] * 10, [-20.0] * 10), contact)
    assert g["verdict"] == "inconclusive"
    assert not g["passed"]
    assert not g["deep_enough_to_judge"]
    assert g["max_depth_over_a"] == pytest.approx(1.16, rel=1e-3)


def test_gate_fails_when_deep_enough_but_never_turns(contact):
    """长够了深度仍不转向 —— 这才是真正的 fail。"""
    deps = np.linspace(2.0e-6, 3.0 * A, 12)
    g = _gate_report(history(deps, [-17.0] * 12, np.linspace(-14.0, -26.0, 12)), contact)
    assert g["verdict"] == "fail" and not g["passed"]
    assert not g["turns_toward_normal"]


def test_surface_window_angle_cannot_replace_tip_direction(contact):
    """回归 OPEN-2：表层倾角一直不动、尖端却已转向时，关卡必须判 pass。

    反过来说，只要判据 3 还用 inclination_deg，这条就会失败 —— 这正是
    第一次生产算例被误判的机制。
    """
    deps = np.linspace(2.0e-6, 2.5 * A, 12)
    incl_frozen = [-16.7] * 12                      # 表层窗口冻结在最初那一段
    tip_turning = list(np.linspace(-16.7, -2.0, 12))
    g = _gate_report(history(deps, incl_frozen, tip_turning), contact)
    assert g["inclination_final_deg"] == pytest.approx(-16.7)
    assert g["tip_direction_final_deg"] == pytest.approx(-2.0)
    assert g["passed"], "尖端已转向，却因表层量不动被判失败"


def test_gate_rejects_nucleation_outside_slip_zone(contact):
    """裂纹长在接触中心（|x|/a ≈ 0）时必须判失败 —— 这是 D-8 的假损伤特征。"""
    deps = np.linspace(2.0e-6, 2.5 * A, 12)
    g = _gate_report(
        history(deps, [-17.0] * 12, np.linspace(-17.0, -3.0, 12), x_surface=-0.03 * A),
        contact,
    )
    assert not g["nucleates_in_slip_zone"]
    assert g["verdict"] == "fail" and not g["passed"]


def test_gate_requires_initial_tilt(contact):
    """一上来就垂直的裂纹不是微动裂纹（判据 2）。"""
    deps = np.linspace(2.0e-6, 2.5 * A, 12)
    g = _gate_report(history(deps, [-1.0] * 12, [-1.0] * 12), contact)
    assert not g["initially_tilted"]
    assert not g["passed"]


def test_gate_reports_no_crack(contact):
    g = _gate_report(
        [{"N": 1.0, "depth": 0.0, "x_surface": None,
          "inclination_deg": None, "tip_direction_deg": None}],
        contact,
    )
    assert not g["passed"]
    assert "没有出现" in g["reason"]


def test_judge_depth_is_configurable(contact):
    """judge_depth_over_a 放宽到 1.0 时，同一段 history 从 inconclusive 变成 fail。"""
    deps = np.linspace(2.0e-6, 1.16 * A, 10)
    h = history(deps, [-17.0] * 10, [-20.0] * 10)
    assert _gate_report(h, contact)["verdict"] == "inconclusive"
    assert _gate_report(h, contact, judge_depth_over_a=1.0)["verdict"] == "fail"
