"""Phase 6.4：循环稳定性检查（PLAN §2.4，D-66）。

测试意图：
  A. 判据（纯函数，合成历史，不跑 FE）：
     - 常数增量（棘轮）→ not_decaying，不外推；
     - 几何衰减（安定，慢）→ decaying，放行；
     - 首周期后为零（滞回环闭合）与全零（弹性安定）→ closed，放行；
     - 慢衰减（比值 0.9 > decay_ratio）→ not_decaying：保守，宁可停也不误放行；
     - 周期数不足 → undetermined，继续显式推进；
     - 增量重新出现（0 之后又非零）→ not_decaying。
  B. 驱动器连线（90 MPa，spectral，FE）：
     - 塑性求解器未给判据配置 → 报错（不静默外推）；
     - 注入棘轮历史后 step() 抛 RatchetDetected，且未外推（N 不变）；
     - 前 n_check 步为 undetermined（显式推进），随后判定 closed 并放行外推；
     - 弹性与弹塑性（90 MPa 下 ε_p ≡ 0）的前 K 个 jump 的 N、max d 逐点一致；
     - 快照/回滚包含 ε_p、p：回滚后塑性历史逐位恢复。

判据阈值（zero_tol、decay_ratio、n_check）待确认，见 DECISIONS.md D-66；此处用的值与 D-66 一致。
"""

import pathlib
import sys
from dataclasses import replace

import numpy as np
import pytest
import yaml

from pfwire import cyclejump as cj
from pfwire.material import load_material
from pfwire.plasticity import load_j2_params

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_alpha_t as cal  # noqa: E402

MATERIAL = ROOT / "configs/material_al1350.yaml"
SIGMA_A = 90.0e6
SHAKE = cj.ShakedownConfig(zero_tol=1.0e-10, decay_ratio=0.5, n_check=3)


# ---------------------------------------------------------------- A. 判据
@pytest.mark.parametrize(
    "history, expected",
    [
        ([1.0e-3] * 6, "not_decaying"),                                   # 棘轮：常数增量
        ([1.5**k * 1.0e-4 for k in range(6)], "not_decaying"),           # 棘轮：增大
        ([1.0e-3 * 0.3**k for k in range(6)], "decaying"),                # 安定：几何衰减
        ([1.0e-3, 0.0, 0.0, 0.0], "closed"),                              # 首周期后闭合
        ([0.0] * 5, "closed"),                                             # 弹性安定
        ([1.0e-3 * 0.9**k for k in range(6)], "not_decaying"),            # 慢衰减：保守拒绝
        ([1.0e-3, 1.0e-4], "undetermined"),                               # 周期数不足
        ([1.0e-3, 0.0, 1.0e-3], "not_decaying"),                          # 零之后重现增量
    ],
)
def test_classify_synthetic_histories(history, expected):
    assert cj.classify_plastic_history(history, SHAKE) == expected


def test_shakedown_config_rejects_bad_values():
    with pytest.raises(ValueError):
        cj.ShakedownConfig(zero_tol=-1.0, decay_ratio=0.5, n_check=3)
    with pytest.raises(ValueError):
        cj.ShakedownConfig(zero_tol=0.0, decay_ratio=1.0, n_check=3)
    with pytest.raises(ValueError):
        cj.ShakedownConfig(zero_tol=0.0, decay_ratio=0.5, n_check=1)


def test_only_closed_and_decaying_are_allowed():
    assert cj.SHAKEDOWN_ALLOWED == ("closed", "decaying")


# ---------------------------------------------------------------- B. 驱动器连线
def _setup_cfg():
    cfg = yaml.safe_load((ROOT / "configs/calib_tension_spectral.yaml").read_text(encoding="utf-8"))
    cfg["solver"]["snes_rtol"] = 1.0e-10
    cfg["solver"]["snes_atol"] = 1.0e-14
    fat = yaml.safe_load(MATERIAL.read_text(encoding="utf-8"))["fatigue"]
    return cfg, float(fat["alpha_T"]), float(fat["kappa"])


def _jump_config(cfg, shakedown):
    c = cfg["loading"]["cycle"]
    return cj.CycleJumpConfig(
        n_sub=int(c["n_sub"]), dN_init=float(c["dN_init"]), dN_min=float(c["dN_min"]),
        dN_max=float(c["dN_max"]), dd_max=float(c["dd_max"]),
        grow_below=float(c["grow_below"]), factor=float(c["factor"]),
        N_max=float(c["N_max"]),
        peak_phase=None if c.get("peak_phase") is None else float(c["peak_phase"]),
        shakedown=shakedown,
    )


def _driver(plastic: bool, shakedown=SHAKE):
    cfg, alpha_T, kappa = _setup_cfg()
    material = load_material(ROOT / cfg["material"])
    prm = load_j2_params(MATERIAL) if plastic else None
    case = cal.build_case(cfg, material, sigma_a=SIGMA_A, alpha_T=alpha_T, kappa=kappa,
                          plastic_prm=prm)
    driver = cj.CycleJumpDriver(case["solver"], case["state"], case["set_phase"],
                                _jump_config(cfg, shakedown))
    return driver, case


def test_plastic_driver_without_shakedown_config_is_refused():
    with pytest.raises(ValueError, match="shakedown"):
        _driver(plastic=True, shakedown=None)


def test_ratchet_history_stops_without_extrapolation():
    driver, _ = _driver(plastic=True)
    driver.plastic_increments[:] = [1.0e-3] * 5      # 注入棘轮历史
    with pytest.raises(cj.RatchetDetected):
        driver.step()
    assert driver.N == 0.0, "棘轮判定后不得外推（N 不应推进）"


def test_undetermined_steps_are_explicit_then_closed_allows_jumps():
    driver, _ = _driver(plastic=True)
    reports = [driver.step() for _ in range(6)]
    tags = [r.shakedown for r in reports]
    assert tags[:3] == ["undetermined"] * 3
    assert tags[3] == "closed"
    # undetermined 步只做显式周期：ΔN = 1（不外推）
    assert all(r.delta_N == 1.0 for r in reports[:3])
    assert all(r.accepted for r in reports)
    # 90 MPa 下 ε_p 始终为零：净增量全为零
    assert max(abs(x) for x in driver.plastic_increments) == 0.0
    assert np.max(np.abs(driver.solver.plastic.ep_n.x.array)) == 0.0


def test_elastic_and_plastic_jumps_agree_over_first_steps():
    """90 MPa 下 ε_p ≡ 0，弹塑性 cycle-jump 应与弹性逐步一致（D-66，6.3.3 的 N 比对）。

    弹塑性在判据未定（前 n_check 步）时只做显式周期（ΔN = 1），弹性没有这个限制。
    为可比，前三步两边都用 ΔN = 1 预热（弹性侧把 dN 限死为 1），然后弹性侧接过弹塑性的
    配置与 ΔN，逐步比较 N、ΔN 与 max d。
    """
    el, _ = _driver(plastic=False, shakedown=None)
    pl, _ = _driver(plastic=True)
    el.config = replace(pl.config, dN_init=1.0, dN_min=1.0, dN_max=1.0)
    el.delta_N = 1.0
    for _ in range(3):
        re, rp = el.step(), pl.step()
        assert rp.N == pytest.approx(re.N, rel=1e-10)
        assert rp.max_d == pytest.approx(re.max_d, rel=1e-8, abs=1e-14)
    el.config = pl.config
    el.delta_N = pl.delta_N
    for _ in range(5):
        re, rp = el.step(), pl.step()
        assert rp.shakedown == "closed"
        assert rp.N == pytest.approx(re.N, rel=1e-10)
        assert rp.delta_N == re.delta_N
        assert rp.max_d == pytest.approx(re.max_d, rel=1e-8, abs=1e-14)


def test_snapshot_round_trips_plastic_history():
    driver, case = _driver(plastic=True)
    driver.step()
    plastic = case["solver"].plastic
    snap = driver._snapshot()
    ep0, p0 = plastic.ep_n.x.array.copy(), plastic.p_n.x.array.copy()
    plastic.ep_n.x.array[:] = 1.0
    plastic.p_n.x.array[:] = 2.0
    driver._restore(snap)
    assert np.array_equal(plastic.ep_n.x.array, ep0)
    assert np.array_equal(plastic.p_n.x.array, p0)
