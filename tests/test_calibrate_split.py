"""Phase 6.3.1：标定脚本的 split 口径（scripts/calibrate_alpha_t.py）。

测试意图：
  - uniaxial_ratios 的解析系数在已知极限下正确：amor 的周期累积在 ν → 0 时回到 1.5（旧 docstring 的特例），
    spectral 的周期累积恰为全能量 1−ν²（两段半周期的 ψ⁺ 互不重叠）；
  - 均匀试样的有限元实测 ψ⁺_peak 与 Δᾱ_cycle 与解析值一致（amor、spectral 各一）。
    均匀场下 P1 位移精确，故这里的容差只需容纳求解器的收敛误差。
"""

import pathlib
import sys

import pytest
import yaml
from pfwire.material import load_material

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_alpha_t as cal  # noqa: E402


def test_uniaxial_ratios_limits():
    assert cal.uniaxial_ratios(0.0, "amor")["cycle"] == pytest.approx(1.5)
    nu = 0.33
    spec = cal.uniaxial_ratios(nu, "spectral")
    # 整周期 Δᾱ 恰为全能量
    assert spec["cycle"] == pytest.approx(1.0 - nu**2, rel=1e-14)
    # 压缩半周期贡献 ν²(1+ν)，拉伸峰值 + 压缩峰 = 全能量
    assert spec["peak"] + nu**2 * (1.0 + nu) == pytest.approx(1.0 - nu**2, rel=1e-14)
    with pytest.raises(ValueError):
        cal.uniaxial_ratios(nu, "rankine")


@pytest.mark.parametrize("split", ["amor", "spectral"])
def test_fe_accumulation_matches_closed_form(split):
    cfg = yaml.safe_load((ROOT / "configs/calib_tension_spectral.yaml").read_text(encoding="utf-8"))
    cfg["model"]["split"] = split
    # 各自用原口径的求解器：amor 走 linear（D-26 的标定路径）。amor 的 newton 在 u = 0 处
    # 落在 max_value(tr, 0) 的拐点上，SNES 线搜索发散（SNESConvergedReason = -6）；
    # 生产 spectral 不走 amor，此处只为测试口径，不修改求解器。
    cfg["solver"]["u_solver"] = "linear" if split == "amor" else "newton"
    material = load_material(ROOT / cfg["material"])
    case = cal.build_case(cfg, material, sigma_a=45.0e6, alpha_T=1.0e6, kappa=0.14)
    acc = cal.measure_accumulation(case, cfg)
    assert acc["split"] == split
    assert acc["psi_peak_over_1d"] == pytest.approx(acc["analytic_peak_over_1d"], rel=1e-4)
    assert acc["delta_alpha_over_1d"] == pytest.approx(acc["analytic_cycle_over_1d"], rel=1e-4)
