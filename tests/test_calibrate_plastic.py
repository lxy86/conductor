"""Phase 6.3.3：塑性不改变标定（scripts/calibrate_alpha_t.py，σ_a = 90 MPa）。

论证：标定试样是力控单轴（σ_yy = 0、ε_zz = 0），von Mises 峰值
    σ_eq,max = σ_a √(1−ν+ν²) ≈ 0.883 σ_a ≈ 79 MPa < σ_y0 = 167 MPa。
B 口径下损伤只把屈服阈值抬到 σ_y/g（g ≥ 0 且随损伤减小），故全寿命内 ε_p ≡ 0。

测试意图：
  T0. 试样确为单轴：σ_a = 90 MPa 下有限元的 max σ_eq 与 σ_a √(1−ν+ν²) 相对误差 < 1e-3。
      整个论证依赖这一点，所以先验它。
  T1. 主检验（σ_a = 90 MPa，spectral，newton，2 个周期，不做 cycle jump）：
      弹塑性的 max|ε_p| 严格为 0；逐周期的 Δᾱ_cyc、ψ⁺_peak、max d 与纯弹性一致（rtol 1e-10）；
      Δᾱ_cyc > 0，防止 0 = 0 的空检验。
  T2. 阳性对照（σ_a = 220 MPa，σ_eq,max ≈ 194 MPa > σ_y0）：塑性分支确实激活，
      max|ε_p| > 0，且 ψ⁺_peak 与弹性解有可测差异。证明 T1 能发现虚假激活。
N 的直接比对推迟到 6.4（cycle jump 放开之后），见 DECISIONS.md D-65。
"""

import math
import pathlib
import sys

import numpy as np
import pytest
import ufl
import yaml
from dolfinx import fem
from mpi4py import MPI

from pfwire.material import load_material
from pfwire.plasticity import load_j2_params

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_alpha_t as cal  # noqa: E402

MATERIAL_YAML = ROOT / "configs/material_al1350.yaml"
SIGMA_A = 90.0e6
SIGMA_CONTROL = 220.0e6


def _setup():
    cfg = yaml.safe_load((ROOT / "configs/calib_tension_spectral.yaml").read_text(encoding="utf-8"))
    material = load_material(ROOT / cfg["material"])
    prm = load_j2_params(MATERIAL_YAML)
    fat = yaml.safe_load(MATERIAL_YAML.read_text(encoding="utf-8"))["fatigue"]
    # 标定值取自材料卡（D-59）：YAML 1.1 可能把无符号指数读成字符串，统一 float()
    return cfg, material, prm, float(fat["alpha_T"]), float(fat["kappa"])


def _run(sigma_a, plastic: bool, n_cycles: int):
    cfg, material, prm, alpha_T, kappa = _setup()
    # 收紧 u 的 Newton 容差：默认 snes_rtol=1e-6 的停止误差约 1e-7，会淹没 ψ⁺ 的比对（实测）
    cfg["solver"]["snes_rtol"] = 1.0e-10
    cfg["solver"]["snes_atol"] = 1.0e-14
    case = cal.build_case(
        cfg, material, sigma_a=sigma_a, alpha_T=alpha_T, kappa=kappa,
        plastic_prm=prm if plastic else None,
    )
    return cal.run_cycles(case, cfg, n_cycles)


def _max_von_mises(msh, u, prm):
    """平面应变（ε_zz = 0）下的 von Mises 应力场，在 DG0 上取最大值。"""
    eps = ufl.sym(ufl.grad(u))
    e3 = ufl.as_tensor([[eps[0, 0], eps[0, 1], 0.0], [eps[1, 0], eps[1, 1], 0.0], [0.0, 0.0, 0.0]])
    I3 = ufl.Identity(3)
    s3 = prm.lam * ufl.tr(e3) * I3 + 2.0 * prm.G * e3
    dev = s3 - ufl.tr(s3) / 3.0 * I3
    sig_eq = ufl.sqrt(1.5) * ufl.sqrt(ufl.inner(dev, dev))
    V0 = fem.functionspace(msh, ("DG", 0))
    f = fem.Function(V0)
    f.interpolate(fem.Expression(sig_eq, V0.element.interpolation_points))
    return msh.comm.allreduce(float(np.max(f.x.array, initial=0.0)), op=MPI.MAX)


def test_calibration_specimen_is_uniaxial_at_90MPa():
    cfg, material, prm, alpha_T, kappa = _setup()
    case = cal.build_case(cfg, material, sigma_a=SIGMA_A, alpha_T=alpha_T, kappa=kappa)
    case["set_phase"](0.25)
    case["solver"].solve_displacement()
    sig_max = _max_von_mises(case["mesh"], case["solver"].u, prm)
    expected = SIGMA_A * math.sqrt(1.0 - prm.nu + prm.nu**2)
    assert sig_max == pytest.approx(expected, rel=1e-3)
    assert sig_max < prm.sigma_y0


def test_plastic_run_matches_elastic_cycle_by_cycle_at_90MPa():
    elastic = _run(SIGMA_A, plastic=False, n_cycles=2)
    plastic = _run(SIGMA_A, plastic=True, n_cycles=2)
    assert len(elastic) == len(plastic) == 2
    for e, p in zip(elastic, plastic):
        assert p["max_eps_p"] == 0.0, "90 MPa 下不应屈服"
        assert e["delta_alpha_cycle"] > 0.0, "空检验：Δᾱ_cyc 为零则比对无意义"
        assert p["delta_alpha_cycle"] == pytest.approx(e["delta_alpha_cycle"], rel=1e-10)
        assert p["psi_peak"] == pytest.approx(e["psi_peak"], rel=1e-10)
        assert p["max_d"] == pytest.approx(e["max_d"], rel=1e-10, abs=1e-14)


def _first_peak(sigma_a, plastic: bool):
    """第一个峰值状态（d = 0）：求解后直接提交，不做损伤更新。

    在损伤已饱和（d → 1，g → k_res）的状态上做控制没有意义：屈服阈值 σ_y/g 被抬到无穷，
    ε_p 恒为零，检不出塑性是否激活。
    """
    cfg, material, prm, alpha_T, kappa = _setup()
    cfg["solver"]["snes_rtol"] = 1.0e-10
    cfg["solver"]["snes_atol"] = 1.0e-14
    case = cal.build_case(
        cfg, material, sigma_a=sigma_a, alpha_T=alpha_T, kappa=kappa,
        plastic_prm=prm if plastic else None,
    )
    solver = case["solver"]
    case["set_phase"](0.25)
    solver.solve_displacement()
    solver.commit_step()
    psi = case["comm"].allreduce(
        fem.assemble_scalar(case["psi_form"]), op=MPI.SUM
    ) / case["area"]
    eps_p = 0.0
    if case["plastic"] is not None:
        eps_p = case["comm"].allreduce(
            float(np.max(np.abs(case["plastic"].ep_n.x.array), initial=0.0)), op=MPI.MAX
        )
    return psi, eps_p


def test_plastic_branch_is_active_at_220MPa_control():
    psi_el, eps_p_el = _first_peak(SIGMA_CONTROL, plastic=False)
    psi_pl, eps_p_pl = _first_peak(SIGMA_CONTROL, plastic=True)
    assert eps_p_el == 0.0
    assert eps_p_pl > 0.0
    assert abs(psi_pl - psi_el) / psi_el > 1e-3
