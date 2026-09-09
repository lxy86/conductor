"""run_case.py 单测：配置解析与静态算例驱动。

测试意图
--------
1. 配置解析：每一段的未知键都必须报错（拼错键而算例照跑是最难查的错）；
2. ``model.ell`` 覆盖必须**重解** (G_c, ell, sigma_c) 三角，使 sigma_c 与新 ell 自洽；
   若材料卡里 ell 本身是导出量，覆盖它自相矛盾，必须拒绝；
3. 材料卡路径：相对 cwd 找不到时退回到工况 yaml 所在目录；
4. 静态驱动：载荷–位移曲线在弹性段线性、出现峰值后因损伤终止；
   result.json 与 config.yaml 落盘；
5. **厚度换算**：二维反力是单位厚度的（N/m），F = F_per_thickness × thickness。
   这一步错了会与文献差 1000 倍（Miehe 的板厚 1 mm）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml
from mpi4py import MPI

from pfwire import run_case as rc

MIEHE_MATERIAL = """
name: bench
elastic: {E: 210.0e+9, nu: 0.3, plane_state: plane_strain}
strength: {sigma_u: 2.824e+9}
fracture: {dissipation: AT2, mode: from_gc_and_ell, Gc: 2700.0, ell: 7.5e-6}
fatigue: {alpha_T: null}
"""


def write_case(tmp_path, *, material_rel="mat.yaml", material_text=None, **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "mat.yaml").write_text(
        MIEHE_MATERIAL if material_text is None else material_text, encoding="utf-8"
    )
    case = {
        "case_type": "static",
        "material": material_rel,
        "model": {"split": "spectral", "dissipation": "AT2", "ell": 7.5e-6},
        "solver": {"u_solver": "newton", "max_iter": 30},
        "mesh": {
            "type": "sent",
            "W": 1.0e-3,
            "H": 1.0e-3,
            "notch_length": 0.5e-3,
            "thickness": 1.0e-3,
            "h_fine": 4.0e-5,
            "h_coarse": 1.2e-4,
            "refine_halfheight": 1.2e-4,
        },
        "loading": {
            "type": "tension_y",
            "stages": [{"du": 1.0e-6, "n": 6}, {"du": 2.0e-7, "n": 20}],
            "stop_damage": 0.9,
        },
        "output": {"dir": str(tmp_path / "out"), "save_fields_every": 0},
    }
    for section, patch in overrides.items():
        if isinstance(patch, dict):
            case.setdefault(section, {}).update(patch)
        else:
            case[section] = patch
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def case_path(tmp_path_factory):
    root = MPI.COMM_WORLD.bcast(tmp_path_factory.mktemp("case"), root=0)
    return write_case(root)


# --------------------------------------------------------------------------
# 1-3. 配置解析
# --------------------------------------------------------------------------
def test_load_case_basic(case_path):
    cfg = rc.load_case(case_path)
    assert cfg.case_type == "static"
    assert cfg.model.split == "spectral"
    assert cfg.model.dissipation == "AT2"
    assert cfg.thickness == pytest.approx(1.0e-3)
    assert cfg.solver.u_solver == "newton"
    assert cfg.material.fracture.ell == pytest.approx(7.5e-6)


@pytest.mark.parametrize(
    "section,bad_key",
    [
        ("model", "elll"),
        ("solver", "tol"),
        ("mesh", "h_finee"),
        ("loading", "stage"),
        ("output", "directory"),
    ],
)
def test_unknown_keys_are_rejected(tmp_path, section: str, bad_key: str):
    path = write_case(tmp_path, **{section: {bad_key: 1.0}})
    with pytest.raises(KeyError, match=section):
        cfg = rc.load_case(path)
        rc.run_static(cfg)  # mesh/loading 的键在这一步才校验


def test_unknown_top_level_key_rejected(tmp_path):
    path = write_case(tmp_path, fatigue={"alpha_T": 1.0})
    with pytest.raises(KeyError):
        rc.load_case(path)


def test_ell_override_rebuilds_fracture_triangle(tmp_path):
    """材料卡是 from_gc_and_ell，覆盖 ell 后 G_c 不变、sigma_c 随之改变。"""
    from pfwire import material as mat

    path = write_case(tmp_path, model={"ell": 15.0e-6})
    cfg = rc.load_case(path)
    frac = cfg.material.fracture
    assert frac.ell == pytest.approx(15.0e-6)
    assert frac.Gc == pytest.approx(2700.0, rel=1e-12)  # G_c 是输入，不变
    assert frac.dissipation == "AT2"
    assert frac.sigma_c == pytest.approx(
        mat.at2_sigma_c_from_gc_ell(210.0e9, 2700.0, 15.0e-6), rel=1e-12
    )
    # ell 变大 -> 峰值应力变小（sigma ∝ 1/sqrt(ell)）
    base = rc.load_case(write_case(tmp_path / "b", model={"ell": 7.5e-6}))
    assert frac.sigma_c < base.material.fracture.sigma_c
    assert frac.sigma_c / base.material.fracture.sigma_c == pytest.approx(
        1.0 / np.sqrt(2.0), rel=1e-9
    )


def test_ell_override_rejected_when_ell_is_derived(tmp_path):
    derived_ell = (
        "name: t\n"
        "elastic: {E: 210.0e+9, nu: 0.3}\n"
        "strength: {sigma_u: 1.0e+9}\n"
        "fracture: {dissipation: AT2, mode: from_gc_and_sigma_c, "
        "Gc: 2700.0, sigma_c: 2.8e+9}\n"
        "fatigue: {alpha_T: null}\n"
    )
    path = write_case(tmp_path, material_text=derived_ell, model={"ell": 1.0e-5})
    with pytest.raises(ValueError, match="自相矛盾"):
        rc.load_case(path)


def test_material_path_falls_back_to_case_dir(tmp_path):
    path = write_case(tmp_path, material_rel="configs/mat.yaml")
    cfg = rc.load_case(path)  # configs/mat.yaml 不存在，应退回同目录的 mat.yaml
    assert cfg.material.name == "bench"


def test_missing_material_raises(tmp_path):
    path = write_case(tmp_path, material_rel="nope.yaml")
    with pytest.raises(FileNotFoundError):
        rc.load_case(path)


def test_bad_case_type(tmp_path):
    path = write_case(tmp_path, case_type="dynamic")
    with pytest.raises(ValueError, match="case_type"):
        rc.load_case(path)


def test_fatigue_case_type_not_yet_implemented(tmp_path):
    path = write_case(tmp_path, case_type="fatigue")
    with pytest.raises(NotImplementedError, match="fatigue"):
        rc.run_case(path)


# --------------------------------------------------------------------------
# 4-5. 静态驱动
# --------------------------------------------------------------------------
def test_run_static_produces_peak_and_writes_files(case_path):
    cfg = rc.load_case(case_path)
    result = rc.run_static(cfg)
    assert result["status"] in ("completed", "damage_limit")
    curve = result["curve"]
    assert len(curve) >= 6

    u = np.array([c["u"] for c in curve])
    F = np.array([c["F"] for c in curve])
    assert np.all(np.diff(u) > 0.0)  # 位移单调增
    # 弹性段（前几步、损伤仍为 0）应当线性：F/u 近似常数
    elastic = [c for c in curve if c["max_d"] == 0.0][:4]
    if len(elastic) >= 3:
        stiffness = [c["F"] / c["u"] for c in elastic]
        assert max(stiffness) / min(stiffness) < 1.05

    # 厚度换算
    for c in curve:
        assert c["F"] == pytest.approx(c["F_per_thickness"] * cfg.thickness, rel=1e-12)
    peak = result["peak"]
    assert peak["F"] == pytest.approx(float(F.max()), rel=1e-12)
    assert peak["F"] == pytest.approx(
        peak["F_per_thickness"] * cfg.thickness, rel=1e-12
    )
    assert peak["F"] > 0.0

    if MPI.COMM_WORLD.rank == 0:
        out = cfg.output_dir
        saved = json.loads((out / "result.json").read_text(encoding="utf-8"))
        assert saved["peak"]["F"] == pytest.approx(peak["F"], rel=1e-12)
        assert (out / "config.yaml").exists()


def test_run_static_rejects_bad_stages(tmp_path):
    path = write_case(tmp_path, loading={"stages": [{"du": -1.0, "n": 3}]})
    cfg = rc.load_case(path)
    with pytest.raises(ValueError, match="du"):
        rc.run_static(cfg)
    path2 = write_case(tmp_path / "b", loading={"stages": []})
    cfg2 = rc.load_case(path2)
    with pytest.raises(KeyError, match="stages"):
        rc.run_static(cfg2)
