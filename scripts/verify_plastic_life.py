"""Phase 6.3.3 的 N 比对（D-66）：σ_a = 90 MPa 下弹性与弹塑性 cycle-jump 的起裂寿命。

与 scripts/calibrate_alpha_t.py 的均匀试样相同（配置 configs/calib_tension_spectral.yaml，
α_T、κ 取自材料卡）。弹塑性一侧加 J2 塑性并给出循环稳定性判据（PLAN §2.4）。
两侧都收紧 u 的 Newton 容差，使比对不受求解器停止误差影响。

用法（单工况用 6 个进程，见 CLAUDE.md）：
    mpirun -n 6 python scripts/verify_plastic_life.py
结果写入 data/verify_plastic_life/result.json（rank 0）。
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from collections import Counter

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import calibrate_alpha_t as cal  # noqa: E402
from mpi4py import MPI  # noqa: E402

from pfwire import cyclejump as cj  # noqa: E402
from pfwire.material import load_material  # noqa: E402
from pfwire.plasticity import load_j2_params  # noqa: E402

MATERIAL = ROOT / "configs/material_al1350.yaml"
OUT = ROOT / "data/verify_plastic_life/result.json"
SIGMA_A = 90.0e6
#: 循环稳定性判据（D-66，待确认的阈值，与 tests/test_shakedown.py 一致）
SHAKE = cj.ShakedownConfig(zero_tol=1.0e-10, decay_ratio=0.5, n_check=3)


def life(plastic: bool) -> dict:
    comm = MPI.COMM_WORLD
    cfg = yaml.safe_load((ROOT / "configs/calib_tension_spectral.yaml").read_text(encoding="utf-8"))
    cfg["solver"]["snes_rtol"] = 1.0e-10
    cfg["solver"]["snes_atol"] = 1.0e-14
    material = load_material(ROOT / cfg["material"])
    fat = yaml.safe_load(MATERIAL.read_text(encoding="utf-8"))["fatigue"]
    case = cal.build_case(
        cfg, material, sigma_a=SIGMA_A, alpha_T=float(fat["alpha_T"]), kappa=float(fat["kappa"]),
        plastic_prm=load_j2_params(MATERIAL) if plastic else None,
    )
    c = cfg["loading"]["cycle"]
    jcfg = cj.CycleJumpConfig(
        n_sub=int(c["n_sub"]), dN_init=float(c["dN_init"]), dN_min=float(c["dN_min"]),
        dN_max=float(c["dN_max"]), dd_max=float(c["dd_max"]),
        grow_below=float(c["grow_below"]), factor=float(c["factor"]),
        N_max=float(c["N_max"]),
        peak_phase=None if c.get("peak_phase") is None else float(c["peak_phase"]),
        shakedown=SHAKE if plastic else None,
    )
    driver = cj.CycleJumpDriver(case["solver"], case["state"], case["set_phase"], jcfg)
    d_nuc = float(cfg["loading"]["nucleation_d"])
    t0 = time.perf_counter()
    reports = driver.run(stop=lambda r: r.max_d >= d_nuc)
    seconds = time.perf_counter() - t0
    last = reports[-1]
    ok = last.accepted and last.max_d >= d_nuc
    return {
        "plastic": plastic,
        "N_nucleation": float(last.N) if ok else float("nan"),
        "reached_nucleation": bool(ok),
        "max_d": float(last.max_d),
        "n_jumps": len(reports),
        "shakedown_counts": dict(Counter(r.shakedown for r in reports)),
        "max_eps_p": 0.0 if case["plastic"] is None else comm.allreduce(
            float(np.max(np.abs(case["plastic"].ep_n.x.array), initial=0.0)), op=MPI.MAX
        ),
        "seconds": seconds,
    }


def main() -> int:
    comm = MPI.COMM_WORLD
    results = {"elastic": life(False), "plastic": life(True)}
    if comm.rank == 0:
        e, p = results["elastic"], results["plastic"]
        rel = abs(p["N_nucleation"] - e["N_nucleation"]) / e["N_nucleation"]
        results["rel_diff_N"] = rel
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
