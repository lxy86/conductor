"""验证：**塑性激活** + cycle-jump 累积能否算到开裂（D-68 问题 21）。

**为什么必须做这个**：6.5 是项目里第一个让塑性真正激活的算例（钳口 359 MPa
≫ σ_y = 167 MPa），而它不起裂。已排除的四个解释（求解器 / 求积误差 / 网格加密 /
缺摩擦）都不足以定位，因为有一个**能力缺口**从未被验证：

    组合                                    验证过能裂？
    spectral + AT1 + cycle-jump（无塑性）      ✅ N = 108904（verify_plastic_life）
    spectral + AT1 + J2 塑性（同时激活）        ❌ 从未验证
    上述 + 罚接触 + 掩码                       ❌ 从未验证

`verify_plastic_life` 的弹塑性路径虽然 `max_d → 1`，但该工况 90 MPa < σ_y，
`max_eps_p = 0` —— 塑性**从未激活**（D-66 原文：「90 MPa 下塑性对寿命无影响」）。
`data/_probe_d_equation.py` 虽有塑性，但只解**一步** d，没跑循环累积。

**故若 6.6 或换工况仍不起裂，无法排除「塑性激活后 d 方程被卡住」，
而非「驱动真的不够」。本脚本补掉这个缺口。**

**做法**：复用 `scripts/calibrate_alpha_t.py::build_case`（标定用的均匀试样，
平面应变单轴、右边 Neumann 面力、R = -1），只把 **σ_a 提到必然屈服的水平**，
跑完整 cycle-jump（显式周期 + 外推 + 安定判定），看 d 能否到 1。

  - 判据（PLAN §2.5）：d ≥ 0.95 且与表面连通 -> N_起裂
  - 预期：塑性激活的档（σ_a > σ_y = 167 MPa）里若有起裂，说明模型链完整，
    6.5/6.6 的「驱动不足」结论可信；若全部 d ≡ 0 或 SNES 失败，
    则 6.5 的根因要重判（不是驱动不够，是塑性激活后 d 被卡住）

**为什么单轴拉伸就够**：ψ⁺ 与门槛都是**局部**关系（同点比值 r = 2ψ⁺/(f σ_c²/E)），
不需要接触或几何集中。均匀场下 r 全场一致，是最干净的判据。

用法::

    mpirun -n 6 python -u scripts/verify_plastic_crack.py
    mpirun -n 6 python -u scripts/verify_plastic_crack.py --sigma 250e6
    mpirun -n 6 python -u scripts/verify_plastic_crack.py --sigma 90e6 200e6 300e6

结果写入 `data/verify_plastic_crack/result.json`（rank 0）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

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
CALIB_CFG = ROOT / "configs/calib_tension_spectral.yaml"
OUT = ROOT / "data/verify_plastic_crack/result.json"
#: 循环稳定性判据（D-66 的阈值；jitter 按 D-68 问题 17 补）
SHAKE = cj.ShakedownConfig(zero_tol=1.0e-10, decay_ratio=0.5, n_check=3,
                           decay_jitter=0.01)
#: 默认扫描的载荷档（Pa）：90 是已验证的「无塑性」基线，其余越过 σ_y = 167 MPa
SIGMAS = [90.0e6, 150.0e6, 200.0e6, 250.0e6, 300.0e6]


def run_one(sigma_a: float, args) -> dict:
    """跑一个载荷档的完整 cycle-jump，返回结果字典。"""
    comm = MPI.COMM_WORLD
    cfg = yaml.safe_load(CALIB_CFG.read_text(encoding="utf-8"))
    # 收紧 u 的 Newton 容差（同 6.3.3 的理由：停止误差会淹没 ψ⁺ 的差异）
    cfg["solver"]["snes_rtol"] = 1.0e-10
    cfg["solver"]["snes_atol"] = 1.0e-14
    material = load_material(MATERIAL)
    fat = yaml.safe_load(MATERIAL.read_text(encoding="utf-8"))["fatigue"]
    # σ_y 从 J2 参数读（Material 没有 .plastic 字段，别写成 getattr 静默得 0）
    j2 = load_j2_params(MATERIAL)

    case = cal.build_case(
        cfg, material, sigma_a=sigma_a,
        alpha_T=float(fat["alpha_T"]), kappa=float(fat["kappa"]),
        plastic_prm=j2,                              # **始终挂塑性**
    )
    c = cfg["loading"]["cycle"]
    jcfg = cj.CycleJumpConfig(
        n_sub=int(c["n_sub"]), dN_init=float(c["dN_init"]), dN_min=float(c["dN_min"]),
        dN_max=args.dN_max, dd_max=float(c["dd_max"]), grow_below=float(c["grow_below"]),
        factor=float(c["factor"]), N_max=args.N_max,
        peak_phase=None if c.get("peak_phase") is None else float(c["peak_phase"]),
        shakedown=SHAKE,
    )
    driver = cj.CycleJumpDriver(case["solver"], case["state"], case["set_phase"], jcfg)

    t0 = time.time()
    reports = []
    error = None
    try:
        # **必须在 d 到 0.95 时停**：不传 stop 的话，driver 会以 ΔN 下界
        # 一路爬到 N_max = 1e8（2026-10-18 实测：200 MPa 档 d 早已饱和 1.0，
        # 却空转 45 分钟不退出，6 核满载）。N_起裂本来就是判据，到了就该收。
        reports = driver.run(stop=lambda r: r.max_d >= 0.95)
    except Exception as exc:
        # **保留部分进度**：driver.reports 在 run() 里逐步 append，
        # 异常时前面已成功的 jump 仍在。早先版本直接丢成 "error"，
        # 把「已经裂了但后续 u 发散」误报成「完全失败」。
        reports = list(driver.reports)
        error = str(exc)
    dt = time.time() - t0

    max_d = max((r.max_d for r in reports), default=0.0)
    max_ep = float(np.max(np.abs(case["solver"].plastic.ep_n.x.array)))
    max_p = float(np.max(np.abs(case["solver"].plastic.p_n.x.array)))
    # 自检：σ_a/σ_y > 1 时塑性必须激活（问题 19 的 ε_p=0 陷阱 ——
    # 本项目已两次被「resume 后 ε_p = 0 导致 ψ⁺ 虚高」坑过）。
    # 若 σ_a > σ_y 而 max_eps_p = 0，说明塑性没算，本档结论不可信。
    if sigma_a > float(j2.sigma_y0) and max_ep == 0.0:
        raise RuntimeError(
            "σ_a/σ_y = %.2f > 1 但 max_eps_p = 0 —— 塑性未激活，"
            "本档结论是假阳性（见 D-68 问题 19 的 ε_p=0 陷阱）"
            % (sigma_a / float(j2.sigma_y0))
        )
    last = reports[-1] if reports else None
    # 起裂判据：d ≥ 0.95（PLAN §2.5）。均匀试样无「表面连通深度」可言，
    # 用 d 首次达 0.95 的 N（与 verify_plastic_life 的 N_nucleation 同口径）。
    n_nuc = None
    for r in reports:
        if r.max_d >= 0.95:
            n_nuc = r.N
            break
    tags = {}
    for r in reports:
        k = str(r.shakedown)
        tags[k] = tags.get(k, 0) + 1
    return {
        "sigma_a": sigma_a,
        "sigma_over_yield": sigma_a / float(j2.sigma_y0),
        "N_final": driver.N,
        "N_nucleation": n_nuc,
        "cracked": n_nuc is not None,
        "max_d": max_d,
        "max_eps_p": max_ep,
        "max_p": max_p,
        "plasticity_active": max_ep > 0.0,
        "n_jumps": len(reports),
        "shakedown_tags": tags,
        "accepted": bool(last.accepted) if last else False,
        "status": ("cracked" if n_nuc is not None else
                   ("ratchet" if (last and not last.accepted) else "runout")),
        "error": error,
        "seconds": dt,
    }


def main():
    ap = argparse.ArgumentParser(description="塑性激活下 cycle-jump 能否算到开裂")
    ap.add_argument("--sigma", type=float, nargs="+", default=None,
                    help="应力幅（Pa），可给多个；缺省扫 %s" % [f"{s:.0e}" for s in SIGMAS])
    ap.add_argument("--dN-max", type=float, default=1.0e4)
    ap.add_argument("--N-max", type=float, default=1.0e8)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    sigmas = args.sigma if args.sigma else SIGMAS
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print("=" * 78)
        print("验证：塑性激活 + cycle-jump 能否算到开裂（D-68 问题 21）")
        print("=" * 78)
        print("")
        print("口径：平面应变单轴均匀试样、R = -1、spectral + AT1 + **J2 塑性**")
        print("参照：scripts/verify_plastic_life.py（90 MPa，该工况 ε_p ≡ 0）")
        print("σ_y = 167 MPa（Lévesque 2011，D-50）")
        print("")
        print("  σ_a[MPa]  σ_a/σ_y   N_起裂      N_final    max_d     max_eps_p  判定     步数")
        print("  " + "-" * 74)

    results = []
    for sigma in sigmas:
        try:
            r = run_one(sigma, args)
            results.append(r)
            if comm.rank == 0:
                print("  %7.1f  %6.2f   %-11s %-10.1f %.6f  %.3e  %-8s %d"
                      % (1e-6 * sigma, r["sigma_over_yield"],
                         ("%.1f" % r["N_nucleation"]) if r["N_nucleation"] else "—",
                         r["N_final"], r["max_d"], r["max_eps_p"],
                         r["status"], r["n_jumps"]), flush=True)
        except Exception as exc:
            results.append({"sigma_a": sigma, "error": str(exc)})
            if comm.rank == 0:
                print("  %7.1f  **失败**：%s" % (1e-6 * sigma, exc), flush=True)
        # error 非空但已有部分进度时，如实标注「部分完成」
        # **每档完成即落盘**：上一版只在全部跑完后写一次，2h 超时把 200/250/300
        # 的结果整批丢掉（90 已裂、200 日志里 d 已到 1.0，全没存下来）。
        if comm.rank == 0:
            out = pathlib.Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    if comm.rank == 0:
        out = pathlib.Path(args.out)
        print("")
        print("=" * 78)
        print("判读：")
        print("  - 塑性激活的档（σ_a > 167 MPa）里有「cracked」-> 模型链完整，")
        print("    6.5/6.6 的「驱动不足」结论可信")
        print("  - 若塑性激活的档全部 d ≡ 0 或失败 -> 6.5 根因要重判：")
        print("    不是驱动不够，是**塑性激活后 d 方程被卡住**")
        print("")
        print("结果已写入 %s" % out)


if __name__ == "__main__":
    main()
