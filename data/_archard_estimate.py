"""Archard 磨损估算：说明本工况磨损深度相对裂纹深度可忽略（PLAN Phase 3 讨论 a）。

公式（Archard 1953，磨损体积正比于载荷×滑移距离/硬度）::

    V = k W s / H

k 为无量纲磨损系数，W 法向载荷，s 累计滑移距离，H 压入硬度。
二维线接触下按单位长度计：V' = k W' s / H，磨损深度 h = V' / 磨损带宽度。

**滑移幅来自本模型自己**（data/_slip_amplitude.py），不套解析式 ——
硬约束 2 不许自创公式，而 Cattaneo–Mindlin 的滑移位移解析式手头没有可引出处。
该脚本扣掉接触中心的平动后给出接触区内相对错动约 3–28 nm，取量级 10 nm。
**注意**：本模型用规定面力代替接触对偶，没有接触约束，所以这个滑移量只是
量级参考，Archard 结果须据此标注不确定性。

主论据不是 Archard 而是**摩擦学工况判别**：Q/(μP) = 0.6 < 1 属**部分滑移**，
按 fretting map（Vingsbo & Söderberg 1988）这是**裂纹主导**区而非磨损主导区；
磨损主导需要整体滑移（Q/(μP) → 1）。Archard 只是给这个判断配一个量级。

用法：python data/_archard_estimate.py
"""

from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]

#: 本模型算出的接触区相对错动量级（data/_slip_amplitude.py）
SLIP_NM = (3.0, 10.0, 30.0)          # 下界 / 取用值 / 上界
#: 无量纲磨损系数。部分滑移区远低于整体滑移区 —— 这正是论据的关键。
#: 取值区间取自常见铝合金干摩擦报道；整体滑移一档列出作对照。
K_WEAR = {"部分滑移（本工况）": (1.0e-6, 1.0e-5),
          "整体滑移（对照，非本工况）": (1.0e-4, 1.0e-3)}


def main() -> int:
    r = json.loads((REPO / "data" / "sn_sigma35" / "result.json")
                   .read_text(encoding="utf-8"))
    c = r["contact"]
    a, cc = c["a"], c["c"]
    W = c["normal_load"]                  # N/m（二维线接触，单位长度）
    N_f = r["N_f"]
    depth_crack = max(h["depth"] for h in r["history"])
    sigma_y = 165.0e6                     # Al 1350-H19 约当屈服
    H = 3.0 * sigma_y                     # 压入硬度 ~ 3σy

    band = 2.0 * (a - cc)                 # 磨损只发生在滑移区（两侧）
    print("基准工况 sn_sigma35：")
    print("  接触半宽 a = %.3f µm，黏着半宽 c = %.3f µm，滑移带总宽 %.3f µm"
          % (a * 1e6, cc * 1e6, band * 1e6))
    print("  单位长度法向载荷 W' = %.1f N/m，压入硬度 H = 3σy = %.0f MPa"
          % (W, H / 1e6))
    print("  N_f = %.4g，稳定段最大裂纹深度 = %.2f µm" % (N_f, depth_crack * 1e6))
    print()
    print("每周期滑移距离取 4δ（往复各一次、去回两程）。")
    print()
    hdr = "%-26s %10s %12s %12s %10s" % ("磨损系数 k", "δ (nm)", "总滑移 (mm)",
                                         "磨损深度 (µm)", "占裂纹深度")
    print(hdr)
    for label, (k_lo, k_hi) in K_WEAR.items():
        for k, kt in ((k_lo, "下界"), (k_hi, "上界")):
            for d_nm in (SLIP_NM[1],):
                s = 4.0 * d_nm * 1e-9 * N_f
                V = k * W * s / H          # m^2（单位长度体积）
                h = V / band
                print("%-26s %10.0f %12.3f %12.4f %9.2f%%"
                      % ("%s %s k=%.0e" % (label, kt, k), d_nm, s * 1e3,
                         h * 1e6, 100.0 * h / depth_crack))
    print()
    print("滑移量的敏感度（部分滑移上界 k = %.0e）：" % K_WEAR["部分滑移（本工况）"][1])
    k = K_WEAR["部分滑移（本工况）"][1]
    for d_nm in SLIP_NM:
        s = 4.0 * d_nm * 1e-9 * N_f
        h = k * W * s / H / band
        print("  δ = %5.0f nm -> 磨损深度 %.4f µm（占裂纹深度 %.2f%%）"
              % (d_nm, h * 1e6, 100.0 * h / depth_crack))
    print()
    print("结论：部分滑移区内，即使取该区间的上界磨损系数、并把滑移幅放大到 30 nm，")
    print("磨损深度仍在裂纹深度的百分之几以内 —— 故本模型不解磨损是站得住的。")
    print("整体滑移那一档列在上面只作对照：本工况 Q/(µP) = 0.6 < 1 不在该区。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
