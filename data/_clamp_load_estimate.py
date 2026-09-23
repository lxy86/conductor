"""从线夹标准反推 R_clamp 与 W_clamp，并用 Hertz 估一下弹性接触压力（PLAN §1.1 / D-47）。

**不手算**：这条链上有五步换算，手算错一步就会把整个新模型的载荷定错。

数据来源（两条都是可引的）：
  1. T/CMBN《金属模锻铝合金悬垂线夹》5.3.3：
     「导线槽的曲率半径应不小于所安装导线外径的 6 倍」-> R_clamp >= 6 D
  2. 同标准 5.4.1：
     「线夹对导线的握力……不应小于被安装导线额定计算拉断力（RTS）的 20%」

握力是**抗滑移**指标，受摩擦限制：F_grip = mu * F_normal，故
     F_normal = F_grip / mu

另有一条二手来源（《线路金具》，转述我国与俄罗斯设计惯例，**出处待核**）：
  - 我国：线槽曲率不小于导线直径 8~10 倍
  - 俄罗斯：线槽中段 R = (150~200) D；**出口段 r = (2~3) D**；船体长度 >= 10 D
出口段那条更贴近 LPC（last point of contact，裂纹实际发生处），作为**最不利**工况列出。

用法：python data/_clamp_load_estimate.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pfwire.loads import hertz_half_width, hertz_peak_pressure
from pfwire.material import contact_effective_modulus

# ---- 导线卡（configs/conductor_jl1_lha1_465_210.yaml）----
D_OVERALL = 33.75e-3      # m，整根导线外径（61 线同心绞，D = 9 d）
D_WIRE = 3.75e-3          # m，铝单线直径
RTS = 137.02e3            # N，计算拉断力
N_OUTER = 24              # 外层铝单线根数

# ---- 材料（configs/material_al1350.yaml）----
E_AL, NU_AL = 69.0e9, 0.33
SIGMA_Y = 165.0e6
MU = 0.7                  # 铝–铝，与旧模型同取值

#: 线夹与导线的接触根数：船体包角 120~180 度 -> 外层 24 根里约 1/3~1/2 参与
N_CONTACT = (8, 12)
#: 握力裕度：标准给的是**下限**，实际设计会高于它
GRIP_FACTORS = (1.0, 1.5)

GEOMS = {
    "标准下限 R = 6 D（T/CMBN 5.3.3，可引）": 6.0 * D_OVERALL,
    "出口段 r = 2 D（二手来源，最不利）": 2.0 * D_OVERALL,
    "出口段 r = 3 D（二手来源）": 3.0 * D_OVERALL,
}


def main() -> int:
    print("导线 D = %.2f mm，单丝 d = %.2f mm，RTS = %.2f kN，外层 %d 根"
          % (D_OVERALL * 1e3, D_WIRE * 1e3, RTS / 1e3, N_OUTER))
    print()

    f_grip_min = 0.20 * RTS
    print("握力下限（标准 5.4.1）F_grip >= 20%% RTS = %.2f kN" % (f_grip_min / 1e3))
    print("法向夹持力   F_N = F_grip / mu = %.2f kN（mu = %.1f）"
          % (f_grip_min / MU / 1e3, MU))
    print()

    # 平面应变下 W' 是**每单位出平面长度**的力。出平面方向是横向，单丝宽 d_wire，
    # 故 W'(N/m) = 该单丝承担的力 / d_wire。
    print("单丝载荷与等效线载荷 W'（W' = F_wire / d_wire）:")
    print("  %-22s %10s %12s %14s" % ("工况", "F_wire (N)", "W' (MN/m)", "备注"))
    cases = []
    for gf in GRIP_FACTORS:
        for nc in N_CONTACT:
            f_wire = f_grip_min * gf / MU / nc
            w = f_wire / D_WIRE
            tag = "握力×%.1f，%d 根接触" % (gf, nc)
            print("  %-22s %10.0f %12.3f %14s" % ("", f_wire, w / 1e6, tag))
            cases.append((tag, w))
    print()

    # 线夹取刚体（PLAN 决策 B）：E* = E/(1-nu^2)，即对偶体刚度无穷大
    e_star = contact_effective_modulus(E_AL, NU_AL, float("inf"), 0.0)
    h_ind = 3.0 * SIGMA_Y
    print("E*（铝–刚体线夹）= %.3f GPa；压入硬度 3 sigma_y = %.0f MPa"
          % (e_star / 1e9, h_ind / 1e6))
    print()
    print("弹性 Hertz 线接触估算（若 p0 超过 3 sigma_y，说明必须上塑性）:")
    print("  %-34s %-22s %9s %9s %9s"
          % ("线夹曲率", "载荷工况", "a (mm)", "p0 (MPa)", "p0/3sy"))
    for gname, r_clamp in GEOMS.items():
        for tag, w in cases:
            a = hertz_half_width(P=w, R=r_clamp, E_star=e_star)
            p0 = hertz_peak_pressure(P=w, a=a)
            print("  %-34s %-22s %9.3f %9.0f %9.2f"
                  % (gname, tag, a * 1e3, p0 / 1e6, p0 / h_ind))
    print()
    print("读法：p0/3sy < 1 表示该工况下接触仍是弹性的；>= 1 表示必然进入塑性。")
    print("标准下限的 R = 6 D 很平缓（202.5 mm），接触相当共形，p0 未必很高；")
    print("而 LPC 处的出口段圆角要小得多（2~3 D），那里才是塑性与裂纹的发生处。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
