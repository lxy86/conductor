"""CT 临界载荷的 LEFM 锚点，并回查原文循环载荷的 R 比。

Griffith/LEFM：K_I = P/(T sqrt(W)) Y(a/W)，令 K_I = K_IC 得临界载荷
    P_c = K_IC T sqrt(W) / Y(a0/W)
Y 用 ASTM E647 的 CT 标定式（= Carrara Eq. B.3）。相场在脆性、ℓ 远小于
韧带尺寸时应当复现 Griffith，故这是判断「我们算得对不对」的独立标尺。
"""

import re

import numpy as np
import pymupdf

from pfwire import paris

E, NU, GC = 6.0e9, 0.22, 2280.0
W, T = 50.0e-3, 1.0e-3
A0_OVER_W = 0.22

kic = np.sqrt(E * GC)  # 平面应力
Y = paris.ct_geometric_factor(A0_OVER_W)
P_c = kic * T * np.sqrt(W) / Y

print("K_IC = sqrt(E Gc) = %.4f MPa sqrt(m)   （原文 3.69）" % (kic / 1e6))
print("Y(a0/W = %.2f) = %.4f" % (A0_OVER_W, Y))
print("LEFM 临界载荷 P_c = K_IC T sqrt(W) / Y = %.1f N" % P_c)
print()
print("本文相场（力控最后收敛）：h=ell/2 190 N、h=ell/4 190 N、尖锥收 185 N")
print("  -> 相对 LEFM %+.1f%% ~ %+.1f%%"
      % (100 * (185 - P_c) / P_c, 100 * (190 - P_c) / P_c))
print()

# Tab. 2 的 Pmax/Pmon 反解：若 Pmax = dP（即 Pmin = 0）
rows = [(10, 6.7), (15, 10.1), (20, 13.5), (40, 26.9), (60, 40.4), (80, 53.8),
        (100, 67.3), (110, 74.0), (120, 80.8), (130, 87.5), (140, 94.2)]
vals = [dp / (pct / 100.0) for dp, pct in rows]
print("Tab.2 反解 (假设 Pmax = dP)：Pmon = %.1f ~ %.1f N，均值 %.1f"
      % (min(vals), max(vals), float(np.mean(vals))))
print("  11 行高度自洽 -> Pmax 确实正比于 dP，但比例常数取决于 R = Pmin/Pmax：")
print("  Pmax = dP/(1-R)  =>  Pmon = %.1f/(1-R)" % float(np.mean(vals)))
for R in (0.0, 0.1, 0.186, 0.2, 0.3):
    print("    R = %.3f -> Pmon = %6.1f N   （LEFM %.1f，本文 185~190）"
          % (R, float(np.mean(vals)) / (1 - R), P_c))
print()
print("=== 回查原文里的 R / Pmin 说法 ===")
d = pymupdf.open("carrara2020.pdf")
pats = [r"load ratio", r"R\s*=\s*0", r"Pmin\s*=", r"P min", r"zero.{0,20}minimum",
        r"pulsating", r"R-ratio", r"stress ratio"]
for i in range(d.page_count):
    t = d[i].get_text()
    for pat in pats:
        for m in re.finditer(pat, t, re.I):
            seg = t[max(0, m.start() - 220): m.start() + 220].replace("\n", " ")
            print("p%-3d [%s] ... %s ..." % (i + 1, pat, seg))
            print("---")
