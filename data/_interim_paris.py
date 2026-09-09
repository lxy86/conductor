"""从运行中的日志抽 a(N) 做**中途** Paris 拟合，早点判断算例是否在正轨上。

日志每 10 个 jump 步打一行 N 与 a（mm，4 位小数），足够做 Δa = 0.25 mm 的采样。
这只是中途估计：ΔK 跨度还没铺开，中间三分之一窗口里的点少，m 会有偏差。
"""

import re
import sys

import numpy as np

from pfwire import paris

W, T, DELTA_P = 50.0e-3, 1.0e-3, 20.0
REF_C, REF_M, REF_NU = 2.01e-2, 3.42, 7288

path = sys.argv[1] if len(sys.argv) > 1 else "/root/ct_h2.log"
pat = re.compile(r"N=([\d.eE+-]+)\s+ΔN=\S+\s+a=([\d.]+) mm")
N, a = [], []
for line in open(path, encoding="utf-8", errors="replace"):
    m = pat.search(line)
    if m:
        N.append(float(m.group(1)))
        a.append(float(m.group(2)) * 1e-3)
N = np.array(N)
a = np.array(a)
print("从日志抽到 %d 个 (N, a) 点" % N.size)
if N.size < 2:
    raise SystemExit("点太少")
print("N: %.6g -> %.6g    a: %.4f -> %.4f mm  (a/W %.3f -> %.3f)"
      % (N[0], N[-1], a[0] * 1e3, a[-1] * 1e3, a[0] / W, a[-1] / W))
print("裂纹已扩展 %.3f mm" % ((a[-1] - a[0]) * 1e3))

rate = paris.crack_growth_rate(a, N, W=W, T=T, delta_P=DELTA_P, delta_a=0.25e-3)
print("da/dN 采样区间数 = %d" % rate.n_intervals)
if rate.n_intervals >= 2:
    print("ΔK 跨度 %.4f ~ %.4f MPa√m"
          % (rate.delta_K.min() / 1e6, rate.delta_K.max() / 1e6))
    for i in range(rate.n_intervals):
        print("   a_mid=%7.3f mm  ΔK=%6.3f MPa√m  da/dN=%.4e mm/cycle"
              % (rate.a_mid[i] * 1e3, rate.delta_K[i] / 1e6, rate.da_dN[i] * 1e3))
try:
    fit = paris.paris_fit(a, N, W=W, T=T, delta_P=DELTA_P, delta_a=0.25e-3)
except ValueError as exc:
    raise SystemExit("中途还拟合不了：%s" % exc)

print()
print("中途 Paris：m = %.4g（原文 %.4g，%+.1f%%）  C_mm_MPa = %.4g（原文 %.4g）"
      % (fit.m, REF_M, 100 * (fit.m - REF_M) / REF_M, fit.C_mm_MPa, REF_C))
print("回归点数 %d，R^2 = %.4f，窗口 %.4f~%.4f MPa√m"
      % (fit.n_points_used, fit.r_squared,
         fit.fit_window[0] / 1e6, fit.fit_window[1] / 1e6))
print()
print("注意：ΔP 用的是 %g N（原文口径），而实际循环是 R = 0.2 的 "
      "P_max = 25 N / P_min = 5 N —— ΔK 的定义式里就该用 ΔP = P_max - P_min = 20 N。"
      % DELTA_P)
