"""看微动裂纹深度随 N 的趋势，判断会长到 100 µm 还是会止裂。

物理预期：微动裂纹在表面附近由接触应力驱动，长得快；一旦越过接触影响区
（深度约 a），驱动力只剩本体应力 sigma_a，可能急剧减速甚至止裂。
"""

import json
import pathlib
import sys

import numpy as np

d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/fretting_base")
r = json.loads((d / "result.json").read_text(encoding="utf-8"))
h = r["history"]
a = r["contact"]["a"]
print("%s  status=%s  步数=%d  N=%.6g  耗时 %.0f s"
      % (d.name, r["status"], r["n_jumps"], r["N_final"], r["timing_s"]["solve"]))
print("a = %.3f µm；失效判据深度 %.0f µm = %.2f a"
      % (a * 1e6, r["failure_criterion"]["depth"] * 1e6 if isinstance(
          r["failure_criterion"], dict) else 100.0,
         (100e-6) / a))

N = np.array([x["N"] for x in h])
dep = np.array([x["depth"] for x in h])
ang = [x["inclination_deg"] for x in h]

first = next((i for i, v in enumerate(dep) if v > 0), None)
if first is None:
    print("尚无裂纹")
    raise SystemExit
print()
print("萌生：N = %.6g，x_init/a = %.3f" % (N[first], h[first]["x_surface_over_a"]))
print()
print("      N        深度(µm)  深度/a   倾角°")
step = max(1, (len(h) - first) // 18)
for i in range(first, len(h), step):
    print("  %9.5g   %7.2f  %6.2f   %s"
          % (N[i], dep[i] * 1e6, dep[i] / a,
             "-" if ang[i] is None else "%.1f" % ang[i]))
if len(h) - 1 > first:
    i = len(h) - 1
    print("  %9.5g   %7.2f  %6.2f   %s  <- 最新"
          % (N[i], dep[i] * 1e6, dep[i] / a,
             "-" if ang[i] is None else "%.1f" % ang[i]))

# 最近一段的平均 da/dN 与外推
tail = slice(max(first, len(h) - 40), len(h))
dN = N[tail][-1] - N[tail][0]
dd = dep[tail][-1] - dep[tail][0]
print()
if dN > 0 and dd > 0:
    rate = dd / dN
    remain = (100e-6 - dep[-1]) / rate
    print("最近 %.0f 周期长了 %.3f µm -> da/dN = %.3e m/cycle" % (dN, dd * 1e6, rate))
    print("按此速率长到 100 µm 还需约 %.3g 周期（N_max = %.0g）"
          % (remain, 1e8))
else:
    print("最近一段深度没有增长（dN = %.0f，dd = %.3g µm）-> 可能已止裂" % (dN, dd * 1e6))
