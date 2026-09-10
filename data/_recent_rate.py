"""看最近一段 a(N) 的局部速率，判断裂纹是真的减速还是采样/停滞假象。"""

import re
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/root/ct_prod.log"
pat = re.compile(r"N=([\d.eE+-]+)\s+ΔN=(\S+)\s+a=([\d.]+) mm")
N, a, dN = [], [], []
for line in open(path, encoding="utf-8", errors="replace"):
    m = pat.search(line)
    if m:
        N.append(float(m.group(1)))
        dN.append(m.group(2))
        a.append(float(m.group(3)))
N = np.array(N)
a = np.array(a)
print("共 %d 个记录点（每 10 个 jump 步一条）" % N.size)
print("最近 25 条：")
print("      N        a(mm)     Δa(mm)   ΔN    局部 da/dN (mm/cycle)")
for i in range(max(1, N.size - 25), N.size):
    da = a[i] - a[i - 1]
    dn = N[i] - N[i - 1]
    print("  %8.1f  %9.4f  %+8.4f  %5.1f   %s"
          % (N[i], a[i], da, dn, ("%.3e" % (da / dn)) if dn > 0 else "-"))

# 分段平均速率，抹掉量化抖动
print()
print("分段平均（每 500 周期）：")
edges = np.arange(N[0], N[-1] + 500, 500)
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (N >= lo) & (N <= hi)
    if sel.sum() < 2:
        continue
    da = a[sel][-1] - a[sel][0]
    dn = N[sel][-1] - N[sel][0]
    print("  N %6.0f~%6.0f: a %7.4f -> %7.4f mm，Δa=%6.4f，平均 da/dN=%.3e mm/cycle"
          % (lo, hi, a[sel][0], a[sel][-1], da, da / dn if dn > 0 else np.nan))
