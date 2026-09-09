"""核对 R = 0.2 假设：用 P_max = 1.25 dP 重算 Tab. 2 的 Pmax/Pmon 全部 11 行。"""

P_MON = 186.4  # Fig. 12b 数字化
R = 0.2
rows = [(10, 6.7), (15, 10.1), (20, 13.5), (40, 26.9), (60, 40.4), (80, 53.8),
        (100, 67.3), (110, 74.0), (120, 80.8), (130, 87.5), (140, 94.2)]
print("假设 R = Pmin/Pmax = %.2f  =>  Pmax = dP/(1-R) = %.3f dP" % (R, 1 / (1 - R)))
print("Pmon = %.1f N（Fig. 12b 数字化）" % P_MON)
print()
print(" dP(N)  原文%%   本假设算出%%   差")
worst = 0.0
for dp, pct in rows:
    p_max = dp / (1.0 - R)
    got = 100.0 * p_max / P_MON
    worst = max(worst, abs(got - pct))
    print("%6.0f  %5.1f   %9.1f   %+5.2f" % (dp, pct, got, got - pct))
print()
print("最大偏差 %.2f 个百分点 -> %s" % (
    worst, "R = 0.2 与全部 11 行一致" if worst < 1.0 else "R = 0.2 不成立"))
print()
print("故参考算例：dP = 20 N -> Pmax = %.1f N、Pmin = %.1f N"
      % (20 / 0.8, 20 * 0.2 / 0.8))
