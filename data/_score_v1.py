"""验证 1 评分：把四个 SENT 算例的峰值与 Miehe Fig.7 数字化值对表。

参考值来自 data/_digitize.py（η=1e-6 曲线，像素覆盖完整）：
  ell1 = 0.0150 mm -> 0.7045 kN;  ell2 = 0.0075 mm -> 0.7373 kN
PLAN 验收：峰值偏差 < 5%，且 h = ell/4 与 ell/8 的峰值差 < 3%。
"""

from __future__ import annotations

import json
import pathlib

REF = {15.0e-6: 704.5, 7.5e-6: 737.3}  # N（板厚 1 mm 已计入）

CASES = [
    ("benchmark_sent", "ell2, h=ell/2"),
    ("benchmark_sent_ell1", "ell1, h=ell/2"),
    ("benchmark_sent_h4", "ell1, h=ell/4"),
    ("benchmark_sent_h8", "ell1, h=ell/8"),
    ("benchmark_sent_ell2_h4", "ell2, h=ell/4"),
]


def descent_stats(forces):
    """峰后连续下降步数与相对跌幅（与 run_case.PEAK_MIN_* 判据一致）。"""
    k = max(range(len(forces)), key=lambda i: forces[i])
    n = 0
    for i in range(k + 1, len(forces)):
        if forces[i] >= forces[i - 1]:
            break
        n += 1
    drop = (forces[k] - min(forces[k + 1 :])) / forces[k] if k < len(forces) - 1 else 0.0
    rise = forces[k] - forces[k - 1] if k > 0 else 0.0
    return k, n, drop, rise


rows = []
for case, label in CASES:
    p = pathlib.Path("data") / case / "result.json"
    if not p.exists():
        print("%-24s 未运行" % case)
        continue
    r = json.loads(p.read_text(encoding="utf-8"))
    curve = r["curve"]
    forces = [c["F"] for c in curve]
    k, n_desc, drop, rise = descent_stats(forces)
    ell = r["model"]["ell"]
    h = r["mesh"]["h_fine"]
    ref = REF[min(REF, key=lambda e: abs(e - ell))]
    ok = n_desc >= 3 or drop >= 1.0e-2
    rows.append(
        dict(
            case=case,
            label=label,
            ell=ell,
            h_over_ell=h / ell,
            n_cells=r["mesh"]["n_cells"],
            F=forces[k],
            u=curve[k]["u"],
            ref=ref,
            dev=100.0 * (forces[k] - ref) / ref,
            status=r["status"],
            reliable=ok,
            n_desc=n_desc,
            drop=drop,
            rise=rise,
            n_steps=r["n_steps"],
            t=r["timing_s"]["solve"],
        )
    )

hdr = (
    "%-14s %-15s %8s %7s %9s %8s %8s %8s  %-8s %s"
    % ("算例", "配置", "h/ell", "单元数", "F_peak(N)", "参考(N)", "偏差%", "峰后降", "峰值可信", "status")
)
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(
        "%-14s %-15s %8.3f %7d %9.3f %8.1f %+8.2f %8d  %-8s %s"
        % (
            r["case"].replace("benchmark_sent", "bs"),
            r["label"],
            r["h_over_ell"],
            r["n_cells"],
            r["F"],
            r["ref"],
            r["dev"],
            r["n_desc"],
            "是" if r["reliable"] else "否(下界)",
            r["status"],
        )
    )

print()
by = {r["label"]: r for r in rows}
if "ell1, h=ell/4" in by and "ell1, h=ell/8" in by:
    f4, f8 = by["ell1, h=ell/4"]["F"], by["ell1, h=ell/8"]["F"]
    d = 100.0 * abs(f4 - f8) / f8
    print(
        "网格无关性 (ell1): h=ell/4 %.3f N vs h=ell/8 %.3f N -> 差 %.2f%%  [PLAN <3%%: %s]"
        % (f4, f8, d, "通过" if d < 3.0 else "未通过")
    )
for r in rows:
    if not r["reliable"]:
        print(
            "  注: %s 的峰值是下界（峰前最后一步增量 %+.4g N，峰后仅降 %d 步）"
            % (r["case"], r["rise"], r["n_desc"])
        )
