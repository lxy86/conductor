"""把运行中的日志转成 result.json 形状，用于中途出图 / 验证作图脚本。

正在跑的那个进程用的是加周期性落盘**之前**的代码，只会在结束时写 result.json，
所以中途只能从日志重建。日志每 10 个 jump 步一条，够画 a(N) 与 da/dN。
重建出来的 status 标成 "running"，并带 note，免得被当成定稿。

用法：python data/_log_to_result.py /root/ct_prod.log data/_ct_partial
"""

import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, "src")
from pfwire import paris  # noqa: E402

log_path = sys.argv[1] if len(sys.argv) > 1 else "/root/ct_prod.log"
out_dir = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "data/_ct_partial")

W, T, DELTA_P, DELTA_A = 50.0e-3, 1.0e-3, 20.0, 0.25e-3

pat = re.compile(
    r"N=([\d.eE+-]+)\s+ΔN=([\d.eE+-]+)\s+a=([\d.]+) mm \(a/W=([\d.]+)\)\s+"
    r"max_d=([\d.]+)\s+ᾱ_max=([\d.eE+-]+)"
)
history = []
for line in open(log_path, encoding="utf-8", errors="replace"):
    m = pat.search(line)
    if not m:
        continue
    a = float(m.group(3)) * 1e-3
    history.append(
        {
            "N": float(m.group(1)),
            "delta_N": float(m.group(2)),
            "a": a,
            "a_over_W": a / W,
            "crack_extension": a - 0.22 * W,
            "max_d": float(m.group(5)),
            "max_alpha_bar": float(m.group(6)),
            "max_dd": float("nan"),
            "rejections": 0,
            "n_surface_components": 1,
        }
    )
if len(history) < 2:
    raise SystemExit("日志里点太少（%d）" % len(history))

a = np.array([h["a"] for h in history])
N = np.array([h["N"] for h in history])
try:
    fit = paris.paris_fit(a, N, W=W, T=T, delta_P=DELTA_P, delta_a=DELTA_A)
    paris_block = {
        "ok": True,
        "delta_a": DELTA_A,
        "C": fit.C,
        "m": fit.m,
        "C_mm_MPa": fit.C_mm_MPa,
        "n_points_used": fit.n_points_used,
        "delta_K_range": list(fit.delta_K_range),
        "fit_window": list(fit.fit_window),
        "r_squared": fit.r_squared,
        "da_dN": fit.rate.da_dN.tolist(),
        "delta_K": fit.rate.delta_K.tolist(),
        "a_mid": fit.rate.a_mid.tolist(),
    }
except ValueError as exc:
    paris_block = {"ok": False, "reason": str(exc)}

result = {
    "case": "benchmark_ct_fatigue.yaml",
    "case_type": "fatigue",
    "status": "running",
    "note": "由日志重建的运行中快照，**不是最终结果**；日志每 10 个 jump 步一条，"
            "故 a(N) 的时间分辨率低于真实 history。",
    "material": "Carrara 2020 CT benchmark",
    "mesh": {"type": "ct", "n_cells": 32999, "W": W},
    "thickness": T,
    "loading": {"delta_P": DELTA_P, "P_min": 5.0},
    "n_jumps": len(history),
    "N_final": float(N[-1]),
    "timing_s": {"mesh": 0.0, "solve": 0.0},
    "history": history,
    "paris": paris_block,
}
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "result.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("已写 %s/result.json（%d 个点，N 到 %.6g，a/W 到 %.4f）"
      % (out_dir, len(history), N[-1], a[-1] / W))
if paris_block["ok"]:
    print("Paris: m = %.4g, C_mm_MPa = %.4g, R^2 = %.4f, 用 %d 点"
          % (paris_block["m"], paris_block["C_mm_MPa"],
             paris_block["r_squared"], paris_block["n_points_used"]))
