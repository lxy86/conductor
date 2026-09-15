"""一眼看清扫描进度：每个工况是没跑、在跑、跑完了，还是出了问题。

「别中间出问题了却不知道」—— 这个脚本就是为此存在的。任何时候跑它，
都能拿到当前真实状态，不依赖谁还盯着终端、也不依赖哪个会话还活着。
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]

#: 终态：跑到这些状态就不用再动了
TERMINAL = {"failed", "failed_unstable", "runout", "completed"}
#: 需要人看一眼的状态
NEEDS_ATTENTION = {"not_converged"}


def rows():
    for cfg in sorted((REPO / "configs" / "sweep").glob("*.yaml")):
        name = cfg.stem
        out = REPO / "data" / name
        res = out / "result.json"
        ck = out / "checkpoint.json"
        if not res.is_file():
            yield name, "未开始" if not ck.is_file() else "有断点未出结果", {}
            continue
        try:
            r = json.loads(res.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            yield name, "result.json 读不了：%s" % exc, {}
            continue
        age = time.time() - res.stat().st_mtime
        yield name, r.get("status", "?"), {
            "N": r.get("N_final"), "N_f": r.get("N_f"),
            "depth_um": (r["history"][-1]["depth"] * 1e6) if r.get("history") else None,
            "gate": (r.get("gate") or {}).get("verdict"),
            "turns": (r.get("gate") or {}).get("turns_toward_normal"),
            "hours": (r.get("timing_s") or {}).get("solve", 0) / 3600.0,
            "age_min": age / 60.0,
        }


def main() -> int:
    print("%-14s %-16s %10s %10s %8s %-14s %6s %8s"
          % ("工况", "状态", "N", "N_f", "深度µm", "关卡", "转向", "已跑h"))
    bad = 0
    for name, status, d in rows():
        if not d:
            print("%-14s %-16s" % (name, status))
            if status not in ("未开始",):
                bad += 1
            continue
        stale = "" if d["age_min"] < 30 else "  (result 已 %.0f min 未更新)" % d["age_min"]
        mark = ""
        if status in NEEDS_ATTENTION:
            mark, bad = "  <-- 需要看", bad + 1
        elif status == "running":
            mark = stale
            if d["age_min"] >= 30:
                bad += 1
        print("%-14s %-16s %10s %10s %8s %-14s %6s %8.2f%s%s"
              % (name, status,
                 "-" if d["N"] is None else "%.4g" % d["N"],
                 "-" if d["N_f"] is None else "%.4g" % d["N_f"],
                 "-" if d["depth_um"] is None else "%.1f" % d["depth_um"],
                 d["gate"] or "-",
                 "-" if d["turns"] is None else str(d["turns"]),
                 d["hours"], mark, stale))
    print()
    print("需要关注的工况数：%d" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
