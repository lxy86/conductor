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

#: 「多久没更新算停滞」。不能取得太紧：max_iter 提到 2000 后（D-35），
#: 一次被拒的 ΔN 尝试就要约 23 min（2000 次交错迭代 × 约 0.7 s），
#: 而临近失稳时连拒三四次是常态 —— 30 min 的阈值会把正常行为报成卡死，
#: 假警报多了就没人看了。取 90 min：既容得下连拒，又不至于真卡住时发现太晚。
STALL_MIN = 90.0


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
        # **活没活着要看最近一次「任何动静」，不能只看 result.json。**
        # result.json 每 dump_every(=25) 个 history 条目才写一次，而临近失稳时
        # 单条要几分钟，写入间隔会被拉到好几小时 —— 只看它就会把一个健康的
        # 长算例报成「可能卡住」（实测：result 6.3 h 没写，run.log 18 min 前
        # 还在更新）。误报一次就够让人去重启一个跑了 16 h 的算例。
        log = REPO / "data" / ("%s_run.log" % name)
        newest = res.stat().st_mtime
        if log.is_file():
            newest = max(newest, log.stat().st_mtime)
        age = time.time() - newest
        yield name, r.get("status", "?"), {
            "N": r.get("N_final"), "N_f": r.get("N_f"),
            "depth_um": (r["history"][-1]["depth"] * 1e6) if r.get("history") else None,
            "gate": (r.get("gate") or {}).get("verdict"),
            "turns": (r.get("gate") or {}).get("turns_toward_normal"),
            "hours": (r.get("timing_s") or {}).get("solve", 0) / 3600.0,
            "result_age_min": (time.time() - res.stat().st_mtime) / 60.0,
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
        # 「多久没更新」只对**还在跑**的工况有意义。已完成的工况本来就不再写
        # result.json，给它们打同样的警告只会淹没真正那条（本工具初版就是
        # 给 5 个已完成工况全打上「356 min 未更新」，噪声盖过信号）。
        stale = ""
        mark = ""
        if status in NEEDS_ATTENTION:
            mark, bad = "  <-- 需要看", bad + 1
        elif status == "running" and d["age_min"] >= STALL_MIN:
            stale = ("  <-- 日志已 %.0f min 没动静（阈值 %.0f），可能卡住"
                     % (d["age_min"], STALL_MIN))
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
