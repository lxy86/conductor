"""用当前的判据口径重算已有工况的关卡，不重跑算例。

关卡口径改过两次（OPEN-2 的表层窗口 -> 尖端方向；D-37 的基准角改固定深度），
每次都要把已算完的工况重新评一遍 —— 否则 result.json 里躺着的是旧口径的结论，
而它看起来和新口径的一模一样，没人分得出来。
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pfwire import loads as pfloads          # noqa: E402
from pfwire.fretting import _gate_report     # noqa: E402


def main() -> int:
    write = "--write" in sys.argv
    for res in sorted((REPO / "data").glob("*/result.json")):
        try:
            r = json.loads(res.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if r.get("case_type") != "fretting" or not r.get("history"):
            continue
        c = r["contact"]
        contact = pfloads.ContactLoads.from_peak_pressure(
            p0=c["p0"], mu=c["mu"], Q_over_mu_P=c["Q_over_muP"],
            E_star=c["E_star"], R=c["R"])
        fc = (r.get("config") or {}).get("failure", {}) if isinstance(
            r.get("config"), dict) else {}
        new = _gate_report(
            r["history"], contact,
            judge_depth_over_a=float(fc.get("judge_depth_over_a", 2.0)),
            initial_tilt_depth_over_a=float(fc.get("initial_tilt_depth_over_a", 0.5)),
        )
        old = r.get("gate", {})
        changed = (old.get("verdict") != new.get("verdict")
                   or old.get("initially_tilted") != new.get("initially_tilted")
                   or old.get("turns_toward_normal") != new.get("turns_toward_normal"))
        print("%-14s %-13s -> %-13s  倾斜 %s->%s  转向 %s->%s  基准角 %s  最斜 %s%s"
              % (res.parent.name,
                 old.get("verdict") or "(无)", new.get("verdict") or "(无)",
                 old.get("initially_tilted"), new.get("initially_tilted"),
                 old.get("turns_toward_normal"), new.get("turns_toward_normal"),
                 "-" if new.get("tip_direction_initial_deg") is None
                 else "%.1f" % new["tip_direction_initial_deg"],
                 "-" if new.get("tip_direction_max_deg") is None
                 else "%.1f@%.2fa" % (new["tip_direction_max_deg"],
                                      new["tip_direction_max_depth_over_a"]),
                 "   <-- 结论变了" if changed else ""))
        if write:
            r["gate"] = new
            tmp = res.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(res)
    if write:
        print("\n已写回 result.json")
    else:
        print("\n（只看不写。加 --write 写回）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
