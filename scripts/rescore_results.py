"""按当前判据重算已有 result.json 的 ``peak`` 块（不重跑算例）。

为什么需要
----------
``peak`` 完全是从 ``curve`` 派生的（峰值位置、可信性、峰后下降步数……），
而 ``curve`` 才是算出来的数据。判据一改（例如 run_case.assess_peak 从
「看运行状态」改成「看曲线本身」），旧的 result.json 里的 ``peak`` 就成了
陈旧元数据 —— 但没必要为此重跑几十分钟的求解。

本脚本对每个 result.json 用 ``run_case.assess_peak`` 重算并就地写回，
只动 ``peak``，``curve`` 与其它字段原样保留。

用法：
    python scripts/rescore_results.py                # 重算 data/ 下全部
    python scripts/rescore_results.py data/xxx       # 只重算指定目录
    python scripts/rescore_results.py --dry-run      # 只报告差异，不写盘
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pfwire.run_case import assess_peak  # noqa: E402


def rescore(path: pathlib.Path, *, dry_run: bool) -> bool:
    """重算一个 result.json。返回 ``peak`` 是否发生变化。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    curve = data.get("curve") or []
    if not curve:
        print("%-34s 无 curve，跳过" % path.parent.name)
        return False

    forces = [c["F"] for c in curve]
    a = assess_peak(forces, status=data.get("status", "completed"))
    thickness = data.get("thickness", 1.0)
    old = data.get("peak", {})
    new = {
        "reliable": a.reliable,
        "note": a.note,
        "n_descending_after": a.n_descending,
        "post_peak_drop_rel": a.post_peak_drop_rel,
        "last_rise_before_peak": a.last_rise,
        "F": float(forces[a.index]),
        "F_per_thickness": float(forces[a.index] / thickness),
        "u": float(curve[a.index]["u"]),
        "step": curve[a.index]["step"],
    }
    changed = any(old.get(k) != v for k, v in new.items())
    flag = "变化" if changed else "不变"
    print(
        "%-34s %s  F=%.3f N  可信=%s->%s  峰后降 %d 步"
        % (
            path.parent.name,
            flag,
            new["F"],
            old.get("reliable"),
            new["reliable"],
            a.n_descending,
        )
    )
    if changed and not dry_run:
        data["peak"] = new
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="*", help="算例输出目录（默认 data/ 下全部）")
    ap.add_argument("--dry-run", action="store_true", help="只报告差异，不写盘")
    args = ap.parse_args(argv)

    if args.dirs:
        paths = [pathlib.Path(d) / "result.json" for d in args.dirs]
    else:
        paths = sorted((ROOT / "data").glob("*/result.json"))
    if not paths:
        raise SystemExit("没找到任何 result.json")

    n_changed = sum(rescore(p, dry_run=args.dry_run) for p in paths if p.exists())
    print()
    print(
        "%d/%d 个 result.json 的 peak 需要更新%s"
        % (n_changed, len(paths), "（--dry-run，未写盘）" if args.dry_run else "，已写回")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
