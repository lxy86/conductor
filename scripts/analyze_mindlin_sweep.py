"""6.6.1：Mindlin 摩擦验证的 Q/(muP) 扫描区间 c/a 定量分析（D-68 问题 22）。

**为什么补这个**：PLAN 6.6.1 要求以 Mindlin 部分滑移作为 Coulomb 摩擦的
独立验证点（c/a = sqrt(1 - Q/(mu P))）。`scripts/verify_contact_mindlin.py`
已经跑过整个部分滑移区间（实测 Q/(muP) = 0.048 ~ 0.983，28 步）且 10 条
判据全过，但 c/a 在那里只是**打印的参考列**，没有跨区间的定量度量。
本脚本补上，并把口径与 D-51 的降级决定对齐。

**口径（D-51，不得违反）**：Cattaneo-Mindlin 是**半空间**解。有限尺寸圆柱的
顶面剪力与接触面反力构成力偶，须由法向力偏心 e = Q h / P 平衡，R/a 约 8 时
e 远大于 a，q(x) 本就不对称 —— FE 与半空间解的系统偏差是**物理差别**，
不是实现错误。故本分析**只作参考测量，不设通过判据**（no pass/fail gate）。
要回答的是三件事：偏差有多大、随 Q/(muP) 怎么变、方向是否与解析一致。

**与 6.6.1 罚函数实现的关系**：罚摩擦实测只能到 Q/(muP) ≈ 0.10（详见
DECISIONS 问题 22 的墙 3），扫不过本分析所用的区间；Nitsche 路径能到 0.98。
这个对比本身就是「独立验证走 Nitsche」这条决策的定量依据。

用法::

    python -u scripts/verify_contact_mindlin.py            # 先跑 FE
    python -u scripts/analyze_mindlin_sweep.py             # 再对 result.json 重判
    python -u scripts/analyze_mindlin_sweep.py --q-min 0.1 --q-max 0.9

结果写回 `data/verify_mindlin/result.json` 的 ``mindlin_sweep`` 段。
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]


def mindlin_c_over_a(q_over_mu_p: float) -> float:
    """Cattaneo-Mindlin 黏着半宽比 c/a = sqrt(1 - Q/(mu P))。"""
    if not 0.0 <= q_over_mu_p <= 1.0:
        return float("nan")
    return math.sqrt(1.0 - q_over_mu_p)


def analyze_sweep(rows: list[dict], *, q_min: float = 0.35,
                  q_max: float = 0.95, min_points: int = 5,
                  slip_key: str = "stick_half") -> dict:
    """跨 Q/(muP) 区间分析黏着半宽比 c/a 的系统偏差与趋势。

    参数
    ----
    rows : `verify_contact_mindlin.analyze_step` 产出的逐步诊断，
        每项含 ``q_ratio`` / ``c_over_a_fe`` / ``c_over_a_mindlin``。
    q_min, q_max : 只在这个 Q/(muP) 区间内做分析。
        下限缺省 **0.35**（不是 0.05）：黏着区外沿受接触斑边界与网格量化
        污染，低 Q 时「黏着」其实是「还没滑到那边」，量不到真实的 c。
        上限排除 Q/(muP) -> 1（整体滑移，Mindlin 本身失效）。
    min_points : 区间内至少这么多点才给统计量，否则只报原始值加注。

    ⚠ **被比较的两个量不是同一个几何量**（这是本分析的关键口径）：
      - FE 侧用 D-52 的**微滑幅**判据量出的 ``stick_half`` = (hi − lo)/2，
        即黏着区的**半宽**；测的是位移（FE 的基本未知量）。
      - Mindlin 侧的 c 是黏着区的**外半径**。
    两者只在对心单连通黏着区且 lo ≈ 0 时才可能重合，故这里的「差」是
    **形状与尺度的一致性检查**，不是逐点误差。要看的是：差随 Q/(muP)
    怎么变、方向是否一致、量级是否在网格分辨率之内。

    返回的字典直接进 result.json 的 ``mindlin_sweep`` 段。
    """
    pts = []
    for r in rows:
        q = r.get("q_ratio")
        a = r.get("a_hertz")
        half = r.get(slip_key)
        if not (math.isfinite(q) and math.isfinite(a) and math.isfinite(half)):
            continue
        if not (q_min <= q <= q_max):
            continue
        if r.get("n_stick", 0) <= 0:
            continue                      # 没有黏着区就没得量
        fe = half / a
        ref = mindlin_c_over_a(q)
        pts.append({"step": r["step"], "q_ratio": q, "c_over_a_fe": fe,
                    "c_over_a_mindlin": ref, "abs_err": abs(fe - ref)})
    out = {
        "q_window": [q_min, q_max],
        "fe_measure": slip_key,
        "caliber_note": ("FE 侧是黏着区半宽 (hi-lo)/2，Mindlin 侧是外半径 c；"
                         "两者非同一几何量，故为形状/尺度一致性检查而非逐点误差"),
        "n_points": len(pts),
        "per_step": pts,
    }
    if not pts:
        out["note"] = "区间内没有可用点（Q/(muP) 窗口可能太窄，或该工况没扫过）"
        return out

    fe = np.array([p["c_over_a_fe"] for p in pts])
    ref = np.array([p["c_over_a_mindlin"] for p in pts])
    q = np.array([p["q_ratio"] for p in pts])
    out.update({
        "c_over_a_fe_range": [float(fe.min()), float(fe.max())],
        "c_over_a_mindlin_range": [float(ref.min()), float(ref.max())],
        "abs_err_max": float(np.max(np.abs(fe - ref))),
        "abs_err_mean": float(np.mean(np.abs(fe - ref))),
        # >0 = FE 的黏着区比半空间解宽
        "signed_bias_mean": float(np.mean(fe - ref)),
        # 解析上 c/a 随 Q/(muP) 严格单调降，看 FE 是否同向
        "trend_monotone_nonincreasing": bool(np.all(np.diff(fe) <= 1.0e-9)),
        # Spearman 秩相关（不用 Pearson：c/a ~ sqrt(1-q) 明显非线性）
        "spearman_vs_q": _spearman(q, fe),
        "n_decreasing_steps": int(np.sum(np.diff(fe) < 0)),
        "n_total_steps": int(len(fe) - 1),
    })
    if len(pts) < min_points:
        out["note"] = "区间内只有 %d 个点（< %d），统计量仅供查看" % (len(pts), min_points)
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相关（自包含，不引 scipy）。"""
    if x.size < 3:
        return float("nan")

    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(a.size, dtype=float)
        r[order] = np.arange(a.size, dtype=float)
        return r

    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = math.sqrt(float(np.sum(rx * rx)) * float(np.sum(ry * ry)))
    return float(np.sum(rx * ry) / den) if den > 0 else float("nan")


def format_report(sw: dict) -> str:
    """给人看的一页摘要（写进日志，也便于贴进 DECISIONS）。"""
    L = ["Mindlin 部分滑移的 c/a 定量分析（Q/(muP) ∈ [%.2f, %.2f]，%d 点）"
         % (sw["q_window"][0], sw["q_window"][1], sw["n_points"]),
         "  FE 侧口径：%s（D-52 的微滑幅判据）" % sw["fe_measure"],
         "  参考口径：c/a = sqrt(1 - Q/(mu P))（Cattaneo-Mindlin 半空间解）",
         "  ⚠ 按 D-51 这是**参考测量，不设通过判据**：半空间解与有限尺寸圆柱的",
         "    系统偏差是物理差别（顶面剪力/反力偶须由法向偏心平衡），不是实现错误。",
         "  ⚠ %s" % sw["caliber_note"]]
    if not sw.get("per_step"):
        L.append("  （区间内无可用点）")
        return "\n".join(L)
    L.append("  %-4s %10s %14s %12s %10s"
             % ("步", "Q/(muP)", "c/a FE(半宽)", "c/a Mindlin", "|差|"))
    for p in sw["per_step"]:
        L.append("  %-4d %10.4f %14.4f %12.4f %10.4f"
                 % (p["step"], p["q_ratio"], p["c_over_a_fe"],
                    p["c_over_a_mindlin"], p["abs_err"]))
    L.append("  c/a FE 范围       [%.4f, %.4f]" % tuple(sw["c_over_a_fe_range"]))
    L.append("  c/a Mindlin 范围  [%.4f, %.4f]" % tuple(sw["c_over_a_mindlin_range"]))
    L.append("  |差| 最大 %.4f、均值 %.4f；带符号偏差均值 %+.4f（+ = FE 黏着区偏宽）"
             % (sw["abs_err_max"], sw["abs_err_mean"], sw["signed_bias_mean"]))
    L.append("  趋势：随 Q/(muP) 单调不增 = %s（降步 %d/%d），Spearman rho = %+.3f"
             % (sw["trend_monotone_nonincreasing"], sw["n_decreasing_steps"],
                sw["n_total_steps"], sw["spearman_vs_q"]))
    if "note" in sw:
        L.append("  注：%s" % sw["note"])
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mindlin c/a 的 Q/(muP) 扫描分析")
    ap.add_argument("--result", default="data/verify_mindlin/result.json")
    ap.add_argument("--q-min", type=float, default=0.35)
    ap.add_argument("--q-max", type=float, default=0.95)
    ap.add_argument("--slip-key", default="stick_half",
                    help="FE 侧的黏着区度量（stick_half = 半宽，D-52 的微滑幅判据）")
    args = ap.parse_args(argv)

    path = REPO / args.result
    data = json.loads(path.read_text(encoding="utf-8"))
    slip = data.get("slip") or []
    if not slip:
        print("result.json 里没有 slip 段（Phase 4.5 的微滑幅诊断）：%s" % path)
        print("先跑 `python -u scripts/verify_contact_mindlin.py` 生成它。")
        return 1
    # slip 段缺 a_hertz（analyze_step 才算），从 steps 段按步号并进来
    a_by_step = {r["step"]: r.get("a_hertz") for r in (data.get("steps") or [])}
    rows = [dict(s, a_hertz=a_by_step.get(s["step"])) for s in slip]
    sw = analyze_sweep(rows, q_min=args.q_min, q_max=args.q_max, slip_key=args.slip_key)
    print(format_report(sw))
    data["mindlin_sweep"] = sw
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写回 %s 的 mindlin_sweep 段" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
