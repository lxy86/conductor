"""验证 1 出图：SENT 载荷–位移曲线 vs Miehe et al. 2010 Fig. 7 + 网格收敛。

PLAN Phase 1 验收要求「两张验证图入 paper/figures/」，本脚本产出第一张。

左图：两个内尺度 ℓ 在**收敛网格**（h = ℓ/4）上的 F–u 曲线，叠加原文数字化曲线。
右图：ℓ₁ 的 h 收敛（h = ℓ/2, ℓ/4, ℓ/8）—— h = ℓ/2 是原文规定的上限，
      裂尖每个 ℓ 只有约 2 个单元，欠解析会**高估**峰值；细化后单调收到原文值。

原文曲线来自 data/_digitize.py 对 Fig. 7 的程序化数字化（需先跑一次该脚本）。
只叠加 η = 1e-6 那条：η = 0 的红线在上升段被绿线覆盖，像素不完整（见该脚本注释）。

图内文字一律用英文：投稿是英文会议，且容器里没有中文字体（DejaVu Sans 无 CJK
字形，用中文会渲染成豆腐块）。中文说明留在本文件与 DECISIONS.md 里。

用法：python scripts/plot_verification1.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "paper" / "figures"
REF_JSON = ROOT / "data" / "_miehe_fig7.json"

#: 左图用的算例：收敛网格 h = ℓ/4。键是 data/ 下的目录名。
MAIN_CASES = [
    ("benchmark_sent_h4", 15.0, "C0", "o"),
    ("benchmark_sent_ell2_h4", 7.5, "C3", "s"),
]
#: 右图的 h 收敛序列（均为 ℓ₁ = 15 µm）。
CONVERGENCE = [
    ("benchmark_sent_ell1", 0.5),
    ("benchmark_sent_h4", 0.25),
    ("benchmark_sent_h8", 0.125),
]
#: 原文数字化曲线在 _miehe_fig7.json 里的键。
REF_KEYS = {15.0: "(a) ell=0.0150 eta=1e-6", 7.5: "(b) ell=0.0075 eta=1e-6"}


def load_result(case: str) -> dict:
    path = ROOT / "data" / case / "result.json"
    if not path.exists():
        raise SystemExit(
            "缺少 %s —— 先跑 python -m pfwire.run_case configs/%s.yaml" % (path, case)
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    if not REF_JSON.exists():
        raise SystemExit(
            "缺少 %s —— 先跑 python data/_digitize.py（需要 miehe2010.pdf 在仓库根目录）"
            % REF_JSON
        )
    ref = json.loads(REF_JSON.read_text(encoding="utf-8"))

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # ---------------- 左图：F–u 曲线对比 ----------------
    for case, ell_um, color, marker in MAIN_CASES:
        r = load_result(case)
        u = [c["u"] * 1e6 for c in r["curve"]]  # m -> µm
        f = [c["F"] * 1e-3 for c in r["curve"]]  # N -> kN
        pk = r["peak"]
        ax.plot(
            u,
            f,
            color=color,
            lw=1.6,
            label=r"present, $\ell$ = %.4g $\mu$m ($h=\ell/4$)" % ell_um,
        )
        ax.plot(
            pk["u"] * 1e6,
            pk["F"] * 1e-3,
            marker=marker,
            color=color,
            ms=6,
            mfc="none",
            mew=1.6,
        )

        key = REF_KEYS[ell_um]
        ax.plot(
            [x * 1e3 for x in ref[key]["u_mm"]],  # mm -> µm
            ref[key]["F_kN"],
            ls="--",
            color=color,
            lw=1.2,
            alpha=0.75,
            label=r"Miehe et al. (2010), $\ell$ = %.4g $\mu$m" % ell_um,
        )

    ax.set_xlabel(r"prescribed displacement $u$  ($\mu$m)")
    ax.set_ylabel(r"reaction force $F$  (kN)")
    ax.set_xlim(0.0, 6.6)
    ax.set_ylim(0.0, 0.85)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.set_title("(a) load-displacement response", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)

    # ---------------- 右图：h 收敛 ----------------
    ref_peak = ref[REF_KEYS[15.0]]["F_peak_kN"]
    xs, ys, cells = [], [], []
    for case, h_over_ell in CONVERGENCE:
        r = load_result(case)
        xs.append(h_over_ell)
        ys.append(r["peak"]["F"] * 1e-3)
        cells.append(r["mesh"]["n_cells"])

    bx.axhline(
        ref_peak,
        color="k",
        ls="--",
        lw=1.2,
        label="Miehe et al. (2010): %.4f kN" % ref_peak,
    )
    bx.plot(xs, ys, "o-", color="C0", lw=1.6, ms=6, label=r"present ($\ell$ = 15 $\mu$m)")
    for i, (x, y, n) in enumerate(zip(xs, ys, cells)):
        # 最后一点贴着右边框，标注要往左放，否则被裁掉
        last = i == len(xs) - 1
        bx.annotate(
            "%.4f kN\n%+.2f%%\n%d cells" % (y, 100.0 * (y - ref_peak) / ref_peak, n),
            (x, y),
            textcoords="offset points",
            xytext=(-8, 8) if last else (8, -2),
            ha="right" if last else "left",
            va="bottom" if last else "top",
            fontsize=7,
        )
    bx.set_xscale("log", base=2)
    bx.set_xticks(xs)
    bx.set_xticklabels([r"$\ell/2$", r"$\ell/4$", r"$\ell/8$"])
    bx.set_xlabel(r"crack-tip element size $h$")
    bx.set_ylabel(r"peak load $F_{\max}$  (kN)")
    bx.set_xlim(0.60, 0.10)  # 从粗到细，向右细化
    bx.legend(fontsize=7.5, loc="upper right", frameon=False)
    bx.set_title(r"(b) mesh convergence ($\ell$ = 15 $\mu$m)", fontsize=9)
    bx.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUT_DIR / ("verification1_sent.%s" % ext)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("已写 %s" % path.relative_to(ROOT))

    # 顺手把验收数字打出来，避免图与结论脱节
    print()
    f4, f8 = ys[1], ys[2]
    print(
        "网格无关性：h=ℓ/4 %.4f kN vs h=ℓ/8 %.4f kN -> 差 %.2f%%（PLAN <3%%）"
        % (f4, f8, 100.0 * abs(f4 - f8) / f8)
    )
    for case, ell_um, _, _ in MAIN_CASES:
        r = load_result(case)
        rp = ref[REF_KEYS[ell_um]]["F_peak_kN"]
        got = r["peak"]["F"] * 1e-3
        print(
            "ℓ=%.4g µm：本文 %.4f kN vs 原文 %.4f kN -> %+.2f%%（PLAN <5%%）%s"
            % (
                ell_um,
                got,
                rp,
                100.0 * (got - rp) / rp,
                ""
                if r["peak"]["reliable"]
                else "  [峰值为下界，峰前最后一步增量 %+.4g N]"
                % r["peak"]["last_rise_before_peak"],
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
