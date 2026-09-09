"""验证 2 出图：CT 疲劳裂纹扩展 a(N) 与 da/dN-ΔK，对拍 Carrara 2020 Tab. 2。

PLAN Phase 1 验收要求「两张验证图入 paper/figures/」，本脚本产出第二张。

左图：a(N)。参考 N_u（原文 Tab. 2，ΔP = 20 N、a0/W = 0.22 时 7288 周期）画竖线。
右图：da/dN-ΔK 双对数。本文的采样点 + 中间三分之一窗口的回归线，
      叠加原文的 Paris 律 C = 2.01e-2、m = 3.42（mm/cycle-MPa√m 口径），
      并标出 K_IC = 3.699 MPa√m 与回归窗口。

单位：图里一律用**原文的口径** mm/cycle 与 MPa√m，这样 C 能直接对表；
result.json 里存的是 SI，故这里显式换算（见 paris.ParisFit 的说明）。

图内文字用英文（投稿是英文会议，且容器无 CJK 字体）。
用法：python scripts/plot_verification2.py [算例目录]
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "paper" / "figures"

#: 原文 Tab. 2（CT、a0/W = 0.22、ΔP = 20 N）。C 的口径是 mm/cycle-MPa√m。
REF_NU = 7288
REF_C_MM_MPA = 2.01e-2
REF_M = 3.42
REF_KIC_MPA = 3.699  # sqrt(E Gc)，与原文给的 3.69 一致


def main(argv: list[str]) -> int:
    case_dir = ROOT / (argv[1] if len(argv) > 1 else "data/benchmark_ct_fatigue")
    path = case_dir / "result.json"
    if not path.exists():
        raise SystemExit(
            "缺少 %s —— 先跑 python -m pfwire.run_case configs/benchmark_ct_fatigue.yaml"
            % path
        )
    r = json.loads(path.read_text(encoding="utf-8"))
    hist = r.get("history") or []
    if not hist:
        raise SystemExit("result.json 里没有 history")

    N = np.array([h["N"] for h in hist])
    a_mm = np.array([h["a"] for h in hist]) * 1e3
    a_over_W = np.array([h["a_over_W"] for h in hist])

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # ---------------- 左图：a(N) ----------------
    ax.plot(N, a_mm, "-", color="C0", lw=1.6, label="present")
    ax.axvline(REF_NU, color="k", ls="--", lw=1.2,
               label="Carrara et al. (2020): $N_u$ = %d" % REF_NU)
    ax.set_xlabel("cycles $N$")
    ax.set_ylabel("crack length $a$  (mm)")
    ax.set_title(r"(a) crack growth ($\Delta P$ = 20 N, $a_0/W$ = 0.22)",
                 fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    # 右轴标 a/W，便于看停止阈值
    ax2 = ax.twinx()
    ax2.set_ylim(np.array(ax.get_ylim()) / (1e3 * r["mesh"]["W"]))
    ax2.set_ylabel("$a/W$")

    # ---------------- 右图：da/dN-ΔK ----------------
    paris = r.get("paris", {})
    if paris.get("ok"):
        # SI -> 原文口径
        dadn = np.array(paris["da_dN"]) * 1e3            # m/cycle -> mm/cycle
        dK = np.array(paris["delta_K"]) / 1e6            # Pa√m -> MPa√m
        bx.loglog(dK, dadn, "o", ms=3.5, mfc="none", mew=1.0, color="C0",
                  label="present (sampled)")

        lo, hi = np.array(paris["fit_window"]) / 1e6
        bx.axvspan(lo, hi, color="C0", alpha=0.08,
                   label=r"fit window (central 1/3 of $\Delta K$)")

        # 本文回归线
        C_mm, m = paris["C_mm_MPa"], paris["m"]
        kk = np.logspace(np.log10(dK.min()), np.log10(dK.max()), 50)
        bx.loglog(kk, C_mm * kk**m, "-", color="C0", lw=1.6,
                  label="present fit: $m$ = %.2f, $C$ = %.3g" % (m, C_mm))

        # 原文 Paris 律
        bx.loglog(kk, REF_C_MM_MPA * kk**REF_M, "--", color="k", lw=1.3,
                  label="Carrara: $m$ = %.2f, $C$ = %.3g" % (REF_M, REF_C_MM_MPA))
    else:
        bx.text(0.5, 0.5, "Paris fit unavailable:\n%s" % paris.get("reason", "?"),
                ha="center", va="center", transform=bx.transAxes, fontsize=8)
        bx.set_xscale("log")
        bx.set_yscale("log")

    bx.axvline(REF_KIC_MPA, color="C3", ls=":", lw=1.2,
               label=r"$K_{IC}$ = %.2f MPa$\sqrt{\mathrm{m}}$" % REF_KIC_MPA)
    bx.set_xlabel(r"$\Delta K$  (MPa$\sqrt{\mathrm{m}}$)")
    bx.set_ylabel(r"$\mathrm{d}a/\mathrm{d}N$  (mm/cycle)")
    bx.set_title("(b) crack growth rate vs Paris law", fontsize=9)
    bx.grid(alpha=0.25, lw=0.5, which="both")
    bx.legend(fontsize=7, loc="lower right", frameon=False)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / ("verification2_ct.%s" % ext)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("已写 %s" % out.relative_to(ROOT))

    # 验收数字
    print()
    print("状态 = %s   jump 步数 = %d   N_final = %.6g   a/W = %.4f"
          % (r["status"], r["n_jumps"], r["N_final"], a_over_W[-1]))
    print("寿命：本文 N = %.6g   原文 N_u = %d   ->  比值 %.3f"
          % (r["N_final"], REF_NU, r["N_final"] / REF_NU))
    if paris.get("ok"):
        m, C_mm = paris["m"], paris["C_mm_MPa"]
        print("Paris m：本文 %.4g vs 原文 %.4g  ->  %+.1f%%（PLAN 口径「同量级」，"
              "D-10 定为 ±30%%：%s）"
              % (m, REF_M, 100.0 * (m - REF_M) / REF_M,
                 "通过" if abs(m - REF_M) / REF_M <= 0.30 else "未通过"))
        print("Paris C：本文 %.4g vs 原文 %.4g（mm/cycle-MPa√m）  ->  比值 %.3g"
              % (C_mm, REF_C_MM_MPA, C_mm / REF_C_MM_MPA))
        print("回归用点数 %d，R^2 = %.4f，ΔK 跨度 %.3g~%.3g MPa√m"
              % (paris["n_points_used"], paris["r_squared"],
                 paris["delta_K_range"][0] / 1e6, paris["delta_K_range"][1] / 1e6))
    else:
        print("Paris 未拟合：%s" % paris.get("reason", "?"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
