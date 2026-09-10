"""验证 2 出图：CT 疲劳裂纹扩展 a(N) 与 da/dN-ΔK，对拍 Carrara 2020 Tab. 2 / Fig. 13。

PLAN Phase 1 验收要求「两张验证图入 paper/figures/」，本脚本产出第二张。

**为什么要多条曲线叠加**：定几何、定 ΔP 时 ΔK 只随 a 变，而 Y(a/W) 在小 a/W 处
很平，单条曲线的 ΔK 跨度天生就窄（ΔP=20/a0=0.22 那条只有 1.41 倍，据此拟合出的
m 强烈依赖窗口）。原文 p26 说得明白：定 ΔK 下 da/dN 应与 ΔP、a0 无关，
Fig. 13 正是把不同 ΔP 与 a0/W 的多条曲线叠起来才铺开约一个数量级的 ΔK。
本脚本照做：每条曲线各自按原文口径拟合，同时画出叠加后的全貌。

左图：各算例的 a(N)，并标原文 Tab. 2 的 N_u。
右图：da/dN-ΔK 双对数叠加 + 各自的回归线 + 原文 Paris 律 + K_IC。

单位一律用原文口径 mm/cycle 与 MPa√m（result.json 里存的是 SI）。
图内文字用英文（投稿是英文会议，且容器无 CJK 字体）。

用法：python scripts/plot_verification2.py [算例目录 ...]
不给参数时用下面 CASES 里存在的那些。
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

#: 默认算例：(目录, 标签, 原文 Tab.2 的 N_u, 原文 C, 原文 m, 颜色)
#: C/m 为 None 表示原文该行没给（Pmax/Pmon 太高、无稳定扩展段）。
CASES = [
    ("data/_ct_h2_norelax", r"$a_0/W$=0.22, $\Delta P$=20 N", 7288, 2.01e-2, 3.42, "C0"),
    ("data/ct_a040_dp20", r"$a_0/W$=0.40, $\Delta P$=20 N", 1483, 1.85e-2, 3.61, "C2"),
    ("data/ct_a050_dp20", r"$a_0/W$=0.50, $\Delta P$=20 N", 512, 1.57e-2, 3.83, "C4"),
    ("data/ct_a022_dp40", r"$a_0/W$=0.22, $\Delta P$=40 N", 817, 1.60e-2, 3.73, "C3"),
]
REF_KIC_MPA = 3.699
#: 原文 CT+TPB 全部算例的平均（p27）
REF_C_MEAN, REF_M_MEAN = 1.78e-2, 3.50


def load(case_dir: pathlib.Path) -> dict | None:
    path = case_dir / "result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    wanted = argv[1:] if len(argv) > 1 else [c[0] for c in CASES]
    meta = {c[0]: c for c in CASES}
    loaded = []
    for d in wanted:
        r = load(ROOT / d)
        if r is None:
            print("跳过（无 result.json）：%s" % d)
            continue
        loaded.append((d, r, meta.get(d, (d, d, None, None, None, "C7"))))
    if not loaded:
        raise SystemExit("没有可用的算例结果")

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    rows = []
    for d, r, (_, label, nu_ref, c_ref, m_ref, color) in loaded:
        hist = r.get("history") or []
        if not hist:
            continue
        N = np.array([h["N"] for h in hist])
        a_mm = np.array([h["a"] for h in hist]) * 1e3
        ax.plot(N, a_mm, "-", color=color, lw=1.5, label=label)
        if nu_ref:
            ax.plot([nu_ref], [a_mm[0]], marker="v", color=color, ms=6,
                    mfc="none", mew=1.4)

        paris = r.get("paris", {})
        if paris.get("ok"):
            dadn = np.array(paris["da_dN"]) * 1e3
            dK = np.array(paris["delta_K"]) / 1e6
            bx.loglog(dK, dadn, "o", ms=3.2, mfc="none", mew=0.9, color=color,
                      label=label)
            kk = np.logspace(np.log10(dK.min()), np.log10(dK.max()), 40)
            bx.loglog(kk, paris["C_mm_MPa"] * kk ** paris["m"], "-",
                      color=color, lw=1.3, alpha=0.85)
        rows.append(
            dict(
                case=d.rsplit("/", 1)[-1],
                status=r["status"],
                N=r["N_final"],
                nu_ref=nu_ref,
                a_over_W=hist[-1]["a_over_W"],
                m=paris.get("m"),
                m_ref=m_ref,
                C=paris.get("C_mm_MPa"),
                C_ref=c_ref,
                r2=paris.get("r_squared"),
                npts=paris.get("n_points_used"),
                dk=paris.get("delta_K_range"),
            )
        )

    ax.set_xlabel("cycles $N$")
    ax.set_ylabel("crack length $a$  (mm)")
    ax.set_title("(a) crack growth", fontsize=9)
    ax.set_xscale("log")
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.plot([], [], "v", color="k", mfc="none", mew=1.4,
            label="Carrara $N_u$")
    ax.legend(fontsize=7, loc="upper left", frameon=False)

    # 原文 Paris 律（平均值那条）覆盖整个叠加后的 ΔK 跨度
    all_dk = [x for r in rows if r["dk"] for x in r["dk"]]
    if all_dk:
        lo, hi = min(all_dk) / 1e6, max(all_dk) / 1e6
        kk = np.logspace(np.log10(lo * 0.9), np.log10(hi * 1.1), 60)
        bx.loglog(kk, REF_C_MEAN * kk ** REF_M_MEAN, "--", color="k", lw=1.4,
                  label="Carrara mean: $m$=%.2f, $C$=%.3g" % (REF_M_MEAN, REF_C_MEAN))
    bx.axvline(REF_KIC_MPA, color="0.4", ls=":", lw=1.2,
               label=r"$K_{IC}$ = %.2f MPa$\sqrt{\mathrm{m}}$" % REF_KIC_MPA)
    bx.set_xlabel(r"$\Delta K$  (MPa$\sqrt{\mathrm{m}}$)")
    bx.set_ylabel(r"$\mathrm{d}a/\mathrm{d}N$  (mm/cycle)")
    bx.set_title("(b) crack growth rate vs Paris law", fontsize=9)
    bx.grid(alpha=0.25, lw=0.5, which="both")
    bx.legend(fontsize=6.5, loc="lower right", frameon=False)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / ("verification2_ct.%s" % ext)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("已写 %s" % out.relative_to(ROOT))

    # ---- 验收对表 ----
    print()
    hdr = "%-16s %-14s %9s %9s %7s %8s %8s %7s %6s" % (
        "算例", "status", "N", "原文N_u", "a/W", "m", "原文m", "C_mm", "R^2")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-16s %-14s %9.4g %9s %7.3f %8s %8s %8s %6s" % (
            r["case"], r["status"], r["N"],
            r["nu_ref"] if r["nu_ref"] else "-",
            r["a_over_W"],
            "%.3f" % r["m"] if r["m"] else "-",
            "%.2f" % r["m_ref"] if r["m_ref"] else "-",
            "%.4g" % r["C"] if r["C"] else "-",
            "%.3f" % r["r2"] if r["r2"] is not None else "-",
        ))
    if all_dk:
        print()
        print("叠加后的 ΔK 跨度：%.3f ~ %.3f MPa√m（%.2f 倍）"
              % (min(all_dk) / 1e6, max(all_dk) / 1e6, max(all_dk) / min(all_dk)))
    ms = [r["m"] for r in rows if r["m"]]
    if ms:
        print("各算例 m：%s  ->  均值 %.3f（原文 CT+TPB 平均 %.2f）"
              % (", ".join("%.3f" % x for x in ms), float(np.mean(ms)), REF_M_MEAN))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
