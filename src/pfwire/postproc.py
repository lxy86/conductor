"""论文图表后处理（PLAN Phase 3 的验收命令：``python -m pfwire.postproc --all``）。

产出 PLAN 第 4 节列的图 4–8；图 2、3（两个验证）由 ``scripts/plot_verification*.py``
产出，``--all`` 会一并调用。图 1（问题示意）是手绘，不在本模块。

**图内文字一律英文**：投稿是英文会议，且容器里没有中文字体
（DejaVu Sans 无 CJK 字形，中文会渲染成豆腐块）。中文说明留在本文件与 DECISIONS.md。

**本模块只读 result.json / snapshots.npz，不重跑算例**：出图是后处理，
任何需要重算的东西都应该先落到 data/ 里再画，否则图与数据会悄悄脱钩。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT_DIR = ROOT / "paper" / "figures"

#: S–N 扫描的六个工况（PLAN 2.5）
SN_CASES = ((20, "sn_sigma20"), (27, "sn_sigma27"), (35, "sn_sigma35"),
            (45, "sn_sigma45"), (55, "sn_sigma55"), (65, "sn_sigma65"))
#: p0 参数影响（PLAN 2.6）。240 就是基准工况，没有单独目录。
P0_CASES = ((210, "p0_210"), (240, "sn_sigma35"), (255, "p0_255"))
#: 网格收敛三档（PLAN 2.7 / D-41）
MESH_CASES = ((4, "sn_sigma35"), (6, "mesh_l6"), (8, "mesh_l8"))
#: D-41 的 Richardson 外推值，图 8 画成参考线
RICHARDSON_NF = 71380.6

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load(case: str) -> dict | None:
    f = DATA / case / "result.json"
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def nucleation_N(r: dict) -> float | None:
    """首次出现与表面连通裂纹（d >= 0.95）的 N。高应力工况可能一直没有。"""
    return next((h["N"] for h in r["history"] if h["depth"] > 0), None)


def _save(fig, name: str) -> pathlib.Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print("  写出 %s" % path.relative_to(ROOT))
    return path


# --------------------------------------------------------------------------
# 图 4：d 场演化 4 帧
# --------------------------------------------------------------------------
def fig4_damage_frames(case: str = "fig_frames",
                       a_um: float = 11.62) -> pathlib.Path | None:
    """萌生 -> 倾斜扩展 -> 转向 -> 贯穿。

    前两帧来自稳定段（收敛解），后两帧来自失稳判别器推进的过程（phase = 2）——
    **转向与贯穿在收敛解里根本不存在**（D-32），这一点必须写进图注，
    否则读者会以为四帧是同一类解。
    """
    f = DATA / case / "snapshots.npz"
    if not f.is_file():
        print("  跳过图 4：缺 %s（需要 output.snapshot_every > 0 的算例）"
              % f.relative_to(ROOT))
        return None
    z = np.load(f)
    d = z["d"]
    x = z["x_coords"] * 1e6
    dep = z["depth_coords"] * 1e6
    phase = z["phase"]
    depth = z["depth"] * 1e6

    stable = np.nonzero(phase < 1.5)[0]
    run = np.nonzero(phase > 1.5)[0]
    picks: list[int] = []
    labels: list[str] = []
    if stable.size:
        cracked = stable[depth[stable] > 0]
        picks.append(int(cracked[0]) if cracked.size else int(stable[0]))
        picks.append(int(stable[-1]))
        # 标题要短：面板只有约 2.3 in 宽，长标签会横向串到隔壁面板上去
        labels += ["(a) nucleation", "(b) stage-I growth"]
    if run.size:
        # **按深度里程碑选帧，不按序号中点**，且标签只说捕获到的东西。
        # 判别器在深度越过 judge_depth（2a）就停，所以失控段只到约 2.1a ——
        # **完整的 mode I 转向发生在约 6a（D-32），不在这批帧里**。
        # 早先把 (c) 的目标设成 2a，结果它与最深帧撞在一起（都是 24.5 µm），
        # 而且标成 turning 名不副实。改成：(c) 取最斜处（约 1.5a），
        # (d) 取捕获到的最深处，两帧之间能看到回转的**开端**。
        deepest = int(run[np.argmax(depth[run])])
        target = 1.5 * a_um if a_um else depth[run].max() * 0.75
        mid = int(run[np.argmin(np.abs(depth[run] - target))])
        if mid == deepest and run.size > 1:      # 兜底：别选出两张一样的
            mid = int(run[np.argsort(depth[run])[-2]])
        picks += [mid, deepest]
        labels += ["(c) unstable, max tilt", "(d) unstable, turning"]
    if not picks:
        print("  跳过图 4：快照里没有可用帧")
        return None

    # **裁到裂纹所在的窗口，并用等比例坐标轴**。采样网格铺到 120 µm 深，
    # 而裂纹只走到 25 µm —— 不裁的话 80% 画面是空黑、裂纹挤成一条缝。
    # 等比例是硬要求：这张图讲的就是倾角与转向，横纵比例不等会让读者
    # 直接读到错误的角度。
    cracked_any = (d.max(axis=0) >= 0.5)
    if cracked_any.any():
        rows = np.nonzero(cracked_any.any(axis=1))[0]
        cols = np.nonzero(cracked_any.any(axis=0))[0]
        pad = 6.0
        y_hi = min(dep.max(), dep[rows.max()] + pad)
        x_lo = max(x.min(), x[cols.min()] - pad)
        x_hi = min(x.max(), x[cols.max()] + pad)
    else:
        y_hi, x_lo, x_hi = dep.max(), x.min(), x.max()

    fig, axes = plt.subplots(1, len(picks),
                             figsize=(2.3 * len(picks), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, i, lab in zip(axes, picks, labels):
        im = ax.pcolormesh(x, dep, d[i], vmin=0.0, vmax=1.0,
                           cmap="inferno", shading="auto", rasterized=True)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_hi, 0.0)          # 深度向下为正
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ [$\mu$m]")
        ax.set_title("%s\ndepth = %.1f $\\mu$m" % (lab, depth[i]), fontsize=8)
    axes[0].set_ylabel(r"depth [$\mu$m]")
    fig.colorbar(im, ax=axes.tolist(), label=r"damage $d$", shrink=0.85)
    fig.text(0.01, -0.04,
             "(c) and (d) are states traversed during unstable propagation; no "
             "equilibrium solution exists beyond (b). The full turn towards "
             "mode I occurs at about 6a, beyond the captured range "
             "(see DECISIONS D-32).", fontsize=6.5)
    return _save(fig, "fig4_damage_frames.png")


# --------------------------------------------------------------------------
# 图 5：基准工况 a–N 曲线 + ᾱ
# --------------------------------------------------------------------------
def fig5_crack_growth(case: str = "sn_sigma35") -> pathlib.Path | None:
    r = load(case)
    if r is None:
        print("  跳过图 5：缺 %s" % case)
        return None
    h = r["history"]
    N = np.array([x["N"] for x in h])
    a_over = np.array([x["depth_over_a"] for x in h])
    alpha = np.array([x["max_alpha_bar"] for x in h])
    nuc = nucleation_N(r)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(N, a_over, lw=1.4, color="C0")
    ax.set_xlabel("cycles $N$")
    ax.set_ylabel(r"crack depth / contact half-width $a$", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    if nuc is not None:
        ax.axvline(nuc, color="0.4", ls=":", lw=1.0)
        ax.annotate("nucleation\n$N$ = %.3g" % nuc, xy=(nuc, 0.05),
                    xytext=(nuc * 1.06, 0.35), fontsize=7, color="0.3")
    ax2 = ax.twinx()
    ax2.plot(N, alpha, lw=1.2, color="C3", ls="--")
    ax2.set_ylabel(r"max accumulated $\bar{\alpha}$ [J/m$^3$]", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax.set_title(r"Base case: $\sigma_a$ = 35 MPa, $p_0$ = 240 MPa", fontsize=9)
    return _save(fig, "fig5_crack_growth.png")


# --------------------------------------------------------------------------
# 图 6：S–N 曲线 + CIGRE 安全边界线
# --------------------------------------------------------------------------
def fig6_sn_curve(conductor: str = "configs/conductor_jl1_lha1_465_210.yaml"
                  ) -> pathlib.Path | None:
    pairs = [(s, load(c)) for s, c in SN_CASES]
    pts = [(s, r["N_f"]) for s, r in pairs if r and r.get("N_f")]
    if len(pts) < 3:
        print("  跳过图 6：有效工况不足（%d 个）" % len(pts))
        return None
    sig = np.array([p[0] for p in pts], dtype=float)
    nf = np.array([p[1] for p in pts], dtype=float)
    m, c = np.polyfit(np.log10(sig), np.log10(nf), 1)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.loglog(nf, sig, "o", color="C0", ms=5, label="phase-field (this work)")
    grid = np.logspace(np.log10(nf.min()) - 0.2, np.log10(nf.max()) + 0.2, 80)
    ax.loglog(grid, 10.0 ** ((np.log10(grid) - c) / m), "-", color="C0", lw=1.0,
              label=r"fit: $N_f \propto \sigma_a^{%.2f}$" % m)

    try:
        from . import loads as pfloads
        cond = pfloads.load_conductor(ROOT / conductor)
        ncig = np.logspace(4, 8, 200)
        ax.loglog(ncig, np.asarray(cond.safe_border_stress(ncig)) / 1e6,
                  "--", color="C3", lw=1.2,
                  label="CIGRE CSBL (multilayer)$^{*}$")
    except Exception as exc:                       # 导线卡缺系数等
        print("  图 6：CIGRE 边界线未画（%s）" % str(exc)[:70])

    ax.set_xlabel("cycles to failure $N_f$")
    ax.set_ylabel(r"bulk stress amplitude $\sigma_a$ [MPa]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right")
    # D-25：TB 332 核对未做，且分段点处有 −3.94% 跳变。图注必须标明，
    # 且**不得**为了好看去平滑它 —— 那等于拿自己编的曲线跟文献比。
    fig.text(0.01, -0.03,
             "$^{*}$ CSBL coefficients taken from PLAN 1; not yet verified "
             "against CIGRE TB 332 (DECISIONS D-25). The -3.94% jump at the "
             "segment point is reproduced as specified, not smoothed.",
             fontsize=6.5)
    return _save(fig, "fig6_sn_curve.png")


# --------------------------------------------------------------------------
# 图 7：p0 影响
# --------------------------------------------------------------------------
def fig7_p0_influence() -> pathlib.Path | None:
    rows = []
    for p0, case in P0_CASES:
        r = load(case)
        if r and r.get("N_f"):
            rows.append((p0, r["N_f"], (r.get("gate") or {}).get("x_init_over_a")))
    if len(rows) < 2:
        print("  跳过图 7：有效工况不足")
        return None
    p0 = np.array([r[0] for r in rows], dtype=float)
    nf = np.array([r[1] for r in rows], dtype=float)
    xi = [r[2] for r in rows]
    n, _ = np.polyfit(np.log(p0), np.log(nf), 1)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax.semilogy(p0, nf, "s-", color="C1", ms=5)
    ax.set_xlabel(r"peak contact pressure $p_0$ [MPa]")
    ax.set_ylabel("cycles to failure $N_f$")
    ax.set_title(r"$N_f \propto p_0^{%.2f}$" % n, fontsize=9)
    ax.grid(True, alpha=0.25)

    ok = [(a, b) for a, b in zip(p0, xi) if b is not None]
    ax2.axhspan(0.632, 1.0, color="0.85", zorder=0, label="slip zone")
    if ok:
        ax2.plot([a for a, _ in ok], [b for _, b in ok], "^", color="C2", ms=7,
                 label="nucleation site")
    ax2.set_xlabel(r"peak contact pressure $p_0$ [MPa]")
    ax2.set_ylabel(r"nucleation site $|x| / a$")
    ax2.set_ylim(0.5, 1.3)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.25)
    return _save(fig, "fig7_p0_influence.png")


# --------------------------------------------------------------------------
# 图 8：网格收敛
# --------------------------------------------------------------------------
def fig8_mesh_convergence() -> pathlib.Path | None:
    rows = []
    for div, case in MESH_CASES:
        r = load(case)
        if r and r.get("N_f"):
            rows.append((div, r["N_f"], nucleation_N(r)))
    if len(rows) < 3:
        print("  跳过图 8：需要三档网格，现有 %d 档" % len(rows))
        return None
    divs = [r[0] for r in rows]
    h = np.array([2.0 / d for d in divs])          # µm；ell = 2 µm
    nf = np.array([r[1] for r in rows], dtype=float)
    nuc = np.array([r[2] if r[2] else np.nan for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(h, nf, "o-", color="C0", ms=6, label=r"$N_f$")
    ax.plot(h, nuc, "s--", color="C2", ms=5, label="nucleation life")
    ax.axhline(RICHARDSON_NF, color="0.4", ls=":", lw=1.2,
               label="Richardson extrapolation")
    for xi, yi, div in zip(h, nf, divs):
        ax.annotate(r"$\ell/%d$" % div, xy=(xi, yi), xytext=(4, 6),
                    textcoords="offset points", fontsize=7)
    ax.set_xlabel(r"element size $h$ [$\mu$m]")
    ax.set_ylabel("cycles")
    ax.set_xlim(0.0, 0.58)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center right")
    ax.set_title("Mesh convergence (oscillatory, $p$ = 3.07, GCI = 3.9%)",
                 fontsize=9)
    return _save(fig, "fig8_mesh_convergence.png")


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="论文图表后处理")
    ap.add_argument("--all", action="store_true",
                    help="产出全部图（含两个验证图）")
    ap.add_argument("--figures", nargs="*", type=int, default=None,
                    help="只出指定编号，如 --figures 5 6")
    args = ap.parse_args(argv)

    table = {4: fig4_damage_frames, 5: fig5_crack_growth,
             6: fig6_sn_curve, 7: fig7_p0_influence, 8: fig8_mesh_convergence}
    want = sorted(table) if (args.all or not args.figures) else args.figures

    print("出图 -> %s" % OUT_DIR.relative_to(ROOT))
    made = 0
    for num in want:
        fn = table.get(num)
        if fn is None:
            print("  未知图号 %r" % num)
            continue
        if fn() is not None:
            made += 1

    if args.all:
        for script in ("plot_verification1.py", "plot_verification2.py"):
            path = ROOT / "scripts" / script
            if not path.is_file():
                continue
            print("  调用 %s" % script)
            subprocess.run([sys.executable, str(path)], check=False)

    print("完成：%d 张" % made)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
