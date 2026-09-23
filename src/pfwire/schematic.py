"""论文图 1：从线夹口微动到 2D 模型与边界条件（PLAN Phase 3 图 1）。

三联图，从左到右是**尺度递降**的三步，读者顺着看就知道 2D 模型是从哪来的：

  (a) 悬垂线夹处的导线：微风振动 -> 线夹出口弯曲，P–S 测点在 89 mm 处；
  (b) 线夹口两根铝单丝的交叉接触：法向压紧 + 往复切向微滑；
  (c) 本模型的 2D 切片：Hertz 压力 p(x) 与 Cattaneo–Mindlin 切向面力 q(x)
      作为面力施加，本体应力 σ_a 加在右边界，底边 u_y = 0、左边 u_x = 0。

**这张图是示意，但所有标注的数值都来自配置与实算结果**，不是随手编的：
a、c/a 由 loads 从 (p0, R, E*) 算出（D-25：网格与面力必须同源），
W/H/加密区来自 configs/fretting_base.yaml。

图内文字一律英文（容器无 CJK 字体，中文会渲染成豆腐块）。
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "figures"

#: 与 configs/fretting_base.yaml / data/sn_sigma35 一致
A_UM = 11.62          # 接触半宽
C_OVER_A = 0.6325     # 黏着半宽比 = sqrt(1 - Q/(mu P))
P0_MPA = 240.0
MU = 0.7
SIGMA_A_MPA = 35.0
R_WIRE_MM = 1.875
W_MM, H_MM = 6.4, 3.2
PS_X_MM = 89.0

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _panel_a(ax) -> None:
    """(a) 悬垂线夹处的导线与微风振动。"""
    # 线夹
    ax.add_patch(mpatches.Rectangle((-1.0, 0.28), 1.0, 0.42,
                                    facecolor="0.75", edgecolor="0.3", lw=0.8))
    ax.text(-0.5, 0.49, "clamp", ha="center", va="center", fontsize=7.5)
    # 导线：从线夹出口伸出并弯曲
    x = np.linspace(0.0, 2.6, 200)
    for off, lw in ((0.0, 2.6), (0.0, 1.0)):
        ax.plot(x, 0.49 + off + 0.16 * np.sin(1.6 * x) * (x / 2.6) ** 1.6,
                color="C0", lw=lw, solid_capstyle="round")
    # 振动示意
    for s in (-1, 1):
        ax.annotate("", xy=(2.2, 0.49 + s * 0.30), xytext=(2.2, 0.49 + s * 0.10),
                    arrowprops=dict(arrowstyle="->", color="C3", lw=1.0))
    ax.text(2.35, 0.49, "aeolian\nvibration", fontsize=7, color="C3", va="center")
    # P–S 测点
    ax.annotate("", xy=(0.0, 0.16), xytext=(1.15, 0.16),
                arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.8))
    ax.text(0.58, 0.09, "$x$ = %.0f mm (P–S)" % PS_X_MM, ha="center", fontsize=7)
    ax.plot([0.0], [0.49], marker="o", ms=4, color="0.2")
    ax.text(0.06, 0.66, "last point\nof contact", fontsize=7, va="center")
    ax.set_xlim(-1.05, 3.3)
    ax.set_ylim(-0.30, 1.30)
    ax.axis("off")


def _panel_b(ax) -> None:
    """(b) 两根铝单丝的交叉接触。"""
    # 上丝（对偶）与下丝（建模对象）
    ax.add_patch(mpatches.Circle((0.0, 0.62), 0.30, facecolor="0.88",
                                 edgecolor="0.35", lw=0.9))
    ax.add_patch(mpatches.Circle((0.0, -0.02), 0.30, facecolor="C0", alpha=0.25,
                                 edgecolor="C0", lw=1.2))
    ax.text(0.40, 0.62, "counterpart wire\n(same alloy)", fontsize=7, va="center")
    ax.text(0.40, -0.02, "modelled wire\n$R$ = %.3f mm" % R_WIRE_MM,
            fontsize=7, va="center", color="C0")
    # 法向压紧
    ax.annotate("", xy=(0.0, 0.40), xytext=(0.0, 0.86),
                arrowprops=dict(arrowstyle="-|>", color="C3", lw=1.4))
    ax.text(-0.06, 0.88, "$P$", fontsize=8.5, color="C3", ha="right")
    # 往复切向
    ax.annotate("", xy=(0.30, 0.30), xytext=(-0.30, 0.30),
                arrowprops=dict(arrowstyle="<->", color="C2", lw=1.2))
    # 标签放圆外：居中会压在对偶丝上
    ax.text(-0.38, 0.30, "$\\pm Q$\n(micro-slip)", fontsize=7,
            color="C2", ha="right", va="center")
    ax.set_xlim(-1.30, 1.85)
    ax.set_ylim(-0.55, 1.30)
    ax.set_aspect("equal")
    ax.axis("off")


def _panel_c(ax) -> None:
    """(c) 2D 切片：面力、本体应力与边界条件。"""
    # 计算域（按真实长宽比 W:H = 2:1）
    ax.add_patch(mpatches.Rectangle((-W_MM / 2, -H_MM), W_MM, H_MM,
                                    facecolor="0.96", edgecolor="0.3", lw=1.0))
    # 接触区（放大画，真实 2a = 23 µm 在 6.4 mm 域上画不出来）。
    # **两条面力都画在表面之上**：它们是施加在 y=0 上的分布面力，
    # 画进域内会被读成「试件内部有个形状」（初版就是这个毛病）。
    half = 0.9
    xs = np.linspace(-half, half, 160)
    p = np.sqrt(np.clip(1.0 - (xs / half) ** 2, 0.0, None))
    cc = C_OVER_A * half
    inner = np.sqrt(np.clip(1.0 - (xs / cc) ** 2, 0.0, None))
    q = MU * (p - C_OVER_A * inner)

    # q(x) 在上（切向），p(x) 在下（法向），都贴着表面向上画
    base_p, base_q = 0.02, 0.62
    ax.fill_between(xs, base_p, base_p + 0.46 * p, color="C3", alpha=0.30, lw=0)
    ax.plot(xs, base_p + 0.46 * p, color="C3", lw=1.1)
    ax.fill_between(xs, base_q, base_q + 0.46 * q / max(q.max(), 1e-9),
                    color="C2", alpha=0.30, lw=0)
    ax.plot(xs, base_q + 0.46 * q / max(q.max(), 1e-9), color="C2", lw=1.1)

    # 三组标签必须排开：初版 p/q/滑移区三者叠在一起，谁也读不清
    ax.text(half + 0.30, base_p + 0.20,
            "$p(x)$ Hertz, $p_0$ = %.0f MPa" % P0_MPA,
            fontsize=7, color="C3", va="center")
    ax.text(half + 0.30, base_q + 0.30,
            "$q(x)$ Cattaneo–Mindlin\n$\\mu$ = %.1f, $Q/\\mu P$ = 0.6, "
            "$c/a$ = %.3f" % (MU, C_OVER_A),
            fontsize=7, color="C2", va="center")
    # 滑移区：标在表面线上，标签放到左侧域外，用引线指过去
    for s in (-1, 1):
        ax.plot([s * cc, s * half], [0.0, 0.0], color="C2", lw=4.0,
                solid_capstyle="butt", alpha=0.9, zorder=5)
    ax.annotate("slip zones\n$c \\leq |x| \\leq a$",
                xy=(-0.5 * (cc + half), 0.0), xytext=(-W_MM / 2 - 0.15, 0.85),
                fontsize=7, color="C2", ha="right", va="center",
                arrowprops=dict(arrowstyle="->", color="C2", lw=0.7))
    # 本体应力
    for y in np.linspace(-H_MM + 0.35, -0.35, 4):
        ax.annotate("", xy=(W_MM / 2 + 0.55, y), xytext=(W_MM / 2, y),
                    arrowprops=dict(arrowstyle="-|>", color="C0", lw=1.0))
    ax.text(W_MM / 2 + 0.62, -H_MM / 2,
            "$\\sigma_a$ = %.0f MPa\n(bulk)" % SIGMA_A_MPA,
            fontsize=7.5, color="C0", va="center")
    # 边界条件
    ax.text(-W_MM / 2 - 0.12, -H_MM / 2, "$u_x = 0$", fontsize=7.5,
            rotation=90, ha="right", va="center")
    for xh in np.linspace(-W_MM / 2 + 0.3, W_MM / 2 - 0.3, 9):
        ax.plot([xh, xh - 0.16], [-H_MM, -H_MM - 0.20], color="0.35", lw=0.7)
    ax.text(0.0, -H_MM - 0.42, "$u_y = 0$", fontsize=7.5, ha="center")
    # 尺寸与加密区
    ax.annotate("", xy=(-W_MM / 2, -H_MM - 0.72), xytext=(W_MM / 2, -H_MM - 0.72),
                arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.7))
    ax.text(0.0, -H_MM - 0.92, "$W$ = %.1f mm" % W_MM, fontsize=7, ha="center")
    ax.text(-W_MM / 2, 1.62,
            "contact patch drawn enlarged: $a$ = %.2f $\\mu$m, "
            "$W/a \\approx$ %.0f" % (A_UM, W_MM * 1e3 / A_UM),
            fontsize=7, ha="left", color="0.3")
    ax.set_xlim(-W_MM / 2 - 2.0, W_MM / 2 + 3.4)
    ax.set_ylim(-H_MM - 1.15, 1.85)
    ax.axis("off")


def make(path: pathlib.Path | None = None) -> pathlib.Path:
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.2),
                             gridspec_kw={"width_ratios": [0.92, 0.62, 1.70]})
    _panel_a(axes[0])
    _panel_b(axes[1])
    _panel_c(axes[2])
    # 三个标题用图级坐标摆在同一高度 —— (b) 是等比例轴，轴框比另外两个矮，
    # 用 ax.set_title 会让它明显下沉。
    for ax, label in zip(axes, ("(a) conductor at a suspension clamp",
                                "(b) wire–wire contact",
                                "(c) 2D plane-strain model")):
        box = ax.get_position()
        fig.text(box.x0 + box.width / 2, 0.97, label,
                 ha="center", va="top", fontsize=9)
    fig.text(0.01, -0.02,
             "Contact tractions are prescribed analytically (Hertz + "
             "Cattaneo–Mindlin) rather than solved as a contact problem; "
             "the counterpart wire is not meshed. The contact patch is drawn "
             "enlarged for clarity.", fontsize=6.5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = path or (OUT_DIR / "fig1_schematic.png")
    fig.savefig(out)
    plt.close(fig)
    print("  写出 %s" % out.relative_to(ROOT))
    return out


if __name__ == "__main__":
    make()
