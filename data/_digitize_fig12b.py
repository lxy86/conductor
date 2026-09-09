"""程序化数字化 Carrara Fig. 12b：读 a0/W = 0.22（蓝线）的单调峰值 = P_mon。

坐标框由黑色长直线定位；轴范围 u: 0~2.0 mm、P: 0~250。纵轴标 "kN"，
但按 Tab. 2 的百分比、LEFM 锚点与弹性柔度三重交叉验证，单位应为 N。

两个坑（都踩过）：
1. **索引系必须统一**。曾用相对列号配绝对行号做图例遮罩，遮错了位置，
   峰值被图例里的蓝色线样污染（读成 209.8 @ u=0.707，而肉眼是 190 @ 0.47）；
2. 斜率只能在**确实线性**的段上拟合。峰值找错会把拟合窗口拉到软化段，
   斜率偏小近 2 倍 —— 这正是第一版把柔度算成 ASTM 的 2.09 倍的原因。
"""

import numpy as np
import PIL.Image
import pymupdf

d = pymupdf.open("carrara2020.pdf")
page = d[27]
r = page.rect
clip = pymupdf.Rect(r.x0 + 0.45 * r.width, r.y0 + 0.02 * r.height,
                    r.x1, r.y0 + 0.42 * r.height)
pix = page.get_pixmap(clip=clip, dpi=400)
pix.save("data/_fig12b.png")

a = np.asarray(PIL.Image.open("data/_fig12b.png").convert("RGB")).astype(int)
H, W, _ = a.shape
R, G, B = a[:, :, 0], a[:, :, 1], a[:, :, 2]
dark = (R < 110) & (G < 110) & (B < 110)


def group(idx, gap=4):
    out, cur = [], [idx[0]]
    for v in idx[1:]:
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            out.append(int(np.mean(cur)))
            cur = [v]
    out.append(int(np.mean(cur)))
    return out


vs = group(np.nonzero(dark.sum(axis=0) > 0.25 * H)[0])
hs = group(np.nonzero(dark.sum(axis=1) > 0.25 * W)[0])
print("图像 %dx%d   候选竖线 %s   候选横线 %s" % (W, H, vs, hs))

# **不能盲取 vs[0]/hs[-1]**：裁剪区里混进了 Fig.12a 的右边框（最左那条竖线）
# 与图注的文字行，取端点会把坐标框认错、两个轴的标定同时错掉。
# 稳健做法：坐标框必须**恰好把彩色曲线像素括住** —— 用曲线自身定位框线。
sat = (np.abs(R - G) > 55) | (np.abs(G - B) > 55) | (np.abs(R - B) > 55)
sat_cols = np.nonzero(sat.any(axis=0))[0]
sat_rows = np.nonzero(sat.any(axis=1))[0]
x_left = max([v for v in vs if v <= sat_cols.min()], default=vs[0])
x_right = min([v for v in vs if v >= sat_cols.max()], default=vs[-1])
y_top = max([h for h in hs if h <= sat_rows.min()], default=hs[0])
y_bot = min([h for h in hs if h >= sat_rows.max()], default=hs[-1])
print("选定坐标框：x [%d, %d]  y [%d, %d]" % (x_left, x_right, y_top, y_bot))
U_MAX, P_MAX = 2.0, 250.0


def to_u(col):
    return (col - x_left) / (x_right - x_left) * U_MAX


def to_P(row):
    return (y_bot - row) / (y_bot - y_top) * P_MAX


# 蓝线 = a0/W = 0.22。**全用绝对索引**
blue = (B > 110) & (B - R > 55) & (B - G > 55)
mask = np.zeros_like(blue)
mask[y_top:y_bot + 1, x_left:x_right + 1] = True          # 只留框内
blue = blue & mask

# 图例遮罩：图例在框内的右上角。用绝对坐标，且按实际观察留足余量。
lx0 = x_left + int(0.50 * (x_right - x_left))
ly1 = y_top + int(0.60 * (y_bot - y_top))
blue[y_top:ly1, lx0:x_right + 1] = False
print("图例遮罩（绝对）：列 >= %d，行 <= %d" % (lx0, ly1))

cols = np.nonzero(blue.any(axis=0))[0]
u = np.array([to_u(c) for c in cols])
# 每列取最**上**的蓝像素 = 该 u 处的最大 P
P = np.array([to_P(np.nonzero(blue[:, c])[0].min()) for c in cols])
i = int(P.argmax())
print()
print("蓝线：%d 列，u 覆盖 %.3f~%.3f mm" % (cols.size, u.min(), u.max()))
print("峰值 P_mon = %.1f  @ u = %.4f mm" % (P[i], u[i]))
order = np.argsort(P)[::-1][:5]
print("前 5 高点：%s" % ", ".join("%.1f@%.3f" % (P[j], u[j]) for j in order))

# 弹性斜率：只用峰值前 60% 且 P 在 20%~70% 之间的段（避开原点与软化）
rise = (u < 0.6 * u[i]) & (P > 0.2 * P[i]) & (P < 0.7 * P[i])
k = float(np.sum(u[rise] * P[rise]) / np.sum(u[rise] ** 2))
print()
print("弹性段拟合用 %d 点（u %.3f~%.3f）" % (rise.sum(), u[rise].min(), u[rise].max()))
print("弹性刚度 k = %.2f [P单位]/mm  -> 柔度 = %.4e mm/[P单位]" % (k, 1.0 / k))

E, T, Wspec, alpha = 6.0e9, 1.0e-3, 50.0e-3, 0.22
rr = (1.0 + alpha) / (1.0 - alpha)
poly = (2.163 + 12.219 * alpha - 20.065 * alpha**2 - 0.9925 * alpha**3
        + 20.609 * alpha**4 - 9.9314 * alpha**5)
v_per_N = rr**2 * poly / (E * T)
print("ASTM E647 载荷线柔度 = %.4e mm/N" % (v_per_N * 1e3))
print("数字化/ASTM = %.3f  ->  图中 P 的单位是 %s"
      % ((1.0 / k) / (v_per_N * 1e3),
         "N" if 0.5 < (1.0 / k) / (v_per_N * 1e3) < 2.0 else "**不是 N，需重查**"))

kic = np.sqrt(E * 2280.0)
Yv = (2 + alpha) / (1 - alpha) ** 1.5 * (
    0.886 + 4.64 * alpha - 13.32 * alpha**2 + 14.72 * alpha**3 - 5.6 * alpha**4
)
print()
print("LEFM P_c = %.1f N；本文相场 190 N；本图数字化 %.1f" % (
    kic * T * np.sqrt(Wspec) / Yv, P[i]))
mean_ratio = 148.6
print("Tab.2 反解 (1-R)Pmon = %.1f  ->  R = 1 - %.1f/%.1f = %.3f"
      % (mean_ratio, mean_ratio, P[i], 1.0 - mean_ratio / P[i]))
