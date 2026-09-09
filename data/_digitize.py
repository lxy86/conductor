"""程序化数字化 Miehe 2010 Fig. 7（修正版）。

坐标框由黑色长直线定位；两个子图的坐标范围均为 x: 0~0.008 mm、y: 0~0.8 kN。
两处必须处理：
  1) 图例色块也是红/绿，必须把左上角图例区排除；
  2) 斜率只能在**上升段**拟合 —— 峰后下降段同样穿过 0.1~0.4 kN，混进去会算错 10 倍。
"""
import numpy as np
import PIL.Image
import pymupdf

doc = pymupdf.open("miehe2010.pdf")
open("data/_f7.png", "wb").write(doc.extract_image(87)["image"])
a = np.asarray(PIL.Image.open("data/_f7.png").convert("RGB")).astype(int)
H, W, _ = a.shape
R, G, B = a[:, :, 0], a[:, :, 1], a[:, :, 2]
dark = (R < 110) & (G < 110) & (B < 110)
green = (G > 110) & (G - R > 55) & (G - B > 55)
red = (R > 110) & (R - G > 55) & (R - B > 55)

col_counts, row_counts = dark.sum(axis=0), dark.sum(axis=1)
def group(idx):
    out, cur = [], [idx[0]]
    for v in idx[1:]:
        if v - cur[-1] <= 3: cur.append(v)
        else: out.append(int(np.mean(cur))); cur = [v]
    out.append(int(np.mean(cur))); return out
vs = group(np.nonzero(col_counts > 0.40 * H)[0])
hs = group(np.nonzero(row_counts > 0.30 * (W / 2))[0])
y_top, y_bot = hs[0], hs[-1]
X_MAX, Y_MAX = 0.008, 0.8
print("坐标框：竖线 %s 横线 %s  -> x:0~%g mm, y:0~%g kN" % (vs, hs, X_MAX, Y_MAX))

results = {}
for name, ell, (xa, xb) in (("(a)", 0.0150, (vs[0], vs[1])), ("(b)", 0.0075, (vs[2], vs[3]))):
    for label, mask in (("eta=0", red), ("eta=1e-6", green)):
        sub = mask[:, xa:xb + 1].copy()
        # 排除左上角图例区（约左 45% 宽、上 40% 高）
        sub[: int(y_top + 0.40 * (y_bot - y_top)), : int(0.45 * (xb - xa))] = False
        cols = np.nonzero(sub.any(axis=0))[0]
        u = cols / (xb - xa) * X_MAX
        F = np.array([(y_bot - np.nonzero(sub[:, c])[0].mean()) / (y_bot - y_top) * Y_MAX
                      for c in cols])
        i = int(F.argmax())
        # 上升段红绿重合、绿色覆盖红色，故红线只在峰后可见 -> 上升段可能为空
        rise = (u < 0.7 * u[i]) & (F > 0.05 * F[i])
        if rise.sum() < 5:
            k = float("nan")
        else:
            # 过原点拟合：F = k u（曲线确实自原点出发）
            k = float(np.sum(u[rise] * F[rise]) / np.sum(u[rise] ** 2))
        results[(name, label)] = (F[i], u[i], k)
        print("%s ell=%.4f mm  %-9s  F_peak=%.4f kN @ u=%.5f mm   k=%.1f kN/mm = %.4e N/m  (%d 列)"
              % (name, ell, label, F[i], u[i], k, k * 1e6, cols.size))
