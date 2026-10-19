"""刚体圆弧线夹的接触障碍：几何、罚函数牵引、单丝网格（PLAN §0.5 决策 B；D-57）。

线夹按决策 B 为**刚体**，因此不需要网格，只是一个解析圆弧障碍：
  圆心 c = (0, R_clamp − δ)，半径 R_clamp（线夹下压 δ 后整体下移）。
  单丝上表面点 X + u 的有符号间隙 g = |X + u − c| − R_clamp（负值 = 穿透）。
  罚函数牵引 p = ε · max(−g, 0)，沿 n = (X + u − c)/|X + u − c| 作用在单丝上。
  （Wriggers, *Computational Contact Mechanics*, 2nd ed., 2006, §7 罚函数法。）

与 dolfinx_contact 的区别：不做最近点搜索，接触力直接由位移算出；因此并行不需要幽灵单元，
接触核也不再是线弹性的，可以与 `PlasticLaw.sigma(u)` 一致耦合（D-56 的两个根因都被绕开）。

坐标约定：单丝上表面在 y = 0，线夹在上方；单丝底面在 y = −depth，全固定。
"""

from __future__ import annotations

import pathlib

import numpy as np
import ufl

from pfwire.mesh import BLOCK_BOTTOM, BLOCK_TOP, block_mesh

__all__ = [
    "BLOCK_TOP", "BLOCK_BOTTOM",
    "clamp_gap", "clamp_normal", "penalty_pressure", "penalty_stiffness",
    "penalty_tangential_stiffness", "tangential_traction",
    "build_block_mesh",
]



def _offset(x, R_clamp, delta):
    """X + u 相对线夹圆心的位移向量 r = X + u − c。这里的 x 已是 X + u。"""
    c = ufl.as_vector([0.0, R_clamp - delta])
    return x - c


def clamp_gap(x, R_clamp, delta):
    """有符号间隙 g = |x − c| − R_clamp。``x`` 为 X + u（UFL 表达式）。负值 = 穿透。"""
    r = _offset(x, R_clamp, delta)
    return ufl.sqrt(ufl.inner(r, r)) - R_clamp


def clamp_normal(x, R_clamp, delta):
    """从圆心指向单丝的单位法向（即线夹对单丝的推力方向，朝下）。"""
    r = _offset(x, R_clamp, delta)
    return r / ufl.sqrt(ufl.inner(r, r))


def lower_clamp_center(R_clamp: float, delta_low, y_bottom: float) -> object:
    """下侧（对称）线夹的圆心：c = (0, y_bottom − R_clamp + δ_low)。

    与上侧 :func:`clamp_gap` 的圆心 c = (0, R_clamp − δ) 关于单丝中面对称。
    单丝下表面在 y = y_bottom（= −H）。δ_low = 0 时圆心在 y_bottom − R，
    线夹顶点恰在 y_bottom（相切，g = 0）。
    """
    return ufl.as_vector([0.0, y_bottom - R_clamp + delta_low])


def clamp_gap_lower(x, R_clamp: float, delta_low, y_bottom: float) -> object:
    """下侧线夹的有符号间隙 g = |x − c_low| − R_clamp（负值 = 穿透）。"""
    c = lower_clamp_center(R_clamp, delta_low, y_bottom)
    r = x - c
    return ufl.sqrt(ufl.inner(r, r)) - R_clamp


def clamp_normal_lower(x, R_clamp: float, delta_low, y_bottom: float) -> object:
    """下侧线夹对单丝的推力方向：从圆心指向单丝，即**向上**（+y 分量）。

    与 :func:`clamp_normal` 同口径（都返回「障碍对单丝的推力方向」）：
    上侧线夹圆心在上方，推力朝下；下侧圆心在下方，推力朝上。
    """
    c = lower_clamp_center(R_clamp, delta_low, y_bottom)
    r = x - c
    # 圆心在下方，r 指向上方，故**不加负号**：与 clamp_normal 一样是 r/|r|。
    # （上侧的 r 指向下方，r/|r| 自然给出朝下的推力；两侧同式。）
    return r / ufl.sqrt(ufl.inner(r, r))


def plane_gap(x, y_plane: float) -> object:
    """水平平面障碍的有符号间隙 g = y_plane − y（负值 = 点在平面上方，即穿透）。

    ``x`` 为 X + u（UFL 表达式），``y_plane`` 为平面的 y 坐标（常数或 Constant）。
    平面在单丝下方，单丝被压下去时 y < y_plane 即穿透。
    """
    return y_plane - x[1]


def plane_normal() -> tuple:
    """平面障碍对单丝的推力方向：平面在下方，把单丝往上推，故为 +y。

    与 :func:`clamp_normal` 同口径——都返回「障碍对单丝的推力方向」，
    残差里一律写成 p·(n·v)。上侧圆弧线夹的推力朝下（−y），下侧平面朝上（+y）。
    """
    return (0.0, 1.0)


def penalty_pressure(g, eps):
    """罚函数接触压力 p = ε · max(−g, 0)（仅穿透时非零，单位 Pa）。"""
    return eps * ufl.max_value(-g, 0.0)


def penalty_stiffness(kappa: float, E: float, h: float) -> float:
    """罚刚度 ε = κ E / h（单位 Pa/m）。κ 为无量纲系数，h 为接触面网格尺寸。

    穿透量 ≈ p / ε ∝ h/(κE)·p；κ 越大穿透越小，但矩阵条件数随之变差。κ 由
    `scripts/verify_obstacle_hertz.py` 的敏感性检查标定，不在代码里写死。
    """
    if kappa <= 0 or E <= 0 or h <= 0:
        raise ValueError("κ、E、h 必须为正：%r %r %r" % (kappa, E, h))
    return float(kappa) * float(E) / float(h)


def penalty_tangential_stiffness(kappa_t: float, E: float, h: float) -> float:
    """切向罚刚度 ε_t = κ_t E / h（单位 Pa/m）。

    与 :func:`penalty_stiffness` 同式，只是系数可独立给。**默认 κ_t = κ**
    （接触面各向同性），此时 ε_t = ε；只有当摩擦项的切向刚度与法向罚刚度
    耦合得过强、矩阵病态时才分开调（见 PLAN 6.6.1 的风险注）。
    """
    if kappa_t <= 0 or E <= 0 or h <= 0:
        raise ValueError("κ_t、E、h 必须为正：%r %r %r" % (kappa_t, E, h))
    return float(kappa_t) * float(E) / float(h)


def tangential_traction(s_trial, p, mu: float, eps_t: float, *, regularized: bool = True):
    """Coulomb 摩擦的切向牵引 q。

    ``regularized=True``（**缺省**）::

        q = μp · (ε_t s) / (μp + ε_t |s|)

    ``regularized=False``（硬截断的试探-惩罚格式）::

        q = ε_t · sign(s) · min(|s|, μp/ε_t)

    出处：Wriggers, *Computational Contact Mechanics*, 2nd ed. (2006), §6.2 的
    **试探-惩罚格式**（trial-penalty）：先按弹性罚刚度由试探切向滑移 s^{tr}
    算切向牵引，再用 Coulomb 上限 μp 截断。截断后超出部分变成真实滑移，
    于是只需一个辅助历史量（本步的切向位移基准），**不需要 KKT 迭代**。

    **为什么缺省用正则化形式**（2026-10-19 实测，教训记在 DECISIONS）：
    硬截断在 |s| = μp/ε_t 处有折点，且 μ = 0 时 `min(|s|, 0)` 恒为 0 但
    **其导数在该点不连续**（`sign(s)` 在 s = 0 无定义）。把切向项装进
    u 方程的 form 之后，这个坏导数会毒化 Jacobian —— 实测**仅仅带着该项**
    （μ 已设 0）就让 δ = 1 µm 的法向步 Newton 发散（SNES -6），
    而同一网格不装该项则步步收敛。正则化形式
    q = μp·ε_t s/(μp + ε_t|s|) 处处光滑（含 μ = 0，此时 q ≡ 0 且导数 ≡ 0），
    且过渡宽度 |s| ~ μp/ε_t = 3e-11 m 比任何物理滑移小 3 个数量级，
    故在物理上就是 Coulomb 律。

    参数
    ----
    s_trial : 试探切向滑移 s = τ·(u − u_prev)，UFL 表达式。
    p       : 同点的法向接触压力（Pa），由 :func:`penalty_pressure` 给出。
    mu      : Coulomb 摩擦系数（无量纲），可为 UFL Constant（运行时改）。
    eps_t   : 切向罚刚度 ε_t（Pa/m）。

    ⚠ **s_trial 必须已经是 `ufl.variable`**（调用方负责）：q 与其导数里
    多次引用 s，包成 variable 才能让 UFL/FFC 把它们识别成同一个中间量。
    """
    mu = float(mu) if not isinstance(mu, ufl.Constant) else mu
    eps_t = float(eps_t)
    if eps_t <= 0.0:
        raise ValueError("ε_t 必须为正，收到 %r" % (eps_t,))
    if regularized:
        # 分子分母同乘 μp：μ = 0 时整式为 0，且对 s 的导数也恒为 0（光滑）。
        # ⚠ 分母必须加下限：接触区外 p = 0（法向没穿透），此时
        # limit = μp = 0、s 也常为 0，0/(0+0) = **NaN** —— 实测 μ=0 时
        # q 的 y 分量读出 [0, nan, nan]，Newton 立刻 SNES -4。
        # 加一个远小于任何物理量的地板项（ε_t × 1e-30 m ≈ 1e-14 Pa）即可，
        # 它不改变黏着区/滑移区的任何行为。
        limit = mu * p                      # Coulomb 上限（Pa）
        return limit * (eps_t * s_trial) / (limit + eps_t * abs(s_trial)
                                            + eps_t * 1.0e-30)
    if float(mu) < 0.0:
        raise ValueError("μ 必须非负，收到 %r" % (mu,))
    return eps_t * ufl.sign(s_trial) * ufl.min_value(abs(s_trial), mu * p / eps_t)


def build_block_mesh(workdir: pathlib.Path, *, width: float, depth: float,
                     h_fine: float, h_coarse: float, fine_halfwidth: float,
                     fine_depth: float, transition: float, order: int = 1,
                     quads: bool = True, comm):
    """单个加密盒 |x| ≤ fine_halfwidth、深 fine_depth 的矩形网格（D-57 S2/S3 用）。

    委托给 `pfwire.mesh.block_mesh`；返回 (mesh, cell_tags, facet_tags)。
    """
    return block_mesh(workdir, width=width, depth=depth,
                      refine_boxes=[(-fine_halfwidth, fine_halfwidth, -fine_depth, 0.0)],
                      h_fine=h_fine, h_coarse=h_coarse, transition=transition,
                      order=order, quads=quads, comm=comm)
