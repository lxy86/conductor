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
