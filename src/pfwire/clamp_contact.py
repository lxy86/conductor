"""6.5 基准工况的线夹接触：无摩擦罚函数法向接触，挂入 PhaseFieldSolver（D-67）。

几何、罚函数与 6.1 的 :class:`pfwire.obstacle.ObstacleContact`（D-57）完全相同，复用其函数：
    g = |X + u − c| − R_clamp，p = ε·max(−g, 0)，ε = κE/h，c = (0, R_clamp − δ)。
区别只在于：这里不持有 u、不建求解器、不带边界条件——接触项是 u 的函数，由相场求解器的 u 方程调用。

摩擦：不包含（D-67 的偏离，PLAN §1.1）。
"""

from __future__ import annotations

import ufl
from dolfinx import fem
from mpi4py import MPI

from pfwire.mesh import BLOCK_BOTTOM, BLOCK_TOP
from pfwire.obstacle import (
    clamp_gap,
    clamp_gap_lower,
    clamp_normal,
    clamp_normal_lower,
    penalty_pressure,
    penalty_stiffness,
    plane_gap,
    plane_normal,
)

__all__ = ["ClampContact"]


class ClampContact:
    """线夹的罚函数法向接触（无摩擦）。

    参数
    ----
    mesh, facet_tags : 网格与面 tag（``BLOCK_TOP`` 为接触面）。
    E                : 弹性模量（SI），用于罚刚度。
    R_clamp          : 线夹轴向曲率半径（m）。
    kappa            : 罚系数 κ（无量纲），ε = κE/h_contact。
    h_contact        : 接触面的单元尺寸（m）。D-67：取 clamp_mesh 加密带的 ℓ/4。
    """

    def __init__(self, mesh, facet_tags, *, E: float, R_clamp: float, kappa: float,
                 h_contact: float, bottom_support: float | None = None,
                 symmetric: bool = False, depth: float | None = None) -> None:
        """``symmetric=True``：单丝被**两个相同线夹对称挤压**（Lévesque 2011）。

        出处：F. Lévesque et al., *Tribology International* 44 (2011) 1014–1023,
        §2："We consider the wire as being squeezed between two similar clamps."
        原模型的边界条件是三个对称面（u_x=0、u_y=0、u_z=0）+ 自由端拉伸 +
        夹持载荷。二维平面应变下只剩轴向锚定的对称面（u_x = 0）。

        为什么必须对称：单侧压紧时，压紧力的反作用力只能由位移约束承担，而
        那个约束（底面固定 / 简支）会同时把弯曲约束掉——实测 σ_a 只剩目标的
        15%，或者 W 只剩 1/7。对称夹持让压紧力**自平衡**，于是只需要一个
        轴向锚定（对称面 u_x = 0），弯曲不被干扰。

        ``bottom_support``（平面支撑）与 ``symmetric``（对称圆弧线夹）二选一：
        前者实测会让接触一发生就发散（支撑面与单丝底面初始重合，罚压力超
        σ_y 八倍），故对称挤压应走 ``symmetric``。
        """
        self.mesh = mesh
        self.facet_tags = facet_tags
        self.R_clamp = float(R_clamp)
        if symmetric and depth is None:
            raise ValueError("symmetric=True 时必须给 depth（单丝厚度），下侧线夹要用它定位")
        self.depth = None if depth is None else float(depth)
        self.eps = penalty_stiffness(kappa, E, h_contact)
        self.h_contact = float(h_contact)
        self.delta = fem.Constant(mesh, 0.0)
        self.bottom_support = bottom_support
        self.symmetric = bool(symmetric)
        if symmetric and bottom_support is not None:
            raise ValueError("symmetric 与 bottom_support 二选一（对称挤压用 symmetric）")
        self.delta_low = fem.Constant(mesh, 0.0)   # 下侧线夹的位移（对称时 = δ/2）
        # 接触项核含 sqrt/max_value（法向、罚压力）。UFL 对残差 F 与对其 u 的导数 J
        # 估计出的多项式阶不同，默认的「按表达式估计求积阶」会让两者落到不同规则，
        # 离散上 F 与 dF/du 不一致（切线校验差到 1.6e-4）。显式固定求积阶 2，
        # 使 F 与 dF/du 用同一规则（线性基函数下 dot(n,v) 的有效阶为 2）。
        self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags,
                              metadata={"quadrature_degree": 2})

    def set_delta(self, delta: float) -> None:
        """线夹下压位置（绝对值，m）。由求解器的外层（W' 搜索或延拓）调用。

        delta 是**每个线夹的行程**（对称模式下上下各走 δ，总相对接近量 2δ）。
        这样单侧穿透量 = δ，与单侧模式的 W(δ)、以及 6.1 的结果可比。
        语义来自 Lévesque 2011："a displacement is applied to the middle of the
        wire"——夹持位移就是两个线夹的相对接近量，对称时每侧各分担一半。
        """
        self.delta.value = float(delta)
        if self.symmetric:
            self.delta_low.value = float(delta)   # 上下同行程，相对接近量 2δ

    def _pressure_and_normal(self, u):
        x = ufl.SpatialCoordinate(self.mesh) + u
        g = clamp_gap(x, self.R_clamp, self.delta)
        return penalty_pressure(g, self.eps), clamp_normal(x, self.R_clamp, self.delta)

    def _bottom_pressure(self, u):
        """下侧障碍的压力（无下侧障碍时为 0）。"""
        if self.bottom_support is not None:
            x = ufl.SpatialCoordinate(self.mesh) + u
            return penalty_pressure(plane_gap(x, self.bottom_support), self.eps)
        if self.symmetric:
            x = ufl.SpatialCoordinate(self.mesh) + u
            g = clamp_gap_lower(x, self.R_clamp, self.delta_low, -self.depth)
            return penalty_pressure(g, self.eps)
        return None

    def _bottom_normal(self, u):
        if self.bottom_support is not None:
            return ufl.as_vector(plane_normal())
        if self.symmetric:
            x = ufl.SpatialCoordinate(self.mesh) + u
            return clamp_normal_lower(x, self.R_clamp, self.delta_low, -self.depth)
        return None

    def residual(self, u, v) -> ufl.Form:
        """接触力的虚功 ∫ p n·v ds（上侧 + 下侧）。求解器在 u 方程里取负号。"""
        p, n = self._pressure_and_normal(u)
        form = p * ufl.dot(n, v) * self.ds(BLOCK_TOP)
        pb = self._bottom_pressure(u)
        if pb is not None:
            form = form + pb * ufl.dot(self._bottom_normal(u), v) * self.ds(BLOCK_BOTTOM)
        return form

    def reaction(self, u) -> float:
        """法向反力 W = −∫ p n_y ds（N/m，单位厚度），与 6.1 的 ObstacleContact.reaction 同式。

        只算**上侧线夹**的反力（下侧支撑的反力在静力平衡里与之相等反号，
        不计入 W）——W 的定义是线夹压紧力。
        """
        p, n = self._pressure_and_normal(u)
        local = fem.assemble_scalar(fem.form(-p * n[1] * self.ds(BLOCK_TOP)))
        return float(self.mesh.comm.allreduce(local, op=MPI.SUM))

    def support_reaction(self, u) -> float:
        """下侧支撑的反力（N/m，单位厚度），诊断用：应与 :meth:`reaction` 接近。"""
        pb = self._bottom_pressure(u)
        if pb is None:
            return 0.0
        local = fem.assemble_scalar(fem.form(pb * self.ds(BLOCK_BOTTOM)))
        return float(self.mesh.comm.allreduce(local, op=MPI.SUM))
