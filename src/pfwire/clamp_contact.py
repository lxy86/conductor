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
    clamp_normal,
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
                 h_contact: float, bottom_support: float | None = None) -> None:
        """``bottom_support``：下侧平面支撑的 y 坐标（m），None 表示没有。

        D-67 §6 实测：单侧线夹压紧时，压紧力的反作用力必须由某个约束承担，
        而这个约束（底面固定 / 简支）会同时把弯曲约束掉（σ_a 只剩 15%）。
        真实的线夹是**上下两个颚**，法向力由夹持自身平衡。故这里加一个下侧
        平面刚体障碍（对称支撑），压紧力自平衡，单丝不需要额外的位移约束。
        平面默认放在单丝底面处（y = −H）。
        """
        self.mesh = mesh
        self.facet_tags = facet_tags
        self.R_clamp = float(R_clamp)
        self.eps = penalty_stiffness(kappa, E, h_contact)
        self.delta = fem.Constant(mesh, 0.0)
        self.bottom_support = bottom_support
        # 接触项核含 sqrt/max_value（法向、罚压力）。UFL 对残差 F 与对其 u 的导数 J
        # 估计出的多项式阶不同，默认的「按表达式估计求积阶」会让两者落到不同规则，
        # 离散上 F 与 dF/du 不一致（切线校验差到 1.6e-4）。显式固定求积阶 2，
        # 使 F 与 dF/du 用同一规则（线性基函数下 dot(n,v) 的有效阶为 2）。
        self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags,
                              metadata={"quadrature_degree": 2})

    def set_delta(self, delta: float) -> None:
        """线夹下压位置（绝对值，m）。由求解器的外层（W' 搜索或延拓）调用。"""
        self.delta.value = float(delta)

    def _pressure_and_normal(self, u):
        x = ufl.SpatialCoordinate(self.mesh) + u
        g = clamp_gap(x, self.R_clamp, self.delta)
        return penalty_pressure(g, self.eps), clamp_normal(x, self.R_clamp, self.delta)

    def _bottom_pressure(self, u):
        """下侧平面支撑的压力（无支撑时为 0）。"""
        if self.bottom_support is None:
            return None
        x = ufl.SpatialCoordinate(self.mesh) + u
        return penalty_pressure(plane_gap(x, self.bottom_support), self.eps)

    def residual(self, u, v) -> ufl.Form:
        """接触力的虚功 ∫ p n·v ds（BLOCK_TOP + 下侧支撑）。求解器在 u 方程里取负号。"""
        p, n = self._pressure_and_normal(u)
        form = p * ufl.dot(n, v) * self.ds(BLOCK_TOP)
        pb = self._bottom_pressure(u)
        if pb is not None:
            nb = ufl.as_vector(plane_normal())
            form = form + pb * ufl.dot(nb, v) * self.ds(BLOCK_BOTTOM)
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
