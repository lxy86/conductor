"""单丝对刚体圆弧线夹的罚函数接触问题（PLAN §0.5 决策 B；D-57 S2）。

与 `pfwire.contact.ContactModel` 的区别（D-56 之后）：
  - 线夹是解析障碍（见 `pfwire.obstacle`），只有单丝一个体，不需要 dolfinx_contact；
  - 接触牵引 p = ε·max(−g, 0) 直接进入残差，与体的本构（线弹性或 `PlasticLaw`）同一套应力，
    因此塑性区的接触压力不会超出体的承载能力（D-56 的根因 2）；
  - 不做最近点搜索，所以并行不需要幽灵单元（D-56 的根因 1）。

弹性本构：平面应变 Lamé 参数（Johnson, *Contact Mechanics*, 1985, §2.5）
  μ = E / (2(1+ν))，λ = E ν / ((1+ν)(1−2ν))，σ = 2μ ε + λ tr(ε) I。
"""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

from pfwire.obstacle import (
    BLOCK_BOTTOM,
    BLOCK_TOP,
    clamp_gap,
    clamp_normal,
    penalty_pressure,
    penalty_stiffness,
)

__all__ = ["ObstacleContact", "plane_strain_lame"]


def plane_strain_lame(E: float, nu: float) -> tuple[float, float]:
    """平面应变的 (μ, λ)。"""
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lmbda


class ObstacleContact:
    """单丝（弹性或弹塑性）+ 刚体圆弧线夹 的罚函数接触求解器。

    参数
    ----
    mesh, facet_tags : 单丝网格与面 tag（`BLOCK_TOP` 为接触面，`BLOCK_BOTTOM` 为固定面）。
    E, nu            : 单丝弹性参数（SI）。
    R_clamp          : 线夹轴向曲率半径（m）。
    kappa            : 罚刚度系数，ε = κ E / h_contact。
    h_contact        : 接触面网格尺寸（m），用于罚刚度。
    solver           : PETSc/SNES 选项字典，直接传给 `NonlinearProblem`。
    plastic          : 可选 `pfwire.plasticity.PlasticLaw`；给了则体本构用它，否则线弹性。
    degree           : 位移元阶（默认 1）。
    """

    def __init__(self, mesh, facet_tags, *, E: float, nu: float, R_clamp: float,
                 kappa: float, h_contact: float, solver: dict, plastic=None, degree: int = 1):
        self.mesh, self.facet_tags = mesh, facet_tags
        self.E, self.nu, self.R_clamp = float(E), float(nu), float(R_clamp)
        self.plastic = plastic
        self.eps = penalty_stiffness(kappa, self.E, h_contact)

        gdim = mesh.geometry.dim
        self.V = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
        self.u = fem.Function(self.V)
        self.delta = fem.Constant(mesh, 0.0)

        if plastic is None:
            mu, lmbda = plane_strain_lame(self.E, self.nu)
            self.dx = ufl.Measure("dx", domain=mesh)

            def stress(w):
                eps = ufl.sym(ufl.grad(w))
                return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(gdim)
        else:
            self.dx = plastic.dx
            stress = plastic.sigma
        self._stress = stress

        # 接触测度固定求积阶 2：与 ClampContact（6.5）同一规则，两者离散的是同一个
        # 接触项，必须逐位一致（T2 对拍）。核含 sqrt/max_value，默认按表达式估计
        # 求积阶会让 F 与 dF/du 落到不同规则（见 ClampContact 的注释，D-67）。
        self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags,
                              metadata={"quadrature_degree": 2})
        x = ufl.SpatialCoordinate(mesh)
        xu = x + self.u
        self.gap = clamp_gap(xu, self.R_clamp, self.delta)
        self.normal = clamp_normal(xu, self.R_clamp, self.delta)
        self.pressure = penalty_pressure(self.gap, self.eps)

        v = ufl.TestFunction(self.V)
        self.F = (ufl.inner(stress(self.u), ufl.sym(ufl.grad(v))) * self.dx
                  - self.pressure * ufl.dot(self.normal, v) * self.ds(BLOCK_TOP))

        # 固定底面：BC 建在求解器自己的空间 V 上（CLAUDE.md 陷阱 1）
        bottom = facet_tags.find(BLOCK_BOTTOM)
        dofs = fem.locate_dofs_topological(self.V, mesh.topology.dim - 1, bottom)
        zero = fem.Constant(mesh, np.zeros(gdim, dtype=np.float64))
        self.bcs = [fem.dirichletbc(zero, dofs, self.V)]

        self.problem = NonlinearProblem(
            self.F, self.u, bcs=self.bcs, petsc_options_prefix="obst_",
            petsc_options=dict(solver))

    # ---- 求解 -----------------------------------------------------------
    def step(self, delta: float) -> int:
        """把线夹下压到 δ（绝对位置，单调递增做延拓）并求解；**不收敛即抛异常**（硬约束 6）。

        返回 Newton 迭代数。
        """
        self.delta.value = float(delta)
        self.problem.solve()
        snes = self.problem.solver
        reason = snes.getConvergedReason()
        if reason <= 0:
            raise RuntimeError("接触 Newton 未收敛（δ = %.4g m，SNES reason = %d）" % (delta, reason))
        return int(snes.getIterationNumber())

    def commit(self) -> None:
        """每个载荷步收敛后把塑性历史写回（弹性时无操作）。"""
        if self.plastic is not None:
            self.plastic.commit(self.u)

    # ---- 后处理 ---------------------------------------------------------
    def reaction(self) -> float:
        """线夹对单丝的法向反力 P = −∫ p n_y ds（单位厚度，N/m），精确面积分。"""
        local = fem.assemble_scalar(fem.form(-self.pressure * self.normal[1] * self.ds(BLOCK_TOP)))
        return float(self.mesh.comm.allreduce(local, op=MPI.SUM))

    def pressure_profile(self):
        """接触面上的 (x, p)：p 在 CG1 节点上插值，按 x 排序，只含接触面节点。

        并行下只取本进程拥有的节点，再 allgather 合并。
        """
        Vs = fem.functionspace(self.mesh, ("Lagrange", 1))
        p_fun = fem.Function(Vs)
        p_fun.interpolate(fem.Expression(self.pressure, Vs.element.interpolation_points))
        dofs = fem.locate_dofs_topological(Vs, self.mesh.topology.dim - 1,
                                           self.facet_tags.find(BLOCK_TOP))
        n_owned = Vs.dofmap.index_map.size_local
        keep = dofs[dofs < n_owned]
        xs = Vs.tabulate_dof_coordinates()[keep, 0]
        comm = self.mesh.comm
        xs = np.concatenate(comm.allgather(xs))
        ps = np.concatenate(comm.allgather(p_fun.x.array[keep]))
        o = np.argsort(xs)
        return xs[o], ps[o]
