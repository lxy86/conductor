"""J2 各向同性线性硬化塑性：径向返回映射 + 一致切线（PLAN §1.4、§2.3、Phase 5）。

本构（决策 A，D-50 的材料数）：
    f = sqrt(3/2) |s| - (sigma_y0 + H p) <= 0，  d(eps_p) = dgamma * sqrt(3/2) * s/|s|，  dp = dgamma
H = d sigma_y / d p 是**塑性模量**（单轴下 E_t = E H/(E+H) 是切线模量）。

返回映射与一致切线的出处：
    J. C. Simo, T. J. R. Hughes, *Computational Inelasticity*, Springer 1998,
    第 3 章（J2 各向同性硬化的径向返回、一致切线
    C = K 1⊗1 + 2G θ (I − 1⊗1/3) − 2G θ̄ n⊗n）。线性硬化下 dgamma = f_tr/(3G + H) 为闭式。
    ⚠️ 具体的 Box / 公式编号**待查书核对**（本文件写于未持有原书时，只引到章；
    公式本身由中心差分与 UFL 自动微分两路独立验证）。
本文件**不自创公式**：NumPy 参考实现与 UFL 实现互相对拍（tests/test_plasticity.py），
一致切线另用返回映射的中心差分独立核对。

约定：应变/应力用 3x3 对称张量。平面应变时 eps_zz = 0，但 sigma_zz、eps_e,zz 不为 0，
必须按三维嵌入处理，否则返回映射是错的。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class J2Params:
    """E, nu, 初始屈服 sigma_y0，塑性模量 H = d sigma_y / d p（均为 SI）。"""

    E: float
    nu: float
    sigma_y0: float
    H: float

    def __post_init__(self):
        if not self.E > 0:
            raise ValueError("E 必须 > 0，得到 %r" % (self.E,))
        if not -1.0 < self.nu < 0.5:
            raise ValueError("nu 须在 (-1, 0.5) 内，得到 %r" % (self.nu,))
        if not self.sigma_y0 > 0:
            raise ValueError("sigma_y0 必须 > 0，得到 %r" % (self.sigma_y0,))
        if not self.H >= 0:
            raise ValueError("H 必须 >= 0（H = 0 为理想塑性），得到 %r" % (self.H,))

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def K(self) -> float:
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    @property
    def lam(self) -> float:
        return self.K - 2.0 * self.G / 3.0


@dataclass(frozen=True)
class ReturnMap:
    sigma: np.ndarray      # (3,3)
    eps_p: np.ndarray      # (3,3)，迹为 0
    p: float
    dgamma: float
    plastic: bool


def load_j2_params(path) -> "J2Params":
    """从材料 yaml（configs/material_al1350.yaml）读 E、nu 与 plastic 段 -> J2Params。

    缺键、或 plastic.H 为 null 都直接报错（硬约束 3：不给默认值）。
    YAML 1.1 会把不带指数符号的 500e6 读成字符串，这里一律 float() 强制转换。
    """
    import pathlib

    import yaml

    raw = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    for top in ("elastic", "plastic"):
        if top not in raw:
            raise KeyError("%s 缺少顶层键 %r" % (path, top))
    pl = raw["plastic"]
    if pl.get("model") != "J2_isotropic_linear_hardening":
        raise ValueError("plastic.model 必须是 J2_isotropic_linear_hardening，得到 %r" % (pl.get("model"),))
    for k in ("sigma_y0", "H"):
        if pl.get(k) is None:
            raise ValueError("%s 的 plastic.%s 未给值" % (path, k))
    return J2Params(E=float(raw["elastic"]["E"]), nu=float(raw["elastic"]["nu"]),
                    sigma_y0=float(pl["sigma_y0"]), H=float(pl["H"]))


_I = np.eye(3)


def _dev(a: np.ndarray) -> np.ndarray:
    return a - np.trace(a) / 3.0 * _I


def _trial(eps, eps_p_n, prm: J2Params):
    ee = np.asarray(eps, float) - np.asarray(eps_p_n, float)
    s_tr = 2.0 * prm.G * _dev(ee)
    sigma_eq_tr = np.sqrt(1.5) * np.linalg.norm(s_tr)
    return ee, s_tr, sigma_eq_tr


def _check_degradation(g: float) -> float:
    if not 0.0 <= g <= 1.0:
        raise ValueError("损伤退化因子 g 须在 [0, 1] 内，得到 %r" % (g,))
    return float(g)


def return_map(eps, eps_p_n, p_n: float, prm: J2Params, g: float = 1.0) -> ReturnMap:
    """弹性试探 -> 判屈服 -> 径向返回（Simo & Hughes 1998, ch. 3）。

    ``g`` 为损伤退化因子 g(d)（D-62，B 口径）：屈服按**实际（退化后）**应力判断，
        F = g · sigma_eq(s_tr) − (sigma_y0 + H p_n) ≤ 0，Δγ = (g σ_eq,tr − σ_y − H p_n)/(3G g + H)。
    g = 1 时即无损伤的经典式。注意：返回的 ``sigma`` 是**未退化**的弹塑性应力；
    退化后用于平衡的应力由能量 ∂E/∂ε_e 给出（见 PLAN §1.5），不是 ``sigma`` 乘 g。
    """
    g = _check_degradation(g)
    ee, s_tr, sig_eq_tr = _trial(eps, eps_p_n, prm)
    f_tr = g * sig_eq_tr - (prm.sigma_y0 + prm.H * p_n)
    pressure_part = prm.K * np.trace(ee) * _I
    if f_tr <= 0.0:
        return ReturnMap(pressure_part + s_tr, np.array(eps_p_n, float), float(p_n), 0.0, False)
    dgamma = f_tr / (3.0 * prm.G * g + prm.H)
    n = s_tr / np.linalg.norm(s_tr)
    s = s_tr - 2.0 * prm.G * dgamma * np.sqrt(1.5) * n
    eps_p = np.asarray(eps_p_n, float) + dgamma * np.sqrt(1.5) * n
    return ReturnMap(pressure_part + s, eps_p, float(p_n + dgamma), float(dgamma), True)


def consistent_tangent(eps, eps_p_n, p_n: float, prm: J2Params, g: float = 1.0) -> np.ndarray:
    """一致弹塑性切线 C_ep = d sigma / d eps，(3,3,3,3)（Simo & Hughes 1998, ch. 3）。

        弹性步：C = K 1⊗1 + 2G (I_sym − 1⊗1/3)
        塑性步：C = K 1⊗1 + 2G θ (I_sym − 1⊗1/3) − 2G θ̄ n⊗n
        θ = 1 − 3G dgamma / sigma_eq_tr，  θ̄ = 3Gg/(3Gg+H) − (1 − θ)

    退化屈服（g < 1）只改 dgamma 对 sigma_eq_tr 的导数：g/(3Gg+H)，故 θ̄ 的第一项带 g。
    """
    g = _check_degradation(g)
    ee, s_tr, sig_eq_tr = _trial(eps, eps_p_n, prm)
    f_tr = g * sig_eq_tr - (prm.sigma_y0 + prm.H * p_n)
    I_sym = 0.5 * (np.einsum("ik,jl->ijkl", _I, _I) + np.einsum("il,jk->ijkl", _I, _I))
    vol = np.einsum("ij,kl->ijkl", _I, _I)
    dev_proj = I_sym - vol / 3.0
    G = prm.G
    if f_tr <= 0.0:
        return prm.K * vol + 2.0 * G * dev_proj
    dgamma = f_tr / (3.0 * G * g + prm.H)
    theta = 1.0 - 3.0 * G * dgamma / sig_eq_tr
    theta_bar = 3.0 * G * g / (3.0 * G * g + prm.H) - (1.0 - theta)
    n = s_tr / np.linalg.norm(s_tr)
    return (prm.K * vol + 2.0 * G * theta * dev_proj
            - 2.0 * G * theta_bar * np.einsum("ij,kl->ijkl", n, n))


# =============================================================================
# UFL / FE 部分：返回映射写成 UFL 表达式，一致切线由自动微分给出
# =============================================================================

#: 防 |s_tr| = 0 处 sqrt 求导出 nan：|s| = sqrt(s:s + delta^2)，delta 取 1e-14 sigma_y0，
#: 相对偏差 ~1e-28，远低于所有测试容差。
_NORM_SOFTENING = 1.0e-14


@dataclass
class UFLUpdate:
    sigma: object          # 2x2（平面内分量）
    sigma3: object         # 3x3（含 sigma_zz）
    ep_new: object         # 3 分量向量 (xx, yy, xy)
    p_new: object
    ee3: object            # 3x3 弹性应变（含 eps_e,zz = -eps_p,zz）
    dgamma: object


def _embed_sym(a_xx, a_yy, a_xy, a_zz):
    import ufl
    return ufl.as_tensor([[a_xx, a_xy, 0.0], [a_xy, a_yy, 0.0], [0.0, 0.0, a_zz]])


def stress_update_ufl(eps2, ep_n, p_n, prm: J2Params, g=1.0) -> UFLUpdate:
    """平面应变返回映射的 UFL 表达式。

    eps2：2x2 总应变（UFL）；ep_n：3 分量 (xx, yy, xy) 的历史塑性应变；p_n：历史 p；
    g：损伤退化因子 g(d)，可为常数或 UFL 表达式（D-62，见 ``return_map``）。
    ε_zz = 0、塑性不可压 ⇒ ε_p,zz = −(ε_p,xx + ε_p,yy)，ε_e,zz = −ε_p,zz。
    判屈服用 ufl.conditional；对它求导（derivative）即得一致切线，不需要手推进求解器。
    """
    import ufl
    G, K = prm.G, prm.K
    ep_zz = -(ep_n[0] + ep_n[1])
    eps3 = _embed_sym(eps2[0, 0], eps2[1, 1], eps2[0, 1], 0.0)
    ep3 = _embed_sym(ep_n[0], ep_n[1], ep_n[2], ep_zz)
    ee3 = eps3 - ep3
    I3 = ufl.Identity(3)
    tr = ufl.tr(ee3)
    s_tr = 2.0 * G * (ee3 - tr / 3.0 * I3)
    delta = _NORM_SOFTENING * prm.sigma_y0
    norm = ufl.sqrt(ufl.inner(s_tr, s_tr) + delta**2)
    f_tr = 1.5**0.5 * norm * g - (prm.sigma_y0 + prm.H * p_n)
    dgamma = ufl.conditional(ufl.gt(f_tr, 0.0), f_tr / (3.0 * G * g + prm.H), 0.0)
    n = s_tr / norm
    s = s_tr - 2.0 * G * 1.5**0.5 * dgamma * n
    sigma3 = K * tr * I3 + s
    ep3_new = ep3 + 1.5**0.5 * dgamma * n
    ep_new = ufl.as_vector([ep3_new[0, 0], ep3_new[1, 1], ep3_new[0, 1]])
    sigma2 = ufl.as_tensor([[sigma3[0, 0], sigma3[0, 1]], [sigma3[1, 0], sigma3[1, 1]]])
    return UFLUpdate(sigma2, sigma3, ep_new, p_n + dgamma, ee3, dgamma)


class PlasticLaw:
    """J2 塑性的 FE 载体：历史量存 DG-2（PLAN §2.3、D-7，不用 quadrature 元，以便 adios4dolfinx 续跑）。

    用法：残差里用 ``law.sigma(u)``，``law.dx`` 作积分测度；每个载荷步 Newton 收敛后调
    ``law.commit(u)`` 把 (ε_p, p) 更新到历史。
    **近似**：返回映射在积分点用 DG-2 插值的历史量，不是逐积分点的精确历史（D-7 的代价），
    其影响由 5.4 的网格加密检验量出。
    """

    def __init__(self, mesh, prm: J2Params, *, degree: int = 2, quad_degree: int = 4):
        import ufl
        from dolfinx import fem
        self.mesh, self.prm = mesh, prm
        self.V_ep = fem.functionspace(mesh, ("DG", degree, (3,)))
        self.V_p = fem.functionspace(mesh, ("DG", degree))
        self.ep_n, self.p_n = fem.Function(self.V_ep), fem.Function(self.V_p)
        self.ep_n.name, self.p_n.name = "eps_p", "p_eq"
        self.dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": quad_degree})

    @staticmethod
    def strain(u):
        import ufl
        return ufl.sym(ufl.grad(u))

    def update(self, u, g=1.0) -> UFLUpdate:
        return stress_update_ufl(self.strain(u), self.ep_n, self.p_n, self.prm, g=g)

    def sigma(self, u, g=1.0):
        return self.update(u, g).sigma

    def elastic_strain(self, u, g=1.0):
        """ε_e = ε − ε_p（2x2 平面内分量），给 Phase 6 的 ψ⁺ 用。g 同 ``update``。"""
        import ufl
        e = self.update(u, g).ee3
        return ufl.as_tensor([[e[0, 0], e[0, 1]], [e[1, 0], e[1, 1]]])

    def commit(self, u, cells=None, g=1.0) -> None:
        """收敛后把本步 (ε_p, p) 写回历史（在 DG 节点处对表达式插值）。

        cells 给定时只更新这些单元（多体问题里只更新塑性体，别的体的历史保持为 0）。
        g 为本步的损伤退化因子（UFL 或常数），须与求解时用的一致。
        """
        from dolfinx import fem
        upd = self.update(u, g)
        ep_tmp, p_tmp = fem.Function(self.V_ep), fem.Function(self.V_p)
        kw = {} if cells is None else {"cells0": cells}
        ep_tmp.interpolate(fem.Expression(upd.ep_new, self.V_ep.element.interpolation_points), **kw)
        p_tmp.interpolate(fem.Expression(upd.p_new, self.V_p.element.interpolation_points), **kw)
        # 先算两者再一起写回，避免 p_new 读到已更新的 ep_n/p_n
        self.ep_n.x.array[:] = ep_tmp.x.array
        self.p_n.x.array[:] = p_tmp.x.array
        self.ep_n.x.scatter_forward()
        self.p_n.x.scatter_forward()

    def fields(self) -> dict:
        return {"eps_p": self.ep_n, "p_eq": self.p_n}
