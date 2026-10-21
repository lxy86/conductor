"""6.5 基准工况的端部载荷：轴向张力 T' 与往复弯矩 M(t)（D-67 §3）。

端面（左/右）施加线性分布的轴向面力

    t_x(y) = T'/H + 12 M(t) (y − H/2) / H³        （每单位厚度，Pa）

其中 y 从底面（y = 0）量起、H 为单丝厚度（D-67 的 y 记号；网格坐标是上表面 y = 0、
底面 y = −H，故 y_grid = y − H）。这个分布对端面的静力等效力系正是「轴力 T' +
弯矩 M(t)」：∫t_x dy = T'、∫t_x (y − H/2) dy = M(t)，由 T3b 用 Simpson 积分钉住。

弯矩幅由目标远场弯曲应力幅反解（D-67 §3）：

    M_a = σ_a H²/6

σ_a 由 `pfwire.loads` 的 P–S 函数给出（`stress_from_bending_amplitude`），本模块
不接受 Y_b，只接受 σ_a —— P–S 是另一层的标定，混进来会让 T3 的判据不纯。

为什么要做成类而不是两个自由函数：`t_x` 里的 M = phase·M_a 要能**只改 Constant
就切换相位**，否则每换一个相位都要重新编译 form（D-16 的教训：一次编译数秒到
数十秒，而疲劳循环要跑上万次相位切换）。

单位：SI。
"""

from __future__ import annotations

import math
from collections.abc import Callable

import ufl
from dolfinx import default_scalar_type, fem

from pfwire.mesh import BLOCK_LEFT, BLOCK_RIGHT

__all__ = [
    "bending_moment_amplitude",
    "end_traction_x",
    "ClampLoads",
]


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("%s 必须为有限正数，收到 %r" % (name, value))
    return value


def bending_moment_amplitude(sigma_a: float, H: float) -> float:
    """M_a = σ_a H²/6（每单位厚度，N）。D-67 §3。

    出处：纯弯曲的截面正应力 σ_xx = M y / I，矩形截面（单位厚度）I = H³/12，
    表面 y = H/2 处 σ_a = M_a (H/2)/(H³/12) = 6 M_a/H²，反解即得。
    """
    return float(sigma_a) * _positive(H, "H") ** 2 / 6.0


def end_traction_x(T, M, H: float, y):
    """t_x(y) = T/H + 12 M (y − H/2)/H³（Pa，每单位厚度）。

    ``y`` 从**底面**量起（0 ≤ y ≤ H），与 D-67 §3 的记号一致。可为 float、
    numpy 数组或 UFL 表达式（因此可直接写进 Neumann 形式）。

    ⚠ ``T`` 与 ``M`` **不得 float()**：它们可能是 UFL 表达式（M = phase·M_a，
    phase 是 fem.Constant）。早期版本对它们做了 float()，把 Constant 的**构造时
    值**烘成字面量，于是 set_phase 改 .value 对已编译的 form 毫无影响——载荷
    完全冻结，psi+ 不随相位变化、alpha_bar 永不累积（实测相位 0 与 0.25 的
    |F_ext| 都是 4358.9）。只有 ``H`` 是真正的几何常数，可以对它做校验。
    """
    H = _positive(H, "H")
    return T / H + 12.0 * M * (y - 0.5 * H) / H**3


class ClampLoads:
    """右端（默认）的轴向张力 + 往复弯矩，以端面 Neumann 形式给出。

    参数
    ----
    mesh, facet_tags : 单丝网格与面 tag（``BLOCK_RIGHT`` / ``BLOCK_LEFT`` 为端面）。
    H                : 单丝厚度（m）。
    sigma_a          : 目标远场弯曲应力幅（Pa），由 P–S 给出。
    T                : 轴向张力 T'（N/m，单位厚度）。
    side             : "right" 或 "left"，施加载荷的端面。

    用法::

        loads = ClampLoads(mesh, ft, H=3.75e-3, sigma_a=sigma_a, T=T_prime)
        loads.set_phase(+1.0)
        solver = PhaseFieldSolver(..., external_work=loads.traction())
    """

    def __init__(self, mesh, facet_tags, *, H: float, sigma_a: float, T: float,
                 side: str = "right", self_equilibrated: bool = False) -> None:
        """``self_equilibrated=True`` 时弯矩与轴力在**两端反向**施加（自平衡）。

        为什么要这个选项：单端加载 +「左端 u_x = 0」的约束下，块体近似自由，
        罚接触力把块推走、压力建立不起来（实测 W 只有 6.1 底面固定时的 1/23）。
        而弯矩若靠「两端钉死」来提供反力矩，端部转动被约束掉，又测不到 σ_a。
        自平衡的两端加载让**载荷自身**合力与合力矩为零，于是底面可以全固定
        （与 6.1 的 ObstacleContact 一致，接触压力能正常建立），同时端部转动
        不受约束（弯曲照常）。左端面力取反号：n = (−1, 0)。
        """
        self.mesh = mesh
        self.facet_tags = facet_tags
        self.H = _positive(H, "H")
        self.sigma_a = float(sigma_a)
        self.T = float(T)
        if side not in ("right", "left"):
            raise ValueError("side 只能取 'right' 或 'left'，收到 %r" % (side,))
        self.side = side
        self.self_equilibrated = bool(self_equilibrated)
        self.tag = BLOCK_RIGHT if side == "right" else BLOCK_LEFT
        self.tag_opposite = BLOCK_LEFT if side == "right" else BLOCK_RIGHT
        self.M_a = bending_moment_amplitude(self.sigma_a, self.H)
        # 相位做成 Constant：切换相位只改 .value，不重编译（D-67 §3，D-16）
        self.phase = fem.Constant(mesh, default_scalar_type(0.0))
        self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    def set_phase(self, phase: float) -> None:
        """设置往复相位 ∈ [−1, 1]：M = phase · M_a。"""
        phase = float(phase)
        if not -1.0 <= phase <= 1.0:
            raise ValueError("phase 应在 [-1, 1] 内，收到 %r" % (phase,))
        self.phase.value = phase

    @property
    def moment(self) -> float:
        """当前弯矩 M = phase · M_a（N，单位厚度）。"""
        return float(self.phase.value) * self.M_a

    @property
    def moment_amplitude(self) -> float:
        """弯矩幅 M_a（N，单位厚度）。"""
        return self.M_a

    def traction_x(self, y_grid) -> ufl.core.expr.Expr:
        """端面面力的 UFL 表达式：y_grid 为网格坐标（上表面 0、底面 −H）。

        网格坐标 y_grid 与 D-67 的 y 差一个平移：y = y_grid + H。
        """
        return end_traction_x(self.T, self.phase * self.M_a, self.H,
                              y_grid + self.H)

    def traction_with(self, v) -> ufl.Form:
        """直接用给定的**测试函数** v 构造端部面力形式（monolithic 用，D-68 方案 a）。

        为什么需要它：``traction()`` 的签名是 ``lambda V: Form``，它自己从 V 取
        TestFunction。混合空间下 ``W.sub(0)`` 的 TestFunction 与
        ``ufl.split(TestFunction(W))[0]`` 是两个编号相同却不同源的 Argument，
        UFL 会报 "Found different Arguments with same number and part"
        （与 BC 那个陷阱同源）。故 monolithic 必须把 v 直接传进来。

        ``v`` 可以是向量（2D 位移）测试函数的任意分量表达式。
        """
        y_grid = ufl.SpatialCoordinate(self.mesh)[1]
        tx = self.traction_x(y_grid)
        form = ufl.dot(ufl.as_vector([tx, 0.0]), v) * self.ds(self.tag)
        if self.self_equilibrated:
            form = form + ufl.dot(ufl.as_vector([-tx, 0.0]), v) * self.ds(
                self.tag_opposite)
        return form

    def traction(self) -> Callable:
        """返回 ``lambda V: ufl.Form``，供 ``PhaseFieldSolver(external_work=...)`` 用。

        必须写成「拿到 V 再构造」的可调用对象：TestFunction 要取自求解器自己的
        ``V_u``，在外部另建一个结构相同的空间，UFL 会看到两个编号相同却不同源的
        Argument 并报错（与 BC 那个陷阱同源，见 phasefield.BCSpec 的注释）。
        """
        y_grid = ufl.SpatialCoordinate(self.mesh)[1]

        def make(V):
            v = ufl.TestFunction(V)
            tx = self.traction_x(y_grid)
            form = ufl.dot(ufl.as_vector([tx, 0.0]), v) * self.ds(self.tag)
            if self.self_equilibrated:
                # 对面施加**反向**的同一分布：合力与合力矩都抵消，整体自平衡
                form = form + ufl.dot(ufl.as_vector([-tx, 0.0]), v) * self.ds(
                    self.tag_opposite)
            return form

        return make
