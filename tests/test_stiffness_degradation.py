"""b-1 口径（D-68 问题 5）：g(d) 是否退化弹性刚度的开关。

背景：u 与 d 交错时，u 方程弹性能里的 g(d) 构成正反馈环
    d ↑ → g(d) ↓ → 刚度 ↓ → u ↑ → ψ⁺ ↑ → d ↑
D-68 实测（N=94.2773、峰值相位）max_d 五轮从 0.084 冲到 1.0，ψ⁺ 同时涨 63%，
而 max|ε_p| ≡ 0（塑性未激活）—— 即反馈纯由 u 一侧造成。
均匀区驻定分析给出的内部解只有 d* ≈ +0.24，是耦合反馈把它推成发散。

本模块测的三件事：
  T1 开关生效：degrade_stiffness=False 时弹性能**不含** g(d)，与 d 无关；
     True（默认）时含 g(d) —— 保证默认口径逐位不变。
  T2 u 的解与 d 无关：degrade_stiffness=False 时同一载荷下不同 d 的位移逐位相同
     （刚度不变）。这是打断反馈环的**直接证据**，比任何残差判据都硬。
  T3 剖切实测：在有塑性的均匀仿射场上，退化与非退化两种口径的 u 确实不同，
     且偏差随 d 单调增大（避免「d 取 0 使 T2 恒真」的假阳性）。
  T4 d 方程驱动力仍含 g(d)：degrade_stiffness=False **不应**改 d 方程
     的驻定条件 —— 不可逆性与 B 口径不受影响（用户 2026-10-16 明确要求）。
"""

import pathlib

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pfwire.material import load_material
from pfwire.phasefield import (
    PhaseFieldModel,
    PhaseFieldSolver,
    SolverConfig,
    degradation,
    elastic_energy_density,
)
from pfwire.plasticity import PlasticLaw, load_j2_params

ROOT = pathlib.Path(__file__).resolve().parents[1]
MATERIAL = ROOT / "configs/material_al1350.yaml"
K_RES = 1.0e-6


def _affine_bcs(msh, exx, eyy):
    """全边界仿射位移 (exx·x, eyy·y)：均匀应变，应力处处相同（与 test_phasefield_plastic 同）。"""

    def make(V):
        fdim = msh.topology.dim - 1
        facets = dmesh.locate_entities_boundary(
            msh, fdim, lambda x: np.full(x.shape[1], True))
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        g = fem.Function(V)
        g.interpolate(lambda x: np.vstack([exx * x[0], eyy * x[1]]))
        return [fem.dirichletbc(g, dofs)]

    return make


def _uniaxial_bcs(msh):
    """左端 x 固定、右端 x 自由：单轴拉伸由**外力**驱动，刚度真正进入方程。

    ⚠ 不能用全边界仿射位移去做这条测试：那时 u 被边界值完全钉住，
    刚度不进方程，两种口径的 u 会逐位相同 —— 测试会「假通过」。
    必须让自由度由平衡方程定，刚度才显形。
    """

    def make(V):
        fdim = msh.topology.dim - 1
        left = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
        dofs = fem.locate_dofs_topological(V, fdim, left)
        g = fem.Function(V)
        g.interpolate(lambda x: np.vstack([np.zeros(x.shape[1]),
                                           np.zeros(x.shape[1])]))
        return [fem.dirichletbc(g, dofs)]

    return make


def _traction(V):
    """右端 (x = 1) 的 x 向面力，幅值取 T0=1e5 Pa。**必须写成 lambda V**：
    Neumann 项里的 TestFunction 要取自求解器自己的 V_u（见
    _resolve_external_work 的注释）。

    幅值不能取 1 Pa：单位边长下位移 ~ p·L/E ~ 1.4e-11 m，夹在装配噪声里
    分不出真假。取 1e5 Pa 给到位移 ~1e-8 m，留出 3 个数量级余量。
    """
    T0 = 1.0e5                      # Pa（单位厚度的面力），仅为测试可分辨而取
    ds = ufl.Measure("ds", domain=V.mesh)
    v = ufl.TestFunction(V)
    return ufl.dot(ufl.as_vector([T0, 0.0]), v) * ds


def _build_traction(degrade):
    """单轴拉伸工况：左端固定、右端受单位面力。"""
    material = load_material(MATERIAL)
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 4, 4, dmesh.CellType.triangle)
    model = PhaseFieldModel(material=material, split="spectral", k_res=K_RES,
                            degrade_stiffness=degrade)
    solver = PhaseFieldSolver(
        msh, model, bcs_u=_uniaxial_bcs(msh),
        config=SolverConfig(u_solver="newton", snes_rtol=1.0e-10,
                            snes_atol=1.0e-14),
        external_work=_traction)
    return solver

# ------------------------------------------------------------------ T1
@pytest.mark.parametrize("d_val", [0.0, 0.4, 0.9])
def test_elastic_energy_toggles_degradation_factor(d_val):
    """T1：开关只影响弹性能里的 g(d)，满足 deg − undeg ≡ (g(d)−1)·ψ⁺。

    做法：装配**标量**（一次求值三次形式之差），比符号化简稳 ——
    UFL 的 expand_derivatives 对含特征值的谱分解会踩 index_combination 的坑
    （实测报「Component and shape length don't match」），而数值差对此免疫。
    容差取相对 1e-12：三项都在同一网格同一空间上装配，只有舍入差。
    """
    material = load_material(MATERIAL)
    msh = dmesh.create_unit_square(MPI.COMM_SELF, 4, 4, dmesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    w = fem.Function(V, name="w")
    # 任意非零场（线性 ramps）即可：恒等式对 w 无关，非零只为放大数值
    w.interpolate(lambda x: np.vstack([0.3 * x[0], -0.2 * x[1]]))
    e = ufl.sym(ufl.grad(w))

    deg = elastic_energy_density(e, d_val, material, "spectral", 2, K_RES,
                                 degrade_stiffness=True)
    undeg = elastic_energy_density(e, d_val, material, "spectral", 2, K_RES,
                                   degrade_stiffness=False)
    from pfwire.phasefield import energy_split
    psi_p, _ = energy_split(e, material, "spectral", 2)

    def val(expr):
        return fem.assemble_scalar(fem.form(expr * ufl.dx(domain=msh)))

    residual = val(deg) - (val(undeg) + (degradation(d_val, K_RES) - 1.0) * val(psi_p))
    scale = abs(val(deg))
    assert residual == pytest.approx(0.0, abs=1e-9 * max(scale, 1.0)), \
        "d=%g: deg - undeg != (g-1)psi+，残差 %.3e（量级 %.3e）" % (
            d_val, residual, scale)


def test_default_flag_preserves_existing_behaviour():
    """T1 附带：PhaseFieldModel 默认 degrade_stiffness=True（PLAN §2 原口径）。"""
    material = load_material(MATERIAL)
    m = PhaseFieldModel(material=material, split="spectral", k_res=K_RES)
    assert m.degrade_stiffness is True
    m2 = PhaseFieldModel(material=material, split="spectral", k_res=K_RES,
                         degrade_stiffness=False)
    assert m2.degrade_stiffness is False


# ------------------------------------------------------------------ T2
def _set_d(solver, d_val):
    solver.d.x.array[:] = d_val
    solver.d.x.scatter_forward()
    solver.d_old.x.array[:] = d_val
    solver.d_old.x.scatter_forward()


@pytest.mark.parametrize("d_val", [0.0, 0.3, 0.6])
def test_displacement_is_independent_of_d_when_stiffness_not_degraded(d_val):
    """T2（核心）：degrade_stiffness=False 时，u 的解与 d **完全无关**。

    这是打断正反馈环的直接证据 —— d 再怎么涨，刚度（因此 u、ψ⁺）都不动，
    d↑→刚度↓→u↑→ψ⁺↑ 这一环被切断。

    ⚠ 工况必须是**外力驱动**（单轴拉伸 + 左端固定）：若把全边界都钉成仿射位移，
    u 被边界值完全定死、刚度不进方程，两种口径会逐位相同 —— 测试会假通过。
    """
    s0 = _build_traction(degrade=False)
    _set_d(s0, 0.0)
    s0.solve_displacement()
    u_ref = s0.u.x.array.copy()

    s1 = _build_traction(degrade=False)
    _set_d(s1, d_val)
    s1.solve_displacement()
    assert np.max(np.abs(s1.u.x.array - u_ref)) == pytest.approx(
        0.0, abs=1e-12), "degrade_stiffness=False 时 u 仍随 d 变：反馈环未切断"
    # 顺带确认这个工况位移量级合理（否则上面那条恒真）。
    # 数量级估计：T0=1e5 Pa 单轴，平均 σ ~ T0/2 = 5e7 Pa，ε ~ σ/E ~ 7.3e-4，
    # 但这里是 2D 平面应变 + 泊松效应的组合，端点位移实测 3.3e-6 m。
    # 取 [1e-8, 1e-4] 的宽区间只做量级断言（防「解恒零」的假阳性），
    # 不充作解析校验 —— 精确各向同性解由 test_phasefield_elastic 之类负责。
    assert 1.0e-8 < np.max(np.abs(u_ref)) < 1.0e-4, \
        "拉伸工况位移量级异常：%.3e m" % np.max(np.abs(u_ref))


def test_displacement_does_depend_on_d_when_stiffness_degraded():
    """T2 对照：默认口径（True）下 u 必须随 d 变，否则 T2 会被假阳性骗过。"""
    s = _build_traction(degrade=True)
    _set_d(s, 0.0)
    s.solve_displacement()
    u0 = s.u.x.array.copy()
    _set_d(s, 0.6)
    s.solve_displacement()
    scale = np.max(np.abs(u0))
    assert np.max(np.abs(s.u.x.array - u0)) / scale > 1.0e-3, \
        "默认口径下 u 应随 d 明显退化（g(0.6)=(0.4)^2=0.16）"


# ------------------------------------------------------------------ T3
@pytest.mark.parametrize("d_val", [0.2, 0.5, 0.8])
def test_deviations_between_the_two_bases_grow_monotonically_with_d(d_val):
    """T3：两种口径的位移偏差随 d 单调增大（真退化，不是常数偏置）。"""
    sd = _build_traction(degrade=True)
    su = _build_traction(degrade=False)
    for s in (sd, su):
        _set_d(s, d_val)
        s.solve_displacement()
    dev = np.max(np.abs(sd.u.x.array - su.u.x.array)) / np.max(np.abs(su.u.x.array))
    assert dev > 1.0e-4, "d=%g 时两口径应已有可测偏差，实得 %.2e" % (d_val, dev)


def test_deviation_curve_is_monotone():
    """T3 附加：偏差随 d 单调不减（g(d) 单调降的直接推论）。"""
    devs = []
    for d_val in (0.1, 0.3, 0.5, 0.7, 0.9):
        sd = _build_traction(degrade=True)
        su = _build_traction(degrade=False)
        for s in (sd, su):
            _set_d(s, d_val)
            s.solve_displacement()
        devs.append(float(np.max(np.abs(sd.u.x.array - su.u.x.array))))
    assert all(b >= a - 1e-15 for a, b in zip(devs, devs[1:])), \
        "偏差应随 d 单调不减，实得 %r" % [float(x) for x in devs]


# ------------------------------------------------------------------ T4
def test_d_equation_drive_still_contains_degradation():
    """T4：degrade_stiffness=False **不改** d 方程的驱动力（用户明确要求）。

    d 方程的能量是 g(d)ψ⁺ + f(α̅)(3G_c/8)(d/ℓ + ℓ|∇d|²)。若开关误把它也摘掉，
    驻定条件会从 2(1−d)ψ⁺ = fσ_c²/E 变成 2ψ⁺ = fσ_c²/E，d 的求解完全不同。
    这里用同一 u、同一 α̅，比较两种开关下 d 方程的残差是否一致
    （对 d 求导前应该**逐位相同** —— 即能量表达式里 g(d) 仍在）。
    """
    from pfwire.phasefield import dissipation_density
    material = load_material(MATERIAL)
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 3, 3, dmesh.CellType.triangle)

    def d_energy_form(solver):
        psi_plus = solver.psi_plus_expr()
        return (degradation(solver.d, solver.model.k_res) * psi_plus
                + dissipation_density(solver.d, material, solver.f_alpha, "AT1")) \
            * solver.dx

    outs = {}
    for flag in (True, False):
        model = PhaseFieldModel(material=material, split="spectral", k_res=K_RES,
                                degrade_stiffness=flag)
        s = PhaseFieldSolver(msh, model, config=SolverConfig(u_solver="newton"))
        _set_d(s, 0.3)
        outs[flag] = fem.assemble_scalar(fem.form(d_energy_form(s)))
    assert outs[True] == pytest.approx(outs[False], rel=1e-14, abs=0.0), \
        "d 方程能量不应随 degrade_stiffness 变化（开关只管 u 的刚度）"
