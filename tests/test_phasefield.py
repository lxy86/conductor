"""Phase 1.1 单测：AT1 泛函、拉压分解、交错求解、SNES 不可逆。

测试意图
--------
A. 分解的数学性质（不依赖求解器，直接在给定均匀应变场上积分）
   1. 能量一致性 psi+ + psi- = psi（两种分解、随机应变）；
   2. 纯静水压缩 -> psi+ = 0；纯静水拉伸 -> psi- = 0（amor 与 spectral 各自的判据）；
   3. 各向同性（frame indifference）：旋转应变张量后 psi± 不变；
   4. 冻结形式在不动点处等于精确形式（linear 路径合法性的前提）。
B. 求解器（均匀单轴应变算例，有闭式解）
   5. **主物理锚点**：AT1 均匀解 d = max(0, 1 - (sigma_c/sigma)^2)。
      这条同时验证了 Gc/ell/sigma_c 三角、退化函数、耗散项系数 3Gc/8 和 SNES 界约束；
   6. 不可逆：加载到 d>0 后卸载，d 不得回落；
   7. 界约束：0 <= d <= 1 恒成立；
   8. linear（冻结 chi）与 newton（精确残差）在拉压混合场上给出同一个 u；
   9. 不收敛必须显式报告（max_iter=1 时 converged=False 且 failure_mode 非空），
      不允许静默返回坏解。
"""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pfwire import material as mat
from pfwire import phasefield as pf

# --------------------------------------------------------------------------
# 公共夹具
# --------------------------------------------------------------------------
E_AL, NU_AL = 69.0e9, 0.33
SIGMA_C, ELL = 185.0e6, 4.0e-6


def make_material(nu: float = NU_AL, E: float = E_AL) -> mat.Material:
    """构造材料卡（绕开 yaml，便于把 nu 设成 0 做解析对照）。"""
    fracture = mat.resolve_fracture_properties(
        E, nu, mode="from_sigma_c_and_ell", sigma_c=SIGMA_C, ell=ELL
    )
    return mat.Material(
        name="test-al",
        elastic=mat.ElasticProperties(E=E, nu=nu),
        fracture=fracture,
        sigma_u=SIGMA_C,
        alpha_T=None,
        sources={},
    )


def unit_mesh(n: int = 2) -> dmesh.Mesh:
    return dmesh.create_unit_square(MPI.COMM_WORLD, n, n, dmesh.CellType.triangle)


def uniform_strain_field(msh: dmesh.Mesh, eps: np.ndarray) -> fem.Function:
    """构造位移 u = A x（A = eps 对称），使 sym(grad u) 恰为给定常应变。"""
    eps = np.asarray(eps, dtype=float)
    assert np.allclose(eps, eps.T), "应变张量必须对称"
    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    u = fem.Function(V)
    u.interpolate(
        lambda x: np.vstack(
            [
                eps[0, 0] * x[0] + eps[0, 1] * x[1],
                eps[1, 0] * x[0] + eps[1, 1] * x[1],
            ]
        )
    )
    return u


def mean_value(expr, msh: dmesh.Mesh) -> float:
    """expr 在网格上的平均值（均匀场时即其点值）。"""
    dx = ufl.Measure("dx", domain=msh)
    total = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(expr * dx)), op=MPI.SUM
    )
    area = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * dx)), op=MPI.SUM
    )
    return total / area


def psi_undamaged(eps: np.ndarray, material: mat.Material) -> float:
    """无损伤弹性能密度 0.5 lambda tr^2 + mu eps:eps（numpy 侧独立算）。"""
    lmbda, mu = mat.lame_parameters(material.elastic.E, material.elastic.nu)
    tr = np.trace(eps)
    return 0.5 * lmbda * tr**2 + mu * np.einsum("ij,ij", eps, eps)


RANDOM_STRAINS = [
    np.array([[1.0e-3, 0.0], [0.0, 1.0e-3]]),        # 静水拉伸
    np.array([[-1.5e-3, 0.0], [0.0, -1.5e-3]]),      # 静水压缩
    np.array([[2.0e-3, 5.0e-4], [5.0e-4, -1.0e-3]]),  # 拉压混合 + 剪切
    np.array([[-2.0e-3, 1.0e-3], [1.0e-3, 5.0e-4]]),  # 反号混合
    np.array([[0.0, 1.2e-3], [1.2e-3, 0.0]]),        # 纯剪
]


# --------------------------------------------------------------------------
# A. 分解的数学性质
# --------------------------------------------------------------------------
@pytest.mark.parametrize("split", pf.SPLITS)
@pytest.mark.parametrize("eps", RANDOM_STRAINS)
def test_split_energy_consistency(split: str, eps: np.ndarray):
    """psi+ + psi- 必须等于无损伤弹性能。"""
    msh = unit_mesh()
    material = make_material()
    u = uniform_strain_field(msh, eps)
    psi_p, psi_m = pf.energy_split(pf.strain(u), material, split, 2)
    total = mean_value(psi_p + psi_m, msh)
    assert total == pytest.approx(psi_undamaged(eps, material), rel=1e-10)


def test_amor_hydrostatic_compression_has_no_driving_force():
    """静水压缩：dev = 0 且 tr < 0，故 psi+ 应为 0（接触区不该有假损伤驱动力）。"""
    msh = unit_mesh()
    material = make_material()
    eps = np.array([[-1.5e-3, 0.0], [0.0, -1.5e-3]])
    u = uniform_strain_field(msh, eps)
    psi_p, psi_m = pf.energy_split(pf.strain(u), material, "amor", 2)
    assert mean_value(psi_p, msh) == pytest.approx(0.0, abs=1e-6)
    assert mean_value(psi_m, msh) == pytest.approx(
        psi_undamaged(eps, material), rel=1e-10
    )


def test_amor_hydrostatic_tension_is_fully_positive():
    msh = unit_mesh()
    material = make_material()
    eps = np.array([[1.0e-3, 0.0], [0.0, 1.0e-3]])
    u = uniform_strain_field(msh, eps)
    psi_p, psi_m = pf.energy_split(pf.strain(u), material, "amor", 2)
    assert mean_value(psi_m, msh) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("sign,expect_zero", [(-1.0, "plus"), (1.0, "minus")])
def test_spectral_split_full_compression_or_tension(sign: float, expect_zero: str):
    """spectral：所有主应变同号时，另一半能量必须为 0。"""
    msh = unit_mesh()
    material = make_material()
    eps = sign * np.array([[2.0e-3, 3.0e-4], [3.0e-4, 1.0e-3]])
    u = uniform_strain_field(msh, eps)
    assert np.all(np.linalg.eigvalsh(eps) * sign > 0)  # 前提：主应变同号
    psi_p, psi_m = pf.energy_split(pf.strain(u), material, "spectral", 2)
    zero, full = (psi_p, psi_m) if expect_zero == "plus" else (psi_m, psi_p)
    assert mean_value(zero, msh) == pytest.approx(0.0, abs=1e-6)
    assert mean_value(full, msh) == pytest.approx(
        psi_undamaged(eps, material), rel=1e-10
    )


@pytest.mark.parametrize("split", pf.SPLITS)
def test_split_is_frame_indifferent(split: str):
    """旋转应变张量后 psi± 不变（各向同性）。"""
    msh = unit_mesh()
    material = make_material()
    eps = RANDOM_STRAINS[2]
    theta = 0.37
    R = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    eps_rot = R @ eps @ R.T

    values = []
    for tensor in (eps, eps_rot):
        u = uniform_strain_field(msh, tensor)
        psi_p, psi_m = pf.energy_split(pf.strain(u), material, split, 2)
        values.append((mean_value(psi_p, msh), mean_value(psi_m, msh)))
    for got, ref in zip(values[1], values[0]):
        assert got == pytest.approx(ref, rel=1e-9, abs=1e-6)


@pytest.mark.parametrize("eps", RANDOM_STRAINS)
def test_frozen_matches_exact_at_fixed_point(eps: np.ndarray):
    """冻结形式在 freeze_from == eps 时必须与精确形式重合（linear 路径的前提）。"""
    msh = unit_mesh()
    material = make_material()
    u = uniform_strain_field(msh, eps)
    exact = pf.energy_split(pf.strain(u), material, "amor", 2)
    frozen = pf.energy_split(
        pf.strain(u), material, "amor", 2, freeze_from=pf.strain(u)
    )
    for got, ref in zip(frozen, exact):
        assert mean_value(got, msh) == pytest.approx(mean_value(ref, msh), rel=1e-10, abs=1e-6)


def test_spectral_frozen_form_is_refused_not_silently_wrong():
    msh = unit_mesh()
    material = make_material()
    u = uniform_strain_field(msh, RANDOM_STRAINS[2])
    with pytest.raises(NotImplementedError):
        pf.energy_split(
            pf.strain(u), material, "spectral", 2, freeze_from=pf.strain(u)
        )


def test_degradation_function_properties():
    k_res = 1.0e-6
    assert pf.degradation(0.0, k_res) == pytest.approx(1.0 + k_res)
    assert pf.degradation(1.0, k_res) == pytest.approx(k_res)
    ds = np.linspace(0.0, 1.0, 21)
    g = np.array([pf.degradation(float(d), k_res) for d in ds])
    assert np.all(np.diff(g) < 0.0)  # 严格单调下降


def test_model_and_config_reject_bad_input():
    material = make_material()
    with pytest.raises(ValueError):
        pf.PhaseFieldModel(material, split="miehe")
    with pytest.raises(ValueError):
        pf.PhaseFieldModel(material, k_res=0.0)
    with pytest.raises(ValueError):
        pf.SolverConfig(u_solver="magic")
    with pytest.raises(ValueError):
        pf.SolverConfig(max_iter=0)


# --------------------------------------------------------------------------
# B. 求解器：均匀单轴应变算例
# --------------------------------------------------------------------------
class UniaxialStrainCase:
    """nu = 0 的平面应变方块，施加 eps = diag(e, 0) 的均匀应变。

    nu = 0 时 lambda = 0、2 mu = E，u = (e x, 0) 是精确解且保持均匀；
    此时 amor 分解给出 psi+ = 0.5 E e^2（体积项与偏量项各占一半），
    于是 AT1 的均匀解有闭式：d = max(0, 1 - (sigma_c/sigma)^2)，sigma = E e。
    """

    def __init__(self, n: int = 4, u_solver: str = "linear", max_iter: int = 50):
        self.material = make_material(nu=0.0)
        self.L = 10.0 * ELL
        self.msh = dmesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([0.0, 0.0]), np.array([self.L, self.L])],
            [n, n],
            cell_type=dmesh.CellType.triangle,
        )
        model = pf.PhaseFieldModel(self.material, split="amor")
        cfg = pf.SolverConfig(u_solver=u_solver, max_iter=max_iter)

        self.u_applied = fem.Constant(self.msh, default_scalar_type(0.0))

        # BC 用 callable 构造：必须建在求解器自己的 V_u 上，否则 dolfinx 静默忽略
        def make_bcs(V):
            fdim = self.msh.topology.dim - 1
            left = dmesh.locate_entities_boundary(
                self.msh, fdim, lambda x: np.isclose(x[0], 0.0)
            )
            right = dmesh.locate_entities_boundary(
                self.msh, fdim, lambda x: np.isclose(x[0], self.L)
            )
            horiz = dmesh.locate_entities_boundary(
                self.msh,
                fdim,
                lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], self.L),
            )
            Vx, Vy = V.sub(0), V.sub(1)
            zero = fem.Constant(self.msh, default_scalar_type(0.0))
            return [
                fem.dirichletbc(zero, fem.locate_dofs_topological(Vx, fdim, left), Vx),
                fem.dirichletbc(
                    self.u_applied,
                    fem.locate_dofs_topological(Vx, fdim, right),
                    Vx,
                ),
                fem.dirichletbc(
                    zero, fem.locate_dofs_topological(Vy, fdim, horiz), Vy
                ),
            ]

        self.make_bcs = make_bcs  # 供 test_cyclejump 复用同一套边界条件
        self.solver = pf.PhaseFieldSolver(self.msh, model, bcs_u=make_bcs, config=cfg)

    @property
    def strain_c(self) -> float:
        """起裂应变 eps_c = sigma_c/E。"""
        return self.material.fracture.sigma_c / self.material.elastic.E

    def apply_ratio(self, s: float) -> pf.StaggeredResult:
        """施加 sigma/sigma_c = s，返回交错求解结果。"""
        self.u_applied.value = s * self.strain_c * self.L
        return self.solver.solve_step()

    @staticmethod
    def analytic_damage(s: float) -> float:
        return max(0.0, 1.0 - 1.0 / s**2)


@pytest.mark.parametrize("s", [0.90, 0.99, 1.05, 1.30, 1.80])
def test_at1_homogeneous_damage_matches_closed_form(s: float):
    """主物理锚点：均匀 AT1 解 d = max(0, 1 - (sigma_c/sigma)^2)。"""
    case = UniaxialStrainCase()
    result = case.apply_ratio(s)
    assert result.converged, result.failure_mode
    expected = case.analytic_damage(s)
    got = case.solver.max_damage()
    if expected == 0.0:
        assert got == pytest.approx(0.0, abs=1e-12)  # 界约束把 d 压在 0
    else:
        assert got == pytest.approx(expected, rel=2e-3)
    # 均匀性：最大最小值应一致
    assert case.solver.min_damage() == pytest.approx(got, rel=1e-6, abs=1e-12)


def test_damage_stays_within_bounds():
    case = UniaxialStrainCase()
    for s in (0.5, 1.2, 2.5, 5.0):
        result = case.apply_ratio(s)
        assert result.converged, result.failure_mode
        assert case.solver.min_damage() >= -1e-12
        assert case.solver.max_damage() <= 1.0 + 1e-12
        case.solver.commit_step()


def test_irreversibility_damage_never_decreases():
    """加载到 d>0、commit、再卸载：d 必须被下界钉住不回落。"""
    case = UniaxialStrainCase()
    loaded = case.apply_ratio(1.8)
    assert loaded.converged, loaded.failure_mode
    d_peak = case.solver.max_damage()
    assert d_peak == pytest.approx(case.analytic_damage(1.8), rel=2e-3)

    case.solver.commit_step()
    unloaded = case.apply_ratio(0.3)  # 远低于起裂应力
    assert unloaded.converged, unloaded.failure_mode
    assert case.solver.max_damage() == pytest.approx(d_peak, rel=1e-9)
    assert case.solver.min_damage() == pytest.approx(d_peak, rel=1e-6)


def test_energies_are_positive_and_consistent():
    case = UniaxialStrainCase()
    result = case.apply_ratio(1.5)
    assert result.converged, result.failure_mode
    e = result.energies
    assert e["elastic"] > 0.0 and e["dissipated"] > 0.0
    assert e["total"] == pytest.approx(e["elastic"] + e["dissipated"], rel=1e-12)
    # 均匀场下耗散能可解析核对：f=1, d 均匀, grad d = 0
    #   E_diss = (3 Gc/8)(d/ell) * area
    d = case.solver.max_damage()
    Gc, ell = case.material.fracture.Gc, case.material.fracture.ell
    expected = 3.0 * Gc / 8.0 * d / ell * case.L**2
    assert e["dissipated"] == pytest.approx(expected, rel=1e-6)


def test_nonconvergence_is_reported_not_silent():
    """max_iter=1 不可能同时收敛 u 与 d：必须显式报未收敛，且带失败原因。"""
    case = UniaxialStrainCase(max_iter=1)
    result = case.apply_ratio(1.8)
    assert result.converged is False
    assert result.failure_mode and "未收敛" in result.failure_mode
    assert result.iterations == 1
    assert result.history  # 逐迭代记录仍在，便于排查


def test_linear_and_newton_u_solvers_agree_on_mixed_field():
    """拉压混合 + 有损伤时，冻结 chi 的 linear 路径应与精确 newton 路径同解。

    构造：方块四边给出一个使 tr eps 在域内变号的位移场（x 向拉、y 向压且带剪切），
    d 人为置为 0.5 使 g(d) < 1、分解真正起作用（d = 0 时分解不影响 u）。
    """
    material = make_material()
    L = 10.0 * ELL
    results = {}
    for u_solver in ("linear", "newton"):
        msh = dmesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([0.0, 0.0]), np.array([L, L])],
            [6, 6],
            cell_type=dmesh.CellType.triangle,
        )
        model = pf.PhaseFieldModel(material, split="amor")
        cfg = pf.SolverConfig(u_solver=u_solver)
        def g(x):
            # tr eps = 2e-3 * (1 - 2 x1/L) -> 在 x1 = L/2 处变号
            return np.vstack(
                [
                    2.0e-3 * x[0] + 5.0e-4 * x[1],
                    -2.0e-3 * x[1] ** 2 / L + 5.0e-4 * x[0],
                ]
            )

        def make_bcs(V):
            fdim = msh.topology.dim - 1
            boundary = dmesh.locate_entities_boundary(
                msh, fdim, lambda x: np.full(x.shape[1], True)
            )
            u_bc = fem.Function(V)
            u_bc.interpolate(g)
            return [
                fem.dirichletbc(u_bc, fem.locate_dofs_topological(V, fdim, boundary))
            ]

        solver = pf.PhaseFieldSolver(msh, model, bcs_u=make_bcs, config=cfg)
        solver.d.x.array[:] = 0.5
        solver.d.x.scatter_forward()
        iterations, flips = solver.solve_displacement(fixed_point_iterations=20)
        assert flips == 0, "chi 的不动点没收敛，linear 路径不可信"
        results[u_solver] = solver.u.x.array.copy()

    diff = np.max(np.abs(results["linear"] - results["newton"]))
    scale = np.max(np.abs(results["newton"]))
    assert scale > 0.0
    assert diff / scale < 1e-8, f"linear 与 newton 的 u 相对差 {diff / scale:.2e} 过大"


def test_linear_solver_refuses_spectral_split():
    material = make_material()
    msh = unit_mesh(2)
    model = pf.PhaseFieldModel(material, split="spectral")
    with pytest.raises(ValueError, match="newton"):
        pf.PhaseFieldSolver(msh, model, config=pf.SolverConfig(u_solver="linear"))


def test_bc_on_foreign_function_space_is_rejected():
    """回归：BC 建在外部同构空间上时 dolfinx 会静默忽略 -> 矩阵奇异但直接解仍报收敛。

    这个坑在 Phase 1.1 开发中真实发生过（u 解出 1e8 量级垃圾、d 在阈值以下也跑到 1），
    所以求解器必须在构造时就拒绝，而不是等到结果离谱才被发现。
    """
    material = make_material()
    msh = unit_mesh(2)
    model = pf.PhaseFieldModel(material)
    foreign_V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    fdim = msh.topology.dim - 1
    facets = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    bc = fem.dirichletbc(
        np.zeros(2, dtype=default_scalar_type),
        fem.locate_dofs_topological(foreign_V, fdim, facets),
        foreign_V,
    )
    with pytest.raises(ValueError, match="静默忽略"):
        pf.PhaseFieldSolver(msh, model, bcs_u=[bc])


def test_tiny_load_increment_still_converges():
    """回归：极小的载荷增量不能把 d 的 SNES 逼到舍入地板以下。

    实测过的失败模式：atol/rtol 设得过紧时，载荷步越细、网格越密，
    ‖F_0‖ 越小、相对目标越先掉到装配的舍入地板下，SNES 必然撞 max_it
    （症状是「越细越早死」，reason=-5）。这里连着施加 1e-4 相对量级的
    微小增量，全部必须收敛。
    """
    case = UniaxialStrainCase()
    s = 1.2
    case.apply_ratio(s)
    case.solver.commit_step()
    for k in range(1, 6):
        result = case.apply_ratio(s + 1.0e-4 * k)  # 每步只增 0.01% 应变
        assert result.converged, f"第 {k} 个微小增量就不收敛了：{result.failure_mode}"
        case.solver.commit_step()
    assert case.solver.max_damage() > 0.0
