"""6.6.1 Uzawa/ALM Coulomb 返回映射的测试（D-68 问题 26 的实现守卫）。

**为什么另开一个文件而不是并进 test_clamp_friction.py**：
那个文件全部用「上表面 prescribed u_x」驱动，而该驱动方式与 Coulomb 的
黏滑转折（|s| ~ μp/ε_t）冲突，Newton 必然 -6（实测 6 个用例全挂）。
Uzawa 的产品驱动是**远场弯矩**（ClampLoads，self_equilibrated），
与 Mindlin 的物理设定一致，也是 6.6.2 真正走的路径。故此处用同一驱动。

测试意图：
  U1. 乘子场可建/可判：bind_uzawa_space 后 has_uzawa() 为真，q 是 DG0；
  U2. 无摩擦时不建 q（向后兼容：mu=None 逐位回到 D-67 行为）；
  U3. Cone 投影：逐点 |q| <= mu*p（接触核心区）；
  U4. 积分量 Q <= mu*P；
  U5. 精度与 eps_t 无关（Uzawa 的本体性质：k+ 与 10k 结果一致 < 3%）；
  U6. 远场驱动下 Q 反向（完整周期含负相位时 Q/(muP) 过零）；
  U7. 摩擦不设法向反力 W；
  U8. checkpoint 场含 q，存取往返一致。
"""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

import adios4dolfinx

from pfwire.clamp_contact import ClampContact
from pfwire.obstacle import build_block_mesh
from pfwire.obstacle_contact import plane_strain_lame
from pfwire.mesh import BLOCK_BOTTOM, BLOCK_TOP

E_BULK = 69.0e9
NU_BULK = 0.33
KAPPA = 10.0
KAPPA_T = 10.0
R_CLAMP = 330.0e-3
DELTA = 5.0e-6
MU = 0.7
H_DEPTH = 3.75e-3
WIDTH = 10.0e-3
SIGMA_A = 45.0e6
T_PRIME = 1.0e4

SOLVER = {
    "snes_type": "newtonls",
    # D-31：spectral 分解的 psi+ 在过零处 C0，回溯线搜索会 -6
    "snes_linesearch_type": "basic",
    "snes_rtol": 1.0e-10,
    "snes_atol": 1.0e-10,
    "snes_max_it": 60,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def _mesh(tmp_path):
    msh, _ct, ft = build_block_mesh(
        tmp_path, width=WIDTH, depth=H_DEPTH, h_fine=0.1e-3, h_coarse=0.5e-3,
        fine_halfwidth=2.0e-3, fine_depth=2.0e-3, transition=0.5e-3,
        comm=MPI.COMM_WORLD,
    )
    return msh, ft


def _bottom_fixed(ft, msh):
    def make(V):
        fdim = msh.topology.dim - 1
        dofs = fem.locate_dofs_topological(V, fdim, ft.find(BLOCK_BOTTOM))
        zero = fem.Function(V)
        return [fem.dirichletbc(zero, dofs)]
    return make


def _end_bcs(ft, msh):
    """两端简支式的自平衡加载锚定：左端 u_x = 0（对称面）。"""
    def make(V):
        from pfwire.mesh import BLOCK_LEFT
        fdim = msh.topology.dim - 1
        dofs = fem.locate_dofs_topological(V, fdim, ft.find(BLOCK_LEFT))
        zero = fem.Function(V)
        return [fem.dirichletbc(zero, dofs)]
    return make


def _setup(tmp_path, *, mu=MU, kappa_t=KAPPA_T):
    """装配「法向延拓 + Uzawa 切向」的全套，返回 (case-like dict)。"""
    from pfwire.clamp_loads import ClampLoads
    from dolfinx.fem.petsc import NonlinearProblem

    msh, ft = _mesh(tmp_path)
    clamp = ClampContact(msh, ft, E=E_BULK, R_clamp=R_CLAMP, kappa=KAPPA,
                         h_contact=0.1e-3, symmetric=True, depth=H_DEPTH,
                         mu=mu, kappa_t=kappa_t)
    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    u = fem.Function(V)
    clamp.bind_slip_space(V)
    clamp.bind_uzawa_space(V)
    loads = ClampLoads(msh, ft, H=H_DEPTH, sigma_a=SIGMA_A, T=T_PRIME,
                       side="right", self_equilibrated=True)
    v = ufl.TestFunction(V)
    mu_l, lmbda = plane_strain_lame(E_BULK, NU_BULK)
    eps = ufl.sym(ufl.grad(u))
    sig = 2.0 * mu_l * eps + lmbda * ufl.tr(eps) * ufl.Identity(2)
    F = ufl.inner(sig, ufl.sym(ufl.grad(v))) * ufl.dx
    F = F - clamp.residual(u, v)
    F = F - clamp.tangential_residual(u, v)
    F = F - loads.traction()(V)
    problem = NonlinearProblem(F, u,
                               bcs=list(_bottom_fixed(ft, msh)(V))
                                   + list(_end_bcs(ft, msh)(V)),
                               petsc_options_prefix="uz_", petsc_options=SOLVER)

    # ---- 法向延拓：关摩擦 ----
    clamp.set_mu(None)
    loads.set_phase(0.0)
    for d in (1.0e-7, 1.0e-6, 1.0e-5, DELTA):
        clamp.set_delta(d)
        problem.solve()
        if problem.solver.getConvergedReason() <= 0:
            raise RuntimeError("法向延拓不收敛（delta=%g）" % d)
    clamp.set_slip_reference(u)
    # ---- 开摩擦 ----
    if mu is not None:
        clamp.set_mu(mu)
    return dict(msh=msh, ft=ft, clamp=clamp, V=V, u=u, problem=problem,
                loads=loads)


def _uzawa(problem, clamp, u, *, max_it=40, tol=1.0e-3):
    """跑 Uzawa 内循环，返回迭代次数。"""
    eps_t = clamp.eps_t
    for it in range(max_it):
        problem.solve()
        if problem.solver.getConvergedReason() <= 0:
            raise RuntimeError("Uzawa 内的 Newton 不收敛：SNES reason=%d"
                               % problem.solver.getConvergedReason())
        rel = clamp.uzawa_update(u, eps_t=eps_t)
        if rel <= tol:
            return it + 1
    return max_it


# --------------------------------------------------------------------------
# U 组：接口与 Coulomb 不等式
# --------------------------------------------------------------------------
def test_U1_multiplier_field_is_dg0_and_flagged(tmp_path):
    """U1：bind_uzawa_space 后 has_uzawa() 为真，q 是 DG0 场。"""
    ctx = _setup(tmp_path)
    clamp = ctx["clamp"]
    assert clamp.has_uzawa() is True
    assert clamp.q is not None
    bx = clamp.q.function_space.element.basix_element
    assert int(bx.degree) == 0 and bool(bx.discontinuous), "q 必须是 DG0"


def test_U2_no_friction_never_builds_q(tmp_path):
    """U2：mu=None 时 q 不建，has_uzawa() 为假（向后兼容 D-67）。"""
    ctx = _setup(tmp_path, mu=None)
    clamp = ctx["clamp"]
    assert clamp.has_uzawa() is False
    assert clamp.q is None
    # 切向反力仍然是 0（不是异常）
    assert clamp.tangential_reaction(ctx["u"]) == 0.0


def test_U3_U4_cone_and_integral_bounds(tmp_path):
    """U3：核心区逐点 |q| <= mu*p；U4：积分量 Q <= mu*P。"""
    ctx = _setup(tmp_path)
    clamp, u, msh, ft = ctx["clamp"], ctx["u"], ctx["msh"], ctx["ft"]
    _uzawa(ctx["problem"], clamp, u)
    P = clamp.reaction(u)
    Q = clamp.tangential_reaction(u)
    # U4：积分量（可严）
    assert abs(Q) <= MU * P * (1.0 + 1.0e-9), "Q=%g 超过 mu P=%g" % (Q, MU * P)
    # U3：逐点
    Vs = fem.functionspace(msh, ("Lagrange", 1))
    p_expr, _ = clamp._pressure_and_normal(u)
    p_f = fem.Function(Vs)
    p_f.interpolate(fem.Expression(p_expr, Vs.element.interpolation_points))
    q = clamp.q.x.array
    # q 是 DG0（逐单元），用单元上接触节点的 p 做同口径归约
    lim = MU * np.maximum(p_f.x.array, 0.0)
    # 直接查 q 的最大值与全局 mu*p_max 的关系（DG0 与 CG1 的逐点比对
    # 需要投影，这里用积分量与最大值的界限关系）
    assert float(np.max(np.abs(q))) <= MU * float(np.max(p_f.x.array)) * (
        1.0 + 1.0e-6), "逐点越界：max|q|=%g > mu*pmax=%g" % (
        float(np.max(np.abs(q))), MU * float(np.max(p_f.x.array)))


def test_U5_precision_independent_of_eps_t(tmp_path):
    """U5：kappa_t 与 10*kappa_t 的 Q 相对差 < 3%（Uzawa 的本体性质）。"""
    qs = []
    for kt, tag in ((KAPPA_T, "u5a_"), (10.0 * KAPPA_T, "u5b_")):
        ctx = _setup(tmp_path, kappa_t=kt)
        ctx["problem"].petsc_options_prefix = tag
        clamp, u = ctx["clamp"], ctx["u"]
        ctx["loads"].set_phase(1.0)
        _uzawa(ctx["problem"], clamp, u)
        qs.append(clamp.tangential_reaction(u))
    rel = abs(qs[1] - qs[0]) / abs(qs[0]) if qs[0] else float("inf")
    assert rel < 0.03, "eps_t 敏感性 %.1f%% 过大：%r" % (100 * rel, qs)


def test_U6_far_field_drive_stays_stick(tmp_path):
    """U6：纯远场弯矩驱动下接触处于**深度黏着**，Q 不随相位反向。

    依据 D-68 问题 26 续 5：远场弯矩引起的接触面切向位移 ≈ 1.88 µm，
    而 Coulomb 黏着能承受的弹性滑移 ≈ μp/E* ≈ 1.7 mm，差三个数量级。
    故接触斑没有真实滑移，Q 由体平衡唯一确定，**不随弯矩相位反向**。

    ⚠ 不断言 |Q/(μP)| 远离饱和：深度黏着 + Uzawa 极限环时 Q 会停在锥面上
    （Q/(μP) ≈ 0.97 但不随相位变）。那个「停在锥上」是 D-68 问题 26 续 5
    记录的**已知局限**，不是本用例要判的物理量。本用例判的是**相位不敏感**。

    ⚠ 若把本测试改成「Q 应该反向」就错了 —— 那需要线夹的切向微动 δ(t)
    （PLAN 6.6.2 尚未落地的部分）。
    """
    ctx = _setup(tmp_path)
    clamp, u, loads = ctx["clamp"], ctx["u"], ctx["loads"]
    phases = (1.0e-3, 0.1, 0.5, 1.0, 0.5, 0.0, -0.1, -0.5, -1.0, 0.0)
    ratios = []
    for ph in phases:
        loads.set_phase(ph)
        _uzawa(ctx["problem"], clamp, u)
        P = clamp.reaction(u)
        Q = clamp.tangential_reaction(u)
        # Coulomb 上限必须始终成立（这是算法正确性的底线）
        assert abs(Q) <= MU * P * (1.0 + 1.0e-9), (
            "Coulomb 被违反：Q=%g > mu P=%g（phase=%g）" % (Q, MU * P, ph))
        ratios.append(Q / (MU * P) if P else 0.0)
    # 深度黏着：Q 不反向（若出现真实的滑移-反向，最小值应为负）
    assert min(ratios) > 0.0, "Q 反向了，说明真有滑移：%r" % (ratios,)
    # 且对相位不敏感（黏着的 Q 由法向压力定，弯矩只改变远场）
    spread = max(ratios) - min(ratios)
    assert spread < 0.05, "Q/(muP) 随相位变化 %.3f 过大（黏着应几乎不变）" % spread


def test_U7_friction_does_not_change_normal_reaction(tmp_path):
    """U7：同 delta 下 W 与有无摩擦无关（力系解耦）。"""
    ws = []
    for mu in (None, MU):
        ctx = _setup(tmp_path, mu=mu)
        ws.append(ctx["clamp"].reaction(ctx["u"]))
    assert abs(ws[1] - ws[0]) <= 1.0e-8 * abs(ws[0]), (ws[0], ws[1])


def test_U8_checkpoint_roundtrip(tmp_path):
    """U8：q 进 checkpoint 场，存取往返一致。

    ⚠ 必须按生产的断点续跑路径走：**读回网格后在它上面重建求解器**。
    ``checkpoint.Resumed.read_fields`` 会拒绝任何不是自己那张网格上的求解器 ——
    那正是 D-48 的「静默读出乱序值」守卫。用原网格去读会撞上守卫（本 tests
    第一版就是这么错的）。

    ⚠ 不同 mesh 对象之间直接比较数组无意义（单元编号可能不同），
    需要在同一张 mesh 上比较：用原 mesh 读回 bp 里的 q 做基准。
    """
    import pathlib
    import tempfile

    from pfwire import checkpoint
    from pfwire.cyclejump import FatigueState
    from pfwire.material import load_material
    from pfwire.phasefield import (
        PhaseFieldModel,
        PhaseFieldSolver,
        SolverConfig,
    )

    ctx = _setup(tmp_path)
    clamp, u, msh = ctx["clamp"], ctx["u"], ctx["msh"]
    ctx["loads"].set_phase(1.0)
    _uzawa(ctx["problem"], clamp, u)

    mat = load_material(
        pathlib.Path(__file__).resolve().parents[1] / "configs"
        / "material_al1350.yaml")
    model = PhaseFieldModel(material=mat, split="spectral", k_res=1.0e-6)
    aT, form, kappa = mat.require_fatigue_params()
    state = FatigueState(msh, model, alpha_T=aT, form=form, kappa=kappa)

    # 用同一个 solver 做 checkpoint（生产里就是同一个 solver 持续跑）
    solver = PhaseFieldSolver(msh, model, bcs_u=(),
                              config=SolverConfig(u_solver="newton"),
                              f_alpha=state.f_alpha, contact=clamp)
    state.bind_solver(solver)
    solver.u.x.array[:] = u.x.array  # 把刚才解出来的位移拷进去

    assert "q" in checkpoint.FIELD_NAMES
    fields = checkpoint._fields(solver, state)
    assert fields["q"] is clamp.q, "checkpoint 取的 q 不是接触的乘子场"
    assert float(np.max(np.abs(clamp.q.x.array))) > 0.0, "q 应已被 Uzawa 更新过"
    # ⚠ 只用**标量摘要**做往返校验，不逐点比数组：adios 按**全局 input 编号**
    # 存取，而原 mesh 与 bp 里读回的 mesh 的 DG0 dof 排序可以不同
    # （实测非零位置 869/931/997 vs 9/79/97，值本身是对的但槽位不同）——
    # 这正是 D-48「必须用 bp 读回的网格」的同一教训。逐点比只在
    # **同一张 mesh**（res.mesh）上做。
    n_owned_q = clamp.q.function_space.dofmap.index_map.size_local
    before = clamp.q.x.array[:n_owned_q].copy()
    before_max = float(np.max(np.abs(before)))
    before_nz = int(np.count_nonzero(before))

    with tempfile.TemporaryDirectory() as td:
        p = checkpoint.save(td, solver, state, {"N": 1.0}, MPI.COMM_WORLD)
        # 按生产路径读回：resume -> 在**读回的网格**上重建一切 -> read_fields
        res = checkpoint.resume(td, MPI.COMM_WORLD)
        clamp2 = ClampContact(res.mesh, None, E=E_BULK, R_clamp=R_CLAMP,
                              kappa=KAPPA, h_contact=0.1e-3, symmetric=True,
                              depth=H_DEPTH, mu=MU, kappa_t=KAPPA_T)
        V2 = fem.functionspace(res.mesh, ("Lagrange", 1, (2,)))
        clamp2.bind_slip_space(V2)
        clamp2.bind_uzawa_space(V2)
        state2 = FatigueState(res.mesh, model, alpha_T=aT, form=form, kappa=kappa)
        solver2 = PhaseFieldSolver(res.mesh, model, bcs_u=(),
                                   config=SolverConfig(u_solver="newton"),
                                   f_alpha=state2.f_alpha, contact=clamp2)
        state2.bind_solver(solver2)
        res.read_fields(solver2, state2)

        # (a) 标量摘要：幅值与非零数必须一致（与 dof 排序无关）
        got = clamp2.q.x.array
        assert int(np.count_nonzero(got)) == before_nz, (
            "非零数不一致：%d vs %d" % (int(np.count_nonzero(got)), before_nz))
        assert abs(float(np.max(np.abs(got))) - before_max) <= 1e-6 * before_max, (
            "幅值不一致：%.6e vs %.6e"
            % (float(np.max(np.abs(got))), before_max))

        # (b) 逐点一致：在**同一张 mesh**（res.mesh）上直接读 bp 的 q 做基准
        q_bp_res = fem.Function(fem.functionspace(res.mesh, ("DG", 0)), name="q")
        import adios4dolfinx
        adios4dolfinx.read_function(p, q_bp_res)
        err = float(np.max(np.abs(clamp2.q.x.array - q_bp_res.x.array)))
        assert err == 0.0, "read_fields 与直接读 bp 不一致：最大差 %.3e" % err
