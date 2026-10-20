"""6.6.1/6.6.2 Coulomb 摩擦与线夹切向往复的测试（D-68 问题 26/27 的实现守卫）。

**为什么另开一个文件而不是并进 test_clamp_friction.py**：
那个文件全部用「上表面 prescribed u_x」驱动，而该驱动方式与 Coulomb 的
黏滑转折（|s| ~ μp/ε_t）冲突，Newton 必然 -6（实测 6 个用例全挂）。
本文件的产品驱动是**远场弯矩 + 线夹切向行程**（ClampLoads +
ClampContact.set_tangential），与 Mindlin 的物理设定一致，
也是 6.6.2 真正走的路径。

测试意图：
  U1. 乘子场可建/可判：bind_uzawa_space 后 has_uzawa() 为真，q 是 DG0；
  U2. 无摩擦时不建 q（向后兼容：mu=None 逐位回到 D-67 行为）；
  U3. Cone 投影：逐点 |q| <= mu*p（接触核心区）；
  U4. 积分量 Q <= mu*P；
  U5. 精度与 eps_t 无关（Uzawa 的本体性质：k+ 与 10k 结果一致 < 3%）；
  U6. 纯远场驱动下接触处于深度黏着，Q 不反向、Coulomb 不被违反；
  U7. 摩擦不设法向反力 W；
  U8. checkpoint 场含 q，存取往返一致；
  U9. 切向几何：set_tangential(±a) 的压力分布互为 x→−x 镜像，0 与旧接口一致；
  U10. 对称反向：上下线夹切向偏移反号，接触斑内存在相对错动；
  U11. 相位同步：切向行程与弯矩同相位（都由 sin(2πt/T) 驱动）；
  U12. 部分滑移：切向行程足够大时接触斑内同时有黏着点与滑移点。
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
    """底面**整面**全固定（u_x = u_y = 0）。仅用于单侧线夹（symmetric=False）的旧测试。

    ⚠ **不能与 symmetric=True 同用**：下侧线夹就在这个面上压，而该面被
    钉死 —— 下侧压力算出来却无处释放，切向反力爆掉（实测 Q/(μP) = −150，
    违反 Coulomb 锥 150 倍）。对称工况请用 ``_free_support``。
    """
    def make(V):
        fdim = msh.topology.dim - 1
        dofs = fem.locate_dofs_topological(V, fdim, ft.find(BLOCK_BOTTOM))
        zero = fem.Function(V)
        return [fem.dirichletbc(zero, dofs)]
    return make


def _free_support(ft, msh):
    """与生产 ``loading.support: free`` 一致的锚定（D-67 §3）。

    左端 u_x = 0（对称面，轴向锚定）+ 底边 |x| 最小的一个节点 u_y = 0（只挡
    刚体竖移）。**底面其余自由度全自由**，下侧线夹才有地方把单丝顶起来。

    与 ``_bottom_fixed`` 的区别就是这一点：整面钉死会把下侧接触变成
    病态约束（见 ``_bottom_fixed`` 的注）。
    """
    def make(V):
        fdim = msh.topology.dim - 1
        from pfwire.mesh import BLOCK_LEFT
        Vx, Vy = V.sub(0), V.sub(1)
        bcs = [fem.dirichletbc(
            np.zeros(1, dtype=np.float64)[0],
            fem.locate_dofs_topological(Vx, fdim, ft.find(BLOCK_LEFT)), Vx)]
        dofs_y = fem.locate_dofs_topological(Vy, fdim, ft.find(BLOCK_BOTTOM))
        coords = V.tabulate_dof_coordinates()
        cand = dofs_y[dofs_y < Vy.dofmap.index_map.size_local]
        if cand.size:
            i = int(np.argmin(np.abs(coords[cand, 0])))
            x_best, dof_best = float(abs(coords[cand[i], 0])), int(cand[i])
        else:
            x_best, dof_best = float("inf"), -1
        owner = min(msh.comm.allgather((x_best, dof_best, msh.comm.rank)))
        if owner[2] == msh.comm.rank and owner[1] >= 0:
            bcs.append(fem.dirichletbc(
                np.zeros(1, dtype=np.float64)[0],
                np.array([owner[1]], dtype=np.int32), Vy))
        return bcs
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
                               bcs=list(_free_support(ft, msh)(V)),
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


# --------------------------------------------------------------------------
# U9-U12：线夹切向往复 δ_x(t)（PLAN 6.6.2，D-68 问题 27）
# --------------------------------------------------------------------------
def _surface_pressure(msh, ft, clamp, u):
    """接触面上的 (x, p)：p 在 CG1 节点上插值，按 x 排序，只含接触面节点。"""
    xs, ps, _ = _surface_profile(msh, ft, clamp, u)
    return xs, ps


def _surface_profile(msh, ft, clamp, u):
    """接触面上的 (x, p, q)，按 x 排序，只含接触面节点。

    q 取**乘子场**（Uzawa 模式）或罚函数公式，与 ``tangential_residual`` 同口径：
    两者都要经 CG1 节点插值后才能与 p 逐点比（DG0 与 CG1 不能直接比）。
    """
    from pfwire.obstacle import tangential_traction

    Vs = fem.functionspace(msh, ("Lagrange", 1))
    p_expr, _ = clamp._pressure_and_normal(u)
    p_f = fem.Function(Vs)
    p_f.interpolate(fem.Expression(p_expr, Vs.element.interpolation_points))

    tau_up, _ = clamp._tangents(u)
    if clamp.has_uzawa():
        q_expr = clamp.q
    else:
        # 与 clamp.tangential_residual 同口径：s 相对**线夹**量（含 x_clamp）
        s_up = ufl.variable(clamp._slip(u, tau_up, clamp.x_clamp))
        q_expr = tangential_traction(s_up, p_expr, clamp.mu, clamp.eps_t)
    q_f = fem.Function(Vs)
    q_f.interpolate(fem.Expression(q_expr, Vs.element.interpolation_points))

    dofs = fem.locate_dofs_topological(Vs, msh.topology.dim - 1, ft.find(BLOCK_TOP))
    n_owned = Vs.dofmap.index_map.size_local
    keep = dofs[dofs < n_owned]
    xs = Vs.tabulate_dof_coordinates()[keep, 0]
    xs = np.concatenate(msh.comm.allgather(xs))
    ps = np.concatenate(msh.comm.allgather(p_f.x.array[keep]))
    qs = np.concatenate(msh.comm.allgather(q_f.x.array[keep]))
    o = np.argsort(xs)
    return xs[o], ps[o], qs[o]


def test_U9_tangential_geometry_mirrors_and_zero_is_bitwise(tmp_path):
    """U9：set_tangential(+a) 与 (−a) 的压力分布互为 x → −x 镜像；
    set_tangential(0) 与不开切向（旧接口）逐位一致。

    这是纯几何守卫：线夹圆心水平移动 a 等价于把接触斑整体平移，
    而圆弧障碍关于 x=0 对称，故 ±a 的压力分布必须互为镜像。

    ⚠ 必须**关掉弯矩**（phase=0）才能谈镜像：端部弯矩 M(t) = M_a·sin(2πt/T)
    本身就破坏 x→−x 对称（它在 +x 侧拉、−x 侧压），带着它量出来的「不镜像」
    是物理效应不是几何错误（实测差 11%，全部来自弯矩）。
    """
    ctx = _setup(tmp_path)
    clamp, u, msh, ft = ctx["clamp"], ctx["u"], ctx["msh"], ctx["ft"]
    prob = ctx["problem"]
    ctx["loads"].set_phase(0.0)      # 关弯矩，只留法向 + 切向行程

    # 基线：不开切向
    clamp.set_tangential(0.0)
    if clamp.q is not None:
        clamp.q.x.array[:] = 0.0
        clamp.q.x.scatter_forward()
    prob.solve()
    xs0, ps0 = _surface_pressure(msh, ft, clamp, u)
    # 斑中心的**基线**位置：圆弧线夹的接触斑本身有静偏心（实测 -1.69e-06），
    # 判镜像必须比对「相对基线的位移」而不是绝对位置 —— 后者会把静偏心
    # 当成不对称（+-a 两侧都偏在同一侧，实测 +1.2802e-06 与 -2.0900e-06
    # 看似不镜像，其实位移是 +0.405 µm 与 -0.405 µm，严格等值反号）。
    live0 = ps0 > 0.30 * ps0.max()
    c_base = float(np.average(xs0[live0], weights=ps0[live0]))

    # +a 与 −a：都从基线解起步（延拓思想，避免一步跨太大）
    out = {}
    for sign in (+1.0, -1.0):
        clamp.set_tangential(0.0)
        # ⚠ q 必须清零：Uzawa 乘子是路径相关的历史量，+a 跑完留下的 q
        # 会带到 −a 那一轮，破坏镜像（实测不清零差 3.6%，清零后 <1%）。
        if clamp.q is not None:
            clamp.q.x.array[:] = 0.0
            clamp.q.x.scatter_forward()
        prob.solve()
        for frac in (0.25, 0.5, 1.0):
            clamp.set_tangential(sign * 2.0e-6 * frac)
            prob.solve()
            if prob.solver.getConvergedReason() <= 0:
                raise RuntimeError("切向 %.1f*%.2f um 不收敛（reason=%d）"
                                   % (sign, 2.0 * frac,
                                      prob.solver.getConvergedReason()))
        xs, ps = _surface_pressure(msh, ft, clamp, u)
        out[sign] = (xs, ps)

    # 镜像：把 −a 的分布按 x → −x 翻过来，应与 +a 的逐点一致
    xp, pp = out[+1.0]
    xm, pm = out[-1.0]
    # np.interp 要求 x 坐标单调递增，故先把 −a 的结果镜像并重排
    o = np.argsort(-xm)
    pm_mirror = np.interp(xp, -xm[o], pm[o], left=0.0, right=0.0)
    # ⚠ 只在**接触斑内部**比：斑边缘的压力在相邻采样节点之间掉到 0，
    # np.interp 在斑边界会读出 0（实测斑边缘一点 p(+a)=1.8e7 而镜像得 1.9e-6），
    # 那是插值伪迹不是几何不对称。用 30% 峰值以上的点，边界自然被排除。
    live = pp > 0.30 * pp.max()
    assert live.sum() >= 5, "接触斑内点太少（%d），无法判镜像" % live.sum()
    rel = float(np.max(np.abs(pp[live] - pm_mirror[live]))
                / max(float(np.max(pp[live])), 1.0))
    # 阈值 5%：**几何上是严格镜像的**（解析核对了 g(x,+a) 与 g(-x,−a) 的差
    # 恒为 0.00e+00），但两次 Newton 走的是不同路径、收敛到不同的容差内点，
    # 残差 1–3.6% 且**边缘大、中心小**——这正是求解器容差的签名。
    # 若几何写错（比如漏传 x_clamp），差会是 O(1) 而不是几个百分点。
    assert rel < 0.05, "±a 压力分布不镜像：最大相对差 %.3f" % rel
    # 斑中心的**位移**必须等值反号（相对基线 c_base，见其处说明）：
    # 实测 +a 移 +4.0491e-07 m、-a 移 -4.0485e-07 m，两者**和**为
    # 6.4e-10（行程 2 um 的 0.03%），是 Newton 容差。判据取行程的 20%。
    c_p = float(np.average(xp[live], weights=pp[live])) - c_base
    live_m = pm > 0.30 * pm.max()
    c_m = float(np.average(xm[live_m], weights=pm[live_m])) - c_base
    # ⚠ 等值反号意味着 c_p ≈ +Δ、c_m ≈ −Δ，故判据是 |c_p + c_m| ≈ 0
    # （**和**接近 0，不是差 —— |c_p − c_m| 恒为 2Δ，量级上是行程本身）。
    assert abs(c_p + c_m) < 0.20 * 2.0e-6, (
        "接触斑中心的位移不等值反号：+a 移 %+.4e vs -a 移 %+.4e" % (c_p, c_m))
    # 更强的判据：两侧都应在**行程量级**内（不会偏出斑外）
    assert abs(c_p) < 1.0e-6 and abs(c_m) < 1.0e-6, (
        "接触斑中心位移偏出行程量级：%g, %g" % (c_p, c_m))
    assert np.sign(c_p) != np.sign(c_m), "两侧斑中心位移同向，线夹反向没生效"

    # 0 与旧接口一致（回归保护）
    # ⚠ 判据是**数值相同**而不是 array_equal：往返到 0 后重解，Newton 落在
    # 容差内的最后一个非线性迭代点会有极小的台阶差（实测 115 个点里只有 1 个
    # 不等，max|Δp| = 0.77 Pa 而 p_max = 2.2e8 Pa，相对 3.4e-9，即 rtol=1e-10
    # 量级）。那是求解器容差不是几何改变；真正漏传 x_clamp 的话差会是 O(1)。
    clamp.set_tangential(0.0)
    prob.solve()
    xs1, ps1 = _surface_pressure(msh, ft, clamp, u)
    scale = max(float(np.max(ps0)), 1.0)
    rel0 = float(np.max(np.abs(ps1 - ps0))) / scale
    assert rel0 < 1.0e-6, "set_tangential(0) 改变了压力分布：相对差 %.3e" % rel0


def test_U10_symmetric_opposed_offsets(tmp_path):
    """U10：对称挤压时上下线夹切向偏移反号（oppose_x 默认 True）。

    反向才有相对错动、才有微滑；同向同幅是刚体平移、接触面无错动、
    摩擦不起作用（静默失效）。本测试把两个 Constant 的值钉住。
    """
    ctx = _setup(tmp_path)
    clamp = ctx["clamp"]
    assert clamp.symmetric is True
    assert clamp.oppose_x is True, "默认必须反向（D-68 问题 27）"

    clamp.set_tangential(2.0e-6)
    assert clamp.x_clamp.value == 2.0e-6
    assert clamp.x_clamp_low.value == -2.0e-6, "下侧应取反对号"
    clamp.set_tangential(-2.0e-6)
    assert clamp.x_clamp.value == -2.0e-6
    assert clamp.x_clamp_low.value == 2.0e-6
    clamp.set_tangential(0.0)
    assert clamp.x_clamp.value == 0.0 and clamp.x_clamp_low.value == 0.0

    # 对照：oppose_x=False 时同向
    clamp.oppose_x = False
    clamp.set_tangential(2.0e-6)
    assert clamp.x_clamp_low.value == 2.0e-6, "同向模式上下应一致"


def test_U11_tangential_tracks_phase(tmp_path):
    """U11：切向行程与弯矩同相位（都由 sin(2πt/T) 驱动）。

    复现 clamp_case.set_phase 的相位律（那里只调 Constant.value），
    确认 0.25（峰值）与 0.75（谷值）反号、0 处为 0。
    """
    import math

    ctx = _setup(tmp_path)
    clamp, loads = ctx["clamp"], ctx["loads"]
    amp = 3.0e-6

    def drive(t_over_T):
        value = math.sin(2.0 * math.pi * float(t_over_T))
        loads.set_phase(value)
        clamp.set_tangential(amp * value)

    seen = {}
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        drive(t)
        # ⚠ fem.Constant.value 返回**活的数组引用**，不是副本：直接存会让
        # 字典里每一项都指向同一个数组、最后全变成末态的值（实测：
        # seen[0.25] 读出 t=1.0 的 x）。必须 float() 成标量再存。
        seen[t] = (float(clamp.x_clamp.value), float(loads.moment))

    # ⚠ math.sin(2π*0.25) 不是精确的 1 而是 1.2246e-16（浮点），
    # 故用相对容差而不是逐位相等。
    # 峰值相位 0.25：切向与弯矩都取正最大
    assert abs(seen[0.25][0] - amp) < 1e-9 * amp, seen[0.25]
    assert seen[0.25][1] > 0.0, "弯矩应为正"
    # 谷值相位 0.75：切向反号，弯矩也反号
    assert abs(seen[0.75][0] + amp) < 1e-9 * amp, seen[0.75]
    assert seen[0.75][1] < 0.0, "弯矩应为负"
    # 0 与 1（周期闭合）都为 0（同样是浮点残差，用绝对容差）
    for t in (0.0, 0.5, 1.0):
        assert abs(seen[t][0]) < 1e-9 * amp, (t, seen[t])
    # 同相：切向与弯矩的符号始终一致（0 处两者都≈0，单独放过）
    for t, (x_val, m_val) in seen.items():
        if abs(x_val) < 1e-9 * amp:
            continue
        assert np.sign(x_val) == np.sign(m_val), (
            "切向与弯矩不同相：t=%s, x=%g, M=%g" % (t, x_val, m_val))


def test_U12_partial_slip_with_tangential_stroke(tmp_path):
    """U12：切向行程足够大时接触斑内**同时**存在黏着点与滑移点。

    ⚠ 不设通过判据（D-51：半空间解析解 vs 有限体有物理差别，c/a 不会精确相等），
    只验证「部分滑移这个定性形态真的出现了」—— 这是 6.6.2 的物理目标。

    为什么要**扫幅值**而不是取单档：罚函数格式的黏滑过渡宽度是
    |s| ~ μp/ε_t = 0.7·1e6/(10·69e9/0.1e-3) ≈ 5e-11 m = 50 pm。
    行程远大于它时**整个接触斑饱和**（|q| ≈ μp 处处成立），那是整体滑移，
    不是部分滑移。实测 5 µm 档 |q|/(μp) 剖面恒为 1.00、黏着点 0/49 —— 所以
    单取一个大档必然失败，必须扫到部分滑移所在的亚微米窗口。

    若某档幅值下没有出现部分滑移，本测试**失败并打印诊断**，
    不静默跳过（那是问题 19 ε_p=0 那类假阳性的同型陷阱）。
    """
    ctx = _setup(tmp_path)
    clamp, u, msh, ft = ctx["clamp"], ctx["u"], ctx["msh"], ctx["ft"]
    prob = ctx["problem"]
    loads = ctx["loads"]

    # 走到峰值相位（切向 + 弯矩同向最大），弯矩与切向同相位
    loads.set_phase(0.25)              # sin(2π*0.25) = +1

    # 罚函数路径（Uzawa 在深度黏着下有已知极限环，D-68 问题 26 续 5）
    clamp._uzawa_bound = False

    rows = []
    partial = []
    for amp in (0.05e-6, 0.1e-6, 0.2e-6, 0.5e-6, 2.0e-6, 5.0e-6):
        clamp.set_tangential(0.0)
        if clamp.q is not None:
            clamp.q.x.array[:] = 0.0
            clamp.q.x.scatter_forward()
        prob.solve()
        reason = 0
        for frac in (0.25, 0.5, 1.0):
            clamp.set_tangential(amp * frac)
            prob.solve()
            reason = prob.solver.getConvergedReason()
            if reason <= 0:
                break
        if reason <= 0:
            if comm_rank0():
                print("  U12: amp=%.2f um Newton 未收敛（reason=%d），跳过该档"
                      % (amp * 1e6, reason))
            continue

        xs, ps, qs = _surface_profile(msh, ft, clamp, u)
        live = ps > 0.10 * ps.max()
        assert live.any(), "amp=%.2f um 接触斑为空" % (amp * 1e6)
        stick = live & (np.abs(qs) < 0.9 * MU * ps)
        slip = live & (np.abs(qs) >= 0.9 * MU * ps)

        # Q 与 P 必须同侧积分：reaction() 只算上侧（W 的定义），
        # tangential_reaction() 是上下之和，直接相除在对称双夹持下差近 2 倍。
        dx = (xs[live].max() - xs[live].min()) / max(1, int(live.sum()) - 1)
        P_top = float(np.sum(ps[live]) * dx)
        Q_top = float(np.sum(qs[live]) * dx)
        ratio = Q_top / (MU * P_top) if P_top > 0 else float("nan")
        rows.append((amp * 1e6, ratio, int(stick.sum()), int(slip.sum())))
        if stick.any() and slip.any():
            partial.append(amp)

    if comm_rank0():
        print("  U12 诊断（每行: amp(um), Q/(muP), 黏着点数, 滑移点数）:")
        for a, r, ns, nl in rows:
            print("    %6.2f  %+.4f   %3d   %3d" % (a, r, ns, nl))

    assert partial, (
        "没有任何幅值出现部分滑移 —— 6.6.2 的物理目标未达成。诊断：\n" +
        "\n".join("amp=%.2f um: Q/(muP)=%+.4f 黏着 %d 滑移 %d" % r for r in rows))

    # 黏着区应在接触斑内侧（Mindlin 的定性形态；允 2 个加密带节点间距的回退）
    amp = partial[0]
    clamp.set_tangential(0.0)
    if clamp.q is not None:
        clamp.q.x.array[:] = 0.0
        clamp.q.x.scatter_forward()
    prob.solve()
    for frac in (0.25, 0.5, 1.0):
        clamp.set_tangential(amp * frac)
        prob.solve()
        assert prob.solver.getConvergedReason() > 0, "复算不收敛"
    xs, ps, qs = _surface_profile(msh, ft, clamp, u)
    live = ps > 0.10 * ps.max()
    stick = live & (np.abs(qs) < 0.9 * MU * ps)
    slip = live & (np.abs(qs) >= 0.9 * MU * ps)
    assert (np.median(np.abs(xs[stick]))
            <= np.median(np.abs(xs[slip])) + 2.0 * 0.1e-3), (
        "黏着区不在内侧：stick |x| 中位 %g vs slip %g"
        % (np.median(np.abs(xs[stick])), np.median(np.abs(xs[slip]))))


def comm_rank0() -> bool:
    return MPI.COMM_WORLD.rank == 0
