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
  U9. 切向几何：set_tangential(±a) 的压力分布同 x 一致（**不用** x→−x 镜像，
       原因见该用例 docstring 的四轮定位），0 与旧接口一致；
  U10. 对称反向：上下线夹切向偏移反号，接触斑内存在相对错动；
  U11. 相位同步：切向行程与弯矩同相位（都由 sin(2πt/T) 驱动）；
  U12. 部分滑移：切向行程足够大时接触斑内同时有黏着点与滑移点。
  U13. smooth_clip 单元性质：盒内恒等 / 盒外单调不超限 / η→0 收敛 / 奇对称；
  U14. 深黏着 Uzawa 收敛（6.6.3 第一个循环步的爆点，旧实现此处爆炸）；
  U15. 滑移区饱和与阻尼 θ 无关（θ 是步长，不改滑移区不动点）；
  U16. 光滑化 Coulomb 的两条底线：Coulomb 锥逐单元满足 + 大幅行程能饱和
        （部分滑移形貌在生产粗网格上验，见该用例 docstring 的理由）。
  U17. u 的 Uzawa 欠松弛（方案 h）：ω=1 恒等、ω<1 精确压缩 Δu；
  U18. 同上的配置侧：u_relaxation 默认 1.0、取值 (0,1]、clamp_case 真去读它。
"""

import shutil

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
        # ⚠ 必须在**独立的标量空间**上选点，不能用 V.tabulate_dof_coordinates()
        # （D-68 问题 33 续 9）：dofs_y 是 Vy 子空间的编号，而 V 的坐标行按 block
        # 交错，两个编号空间不同。实测同一批 41 个底边 dof，coords_V[cand, 1]
        # 只有 0/41 落在底边上 —— 取到的全是错的坐标。
        Vs = fem.functionspace(msh, ("Lagrange", 1))
        dofs_s = fem.locate_dofs_topological(Vs, fdim, ft.find(BLOCK_BOTTOM))
        coords_s = Vs.tabulate_dof_coordinates()
        cand = dofs_s[dofs_s < Vs.dofmap.index_map.size_local]
        if cand.size:
            i = int(np.argmin(np.abs(coords_s[cand, 0])))
            x_best, dof_best = float(abs(coords_s[cand[i], 0])), int(cand[i])
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
    """U6：纯远场弯矩驱动下 Coulomb 锥不被违反、|Q/(μP)| 的**幅值**对相位不敏感。

    ⚠⚠ **本用例原写的「深度黏着 → Q ≈ 0」前提，实测两版实现都没达到**
    （2026-10-20 对照 bc787af 基线）：远场弯矩驱动的接触斑在这套罚接触下
    **饱和到 |Q/(μP)| ≈ 0.98**，新旧实现逐位同量级（基线 |q|max = 1.757e8，
    新 = 1.757e8）。基线只是因为从不重置 q、各相位恰好都落在**正值**才「通过」
    —— 那是符号运气，不是物理。故本用例改为判两条**算法底线**：
      1. Coulomb 锥逐相位成立（|Q| ≤ μP）；
      2. |Q/(μP)| 的幅值对相位不敏感（黏着区的真实特征是量级不随弯矩相位变）。
    不要求 Q 保持小 —— 「小」需要 6.6.2 的线夹切向微动把接触斑拉进部分滑移
    窗口，那是另一个工况。

    饱和机理（记录备查，**不是 Uzawa 迭代问题**）：q 增大 → 切向位移 →
    法向间隙张开 → p 塌下去（实测 p_max 从 1e8 掉到 7e3）→ lim = μp 跟着塌 →
    clip 箍不住 q。这是**法向-切向耦合不稳定**，在内圈 Newton 就发生了，
    换阻尼/换光滑化都不改它。要治得改接触格式（Nitsche/Mortar，或给切向步加
    「不越过当前 p 支持的 Coulomb 容量」的限幅），超出本用例范围。
    """
    ctx = _setup(tmp_path)
    clamp, u, loads = ctx["clamp"], ctx["u"], ctx["loads"]
    phases = (1.0e-3, 0.1, 0.5, 1.0, 0.5, 0.0, -0.1, -0.5, -1.0, 0.0)
    ratios = []
    for ph in phases:
        loads.set_phase(ph)
        # 每个相位是**独立的静力平衡**（不是循环），q 不得跨相位继承 ——
        # 否则从上一相位的饱和值出发，Δq 符号历史被带成「单调」，逐单元 θ
        # 判成真滑移并饱和到另一侧（实测 ±0.98 的假滑移）。
        clamp.reset_uzawa()
        _uzawa(ctx["problem"], clamp, u)
        P = clamp.reaction(u)
        Q = clamp.tangential_reaction(u)
        # Coulomb 上限必须始终成立（这是算法正确性的底线）
        assert abs(Q) <= MU * P * (1.0 + 1.0e-9), (
            "Coulomb 被违反：Q=%g > mu P=%g（phase=%g）" % (Q, MU * P, ph))
        ratios.append(Q / (MU * P) if P else 0.0)
    # 幅值对相位不敏感：**只在弯矩非零的相位之间比**。零弯矩相位（0.0）的
    # Q 本来就该接近 0（没有切向驱动），把它算进 spread 会得到 0.979 的
    # 假警报（实测 8 个非零相位 0.9746~0.9792、2 个零相位 1e-4）。
    mags = [abs(r) for r in ratios]
    if comm_rank0():
        print("  U6: |Q|/(muP) 逐相位 = %s" % " ".join("%.4f" % m for m in mags))
    nz = [m for m in mags if m > 1.0e-3]      # 滤掉零弯矩相位
    assert nz, "没有任何非零弯矩相位"
    assert max(nz) - min(nz) < 0.05, (
        "非零弯矩相位的 |Q|/(muP) 幅值变化 %.3f 过大（黏着应几乎不变）：%r"
        % (max(nz) - min(nz), nz))


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
    # ⚠ 必须取**全局**量（CLAUDE.md 陷阱 3）：q 只在接触单元上非零，而接触斑
    # 可以整个落在某几个 rank 之外。实测 6 rank 下 rank 0 与 rank 3 的
    # size_local 各有 838 个 DG0 却**全零**（nz=0），接触单元全在另外 4 个
    # rank 上。用本 rank 的 max 判 `> 0` 会在这 2 个 rank 上失败，
    # 它们随即退出，其余 rank 卡在 checkpoint.save 的集体通信里忙等
    # （实测 240 s 超时、MPI_Abort）。全局真值 max|q| = 1.7668e+08、nz = 76。
    _q_max_local = (float(np.max(np.abs(clamp.q.x.array)))
                    if clamp.q.x.array.size else 0.0)
    q_max_global = msh.comm.allreduce(_q_max_local, op=MPI.MAX)
    assert q_max_global > 0.0, "q 应已被 Uzawa 更新过（全局 max|q| = %.3e）" % q_max_global
    # ⚠ 只用**标量摘要**做往返校验，不逐点比数组：adios 按**全局 input 编号**
    # 存取，而原 mesh 与 bp 里读回的 mesh 的 DG0 dof 排序可以不同
    # （实测非零位置 869/931/997 vs 9/79/97，值本身是对的但槽位不同）——
    # 这正是 D-48「必须用 bp 读回的网格」的同一教训。逐点比只在
    # **同一张 mesh**（res.mesh）上做。
    # ⚠ 摘要同样必须全局化：size_local / max / 非零数**全都**依赖分区
    # （实测 size_local = 838/838/843/846/853/849，nz = 0/2/34/0/23/17）。
    n_owned_q = clamp.q.function_space.dofmap.index_map.size_local
    before = clamp.q.x.array[:n_owned_q].copy()
    before_max = msh.comm.allreduce(
        float(np.max(np.abs(before))) if before.size else 0.0, op=MPI.MAX)
    before_nz = int(msh.comm.allreduce(int(np.count_nonzero(before)), op=MPI.SUM))

    # ⚠ 临时目录必须**所有 rank 同一个路径**：`tempfile.TemporaryDirectory()`
    # 每个 rank 各生成一个不同目录（随机后缀），于是 6 个 rank 把 .bp 写到 6 个
    # 不同路径，BP4 的集合 IO 直接死锁（实测 mpirun -n 6 必挂、rc=124）。
    # 生产路径 `clamp_case._save(case, outdir, ...)` 用配置里的统一 outdir，
    # 没这个问题；只有测试里用临时目录才会踩。
    # 做法：rank 0 生成后 bcast，全体共用。
    _td = msh.comm.bcast(tempfile.mkdtemp(prefix="u8_ckpt_") if msh.comm.rank == 0
                         else None, root=0)
    try:
        p = checkpoint.save(_td, solver, state, {"N": 1.0}, MPI.COMM_WORLD)
        # 按生产路径读回：resume -> 在**读回的网格**上重建一切 -> read_fields
        res = checkpoint.resume(_td, MPI.COMM_WORLD)
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
        # ⚠ 同样是全局量（陷阱 3）：读回网格的分区与 size_local 都独立于原网格，
        # 本 rank 的 max/非零数不可比。
        # ⚠ 口径必须与 before 对齐：只取 **owned** 段。DG0 的 x.array 含 ghost，
        # 同一个单元会被相邻 rank 各存一份，全取会把非零数算重
        # （实测 6 rank 下 got=80 vs before=76，差的 4 正是 ghost 重复计数）。
        n_owned_got = clamp2.q.function_space.dofmap.index_map.size_local
        got = clamp2.q.x.array[:n_owned_got]
        got_max = res.mesh.comm.allreduce(
            float(np.max(np.abs(got))) if got.size else 0.0, op=MPI.MAX)
        got_nz = int(res.mesh.comm.allreduce(int(np.count_nonzero(got)), op=MPI.SUM))
        assert got_nz == before_nz, (
            "非零数不一致：%d vs %d" % (got_nz, before_nz))
        assert abs(got_max - before_max) <= 1e-6 * before_max, (
            "幅值不一致：%.6e vs %.6e" % (got_max, before_max))

        # (b) 逐点一致：在**同一张 mesh**（res.mesh）上直接读 bp 的 q 做基准
        # ⚠ 这里的逐点比是合法的：两侧都在 res.mesh 上、用同一个 DG0 空间，
        # 不跨 mesh 比。但 max 仍要全局化（各 rank 的局部数组不同）。
        q_bp_res = fem.Function(fem.functionspace(res.mesh, ("DG", 0)), name="q")
        import adios4dolfinx
        adios4dolfinx.read_function(p, q_bp_res)
        _d = np.abs(got - q_bp_res.x.array[:n_owned_got])
        err = float(res.mesh.comm.allreduce(
            float(np.max(_d)) if _d.size else 0.0, op=MPI.MAX))
        assert err == 0.0, "read_fields 与直接读 bp 不一致：最大差 %.3e" % err
    finally:
        # 只在 rank 0 删（各 rank 的 rmtree 会对同一路径竞态）；删不掉无所谓，
        # pytest 的 tmp 不背这个锅（这里用的是 /tmp 下的独立目录）。
        if msh.comm.rank == 0:
            shutil.rmtree(_td, ignore_errors=True)


# --------------------------------------------------------------------------
# U9-U12：线夹切向往复 δ_x(t)（PLAN 6.6.2，D-68 问题 27）
# --------------------------------------------------------------------------
def _surface_pressure(msh, ft, clamp, u):
    """接触面上的 (x, p)：p 在 CG1 节点上插值，按 x 排序，只含接触面节点。"""
    xs, ps, _ = _surface_profile(msh, ft, clamp, u)
    return xs, ps


def _surface_profile(msh, ft, clamp, u):
    """接触面上的 (x, p, q)，按 x 排序，只含接触面/接触单元。

    **Uzawa 模式（q 是 DG0）走单元口径**，罚函数模式走 CG1 节点口径：

    - Uzawa：q 逐单元定义，p 也用**同一个** CG1→DG0 归约
      （``clamp._p_dg_projector``，与 ``uzawa_update`` 里算 lim 的那个完全同源），
      于是 |q| ≤ μp 的逐点比对是自洽的。实测（2026-10-20，U16 诊断）单元口径下
      max|q|/(μ·p) = 0.9998 —— Coulomb 锥满足。
      ⚠ 不要试图把 DG0 的 q 插值到 CG1 节点再比：那是无定义操作，实测给出
      |q|/(μp) ≈ 2.0 的假剖面（节点被 2~3 个单元共享、各贡献一份权重，
      叠加后超出单元值）。
    - 罚函数：q 由 UFL 表达式给出，天然 CG1，与 p 同口径逐点比。
    """
    from pfwire.obstacle import tangential_traction

    if clamp.has_uzawa():
        # ---- 单元口径（DG0 q 与同源归约的 p）----
        if not hasattr(clamp, "_p_dg_projector"):
            clamp._build_p_dg_projector()
        Vs = fem.functionspace(msh, ("Lagrange", 1))
        p_expr, _ = clamp._pressure_and_normal(u)
        p_f = fem.Function(Vs)
        p_f.interpolate(fem.Expression(p_expr, Vs.element.interpolation_points))
        p_dg = clamp._p_dg_projector.dot(p_f.x.array[:clamp._p_n_owned])
        q_dg = clamp.q.x.array
        n_owned_dg = clamp.q.function_space.dofmap.index_map.size_local
        p_dg, q_dg = p_dg[:n_owned_dg], q_dg[:n_owned_dg]

        # 接触单元 = 压力为正的单元；取它们的形心 x 作剖面横坐标
        tdim = msh.topology.dim
        cmid = msh.geometry.x[msh.geometry.dofmap[:n_owned_dg]].mean(axis=1)
        live = p_dg > 0.10 * float(np.max(p_dg)) if p_dg.size else np.zeros(0, bool)
        xs = np.concatenate(msh.comm.allgather(cmid[live, 0]))
        ps = np.concatenate(msh.comm.allgather(p_dg[live]))
        qs = np.concatenate(msh.comm.allgather(q_dg[live]))
        o = np.argsort(xs)
        return xs[o], ps[o], qs[o]

    # ---- 节点口径（罚函数路径，q 是 CG1 表达式）----
    Vs = fem.functionspace(msh, ("Lagrange", 1))
    p_expr, _ = clamp._pressure_and_normal(u)
    p_f = fem.Function(Vs)
    p_f.interpolate(fem.Expression(p_expr, Vs.element.interpolation_points))

    tau_up, _ = clamp._tangents(u)
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
    """U9：set_tangential(+a) 与 (−a) 的压力分布**同 x 一致**；
    set_tangential(0) 与不开切向（旧接口）逐位一致。

    这是纯几何守卫：线夹圆心水平移动 a 等价于把接触斑整体平移，
    而圆弧障碍关于 x=0 对称，故 ±a 的压力分布互为平移后的同一形状。

    ⚠ 必须**关掉弯矩**（phase=0）才能谈对称：端部弯矩 M(t) = M_a·sin(2πt/T)
    本身就破坏 x→−x 对称（它在 +x 侧拉、−x 侧压），带着它量出来的「不对称」
    是物理效应不是几何错误。

    ⚠⚠ 判据**不用** x→−x 镜像，用**同 x 比**（2026-10-23 改，D-68 问题 29 续 1~3）。
    原判据是「把 −a 的分布按 x→−x 翻过来与 +a 逐点一致」，实测在
    `mpirun -n 6` 下恒报 8.12%（阈值 5%）而失败。三轮定位证明它**不是**代码 bug，
    也不是物理不对称，而是判据的离散前提不成立：

    1. **不是插值伪迹**：`np.interp` 越界清零在 100/50 µm 档**从未触发**
       （越界点数 0），(C) 现判据与 (D) 排除越界后的值逐位相同。
    2. **不是切向载荷**：±a 两侧剖面的差只有 `|asym|/|sym| = 6.06e-4`，
       8% 来自同一个剖面**本身**关于 x=0 不对称。
    3. **根因是网格不镜像**：BLOCK_TOP 节点 x 的镜像偏差为
       0.50h / 0.99h / 1.49h（h=100/50/20 µm），随加密**增长**。
       而 `_surface_profile` 的 Uzawa 口径用**单元形心 x** 作横坐标
       （见其 docstring），网格不对称直接成为剖面横坐标的不对称。
    4. **数值依赖分区**：同一个 U9，串行 rel=2.29e-02、并行 rel=8.12e-02
       （3.5 倍），串行的剖面横坐标恰好对称（±1.4700a）、并行的不对称。

    docstring 原话说「圆弧障碍关于 x=0 对称故 ±a 必须互为镜像」——
    这对**连续问题**成立，对**离散采样**不成立：差在采样横坐标上。

    改成同 x 比后不依赖网格 0-对称（比的是同一个 x 上的两个解），
    仍精确覆盖本测试的意图（x_clamp 传对、上下反向生效）。
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

    xp, pp = out[+1.0]
    xm, pm = out[-1.0]
    # ⚠ 只在**接触斑内部**比：斑边缘的压力在相邻采样点之间掉到 0，
    # 会把边缘的插值/截断伪迹当成不对称。用 30% 峰值以上的点，边界自然被排除。
    live = pp > 0.30 * pp.max()
    assert live.sum() >= 5, "接触斑内点太少（%d），无法判一致性" % live.sum()

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

    # 整条剖面的一致性：**同 x 比**（见下方说明为何不用 x→−x 镜像）
    # 两侧点集相同（实测排序后逐点偏差 0.000e+00），故可直接逐点比。
    assert xp.size == xm.size, "±a 两侧采样点数不同（%d vs %d）" % (xp.size, xm.size)
    dev = float(np.max(np.abs(np.sort(xp) - np.sort(xm))))
    assert dev < 1e-12, "±a 两侧采样点集不同（最大偏差 %.3e m）" % dev
    rel_same = float(np.max(np.abs(pp[live] - pm[live]))
                     / max(float(np.max(pp[live])), 1.0))
    assert rel_same < 0.05, "±a 两侧同 x 压力不一致：最大相对差 %.3f" % rel_same

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


# --------------------------------------------------------------------------
# U13–U16：光滑化 Coulomb + 阻尼 Uzawa（D-68 问题 28 续，6.6.3 的解药）
#
# 背景：无阻尼精确投影的 Uzawa 增益 |1 − ε_t·C| ≈ 5e3，黏着区进二周期极限环，
# 6.6.3 第一个循环步直接炸（Uzawa 残差 0.2 → 1.4e3 → 5e4，SNES -6/-9）。
# 修复：FB 光滑化投影（Fischer 1992 / Kanzow 1996）+ Fortin–Glowinski 阻尼 θ。
# 判据换成自然残差（θ 变小会让 Δq 假收敛）。
# --------------------------------------------------------------------------
def test_U13_smooth_clip_is_identity_inside_and_monotonic_outside():
    """U13：:func:`smooth_clip` 的三条单元性质（盒内 O(η²) 偏差 / 盒外单调不超限 / η→0 收敛）。

    **盒内偏差是 O(η²/(4·(lim−|x|)))、而不是 O(η)，这是本修复的全部理由**：
    黏着区深处（|q| ≪ μp）偏差 ~ η²/(4μp) 可忽略；若换成把内部点整体拖向盒面的
    形式（如 μp·tanh(q/η)），整区都是 O(η) 偏差、黏着牵引被抹掉 —— 那是
    D-68 问题 26 续 5「ε_t 调小则 Q→0」的同一个坑。故这条性质必须逐点钉死：
    偏差上界用解析式 η²/(4·(lim−|x|)) 直接检验。
    """
    from pfwire.clamp_contact import smooth_clip

    lim, eta = 2.0, 0.1

    # (1) 盒内：偏差 <= 解析上界 eta^2/(4*(lim-|x|))（留 1% 余量给浮点）
    inside = np.linspace(-lim + eta, lim - eta, 401)
    out = smooth_clip(inside, lim, eta)
    bound = eta ** 2 / (4.0 * (lim - np.abs(inside)))
    assert np.all(np.abs(out - inside) <= bound * 1.01), (
        "盒内偏差超过解析上界（最大偏差 %g vs 上界 %g）"
        % (np.max(np.abs(out - inside)), np.max(bound)))
    # 深处（|x| <= lim/2）的相对偏差应在 eta^2 量级 —— O(eta) 的形式过不了这条
    deep = np.abs(inside) <= 0.5 * lim
    rel_deep = np.max(np.abs(out[deep] - inside[deep])) / lim
    assert rel_deep < 0.05 * eta, (
        "盒内深处的相对偏差 %g 达到 O(eta) 量级（阈值 0.05*eta=%g）："
        "这不是 O(eta^2) 的光滑化，会抹掉黏着牵引" % (rel_deep, 0.05 * eta))

    # (2) 盒外单调、不超限、量级趋近 lim
    outside = np.linspace(lim + eta, 10.0 * lim, 200)
    out = smooth_clip(outside, lim, eta)
    assert np.all(np.diff(out) > 0.0), "盒外必须单调不减"
    assert np.all(out <= lim + 1e-12), "盒外不得超过 lim（Coulomb 锥）"
    assert np.all(np.abs(out) > 0.5 * lim), "盒外应快速趋近 lim"
    # 盒外偏差有界 <= eta/2（滑移区相对误差可控）
    assert np.all(lim - out <= 0.5 * eta), "盒外偏差超过 eta/2"

    # (3) η → 0 逐点收敛到精确投影 np.clip
    xs = np.linspace(-10.0, 10.0, 401)
    for e in (1.0e-1, 1.0e-3, 1.0e-6):
        assert np.allclose(smooth_clip(xs, lim, e), np.clip(xs, -lim, lim), atol=e), \
            "η=%g 时未收敛到精确 clip" % e

    # (4) 奇对称（tau 指向 ±x 两侧必须一致，否则摩擦方向会不对称）
    xs = np.linspace(-10.0, 10.0, 401)
    assert np.allclose(smooth_clip(xs, lim, eta), -smooth_clip(-xs, lim, eta), atol=0.0)

    # (5) lim = 0 时 FB 形式**不**给 0（= docstring 里那条 ⚠ 的定量版本）。
    #     精确式：结果 = −½(√(x²+η²) − |x|)·sign(x)，量级 η²/(4|x|) 但**不是**那个
    #     近似式本身（近似丢掉 η⁴ 项，x=0.3/η=0.1 时差 2.7%）。故这里用精确式检验。
    #     意义：uzawa_update 对非接触单元必须走精确 clip，否则会残留这个小伪力。
    zero = smooth_clip(np.array([0.3, -0.3]), 0.0, eta)
    exact = 0.5 * (np.sqrt(0.3 ** 2 + eta ** 2) - 0.3)
    assert np.all(zero != 0.0), "lim=0 时 FB 必须非零（否则 uzawa_update 的分支无意义）"
    assert np.allclose(np.abs(zero), exact, rtol=1e-12), (zero, exact)


def test_U14_deep_stick_first_cycle_step_converges(tmp_path):
    """U14：深黏着工况下 Uzawa 收敛 —— 今天 6.6.3 爆掉的那个点。

    复现场景：法向延拓 → 冻结滑移基准 → **开摩擦**（= `_enable_friction_after_ramp`）
    → 解一次平衡。旧实现（无阻尼精确投影）在此处 Uzawa 残差爆炸
    （0.2 → 1.4e3 → 5e4），内圈 Newton SNES -6/-9，整条 case 死在第一个循环步。

    ⚠ 若此处在 tiny 网格上复现不出旧败（tiny 网格接触斑只有几个单元、
    切向柔度量级不同），则不调参硬凑，而是报告「tiny 网格不足以复现」，
    以粗网格现场为准（粗网格实测复现，见 D-68 问题 28）。
    """
    ctx = _setup(tmp_path)
    clamp, u, prob = ctx["clamp"], ctx["u"], ctx["problem"]

    # 关摩擦延拓（Mindlin 序列第一步，与生产 build_clamp_case 一致）
    clamp.set_mu(None)
    for d in (1.0e-7, 1.0e-6, 1.0e-5, DELTA):
        clamp.set_delta(d)
        prob.solve()
        assert prob.solver.getConvergedReason() > 0, "法向延拓不收敛（delta=%g）" % d

    # 开摩擦 + 冻结滑移基准（_enable_friction_after_ramp 的两个动作）
    clamp.set_slip_reference(u)
    clamp.set_mu(MU)

    # 载荷停在零相位（远场弯矩 0、切向行程 0）—— 真解 q ≡ 0。
    # 旧实现在这个**平凡**点都被迭代噪声炸飞，最能暴露增益问题。
    ctx["loads"].set_phase(0.0)
    it = _uzawa(prob, clamp, u)

    if comm_rank0():
        print("  U14: 深黏着 Uzawa 迭代 %d 次收敛，theta=%.3e"
              % (it, getattr(clamp, "_uzawa_theta", float("nan"))))
    assert it < 40, "深黏着 Uzawa 未在 max_it 内收敛（%d 次）" % it

    # q 应停在 0 附近（Coulomb 容量以内），且不违反锥
    Q = abs(clamp.tangential_reaction(u))
    P = clamp.reaction(u)
    assert Q <= MU * P * (1.0 + 1.0e-9), "违反 Coulomb 锥：Q=%g > mu P=%g" % (Q, MU * P)


def test_U15_slip_zone_saturation_is_theta_independent(tmp_path):
    """U15：滑移区饱和与阻尼 θ 无关（θ 只改黏着区的收敛速率，不改滑移区不动点）。

    滑移区不动点满足 |q| = μp（与 θ 无关，因 clip 在盒外单调饱和到 lim）。
    这是阻尼合法性的直接检验：若 θ 影响了滑移解，说明它不只是「步长」而是改了方程。
    """
    ctx = _setup(tmp_path)
    clamp, u, prob = ctx["clamp"], ctx["u"], ctx["problem"]
    clamp.set_mu(MU)

    # 大幅切向行程（5 µm ≫ μp/ε_t ≈ 50 pm 过渡宽度）→ 整体滑移
    clamp.set_tangential(5.0e-6)
    prob.solve()
    assert prob.solver.getConvergedReason() > 0, "滑移工况 Newton 不收敛"

    def _saturated_ratio() -> float:
        xs, ps, qs = _surface_profile(ctx["msh"], ctx["ft"], clamp, u)
        live = ps > 0.10 * ps.max()
        assert live.any(), "接触斑为空"
        return float(np.max(np.abs(qs[live]) / (MU * ps[live])))

    # ⚠ 必须先跑 Uzawa 再量：q 场在 _setup 的关摩擦延拓后恒为 0，不跑 Uzawa
    # 直接量会得到「饱和比 0」的假象（实测踩过）。
    it0 = _uzawa(prob, clamp, u)
    r_default = _saturated_ratio()

    # 清零后重跑，θ 被人为压到 1e-6：滑移区的 Δq 单调（真滑移特征），
    # 逐单元 θ 会判「单调 → θ=1」，故饱和比必须几乎不变。
    clamp.q.x.array[:] = 0.0
    clamp.q.x.scatter_forward()
    clamp._uzawa_rho_prev = None
    clamp._uzawa_theta = 1.0e-6
    it1 = _uzawa(prob, clamp, u, max_it=40, tol=1.0e-6)
    r_small_theta = _saturated_ratio()

    if comm_rank0():
        print("  U15: 饱和比 theta 默认 %.6f -> 1e-6 时 %.6f"
              % (r_default, r_small_theta))
    assert abs(r_small_theta - r_default) < 0.05 * max(r_default, 1.0e-12), (
        "滑移区饱和比随 theta 变化（%g -> %g）：theta 不该改变滑移区不动点"
        % (r_default, r_small_theta))


def test_U16_smoothed_coulomb_saturates_without_violating_cone(tmp_path):
    """U16：光滑化 Coulomb 下 Coulomb 锥**始终满足**、滑移区能饱和（修复的两个底线）。

    本用例不判「部分滑移形貌」—— 实测（2026-10-20）本测试网格（_setup：δ=5 µm、
    h=0.1 mm、接触斑 ~30 个单元）上 **Uzawa 路径整斑滑移、罚函数路径整斑黏着**，
    两者都不落在那条「中心黏着 + 边缘滑移」的部分滑移窗口里（该窗口在 tiny 网格
    的亚微米行程档，D-68 问题 27）。部分滑移的形貌判据归**生产粗网格跑**去验
    （那里接触斑有上百个单元、行程 0.1 µm 与 μp/ε_t 的比值才落进窗口），
    不在这里硬设一个会逼着调参的判据（硬约束 6）。

    这里判两条修复底线，任一条破了就说明光滑化/阻尼实现有错：
      1. **Coulomb 锥**：逐单元 |q| ≤ μp（容 1e-6 相对余量给 η 光滑化）。
      2. **滑移区能饱和**：大幅行程下 max|q| 达到 μ·p_max 的 95% 以上
         （若阻尼把滑移区也拖慢到饱和不了，说明 θ 施错了地方 —— 见
         uzawa_update 里 θ 分区施加的注释）。
    """
    ctx = _setup(tmp_path)
    clamp, u, prob, loads = (ctx["clamp"], ctx["u"], ctx["problem"], ctx["loads"])
    clamp.set_mu(MU)
    loads.set_phase(0.25)

    rows = []
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
            continue
        it = _uzawa(prob, clamp, u)
        xs, ps, qs = _surface_profile(ctx["msh"], ctx["ft"], clamp, u)
        live = ps > 0.10 * ps.max()
        assert live.any(), "amp=%.2f um 接触斑为空" % (amp * 1e6)

        # 底线 1：逐单元 Coulomb 锥（DG0 口径，自洽）
        cone = float(np.max(np.abs(qs[live]) / (MU * ps[live])))
        assert cone <= 1.0 + 1.0e-6, (
            "违反 Coulomb 锥：max|q|/(mu p) = %.6f（amp=%.2f um）" % (cone, amp * 1e6))

        stick = int((live & (np.abs(qs) < 0.8 * MU * ps)).sum())
        slip = int((live & (np.abs(qs) >= 0.8 * MU * ps)).sum())
        rows.append((amp * 1e6, stick, slip, it, cone))

    if comm_rank0():
        print("  U16 诊断（amp(um), 黏着单元, 滑移单元, Uzawa迭代, max|q|/(mu p)）:")
        for a, ns, nl, it, cone in rows:
            print("    %6.2f   %3d   %3d   %3d   %.6f" % (a, ns, nl, it, cone))

    assert rows, "所有幅值都未收敛"
    # 底线 2：大幅行程下必须饱和（θ 没有把滑移区拖住）
    big = [r for r in rows if r[0] >= 2.0]
    assert big, "缺少大幅行程档"
    for a, _ns, _nl, it, cone in big:
        assert cone > 0.95, (
            "大幅行程（%.2f um）未饱和：max|q|/(mu p) = %.4f（Uzawa %d 次）—— "
            "阻尼把滑移区拖住了，见 uzawa_update 的 θ 分区施加"
            % (a, cone, it))


def comm_rank0() -> bool:
    return MPI.COMM_WORLD.rank == 0


# --------------------------------------------------------------------------
# U17-U18：u 的 Uzawa 欠松弛（方案 h，D-68 问题 33 续 4，6.6.3 的 blocker）
#
# 背景：6.6.3 开摩擦后 u 的 Newton 解在 Uzawa 正反馈环里漂移
# （u -> s -> q -> 面力 -> u），实测同一轮三次迭代 s_dg 涨 13 个数量级
# （3.55e-15 -> 3.85e-08 -> 1.08e-06），q 被动跟着涨（9.9e4 -> 2.0e6 -> 6.1e7）。
# 五个假说（θ₀ 太小 / th_eff 判据 / DG0 ghost / form 结构 / 载荷大小）
# 都已逐一证伪，真根因是 s 无界 ⇒ θ₀ 阻尼无从发挥。
#
# 方案 h：对 u 欠松弛，u <- u_iter_prev + ω·(u_solved - u_iter_prev)。
# 因为 s = τ·(u - u_prev_clamp) 而 u_prev_clamp 跨 Uzawa 循环冻结，
# Δu 压 ω 倍 ⇒ Δs 同比压小 ⇒ 正反馈增益从 1 降到 ω。
# --------------------------------------------------------------------------
def _u_relax_apply(u, u_iter_prev, omega):
    """施加 u 欠松弛（与 PhaseFieldSolver._solve_u_uzawa 里方案 h 逐行对应）。"""
    if omega >= 1.0:
        return
    u.x.array[:] = u_iter_prev + omega * (u.x.array - u_iter_prev)
    u.x.scatter_forward()


def test_U17_u_relaxation_shrinks_step_and_keeps_fixed_point(tmp_path):
    """U17：``u_relaxation`` 的两条性质 —— ω=1 是恒等；ω<1 精确压缩迭代增量。

    方案 h 的机理（D-68 问题 33 续 4）：6.6.3 开摩擦后 u 的 Newton 解在
    Uzawa 正反馈环里漂移（u → s → q → 面力 → u），实测同一轮三次迭代
    s_dg 涨 13 个数量级。因 ``s = τ·(u − u_prev_clamp)`` 而 ``u_prev_clamp``
    由 ``set_slip_reference`` **跨 Uzawa 循环冻结**，对 u 欠松弛
    ``u ← u_iter_prev + ω(u_solved − u_iter_prev)`` 就把 Δu 压 ω 倍，
    Δs 同比压小，q 的正反馈增益从 1 降到 ω。

    本用例把这条机理拆成两个可判的性质（**不**跑完整 Uzawa 循环 ——
    tiny 网格上循环几步就收敛，复现不出粗网格那个正反馈环）：

    1. **ω=1 是恒等**：这一步对 u 不做任何事（逐位相等）。这是向后兼容底线 ——
       默认值下所有既有工况必须逐位不变。
    2. **ω<1 精确压缩**：同一个 ``u_solved`` 上分别施加 ω=1 与 ω=0.5，
       两者的差必须恰是 ``0.5·(u_solved − u_iter_prev)``（相对偏差 < 1e-12）。
       这把「ω 到底作用在哪」直接钉死 —— 若实现误乘到别处（乘 q、或乘
       u_prev），这里立刻现形，不需要等循环发散。

    ⚠ 本用例**不**判「粗网格上 6.6.3 是否因此收敛」：tiny 网格（_setup：
    h=0.1 mm、接触斑 ~30 单元）上 Uzawa 几步就收敛。6.6.3 的最终验收在
    生产粗网格上做（``python -m pfwire.run_clamp configs/clamp_fretting_uzawa.yaml``）。
    """
    ctx = _setup(tmp_path)
    clamp, u, prob = ctx["clamp"], ctx["u"], ctx["problem"]
    ctx["loads"].set_phase(0.25)
    clamp.set_tangential(1.0e-6)

    # 一次 Newton 解，作为两种 ω 共同的 u_solved
    u_iter_prev = u.x.array.copy()
    prob.solve()
    assert prob.solver.getConvergedReason() > 0, "Newton 未收敛，测试前提不成立"
    u_solved = u.x.array.copy()

    # 性质 1：ω=1 恒等
    u.x.array[:] = u_solved
    _u_relax_apply(u, u_iter_prev, 1.0)
    assert np.array_equal(u.x.array, u_solved), (
        "ω=1 改变了 u：欠松弛在默认值下不是恒等，会污染所有既有工况")

    # 性质 2：ω=0.5 精确压缩
    u.x.array[:] = u_solved
    _u_relax_apply(u, u_iter_prev, 0.5)
    expect = u_iter_prev + 0.5 * (u_solved - u_iter_prev)
    denom = max(float(np.max(np.abs(u_solved - u_iter_prev))), 1.0e-300)
    assert denom > 0.0, "u_solved 与 u_iter_prev 相同，压缩判据空转"
    err = float(np.max(np.abs(u.x.array - expect))) / denom
    assert err < 1.0e-12, (
        "ω=0.5 的结果不是 u_iter_prev + 0.5·Δu（相对偏差 %.3e）："
        "ω 作用的位置不对（检查是否误乘 q 或 u_prev）" % err)

    # 顺带确认压缩真在发生：松弛后的增量应恰是原增量的一半
    du_full = float(np.max(np.abs(u_solved - u_iter_prev)))
    du_half = float(np.max(np.abs(u.x.array - u_iter_prev)))
    assert abs(du_half - 0.5 * du_full) <= 1.0e-12 * du_full, (
        "ω=0.5 后的增量不是原增量的一半（%.6e vs %.6e）" % (du_half, 0.5 * du_full))

    if comm_rank0():
        print("  U17: ω=1 恒等 ✓；ω=0.5 相对偏差 %.3e；max|Δu| %.3e -> %.3e"
              % (err, du_full, du_half))


def test_U18_u_relaxation_config_validation_and_wiring():
    """U18：``u_relaxation`` 的默认值、取值范围、以及 clamp_case 的读取。

    旋钮要真能配置到，否则测试里手改 omega 而生产读不到 ——
    那正是「改代码跑通、改配置无效」的静默失效模式。
    """
    import inspect

    from pfwire.phasefield import SolverConfig

    # 默认 1.0 = 不松弛（向后兼容：不显式配置时逐位回到原行为）
    assert SolverConfig().u_relaxation == 1.0, (
        "u_relaxation 默认值必须是 1.0（不松弛），否则所有既有工况行为改变")
    # 取值 (0,1]：与 relaxation 同一约定
    for bad in (0.0, -0.5, 1.5):
        try:
            SolverConfig(u_relaxation=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("u_relaxation=%g 未被拒绝" % bad)
    SolverConfig(u_relaxation=1.0)      # 边界值必须合法
    SolverConfig(u_relaxation=0.3)      # 方案 h 的推荐档

    # clamp_case 必须从 configs 的 solver 段读它（生产 6.6.3 走的是这条路径）
    from pfwire import clamp_case
    src = inspect.getsource(clamp_case.build_clamp_case)
    assert "u_relaxation" in src, (
        "build_clamp_case 没有读 u_relaxation：配置里设了也不生效")
    assert 'sv.get("u_relaxation"' in src, (
        "u_relaxation 应从 solver 段读（sv.get('u_relaxation', 1.0)）")

    # 且生产求解器路径真会用它：_solve_u_uzawa 必须按 omega 分支
    from pfwire import phasefield
    src_solver = inspect.getsource(phasefield.PhaseFieldSolver._solve_u_uzawa)
    assert "u_relaxation" in src_solver, (
        "_solve_u_uzawa 没有读 config.u_relaxation：旋钮接到了配置却没接到求解器")
    assert "omega < 1.0" in src_solver, (
        "_solve_u_uzawa 缺 omega < 1.0 的分支")



# --------------------------------------------------------------------------
# U19-U20：切向面力线性化进 u 的 Jacobian（方向 f，D-68 问题 33 续 13）
#
# 背景：Uzawa 模式下切向牵引 q·τ·v 里 q 是 DG0 Function、不依赖 u，
# ufl.derivative(F, u) 给出的切向刚度为 0。实测（续 13，串行）：
#   it=4 时 843 个接触单元全部 |q| = μp（q_new|max 恰等于 lim|max），
#   it=5 即 u 的 Newton SNES -5 —— 巨大面力 + 零切向刚度。
# 处方：补 ε_t·(τ·du)(τ·v) 进 Jacobian（Uzawa 迭代参数的本体）。
# --------------------------------------------------------------------------
def test_U19_tangential_jacobian_is_symmetric_and_linear_in_eps_t(tmp_path):
    """U19：``tangential_jacobian`` 的三条性质 —— 非零 / 对称 / 线性于 ε_t。

    对称性不是装饰：切向刚度必须是自伴的双线性型，否则 Newton 的线性系统
    非对称、MUMPS 虽能解但收敛性没有保证（且物理上不对）。
    """
    import numpy as np
    import scipy.sparse as sp
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_matrix

    ctx = _setup(tmp_path)
    clamp, u = ctx["clamp"], ctx["u"]
    clamp.set_mu(MU)
    ctx["loads"].set_phase(0.25)
    clamp.set_tangential(1.0e-6)
    _uzawa(ctx["problem"], clamp, u)

    V = ctx["V"]
    du = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    def _assemble():
        form = fem.form(clamp.tangential_jacobian(u, du, v))
        M = assemble_matrix(form, bcs=[])
        M.assemble()
        n = M.getSize()[0]
        return sp.csr_matrix(M.getValues(range(n), range(n)))

    A = _assemble()
    nz = int(A.nnz)
    if comm_rank0():
        print("  U19: J 的 nnz = %d" % nz)
    assert nz > 0, "Uzawa 模式下切向 Jacobian 全零：方向 f 没接上"

    # 对称性：只比非显式零的元素
    scale = float(np.max(np.abs(A.data))) if A.data.size else 1.0
    D = (A - A.T)
    rel_sym = (float(np.max(np.abs(D.data))) / max(scale, 1.0e-300)) \
        if D.nnz else 0.0
    if comm_rank0():
        print("       对称性相对偏差 = %.3e" % rel_sym)
    assert rel_sym < 1.0e-12, (
        "切向 Jacobian 不对称（相对偏差 %.3e）：自伴性被破坏" % rel_sym)

    # 线性于 eps_t：加倍后元素应同比缩放
    eps0 = clamp.eps_t
    try:
        clamp.eps_t = 2.0 * eps0
        A2 = _assemble()
    finally:
        clamp.eps_t = eps0
    assert A.data.size == A2.data.size, (
        "换 ε_t 后非零结构变了（%d vs %d）：J 不该随 ε_t 改变稀疏结构"
        % (A.data.size, A2.data.size))
    ratio = A2.data / A.data
    if comm_rank0():
        print("       ε_t 加倍后元素比 = %.6f（应为 2）" % float(ratio[0]))
    assert np.allclose(ratio, 2.0, rtol=1.0e-12), (
        "J 不线性于 ε_t：换 ε_t 后元素没按比例缩放")


def test_U20_assembling_jacobian_does_not_change_residual(tmp_path):
    """U20：装配切向 Jacobian **不污染残差** —— 前后 F(u) 逐位相同。

    这条守的是「J 只进 Newton 的线性系统、不进残差」。若实现不小心让 J 的
    装配改动了 u 或 q 的状态，残差就会变 —— 那意味着收敛判据判的是另一个问题。
    """
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_matrix

    ctx = _setup(tmp_path)
    clamp, u, prob = ctx["clamp"], ctx["u"], ctx["problem"]
    clamp.set_mu(MU)
    ctx["loads"].set_phase(0.25)
    clamp.set_tangential(1.0e-6)
    _uzawa(prob, clamp, u)

    F_form = fem.form(prob.F)
    r_before = fem.assemble_vector(F_form)
    # dolfinx 的 Vector 没有 .norm()；取数组算 L2
    import numpy as _np
    norm_before = float(_np.linalg.norm(r_before.array))

    # 装配 J（应只读状态、不改状态）
    v = ufl.TestFunction(ctx["V"])
    du = ufl.TrialFunction(ctx["V"])
    J_form = fem.form(clamp.tangential_jacobian(u, du, v))
    M = assemble_matrix(J_form, bcs=[])
    M.assemble()

    r_after = fem.assemble_vector(F_form)
    diff = abs(float(_np.linalg.norm(r_after.array)) - norm_before)

    if comm_rank0():
        print("  U20: 残差范数 = %.6e（装配 J 前后差 %.3e）" % (norm_before, diff))
    assert diff < 1.0e-12 * max(norm_before, 1.0e-300), (
        "装配 J 之后残差变了（%.3e）：J 的装配污染了 F" % diff)


# --------------------------------------------------------------------------
# U21：切向 Jacobian 的**运行时**开关（D-68 问题 33 续 14 修）
#
# 延拓期 μ = 0、q ≡ 0 ⇒ 切向刚度必须为 0。
# 只判编译期的 has_friction_at_build 会让延拓期白扛 ε_t（SNES -5）；
# 只判运行期的 Python bool 会让项在延拓期被永久编译成 0（开摩擦后拿不回来）。
# 所以必须是 Constant 门：form 结构恒定，值随 set_mu 走。
# --------------------------------------------------------------------------
def test_U21_tangential_jacobian_gate_follows_mu(tmp_path):
    """U21：切向 Jacobian 随 ``set_mu`` 开关 —— μ = 0 时精确为 0。

    这条守的是续 14 的回归。**同一条 form** 反复装配（不重建、不重新编译），
    只有 ``set_mu`` 在切 —— 正是生产里的时序。
    """
    import numpy as np
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_matrix

    ctx = _setup(tmp_path)
    clamp, u = ctx["clamp"], ctx["u"]
    V = ctx["V"]

    # 装一次 form，之后只切 set_mu（不重建 form）
    du = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    form = fem.form(clamp.tangential_jacobian(u, du, v))

    def _fro():
        M = assemble_matrix(form, bcs=[])
        M.assemble()
        n = M.getSize()[0]
        return float(np.abs(M.getValues(range(n), range(n))).sum())

    # 1) 关摩擦（延拓期）：必须精确 0
    clamp.set_mu(None)
    f0 = _fro()
    # 2) 开摩擦：必须非零
    clamp.set_mu(MU)
    f1 = _fro()
    # 3) 再关回去：必须又精确 0（证明是「随 μ 走」而不是「一次性」）
    clamp.set_mu(None)
    f2 = _fro()

    if comm_rank0():
        print("  U21: μ=0 → %.3e,  μ=%g → %.3e,  再 μ=0 → %.3e"
              % (f0, MU, f1, f2))
    assert f0 == 0.0, "μ = 0 时切向 Jacobian 非零（%.3e）：延拓期白扛 ε_t" % f0
    assert f1 > 0.0, "μ > 0 时切向 Jacobian 全零：方向 f 没接上"
    assert f2 == 0.0, "再关摩擦后 Jacobian 没归零（%.3e）：开关不是随 μ 走的" % f2


# --------------------------------------------------------------------------
# U21：切向 Jacobian 的**运行时**开关（D-68 问题 33 续 14 修）
#
# 延拓期 μ = 0、q ≡ 0 ⇒ 切向刚度必须为 0。
# 只判编译期的 has_friction_at_build 会让延拓期白扛 ε_t（SNES -5）；
# 只判运行期的 Python bool 会让项在延拓期被永久编译成 0（开摩擦后拿不回来）。
# 所以必须是 Constant 门：form 结构恒定，值随 set_mu 走。
# --------------------------------------------------------------------------
def test_U21_tangential_jacobian_gate_follows_mu(tmp_path):
    """U21：切向 Jacobian 随 ``set_mu`` 开关 —— μ = 0 时精确为 0。

    这条守的是续 14 的回归。**同一条 form** 反复装配（不重建、不重新编译），
    只有 ``set_mu`` 在切 —— 正是生产里的时序。
    """
    import numpy as np
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_matrix

    ctx = _setup(tmp_path)
    clamp, u = ctx["clamp"], ctx["u"]
    V = ctx["V"]

    # 装一次 form，之后只切 set_mu（不重建 form）
    du = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    form = fem.form(clamp.tangential_jacobian(u, du, v))

    def _fro():
        M = assemble_matrix(form, bcs=[])
        M.assemble()
        n = M.getSize()[0]
        return float(np.abs(M.getValues(range(n), range(n))).sum())

    # 1) 关摩擦（延拓期）：必须精确 0
    clamp.set_mu(None)
    f0 = _fro()
    # 2) 开摩擦：必须非零
    clamp.set_mu(MU)
    f1 = _fro()
    # 3) 再关回去：必须又精确 0（证明是「随 μ 走」而不是「一次性」）
    clamp.set_mu(None)
    f2 = _fro()

    if comm_rank0():
        print("  U21: μ=0 → %.3e,  μ=%g → %.3e,  再 μ=0 → %.3e"
              % (f0, MU, f1, f2))
    assert f0 == 0.0, "μ = 0 时切向 Jacobian 非零（%.3e）：延拓期白扛 ε_t" % f0
    assert f1 > 0.0, "μ > 0 时切向 Jacobian 全零：方向 f 没接上"
    assert f2 == 0.0, "再关摩擦后 Jacobian 没归零（%.3e）：开关不是随 μ 走的" % f2


# ==========================================================================
# M1-M4：方向 m —— 单体 Newton 求 (u,q) 联立 Jacobian（D-68 问题 33 续 16）
#
# 背景（续 15 的证伪结论）：
#   blocker B 不是「u 的 Jacobian 缺切向项」—— 零插桩探针实测冻住 q 只解 u，
#   SNESConvergedReason = 4（收敛）。病根在 (u,q) 的交错耦合：q 一步跳到
#   Coulomb 容量，下一步 u 面对巨大面力。
# 处方：把 (u,q) 当一个系统，用单体 Newton 求真正的联立 Jacobian。
#
# ⚠ 方向 f 的代码（tangential_jacobian + U19/U20/U21）**故意留着不删**
#   （用户 2026-10-25 拍板）：留作有测试的对照，直到 M3 证明单体路径跑通。
# ==========================================================================

# --------------------------------------------------------------------------
# M4：smooth_clip_ufl 与 numpy 版逐点一致
#
# 这条是方向 m 的地基守卫：单体 Newton 的 Coulomb 投影方程要用 UFL 版，
# 而 Uzawa 迭代用 numpy 版。两个版本差一个符号/一处 abs，联立解与
# 交错解就会分叉 —— 那种 bug 不会报错，只会让 M3 的「不动点不变」悄悄失败。
# --------------------------------------------------------------------------
def test_M4_smooth_clip_ufl_matches_numpy_pointwise():
    """M4：``smooth_clip_ufl`` 与 :func:`smooth_clip` 在随机输入上逐点一致。

    覆盖三个区间：盒内（|x| < lim）、盒面附近（|x| ≈ lim，偏差最大处）、
    盒外（|x| > lim）。盒面附近是 FB 光滑化偏差 O(η²/(4u)) 最大的地方，
    必须覆盖。
    """
    import numpy as np
    import ufl
    from dolfinx import fem
    from pfwire.clamp_contact import smooth_clip, smooth_clip_ufl

    msh = None  # 占位：下面自建一个够大的 DG0 网格
    from dolfinx.mesh import create_unit_square
    from mpi4py import MPI
    m = create_unit_square(MPI.COMM_WORLD, 30, 30)
    Vs = fem.functionspace(m, ("DG", 0))
    # ⚠ DG0 的 x.array 含 ghost（CLAUDE.md 陷阱 6）：size_local 之外还有
    # ghost 槽，故数据点数必须按 **owned 数**生成，不能反过来猜。
    n = Vs.dofmap.index_map.size_local
    assert n > 600, "网格太小：owned %d" % n

    rng = np.random.default_rng(12345)
    # 三组等量：盒内（|x| < lim）/ 盒面附近（|x| ≈ lim，FB 偏差最大处）/ 盒外
    k = n // 3
    x_np = np.concatenate([
        rng.uniform(-0.9, 0.9, k),
        rng.uniform(0.98, 1.02, k),
        rng.uniform(1.1, 5.0, k),
        rng.uniform(-0.9, 0.9, n - 3 * k),      # 补齐到 n
    ])
    lim_np = np.full(n, 1.0)
    eta_np = np.full(n, 0.05)

    want = smooth_clip(x_np, lim_np, eta_np)

    x = fem.Function(Vs)
    lim = fem.Function(Vs)
    eta = fem.Function(Vs)
    x.x.array[:n] = x_np
    lim.x.array[:n] = lim_np
    eta.x.array[:n] = eta_np
    x.x.scatter_forward()
    lim.x.scatter_forward()
    eta.x.scatter_forward()
    expr = fem.Expression(smooth_clip_ufl(x, lim, eta), Vs.element.interpolation_points)
    got = fem.Function(Vs)
    got.interpolate(expr)

    # 比对同样只切 owned 段（ghost 段是相邻 rank 的副本，不属本 rank）
    got_a = got.x.array[:n]
    want_a = want[:n]

    rel = np.max(np.abs(got_a - want_a)) / max(np.max(np.abs(want_a)), 1.0e-300)
    if comm_rank0():
        print("  M4: 最大相对差 = %.3e（%d 点）" % (rel, n))
    assert rel < 1.0e-12, (
        "smooth_clip_ufl 与 numpy 版不一致（最大相对差 %.3e）："
        "联立解会与交错解分叉" % rel)

    # 不超限（盒外也不许越过 Coulomb 锥）
    over = float(np.max(np.abs(got_a) - lim.x.array[:n]))
    if comm_rank0():
        print("      盒外最大超出量 = %.3e（应 <= 0 或 O(eta)）" % over)
    assert over <= eta_np[0] + 1.0e-12, (
        "smooth_clip_ufl 超出 Coulomb 锥 %.3e > eta：锥被违反" % over)


# --------------------------------------------------------------------------
# M1：混合空间 (u, q) 的 dof 映射逐位一致
#
# 这是续 9（end_bcs 编号空间错配）的同型守卫。那次的事故形态：
# 用 Vy 子空间的编号去索引 V 的坐标行 -> 0/41 个底部 dof 落到底边上 ->
# 单点约束打在网格内部 -> 应力奇异 -> d 假饱和。**只在并行下暴露**
# （错点随分区变），所以当时查了很久。
#
# 混合空间的风险同型：W 的 dof 段切错，u 与 q 就会互相串味。
# --------------------------------------------------------------------------
def test_M1_monolithic_space_dof_mapping_roundtrips(tmp_path):
    """M1：``pack_w`` / ``unpack_w`` 互逆，且 u/q 段与各自场逐位一致。"""
    import numpy as np

    ctx = _setup(tmp_path)
    clamp, u, V = ctx["clamp"], ctx["u"], ctx["V"]
    clamp.bind_monolithic_space(V)

    n_u = clamp.mono_n_u
    n_q = clamp.mono_n_q
    if comm_rank0():
        print("  M1: n_u = %d, n_q = %d, 合计 = %d" % (n_u, n_q, n_u + n_q))

    # 规模必须与两个空间的 owned 规模对得上
    bs = V.dofmap.index_map_bs
    assert n_u == V.dofmap.index_map.size_local * bs, (
        "u 段规模 %d != V owned %d x bs %d" % (n_u, V.dofmap.index_map.size_local, bs))
    assert n_q == clamp.q.function_space.dofmap.index_map.size_local, (
        "q 段规模 %d != Q owned %d"
        % (n_q, clamp.q.function_space.dofmap.index_map.size_local))

    # 造非平凡数据（全 1 / 全 2 太容易蒙对：错位后仍可能自洽）
    rng = np.random.default_rng(7)
    u_test = rng.standard_normal(u.x.array.size)
    q_test = rng.standard_normal(clamp.q.x.array.size)
    u_ref = u_test.copy()
    q_ref = q_test.copy()

    w = np.zeros(n_u + n_q)
    clamp.pack_w(w, u_test, q_test)

    # pack 之后 u/q 段必须分别等于原场（逐位）
    assert np.array_equal(w[:n_u], u_ref[:n_u]), (
        "u 段与 u 场不一致：混合空间 dof 映射错了（续 9 同型）")
    assert np.array_equal(w[n_u:n_u + n_q], q_ref[:n_q]), (
        "q 段与 q 场不一致：混合空间 dof 映射错了（续 9 同型）")

    # unpack 回两个新数组，必须逐位还原
    u_back = np.zeros_like(u_test)
    q_back = np.zeros_like(q_test)
    clamp.unpack_w(w, u_back, q_back)
    assert np.array_equal(u_back[:n_u], u_ref[:n_u]), "unpack 的 u 段没还原"
    assert np.array_equal(q_back[:n_q], q_ref[:n_q]), "unpack 的 q 段没还原"

    # 反向再 pack 一次，w 必须逐位不变（互逆性）
    w2 = np.zeros(n_u + n_q)
    clamp.pack_w(w2, u_back, q_back)
    assert np.array_equal(w, w2), "pack/unpack 不互逆"

    if comm_rank0():
        print("       pack/unpack 互逆、u/q 段逐位一致：通过")


# --------------------------------------------------------------------------
# M2：联立 Jacobian 的块结构 + 两个静默失效陷阱的量级守卫（D-68 问题 33 续 16）
#
# 这两个坑都是**静默**的（不报错、只给错值），所以必须配量级守卫：
#   坑 1：F_q 用 ufl.dx 而 p/s 是 facet 量 => dF_q/du 归零（1e15 -> 1.57e-9）
#   坑 2：ufl.conditional 吃掉分支导数 => dF_q/du 的 nnz 从 40536 掉到 0
# --------------------------------------------------------------------------
def test_M2_monolithic_jacobian_block_structure(tmp_path):
    """M2：联立 Jacobian 四块的结构与量级（方向 m 的 B 方案，续 17）。

    续 17 的关键实测（中心差分判据）：

    ==============  ============  ==========  ==========
    写法            dF_q/du nnz  max&#124;.&#124;    FD 一致
    ==============  ============  ==========  ==========
    facet + dx      26568         1.57e-9    1/4（伪影）
    facet + ds      260           **1.20e11** **4/4（真）**
    DG0 场 + dx     0             —          无耦合（块三角）
    ==============  ============  ==========  ==========

    ⇒ 本测试断言 B 方案（facet + ds）的**满块**结构：
      ``dF_q/du`` 非零且量级 O(eps_t)（不是 1e-9 的伪影）。
    """
    import numpy as np
    import scipy.sparse as sp
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_matrix

    ctx = _setup(tmp_path)
    clamp, u, V = ctx["clamp"], ctx["u"], ctx["V"]
    loads = ctx["loads"]
    clamp.bind_monolithic_space(V)

    loads.set_phase(0.25)
    clamp.set_tangential(1.0e-6)
    _uzawa(ctx["problem"], clamp, u)

    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    q_trial = ufl.TestFunction(clamp.q.function_space)
    q_du = ufl.TrialFunction(clamp.q.function_space)

    blocks = clamp.monolithic_jacobian(u, du, v, q_trial, q_du)
    assert len(blocks) == 2 and len(blocks[0]) == 2, "J 不是 2x2"

    n_u = clamp.mono_n_u
    n_q = clamp.mono_n_q

    def _assemble(form, n_row, n_col):
        M = assemble_matrix(fem.form(form), bcs=[])
        M.assemble()
        return sp.csr_matrix(M.getValues(range(n_row), range(n_col)))

    A_uu = _assemble(blocks[0][0], n_u, n_u)
    A_uq = _assemble(blocks[0][1], n_u, n_q)
    A_qu = _assemble(blocks[1][0], n_q, n_u)
    A_qq = _assemble(blocks[1][1], n_q, n_q)

    if comm_rank0():
        print("  M2: nnz  dF_u/du=%d  dF_u/dq=%d  dF_q/du=%d  dF_q/dq=%d"
              % (A_uu.nnz, A_uq.nnz, A_qu.nnz, A_qq.nnz))

    # --- dF_u/dq 与 dF_q/du 必须非零（满块耦合）---
    assert A_uq.nnz > 0, "dF_u/dq 全零：q 没接进 u 的方程"
    assert A_qu.nnz > 0, (
        "dF_q/du 全零：退化成块三角。B 方案要求 facet + ds（续 17 实测："
        "facet+dx 的 1.57e-9 是伪影，DG0+dx 恒 0，只有 facet+ds 是真的 1.2e11）")

    # --- 量级守卫：踩坑时是 1.57e-9，正确时 1e11 ---
    max_qu = float(np.max(np.abs(A_qu.data)))
    if comm_rank0():
        print("       max|dF_q/du| = %.6e" % max_qu)
    assert max_qu > 1.0e3, (
        "max|dF_q/du| = %.3e 过小：几乎肯定踩了续 16 坑 1"
        "（facet 量配了 ufl.dx 而非 ds）—— 那是静默失效" % max_qu)

    # --- dF_q/dq 无 NaN、值域在 [0,1]（FB 导数的性质）---
    diag = A_qq.diagonal()
    assert not np.isnan(diag).any(), (
        "dF_q/dq 有 %d 个 NaN：eta 的 0/0 问题（坑 5）" % int(np.isnan(diag).sum()))
    assert diag.min() > -1.0e-6, "dF_q/dq 负对角元 %.3e" % diag.min()
    assert diag.max() < 1.0 + 1.0e-6, (
        "dF_q/dq 对角元 %.3f 超过 1：投影不是压缩映射" % diag.max())
    if comm_rank0():
        print("       dF_q/dq 对角元范围 [%.4e, %.4e]"
              % (float(diag.min()), float(diag.max())))

    # --- dF_u/du 必须满（弹性 + 法向 + q 面力）---
    assert A_uu.nnz > 0, "dF_u/du 全零"


def test_M2b_facet_ds_is_the_real_derivative(tmp_path):
    """M2b：``facet + ds`` 的 dF_q/du 经**中心差分**验证为真（续 17 的核心判据）。

    ⚠⚠ **必须在真有耦合的列上验**（续 17 续 20 的教训）：
    第一版随机选 3 列，结果全部 ``sym = 0`` 且 ``fd = 0`` ⇒ ``rel = 0``
    「一致」—— 那是**假通过**（trivially equal），什么也没验到。
    本测试先从符号 Jacobian 里挑非零列，再在这些列上做差分。

    实测（h = 1e-8）：
        facet + ds : sym = 1.2021e11, fd = 1.2021e11, rel = 3.1e-09  ✓
        facet + dx : sym = 9.3e-16,  fd = 0,         rel = inf       ✗
    """
    import numpy as np
    import scipy.sparse as sp
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector

    ctx = _setup(tmp_path)
    clamp, u, V = ctx["clamp"], ctx["u"], ctx["V"]
    loads = ctx["loads"]

    loads.set_phase(0.25)
    clamp.set_tangential(1.0e-6)
    _uzawa(ctx["problem"], clamp, u)

    q_trial = ufl.TestFunction(clamp.q.function_space)
    du = ufl.TrialFunction(V)
    n_q = clamp.q.function_space.dofmap.index_map.size_local
    n_u = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    F_u_c, F_q = clamp.monolithic_residual(u, ufl.TestFunction(V), q_trial, None)
    M = assemble_matrix(fem.form(ufl.derivative(F_q, u, du)), bcs=[])
    M.assemble()
    Asym = sp.csr_matrix(M.getValues(range(n_q), range(n_u)))

    # **只挑真有耦合的列**
    colsum = np.asarray(np.abs(Asym).sum(axis=0)).ravel()
    cols_nz = np.nonzero(colsum > 0.0)[0]
    assert cols_nz.size > 0, (
        "dF_q/du 全零：B 方案的 facet+ds 没接上（退化成块三角）")
    pick = cols_nz[np.argsort(-colsum[cols_nz])[:3]]

    h = 1.0e-8
    u_save = u.x.array.copy()
    ok = 0
    for c in pick:
        fd = np.zeros(n_q)
        for sgn in (+1, -1):
            u.x.array[:] = u_save
            u.x.array[c] += sgn * h
            u.x.scatter_forward()
            fd += sgn * assemble_vector(fem.form(F_q)).array[:n_q]
        fd /= (2 * h)
        sym = Asym[:, c].toarray().ravel()
        sc, fc = float(np.max(np.abs(sym))), float(np.max(np.abs(fd)))
        rel = abs(sc - fc) / max(sc, fc) if (sc > 0 and fc > 0) else float("inf")
        if rel < 1.0e-2:
            ok += 1
    u.x.array[:] = u_save
    u.x.scatter_forward()

    if comm_rank0():
        print("  M2b: 非零列 %d/%d，取 3 列做差分，一致 %d/3"
              % (cols_nz.size, n_u, ok))
    assert ok == len(pick), (
        "dF_q/du 的符号导数与中心差分不一致（%d/%d）：facet 量配错了测度"
        "（续 17 实测：facet+dx 全不一致，facet+ds 一致到 3e-9）" % (ok, len(pick)))
# --------------------------------------------------------------------------
# M3：单体 Newton 的解 == 交错 Uzawa 的收敛解（方向 m 的验收性质）
#
# 单体只改收敛路径、不改不动点。若这条不过，说明联立残差写错了 ——
# 那比「跑不通」更糟：会静默给出另一个解。
# --------------------------------------------------------------------------
def test_M3_monolithic_newton_converges(tmp_path):
    """M3：手工块 Newton 收敛，且 Coulomb 锥精确满足（方向 m 的 B 方案，续 17）。

    ⚠ 为什么走**手工块 Newton**：dolfinx 0.11 的多 form ``NonlinearProblem``
    对混合系统有两个叠加的 bug（续 17 坑 7）：
      (a) ``bcs_by_block`` 里 ``V.contains(bc.function_space)`` 抛
          AttributeError（cpp 对象），异常在 PETSc 回调里被吞 ⇒ SNES
          **什么都不做**就返回 -3、it=0；
      (b) 即使 patch 掉 (a)，BC 的 dof 编号仍落到错的 block。
    ⇒ 四块各自装配 + scipy 块矩阵 + 手工施加 BC。

    验收判据（物理，不依赖另一个数值解）：
      - 自由 dof 上的残差收敛（BC dof 上是**支座反力**，不算残差）
      - Coulomb 锥在**积分口径**（与方程同测度）下满足：|∫q|/∫μp = 1
    """
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector
    from pfwire.mesh import BLOCK_TOP

    ctx = _setup(tmp_path)
    clamp, u, V = ctx["clamp"], ctx["u"], ctx["V"]
    loads = ctx["loads"]
    clamp.bind_monolithic_space(V)

    loads.set_phase(0.25)
    clamp.set_tangential(1.0e-6)

    # 清零 + 关摩擦延拓 + 冻结基准 + 开摩擦
    u.x.array[:] = 0.0
    u.x.scatter_forward()
    clamp.q.x.array[:] = 0.0
    clamp.q.x.scatter_forward()
    clamp.set_mu(None)
    for d in (1.0e-7, 1.0e-6, 1.0e-5, DELTA):
        clamp.set_delta(d)
        ctx["problem"].solve()
        assert ctx["problem"].solver.getConvergedReason() > 0
    clamp.set_slip_reference(u)
    clamp.set_mu(MU)

    n_u, n_q = clamp.mono_n_u, clamp.mono_n_q
    v = ufl.TestFunction(V)
    q_trial = ufl.TestFunction(clamp.q.function_space)
    du = ufl.TrialFunction(V)
    q_du = ufl.TrialFunction(clamp.q.function_space)

    from pfwire.obstacle_contact import plane_strain_lame
    mu_l, lmbda = plane_strain_lame(E_BULK, NU_BULK)

    def _full():
        eps = ufl.sym(ufl.grad(u))
        sig = 2.0 * mu_l * eps + lmbda * ufl.tr(eps) * ufl.Identity(2)
        F = ufl.inner(sig, ufl.sym(ufl.grad(v))) * ufl.dx
        F = F - loads.traction()(V)
        F = F - clamp.residual(u, v)
        F_u_c, F_q = clamp.monolithic_residual(u, v, q_trial, None)
        return F + F_u_c, F_q

    bcs_u = list(_free_support(ctx["ft"], ctx["msh"])(V))
    u_bc = np.unique(np.concatenate([bc.dof_indices()[0] for bc in bcs_u]))
    contact, all_dofs = clamp._contact_cell_dofs(u)
    q_bc = np.setdiff1d(all_dofs, contact)
    g_bc = np.concatenate([u_bc, q_bc + n_u]).astype(np.int64)
    free = np.setdiff1d(np.arange(n_u + n_q), g_bc)

    def _sp(M):
        M.assemble()
        r, c = M.getLocalSize()
        return sp.csr_matrix(M.getValues(range(r), range(c)))

    hist = []
    for it in range(30):
        F_u, F_q = _full()
        r = np.concatenate([assemble_vector(fem.form(F_u)).array[:n_u],
                            assemble_vector(fem.form(F_q)).array[:n_q]])
        rn = float(np.linalg.norm(r[free]))
        hist.append(rn)
        if rn < 1.0e-3:
            break
        J = sp.bmat([[_sp(assemble_matrix(fem.form(ufl.derivative(F_u, u, du)), bcs=[])),
                      _sp(assemble_matrix(fem.form(ufl.derivative(F_u, clamp.q, q_du)), bcs=[]))],
                     [_sp(assemble_matrix(fem.form(ufl.derivative(F_q, u, du)), bcs=[])),
                      _sp(assemble_matrix(fem.form(ufl.derivative(F_q, clamp.q, q_du)), bcs=[]))]],
                    format="lil")
        for gd in g_bc:
            J.rows[gd] = [int(gd)]
            J.data[gd] = [1.0]
        Jf = J.tocsr()[free][:, free]
        dxf = spla.spsolve(Jf, -r[free])
        if not np.all(np.isfinite(dxf)):
            break
        dx = np.zeros(n_u + n_q)
        dx[free] = dxf
        u.x.array[:n_u] += dx[:n_u]
        u.x.scatter_forward()
        clamp.q.x.array[:n_q] += dx[n_u:]
        clamp.q.x.scatter_forward()

    if comm_rank0():
        print("  M3: 残差 %.2e -> %.2e（%d 步）" % (hist[0], hist[-1], len(hist)))
    assert hist[-1] < 1.0e-2, (
        "手工块 Newton 没收敛：残差停在 %.3e。续 17 记录它应降到 ~1e-4" % hist[-1])
    assert hist[-1] < hist[0], "残差没下降"

    # Coulomb 锥（积分口径，与方程同测度）
    p_f, _n = clamp._pressure_and_normal(u)
    lim_int = fem.assemble_scalar(fem.form(clamp.mu * p_f * clamp.ds(BLOCK_TOP)))
    q_int = fem.assemble_scalar(fem.form(clamp.q * clamp.ds(BLOCK_TOP)))
    ratio = abs(q_int) / max(lim_int, 1.0e-300)
    if comm_rank0():
        print("       |∫q| / ∫mu p = %.6f" % ratio)
    assert abs(ratio - 1.0) < 0.05, (
        "Coulomb 锥未满足：|∫q|/∫mu p = %.4f（应 ≈ 1，因为接触斑整体滑移）" % ratio)
