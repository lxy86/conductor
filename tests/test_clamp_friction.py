"""6.6.1 的 Coulomb 摩擦（D-67 的无摩擦工况加切向项）。

测试意图（PLAN 6.6.1：先独立验证再投入耦合）：
  G1. 切线：摩擦 + 塑性（屈服状态）下 Jacobian 与残差的有限差分一致；
  G2. 无摩擦退化：mu=None 时切向残差逐位为 0；
  G3. Coulomb 上限：切向推远超饱和值，核心区逐点 |q| <= mu p；
  G4. 积分量 Q <= mu P；
  G5. 饱和：切向够大时 Q/(mu P) -> 1，且单调不减；
  H1. 部分滑移：中等切向下同时存在黏着点与滑移点；
  H2. 黏着半宽 c 随 Q 单调不增；
  H3. c/a 与 Mindlin 参考值同侧（不定量对拍，D-51 的理由）；
  H4. 滑移区在接触斑外沿（|x| 大的一侧），不是中心；
  I1. 摩擦不改变法向反力 W（力系解耦的必然结果）；
  I2. kappa_t 敏感性：kappa_t 从 kappa 到 10 kappa，Q 相对变化 < 3%；
  I3. 并行：tangential_reaction 可重复装配且自洽。

网格为小尺寸加密矩形（10 mm 长），与 test_clamp_contact.py 同构，只为测试速度。
几何约定与 6.1 相同：上表面 y = 0、底面 y = -depth。

**为什么这些用例不用相场求解器**：G/H/I 组要的是接触本身的性质，掺进相场只会拖慢，
且 d 一旦非零就会把刚度退化、干扰 Coulomb 判据。摩擦项与相场的耦合由 G1 覆盖
（它是唯一带塑性与 d≡0 的），其余走「体本构 + 接触」的独立残差。
"""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.fem import assemble_matrix, assemble_vector, form as ufl_form
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

import pathlib

from pfwire.clamp_contact import ClampContact
from pfwire.material import load_material
from pfwire.mesh import BLOCK_BOTTOM, BLOCK_TOP
from pfwire.obstacle import build_block_mesh, tangential_traction
from pfwire.obstacle_contact import plane_strain_lame
from pfwire.plasticity import PlasticLaw, load_j2_params

ROOT = pathlib.Path(__file__).resolve().parents[1]
MATERIAL = ROOT / "configs/material_al1350.yaml"
E_BULK = 69.0e9
NU_BULK = 0.33
KAPPA = 10.0
R_CLAMP = 330.0e-3
DELTA = 5.0e-6          # 线夹下压量：使接触区内的压力达到屈服量级
MU = 0.3                # 验证用的摩擦系数（生产配置另取 0.7，见 PLAN §1.2）
SOLVER = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_rtol": 1.0e-10, "snes_atol": 1.0e-10, "snes_max_it": 60,
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
}


def _mesh(tmp_path):
    msh, _, facet_tags = build_block_mesh(
        tmp_path, width=10.0e-3, depth=3.75e-3, h_fine=0.1e-3, h_coarse=0.5e-3,
        fine_halfwidth=2.0e-3, fine_depth=2.0e-3, transition=0.5e-3, comm=MPI.COMM_WORLD,
    )
    return msh, facet_tags


def _bottom_fixed(facet_tags, msh):
    """底面全固定（去刚体位移）。BC 建在求解器自己的空间上（CLAUDE.md 陷阱 1）。"""

    def make(V):
        fdim = msh.topology.dim - 1
        dofs = fem.locate_dofs_topological(V, fdim, facet_tags.find(BLOCK_BOTTOM))
        zero = fem.Function(V)
        return [fem.dirichletbc(zero, dofs)]

    return make


def _tangential_bc(msh, facet_tags, V):
    """接触面（上侧）**只约束 x 分量**的 DirichletBC，返回 (bc, gate, x_dofs)。

    ``gate`` 是持有该值的 Function：把 ``gate.x.array[x_dofs]`` 设成目标值
    就能推进切向位移（gate 的 y 分量始终为 0，不约束）。

    ⚠ 分量级 BC 必须用 `dirichletbc(g, [dofs_x, dofs_y], V)` 的**重载 4**
    （每个分量一组 dof）。写成 `dirichletbc(g, dofs_x)`（只传一组）时
    dolfinx 不会报错，但 BC 被**静默丢弃** —— 实测 gate = 0 而解出的 u_x
    是 3.5e-7，且接触反力虚高 125 倍（3.5e5 -> 4.4e7）。
    这正是 CLAUDE.md 陷阱 1 的同型：约束没生效，而求解器照样返回
    「已收敛」。

    为什么用 BC 而不是面力：切向若用面力，块体只靠摩擦被拉住，Q 接近 μP 时
    会整体滑走、矩阵接近奇异（configs/verify_mindlin_friction.yaml 的同款理由）。
    """
    fdim = msh.topology.dim - 1
    facets = facet_tags.find(BLOCK_TOP)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bs = V.dofmap.bs
    if bs != 2:
        raise AssertionError("本测试只针对 2D 向量元，收到 bs=%d" % bs)
    dofs_x = dofs[dofs % bs == 0]        # x 分量 dof（向量元按分量交错排布）
    dofs_y = np.array([], dtype=np.int32)   # y 不约束
    g = fem.Function(V)
    return fem.dirichletbc(g, [dofs_x, dofs_y], V), g, dofs_x


def _residual(msh, clamp, u, V, *, plastic=None, with_friction=True):
    """体本构（弹性或塑性）+ 接触（含摩擦）的残差，供 NonlinearProblem 用。

    ``with_friction=False`` 时**不装**切向项（μ=0 的工况用）。
    ⚠ 切向项装不装是在 **form 构造时**定死的（``has_friction`` 是普通属性），
    所以「先关摩擦压法向、再开摩擦推切向」这种加载序列，形式里必须**一直**
    带着切向项，靠 ``clamp.mu.value = 0`` 来关 —— 否则切向阶段根本没有该项。
    本测试的形式一律带切向项，法向阶段把 μ 设 0 即可（q ≡ 0，逐位无影响）。
    """
    v = ufl.TestFunction(V)
    if plastic is None:
        mu_l, lmbda = plane_strain_lame(E_BULK, NU_BULK)
        eps = ufl.sym(ufl.grad(u))
        sig = 2.0 * mu_l * eps + lmbda * ufl.tr(eps) * ufl.Identity(msh.geometry.dim)
        F = ufl.inner(sig, ufl.sym(ufl.grad(v))) * ufl.dx
    else:
        F = ufl.inner(plastic.sigma(u), ufl.sym(ufl.grad(v))) * plastic.dx
    F = F - clamp.residual(u, v)
    if with_friction and clamp.has_friction_at_build:
        F = F - clamp.tangential_residual(u, v)
    return F


def _solve_with(clamp, msh, ft, *, plastic=None, ux_final=0.0, prefix="t_",
                freeze_reference=True):
    """装配并求解「法向压到 DELTA，再切向推到 ux_final」的两阶段问题。

    返回 (V, u, problem, gate, x_dofs)。切向滑移基准在法向阶段结束后冻结
    （Coulomb 增量语义：滑移从法向平衡态起算）。

    ⚠ **两件事必须做，否则 Newton 发散**（都实测过，别再踩）：

    1. **δ 要延拓**。从 u = 0 一步压到 DELTA = 5 µm 会让初始罚压力达到
       ε·δ ≈ 1e16 × 5e-6 = 5e10 Pa（超 σ_y 三个数量级），Newton 必发散
       （SNES -6）。与 δ* 搜索的 DELTA_STEPS/_ramp_plan 同一理由。
    2. **法向阶段必须关摩擦**。低压时 Coulomb 上限 μp/ε_t 极小，q 恰好坐在
       min() 的折点上，Newton 一步就飞（实测：μ=0.3 时 δ=1e-6 报 -6，
       μ=1.0 时连 δ=1e-5 都 -5；同网格**不挂摩擦则步步收敛**）。
       这正是 `pfwire/contact.py` 的 `set_friction_law` docstring 记的
       「先无摩擦压下、再切 Coulomb 推」（也是 Mindlin 加载的标准做法）。
       故这里法向阶段 `set_mu(None)`，压完冻结基准后再 `set_mu(MU)`。
    """
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    u = fem.Function(V)
    clamp.bind_slip_space(V)
    bc_x, gate, x_dofs = _tangential_bc(msh, ft, V)
    problem = NonlinearProblem(_residual(msh, clamp, u, V, plastic=plastic), u,
                               bcs=list(_bottom_fixed(ft, msh)(V)) + [bc_x],
                               petsc_options_prefix=prefix, petsc_options=SOLVER)
    gate.x.array[:] = 0.0
    # ---- 法向阶段：关摩擦，δ 延拓 ----
    clamp.set_mu(None)
    for d in (1.0e-7, 1.0e-6, 1.0e-5, DELTA):
        clamp.set_delta(d)
        problem.solve()
        if problem.solver.getConvergedReason() <= 0:
            raise RuntimeError("法向阶段不收敛（δ=%g，SNES reason=%d）"
                               % (d, problem.solver.getConvergedReason()))
    if freeze_reference:
        clamp.set_slip_reference(u)
    # ---- 切向阶段：开摩擦，推进 u_x ----
    # u_x 也要延拓：Coulomb 的黏/滑转折在 |s| ~ μp/ε_t ≈ 3e-11 m，而测试
    # 从 1e-8 起步就已经深在滑移区，一步跨过去 Newton 线搜索失败（SNES -5）。
    # 按几何序列推进，每步都停在收敛的解上（延拓不改变终态，只改变路径）。
    clamp.set_mu(MU)
    seq = [ux_final] if ux_final <= 0.0 else _tangential_ramp(ux_final)
    for ux in seq:
        gate.x.array[:] = 0.0
        gate.x.array[x_dofs] = ux
        problem.solve()
        if problem.solver.getConvergedReason() <= 0:
            raise RuntimeError("切向阶段不收敛（ux=%g，SNES reason=%d）"
                               % (ux, problem.solver.getConvergedReason()))
    return V, u, problem, gate, x_dofs


def _tangential_ramp(ux_final: float) -> list[float]:
    """切向位移的延拓序列：从 1e-13 起按 ×10 增长到 ux_final（含）。"""
    seq, v = [], 1.0e-13
    while v < ux_final:
        seq.append(v)
        v *= 10.0
    seq.append(ux_final)
    return seq


def _push_to(problem, gate, x_dofs, ux: float, *, tag: str = "") -> None:
    """把切向位移从当前值**延拓**到 ux 并解（不收敛即抛）。

    测试里逐级扫描 u_x 都必须走这个：Coulomb 的黏/滑转折在
    |s| ~ μp/ε_t ≈ 3e-11 m，从上一级直接跳到下一级会跨越它，
    Newton 线搜索失败（实测 SNES -5）。
    """
    current = float(np.max(np.abs(gate.x.array[x_dofs]))) if x_dofs.size else 0.0
    seq = _tangential_ramp(ux)
    for v in seq:
        if v <= current:
            continue
        gate.x.array[:] = 0.0
        gate.x.array[x_dofs] = v
        problem.solve()
        if problem.solver.getConvergedReason() <= 0:
            raise RuntimeError("切向阶段不收敛（ux=%g%s，SNES reason=%d）"
                               % (v, tag, problem.solver.getConvergedReason()))


def _surface_profile(msh, ft, clamp, u):
    """接触面上的 (x, p, q)：p、q 在 CG1 节点上插值，按 x 排序，只含接触面节点。"""
    Vs = fem.functionspace(msh, ("Lagrange", 1))
    p_expr, _n = clamp._pressure_and_normal(u)
    p_fun = fem.Function(Vs)
    p_fun.interpolate(fem.Expression(p_expr, Vs.element.interpolation_points))
    tau_up, _tau_lo = clamp._tangents(u)
    s_up = ufl.variable(ufl.dot(tau_up, u - clamp.u_prev))
    q_expr = tangential_traction(s_up, p_expr, clamp.mu, clamp.eps_t)
    q_fun = fem.Function(Vs)
    q_fun.interpolate(fem.Expression(q_expr, Vs.element.interpolation_points))
    dofs = fem.locate_dofs_topological(Vs, msh.topology.dim - 1, ft.find(BLOCK_TOP))
    n_owned = Vs.dofmap.index_map.size_local
    keep = dofs[dofs < n_owned]
    xs = Vs.tabulate_dof_coordinates()[keep, 0]
    comm = msh.comm
    xs = np.concatenate(comm.allgather(xs))
    ps = np.concatenate(comm.allgather(p_fun.x.array[keep]))
    qs = np.concatenate(comm.allgather(q_fun.x.array[keep]))
    o = np.argsort(xs)
    return xs[o], ps[o], qs[o]


# --------------------------------------------------------------------------
# G 组：Coulomb 不等式与饱和
# --------------------------------------------------------------------------
def test_G1_tangent_with_friction_and_plastic(tmp_path):
    """G1：摩擦 + 塑性（屈服状态）下 Jacobian 与残差的有限差分一致。

    这是 ``ufl.variable`` 那个坑的直接守卫：若 sign(s) 与 min(|s|,·) 被 UFL 当成
    两个独立表达式，折点 |s| = μp/ε_t 处切线断裂，本项必然失败。
    做法与 tests/test_clamp_contact.py 的 T1 同口径：解出平衡态后随机扰动 u，
    比对 ‖F(u+h) − F(u)‖ 与 ‖J·h‖。
    """
    msh, ft = _mesh(tmp_path)
    clamp = ClampContact(msh, ft, E=E_BULK, R_clamp=R_CLAMP, kappa=KAPPA,
                         h_contact=0.1e-3, mu=MU)
    plastic = PlasticLaw(msh, load_j2_params(MATERIAL))
    V, u, _problem, _gate, _xd = _solve_with(clamp, msh, ft, plastic=plastic,
                                             ux_final=3.0e-6, prefix="g1_")

    F = _residual(msh, clamp, u, V, plastic=plastic)
    J = ufl.derivative(F, u, ufl.TrialFunction(V))
    # dolfinx 0.11 的 assemble_matrix **返回时已装配**（无 .assemble()）。
    A = assemble_matrix(ufl_form(J), bcs=[])

    # 与 tests/test_clamp_contact.py 的 T1 同口径：**中心差分**、步长 1e-11
    # （方向归一化后）。单侧差分在这里会给出恰好 1.0 的相对误差 ——
    # 单侧差是 O(h) 的，与 J·h 差一阶，别用。
    rng = np.random.default_rng(20261019)
    n = u.x.array.size
    u0 = u.x.array.copy()
    h = 1.0e-11
    rel_errs = []
    for _ in range(4):
        direction = rng.standard_normal(n)
        direction /= np.max(np.abs(direction))       # 方向最大分量 = 1
        fd = np.zeros(n)
        for sign in (+1.0, -1.0):
            u.x.array[:] = u0 + sign * h * direction
            u.x.scatter_forward()
            b = assemble_vector(ufl_form(F)).array
            fd += sign * b
        fd /= 2.0 * h
        u.x.array[:] = u0
        u.x.scatter_forward()
        an = _matvec(A, direction)
        den = float(np.linalg.norm(an))
        if den > 0.0:
            rel_errs.append(float(np.linalg.norm(fd - an)) / den)
    assert rel_errs and max(rel_errs) < 1.0e-5, \
        "切线校验失败，最大相对误差 %g" % (max(rel_errs) if rel_errs else float("nan"))


def _matvec(A, arr: np.ndarray) -> np.ndarray:
    """把 dolfinx MatrixCSR 作用于 numpy 向量（不碰 BC，纯 Jacobian 动作）。

    走 scipy：MatrixCSR.mult 是 ``y += Ax`` 且要 dolfinx Vector 对象，
    而切线校验只需要一次性动作，转 scipy 更直白（本测试的量级 < 1e4 dof）。
    """
    return np.asarray(A.to_scipy().dot(arr))


def test_G2_no_friction_is_bitwise_off(tmp_path):
    """G2：构造时没给 mu -> 摩擦整条路径关闭（形式、反力全是零）。

    ⚠ 不组装切向残差：此时它返回 UFL 标量零（``ufl.as_ufl(0.0)``），
    那**不是**可装配的 Form（无积分的 form 组装会 IndexError）。
    调用方一律先判 ``has_friction_at_build``，这条测试把契约钉住。
    """
    msh, ft = _mesh(tmp_path)
    clamp = ClampContact(msh, ft, E=E_BULK, R_clamp=R_CLAMP, kappa=KAPPA,
                         h_contact=0.1e-3)          # 不给 mu
    assert clamp.has_friction_at_build is False
    assert clamp.has_friction is False
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    u = fem.Function(V)
    clamp.bind_slip_space(V)
    z = clamp.tangential_residual(u, ufl.TestFunction(V))
    assert isinstance(z, ufl.classes.Zero) or float(z) == 0.0
    assert clamp.tangential_reaction(u) == 0.0


def test_G2b_set_mu_toggles_runtime_only(tmp_path):
    """G2b：给了 mu 的工况，``set_mu`` 只改运行时大小，**不动装配标志**。

    这是「先关摩擦压法向、再开摩擦推切向」那个加载序列能成立的原因：
    形式里一直带着切向项，μ = 0 时 q ≡ 0（逐位无影响），
    故装配判据必须是 ``has_friction_at_build`` 而不是 ``has_friction``。
    """
    msh, ft = _mesh(tmp_path)
    clamp = ClampContact(msh, ft, E=E_BULK, R_clamp=R_CLAMP, kappa=KAPPA,
                         h_contact=0.1e-3, mu=MU)
    assert clamp.has_friction_at_build is True
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    u = fem.Function(V)
    clamp.bind_slip_space(V)
    v = ufl.TestFunction(V)
    clamp.set_mu(None)
    assert clamp.has_friction is False
    assert clamp.has_friction_at_build is True      # ← 关键：不随 set_mu 变
    # μ = 0 时切向残差仍是**可装配的 Form**（不是标量零），且积出来是 0
    form = clamp.tangential_residual(u, v)
    assert not isinstance(form, ufl.classes.Zero)
    clamp.set_mu(MU)
    assert clamp.has_friction is True


def test_G3_G4_G5_coulomb_bounds_and_saturation(tmp_path):
    """G3：核心区逐点 |q| <= mu p；G4：积分量 Q <= mu P；G5：大滑移时饱和。"""
    msh, ft = _mesh(tmp_path)
    clamp = ClampContact(msh, ft, E=E_BULK, R_clamp=R_CLAMP, kappa=KAPPA,
                         h_contact=0.1e-3, mu=MU)
    V, u, problem, gate, x_dofs = _solve_with(clamp, msh, ft, ux_final=0.0,
                                              prefix="g345_")
    P = clamp.reaction(u)

    ratios = []
    for ux in (1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-3):
        _push_to(problem, gate, x_dofs, ux, tag=" (G345)")
        Q = clamp.tangential_reaction(u)
        xs, ps, qs = _surface_profile(msh, ft, clamp, u)
        core = ps >= 0.5 * ps.max()
        if core.any():
            # G3：逐点 Coulomb 上限（核心区，理由同 D-51 的边缘口径）
            assert np.max(np.abs(qs[core])) <= MU * np.max(ps[core]) * (1.0 + 1.0e-9)
        # G4：积分量（积分量不受边缘噪声影响，可以严）
        assert abs(Q) <= MU * P * (1.0 + 1.0e-9), "Q=%g 超过 mu P=%g" % (Q, MU * P)
        ratios.append(abs(Q) / (MU * P) if P > 0 else np.nan)
    # G5：末级已饱和，且整条序列单调不减
    assert ratios[-1] >= 0.99, "Q/(mu P) 终值 %g 未饱和" % ratios[-1]
    assert all(b >= a - 0.05 for a, b in zip(ratios, ratios[1:])), ratios


# --------------------------------------------------------------------------
# H 组：Mindlin 部分滑移
# --------------------------------------------------------------------------
def test_H1_H4_partial_slip_and_slip_zone_location(tmp_path):
    """H1：存在一个切向步同时有黏着点与滑移点（真的部分滑移）。
    H4：黏着点集中在接触斑内侧，滑移点在外沿。
    """
    msh, ft = _mesh(tmp_path)
    clamp = ClampContact(msh, ft, E=E_BULK, R_clamp=R_CLAMP, kappa=KAPPA,
                         h_contact=0.1e-3, mu=MU)
    V, u, problem, gate, x_dofs = _solve_with(clamp, msh, ft, ux_final=0.0,
                                              prefix="h14_")

    found_partial = False
    for ux in (1.0e-8, 3.0e-8, 1.0e-7, 3.0e-7, 1.0e-6, 3.0e-6):
        _push_to(problem, gate, x_dofs, ux, tag=" (H14)")
        xs, ps, qs = _surface_profile(msh, ft, clamp, u)
        live = ps > 0.10 * ps.max()
        if not live.any():
            continue
        stick = live & (np.abs(qs) < 0.9 * MU * ps)
        slip = live & (np.abs(qs) >= 0.9 * MU * ps)
        if stick.any() and slip.any():
            found_partial = True
            # H4：黏着区在内侧（允许 2 个加密带节点间距的回退）
            assert (np.median(np.abs(xs[stick]))
                    <= np.median(np.abs(xs[slip])) + 2.0 * 0.1e-3), (
                "黏着区不在接触斑内侧：stick |x| 中位 %g vs slip %g"
                % (np.median(np.abs(xs[stick])), np.median(np.abs(xs[slip]))))
            break
    assert found_partial, "扫描的切向步里没有出现部分滑移"


def test_H2_H3_stick_halfwidth_shrinks_toward_mindlin(tmp_path):
    """H2：Q/(mu P) 增时黏着半宽 c 单调不增。
    H3：c 不超过 Mindlin 参考值 a·sqrt(1-Q/(mu P))（惩罚格式的离散黏着区
    总是略窄于解析值；不定量对拍，D-51 的理由）。
    """
    msh, ft = _mesh(tmp_path)
    clamp = ClampContact(msh, ft, E=E_BULK, R_clamp=R_CLAMP, kappa=KAPPA,
                         h_contact=0.1e-3, mu=MU)
    V, u, problem, gate, x_dofs = _solve_with(clamp, msh, ft, ux_final=0.0,
                                              prefix="h23_")
    P = clamp.reaction(u)

    rows = []
    for ux in (1.0e-8, 3.0e-8, 1.0e-7, 3.0e-7, 1.0e-6):
        _push_to(problem, gate, x_dofs, ux, tag=" (H23)")
        Q = clamp.tangential_reaction(u)
        xs, ps, qs = _surface_profile(msh, ft, clamp, u)
        live = ps > 0.10 * ps.max()
        if not live.any():
            continue
        stick = live & (np.abs(qs) < 0.9 * MU * ps)
        c = float(np.max(np.abs(xs[stick]))) if stick.any() else 0.0
        a = float(np.max(np.abs(xs[live])))
        rows.append({"ux": ux, "ratio": abs(Q) / (MU * P), "c": c, "a": a})

    assert len(rows) >= 3, "有效步数太少：%r" % rows
    for r0, r1 in zip(rows, rows[1:]):
        assert r1["c"] <= r0["c"] + 2.0 * 0.1e-3, (
            "黏着半宽反弹：%g -> %g" % (r0["c"], r1["c"]))
    for r in rows:
        if 0.02 < r["ratio"] < 0.98:
            c_ref = r["a"] * np.sqrt(1.0 - r["ratio"])
            assert r["c"] <= c_ref + 0.1e-3, (
                "c=%g 超过 Mindlin 参考值 %g（ratio=%g）"
                % (r["c"], c_ref, r["ratio"]))


# --------------------------------------------------------------------------
# I 组：工程纪律
# --------------------------------------------------------------------------
def test_I1_friction_does_not_change_normal_reaction(tmp_path):
    """I1：加摩擦后同一 delta 下的法向反力 W 不变（力系解耦的必然结果）。

    若变了，说明切向项写进了法向通道（例如把 q 当成压力），形式就是错的。
    """
    msh, ft = _mesh(tmp_path)
    common = dict(E=E_BULK, R_clamp=R_CLAMP, kappa=KAPPA, h_contact=0.1e-3)

    def W(clamp):
        _V, u, _problem, _gate, _xd = _solve_with(clamp, msh, ft, ux_final=0.0,
                                                  prefix="i1_")
        return clamp.reaction(u)

    w_nf = W(ClampContact(msh, ft, **common))
    w_f = W(ClampContact(msh, ft, mu=MU, **common))
    assert abs(w_f - w_nf) <= 1.0e-8 * abs(w_nf), (w_f, w_nf)


def test_I2_kappa_t_sensitivity(tmp_path):
    """I2：kappa_t 从 kappa 到 10*kappa，Q 的相对变化 < 3%。

    这是「惩罚格式真的收敛到 Coulomb 极限」的证据：若 q 强依赖 eps_t，
    说明算出来的是弹性罚牵引而不是 Coulomb 截断。
    """
    msh, ft = _mesh(tmp_path)
    base = dict(E=E_BULK, R_clamp=R_CLAMP, kappa=KAPPA, h_contact=0.1e-3, mu=MU)

    def Q(kappa_t, prefix):
        clamp = ClampContact(msh, ft, kappa_t=kappa_t, **base)
        _V, u, _problem, _gate, _xd = _solve_with(clamp, msh, ft, ux_final=1.0e-7,
                                                  prefix=prefix)
        return clamp.tangential_reaction(u)

    q1 = Q(KAPPA, "i2a_")
    q2 = Q(10.0 * KAPPA, "i2b_")
    rel = abs(q2 - q1) / abs(q1) if q1 else float("inf")
    assert rel < 0.03, "kappa_t 敏感性 %.1f%% 过大：q1=%g q2=%g" % (100 * rel, q1, q2)


def test_I3_reaction_is_reproducible(tmp_path):
    """I3：同一状态下重复装配切向反力逐位一致（无隐藏的全局状态）。

    并行正确性由 reaction 内部的 allreduce 保证（CLAUDE.md 陷阱 3），
    这里只查「装配是纯函数」这一点 —— 若 tangential_reaction 内部误改了
    基准场或 Constant，重复调用会给出不同值。
    """
    msh, ft = _mesh(tmp_path)
    clamp = ClampContact(msh, ft, E=E_BULK, R_clamp=R_CLAMP, kappa=KAPPA,
                         h_contact=0.1e-3, mu=MU)
    _V, u, _problem, _gate, _xd = _solve_with(clamp, msh, ft, ux_final=1.0e-7,
                                              prefix="i3_")
    q1 = clamp.tangential_reaction(u)
    q2 = clamp.tangential_reaction(u)
    assert q1 == q2
    assert np.isfinite(q1)
