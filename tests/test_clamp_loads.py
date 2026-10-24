"""6.5 T3：端部载荷（轴向张力 + 往复弯矩）的解析与 FE 校验（D-67 §3）。

测试意图：
  T3a M_a 自洽：∫ t_x(T=0, M_a, y)(y−H/2) dy == M_a
      —— D-67 §3 同时给了「M_a = σ_a H²/6」与「t_x(y) = T/H + 12M(y−H/2)/H³」
         两个式子，这条把两者串起来，防止各写各的。
  T3b 静力等效：∫t_x dy == T'、∫t_x(y−H/2) dy == M
      —— 端部施加的线性分布面力必须等价于「轴力 + 弯矩」这个力系。积分用
         Simpson（对二次核精确），独立于实现闭式地核对系数。
  T3c 远场 σ_xx 幅值 == σ_a（< 1%）：σ_a 是整个疲劳寿命的输入量，端部加载错一个
      因子寿命就错，必须在真实网格（clamp_mesh 布局 C）上坐实。
  T3d 相位反对称：phase=+1 与 −1 的位移逐点相反（纯往复、无静水偏置）。

T3c 需要 clamp_mesh 布局 C（约 145 万单元）：串行下几 GB、几十分钟，故
**只在 MPI 规模 >= 4 时运行**（``mpirun -n 6``），串行自动跳过。
"""

import math

import numpy as np
import pytest
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

from pfwire.clamp_loads import ClampLoads, bending_moment_amplitude, end_traction_x
from pfwire.material import load_material
from pfwire.mesh import BLOCK_BOTTOM, BLOCK_LEFT, block_mesh

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MATERIAL = ROOT / "configs/material_al1350.yaml"
H = 3.75e-3                 # 单丝厚度（R_wire = 1.875 mm）
SIGMA_A = 45.0e6            # 远场弯曲应力幅（Pa），S-N 扫描里的中等工况
T_PRIME = 1.0e4             # 轴向张力（N/m，单位厚度）


def _simpson(f, a: float, b: float, n: int) -> float:
    """复合 Simpson 积分（对三次以下多项式精确）。n 为偶数个区间。"""
    if n % 2:
        n += 1
    y = np.linspace(a, b, n + 1)
    fv = np.asarray(f(y), dtype=float)
    h = (b - a) / n
    return h / 3.0 * (fv[0] + fv[-1] + 4.0 * fv[1:-1:2].sum() + 2.0 * fv[2:-1:2].sum())


def test_moment_amplitude_is_consistent_with_traction_distribution():
    """T3a：M_a 与 t_x 分布的一阶矩必须互相自洽（相对 < 1e-12）。"""
    M_a = bending_moment_amplitude(SIGMA_A, H)
    # t_x 取 T=0、M=M_a，对 y 求一阶矩应得回 M_a（Simpson 对二次核精确）
    f = lambda y: end_traction_x(0.0, M_a, H, y)
    moment = _simpson(lambda y: f(y) * (y - 0.5 * H), 0.0, H, 20000)
    assert moment == pytest.approx(M_a, rel=1e-12)
    # 纯弯曲（T=0）时合力必须为零：正负半部抵消
    assert _simpson(f, 0.0, H, 20000) == pytest.approx(0.0, abs=1e-12 * abs(M_a) / H)


def test_end_traction_resultants_equal_T_and_M():
    """T3b：线性分布面力的合力 == T'、合力矩 == M（Simpson 积分独立核对）。"""
    for M in (0.0, 1.0e2, -3.7e2, bending_moment_amplitude(SIGMA_A, H)):
        f = lambda y: end_traction_x(T_PRIME, M, H, y)
        assert _simpson(f, 0.0, H, 20000) == pytest.approx(T_PRIME, rel=1e-12)
        assert _simpson(lambda y: f(y) * (y - 0.5 * H), 0.0, H, 20000) == \
            pytest.approx(M, rel=1e-12, abs=1e-12)
        # 轴力项（M=0）在截面上必须均匀：t_x == T/H 恒成立
        y = np.linspace(0.0, H, 11)
        assert np.allclose(end_traction_x(T_PRIME, 0.0, H, y), T_PRIME / H)


def test_clamp_loads_phase_switches_moment_without_recompiling(tmp_path):
    """T3 辅助：set_phase 只改 Constant，弯矩随之切换（不重编译）。"""
    msh, _ct, ft = _small_mesh(tmp_path)
    loads = ClampLoads(msh, ft, H=H, sigma_a=SIGMA_A, T=T_PRIME)
    M_a = loads.moment_amplitude
    assert M_a == pytest.approx(SIGMA_A * H**2 / 6.0, rel=1e-12)

    loads.set_phase(1.0)
    assert loads.moment == pytest.approx(M_a, rel=1e-12)
    loads.set_phase(-1.0)
    assert loads.moment == pytest.approx(-M_a, rel=1e-12)
    loads.set_phase(0.5)
    assert loads.moment == pytest.approx(0.5 * M_a, rel=1e-12)
    with pytest.raises(ValueError):
        loads.set_phase(1.5)
    # traction() 是「拿到 V 再构造」的可调用（CLAUDE.md 陷阱 1）
    form = loads.traction()(fem.functionspace(msh, ("Lagrange", 1, (2,))))
    assert isinstance(form, ufl.Form)


def test_phase_changes_an_already_compiled_form(tmp_path):
    """回归：set_phase 必须改变**已构造好的** form（D-68 的真 bug）。

    症状（修复前）：`end_traction_x` 里 `float(M)` 把 `phase * M_a` 这个 UFL
    表达式在构造时求值成 Python 字面量，于是 set_phase 改 `.value` 对已编译的
    form 毫无影响——载荷冻结、ψ⁺ 不随相位变、alpha_bar 永不累积、N 跑到几十万
    也不起裂。这条测试用**同一个 form** 在两个相位下装配，两者必须不同。
    """
    msh, _ct, ft = _small_mesh(tmp_path)
    loads = ClampLoads(msh, ft, H=H, sigma_a=SIGMA_A, T=0.0)   # 纯弯曲，只看相位
    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    form = fem.form(loads.traction()(V))          # 只构造一次
    from dolfinx.fem.petsc import assemble_vector

    loads.set_phase(0.0)
    n0 = float(np.linalg.norm(assemble_vector(form).array))
    loads.set_phase(1.0)
    n1 = float(np.linalg.norm(assemble_vector(form).array))
    assert n0 == pytest.approx(0.0, abs=1e-6), "phase=0 时载荷应为零，实得 %g" % n0
    assert n1 > 0.0, "phase=1 时载荷不应为零"
    assert n1 != pytest.approx(n0), \
        "set_phase 没有改变已编译的 form（载荷被冻结，D-68 的 bug 复现）"


def test_phase_antisymmetry_of_displacement(tmp_path):
    """T3d：纯弯曲（T=0）时 phase=+1 与 −1 的位移逐点相反。

    t_x = T'/H + 12M(y−H/2)/H³ 里 T' 是**恒定轴力**（不随相位变号），只有弯矩项
    往复。因此「相位反对称」只对弯矩分量成立：取 T=0 时载荷就是纯往复弯矩，
    位移应逐点相反；恒定的轴力偏置不是往复，不能混进这条判据（D-67 §3）。
    """
    msh, _ct, ft = _small_mesh(tmp_path)
    material = load_material(MATERIAL)
    loads = ClampLoads(msh, ft, H=H, sigma_a=SIGMA_A, T=0.0)  # 纯弯曲
    u = {s: _solve_elastic(msh, ft, material, loads, s) for s in (+1.0, -1.0)}
    scale = np.max(np.abs(u[+1.0]))
    assert scale > 0.0
    err = np.max(np.abs(u[+1.0] + u[-1.0])) / scale
    assert err < 1e-8, "纯弯曲相位 ±1 的位移不反对称，相对残差 %.3e" % err


def test_axial_tension_is_phase_independent(tmp_path):
    """T3 辅助：恒定的轴力项不随相位变号（与往复弯矩区分）。"""
    msh, _ct, ft = _small_mesh(tmp_path)
    material = load_material(MATERIAL)
    loads = ClampLoads(msh, ft, H=H, sigma_a=SIGMA_A, T=T_PRIME)
    u_p = _solve_elastic(msh, ft, material, loads, +1.0)
    u_m = _solve_elastic(msh, ft, material, loads, -1.0)
    # 轴力偏置 = 两相位位移的平均（弯矩贡献被差分消掉），应非零
    bias = 0.5 * (u_p + u_m)
    assert np.max(np.abs(bias)) > 0.0, "轴力偏置不应为零（T' 是恒定的）"
    # 弯矩往复分量 = 两相位之差的一半，应非零
    amp = 0.5 * (u_p - u_m)
    assert np.max(np.abs(amp)) > 0.0, "弯矩往复分量不应为零"


@pytest.mark.skipif(MPI.COMM_WORLD.size < 4,
                    reason="T3c 用 clamp_mesh 布局 C（约 145 万单元），需 mpirun -n 6")
def test_far_field_sigma_xx_amplitude_matches_sigma_a(tmp_path):
    """T3c：真实网格（布局 C）上，远场 σ_xx 的幅值 == σ_a（< 1%）。"""
    msh, _ct, ft = _layout_c_mesh(tmp_path)
    material = load_material(MATERIAL)
    loads = ClampLoads(msh, ft, H=H, sigma_a=SIGMA_A, T=T_PRIME)
    rows = {}
    for phase in (+1.0, -1.0):
        rows[phase] = _sigma_xx_profile(msh, ft, material, loads, phase)
    # 幅值 = 两个相反相位之差的一半（零点漂移由此消掉）；两个相位的 y 一致
    amp = 0.5 * (rows[+1.0][1] - rows[-1.0][1])
    sigma_a_fe = _bending_amplitude_from_profile(rows[+1.0][0], amp)
    rel = abs(sigma_a_fe - SIGMA_A) / SIGMA_A
    MPI.COMM_WORLD.barrier()
    assert rel < 0.01, "远场 σ_xx 幅值 %.4g 与 σ_a = %.4g 差 %.2f%%" % (
        sigma_a_fe, SIGMA_A, 100 * rel)


# --------------------------------------------------------------------------
# 辅助：网格、弹性解、σ_xx 剖面
# --------------------------------------------------------------------------
def _small_mesh(tmp_path):
    """T3d 用的小网格（快）：40 × 3.75 mm，粗网格即可验证反对称。"""
    return block_mesh(tmp_path / "small", width=40.0e-3, depth=H,
                      refine_boxes=[(-1.0e-3, 1.0e-3, -0.5e-3, 0.0)],
                      h_fine=0.5e-3, h_coarse=2.0e-3, transition=1.0e-3,
                      comm=MPI.COMM_WORLD)


def _layout_c_mesh(tmp_path):
    """T3c 用的真实网格：clamp_mesh 布局 C（D-67 §3，用户 2026-10-14 指定）。"""
    a_contact, w, d = 2.917e-3, 0.3e-3, 0.3e-3
    boxes = [(a_contact - w, a_contact + w, -d, 0.0),
             (-a_contact - w, -a_contact + w, -d, 0.0)]
    return block_mesh(tmp_path / "layoutC", width=40.0e-3, depth=H, refine_boxes=boxes,
                      h_fine=1.1111e-6, h_coarse=750.0e-6, transition=1.0e-3,
                      comm=MPI.COMM_WORLD)


def _solve_elastic(msh, ft, material, loads, phase) -> np.ndarray:
    """线弹性解一次，返回位移向量（左端 u_x = 0，底边中点 u_y = 0，右端加载）。"""
    loads.set_phase(phase)
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    u = fem.Function(V)
    du = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    mu, lmbda = material.elastic.lame
    ident = ufl.Identity(msh.geometry.dim)

    def stress(w):
        eps = ufl.sym(ufl.grad(w))
        return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ident

    a = ufl.inner(stress(du), ufl.sym(ufl.grad(v))) * ufl.dx
    L = loads.traction()(V)
    bcs = _end_bcs(V, msh, ft)
    problem = LinearProblem(a, L, bcs=bcs, u=u, petsc_options_prefix="t3_elastic_",
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                           "pc_factor_mat_solver_type": "mumps"})
    problem.solve()
    if problem.solver.getConvergedReason() <= 0:
        raise RuntimeError("T3 线弹性求解未收敛：KSP reason=%d"
                           % problem.solver.getConvergedReason())
    return u.x.array.copy()


def _end_bcs(V, msh, ft):
    """左端 u_x = 0（只压 x 分量）+ 底边中点 u_y = 0（D-67 §3）。

    左端只压 x：完全钉住端面会把弯曲挡在外面（端部无法转动/错动），测不到 σ_a；
    只压 x 即消掉轴向刚体平移。竖直与转动由底边中点的单点 u_y = 0 约束掉。
    写法照 fretting.py 的既有约定：值给标量、空间给 V.sub(i)、dofs 用
    ``locate_dofs_topological(V.sub(i), ...)``。
    """
    fdim = msh.topology.dim - 1
    Vx, Vy = V.sub(0), V.sub(1)
    bcs = [fem.dirichletbc(default_scalar_type(0.0),
                           fem.locate_dofs_topological(Vx, fdim, ft.find(BLOCK_LEFT)),
                           Vx)]
    # 底边中点：单个节点、只压 y。
    # ⚠ **必须在独立的标量空间上选点**，不能用 V.tabulate_dof_coordinates()
    # （D-68 问题 33 续 9）。原注释写「坐标从父空间 V 取（子空间不能 tabulate）」
    # —— 那个前提是错的：`locate_dofs_topological(Vy, ...)` 返回的是 Vy 子空间的
    # dof 编号，而 V 的坐标行号按 block 交错（Vx 走偶数行、Vy 走奇数行），
    # 两者不是同一个编号空间。实测同一批 41 个底边 dof：
    #     coords_V[cand, 1] ∈ [-0.003404, -0.000148]  ← **0/41 落在底边上**
    # 即取到的全是错的坐标。CG1 的 dof 与节点一一对应，故在同阶标量空间上
    # locate 再取节点编号，该编号可直接用于 Vy。
    Vs = fem.functionspace(msh, ("Lagrange", 1))
    dofs_s = fem.locate_dofs_topological(Vs, fdim, ft.find(BLOCK_BOTTOM))
    coords_s = Vs.tabulate_dof_coordinates()
    nloc = Vs.dofmap.index_map.size_local
    cand = dofs_s[dofs_s < nloc]
    if cand.size:
        xs = np.abs(coords_s[cand, 0])
        i = int(np.argmin(xs))
        x_best, dof_best = float(xs[i]), int(cand[i])
    else:
        x_best, dof_best = math.inf, -1
    comm = msh.comm
    owner = min(comm.allgather((x_best, dof_best, comm.rank)))
    if owner[2] == comm.rank and owner[1] >= 0:
        bcs.append(fem.dirichletbc(
            default_scalar_type(0.0),
            np.array([owner[1]], dtype=np.int32), Vy))
    return bcs


def test_bottom_midpoint_bc_really_sits_on_the_bottom_edge(tmp_path):
    """守卫：free 构型的单点 u_y 约束必须真的落在**底边**上（D-68 问题 33 续 9）。

    原实现用 ``V.tabulate_dof_coordinates()[cand]`` 取坐标，而 ``cand`` 是
    ``Vy = V.sub(1)`` 子空间的 dof 编号 —— 混合空间 V 的坐标行按 block 交错
    （Vx 走偶数行、Vy 走奇数行），两个编号空间不同。实测同一批 41 个底边 dof：

        coords_V[cand, 1] ∈ [-0.003404, -0.000148]   ← 0/41 落在底边（y = -H）

    于是 argmin 选出一个**网格内部**的任意点，BC 照建、不报错，只是约束在错位置。
    后果：既没提供竖向反力，又制造应力奇异 —— 无摩擦基准的 max_d 从 0 跳到 1.0
    而 depth_max 恒为 0（假损伤），且只在 mpirun -n 6 下复现、串行不复现
    （分区不同 → 选出的错点不同）。

    本用例直接核对被约束 dof 的坐标，**不依赖编号映射**：
    把该 dof 的值置 1、其余 0，插值出这个「基函数」在底边处的值 ——
    若 dof 真在底边，则底边上的值应≈1。
    """
    import numpy as np
    import ufl
    from dolfinx import default_scalar_type, fem

    from pfwire.mesh import BLOCK_BOTTOM

    msh, _ct, ft = _small_mesh(tmp_path)
    V = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    bcs = _end_bcs(V, msh, ft)

    assert len(bcs) == 2, "free 构型应有 2 个 BC（左端 u_x + 底边单点 u_y）"
    # 第二个 BC 是底边单点
    parts = bcs[1].dof_indices()
    idx = np.asarray(parts[0])
    assert idx.size == 1, "底边 BC 应是单点约束，收到 %d 个 dof" % idx.size

    # 该 dof 的坐标：Vy 子空间的编号，用独立标量空间取（同一网格同阶，
    # CG1 的 dof 与节点一一对应，编号可直接搬）
    Vs = fem.functionspace(msh, ("Lagrange", 1))
    coords = Vs.tabulate_dof_coordinates()
    node = int(idx[0])
    x_dof, y_dof = float(coords[node, 0]), float(coords[node, 1])

    H = 3.75e-3      # 块体深度（与 _mesh 一致）
    if MPI.COMM_WORLD.rank == 0:
        print("  底边单点 dof = %d，坐标 = (%.6f, %.6f)" % (node, x_dof, y_dof))

    # 判据 1：必须在底边上（y = -H）
    assert abs(y_dof - (-H)) < 1.0e-9, (
        "被约束的 dof 不在底边上：y = %.9f，应为 %.9f。"
        "这是 D-68 问题 33 续 9 的编号空间错配 bug（Vy 编号 vs V 坐标行）"
        % (y_dof, -H))

    # 判据 2：必须接近 x = 0（底边中点；网格对称时 |x| 最小 = 0）
    x_bottom = np.abs(coords[:, 0])
    # 只统计底边节点
    fdim = msh.topology.dim - 1
    d_bottom = fem.locate_dofs_topological(Vs, fdim, ft.find(BLOCK_BOTTOM))
    nb = Vs.dofmap.index_map.size_local
    x_min_bottom = float(np.min(np.abs(coords[d_bottom[d_bottom < nb], 0])))
    assert abs(abs(x_dof) - x_min_bottom) < 1.0e-12, (
        "被约束的 dof 不是 |x| 最小的底边节点：|x| = %.9f，应为 %.9f"
        % (abs(x_dof), x_min_bottom))

#: T3c 的远场截面 x 坐标。取 x = 0：布局 C 的两条加密带在 x = ±a_contact
#: (≈ ±2.92 mm)，40 mm 宽的单丝上 x = 0 距两端 20 mm（5.3 H）、距加密带 2.9 mm，
#: 既避开端部圣维南区也避开接触区。
FAR_FIELD_X = 0.0
#: 截面上取样的容差：|x − FAR_FIELD_X| <= X_TOL 的单元都算在该截面上。
X_TOL = 100.0e-6


def _sigma_xx_profile(msh, ft, material, loads, phase) -> tuple[np.ndarray, np.ndarray]:
    """解一次，在远场截面上取 (y, σ_xx) 剖面（按 y 排序）。"""
    u_arr = _solve_elastic(msh, ft, material, loads, phase)
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    u = fem.Function(V)
    u.x.array[:] = u_arr
    u.x.scatter_forward()

    # σ_xx 投到 DG0（单元常数），再按单元中心坐标挑截面
    V0 = fem.functionspace(msh, ("DG", 0))
    mu, lmbda = material.elastic.lame
    ident = ufl.Identity(msh.geometry.dim)
    eps = ufl.sym(ufl.grad(u))
    sxx = (2.0 * mu * eps + lmbda * ufl.tr(eps) * ident)[0, 0]
    s_fn = fem.Function(V0)
    s_fn.interpolate(fem.Expression(sxx, V0.element.interpolation_points))
    from dolfinx.mesh import compute_midpoints
    nloc = msh.topology.index_map(msh.topology.dim).size_local
    mid = compute_midpoints(msh, msh.topology.dim, np.arange(nloc, dtype=np.int32))
    on = np.abs(mid[:, 0] - FAR_FIELD_X) <= X_TOL
    ys = np.concatenate(msh.comm.allgather(mid[on, 1]))
    ss = np.concatenate(msh.comm.allgather(s_fn.x.array[:nloc][on]))
    o = np.argsort(ys)
    return ys[o], ss[o]


def _bending_amplitude_from_profile(y, sigma_xx) -> float:
    """由 σ_xx(y) 剖面反解弯曲应力幅：线性拟合斜率 × (H/2)。

    用**拟合斜率**而不是表面点取值（用户 2026-10-14 确认）：避开表面节点的插值
    误差与两端圣维南区的残余。纯弯曲下 σ_xx 对 y 是线性的，斜率 k 满足
    σ_a = k·(H/2)；y 用「距底面的高度」代入，符号不影响幅值。
    """
    if y.size < 2:
        return 0.0
    k = np.polyfit(y, sigma_xx, 1)[0]      # dσ_xx/dy
    return abs(k) * 0.5 * H
