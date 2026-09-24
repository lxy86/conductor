"""Phase 4.3：FE 摩擦接触解与 Hertz 解析解对拍（PLAN §4 Phase 4.3）。

**为什么自己写而不用上游的 demo**：asimov-contact 的 `ContactWriter` 在
dolfinx 0.11 / UFL 2026.1 下已失效（`create_submesh` 改返回 `EntityMap`，
且它还在用已删除的 legacy `ufl.FiniteElement`），见 DECISIONS.md D-48。
问题装配沿用上游算例的写法（那是该库的标准用法），**接触压力的提取与
对拍是本脚本自己做的**。

**压力怎么取**：下方块体的上表面是平的、外法向为 (0, 1)，故
    p(x) = -sigma_nn = -sigma_yy
不需要碰 FacetNormal，也不需要建 submesh —— 正好绕开 D-48 的两个坑。
把 sigma_yy 插值到 CG1（只在块体单元上），再取上表面那圈自由度的
坐标与数值即可。

对拍口径（PLAN Phase 4.3）：接触半宽 a、峰值压力 p0 相对解析解偏差 < 3%；
另加一条更严的**合力校核** ∫p dx 与施加的 P 之差 < 2%（积分量对噪声不敏感，
能查出"形状对了但量级错了"这种 p0/a 侥幸通过的情形）。

用法：
    python scripts/verify_contact_hertz.py [configs/verify_hertz_contact.yaml]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import ufl
import yaml
from dolfinx import default_scalar_type
from dolfinx.fem import (
    Constant,
    assemble_scalar,
    Expression,
    Function,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem import extract_function_spaces
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_vector,
    set_bc,
)
from dolfinx.geometry import bb_tree, compute_closest_entity
from dolfinx.graph import adjacencylist
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py.PETSc import InsertMode, ScatterMode  # type: ignore

from dolfinx_contact.cpp import ContactMode
from dolfinx_contact.general_contact.contact_problem import ContactProblem, FrictionLaw
from dolfinx_contact.helpers import (
    epsilon,
    lame_parameters,
    rigid_motions_nullspace_subdomains,
    sigma_func,
)
from dolfinx_contact.meshing import convert_mesh, create_halfdisk_plane_mesh
from dolfinx_contact.newton_solver import NewtonSolver

from pfwire.loads import hertz_half_width, hertz_peak_pressure
from pfwire.material import contact_effective_modulus

REPO = pathlib.Path(__file__).resolve().parents[1]

#: 上游 `create_halfdisk_plane_mesh` 写死的物理组编号（见该函数源码）。
#: 放在这里是为了让"魔数"至少有出处；上游一改就会在断言里炸出来。
TAG_DISK_CONTACT = 7      # 半圆柱的弧面（接触面 1）
TAG_BLOCK_CONTACT = 6     # 块体上表面（接触面 2）
TAG_DIRICHLET = 4         # 块体底面：全固定
TAG_NEUMANN = 8           # 半圆柱顶面：施加面力
CELL_DISK = 1
CELL_BLOCK = 2


def _f(node, *keys):
    """取配置值并强制成 float —— PyYAML 走 YAML 1.1，`1e-7` 会是字符串。"""
    for k in keys[:-1]:
        node = node[k]
    return float(node[keys[-1]])


def load_config(path: pathlib.Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("配置根节点必须是映射：%s" % path)
    return cfg


def build_mesh(cfg: dict, workdir: pathlib.Path):
    """生成半圆柱–块体网格并读回 dolfinx。"""
    g, m = cfg["geometry"], cfg["mesh"]
    workdir.mkdir(parents=True, exist_ok=True)
    stem = str(workdir / "halfdisk_plane")
    if MPI.COMM_WORLD.rank == 0:
        create_halfdisk_plane_mesh(
            filename=f"{stem}.msh",
            res=_f(m, "res"),
            order=int(m["order"]),
            quads=bool(m["quads"]),
            r=_f(g, "R"),
            height=_f(g, "H"),
            length=_f(g, "L"),
            gap=_f(g, "gap"),
        )
        convert_mesh(stem, f"{stem}.xdmf", gdim=2)
    MPI.COMM_WORLD.barrier()
    with XDMFFile(MPI.COMM_WORLD, f"{stem}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()
        domain_marker = xdmf.read_meshtags(mesh, name="cell_marker")
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")
    return mesh, domain_marker, facet_marker


def _closest_node(mesh, point):
    tree = bb_tree(mesh, 0)
    return compute_closest_entity(tree, tree, mesh, np.asarray(point, dtype=np.float64))


def solve_contact(cfg: dict, mesh, domain_marker, facet_marker):
    """按上游算例的写法装配并求解；返回位移 u 与材料参数。"""
    g, mat, ld, sol = cfg["geometry"], cfg["material"], cfg["loading"], cfg["solver"]
    R, H = _f(g, "R"), _f(g, "H")
    E, nu = _f(mat, "E"), _f(mat, "nu")
    P_total, steps = _f(ld, "P"), int(ld["steps"])
    order = int(cfg["mesh"]["order"])
    tdim = mesh.topology.dim

    V = functionspace(mesh, ("CG", order, (mesh.geometry.dim,)))

    # 块体底面全固定；另外在圆柱内部钉两个节点的 x 位移，消掉水平刚体位移
    node1 = _closest_node(mesh, [0.0, -R / 2.5, 0.0])
    node2 = _closest_node(mesh, [0.0, -R / 5.0, 0.0])
    dofs_x = locate_dofs_topological(V.sub(0), 0, np.hstack([node1, node2]))
    dofs_fix = locate_dofs_topological(V, tdim - 1, facet_marker.find(TAG_DIRICHLET))
    g0 = Constant(mesh, default_scalar_type(0.0))
    gv = Constant(mesh, default_scalar_type((0.0, 0.0)))
    bcs = [dirichletbc(g0, dofs_x, V.sub(0)), dirichletbc(gv, dofs_fix, V)]

    # 面力：把总法向力 P 摊在圆柱顶面宽度 2R 上
    traction = Constant(mesh, default_scalar_type((0.0, 0.0)))

    mu_f, lmbda_f = lame_parameters(True)          # True = 平面应变
    V0 = functionspace(mesh, ("DG", 0))
    mu_c, lmbda_c = Function(V0), Function(V0)
    for cells in (domain_marker.find(CELL_DISK), domain_marker.find(CELL_BLOCK)):
        mu_c.interpolate(lambda x: np.full((1, x.shape[1]), mu_f(E, nu)), cells)
        lmbda_c.interpolate(lambda x: np.full((1, x.shape[1]), lmbda_f(E, nu)), cells)
    sigma = sigma_func(mu_c, lmbda_c)

    u, du = Function(V), Function(V)
    v, w = ufl.TestFunction(V), ufl.TrialFunction(V)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)

    F = ufl.inner(sigma(u), epsilon(v)) * dx - ufl.inner(traction, v) * ds(TAG_NEUMANN)
    F = ufl.replace(F, {u: u + du})
    J = ufl.derivative(F, du, w)
    F_c, J_c = form(F), form(J)

    surfaces = adjacencylist(
        np.array([TAG_DISK_CONTACT, TAG_BLOCK_CONTACT], dtype=np.int32),
        np.array([0, 2], dtype=np.int32),
    )
    pairs = [(0, 1), (1, 0)]
    problem = ContactProblem(
        [facet_marker], surfaces, pairs, mesh, int(cfg["solver"].get("quadrature_degree", 5)),
        [ContactMode.ClosestPoint, ContactMode.ClosestPoint],
    )
    gamma = E * _f(sol, "nitsche_gamma_factor") * order**2
    problem.generate_contact_data(
        FrictionLaw.Frictionless, V,
        {"u": u, "du": du, "mu": mu_c, "lambda": lmbda_c},
        gamma, _f(sol, "nitsche_theta"),
    )

    def coefficients(x, coeffs):
        du.x.scatter_forward()
        problem.update_contact_data(du)

    def residual(x, b, coeffs):
        b.zeroEntries()
        b.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        problem.assemble_vector(b, V)
        assemble_vector(b, F_c)
        apply_lifting(b, [J_c], bcs=[bcs], x0=[x], alpha=-1.0)
        b.ghostUpdate(addv=InsertMode.ADD, mode=ScatterMode.REVERSE)
        set_bc(b, bcs, x, -1.0)

    def jacobian(x, A, coeffs):
        A.zeroEntries()
        problem.assemble_matrix(A, V)
        assemble_matrix(A, J_c, bcs=bcs)
        A.assemble()

    A = problem.create_matrix(J_c)
    b = create_vector(extract_function_spaces(F_c))
    newton = NewtonSolver(mesh.comm, A, b, problem.coeffs)
    newton.set_residual(residual)
    newton.set_jacobian(jacobian)
    newton.set_coefficients(coefficients)
    newton.A.setNearNullSpace(
        rigid_motions_nullspace_subdomains(V, domain_marker, np.unique(domain_marker.values)))
    newton.set_newton_options({
        "relaxation_parameter": 1.0,
        "atol": _f(sol, "newton_tol"), "rtol": _f(sol, "newton_tol"),
        "convergence_criterion": "residual",
        "max_it": int(sol["max_it"]), "error_on_nonconvergence": True,
    })
    ksp_tol = _f(sol, "ksp_tol")
    newton.set_krylov_options({"ksp_type": "preonly", "pc_type": "lu",
                               "ksp_rtol": ksp_tol, "ksp_atol": ksp_tol})

    # 初值：让圆柱先"落下"一点，否则第一步接触搜索找不到对面
    du.interpolate(lambda x: np.vstack([np.zeros(x.shape[1]),
                                        np.full(x.shape[1], -H / 100.0 - _f(g, "gap"))]),
                   domain_marker.find(CELL_DISK))

    for i in range(steps):
        traction.value[-1] = -P_total * (i + 1) / (steps * 2.0 * R)
        n_it, converged = newton.solve(du, write_solution=True)
        if not converged:                     # 硬约束 6：不收敛必须报出来，不能静默继续
            raise RuntimeError("第 %d/%d 载荷步 Newton 未收敛" % (i + 1, steps))
        du.x.scatter_forward()
        u.x.array[:] += du.x.array[:]
        if mesh.comm.rank == 0:
            print("  载荷步 %d/%d：Newton %d 次迭代收敛" % (i + 1, steps, n_it), flush=True)
        if i < steps - 1:
            problem.update_contact_detection(u)
            A = problem.create_matrix(J_c)
            A.setNearNullSpace(rigid_motions_nullspace_subdomains(
                V, domain_marker, np.unique(domain_marker.values)))
            newton.set_petsc_matrix(A)
            du.x.array[:] = 0.1 * du.x.array[:]
            problem.update_contact_data(du)
    return u, sigma, V


def contact_resultant(mesh, facet_marker, u, sigma) -> float:
    """接触面上的法向合力，用**精确面积分**而不是对节点值做梯形。

    平衡要求它等于施加的总法向力 P，所以这是比 p0/a 更硬的校核。
    第一版用 np.trapezoid 对 CG1 节点值积分，偏高 3.1% —— 多出来的部分
    是接触区外的数值残尾被算了进去（CG1 插值把 DG 应力抹平到了边缘之外），
    是**测量方法的误差，不是解的误差**。换成 assemble_scalar 就没有这个问题。
    """
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
    local = assemble_scalar(form(-sigma(u)[1, 1] * ds(TAG_BLOCK_CONTACT)))
    return float(mesh.comm.allreduce(local, op=MPI.SUM))


def extract_pressure(mesh, domain_marker, facet_marker, u, sigma):
    """取块体上表面的接触压力 p(x) = -sigma_yy。

    上表面是平面、外法向 (0,1)，故法向应力就是 sigma_yy，不必用 FacetNormal，
    也不必建 submesh（D-48：submesh 那条路在 0.11 下有 EntityMap 的坑）。
    """
    Vs = functionspace(mesh, ("CG", 1))
    syy = Function(Vs)
    syy.interpolate(Expression(sigma(u)[1, 1], Vs.element.interpolation_points),
                    cells0=domain_marker.find(CELL_BLOCK))
    dofs = locate_dofs_topological(Vs, mesh.topology.dim - 1,
                                   facet_marker.find(TAG_BLOCK_CONTACT))
    coords = Vs.tabulate_dof_coordinates()[dofs]
    p = -syy.x.array[dofs]
    n_owned = Vs.dofmap.index_map.size_local
    keep = dofs < n_owned                       # 并行下只留本进程拥有的自由度
    xs = np.concatenate(mesh.comm.allgather(coords[keep, 0]))
    ps = np.concatenate(mesh.comm.allgather(p[keep]))
    order = np.argsort(xs)
    return xs[order], ps[order]


def compare(cfg: dict, xs, ps, load_exact: float) -> dict:
    """与 Hertz 解析解对拍。"""
    g, mat, ld = cfg["geometry"], cfg["material"], cfg["loading"]
    R, E, nu, P = _f(g, "R"), _f(mat, "E"), _f(mat, "nu"), _f(ld, "P")
    e_star = contact_effective_modulus(E, nu, E, nu)
    a_ref = hertz_half_width(P=P, R=R, E_star=e_star)
    p0_ref = hertz_peak_pressure(P=P, a=a_ref)

    p0_fe = float(ps.max())
    # 接触半宽：压力降到峰值 1% 以下即视为脱开。用阈值而不是"首个零"，
    # 因为 CG1 插值会在边缘留下小的数值残余。
    inside = ps > 0.01 * p0_fe
    a_fe = float(np.max(np.abs(xs[inside]))) if inside.any() else 0.0
    load_trapz = float(np.trapezoid(np.clip(ps, 0.0, None), xs))   # 仅作诊断对照

    return {
        "E_star": e_star,
        "a_ref": a_ref, "a_fe": a_fe, "a_rel": abs(a_fe - a_ref) / a_ref,
        "p0_ref": p0_ref, "p0_fe": p0_fe, "p0_rel": abs(p0_fe - p0_ref) / p0_ref,
        "P_ref": P, "P_fe": load_exact, "load_rel": abs(load_exact - P) / P,
        "P_trapz": load_trapz, "load_trapz_rel": abs(load_trapz - P) / P,
        "n_points_in_contact": int(inside.sum()),
        "a_over_R": a_ref / R,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 4.3 Hertz 接触对拍")
    ap.add_argument("config", nargs="?", default="configs/verify_hertz_contact.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    comm = MPI.COMM_WORLD
    outdir = REPO / cfg["output"]["dir"]

    def say(msg):
        if comm.rank == 0:
            print(msg, flush=True)

    say("Phase 4.3：FE 摩擦接触 vs Hertz 解析解")
    mesh, dm, fm = build_mesh(cfg, outdir)
    say("  网格：%d 单元" % mesh.topology.index_map(mesh.topology.dim).size_global)
    u, sigma, _ = solve_contact(cfg, mesh, dm, fm)
    xs, ps = extract_pressure(mesh, dm, fm, u, sigma)
    load_exact = contact_resultant(mesh, fm, u, sigma)
    r = compare(cfg, xs, ps, load_exact)

    tol = cfg["tolerance"]
    checks = [("接触半宽 a", r["a_fe"], r["a_ref"], r["a_rel"], _f(tol, "a_rel")),
              ("峰值压力 p0", r["p0_fe"], r["p0_ref"], r["p0_rel"], _f(tol, "p0_rel")),
              ("合力（面积分）", r["P_fe"], r["P_ref"], r["load_rel"], _f(tol, "load_rel"))]
    say("")
    say("  E* = %.6g，a/R = %.4f（Hertz 要求接触区远小于曲率半径）" % (r["E_star"], r["a_over_R"]))
    say("  接触区内取到 %d 个采样点" % r["n_points_in_contact"])
    say("  （对照：对 CG1 节点值做梯形积分得 %.6g，偏差 %.2f%% —— 含接触区外的数值残尾，仅供诊断）"
        % (r["P_trapz"], 100 * r["load_trapz_rel"]))
    say("")
    say("  %-14s %12s %12s %10s %10s %s" % ("量", "FE", "解析", "相对差", "阈值", "判定"))
    ok = True
    for name, fe, ref, rel, lim in checks:
        passed = rel < lim
        ok = ok and passed
        say("  %-14s %12.6g %12.6g %9.2f%% %9.1f%% %s"
            % (name, fe, ref, 100 * rel, 100 * lim, "通过" if passed else "**未通过**"))

    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "result.json").write_text(
            json.dumps({"passed": ok, **r}, ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez_compressed(outdir / "pressure.npz", x=xs, p=ps)
    say("")
    say("Phase 4.3 %s" % ("通过" if ok else "未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
