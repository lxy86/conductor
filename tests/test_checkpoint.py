"""断点续跑（硬约束 5）的单测。

最要紧的一条是 `test_resume_matches_uninterrupted_run`：续跑必须与**没中断过**
的那一次逐位一致。只测「能存能读、不抛异常」是不够的 —— 本模块要防的正是
「读到了、但值是错的、还不报错」那种失败。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pfwire import checkpoint as ck
from pfwire import cyclejump as cj
from pfwire import phasefield as pf
from tests.test_phasefield import UniaxialStrainCase


def tmp_shared(tmp_path):
    """pytest 的 tmp_path 每个 rank 各不相同；并行下必须统一，否则集合写入死锁。"""
    return MPI.COMM_WORLD.bcast(tmp_path, root=0)


#: 让损伤在约 2 个循环后就起来：alpha_T 由闭式 N = a_T (2/s - 1)/psi_max 反解
#: （与 test_cyclejump 同一个关系式）。d 必须真的长起来，否则存的是平凡状态、
#: 「续跑一致」就成了「两边都是 0」这种没有信息量的相等。
_S = 0.8
_N_TARGET = 2.0


def alpha_T_for(case) -> float:
    sigma_c = case.material.fracture.sigma_c
    psi_max = 0.5 * (_S * sigma_c) ** 2 / case.material.elastic.E
    return _N_TARGET * psi_max / (2.0 / _S - 1.0)


def make_case(mesh=None):
    case = UniaxialStrainCase(mesh=mesh)
    state = cj.FatigueState(
        case.msh, case.solver.model, alpha_source="undegraded",
        alpha_T=alpha_T_for(case))
    # 必须把 f(alpha_bar) 接回求解器再重建一次 —— 只建 FatigueState 而不接进去，
    # 疲劳退化根本不进方程，d 会一直是 0（本测试初版就是这么写错的）。
    case.solver = pf.PhaseFieldSolver(
        case.msh,
        case.solver.model,
        bcs_u=case.make_bcs,
        f_alpha=state.f_alpha,
        config=case.solver.config,
    )
    state.bind_displacement(case.solver.u)
    return case, state


def advance(case, state, cycles, s=_S):
    """跑若干个完整循环（峰值 -> 卸载到 0），制造一个非平凡的状态。

    **卸载那半程不能省**：accumulate() 只在 psi+ 上升时累积（Carrara 式 35），
    一直把 u 压在峰值的话 psi+ 不再上升，alpha_bar 会卡在 1×psi_max 永远不动，
    d 自然一直是 0 —— 本测试初版就是这么写的，白跑了两轮。
    """
    for _ in range(cycles):
        case.u_applied.value = s * case.strain_c * case.L
        case.solver.solve_displacement()
        state.accumulate()
        case.solver.solve_damage()
        case.solver.commit_step()
        case.u_applied.value = 0.0
        case.solver.solve_displacement()
        state.accumulate()


def gather_max(fn):
    n = fn.function_space.dofmap.index_map.size_local
    return MPI.COMM_WORLD.allreduce(float(np.max(fn.x.array[:n])), op=MPI.MAX)


def test_has_checkpoint_is_false_on_empty_dir(tmp_path):
    assert not ck.has_checkpoint(tmp_shared(tmp_path))


def test_resume_on_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="没有可用的断点"):
        ck.resume(tmp_shared(tmp_path))


def test_round_trip_restores_fields_and_meta(tmp_path):
    d = tmp_shared(tmp_path)
    case, state = make_case()
    advance(case, state, 3)
    ref = {n: gather_max(f) for n, f in (
        ("d", case.solver.d), ("d_old", case.solver.d_old),
        ("alpha_bar", state.alpha_bar), ("psi_plus_prev", state.psi_plus_prev))}
    assert ref["alpha_bar"] > 0.0 and ref["d"] > 0.0, "构造前提：状态非平凡"

    ck.save(d, case.solver, state, {"N": 12.5, "n_jumps": 3})
    assert ck.has_checkpoint(d)

    r = ck.resume(d)
    assert r.meta == {"N": 12.5, "n_jumps": 3}
    case2, state2 = make_case(mesh=r.mesh)
    r.read_fields(case2.solver, state2)
    for name, fn in (("d", case2.solver.d), ("d_old", case2.solver.d_old),
                     ("alpha_bar", state2.alpha_bar),
                     ("psi_plus_prev", state2.psi_plus_prev)):
        assert gather_max(fn) == pytest.approx(ref[name], rel=1e-12), name


def test_reading_onto_a_rebuilt_mesh_is_refused(tmp_path):
    """**核心防呆**：网格重建后读函数会静默读出乱序的值，必须拒绝而不是照读。

    实测（data/_ckpt_probe.py，12x9 矩形 P1 解析场）：重建网格读回的逐点误差
    1.995，与场本身同量级，且 read_function 不抛任何异常。
    """
    d = tmp_shared(tmp_path)
    case, state = make_case()
    advance(case, state, 2)
    ck.save(d, case.solver, state, {"N": 2.0})

    r = ck.resume(d)
    rebuilt, state_rebuilt = make_case()          # 另起一张结构相同的网格
    assert rebuilt.msh is not r.mesh
    with pytest.raises(ValueError, match="不是建在 checkpoint 读出来的那张网格"):
        r.read_fields(rebuilt.solver, state_rebuilt)


def test_resume_matches_uninterrupted_run(tmp_path):
    """续跑必须与没中断过的那一次**逐位一致**。这条是断点续跑的验收判据。"""
    d = tmp_shared(tmp_path)

    straight, s_state = make_case()
    advance(straight, s_state, 6)
    want_d = gather_max(straight.solver.d)
    want_a = gather_max(s_state.alpha_bar)
    assert want_d > 0.0, "构造前提：直跑那次必须真的长出损伤"

    first, f_state = make_case()
    advance(first, f_state, 3)
    ck.save(d, first.solver, f_state, {"done": 3})

    r = ck.resume(d)
    second, sec_state = make_case(mesh=r.mesh)
    r.read_fields(second.solver, sec_state)
    assert r.meta["done"] == 3
    advance(second, sec_state, 3)                  # 补上剩下的 3 个

    assert gather_max(second.solver.d) == pytest.approx(want_d, rel=1e-12)
    assert gather_max(sec_state.alpha_bar) == pytest.approx(want_a, rel=1e-12)


def test_write_alternates_slots_so_a_killed_write_keeps_the_old_one(tmp_path):
    """两槽轮换：第二次写进另一个槽，指针最后才翻过去。

    写到一半被杀（本轮就 kill -9 过好几次）只该毁掉空闲槽。
    """
    d = tmp_shared(tmp_path)
    case, state = make_case()
    advance(case, state, 1)

    p1 = ck.save(d, case.solver, state, {"k": 1})
    p2 = ck.save(d, case.solver, state, {"k": 2})
    assert p1 != p2, "第二次写必须换槽，不能原地覆盖"
    assert {p1.name, p2.name} == set(ck.SLOTS)
    if MPI.COMM_WORLD.rank == 0:
        ptr = json.loads((d / ck.POINTER).read_text(encoding="utf-8"))
        assert ptr["meta"]["k"] == 2
        assert (d / ck.SLOTS[ptr["slot"]]) == p2
        assert p1.is_dir(), "旧槽要留着，它是写坏时的退路"


def test_facet_tags_survive_the_round_trip(tmp_path):
    """面标签也要存：从 .bp 读回的网格不带物理组，不存就续不了跑。

    微动工况的接触面力与边界条件全按 facet_tags 施加 —— 丢了它，
    续跑出来的是一个**没有外载**的算例，而且不会报错。
    """
    d = tmp_shared(tmp_path)
    case, state = make_case()
    advance(case, state, 1)

    msh = case.msh
    fdim = msh.topology.dim - 1
    top = dmesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], case.L))
    tags = dmesh.meshtags(
        msh, fdim, np.sort(top), np.full(top.size, 7, dtype=np.int32))
    n_tagged = MPI.COMM_WORLD.allreduce(int(top.size), op=MPI.SUM)
    assert n_tagged > 0

    ck.save(d, case.solver, state, {"N": 1.0}, facet_tags=tags)
    r = ck.resume(d)
    back = r.read_facet_tags()
    assert back.dim == fdim
    # ⚠ 必须用**全局**量（CLAUDE.md 陷阱 3）：`back.values` 是本 rank 局部的，
    # 被标记的 facet 可以整个落在别的 rank 上，本 rank 读到 0 个 ——
    # 实测 6 rank 下分布是 (0,2,0,0,2,0)，rank 0/2/3/5 读到空，
    # `np.unique(back.values)` 在它们上面得空集，`== {7}` 直接挂。
    # 这是断言口径 bug，不是 facet_tags 丢了（串行能过纯属 size==1 无分区）。
    all_vals = np.concatenate(MPI.COMM_WORLD.allgather(np.asarray(back.values)))
    assert set(np.unique(all_vals).tolist()) == {7}
    # 且数量必须与写入时一致（adios4dolfinx 并行下会把 rank 交界的 ghost facet
    # 也读回来，实测轻量算例 4→5、真实配置 TOP 820→826）。
    # 已定量证明那些 ghost **不被 ds(tag) 计入**（D-68 问题 31 续：
    # 两套 tag 的 ∫1ds 与 ∫x ds 相对差全部精确 0.000e+00），故这里只要求
    # 「不少于」而不是「等于」，并要求 tag 值集合正确。
    n_back = MPI.COMM_WORLD.allreduce(int(back.indices.size), op=MPI.SUM)
    assert n_back >= n_tagged, (
        "读回的 facet 数 (%d) 少于写入时 (%d)，面标签真的丢了" % (n_back, n_tagged))
