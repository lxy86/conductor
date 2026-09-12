"""微动工况的断点续跑端到端验收（硬约束 5）。

判据只有一条：**续跑的结果必须与没中断过的那次逐位一致**。
`tests/test_checkpoint.py` 测的是存取模块本身，这里测的是接到 run_case 之后
整条路走通 —— 两者都要，因为这条路上真出过两个问题：
算例失败退出时不存断点（丢掉全部进度），以及 runout 那一步不进 history
（N_final 少报一整个 jump）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from mpi4py import MPI

from pfwire.run_case import run_case

REPO = Path(__file__).resolve().parents[1]

#: ΔN 必须两边走同一串，否则比的是错的东西。
#: delta_N 会被 ``min(delta_N, N_max - N)`` 钳制，中断那次的 N_max 更小、
#: 最后一步就被截短；而 cycle-jump 是外推格式，不同步长序列本来就给不同结果。
#: 令 grow_below = 0 使「ΔN 翻倍」的条件恒假，ΔN 恒为 dN_init。
def _case(out_dir: Path, n_max: float) -> dict:
    return {
        "case_type": "fretting",
        "material": "configs/material_al1350.yaml",
        "model": {"split": "spectral", "dissipation": "AT1", "k_res": 1.0e-6},
        "solver": {"u_solver": "newton", "u_linesearch": "basic",
                   "tol_d": 1.0e-4, "max_iter": 300, "relaxation": 0.5},
        "contact": {"p0": 240.0e6, "mu": 0.7, "Q_over_muP": 0.6,
                    "R_wire": 1.875e-3, "counterpart": "aluminum"},
        # 粗网格：这里验的是续跑的管路，不是数值精度
        "mesh": {"type": "fretting", "W": 6.4e-3, "H": 3.2e-3,
                 "refine_halfwidth_factor": 3.0, "refine_depth": 30.0e-6,
                 "h_fine": 2.0e-6, "h_coarse": 4.0e-4, "transition_factor": 8.0},
        "loading": {
            "type": "fretting_contact", "sigma_a": 35.0e6,
            "cycle": {"n_sub": 8, "dN_init": 100.0, "dN_min": 1.0,
                      "dN_max": 1.0e5, "dd_max": 0.02, "grow_below": 0.0,
                      "factor": 2.0, "N_max": n_max, "peak_phase": None},
            "dump_every": 1,
        },
        "failure": {"d_threshold": 0.95, "depth": 100.0e-6,
                    "sample_halfwidth_factor": 3.0, "sample_depth": 25.0e-6,
                    "sample_spacing": 2.0e-6, "inclination_depth": 6.0e-6,
                    "tip_window": 6.0e-6, "judge_depth_over_a": 2.0},
        "output": {"dir": str(out_dir), "save_fields_every": 1},
    }


def _write(path: Path, cfg: dict) -> Path:
    if MPI.COMM_WORLD.rank == 0:
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    MPI.COMM_WORLD.Barrier()
    return path


def _last(out_dir: Path) -> tuple[dict, dict]:
    r = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    return r, r["history"][-1]


@pytest.mark.slow
def test_resumed_fretting_run_matches_uninterrupted(tmp_path, monkeypatch):
    """直跑到 N=400 vs 跑到 200 存断点再续到 400 —— 必须一致。"""
    monkeypatch.chdir(REPO)          # 配置里的 material 路径是相对仓库根的
    shared = MPI.COMM_WORLD.bcast(tmp_path, root=0)

    straight_dir = shared / "straight"
    c = _write(shared / "straight.yaml", _case(straight_dir, 400.0))
    run_case(c, MPI.COMM_WORLD)
    ref, ref_last = _last(straight_dir)
    assert ref["n_jumps"] == 4, "ΔN 应当恒为 100，4 步到 400"
    assert ref_last["max_alpha_bar"] > 0.0, "构造前提：状态非平凡"

    split_dir = shared / "split"
    half = _write(shared / "half.yaml", _case(split_dir, 200.0))
    run_case(half, MPI.COMM_WORLD)
    assert (split_dir / "checkpoint.json").is_file() or MPI.COMM_WORLD.rank != 0

    full = _write(shared / "full.yaml", _case(split_dir, 400.0))
    run_case(full, MPI.COMM_WORLD, resume=True)
    got, got_last = _last(split_dir)

    assert got["N_final"] == pytest.approx(ref["N_final"], rel=1e-12)
    assert got["n_jumps"] == ref["n_jumps"]
    for key in ("max_d", "max_alpha_bar", "depth"):
        assert got_last[key] == pytest.approx(ref_last[key], rel=1e-10), key


def test_resume_is_refused_for_case_types_that_cannot_do_it(tmp_path, monkeypatch):
    """非 fretting 工况传 --resume 必须报错。

    静默忽略的话会从头重跑，而日志看起来像续上了 —— 这种「看着对、其实白跑」
    是本项目反复吃亏的失败模式。
    """
    monkeypatch.chdir(REPO)
    with pytest.raises(NotImplementedError, match="还没实现续跑"):
        run_case("configs/benchmark_sent.yaml", MPI.COMM_WORLD, resume=True)
