"""单工况入口：yaml -> result.json + 场输出（PLAN.md 第 3 节）。

用法::

    python -m pfwire.run_case configs/benchmark_sent.yaml

当前实现 ``case_type: static``（单调加载，产出载荷–位移曲线与峰值载荷），
用于验证 1（Miehe et al. 2010 SENT）。``case_type: fatigue``（cycle-jump 驱动、
产出 N_f 与 a–N 曲线）将在验证 2 与 Phase 2 加入。

配置约定
--------
- 材料参数在 material yaml 里；工况 yaml 可用 ``model.ell`` **覆盖** ell
  （验证 1 要在同一材料下扫 ell = 0.0150/0.0075 mm，覆盖比复制材料卡干净）；
  覆盖后会按材料卡的 mode/dissipation 重新解一遍 (G_c, ell, sigma_c) 三角。
- 所有数值都过 float()，因为 PyYAML 是 YAML 1.1，`500e6` 这类写法会被解析成字符串
  （见 material.py 的 _coerce 与 CLAUDE.md 已知陷阱 2）。
- 未知配置键一律报错，不静默忽略——拼错一个键而算例照跑是最难查的错。

单位：SI。
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import ufl
import yaml
from dolfinx import default_scalar_type, fem
from mpi4py import MPI

from . import mesh as pfmesh
from . import paris as pfparis
from .cyclejump import CycleJumpConfig, CycleJumpDriver, FatigueState
from .failure import CrackPathGrid
from .material import Material, load_material, resolve_fracture_properties
from .phasefield import PhaseFieldModel, PhaseFieldSolver, SolverConfig

__all__ = [
    "CaseConfig",
    "PeakAssessment",
    "assess_peak",
    "load_case",
    "run_static",
    "run_fatigue",
    "run_case",
    "main",
]

logger = logging.getLogger(__name__)

CASE_TYPES = ("static", "fatigue", "fretting")


def _require_keys(cfg: dict, allowed: set[str], where: str) -> None:
    unknown = set(cfg) - allowed
    if unknown:
        raise KeyError("%s 段有未知键 %r（拼写错误？）" % (where, sorted(unknown)))


def _f(cfg: dict, key: str, default=None) -> float:
    value = cfg.get(key, default)
    if value is None:
        raise KeyError("缺少必需的数值项 %r" % (key,))
    return float(value)


@dataclass
class CaseConfig:
    """一个工况的完整配置（已解析、已 coerce）。"""

    path: Path
    case_type: str
    material: Material
    model: PhaseFieldModel
    solver: SolverConfig
    mesh_cfg: dict[str, Any]
    loading: dict[str, Any]
    output_dir: Path
    save_fields_every: int
    thickness: float = 1.0
    """面外厚度（m）。二维平面应变算出的支反力是**单位厚度**的（N/m），
    乘以厚度才是物理力。Miehe 2010 的 SENT 板厚 1 mm，故 thickness = 1.0e-3，
    否则会与原文的 kN 量级差 1000 倍。"""
    raw: dict[str, Any] = field(default_factory=dict)


#: 峰值可信性判据（见 run_static）。峰后连续下降步数阈值：
#: 取 3 是因为交错容差 tol_d 造成的抖动是无偏的，连续 3 步同向下降概率极低，
#: 而真正的极限点之后载荷是单调下降的。
PEAK_MIN_DESCENT = 3
#: 相对跌幅阈值：跌到峰值的 99% 以下时，无论下降了几步都已确定过峰。
PEAK_MIN_DROP_REL = 1.0e-2


@dataclass(frozen=True)
class PeakAssessment:
    """载荷–位移曲线的峰值及其**可信性**。

    为什么要单独判「可信」
    ----------------------
    验证 1 的口径是峰值载荷偏差 < 5%，所以 ``argmax`` 到底是真峰值还是「曲线
    还在上升就被打断」的下界，直接决定结论能不能成立。曾经因为把下界当峰值
    报出过错误的 "在 5% 以内"。

    判据只看曲线本身，**不看运行状态**：``run_static`` 在未收敛时是先 break
    再 append，故 curve 里只有收敛步 —— 「峰后有没有更低的载荷」这件事不因
    运行怎么结束而改变。（早期版本要求 status ∈ {completed, past_peak,
    damage_limit}，于是把明明已经过峰、只是峰后才崩的算例误标成下界。）

    - ``n_descending``：峰后**连续**下降的步数。交错容差 tol_d 造成的抖动是
      无偏的，连续同向下降数步的概率极低，而真极限点之后载荷单调下降；
    - ``post_peak_drop_rel``：峰后相对跌幅。跌破 PEAK_MIN_DROP_REL 时无论
      降了几步都已确定过峰（用于 stop_after_peak_drop 一步跌很多的情形）；
    - ``last_rise``：峰前最后一步的载荷增量。曲线仍在上升时，它给出「真峰值
      还能高多少」的量级参考。
    """

    index: int
    reliable: bool
    n_descending: int
    post_peak_drop_rel: float
    last_rise: float
    note: str


def assess_peak(forces, *, status: str = "completed") -> PeakAssessment:
    """定位峰值并判断其可信性。``forces`` 为按加载顺序排列的载荷序列。"""
    values = np.asarray(forces, dtype=float)
    if values.size == 0:
        return PeakAssessment(-1, False, 0, 0.0, 0.0, "曲线为空，无峰值")

    index = int(np.argmax(values))
    is_interior = index < values.size - 1

    n_descending = 0
    for i in range(index + 1, values.size):
        if values[i] >= values[i - 1]:
            break
        n_descending += 1

    peak = float(values[index])
    post_peak_drop_rel = (
        float((peak - values[index + 1 :].min()) / peak)
        if is_interior and peak > 0.0
        else 0.0
    )
    last_rise = float(values[index] - values[index - 1]) if index > 0 else 0.0
    reliable = bool(
        is_interior
        and (
            n_descending >= PEAK_MIN_DESCENT
            or post_peak_drop_rel >= PEAK_MIN_DROP_REL
        )
    )
    if reliable:
        note = "峰值可信：峰后连续下降 %d 步、相对跌幅 %.2e" % (
            n_descending,
            post_peak_drop_rel,
        )
    else:
        note = (
            "**不是真峰值**：曲线到最后一个收敛步仍在上升（status=%s，"
            "峰后连续下降 %d 步、跌幅 %.2e），此值只是下界；"
            "峰前最后一步增量 %+.4g 可作「还差多少」的量级参考"
            % (status, n_descending, post_peak_drop_rel, last_rise)
        )
    return PeakAssessment(
        index=index,
        reliable=reliable,
        n_descending=n_descending,
        post_peak_drop_rel=post_peak_drop_rel,
        last_rise=last_rise,
        note=note,
    )


def load_case(path: str | Path) -> CaseConfig:
    """读工况 yaml 并组装成 :class:`CaseConfig`。"""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError("%s 不是 mapping" % (path,))
    _require_keys(
        raw,
        {
            "case_type", "material", "model", "solver", "mesh", "loading",
            "output",
            # fretting 工况专有：接触参数与失效判据各自成段
            "contact", "failure",
        },
        str(path),
    )

    case_type = str(raw.get("case_type", "static"))
    if case_type not in CASE_TYPES:
        raise ValueError("case_type 只能取 %r，收到 %r" % (CASE_TYPES, case_type))

    # material 路径：先按给定的（相对 cwd）找，再退回到工况 yaml 所在目录
    material_path = Path(raw["material"])
    if not material_path.is_absolute() and not material_path.exists():
        candidate = path.parent / material_path.name
        if not candidate.exists():
            raise FileNotFoundError(
                "找不到材料卡 %s（也不在 %s 下）" % (material_path, path.parent)
            )
        material_path = candidate
    material = load_material(material_path)

    # ⚠ dict(raw.get("model", {})) 在「model: 后只有注释」时为 dict(None) ->
    # TypeError。YAML 把空的 ``model:`` 解析成 null，而 get 的默认值只管
    # **键不存在**的情形。
    model_cfg = dict(raw.get("model") or {})
    _require_keys(model_cfg, {"split", "dissipation", "ell", "k_res"}, "model")
    ell_override = model_cfg.get("ell")
    if ell_override is not None:
        # 按材料卡的 mode/dissipation 重解三角，保证 sigma_c 与新 ell 自洽
        frac = material.fracture
        if frac.derived == "ell":
            raise ValueError(
                "材料卡的 fracture.mode 让 ell 成为导出量，工况里再覆盖 ell 自相矛盾。"
                "请把材料卡改成 from_gc_and_ell 或 from_sigma_c_and_ell。"
            )
        # 保留材料卡原本的两个**输入**中除 ell 以外的那个，用新 ell 重解三角
        kwargs: dict[str, float] = {"ell": float(ell_override)}
        if frac.derived == "Gc":  # 输入曾是 (sigma_c, ell)
            mode = "from_sigma_c_and_ell"
            kwargs["sigma_c"] = frac.sigma_c
        else:  # derived == "sigma_c"，输入曾是 (Gc, ell)
            mode = "from_gc_and_ell"
            kwargs["Gc"] = frac.Gc
        new_fracture = resolve_fracture_properties(
            material.elastic.E,
            material.elastic.nu,
            mode=mode,
            dissipation=frac.dissipation,
            **kwargs,
        )
        material = Material(
            name=material.name,
            elastic=material.elastic,
            fracture=new_fracture,
            sigma_u=material.sigma_u,
            alpha_T=material.alpha_T,
            sources=material.sources,
            fatigue_form=material.fatigue_form,
            kappa=material.kappa,
        )

    model = PhaseFieldModel(
        material=material,
        split=str(model_cfg.get("split", "amor")),
        k_res=float(model_cfg.get("k_res", 1.0e-6)),
        dissipation=str(model_cfg.get("dissipation", material.fracture.dissipation)),
        # b-1 口径开关（D-68 问题 5）。默认 True 即 PLAN §2 原口径不变。
        degrade_stiffness=bool(model_cfg.get("degrade_stiffness", True)),
    )

    solver_cfg = dict(raw.get("solver", {}))
    _require_keys(
        solver_cfg,
        {"u_solver", "tol_d", "max_iter", "degree_u", "degree_d",
         "log_iterations", "relaxation", "u_linesearch"},
        "solver",
    )
    solver = SolverConfig(
        u_solver=str(solver_cfg.get("u_solver", "linear")),
        tol_d=float(solver_cfg.get("tol_d", 1.0e-4)),
        max_iter=int(solver_cfg.get("max_iter", 50)),
        degree_u=int(solver_cfg.get("degree_u", 1)),
        degree_d=int(solver_cfg.get("degree_d", 1)),
        log_iterations=bool(solver_cfg.get("log_iterations", False)),
        relaxation=float(solver_cfg.get("relaxation", 1.0)),
        u_relaxation=float(solver_cfg.get("u_relaxation", 1.0)),
        u_linesearch=str(solver_cfg.get("u_linesearch", "bt")),
        u_linesearch_cyclic=(str(solver_cfg["u_linesearch_cyclic"])
                             if solver_cfg.get("u_linesearch_cyclic") else None),
    )

    output_cfg = dict(raw.get("output", {}))
    # snapshot_every 与 save_fields_every 是两回事，别混：
    #   save_fields_every -> checkpoint（续跑用的状态，两槽轮换只留最后两个）
    #   snapshot_every    -> 采样网格上的 d 场时间序列（论文图 4 用，只增不删）
    # 当初两者共用一个键，结果生产跑完只剩两个断点、没有场序列可画（D-31）。
    _require_keys(
        output_cfg, {"dir", "save_fields_every", "snapshot_every"}, "output")

    return CaseConfig(
        path=path,
        case_type=case_type,
        material=material,
        model=model,
        solver=solver,
        mesh_cfg=dict(raw.get("mesh", {})),
        loading=dict(raw.get("loading", {})),
        output_dir=Path(output_cfg.get("dir", "data/%s" % path.stem)),
        save_fields_every=int(output_cfg.get("save_fields_every", 0)),
        thickness=float(dict(raw.get("mesh", {})).get("thickness", 1.0)),
        raw=raw,
    )


def _build_mesh(cfg: CaseConfig, comm: MPI.Comm):
    mesh_cfg = dict(cfg.mesh_cfg)
    mesh_type = str(mesh_cfg.pop("type", "sent"))
    mesh_cfg.pop("thickness", None)  # 厚度只用于把 N/m 换成 N，不进几何
    if mesh_type == "sent":
        _require_keys(
            mesh_cfg,
            {"W", "H", "notch_length", "h_fine", "h_coarse", "refine_halfheight"},
            "mesh",
        )
        spec = pfmesh.SentSpec(**{k: float(v) for k, v in mesh_cfg.items()})
        return pfmesh.sent_mesh(spec, comm)
    if mesh_type == "ct":
        _require_keys(
            mesh_cfg,
            {
                "W",
                "a0_over_W",
                "total_width_over_W",
                "halfheight_over_W",
                "hole_diameter_over_W",
                "hole_offset_over_W",
                "notch_height_over_W",
                "notch_taper_length",
                "h_fine",
                "h_coarse",
                "refine_halfheight",
                "refine_x_margin",
            },
            "mesh",
        )
        spec = pfmesh.CtSpec(**{k: float(v) for k, v in mesh_cfg.items()})
        return pfmesh.ct_mesh(spec, comm)
    if mesh_type == "fretting":
        # 微动试样的 a 由 loads 从 (p0, R, E*) 算出后传入，不走这里（见 D-25）；
        # fretting.run_fretting 自己建网格，故这里不该被调用到。
        raise RuntimeError(
            "mesh.type='fretting' 的网格由 fretting.run_fretting 建（a 需与面力同源，"
            "见 DECISIONS.md D-25），不应经由 _build_mesh"
        )
    raise NotImplementedError(
        "mesh.type 只支持 'sent'（验证 1）、'ct'（验证 2）与 'fretting'（Phase 2）。"
    )


def _load_stages(loading: dict) -> list[tuple[float, int]]:
    stages = loading.get("stages")
    if not stages:
        raise KeyError("loading 段缺少 stages（每项 {du, n}）")
    out = []
    for i, stage in enumerate(stages):
        _require_keys(dict(stage), {"du", "n"}, "loading.stages[%d]" % i)
        du = float(stage["du"])
        n = int(stage["n"])
        if du <= 0.0 or n < 1:
            raise ValueError("loading.stages[%d]：du 必须为正、n >= 1" % i)
        out.append((du, n))
    return out


def run_static(cfg: CaseConfig, comm: MPI.Comm = MPI.COMM_WORLD) -> dict[str, Any]:
    """单调位移加载，返回载荷–位移曲线与峰值载荷。

    载荷方式（``loading.type = "tension_y"``）：底边全约束，顶边给竖向位移，
    支反力由 :meth:`PhaseFieldSolver.reaction` 沿 +y 探针提取。
    """
    _require_keys(
        dict(cfg.loading),
        {"type", "stages", "stop_damage", "stop_after_peak_drop"},
        "loading",
    )
    loading_type = str(cfg.loading.get("type", "tension_y"))
    if loading_type != "tension_y":
        raise NotImplementedError("目前只实现 loading.type='tension_y'")
    stop_damage = float(cfg.loading.get("stop_damage", 0.99))
    # 峰后跌到 (1 - stop_after_peak_drop)·F_peak 以下即停：验证 1 只要峰值，
    # 峰后那段「brutal」下降对验收无用却很费时。None 表示不启用。
    peak_drop = cfg.loading.get("stop_after_peak_drop")
    peak_drop = None if peak_drop is None else float(peak_drop)
    if peak_drop is not None and not 0.0 < peak_drop < 1.0:
        raise ValueError("stop_after_peak_drop 须在 (0,1) 内，收到 %r" % (peak_drop,))
    f_peak_so_far = -np.inf

    is_root = comm.rank == 0

    def log(level, msg, *args):
        if is_root:
            getattr(logger, level)(msg, *args)

    t_start = time.perf_counter()
    sent = _build_mesh(cfg, comm)
    msh, tags = sent.mesh, sent.facet_tags
    fdim = msh.topology.dim - 1
    n_cells = msh.topology.index_map(msh.topology.dim).size_global
    t_mesh = time.perf_counter() - t_start

    u_applied = fem.Constant(msh, default_scalar_type(0.0))

    def make_bcs(V):
        bottom = tags.find(pfmesh.SENT_TAG_BOTTOM)
        top = tags.find(pfmesh.SENT_TAG_TOP)
        Vy = V.sub(1)
        zero_vec = np.zeros(msh.geometry.dim, dtype=default_scalar_type)
        return [
            fem.dirichletbc(
                zero_vec, fem.locate_dofs_topological(V, fdim, bottom), V
            ),
            fem.dirichletbc(
                u_applied, fem.locate_dofs_topological(Vy, fdim, top), Vy
            ),
        ]

    def make_probe_bcs(V):
        top = tags.find(pfmesh.SENT_TAG_TOP)
        Vy = V.sub(1)
        one = fem.Constant(msh, default_scalar_type(1.0))
        return [fem.dirichletbc(one, fem.locate_dofs_topological(Vy, fdim, top), Vy)]

    solver = PhaseFieldSolver(msh, cfg.model, bcs_u=make_bcs, config=cfg.solver)
    probe = solver.make_probe(make_probe_bcs)

    stages = _load_stages(cfg.loading)
    n_total = sum(n for _, n in stages)
    progress_every = max(1, n_total // 20)  # 约 20 条进度行
    curve: list[dict[str, float]] = []
    u_value = 0.0
    step = 0
    stopped = "completed"
    t_solve = time.perf_counter()
    log("info", "开始加载：%d 个阶段共 %d 步，网格 %d 单元", len(stages), n_total, n_cells)
    for du, n_steps in stages:
        for _ in range(n_steps):
            u_value += du
            u_applied.value = u_value
            result = solver.solve_step()
            step += 1
            if not result.converged:
                stopped = "not_converged"
                log("error", "第 %d 步未收敛：%s", step, result.failure_mode)
                break
            solver.commit_step()
            curve.append(
                {
                    "step": step,
                    "u": u_value,
                    "F_per_thickness": solver.reaction(probe),
                    "F": solver.reaction(probe) * cfg.thickness,
                    "max_d": result.max_d,
                    "iterations": result.iterations,
                }
            )
            f_now = curve[-1]["F"]
            f_peak_so_far = max(f_peak_so_far, f_now)
            if step % progress_every == 0 or result.max_d > 0.0:
                log(
                    "info",
                    "步 %d/%d  u=%.4e m  F=%.6g N  max_d=%.4f  交错 %d 次  已用 %.0f s",
                    step,
                    n_total,
                    u_value,
                    curve[-1]["F"],
                    result.max_d,
                    result.iterations,
                    time.perf_counter() - t_solve,
                )
            if (
                peak_drop is not None
                and f_peak_so_far > 0.0
                and f_now < (1.0 - peak_drop) * f_peak_so_far
            ):
                stopped = "past_peak"
                log(
                    "info",
                    "峰后已跌至峰值的 %.0f%%，停止（峰值 %.6g N）",
                    100.0 * f_now / f_peak_so_far,
                    f_peak_so_far,
                )
                break
            if result.max_d >= stop_damage:
                stopped = "damage_limit"
                break
        if stopped != "completed":
            break
    t_solve = time.perf_counter() - t_solve

    forces = np.array([c["F"] for c in curve])
    displacements = np.array([c["u"] for c in curve])
    assessment = assess_peak(forces, status=stopped)
    peak_index = assessment.index
    result: dict[str, Any] = {
        "case": cfg.path.name,
        "case_type": cfg.case_type,
        "status": stopped,
        "material": cfg.material.name,
        "model": {
            "split": cfg.model.split,
            "dissipation": cfg.model.dissipation,
            "k_res": cfg.model.k_res,
            "ell": cfg.material.fracture.ell,
            "Gc": cfg.material.fracture.Gc,
            "sigma_c": cfg.material.fracture.sigma_c,
        },
        "mesh": {"type": "sent", "n_cells": int(n_cells), **cfg.mesh_cfg},
        "thickness": cfg.thickness,
        "peak": {
            "reliable": assessment.reliable,
            "note": assessment.note,
            "n_descending_after": assessment.n_descending,
            "post_peak_drop_rel": assessment.post_peak_drop_rel,
            "last_rise_before_peak": assessment.last_rise,
            "F": float(forces[peak_index]) if peak_index >= 0 else None,
            "F_per_thickness": (
                float(forces[peak_index] / cfg.thickness) if peak_index >= 0 else None
            ),
            "u": float(displacements[peak_index]) if peak_index >= 0 else None,
            "step": peak_index + 1 if peak_index >= 0 else None,
        },
        "n_steps": step,
        "timing_s": {"mesh": t_mesh, "solve": t_solve},
        "curve": curve,
    }

    if comm.rank == 0:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (cfg.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (cfg.output_dir / "config.yaml").write_text(
            yaml.safe_dump(cfg.raw, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return result


def _triangular_wave(phase: float) -> float:
    """周期三角波：相位 0 -> 0，0.5 -> 1，1 -> 0（连续、闭合）。

    Carrara Fig. 11 的循环载荷是在 P_min 与 P_max 之间往复。用三角波而不是
    正弦：ᾱ 的累积由 Eq. 35 的加载/卸载判据控制（只在 ψ⁺ 上升时累积），
    与波形细节无关，而三角波让 n_sub 个子步在载荷上等间距、外推最稳。
    """
    x = float(phase) % 1.0
    return 2.0 * x if x <= 0.5 else 2.0 * (1.0 - x)


def run_fatigue(cfg: CaseConfig, comm: MPI.Comm = MPI.COMM_WORLD) -> dict[str, Any]:
    """力控循环加载 + cycle-jump，输出 a(N) 与 Paris 拟合（验证 2）。

    载荷方式（``loading.type = "ct_pin_force"``，Carrara Fig. 12a）：
    下销孔全约束（原文的 ū = 0），上销孔施加合力为 P(t) 的竖向面力。

    **两个显式的建模选择**（原文只画了箭头，没给分布）：
    1. 面力在**整个上孔周边均匀**施加，合力 = P；真实销接触只压半边，但裂纹面
       距孔心 0.275W ≈ 2.2 个孔半径，圣维南原理下分布细节对 ΔK 无影响，
       而 ΔK 本身用的是 ASTM 远场标定式 Y(a/W)；
    2. 二维是**单位厚度**求解，故面力 t = P / (T · L_hole)，其中 L_hole 是
       实测的孔周长（不是解析弧长 —— 网格上是内接多边形，短 O(1/n²)，
       用解析值会让合力偏大那么多）。
    """
    _require_keys(
        dict(cfg.loading),
        {
            "type",
            "delta_P",
            "P_min",
            "cycle",
            "crack",
            "stop_at_a_over_W",
            "paris_delta_a",
            "dump_every",
        },
        "loading",
    )
    loading_type = str(cfg.loading.get("type", "ct_pin_force"))
    if loading_type != "ct_pin_force":
        raise NotImplementedError(
            "fatigue 目前只实现 loading.type='ct_pin_force'（验证 2）"
        )
    delta_P = float(cfg.loading["delta_P"])
    P_min = float(cfg.loading.get("P_min", 0.0))
    if delta_P <= 0.0:
        raise ValueError("delta_P 必须为正")
    if P_min < 0.0:
        raise ValueError(
            "P_min < 0 表示压缩相，本驱动只处理正向（拉伸）循环，"
            "与 Carrara §5.2 的参考算例一致"
        )
    stop_at = float(cfg.loading.get("stop_at_a_over_W", 0.7))
    #: 每多少个 jump 步把中途结果落盘一次；<= 0 表示只在结束时写
    dump_every = int(cfg.loading.get("dump_every", 50))

    is_root = comm.rank == 0

    def log(level, msg, *args):
        if is_root:
            getattr(logger, level)(msg, *args)

    t_start = time.perf_counter()
    ct = _build_mesh(cfg, comm)
    msh, tags, spec = ct.mesh, ct.facet_tags, ct.spec
    fdim = msh.topology.dim - 1
    n_cells = msh.topology.index_map(msh.topology.dim).size_global
    t_mesh = time.perf_counter() - t_start

    # 上孔周长按**网格实测**取（见 docstring 第 2 点）
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    hole_length = comm.allreduce(
        fem.assemble_scalar(
            fem.form(fem.Constant(msh, default_scalar_type(1.0))
                     * ds(pfmesh.CT_TAG_HOLE_TOP))
        ),
        op=MPI.SUM,
    )
    if hole_length <= 0.0:
        raise RuntimeError("上销孔周长为 0 —— 物理组 CT_TAG_HOLE_TOP 没匹配到边界")

    traction = fem.Constant(msh, default_scalar_type(0.0))  # N/m^2，竖向

    def make_bcs(V):
        lower = tags.find(pfmesh.CT_TAG_HOLE_BOTTOM)
        zero_vec = np.zeros(msh.geometry.dim, dtype=default_scalar_type)
        return [
            fem.dirichletbc(
                zero_vec, fem.locate_dofs_topological(V, fdim, lower), V
            )
        ]

    # 必须用可调用形式：TestFunction 要取自求解器自己的 V_u，在外部另建同构空间
    # 会让 UFL 看到两个同号不同源的 Argument 并报错（见 phasefield.WorkSpec）。
    zero = fem.Constant(msh, default_scalar_type(0.0))

    def make_external_work(V):
        v = ufl.TestFunction(V)
        t = ufl.as_vector([zero, traction])
        return ufl.inner(t, v) * ds(pfmesh.CT_TAG_HOLE_TOP)

    alpha_T, form, kappa = cfg.material.require_fatigue_params()
    state = FatigueState(msh, cfg.model, alpha_T=alpha_T, form=form, kappa=kappa)
    solver = PhaseFieldSolver(
        msh,
        cfg.model,
        bcs_u=make_bcs,
        external_work=make_external_work,
        f_alpha=state.f_alpha,
        config=cfg.solver,
    )
    state.bind_solver(solver)

    def set_phase(phase: float) -> None:
        P = P_min + delta_P * _triangular_wave(phase)
        traction.value = P / (cfg.thickness * hole_length)

    cycle_cfg = dict(cfg.loading.get("cycle", {}))
    _require_keys(
        cycle_cfg,
        {"n_sub", "dN_init", "dN_min", "dN_max", "dd_max", "grow_below",
         "factor", "N_max", "peak_phase"},
        "loading.cycle",
    )
    jump_cfg = CycleJumpConfig(
        n_sub=int(cycle_cfg.get("n_sub", 8)),
        dN_init=float(cycle_cfg.get("dN_init", 100.0)),
        dN_min=float(cycle_cfg.get("dN_min", 1.0)),
        dN_max=float(cycle_cfg.get("dN_max", 1.0e5)),
        dd_max=float(cycle_cfg.get("dd_max", 0.02)),
        grow_below=float(cycle_cfg.get("grow_below", 0.5)),
        factor=float(cycle_cfg.get("factor", 2.0)),
        N_max=float(cycle_cfg.get("N_max", 1.0e8)),
        peak_phase=(
            None if cycle_cfg.get("peak_phase") is None
            else float(cycle_cfg["peak_phase"])
        ),
    )
    driver = CycleJumpDriver(solver, state, set_phase, jump_cfg)

    crack_cfg = dict(cfg.loading.get("crack", {}))
    _require_keys(
        crack_cfg, {"threshold", "spacing", "halfwidth", "growth_length"}, "loading.crack"
    )
    threshold = float(crack_cfg.get("threshold", 0.95))
    # 裂纹自缺口尖端 (a0, 0) 沿 +x 生长；默认量到背面前留一点余量
    growth_length = float(
        crack_cfg.get("growth_length", 0.95 * (spec.x_back - spec.a0))
    )
    grid = CrackPathGrid(
        msh,
        origin=(spec.a0, 0.0),
        growth_length=growth_length,
        halfwidth=float(crack_cfg["halfwidth"]),
        spacing=float(crack_cfg["spacing"]),
        growth_dir=(1.0, 0.0),
        h_fine=spec.h_fine,
    )

    history: list[dict[str, float]] = []
    stopped = "completed"
    t_solve = time.perf_counter()
    log(
        "info",
        "CT 疲劳算例：ΔP=%.4g N，P_min=%.4g N，a0/W=%.3f，网格 %d 单元，"
        "上孔周长 %.6f m（解析 %.6f）",
        delta_P,
        P_min,
        spec.a0_over_W,
        n_cells,
        hole_length,
        2.0 * np.pi * spec.hole_radius,
    )

    # 分项计时：裂纹采样每个 jump 步都做一次，网格细化后它可能反客为主，
    # 而只看总耗时是分辨不出「求解慢」还是「量裂纹慢」的。
    timers = {"record": 0.0}

    def record(report) -> None:
        t_rec = time.perf_counter()
        metrics = grid.crack_extension(solver.d, threshold=threshold)
        timers["record"] += time.perf_counter() - t_rec
        a = spec.a0 + metrics.depth
        history.append(
            {
                "N": report.N,
                "delta_N": report.delta_N,
                "a": a,
                "a_over_W": a / spec.W,
                "crack_extension": metrics.depth,
                "max_d": report.max_d,
                "max_alpha_bar": report.max_alpha_bar,
                # 失败步的 max_dd 是 NaN，裸 NaN 不是合法 JSON
                "max_dd": (None if report.max_dd != report.max_dd
                           else float(report.max_dd)),
                "rejections": report.rejections,
                "n_surface_components": metrics.n_surface_components,
            }
        )

    def make_result(status: str, n_jumps: int, solve_seconds: float) -> dict[str, Any]:
        """组装结果字典。中途快照与最终结果共用，避免两处逻辑漂移。"""
        out: dict[str, Any] = {
            "case": cfg.path.name,
            "case_type": cfg.case_type,
            "status": status,
            "material": cfg.material.name,
            "model": {
                "split": cfg.model.split,
                "dissipation": cfg.model.dissipation,
                "k_res": cfg.model.k_res,
                "ell": cfg.material.fracture.ell,
                "Gc": cfg.material.fracture.Gc,
                "sigma_c": cfg.material.fracture.sigma_c,
                "alpha_T": alpha_T,
                "fatigue_form": form,
                "kappa": kappa,
            },
            "mesh": {"type": "ct", "n_cells": int(n_cells), **cfg.mesh_cfg},
            "thickness": cfg.thickness,
            "loading": {
                "delta_P": delta_P,
                "P_min": P_min,
                "hole_length_mesh": float(hole_length),
                "hole_length_analytic": float(2.0 * np.pi * spec.hole_radius),
            },
            "n_jumps": n_jumps,
            "N_final": float(history[-1]["N"]) if history else 0.0,
            "timing_s": {"mesh": t_mesh, "solve": solve_seconds},
            "history": history,
        }
        if status == "running":
            out["note"] = _partial_result_note(status)
        # Paris 拟合。点数不足时**不报错、只记原因** —— 算例本身可能只是还没跑到
        # 稳定扩展段，那不是 bug，但也绝不能悄悄给个假的 C/m。
        out["paris"] = _fit_paris_or_explain(history, spec, cfg, delta_P)
        return out

    def write_result(out: dict[str, Any]) -> None:
        if comm.rank != 0:
            return
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        # 先写临时文件再 rename：中途落盘时进程可能随时被打断，
        # 直接覆盖会留下一个截断的 JSON，比没有还糟（读的人以为是完整结果）。
        tmp = cfg.output_dir / "result.json.tmp"
        tmp.write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(cfg.output_dir / "result.json")
        (cfg.output_dir / "config.yaml").write_text(
            yaml.safe_dump(cfg.raw, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def stop(report) -> bool:
        record(report)
        last = history[-1]
        # 周期性落盘：疲劳算例动辄跑几小时到一天，只在结束时写 result.json
        # 意味着任何中断都让全部 history 化为乌有（日志里每 10 步一条只是残影）。
        if dump_every > 0 and len(history) % dump_every == 0:
            write_result(
                make_result("running", len(driver.reports),
                            time.perf_counter() - t_solve)
            )
        # 前 5 步逐步打：长算例最怕「起步就卡住却看不见」，而每 10 步一行时
        # 第一行要等 10 步才出现，卡在第 3 步就完全没有输出可看。
        if len(history) <= 5 or len(history) % 10 == 0 or last["a_over_W"] >= stop_at:
            elapsed = time.perf_counter() - t_solve
            log(
                "info",
                "N=%.4g  ΔN=%.4g  a=%.4f mm (a/W=%.3f)  max_d=%.4f  ᾱ_max=%.4g  "
                "已用 %.0f s（其中取裂纹长度 %.0f s = %.0f%%，%d 步）",
                report.N,
                report.delta_N,
                1e3 * last["a"],
                last["a_over_W"],
                report.max_d,
                report.max_alpha_bar,
                elapsed,
                timers["record"],
                100.0 * timers["record"] / max(elapsed, 1e-12),
                len(history),
            )
        return last["a_over_W"] >= stop_at

    reports = driver.run(stop=stop)
    t_solve = time.perf_counter() - t_solve

    final = reports[-1] if reports else None
    if final is not None and not final.accepted:
        stopped = "not_converged"
        log("error", "cycle-jump 在 N=%.4g 失败：%s", final.N, final.failure_mode)
    elif final is not None and final.runout:
        stopped = "runout"
    elif history and history[-1]["a_over_W"] >= stop_at:
        stopped = "crack_limit"

    result = make_result(stopped, len(reports), t_solve)
    write_result(result)
    return result


def _partial_result_note(status: str) -> str:
    return (
        "**这是运行中的快照，不是最终结果**（status=%s）。算例还在跑，"
        "history 与 paris 都只反映已完成的部分；ΔK 跨度不足时拟合出的 m 会偏。" % status
    )


def _fit_paris_or_explain(history, spec, cfg: CaseConfig, delta_P: float) -> dict:
    """对 a(N) 做 Paris 拟合；失败时返回 ``{"ok": False, "reason": ...}``。"""
    if len(history) < 2:
        return {"ok": False, "reason": "只有 %d 个 a(N) 点" % len(history)}
    a = np.array([h["a"] for h in history])
    N = np.array([h["N"] for h in history])
    # 裂纹一步也没长时不必拟合
    if a[-1] - a[0] <= 0.0:
        return {"ok": False, "reason": "裂纹未扩展（a 始终为 a0）"}
    delta_a = float(cfg.loading.get("paris_delta_a", 0.25e-3))
    try:
        fit = pfparis.paris_fit(
            a,
            N,
            W=spec.W,
            T=cfg.thickness,
            delta_P=delta_P,
            delta_a=delta_a,
        )
    except ValueError as exc:
        return {"ok": False, "reason": str(exc), "delta_a": delta_a}
    return {
        "ok": True,
        "delta_a": delta_a,
        "C": fit.C,
        "m": fit.m,
        "C_mm_MPa": fit.C_mm_MPa,
        "n_points_used": fit.n_points_used,
        "delta_K_range": list(fit.delta_K_range),
        "fit_window": list(fit.fit_window),
        "r_squared": fit.r_squared,
        "da_dN": fit.rate.da_dN.tolist(),
        "delta_K": fit.rate.delta_K.tolist(),
        "a_mid": fit.rate.a_mid.tolist(),
    }


def run_case(
    path: str | Path, comm: MPI.Comm = MPI.COMM_WORLD, *, resume: bool = False
) -> dict[str, Any]:
    cfg = load_case(path)
    # 护栏必须排在 dispatch **之前**：放在后面的话 static/fatigue 会先 return，
    # 护栏永远执行不到 —— 传了 --resume 却从头重跑，日志还看着像续上了。
    if resume and cfg.case_type != "fretting":
        raise NotImplementedError(
            "case_type=%r 还没实现续跑；不要静默忽略 --resume，"
            "否则会从头重跑却看起来像续上了。" % (cfg.case_type,)
        )
    if cfg.case_type == "static":
        return run_static(cfg, comm)
    if cfg.case_type == "fatigue":
        return run_fatigue(cfg, comm)
    if cfg.case_type == "fretting":
        from .fretting import run_fretting

        return run_fretting(cfg, comm, resume=resume)
    raise NotImplementedError("未知 case_type: %r" % (cfg.case_type,))


def _summarize(result: dict[str, Any]) -> str:
    """一行汇总。static 报峰值载荷，fatigue 报寿命与 Paris 参数，fretting 报关卡。"""
    if result.get("case_type") == "fretting":
        gate = result.get("gate", {})
        hist = result.get("history") or []
        depth = hist[-1]["depth"] * 1e6 if hist else float("nan")
        verdict = {"pass": "通过", "fail": "**未通过**",
                   "inconclusive": "**判不了**（深度没到判定门槛）"}.get(
            gate.get("verdict"), "**未通过**")
        run = result.get("runaway")
        extra = ""
        if run is not None:
            extra = "  失稳判别=%s（%.2f -> %.2f µm，%d 次迭代）" % (
                run["verdict"], run["depth_start"] * 1e6,
                run["depth_end"] * 1e6, run["n_iterations"])
        if result.get("N_f") is not None:
            extra += "  N_f=%.6g" % result["N_f"]
        return (
            "状态=%s  jump 步数=%d  N=%.6g  裂纹深度=%.2f µm(%.2f a)  "
            "关卡=%s（滑移区萌生 %s，初始倾斜 %s，尖端转向 %s）%s  耗时 %.1f s"
            % (
                result["status"], result["n_jumps"], result["N_final"], depth,
                gate.get("max_depth_over_a", float("nan")),
                verdict,
                gate.get("nucleates_in_slip_zone"),
                gate.get("initially_tilted"),
                gate.get("turns_toward_normal"),
                extra,
                result["timing_s"]["solve"],
            )
        )
    if result.get("case_type") == "fatigue":
        paris = result.get("paris", {})
        tail = (
            "C=%.4g m=%.4g (C_mm_MPa=%.4g, R^2=%.4f, %d 点)"
            % (
                paris["C"],
                paris["m"],
                paris["C_mm_MPa"],
                paris["r_squared"],
                paris["n_points_used"],
            )
            if paris.get("ok")
            else "Paris 未拟合：%s" % paris.get("reason", "未知")
        )
        hist = result.get("history") or []
        a_over_W = hist[-1]["a_over_W"] if hist else float("nan")
        return "状态=%s  jump 步数=%d  N=%.6g  a/W=%.4f  %s  耗时 %.1f s" % (
            result["status"],
            result["n_jumps"],
            result["N_final"],
            a_over_W,
            tail,
            result["timing_s"]["solve"],
        )
    peak = result.get("peak", {})
    return "状态=%s  步数=%d  峰值 F=%.6g N (u=%.6g m)  耗时 %.1f s" % (
        result["status"],
        result["n_steps"],
        peak.get("F") or float("nan"),
        peak.get("u") or float("nan"),
        result["timing_s"]["solve"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行单个工况")
    parser.add_argument("config", help="工况 yaml 路径")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="从 output.dir 下的断点接着跑（硬约束 5）。断点每 "
             "output.save_fields_every 个 jump 步存一次。",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    comm = MPI.COMM_WORLD
    result = run_case(args.config, comm, resume=args.resume)
    if comm.rank == 0:
        print(_summarize(result))
    return (
        0
        if result["status"]
        in ("completed", "damage_limit", "past_peak", "crack_limit", "runout",
            "failed", "failed_unstable")
        else 1
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
