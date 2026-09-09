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
import yaml
from dolfinx import default_scalar_type, fem
from mpi4py import MPI

from . import mesh as pfmesh
from .material import Material, load_material, resolve_fracture_properties
from .phasefield import PhaseFieldModel, PhaseFieldSolver, SolverConfig

__all__ = ["CaseConfig", "load_case", "run_static", "run_case", "main"]

logger = logging.getLogger(__name__)

CASE_TYPES = ("static", "fatigue")


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


def load_case(path: str | Path) -> CaseConfig:
    """读工况 yaml 并组装成 :class:`CaseConfig`。"""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError("%s 不是 mapping" % (path,))
    _require_keys(
        raw,
        {"case_type", "material", "model", "solver", "mesh", "loading", "output"},
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

    model_cfg = dict(raw.get("model", {}))
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
    )

    solver_cfg = dict(raw.get("solver", {}))
    _require_keys(
        solver_cfg,
        {"u_solver", "tol_d", "max_iter", "degree_u", "degree_d",
         "log_iterations", "relaxation"},
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
    )

    output_cfg = dict(raw.get("output", {}))
    _require_keys(output_cfg, {"dir", "save_fields_every"}, "output")

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
    if mesh_type != "sent":
        raise NotImplementedError(
            "目前只实现了 mesh.type='sent'（验证 1）。微动试样属 Phase 2.1。"
        )
    _require_keys(
        mesh_cfg,
        {"W", "H", "notch_length", "h_fine", "h_coarse", "refine_halfheight"},
        "mesh",
    )
    spec = pfmesh.SentSpec(**{k: float(v) for k, v in mesh_cfg.items()})
    return pfmesh.sent_mesh(spec, comm)


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
    peak_index = int(np.argmax(forces)) if forces.size else -1
    # **诚实报告**：若"峰值"就是最后一个记录点、且运行不是因为过峰/损伤上限而停，
    # 那它只是载荷的**下界**（曲线还在上升就被打断了），绝不能当峰值用。
    peak_is_interior = 0 <= peak_index < len(curve) - 1
    peak_reliable = bool(
        peak_is_interior and stopped in ("completed", "past_peak", "damage_limit")
    )
    peak_note = (
        "峰值可信：其后有载荷更低的步"
        if peak_reliable
        else "**不是真峰值**：曲线到最后一步仍在上升（status=%s），此值只是下界" % stopped
    )
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
            "reliable": peak_reliable,
            "note": peak_note,
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


def run_case(path: str | Path, comm: MPI.Comm = MPI.COMM_WORLD) -> dict[str, Any]:
    cfg = load_case(path)
    if cfg.case_type == "static":
        return run_static(cfg, comm)
    raise NotImplementedError(
        "case_type='fatigue' 尚未实现（验证 2 与 Phase 2 会加入）"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行单个工况")
    parser.add_argument("config", help="工况 yaml 路径")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    comm = MPI.COMM_WORLD
    result = run_case(args.config, comm)
    if comm.rank == 0:
        peak = result.get("peak", {})
        print(
            "状态=%s  步数=%d  峰值 F=%.6g N (u=%.6g m)  耗时 %.1f s"
            % (
                result["status"],
                result["n_steps"],
                peak.get("F") or float("nan"),
                peak.get("u") or float("nan"),
                result["timing_s"]["solve"],
            )
        )
    return 0 if result["status"] in ("completed", "damage_limit", "past_peak") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
