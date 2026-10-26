"""6.5 决定性探针：h=1 µm 裂纹带上，d 方程能否收敛（一次 solve_step）。

背景：h=50 µm 与 h=5 µm 时 d 的 SNES 都不收敛（D-68）。若 h=1 µm 仍不收敛，
则不必花数天跑完整疲劳。本探针只做：装配 -> W' 搜索到 δ* -> 一个循环的
solve_step（u + d 交错），报告 d 的 SNES 是否收敛。

用法：mpirun -n 6 python data/_probe_d_converge.py
"""
import pathlib
import tempfile

import yaml
from mpi4py import MPI

from pfwire.clamp_case import build_clamp_case, find_working_delta

base = yaml.safe_load(pathlib.Path(__file__).resolve().parents[1] / "configs" / "clamp_base_c.yaml".read_text(encoding="utf-8"))
a_c = 2.917e-3
raw = yaml.safe_load(yaml.safe_dump(base))
raw["mesh"] = {
    "from_config": "configs/clamp_mesh.yaml", "layout": "C",
    "workdir": str(pathlib.Path(tempfile.mkdtemp(prefix="probe_dc_")) / "mesh"),
    "edge_target": 50.0e-6,
    "refine_boxes": [
        [-3.5e-3, 3.5e-3, -0.3e-3, 0.0, 50.0e-6],
        [a_c - 0.2e-3, a_c + 0.2e-3, -0.12e-3, 0.0, 1.0e-6],
        [-a_c - 0.2e-3, -a_c + 0.2e-3, -0.12e-3, 0.0, 1.0e-6],
    ],
}
raw["contact"]["h_contact"] = 50.0e-6
raw["sampling"]["spacing"] = 1.0e-6
raw["output"]["dir"] = str(pathlib.Path(tempfile.mkdtemp(prefix="probe_dc_out_")))
raw["output"]["save_every"] = 0
p = pathlib.Path(tempfile.mkdtemp(prefix="probe_dc_cfg_")) / "cfg.yaml"
p.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")

comm = MPI.COMM_WORLD


def say(m):
    if comm.rank == 0:
        print(m, flush=True)


case = build_clamp_case(p, comm)
say("cells = %d" % case.mesh.topology.index_map(2).size_global)
delta = find_working_delta(case, float(raw["loading"]["W_target"]),
                           delta0=17.0e-6, rtol=1.0e-2, max_iter=20)
say("delta* = %.4g m, W = %.6g" % (delta, case.clamp.reaction(case.solver.u)))

# 一个循环的 solve_step（u + d 交错）
case.loads.set_phase(0.25)          # 峰值相位
try:
    result = case.solver.solve_step()
    say("solve_step: converged=%s iters=%d delta_d_inf=%.3e max_d=%.4g"
        % (result.converged, result.iterations, result.delta_d_inf, result.max_d))
    if not result.converged:
        say("FAILURE: %s" % result.failure_mode)
except RuntimeError as exc:
    say("FAIL: %s" % exc)
