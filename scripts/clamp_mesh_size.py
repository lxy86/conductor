"""Phase 6.2：线夹压单丝网格的加密布局报告（configs/clamp_mesh.yaml）。

对每个布局给出单元数：
  - 布局 A、B：只做估算（单元数 ≈ 加密区面积 / 单元边长²），不生成网格（过大）；
  - 布局 C（configs 中 generate 列出的）：实际用 gmsh 生成，报告真实单元数、
    加密带内的实际单元边长中位数（相对 edge_target = ℓ/4 的比值），并检查是否达标。

用法：python scripts/clamp_mesh_size.py [configs/clamp_mesh.yaml]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from mpi4py import MPI

from pfwire.mesh import block_mesh
from pfwire.verify_contact_common import fnum, load_config

REPO = pathlib.Path(__file__).resolve().parents[1]


def estimate_cells(area: float, edge: float) -> float:
    """四边形网格的单元数估算：面积 / 单元边长²。"""
    return area / edge**2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 6.2：加密布局的单元数报告")
    ap.add_argument("config", nargs="?", default="configs/clamp_mesh.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    comm = MPI.COMM_WORLD
    inp, geo, el = cfg["inputs"], cfg["geometry"], cfg["element"]
    a_pl, a_c = fnum(inp, "a_pl"), fnum(inp, "a_contact")
    D = fnum(geo, "depth")
    edge_t = fnum(el, "edge_target")
    ratio = fnum(el, "edge_ratio")
    h_fine, h_coarse = fnum(el, "h_fine"), fnum(el, "h_coarse")
    trans = fnum(el, "transition")

    def say(msg):
        if comm.rank == 0:
            print(msg, flush=True)

    say("Phase 6.2 加密布局（目标单元边长 ℓ/4 = %.2f µm；h_fine 设定 %.3f µm → 实际边长 ≈ %.2f µm）"
        % (1e6 * edge_t, 1e6 * h_fine, 1e6 * ratio * h_fine))
    say("  输入：a_pl = %.3f mm，a（接触半宽）= %.3f mm（6.1 工作点，δ = 17 µm）"
        % (1e3 * a_pl, 1e3 * a_c))
    say("")
    lay = cfg["layouts"]
    for name in ("A", "B"):
        L = lay[name]
        hw = fnum(L, "halfwidth_factor") * a_pl
        area = 2 * hw * fnum(L, "fine_depth")
        say("  布局 %s（估算）：|x| ≤ %.2f a_pl = %.2f mm，深 %.3f mm  →  加密面积 %.2e m²，单元 ≈ %.2e"
            % (name, fnum(L, "halfwidth_factor"), 1e3 * hw, 1e3 * fnum(L, "fine_depth"),
               area, estimate_cells(area, edge_t)))

    Lc = lay["C"]
    w, d = fnum(Lc, "crack_band_halfwidth"), fnum(Lc, "crack_band_depth")
    area_c = 2 * (2 * w) * d
    say("  布局 C（估算）：两个边缘带 ±a 各宽 2w = %.2f mm、深 %.2f mm  →  面积 %.2e m²，单元 ≈ %.2e"
        % (1e3 * 2 * w, 1e3 * d, area_c, estimate_cells(area_c, edge_t)))

    if "C" in cfg.get("generate", []):
        boxes = [(a_c - w, a_c + w, -d, 0.0), (-a_c - w, -a_c + w, -d, 0.0)]
        mesh, _ct, _ft = block_mesh(
            REPO / "data" / "clamp_mesh_C", width=fnum(geo, "width"), depth=D,
            refine_boxes=boxes, h_fine=h_fine, h_coarse=h_coarse, transition=trans, comm=comm)
        n = comm.allreduce(mesh.topology.index_map(mesh.topology.dim).size_local, op=MPI.SUM)
        from dolfinx import fem
        import ufl
        from dolfinx.mesh import compute_midpoints

        V0 = fem.functionspace(mesh, ("DG", 0))
        vec = fem.assemble_vector(fem.form(ufl.TestFunction(V0) * ufl.dx(domain=mesh)))
        nloc = mesh.topology.index_map(mesh.topology.dim).size_local
        area = np.asarray(vec.array[:nloc])
        mid = compute_midpoints(mesh, mesh.topology.dim, np.arange(nloc, dtype=np.int32))
        in_band = np.zeros(nloc, dtype=bool)
        for x0, x1, y0, y1 in boxes:
            in_band |= (mid[:, 0] >= x0) & (mid[:, 0] <= x1) & (mid[:, 1] >= y0) & (mid[:, 1] <= y1)
        edge_band = comm.allreduce(
            float(np.median(np.sqrt(area[in_band]))) if in_band.any() else float("nan"), op=MPI.MIN)
        say("")
        say("  布局 C（实际生成）：总单元 %d；加密带内单元边长中位数 %.3f µm（目标 %.3f µm，比值 %.2f）"
            % (n, 1e6 * edge_band, 1e6 * edge_t, edge_band / edge_t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
