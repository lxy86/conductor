"""按**实测单元尺寸**标出「损伤计算的可信区」掩码（D-68 路线 A，A-b）。

为什么按实测尺寸、不按 refine_boxes 的矩形盒：
问题 12 的判据本身就是 h/ℓ（实测 42~945），掩码口径要与它一致；
而 gmsh 的实际边长与设定值差 edge_ratio ≈ 0.45（D-58），盒边界也与真实
尺寸分布不完全吻合。故直接量每个 cell 的最大边长，h ≤ h_max 才计入。

为什么需要掩码（D-68 问题 12/13）：相场只在 h ≲ ℓ 处分辨得出裂纹带。
粗网格区 ψ⁺ 反推的等效应力达 451 MPa，超出 J2 屈服面上界 σ_y²/(6E)
**7.3 倍**（σ_y0 = 167 MPa）—— 在 J2 模型里不可能存在，是离散误差。
用它当起裂判据会把「不该起裂的状态」推进成 d 的棘轮发散。

用法::

    mask = damage_region_mask(msh, h_max=3.0 * ell)
    model = PhaseFieldModel(material=..., split="spectral", damage_region=mask)
"""
from __future__ import annotations

import numpy as np
from dolfinx import fem
from mpi4py import MPI


def cell_max_edge(msh) -> np.ndarray:
    """每个**本 rank 拥有的** cell 的最大边长，返回 shape=(n_cell_local,)。

    纯 numpy 从几何/拓扑算，不用 UFL/JIT —— `ufl.Circumradius(msh)` 在
    dolfinx 0.11 上 JIT 失败（实测，见 D-68 问题 12 的探针 28）。
    三角/四面体取 3(6) 条边；四边形再补两条对角线（避免退化成菱形时漏算）。
    """
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)
    ncell = msh.topology.index_map(tdim).size_local
    if ncell == 0:
        return np.zeros(0)

    offsets, verts = c2v.offsets, c2v.array
    xg = msh.geometry.x[:, : msh.geometry.dim]
    counts = np.diff(offsets[: ncell + 1])
    kmax = int(counts.max())

    # 向量化取顶点：逐 cell Python 循环在百万级单元上要几分钟，
    # 那是工具函数本身成为瓶颈（D-68 探针 28 踩过）。
    P = np.zeros((ncell, kmax, xg.shape[1]))
    mask = np.zeros((ncell, kmax), dtype=bool)
    for j in range(kmax):
        sel = counts > j
        P[sel, j, :] = xg[verts[offsets[:ncell][sel] + j]]
        mask[sel, j] = True

    best = np.zeros(ncell)
    # 相邻顶点对（首尾闭合）
    for i in range(kmax):
        j = (i + 1) % kmax
        sel = mask[:, i] & mask[:, j]
        if sel.any():
            dd = np.linalg.norm(P[sel, i, :] - P[sel, j, :], axis=1)
            best[sel] = np.maximum(best[sel], dd)
    if kmax >= 4:                       # 四边形：两条对角线
        for i, j in ((0, 2), (1, 3)):
            sel = mask[:, i] & mask[:, j]
            if sel.any():
                dd = np.linalg.norm(P[sel, i, :] - P[sel, j, :], axis=1)
                best[sel] = np.maximum(best[sel], dd)
    return best


def damage_region_mask(msh, h_max: float) -> fem.Function:
    """DG0 的 0/1 掩码：cell 的最大边长 ≤ ``h_max`` 记 1，否则 0。

    ``h_max`` 的取法：相场要分辨裂纹带，通常取 ℓ 的若干倍。
    D-68 问题 12 的实测是 h/ℓ = 42~945 处 ψ⁺ 非物理，而加密区 h ≈ 0.5 µm
    与 ℓ = 2 µm 同量级。本工况用 ``3ℓ = 6 µm``（既有加密盒 edge_target
    5 µm 仍在窗内，粗区 84 µm 以上被排除）。
    """
    if h_max <= 0.0:
        raise ValueError("h_max 必须为正，收到 %r" % (h_max,))
    V0 = fem.functionspace(msh, ("DG", 0))
    mask = fem.Function(V0, name="damage_region")
    h = cell_max_edge(msh)
    n = V0.dofmap.index_map.size_local
    k = min(n, h.size)
    mask.x.array[:k] = (h[:k] <= float(h_max)).astype(mask.x.array.dtype)
    mask.x.scatter_forward()
    return mask


def damage_region_stats(msh, h_max: float, comm: MPI.Comm) -> dict[str, float]:
    """掩码的覆盖率与尺寸分布，供日志/测试断言用（并行安全）。"""
    h = cell_max_edge(msh)
    n_loc = h.size
    n_in = int(np.count_nonzero(h <= float(h_max)))
    tot = comm.allreduce(n_loc, op=MPI.SUM)
    inside = comm.allreduce(n_in, op=MPI.SUM)
    hmin = comm.allreduce(float(np.min(h, initial=1e30)), op=MPI.MIN) if n_loc else 0.0
    hmax = comm.allreduce(float(np.max(h, initial=0.0)), op=MPI.MAX) if n_loc else 0.0
    return {
        "n_cell": float(tot),
        "n_in_region": float(inside),
        "fraction": (inside / tot) if tot else 0.0,
        "h_min": hmin,
        "h_max": hmax,
    }
