"""断点续跑（硬约束 5）。

**为什么网格必须从 .bp 里读回来，而不是用 gmsh 重建**：
adios4dolfinx 按网格的 *input* 全局编号存函数值。重建的网格编号顺序不同，
``read_function`` 照样"成功"返回，但值是乱的 —— 实测（12x9 矩形、P1、
解析场）逐点误差 1.995，量级与场本身相当，而**不抛任何异常**
（见 ``data/_ckpt_probe.py``）。从 .bp 读网格则精确：串行误差 0、
2 进程 2.22e-16，且写入与读取的进程数可以不同。

所以本模块的接口设计成「网格只能由 :func:`resume` 给出」：
:meth:`Resumed.read_fields` 会拒绝任何不是自己那张网格上的求解器。
这和项目里 BC 必须建在求解器自己的函数空间上（``FunctionSpace.contains``）
是同一类防呆 —— 靠纪律记不住，靠接口才拦得住。

**写入是两槽轮换的**：先把 .bp 写进空闲槽，屏障同步后再由 rank 0 原子替换
指针文件 ``checkpoint.json``。于是写到一半被杀（本项目这轮就 kill -9 过好几次）
只会毁掉空闲槽，上一个完好的断点仍然可用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mpi4py import MPI

#: 续跑需要的全部场。``d_old`` 不能漏 —— 它是 d 的 SNES 界约束下界，
#: 少了它续跑就允许损伤倒退，不可逆性被悄悄破坏。
FIELD_NAMES = ("u", "d", "d_old", "alpha_bar", "psi_plus_prev")

POINTER = "checkpoint.json"
SLOTS = ("ckpt_0.bp", "ckpt_1.bp")


def _fields(solver, state) -> dict[str, Any]:
    got = {
        "u": solver.u,
        "d": solver.d,
        "d_old": solver.d_old,
        "alpha_bar": state.alpha_bar,
        "psi_plus_prev": state.psi_plus_prev,
    }
    missing = [n for n in FIELD_NAMES if got.get(n) is None]
    if missing:
        raise ValueError("checkpoint 缺少场 %r" % (missing,))
    return got


def _read_pointer(directory: Path) -> dict[str, Any] | None:
    p = directory / POINTER
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def has_checkpoint(directory: str | Path, comm: MPI.Comm = MPI.COMM_WORLD) -> bool:
    """是否存在**可用**的断点（指针文件在，且它指向的 .bp 目录也在）。"""
    directory = Path(directory)
    ok = False
    if comm.rank == 0:
        ptr = _read_pointer(directory)
        ok = bool(ptr) and (directory / SLOTS[int(ptr["slot"])]).is_dir()
    return bool(comm.bcast(ok, root=0))


FACET_TAGS = "facet_tags"


def save(
    directory: str | Path,
    solver,
    state,
    meta: dict[str, Any],
    comm: MPI.Comm = MPI.COMM_WORLD,
    facet_tags=None,
) -> Path:
    """把网格、五个场与标量 ``meta`` 存成一个断点，返回写入的 .bp 路径。

    ``meta`` 必须是可 JSON 序列化的（N、ΔN、已走的 jump 步数、history 等）。
    ``facet_tags`` 给定时一并存下 —— 面力与边界条件都按它施加，
    而从 .bp 读回的网格**不带**物理组，不存就续不了跑。
    """
    import adios4dolfinx

    directory = Path(directory)
    if comm.rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    ptr = _read_pointer(directory) if comm.rank == 0 else None
    ptr = comm.bcast(ptr, root=0)
    # 写空闲槽：写坏了也毁不掉当前有效的那个
    slot = 1 - int(ptr["slot"]) if ptr else 0
    path = directory / SLOTS[slot]

    if comm.rank == 0 and path.exists():
        import shutil

        shutil.rmtree(path)
    comm.Barrier()

    adios4dolfinx.write_mesh(path, solver.mesh)
    for name in FIELD_NAMES:
        adios4dolfinx.write_function(path, _fields(solver, state)[name])
    if facet_tags is not None:
        adios4dolfinx.write_meshtags(
            path, solver.mesh, facet_tags, meshtag_name=FACET_TAGS
        )
    comm.Barrier()          # .bp 完整落盘后，才允许指针指过去

    if comm.rank == 0:
        tmp = directory / (POINTER + ".tmp")
        tmp.write_text(
            json.dumps({"slot": slot, "meta": meta}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(directory / POINTER)
    comm.Barrier()
    return path


@dataclass
class Resumed:
    """:func:`resume` 的返回值：网格 + 标量；场要等求解器建好后再读。"""

    mesh: Any
    meta: dict[str, Any]
    path: Path

    def read_fields(self, solver, state) -> None:
        """把五个场读回 ``solver``/``state``。

        **求解器必须建在 ``self.mesh`` 上**；否则 adios4dolfinx 会读出乱序的值
        而不报错（模块文档里那 1.995 的逐点误差就是这么来的）。
        """
        if solver.mesh is not self.mesh:
            raise ValueError(
                "求解器不是建在 checkpoint 读出来的那张网格上。"
                "重建的网格 input 编号顺序与存档不同，read_function 会"
                "**静默**读出乱序的值（实测逐点误差与场本身同量级）。"
                "请用 resume(...).mesh 建求解器。"
            )
        import adios4dolfinx

        fields = _fields(solver, state)
        for name in FIELD_NAMES:
            fn = fields[name]
            adios4dolfinx.read_function(self.path, fn)
            fn.x.scatter_forward()

    def read_facet_tags(self):
        """读回面标签。存档里没有就抛错 —— 静默返回 None 会让面力整个消失。"""
        import adios4dolfinx

        return adios4dolfinx.read_meshtags(
            self.path, self.mesh, meshtag_name=FACET_TAGS
        )


def resume(directory: str | Path, comm: MPI.Comm = MPI.COMM_WORLD) -> Resumed:
    """读回网格与标量 meta。场由 :meth:`Resumed.read_fields` 单独读。"""
    import adios4dolfinx

    directory = Path(directory)
    ptr = _read_pointer(directory) if comm.rank == 0 else None
    ptr = comm.bcast(ptr, root=0)
    if not ptr:
        raise FileNotFoundError("%s 下没有可用的断点（缺 %s）" % (directory, POINTER))
    path = directory / SLOTS[int(ptr["slot"])]
    mesh = adios4dolfinx.read_mesh(path, comm)
    return Resumed(mesh=mesh, meta=dict(ptr["meta"]), path=path)
