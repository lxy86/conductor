"""pytest 全局配置：并行下禁用 Python 自动 GC。

## 为什么（D-68 问题 30 续 2，2026-10-23 实测）

`mpirun -n 6 python -m pytest tests/test_cyclejump.py` 会**死锁**
（rc=124、内存不再变化、零 MPI_Abort），而：
  - 该文件 13 个用例**逐个**并行跑，13/13 全过；
  - 串行跑整个文件也全过；
  - 必须连跑足够多的用例才挂。

py-spy 抓 6 个 rank 的栈，结论一致且明确：

```
rank 0 : __del__ (dolfinx/fem/petsc.py:884)   ← GC 在析构 PETSc 对象
         fillFormat (numpy/_core/arrayprint.py:1073)
rank 1..5: mpi_jit (dolfinx/jit.py:79)        ← 全部卡在 comm.bcast 等 rank 0
```

即：**Python 的自动 GC 在 rank 0 上触发，去析构 `dolfinx.fem.petsc` 的
`LinearProblem`/`NonlinearProblem`（其 `__del__` 调 PETSc `obj.destroy()`），
而其他 rank 正卡在 `mpi_jit` 的 `comm.bcast(status, root=0)` 等 rank 0
编译/响应。GC 的触发时机各 rank 独立，于是 rank 0 迟迟不回到 bcast，
其余 5 个 rank 干等 → 死锁。

串行不死锁是因为 `dolfinx/jit.py` 的 `mpi_jit` 开头有
`if comm.size == 1: return local_jit(...)`，**根本没有集合通信**。

## 修法

禁用自动 GC（`gc.disable()`）。引用计数驱动的即时释放仍然有效
（CPython 的主体回收路径），只是**周期性**的那一遍不跑了 ——
而那正是触发时机不一致的那一遍。

实测（`scripts/probe_cj_gc.sh`，A 段 7 个用例，mpirun -n 6）：

```
默认（自动 GC）  → rc=124 死锁（24 个点后停）
gc.disable()     → rc=0，42 个点全过
```

⚠ 这不掩盖任何真实失败：禁 GC 后 42 个点全是 `.`（通过），
若有用例真失败会照样报 F。
"""

import gc


def pytest_configure(config):
    # 只在并行下禁用（串行本来就没这个问题，保留默认行为便于调试）
    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.size > 1:
            gc.disable()
    except Exception:
        # mpi4py 不可用时不影响串行测试
        pass
