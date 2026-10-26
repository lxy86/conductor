# CLAUDE.md — 工作约定

## 项目
见 PLAN.md：FEniCSx 相场疲劳模型预测铝单丝微动疲劳寿命。只做 PLAN.md 中列出的内容，
按 Phase 顺序推进，每完成一个子任务在 PLAN.md 勾选。

## ⚠ 接手前先读
**`HANDOFF.md`**（2026-10-26，提交 `f187dfa`）—— 当前进度、换机器的环境重建步骤、
已证伪的假设清单、测试基线。**再读 `DECISIONS.md` 的 D-68 问题 33 续 13~20**
（当前工作的全部上下文）。

**一句话状态**：6.6.3 卡在「单体 Newton 在滑移态没有根」（格式缺陷，非调参可解）。
失败处理刚修好，现在会明确失败而非吐出 `alpha_bar = 3.29e31` 的假结果。
下一步是换 `F_q` 的表述或回交错，**等用户拍板**。

## 环境
WSL2 里的 conda 环境 `pf-wire`，dolfinx **0.11.0**（real PETSc）。
所有命令在此环境运行。

⚠ **发行版名与路径随机器变，不要照抄**。历史上两台机器：

| | 旧机器（2026-10 前） | 本机（2026-10-26 接手） |
|---|---|---|
| WSL 发行版 | `Ubuntu-26` | `Ubuntu` |
| 用户 / conda | `root` / `/root/miniforge3` | `code` / `/home/code/miniforge3` |
| 项目 ASCII 软链 | `/root/pf-wire` | `/home/code/pf-wire` → `/mnt/d/疲劳寿命` |

先确认再跑：

```bash
wsl -l -v                      # 看发行版真名（别假设 Ubuntu-26）
wsl -d <真名>
```

进入环境**不要再手写 `source .../conda.sh && cd /root/pf-wire`** ——
`scripts/*.sh` 已统一 source `scripts/_env.sh`，它按
`$PFWIRE_CONDA_BASE` → `conda info --base` → 常见安装位置三级定位 conda，
按 `$PFWIRE_HOME` → 本文件上两级目录定位仓库根。
手工会话里同样可以用：

```bash
source scripts/_env.sh && conda activate "${PFWIRE_ENV:-pf-wire}" && cd "$PFWIRE_HOME"
```

建/更新环境用 `bash scripts/setup_env.sh`，**不要**裸跑 `conda env create`
（adios4dolfinx 0.11 只在 PyPI 上，须 `pip install --no-deps`，见 DECISIONS.md D-0）。
机器（**随机器变，先 `lscpu` 确认再定 `-n`**）：

| | 原机器 | 本机（2026-10-26 接手实测） |
|---|---|---|
| CPU | Xeon E5-1650 v4 | Xeon Gold 5218R @ 2.10 GHz |
| 物理核 | 6 | **20**（`nproc` 报 40 = 超线程数） |
| 内存 | 47 GB | **31 GB** |

并行策略：**单工况用「一核一 rank」** 并设 `OMP_NUM_THREADS=1`
（纯 MPI 比 MPI+BLAS 线程混用更快、也不抢核）；sweep 阶段再按实测权衡
「单工况多进程」与「多工况并发」。**不设单工况耗时上限**（用户 2026-09-09 指示），见 D-4。

⚠️ **绝对不要按 `nproc` 的值设 `-n`**：`nproc` 是逻辑核（超线程数），
按它设就是超订物理核，MPICH 忙等下会塌方。
原机器（6 物理核）2026-09-10 实测（CT 疲劳算例，跑 5 个 jump 步）：
`-n 6` 21 s、`-n 2` 49 s、`-n 4` 70 s、**`-n 12` 约 700 s（慢 33 倍）**。
各进程数结果逐位相同，只是慢，所以**不会报错、只会白等**——这正是它难发现的地方。

⚠ **本机的 `-n` 还没测过。** 20 物理核下 `-n 6` 只用掉 30% 的机器，
合理起点是 `-n 20`，但**必须先跑一次 `-n` 扫描确认**（比墙钟），
不要直接把 6 或 20 写进配置。
本项目早期的长算例都误用了 `-n 12`，白白多花了几十倍时间（见 D-21）。

## 硬约束
1. dolfinx 锁定 **0.11.0**（用户 2026-09-09 改，原为 0.9.0）。不确定 API 先查源码/`help()`，
   不凭记忆写；禁止 legacy FEniCS 写法。已知 0.9→0.11 改动：`dolfinx.io.gmshio` 改名为
   `dolfinx.io.gmsh`，`LinearProblem`/`NonlinearProblem` 新增必需关键字 `petsc_options_prefix`，
   `NonlinearProblem` 本身已是 SNES 封装（旧 Newton 为 `NewtonSolverNonlinearProblem`）。
2. 所有公式必须来自 PLAN.md 或注释中给出文献出处，不自创公式。
   PLAN.md 与实际取值冲突时以 DECISIONS.md 为准（该文件记录所有修订与待确认项）。
3. 参数全部在 configs/*.yaml，代码不出现魔数；单位统一 SI。
   注意 PyYAML 走 YAML 1.1：`500e6` 这类不带指数符号的写法会被解析成**字符串**，
   配置载入必须 coerce 成 float（见 material.py 的 `_coerce`）。
4. 每个模块有 pytest；有解析解的（Hertz、Mindlin、P–S、CIGRE、悬臂梁）必须比对。
5. 长算例支持断点续跑（adios4dolfinx），sweep 脚本跳过已完成工况。
6. 数值结果异常（裂纹不在接触边缘、N_f 数量级不合理、假损伤）时停止并报告，
   不为"跑通"随意调参。
7. 不做 PLAN.md 明确排除的内容。

## 已知陷阱（踩过的，别再踩）
1. **BC 必须建在求解器自己的函数空间上**。在外部另建一个结构相同的空间做 DirichletBC，
   dolfinx 装配时不认，矩阵缺约束变奇异，而 MUMPS/preonly 仍返回 `KSPConvergedReason>0`，
   得到 1e8 量级的垃圾解。`PhaseFieldSolver` 已在构造时用 `FunctionSpace.contains` 拒绝；
   BC 一律写成 `bcs_u=lambda V: [...]` 的可调用形式。
   注意 `DirichletBC.function_space` 返回 cpp 对象，需用 `V._cpp_object.contains(...)`。
2. **PyYAML 是 YAML 1.1**：`500e6` / `1.0e8`（指数不带符号）会被解析成字符串。
3. **并行下不要写按 rank 局部的断言**：`facets.size > 0`、`x.array.size`（含 ghost）
   都依赖分区，要用 `allreduce` 或 `index_map.size_global`。
4. Amor/spectral 分解使 u 问题实际非线性；用 `u_solver="linear"` 时必须确认
   `sign_flips == 0`（χ 的不动点已收敛），否则解不满足真正的平衡方程。
5. **接触（dolfinx_contact）目前只能串行**：`mpirun -n 6` 下接触检测失败，压下后 P≈5e-14、
   上体穿过下体，**不报错**。接触验证脚本一律串行（约 15 分钟）。另外上游的 `create_*_mesh` /
   `convert_mesh` 内部自带 rank-0 守卫和 COMM_WORLD Barrier，**必须全体 rank 调用**，
   包进 `if rank == 0` 会死锁（D-51）。
   ⚠⚠ **`psi+` 的 DG-2 采样在并行下不可信**（D-68 问题 33 续 11，2026-10-25 实测）：
   同一 config、同一相位（`clamp_fretting_coarse_mu0`，phase 0.25）下 ——
   | 跑法 | `max|psi+|` | `max|p|` |
   | 串行 | 7.06e5（α_T 的 **4.0 倍**） | 1.44 σ_y |
   | `mpirun -n 6` | 1.02e10（α_T 的 **57933 倍**） | — |
   而 `u` 与 `δ` 同量级/逐位相同（`max|ux|` 4.74e-5 vs 7.10e-5、δ 逐位相同），
   故**不是**接触罚伪影、不是网格离散误差、不是 `u` 跑飞。三个并行探针
   都崩在 `state.sample_psi_plus()`（`fem.Expression` 插值到 DG-2）处
   （`MPI_Abort(59)`），串行下同一调用完全正常。
   **⇒ 6.6.3 的验收在串行结果上判**（`bash scripts/run_663_serial.sh`）。
   并行路径在此问题查清前不作为依据。
6. **并行下造临时目录，路径必须 rank 0 生成后 bcast**（D-68 问题 30 续）。
   `tempfile.mkdtemp()` / `TemporaryDirectory()` 每 rank 一个不同目录，而 adios4dolfinx
   的 `.bp` 写入是按 `COMM_WORLD` 的**集合操作** —— 路径不一致直接死锁（不报错、只白等）。
   `tests/test_cyclejump.py:241` 的 docstring 早就记过这个坑；改任何并行 checkpoint 代码前
   先看那里。正确写法：
   ```python
   path = MPI.COMM_WORLD.bcast(tmp_path / "x.bp", root=0)   # 或 rank 0 mkdtemp 后 bcast
   ```
   ⚠ 另注意 DG0 的 `x.array` 含 **ghost**：统计非零数/取 max 时必须先切 `[:size_local]`，
   否则同一单元被相邻 rank 重复计数（实测 6 rank 下 80 vs 76）。
7. **从 Windows 侧进 WSL 跑脚本，循环变量与路径都会被 Git Bash 吃掉**（2026-10-24）。
   两个已实测的形态：
   - `wsl -d Ubuntu-26 bash -lc 'for w in 0.5 0.3; do ...; done'` —— 外层 Git Bash 先把
     `$w` 展开成空，WSL 里拿到的是空串（于是 `float(sys.argv[1])` 收到 `''`/默认值，
     **静默跑错参数**，比报错更难发现）。
   - `wsl -d Ubuntu-26 bash /root/pf-wire/data/x.sh` —— Git Bash 的 MSYS 路径转换把
     `/root/...` 改写成 `C:/Program Files/Git/root/...`，报 No such file or directory。
   正确做法：脚本写到项目里（Windows 侧可见的那个 ASCII 软链），再用
   `wsl -d Ubuntu-26 bash -lc 'bash /root/pf-wire/data/_sweep_omega.sh'` 跑。

8. **不要往求解器方法里插代码做诊断 —— 插桩会污染被测对象**（D-68 问题 33 续 13）。
   2026-10-25 实证：给 `ClampContact.uzawa_update` 插打印（环境变量控制），
   报告 `mu.value=0.0`、`lim|max=0`、`contact=0`，我据此追了**五轮**假说
   （传参 bug / pycache 缓存 / 模块影子 / 补丁落点 / 两个实例），全部否掉。
   最后用**零插桩**探针（公开 API + 与源码逐行同源的公式在外部复算）
   拿到真相：`mu.value` 全程 0.7、`lim|max = 1.63e8`、`contact = 839`，
   且外部复算的 `q_new` 与真实 `uzawa_update` 的 `max|q|` **逐位一致**。
   ⇒ 插桩报告的一切都是假象。
   最可能机理：插桩代码里的 `import` / 文件 IO 触发 Python GC，而 PETSc 对象的
   `__del__` 与 `mpi_jit` 的 `comm.bcast` 不同步（与陷阱 6 的 GC 死锁同源，
   只是这版**不死锁、而是让被插桩的方法读到错值**，更难发现）。
   **正确做法**：定位数值问题用零插桩探针 ——
   调公开 API 拿到状态，再在**探针里**复算源码的公式并与真实调用结果对照。
   真要在源码里插，先用 `gc.disable()` 包住（`tests/conftest.py` 已有先例）。

9. **验证「失败处理」的探针本身不能被前一次失败污染**（2026-10-26）。
   一次失败会**就地改写** u/q，而「把数组抄回去」**恢复不了全部状态**
   （滑移基准、内部迭代量都在变）。本轮第一次跑验收探针就踩了：
   检查 (1) 故意触发失败后抄回 u/q 去跑检查 (3)，结果在相位 0 就失败，
   误报「failure_mode 没带相位」——差点以为修复没生效。
   **正确做法**：要么每次重建 case，要么把**最强的断言放最前面**、
   从干净状态起步。同类：陷阱 8 的「诊断工具扰动被测对象」。

10. **`.gitignore` 有 `data/_*.py` 与 `data/_*.log`**：要保留探针/日志必须
    `git add -f`。项目既有约定是「值得留的结论写进 DECISIONS.md，
    值得留的检查写成 tests/ 里的回归」——探针只是脚手架，
    但**证明某个结论的探针要入库**，否则换机器后无法复现。


## 工作流
- 每个子任务：先给出函数签名和测试意图 → 用户确认 → 实现 → `pytest -q` → 汇报
  （通过/失败、耗时、遗留问题）。
- PLAN.md 未覆盖的决策：列 2–3 个选项及利弊，选默认项并标"待确认"。
- 每个子任务完成后 git commit，message 写明子任务编号。

## 验收命令
- Phase 0: `pytest tests/test_env.py`（已通过；全量 `pytest -q` 26 项、约 16 s，1/2/4 进程均绿）
- Phase 1: `python -m pfwire.run_case configs/benchmark_sent.yaml`
           `python -m pfwire.run_case configs/benchmark_fatigue.yaml`
- Phase 2: `python -m pfwire.run_case configs/fretting_base.yaml`
           `python scripts/make_sweep.py && bash scripts/run_sweep.sh`
- Phase 3: `python -m pfwire.postproc --all`
- **6.6.3（当前卡点）**：`bash scripts/run_663_serial.sh`（**必须串行**，见陷阱 5）
- **回归基线**（2026-10-26 换机器重建后实测）：
  `pytest tests/ -q --ignore=tests/test_fretting.py --ignore=tests/test_fretting_restart.py --ignore=tests/test_clamp_case.py`
  → **6 failed**，全部在 `tests/test_clamp_friction.py`
  （`切向阶段不收敛（ux=1e-08，SNES reason=-6）`，**既有问题、非回归**，
  已用 `git stash` 撤掉改动重跑验证过；换机器后逐字复现）。
  ⚠ **判据只看失败名单与数量，不看收集数** —— 收集数随磁盘上不入库的
  `data/_*.sh` 脚手架数量浮动（实测 598~602，旧记的 632 是原机器快照）。
  **失败数超过 6 才是新回归。**
- 环境自检：`pytest tests/test_env.py -q`（8 项，含 PETSc real 变体与 abi3 断言）
- 失败处理专项：`pytest tests/test_solver_failure_surfaces.py -q`（7 项，秒级）
