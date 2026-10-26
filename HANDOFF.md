# HANDOFF — 换电脑交接说明

> **写给完全没有上下文的新会话 / 新机器。**
> 上一版 HANDOFF 是 2026-10-19 的，已过时（那时还停在 6.6.1 的 Nitsche 阶段）。
> 本文件覆盖到 **2026-10-26**，对应提交 `f187dfa`。
>
> **先读顺序**：本文件 → `CLAUDE.md`（工作约定与陷阱）→ `PLAN.md` 的 6.6 节
> → `DECISIONS.md` 的 **D-68 问题 33 续 13~20**（当前工作的全部上下文）。

---

## 0. 三十秒摘要

**项目**：FEniCSx 相场疲劳模型，预测铝单丝微动疲劳寿命。
代码在 `src/pfwire/`，算例在 `configs/`，全部结论在 `DECISIONS.md`（8408 行）。

**当前卡点**：PLAN 的 **6.6.3 验收**（裂纹在钳口接触边缘萌生、初始倾斜、随后转向）。
延拓、W' 搜索、κ 敏感性**全部正常**；卡在**第一个开摩擦的循环子步**的
(u,q) 求解上。

**本轮刚做完的事（D-68 问题 33 续 19/20）**：
1. 查明粗网格 6.6.3 首跑给出 `max_alpha_bar = 3.29e31` 的真因 —— 不是损伤，
   是**单体 Newton 未收敛却被静默放过**，未收敛的 u（`max|u| = 6.1e7 m`）
   被拿去算 ψ⁺。
2. 已修：单体未收敛**立即抛** `MonolithicNotConverged`，显式周期失败转成
   `accepted=False` 的 report，`N` 不推进。**这条已完成、已测、已提交。**
3. 剩下的**真问题**：单体格式的 `F_q = q − clip(q − ε_t·s) = 0` 在**滑移态没有根**
   （解析：黏着要求 s=0，滑移要求 s 与 q 反号，而 s 由 u 决定、u 又要与 q=±μp 平衡）。
   这是格式缺陷，**不是调参能解决的**。三条路待用户拍板，见 §5。

---

## 1. 换机器第一步：环境重建（最容易翻车的地方）

### 1.1 版本必须逐位复现

这台机器上**实测**的版本组合（不是照抄 environment.yml，是实际 `import` 出来的）：

```
dolfinx        0.11.0
basix          0.11.0
petsc4py       3.25.5      （PETSc 必须是 real 变体，不是 complex）
numpy          2.5.3
scipy          1.18.1
adios4dolfinx  0.11.0
dolfinx_contact  <conda env>/lib/python3.12/site-packages/dolfinx_contact/
```

`environment.yml` 里 `fenics-dolfinx=0.11.0` / `petsc=3.25.*=real_*` 已 pin 好。
**dolfinx 版本是硬约束**（CLAUDE.md 硬约束 1）：0.9→0.11 有 API 改名
（`dolfinx.io.gmshio` → `dolfinx.io.gmsh`，`LinearProblem`/`NonlinearProblem`
新增必需关键字 `petsc_options_prefix`）。**不确定 API 先查源码 / `help()`，不凭记忆写。**

### 1.2 建环境**只能用** `scripts/setup_env.sh`

```bash
bash scripts/setup_env.sh
```

**不要裸跑 `conda env create`。** 两个包 conda-forge 上没有或版本不对：

| 包 | 问题 | 处理 |
|---|---|---|
| `adios4dolfinx` | conda-forge 最高 0.10.*，与 dolfinx 0.11 冲突；0.11 只在 PyPI | `pip install --no-deps --no-build-isolation adios4dolfinx==0.11.0` |
| `dolfinx_contact` | conda-forge 没有，只能源码编译 | 见下 |

`dolfinx_contact` 来自 `Wells-Group/asimov-contact`，**钉死在 commit
`f605fffdcabe470e6d20525c424aa985b36b3c42`**（不跟 main 漂，D-46）。
`setup_env.sh` 会自动 clone 到 `.ext/asimov-contact`（该目录在 .gitignore 里，
**不随仓库搬运**，必须由脚本重新拉）。

**这个库要打两个补丁才装得上**（D-48），`setup_env.sh` 已内置：

1. `find_package(Python ... Development.SABIModule ...)` —— 否则 nanobind 找不到 SABI
2. `nanobind_add_module(... STABLE_ABI ...)` —— 否则产物是 `cpp.cpython-312-*.so`

**为什么必须**：conda-forge 的 dolfinx 绑定是 **stable ABI**（`cpp.abi3.so`）。
两者 ABI 不同就不共享 nanobind 类型注册表，`ContactProblem` 构造时报
`incompatible function arguments ... dolfinx::mesh::MeshTags<int>`
（签名里打印裸 C++ 类型名 = 认不出对方类型）。

**装完必须验这一条**：

```bash
python -c 'import dolfinx_contact; print(dolfinx_contact.__file__)'
# 必须是 cpp.abi3.so；若是 cpp.cpython-312-*.so 则 stable-ABI 补丁没生效
```

### 1.3 WSL 侧要重建的 ASCII 软链接

项目路径含中文（旧机器 `C:\Users\Administrator\Desktop\疲劳寿命`，
本机 `D:\疲劳寿命`），**WSL 里一律走 ASCII 软链接**：

```bash
ln -sfn "/mnt/d/疲劳寿命" "$HOME/pf-wire"
```

⚠ **`scripts/*.sh` 已不再硬编码 `/root/pf-wire`**（2026-10-26 改）：
统一 source `scripts/_env.sh`，按 `$PFWIRE_HOME` / `conda info --base` /
常见安装位置三级定位。所以**软链不是必须的**，但保留它有两个好处：
路径短、且与历史文档一致。真正必须的是 `scripts/_env.sh` 能找到 conda。

若 conda 不在 `$HOME/miniforge3`，用环境变量显式指定：

```bash
export PFWIRE_CONDA_BASE=/path/to/conda
export PFWIRE_HOME=/mnt/d/疲劳寿命     # 想绕开软链时
```

### 1.4 验收环境

```bash
wsl -d Ubuntu-26
source /root/miniforge3/etc/profile.d/conda.sh && conda activate pf-wire
cd /root/pf-wire
pytest tests/test_env.py -q          # 必须全绿（含 PETSc real 变体断言）
```

### 1.5 机器与并行策略（新机器可能不同，务必重测）

原机器：Xeon E5-1650 v4，**6 物理核 / 12 线程**（`nproc` 报 12 是超线程数）/ 47 GB。

- **单工况用 `mpirun -n 6`**（一核一 rank），`OMP_NUM_THREADS=1`
- ⚠⚠ **绝对不要用 `-n 12`**：超订物理核，MPICH 忙等下**塌方**。
  2026-09-10 实测（CT 疲劳算例，5 个 jump 步）：`-n 6` 21 s、`-n 2` 49 s、
  `-n 4` 70 s、**`-n 12` 约 700 s（慢 33 倍）**。
  **各进程数结果逐位相同，只是慢 —— 不会报错、只会白等**，这是它难发现的地方。
- ⚠ **接触（dolfinx_contact）目前只能串行**（CLAUDE.md 陷阱 5）。
  `mpirun -n 6` 下接触检测失败，压下后 `P ≈ 5e-14`、上体穿过下体，**不报错**。
  **6.6.3 的验收一律串行判**：`bash scripts/run_663_serial.sh`。
- ⚠⚠ **`psi+` 的 DG-2 采样在并行下不可信**（D-68 问题 33 续 11，2026-10-25）：
  同 config 同相位，串行 `max|psi+| = 7.06e5`（α_T 的 4.0 倍），
  `mpirun -n 6` 给出 `1.02e10`（**57933 倍**），而 u 与 δ 同量级/逐位相同。
  三个并行探针都崩在 `state.sample_psi_plus()`（`MPI_Abort(59)`）。

> 新机器若核数不同，**重测一遍 `-n` 扫描再定策略**，不要照搬 6。

---

## 2. 把仓库搬过去（**当前没有 git remote，这是最大风险**）

### 2.1 现状

```
分支     master
remote   origin = https://github.com/lxy86/conductor.git   （2026-10-26 确认已同步）
```

**远程库已建好并同步**（提交 `5251112`，即「公开前准备」那次）。
旧版 HANDOFF 写的「无 remote、硬盘坏了全没了」**已过时**。

⚠ 远程还有一条 `main` 分支（`ceb38cd`），与 `master` 不同 ——
本项目的工作分支是 **`master`**，别误推到 `main`。

### 2.2 换机器做法

**远程已就绪，新机器直接 clone 即可**：

```bash
git clone https://github.com/lxy86/conductor.git
cd conductor
git checkout master          # 工作分支是 master，不是 main
```

`DECISIONS.md` 的全部历史都在里面。仓库里的 `scripts/`、`tests/`、
`configs/` 都是自包含的；只有 `.ext/asimov-contact` 需要 `setup_env.sh` 重新拉。

**如果没网或不想走远程**：直接拷整个目录（含 `.git`）。
⚠ 若走 U 盘/FAT32，注意中文文件名与 2 GB 单文件限制。
`data/**/*.bp/` 有 ~1.9 GB 且可重算，不必拷。

### 2.3 搬什么、不搬什么

| 内容 | 大小 | 是否在 git 里 | 建议 |
|---|---|---|---|
| `src/` `tests/` `configs/` `scripts/` `*.md` `*.yml` | ~5 MB | ✅ 全部跟踪 | 随仓库走 |
| `.git/` | 27 MB | — | 随仓库走 |
| `data/` 里**已跟踪**的部分 | 7.3 MB | ✅ | 随仓库走（含 result.json、探针、日志） |
| `data/**/*.bp/`（场数据） | ~1.9 GB | ❌ 被 .gitignore | **不用搬，可重算** |
| `data/_*.py`（247 个一次性探针） | 小 | ❌ 被 .gitignore | **可选**。本轮关键的 11 个已强制入库 |
| `data/_*.log`（83+ 个） | 小 | ❌ 被 .gitignore | 同上 |
| `.ext/asimov-contact` | — | ❌ 被 .gitignore | **必须由 `setup_env.sh` 重新拉** |
| `_probe_nitsche/`（4 个文件，24 KB） | 小 | ❌ 未跟踪 | 可选，Nitsche 预研残留 |

⚠ **本轮新增的 11 个探针 + 8 个日志已经 `git add -f` 入库**
（`.gitignore` 里有 `data/_*.py`，普通 `git add` 会被挡）。
换机器后**不需要**再去找它们：

```
data/_probe_ckpt.py        data/_probe_u_loc.py      data/_probe_cyc.py
data/_probe_coarse_cyc.py  data/_probe_fd_coarse.py  data/_probe_damp.py
data/_probe_step.py        data/_probe_armijo.py     data/_probe_eta.py
data/_probe_sign.py        data/_probe_fail_fast.py
+ data/probe_*.log 共 8 个
```

### 2.4 工作区里未提交的东西

**2026-10-26 接手时实测**（旧版 HANDOFF 记的 4 项里已少 2 项）：

```
M data/fretting_base/config.yaml    （max_iter 300→2000、加 initial_tilt_depth_over_a）
M data/fretting_base/result.json    （-8901 行，status 仍是 "running"）
```

`configs/clamp_base_c_coarse.yaml` 与 `_probe_nitsche/` **已不存在** ——
旧 HANDOFF 的那两行是过时记录，别再去找。

⚠ **`data/fretting_base/result.json` 这个改动仍未处理**：删了 8901 行、
`status` 是 `"running"`，是某次运行中途写的。**需要用户决定**
是提交还是 `git checkout --` 丢弃。它属于旧的 `fretting_base` 工况，
与当前 6.6.3 无关，不影响 6.6.3 的验收。

---

## 3. 当前技术状态（必读，否则会重复已证伪的路）

### 3.1 已经**正常**的部分

```
δ* 搜索     δ* = 2.72527e-05 m，W = 1.74844e6（目标 1.75e6，偏差 0.089%）
κ 敏感性    κ=10→1，接触反力变化 2.624%（阈值 3%）
```

这两条说明**法向接触链路是好的**，问题不在那儿。

### 3.2 卡点：6.6.3 的第一个开摩擦循环子步

用 `configs/clamp_fretting_coarse.yaml`（`monolithic_friction: true`），
粗网格 30480 单元，串行：

```
相位 0.00  ✅ 收敛（延拓末）
相位 0.05  ❌ 单体 Newton 50 次不收敛，max|q| = 5.8e13（= μp 的 1e5 倍）
```

### 3.3 已**排除**的假设（都有实测，别再捡起来）

| 假设 | 判据 | 结果 |
|---|---|---|
| Jacobian 算错 | 粗网格整块有限差分，rel ∝ ε | **否证**：rel = 6.241e-6（ε=1e-6），纯截断误差 |
| η 光滑化造成残差地板 | `uzawa_eta_rel` 从 0.05 扫到 0.0005 | **否证**：残差 4.52e4 → 4.50e4，几乎不动 |
| 方向 f（补 ε_t·(τ·du)(τ·v)） | 零插桩探针 | **否证**：UFL 自动求导的切向刚度不是 0（占比 4.6e-9），我加的那项是它的 **9.5e7 倍**，是凭空加弹簧 |
| 并行特有的 ψ⁺ 采样问题 | 串行 vs `-n 6` 对照 | **已定位**，但不影响串行；6.6.3 走串行 |
| 插桩能诊断 | 给 `uzawa_update` 插打印 | **否证**：插桩报告 `mu=0/lim=0/contact=0` **全是假象**（GC 与 mpi_jit 不同步），追了五轮假说。见 CLAUDE.md 陷阱 8 |

### 3.4 真因（D-68 问题 33 续 19）

单体格式把 Coulomb 投影写成点态残差方程：

```
F_q = q − clip_η(q − ε_t·s(u); μp, η) = 0
```

它的不动点条件 `q = Proj_{|q|≤μp}(q − ε_t·s)` 的**解析解**是：

| 状态 | 条件 |
|---|---|
| 黏着 `\|q\| < μp` | 要求 **s = 0** |
| 滑移 `q = +μp` | 要求 **s < 0** |
| 滑移 `q = −μp` | 要求 **s > 0** |

**滑移态要求 s 与 q 反号。** 而 `s = τ·(u − u_prev) − τ_x·x_clamp` 由 u 唯一决定，
u 又要与「q = ±μp」的面力平衡 —— **这个组合一般无解**，`F_q` 没有根。

实测证据（`data/probe_armijo.log`）：探针内实现 Armijo 回溯后，
步长能被接受（θ=2e-3），残差却**卡在 4.8e4 不动**，
同时 `max|q| → 1.358e8 ≈ μp`、`max|u| → 7.16e-5`（状态物理合理）。
而该状态 `uzawa_update` 的 `rel = 0.97` ⇒ **不是**交错格式的不动点。

⇒ **交错格式从不要求 `F_q = 0`**（它把残差当**更新量**），所以能工作；
单体格式把它当**误差**，所以在滑移态必然卡住。这是「交错能跑、单体不能」的机制。

### 3.5 已完成的修复（D-68 问题 33 续 20，提交 `f187dfa`）

- `phasefield.MonolithicNotConverged(RuntimeError)`：携带 `reason/iterations/residual/tol`
- `_solve_u_monolithic` 未收敛**立即抛**（原来只 warn，且 `log_iterations` 默认 False
  ⇒ 生产配置下**连 warning 都没有**）
- `cyclejump.ExplicitCycleFailed` + `step()` 转成 `accepted=False` 的 report
  （`N` 不推进、`rejections=0`、`failure_mode` 写明相位）
- 新增 `tests/test_solver_failure_surfaces.py`（7 项，秒级）

**效果**：同一个点位现在**立刻**以
`显式周期在相位 0.050 的 u 求解失败 —— reason=max_it，迭代 50 次，||F||=2.3e12`
停下，而不是吐出 `alpha_bar = 3.29e31` 的假结果。

---

## 4. 测试基线（换机器后用来对拍）

```bash
pytest tests/ -q --ignore=tests/test_fretting.py \
                --ignore=tests/test_fretting_restart.py \
                --ignore=tests/test_clamp_case.py
```

**收集 632 项，6 项失败，全部在 `tests/test_clamp_friction.py`。**

那 6 项失败是**既有问题，不是回归**（已用 `git stash` 把三处源码改动撤掉后
重跑验证过，同样 6 项失败）：

```
test_G1_tangent_with_friction_and_plastic
test_G3_G4_G5_coulomb_bounds_and_saturation
test_H1_H4_partial_slip_and_slip_zone_location
test_H2_H3_stick_halfwidth_shrinks_toward_mindlin
test_I2_kappa_t_sensitivity
test_I3_reaction_is_reproducible
```

失败信息统一是 `切向阶段不收敛（ux=1e-08，SNES reason=-6）` ——
**交错切向路径的既有问题**，与单体/失败处理无关。要不要修属于单独议题。

> 被 ignore 的三个文件是端到端算例级，很慢；需要时单独跑。

---

## 5. 下一步：三条路，**待用户拍板**

续 19 留下的三选一，**(a) 已完成**，剩 (b) / (c)：

### (b) 换 `F_q` 的表述，让滑移态有根 ⭐ 技术正解

把点态残差改成**互补条件**形式，使滑移态也有根。候选：

- Fischer–Burmeister 原始形式 `φ(a,b) = a + b − √(a²+b²) = 0`
  （即 `min(a,b)=0` 的光滑化），配 `a = μp − q`、`b = μp + q` 之类的盒约束
- 或直接写 `min(μp − q, max(−μp − q, ε_t·s))` 形式的互补残差

⚠ **硬约束 2：所有公式必须来自 PLAN.md 或注释中给出文献出处，不自创公式。**
`clamp_contact.py` 里已有 Fischer 1992 / Kanzow 1996 的引用
（见 `smooth_clip` 的 docstring），是现成的出发点。
**动手前先把出处补进注释，再写代码。**

### (c) 放弃单体，回到交错

交错格式在粗网格相位 0.05 之前是能跑的（tiny 网格 20 子步全绿）。
但 D-68 问题 33 **续 13** 记的 **blocker B**（深度黏着下 u 的 SNES -5）会回来。
那条路当时就是走不通才转向单体的。

### (d) 还有一个未探索的角度

续 13 记的 blocker B 与续 19 的「没有根」**可能是同一件事的两面**：
交错在黏着态收敛慢/振荡，单体在滑移态没根。若能把**黏着态交给单体、
滑移态交给交错**（或按 `|q|/μp` 分单元切换），也许绕开两边的病。
⚠ 这只是猜想，**没有实测支撑**，列在这里是防止它被忘掉，不是建议直接做。

---

## 6. 工作约定（`CLAUDE.md` 的要点，务必遵守）

### 流程

> **每个子任务**：先给出函数签名和测试意图 → 用户确认 → 实现 → `pytest -q`
> → 汇报（通过/失败、耗时、遗留问题）。每个子任务完成后 git commit，
> message 写明子任务编号。

### 硬约束（节选，全文见 CLAUDE.md）

1. dolfinx 锁 **0.11.0**，不确定 API 先查源码，禁止 legacy FEniCS 写法
2. **公式必须来自 PLAN.md 或有文献出处，不自创**
3. 参数全在 `configs/*.yaml`，代码不出现魔数；单位统一 SI
4. 每个模块有 pytest；有解析解的（Hertz/Mindlin/P–S/CIGRE/悬臂梁）必须比对
5. 长算例支持断点续跑，sweep 跳过已完成工况
6. **数值结果异常时停止并报告，不为「跑通」随意调参** ← 本轮严格遵守了这条
7. 不做 PLAN.md 明确排除的内容

### 最容易踩的坑（全文见 CLAUDE.md「已知陷阱」）

1. **BC 必须建在求解器自己的函数空间上**（否则矩阵缺约束变奇异，
   而 MUMPS 仍返回 `KSPConvergedReason>0`，得到 1e8 量级的垃圾解）
2. **PyYAML 是 YAML 1.1**：`500e6` / `1.0e8`（指数不带符号）会被解析成**字符串**
3. **并行下不要写按 rank 局部的断言**（`facets.size > 0`、`x.array.size` 都依赖分区）
4. **Amor/spectral 分解使 u 问题实际非线性**；用 `u_solver="linear"` 时必须确认
   `sign_flips == 0`
5. **接触只能串行**；⚠ **`psi+` 的 DG-2 采样在并行下不可信**
6. **并行下造临时目录，路径必须 rank 0 生成后 bcast**（否则 adios4dolfinx 死锁，
   不报错只白等）。DG0 的 `x.array` 含 ghost，统计前要切 `[:size_local]`
7. **从 Windows 侧进 WSL 跑脚本，循环变量与路径会被 Git Bash 吃掉**：
   `wsl ... bash -lc 'for w in ...; do ...'` 里 `$w` 会先被外层展开成空；
   `/root/...` 会被 MSYS 改写成 `C:/Program Files/Git/root/...`。
   **正确做法**：脚本写到项目里，用
   `wsl -d Ubuntu-26 bash -lc 'bash /root/pf-wire/data/_x.sh'` 跑
8. ⚠⚠ **不要往求解器方法里插代码做诊断 —— 插桩会污染被测对象**。
   2026-10-25 实证：给 `uzawa_update` 插打印，报告 `mu=0/lim=0/contact=0`
   **全是假象**，据此追了五轮假说全错。**定位数值问题用零插桩探针**
   （调公开 API 拿状态，在**探针里**复算源码公式并与真实调用结果对照）。
   真要在源码里插，先用 `gc.disable()` 包住（`tests/conftest.py` 有先例）。

### 本轮补充的两条经验

9. **验证「失败处理」的探针本身不能被前一次失败污染**。
   一次失败会就地改写 u/q，**抄回数组恢复不了全部状态**。
   要么每次重建 case，要么把最强的断言放最前面、从干净状态起步。
   （本轮第一次跑验收探针就踩了：误报「failure_mode 没带相位」。）

10. **`.gitignore` 里有 `data/_*.py`**。要保留探针必须 `git add -f`。
    「值得留的结论写进 DECISIONS.md，值得留的检查写成 tests/ 里的回归」——
    这是项目既有约定，探针只是脚手架。

---

## 7. 换机器后的验收清单

按顺序做完，确认「无缝衔接」：

```bash
# 1. 环境（发行版名先确认，别假设 Ubuntu-26；见 §1.3）
wsl -l -v
wsl -d <真名>
source scripts/_env.sh && conda activate "${PFWIRE_ENV:-pf-wire}" && cd "$PFWIRE_HOME"

# 2. 版本对拍（必须与 §1.1 一致）
python -c "import dolfinx, petsc4py, numpy, scipy, adios4dolfinx, basix
print('dolfinx', dolfinx.__version__); print('petsc4py', petsc4py.__version__)
print('numpy', numpy.__version__); print('scipy', scipy.__version__)
print('adios4dolfinx', adios4dolfinx.__version__); print('basix', basix.__version__)"

# 3. dolfinx_contact 必须是 abi3
python -c "import dolfinx_contact; print(dolfinx_contact.__file__)"

# 4. 环境自检
pytest tests/test_env.py -q

# 5. 本轮新增的失败处理测试
pytest tests/test_solver_failure_surfaces.py -q     # 7 passed

# 6. 全量基线（期望 632 收集 / 6 failed，且失败全在 test_clamp_friction.py）
pytest tests/ -q --ignore=tests/test_fretting.py \
                --ignore=tests/test_fretting_restart.py \
                --ignore=tests/test_clamp_case.py

# 7. git 状态
git log --oneline -3        # 应看到 5251112 起（含 f187dfa / 27b4429）
git status                  # 应只有 §2.4 那几个未提交项
```

**全部对上 = 衔接完成。** 然后回到 §5，等用户决定 (b) 还是 (c)。

### 7.1 本机（2026-10-26）实际卡在哪

接手时环境**没建起来**，原因不是版本冲突，是 **WSL 的网络栈坏了**：

```
$ ip link show          →  只有 lo，没有 eth0
$ getent hosts github.com →  空
$ conda env create ...  →  NameResolutionError: conda.anaconda.org
```

Windows 宿主侧网络正常（`git ls-remote origin` 成功、`git clone` 可行），
**只有 WSL 里没网**。`wsl --shutdown` + 重启试过两次，`eth0` 仍不出现。
这是 WSL 宿主的 HNS 虚拟交换机问题，**需要管理员权限**修：

```powershell
# 管理员 PowerShell
wsl --shutdown
Get-NetAdapter | Where-Object Name -like "*WSL*" | Restart-NetAdapter
# 或：重启 Hyper-V 主机计算服务
Restart-Service hns
```

**在 eth0 恢复前，本清单第 2~6 步全部做不了**（装不了包、跑不了 pytest）。
第 1 步与第 7 步可以做。

---

## 8. 一句话给新会话

> 6.6.3 卡在「单体 Newton 在滑移态没有根」，已排除 Jacobian / η / 并行 / 方向 f
> 四个方向（都有实测）。失败处理已修好（`f187dfa`），现在会明确失败而不是
> 吐出 `alpha_bar = 3.29e31` 的假结果。**下一步是换 `F_q` 的表述或回交错，
> 等用户拍板。动手前先读 `DECISIONS.md` 的 D-68 问题 33 续 13~20。**
>
> **2026-10-26 接手时**：环境没建起来（WSL 无网，见 §7.1），
> 代码与远程库都是好的。这轮做了「换机器路径自适应」：
> `scripts/*.sh` 不再硬编码 `/root`，统一 `scripts/_env.sh`。
