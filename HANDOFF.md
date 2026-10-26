# HANDOFF — 换电脑交接说明

> **写给完全没有上下文的新会话 / 新机器。**
> 上一版 HANDOFF 是 2026-10-19 的，已过时（那时还停在 6.6.1 的 Nitsche 阶段）。
> 本文件覆盖到 **2026-10-26**，对应提交 `43a8aab`。
>
> ⚠ **旧版引用的 `f187dfa` 在仓库里不存在**：`5251112`「公开前准备：历史脱敏」
> 重写过历史，脱敏前的哈希全部失效。**核对提交号一律以 `git log` 为准**，
> 不要相信文档里的旧哈希。
>
> **先读顺序**：本文件 → `CLAUDE.md`（工作约定与陷阱）→ `PLAN.md` 的 6.6 节
> → `DECISIONS.md` 的 **D-68 问题 33 续 13~22**（当前工作的全部上下文）。
> ⚠ **续 22 推翻了续 19/21 给 (b) 定的前提**（详见 §3.4 与 §5），
> 只读到续 20 会重复已证伪的路。

---

## 0. 三十秒摘要

**项目**：FEniCSx 相场疲劳模型，预测铝单丝微动疲劳寿命。
代码在 `src/pfwire/`，算例在 `configs/`，全部结论在 `DECISIONS.md`（8942 行）。

**当前卡点**：PLAN 的 **6.6.3 验收**（裂纹在钳口接触边缘萌生、初始倾斜、随后转向）。
延拓、W' 搜索、κ 敏感性**全部正常**；卡在**第一个开摩擦的循环子步**的
(u,q) 求解上。

**本轮刚做完的事（D-68 问题 33 续 21/22，2026-10-26）**：
1. **续 21**：续 19 的「滑移态符号矛盾」是**错的推论**。零插桩复算 `F_q` 表明
   它在滑移态**基本不看 s 的符号**（扫 s 全程相对变化仅 2.9e-4）。真因是
   `clip_η` **在盒内除 q=0 外处处不恒等**，边界处地板严格等于 `η/2`
   ⇒ 单体 `F_q` 在**所有**有摩擦单元上都有残差地板，不只滑移单元。
2. **续 22**：三条路线（η=0 半光滑 / 乘子 λ / 回交错）的**共同前提被推翻** ——
   它们都在改 `F_q`，但实测失败点在 **u 块**：
   - q 冻结为 0、只解 u，相位 0.05 仍 `SNES -5`（跑满 200 次、fnorm 停在 4763）；
     相位 0 只需 1 次迭代 ⇒ **失败与 q、与摩擦力无关**。
   - 第一层病因是**方向 f 遗留的假弹簧**：`_tang_jac_gate=1` 时 `-5` 跑满；
     置 0 后 `reason=4`、**4 次迭代**收敛。续 15 的形式判据至此得到数值实证。
   - 单体路径**不受污染**（`monolithic_jacobian` 的 `J_uu` 不含该项）⇒ 与续 21
     的 `F_q` 地板是**两个独立的病**。
   - 但 gate=0 跑完整 `explicit_cycle` 仍在 **Uzawa 第 2 轮**失败
     ⇒ 假弹簧只是第一层，第二层（续 13 的 blocker B 本体）仍在。
3. 已修（续 20，提交 `27b4429`）：单体未收敛**立即抛** `MonolithicNotConverged`，
   显式周期失败转成 `accepted=False` 的 report，`N` 不推进。**已完成、已测。**
   ⚠ 旧文档把这条记在 `f187dfa` 名下，那个哈希已因历史脱敏失效。

---

## 1. 换机器第一步：环境重建（最容易翻车的地方）

### 1.1 版本必须逐位复现

**实测**的版本组合（不是照抄 environment.yml，是实际 `import` 出来的）：

| 包 | 原机器 | 本机（2026-10-26 重建） | 硬 pin？ |
|---|---|---|---|
| dolfinx | 0.11.0 | 0.11.0 | **是**（硬约束 1） |
| basix | 0.11.0 | 0.11.0 | 是 |
| petsc4py | 3.25.5 | **3.25.6** | 否（`petsc=3.25.*=real_*`） |
| numpy | 2.5.3 | **2.5.4** | 否（`2.5.*`） |
| scipy | 1.18.1 | 1.18.1 | 否 |
| adios4dolfinx | 0.11.0 | 0.11.0 | 是（PyPI，`--no-deps`） |

`petsc4py` / `numpy` 的小版本漂移**不影响任何测试** ——
`tests/test_env.py` 只断言 `dolfinx.startswith("0.11")`、
`adios4dolfinx.startswith("0.11")` 和 **PETSc 是 real 变体**，
不 pin 具体小版本。重建环境时看到这两个漂移不用当失败。

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
wsl -l -v                            # 先确认发行版真名，别假设 Ubuntu-26
wsl -d <真名>
source scripts/_env.sh && conda activate "${PFWIRE_ENV:-pf-wire}" && cd "$PFWIRE_HOME"
pytest tests/test_env.py -q          # 必须全绿（含 PETSc real 变体断言）
```

### 1.5 机器与并行策略（新机器可能不同，务必重测）

**两台机器的硬件不同 —— `-n` 不要照搬。**

| | 原机器 | 本机（2026-10-26 接手实测） |
|---|---|---|
| CPU | Xeon E5-1650 v4 | **Xeon Gold 5218R @ 2.10 GHz** |
| 物理核 | 6 | **20**（`nproc` 报 40 = 超线程数） |
| 内存 | 47 GB | **31 GB** |
| 拓扑 | 1 socket | 1 socket / 20 core / 2 thread |

原机器结论（**只对 6 核成立，不能外推**）：

- **单工况用 `mpirun -n 6`**（一核一 rank），`OMP_NUM_THREADS=1`
- ⚠⚠ **绝对不要用 `-n 12`**：超订物理核，MPICH 忙等下**塌方**。
  2026-09-10 实测（CT 疲劳算例，5 个 jump 步）：`-n 6` 21 s、`-n 2` 49 s、
  `-n 4` 70 s、**`-n 12` 约 700 s（慢 33 倍）**。
  **各进程数结果逐位相同，只是慢 —— 不会报错、只会白等**，这是它难发现的地方。

> ⚠ **本机的 `-n` 尚未测过。** 20 物理核下 `-n 6` 只用掉 30% 的机器；
> 合理的起点是 `-n 20`（一核一 rank），但**必须重测扫描确认**再写进配置。
> 同理**绝不要超过 20**（40 逻辑核 = 超订，会重现上面的塌方）。
> 扫描脚本参考 D-21 的做法：同一 CT 疲劳算例跑 5 个 jump 步，比墙钟。

- ⚠ **接触（dolfinx_contact）目前只能串行**（CLAUDE.md 陷阱 5）。
  `mpirun -n 6` 下接触检测失败，压下后 `P ≈ 5e-14`、上体穿过下体，**不报错**。
  **6.6.3 的验收一律串行判**：`bash scripts/run_663_serial.sh`。
- ⚠⚠ **`psi+` 的 DG-2 采样在并行下不可信**（D-68 问题 33 续 11，2026-10-25）：
  同 config 同相位，串行 `max|psi+| = 7.06e5`（α_T 的 4.0 倍），
  `mpirun -n 6` 给出 `1.02e10`（**57933 倍**），而 u 与 δ 同量级/逐位相同。
  三个并行探针都崩在 `state.sample_psi_plus()`（`MPI_Abort(59)`）。

---

## 2. 把仓库搬过去（**当前没有 git remote，这是最大风险**）

### 2.1 现状

```
分支     master
remote   origin = https://github.com/lxy86/conductor.git   （2026-10-26 确认已同步）
```

**远程库已建好并同步**（截至 2026-10-26 为提交 `43a8aab`，含续 21/22）。
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
| `data/_*.py`（磁盘上 415 个一次性探针，**41 个已入库**） | 小 | ❌ 被 .gitignore | **可选**。证明结论的探针已 `git add -f` 入库（陷阱 10） |
| `data/_*.log`（大量） | 小 | ❌ 被 .gitignore | 同上 |
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

⚠ **上面是单体路径（续 18/20 的现场）。续 22 实测同一相位在
交错路径（`monolithic_friction: false`）下也失败，但失败点不同**：

| 路径 | 相位 0.05 的失败 |
|---|---|
| 单体 | `max_it`，50 次，`\|\|F\|\|`≈2.8~5e13（`F_q` 残差地板，续 21） |
| 交错 | u 的 `SNES -5`（跑满 200 次），**q 尚未更新** —— 假弹簧（续 22） |

⇒ 「两条路都卡在同一相位」曾经被当作同一病因的证据，**实际是两个独立的病**。

### 3.3 已**排除**的假设（都有实测，别再捡起来）

| 假设 | 判据 | 结果 |
|---|---|---|
| Jacobian 算错 | 粗网格整块有限差分，rel ∝ ε | **否证**：rel = 6.241e-6（ε=1e-6），纯截断误差 |
| η 光滑化造成残差地板 | `uzawa_eta_rel` 从 0.05 扫到 0.0005 | **否证**：残差 4.52e4 → 4.50e4，几乎不动 |
| 方向 f（补 ε_t·(τ·du)(τ·v)） | 零插桩探针 | **否证**：UFL 自动求导的切向刚度不是 0（占比 4.6e-9），我加的那项是它的 **9.5e7 倍**，是凭空加弹簧 |
| 并行特有的 ψ⁺ 采样问题 | 串行 vs `-n 6` 对照 | **已定位**，但不影响串行；6.6.3 走串行 |
| 插桩能诊断 | 给 `uzawa_update` 插打印 | **否证**：插桩报告 `mu=0/lim=0/contact=0` **全是假象**（GC 与 mpi_jit 不同步），追了五轮假说。见 CLAUDE.md 陷阱 8 |

### 3.4 真因（**已两次修正：续 19 → 续 21 → 续 22**）

⚠ **本节曾被改写过两次，只读续 19 会走错路。当前有效的是续 22。**

**（旧）续 19 的说法**：单体格式把 Coulomb 投影写成点态残差方程

```
F_q = q − clip_η(q − ε_t·s(u); μp, η) = 0
```

其不动点解析解为「黏着要求 s=0；滑移要求 s 与 q 反号」，而 s 由 u 决定、
u 又要与 q=±μp 的面力平衡 ⇒ **这个组合一般无解**。

**续 21 推翻了它**：零插桩复算（`data/_probe_fq_root.py`）表明，
在 `q = +μp` 处扫 s 从 −1e-5 到 +1e-5，`F_q` 只变 2.9e-4（相对）
⇒ **`F_q` 基本不看 s 的符号**。真因是 `clip_η` **在盒内除 q=0 外处处不恒等**
（`q/μp` = 0.5/0.9/0.99/1.0 时 `F_q` = 1.7e5/8.0e5/2.8e6/3.4e6），
边界处地板**严格等于 `η/2`**。⇒ 单体 `F_q` 在**所有**有摩擦单元上都有地板。

**续 22 再次转向：失败点根本不在 q 块。** 实测（`data/_probe_u_block.py`）：

| 条件 | SNES reason | 迭代 | fnorm |
|---|---|---|---|
| 相位 0.00（对照） | `4` | **1** | 2.9e-4 |
| 相位 0.05，**q 冻结为 0、只解 u** | **`-5`** | **200（跑满）** | **4763 不降** |

即**q 还没被碰过，u 自己就解不出来**。而第一层病因是
**方向 f 遗留的假弹簧**（`data/_probe_gate.py`）：

| `_tang_jac_gate` | SNES reason | 迭代 | fnorm |
|---|---|---|---|
| **1.0**（当前配置） | `-5` | **200（跑满）** | 4763 |
| **0.0**（关掉） | **`4`** | **4** | **2.94e-4** |

续 15 用**形式判据**证伪方向 f（`ε_t·(τ·du)(τ·v)` 在 `F` 里没有对应项，
量级是真导数的 **9.5e7 倍**），但**代码一直留在 `_build_u_problem` 里**，
只由 `_tang_jac_gate` 控制，开摩擦后门 = 1.0。

⚠ **假弹簧只是第一层**：`gate=0` 跑完整 `explicit_cycle`，仍在
**Uzawa 第 2 轮**（相位 0.05、q 已非零）失败。第二层 = 续 13 的 blocker B 本体，
**尚未定位**。

⚠ 单体路径**不受**假弹簧污染（`monolithic_jacobian` 的 `J_uu = derivative(F_u, u)`，
`F_u` 不含该项）⇒ 续 21 的 `F_q` 地板与交错的 `-5` 是**两个独立的病**。

⚠ **`-5` 的含义**：`DIVERGED_MAX_IT`（跑满迭代），**不是** `LINE_SEARCH`（`-6`）。
续 15 原文误记，已在 DECISIONS 就地更正。

### 3.5 已完成的修复（D-68 问题 33 续 20，提交 `27b4429`）

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

**判据是「6 failed，且全是下面这 6 个」，不是收集数。**

⚠ **收集数不可靠，别拿它对拍。** 本机 2026-10-26 实测是 **598~602**（同一台机器
连跑两次都不同），旧文档记的「632」是原机器某时刻的快照。原因：
`tests/test_repo_hygiene.py` 的 LF 行尾测试对**磁盘上的** `scripts/*.sh` **和
`data/*.sh`** 逐个 parametrize，而 `data/_*.sh` 是**不入库的一次性脚手架**
（`.gitignore` 排除，陷阱 10）—— 仓库里 4 个、磁盘上 112 个，差量全在这儿。

⇒ **只看失败名单与数量。失败数超过 6 才是新回归。**

那 6 项失败是**既有问题，不是回归**（已用 `git stash` 把三处源码改动撤掉后
重跑验证过，同样 6 项失败；2026-10-26 换机器重建环境后再次逐字复现）：

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

## 5. 下一步：**续 22 已推翻原来的三条路**，待用户拍板

⚠⚠ **读这一节之前先看 §3.4。** 下面的 (b)/(c) 是续 19 时代的选项，
它们共同的前提（「失败点在 q 块」）已被续 22 的实测推翻。
**保留原文是为了留下判断轨迹，不是建议直接做。**

### 5.0 续 22 之后的正确顺序

| 步 | 内容 | 状态 |
|---|---|---|
| 1 | **撤 / 默认关闭方向 f 的假弹簧**（`_tang_jac_gate` 默认 0，或拆掉 `tangential_jacobian` 的接线） | 待做。机理清楚（续 15 形式判据 + 续 22 实测），改动明确。⚠ 用户 2026-10-25 说过「方向 f 代码先留着不删、留作对照」——**保留代码、默认关门**与该指示兼容 |
| 2 | **重测第二层**：为什么 q 非零后 u 的 Newton 在相位 0.05 发散 | 待做。⚠ 续 13 的现场数据（it=4 时 843 单元全 `\|q\| = μp`、it=5 崩）是**假弹簧在场**时测的，必须重测 |
| 3 | 单体路线的 `F_q` 残差地板（续 21） | **降为第二优先**。先让交错能跑，才知道 6.6.3 的物理结果长什么样 |

⚠ **三条路（η=0 / 乘子 λ / 回交错）都不能单独解决 6.6.3。**

### （旧）(b) 换 `F_q` 的表述，让滑移态有根

把点态残差改成**互补条件**形式，使滑移态也有根。候选：

- Fischer–Burmeister 原始形式 `φ(a,b) = a + b − √(a²+b²) = 0`
  （即 `min(a,b)=0` 的光滑化），配 `a = μp − q`、`b = μp + q` 之类的盒约束
- 或直接写 `min(μp − q, max(−μp − q, ε_t·s))` 形式的互补残差

⚠ **硬约束 2：所有公式必须来自 PLAN.md 或注释中给出文献出处，不自创公式。**
`clamp_contact.py` 里已有 Fischer 1992 / Kanzow 1996 的引用
（见 `smooth_clip` 的 docstring），是现成的出发点。
**动手前先把出处补进注释，再写代码。**

### （旧）(c) 放弃单体，回到交错

交错格式在粗网格相位 0.05 之前是能跑的（tiny 网格 20 子步全绿）。
但 D-68 问题 33 **续 13** 记的 **blocker B**（深度黏着下 u 的 SNES -5）会回来。
那条路当时就是走不通才转向单体的。

⚠ **续 22 实测补充**：交错路径在相位 0.05 也失败（`data/_probe_eta_stag.py`），
但**第一处失败是假弹簧**（gate=0 后该处消失），第二层才是 blocker B。
所以「回交错」不是退回原点，而是**必须先关掉假弹簧再看**。

### （旧）(d) 还有一个未探索的角度

续 13 记的 blocker B 与续 19 的「没有根」**可能是同一件事的两面**：
交错在黏着态收敛慢/振荡，单体在滑移态没根。若能把**黏着态交给单体、
滑移态交给交错**（或按 `|q|/μp` 分单元切换），也许绕开两边的病。

⚠ **续 22 已否掉这个猜想的立论基础**：两者**不是同一件事的两面** ——
单体不受假弹簧污染，交错受；单体的病在 `F_q` 地板，交错的病在 u 块。
两个独立的病，按 `|q|/μp` 切换单元绕不开。保留原文作判断轨迹。

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
   `/root/...` 这类绝对路径会被 MSYS 改写成 `C:/Program Files/Git/root/...`。
   **正确做法**：脚本写到项目里（Windows 侧可见的那个目录），用
   `wsl -d <真名> bash -lc 'bash /mnt/d/疲劳寿命/data/_x.sh'` 跑
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

# 6. 全量基线（期望 6 failed，且失败全在 test_clamp_friction.py；收集数不看）
pytest tests/ -q --ignore=tests/test_fretting.py \
                --ignore=tests/test_fretting_restart.py \
                --ignore=tests/test_clamp_case.py

# 7. git 状态
git log --oneline -3        # 应看到 43a8aab（续 22）
git status                  # 应只有 §2.4 那几个未提交项
```

⚠ **不要拿文档里的旧哈希核对**：`5251112` 那次「历史脱敏」重写过历史，
旧文档里的 `f187dfa` **在仓库里根本不存在**（`git cat-file -t` 报 Not a valid object）。
一律以 `git log` 的实际输出为准。

**全部对上 = 衔接完成。** 然后回到 §5：**先看 §5.0**（续 22 之后的三步），
不是原来的 (b)/(c)。

### 7.0 本机（2026-10-26）已验收通过 ✅

| 项 | 结果 |
|---|---|
| `setup_env.sh` 全 5 步 | ✅ |
| dolfinx / basix | 0.11.0 / 0.11.0 |
| petsc4py / numpy | 3.25.6 / 2.5.4（原记 3.25.5 / 2.5.3，范围内） |
| PETSc scalar | `float64`（real） |
| `dolfinx_contact` | **`cpp.abi3.so`**（stable-ABI 生效） |
| `test_env.py` | **8 passed** |
| `test_solver_failure_surfaces.py` | **7 passed** |
| 全量基线 | **6 failed，名单与 §4 逐字一致** |

⇒ **衔接完成。** 6.6.3 仍未推进，等 (b)/(c) 拍板。

### 7.1 本机（2026-10-26）卡过的四个坑（都已解决，但换机器会再遇到）

**(1) WSL 里完全没有网络 —— 真因是 `.wslconfig` 把网络关了，不是 HNS 故障**

```
$ ip link show            →  只有 lo，没有 eth0
$ getent hosts github.com →  空
$ conda env create ...    →  NameResolutionError: conda.anaconda.org
```

Windows 宿主侧网络正常（`git ls-remote` 成功），只有 WSL 里没网。
一度以为是 HNS 虚拟交换机坏了，`Restart-Service hns` + `wsl --shutdown`
试了两轮**都无效**。真相是 `C:\Users\<用户名>\.wslconfig`：

```ini
[wsl2]
networkingMode=none      ← 2026-09-28 设的，等于「WSL 不要网络」
```

`networkingMode=none` 下 **eth0 从来就不会被创建**，所以任何网络排查
都看不到它。改成 `mirrored`（Win11）或删掉该行用默认 NAT（Win10）即可：

```ini
[wsl2]
networkingMode=mirrored
```

⚠ **Win10 (19045) 不支持 mirrored**，WSL 会自动回退到 NAT 并打印一句
`检测到 Windows 版本 19045... 回退到 NAT 模式` —— 这是正常提示，不是错误。
改完**必须 `wsl --shutdown`** 才生效（只重启服务不够，旧 VM 会继续跑旧配置）。

> 排查顺序建议：先看 `.wslconfig`，再看 `ip link`，最后才怀疑 HNS。
> `networkingMode=none` 是个安静的配置项，症状和「网卡坏了」一模一样。

**(2) `setup_env.sh` 自己也倒在 conda 定位上**

原第 23 行是 `source "$(conda info --base)/etc/profile.d/conda.sh"`。
非交互式 `bash -lc` 不走 `.bashrc`，conda 不在 PATH，于是它展开成
`source "/etc/profile.d/conda.sh"`，报 `conda: command not found`。
**已改成 source `scripts/_env.sh`**（与其余 25 个脚本一致）。

⚠ 更阴的一点：当时用 `wsl ... bash -lc 'bash setup_env.sh' > log 2>&1; echo EXIT=$?`
拿到 **exit 0** —— 因为 `echo` 成功了，脚本的真实失败被吞掉。
**看后台任务结果要读日志内容，不能只看退出码。**

**(3) `_env.sh` 依赖 `$HOME`**

`wsl -d X -- bash -c '...'` 这种非登录非交互调用里 `$HOME` 可能是**空的**，
于是 `${HOME}/miniforge3` 展开成 `/miniforge3`，兜底分支全部落空。
已在 `_env.sh` 里用 `getent passwd $(id -u)` 兜一层。

**(4) 旧 `build-contact/` 的 CMake 缓存**

`.ext/asimov-contact/build-contact/CMakeCache.txt` 里写的是**绝对路径**。
目录从旧机器拷过来后（cache 里记的是 `/root/pf-wire/...`），cmake 拒绝复用：

```
CMake Error: The current CMakeCache.txt directory /mnt/d/疲劳寿命/... is
different than the directory /root/pf-wire/... where it was created.
```

`.ext/` 被 `.gitignore`，`build-contact/` 又不在清理范围，**只要拷过目录就必踩**。
已在 `setup_env.sh` 里加自动检测（读 cache 里的 `# For build in directory:`，
与当前路径不符就删掉重建）。

**(5) 不要在 shell 脚本正在运行时编辑它**

同一天犯了两次。bash **按字节偏移**增量读正在执行的脚本，运行中改前面几行，
后续读取就错位 —— 表现是从一段 Python heredoc 中间开始解析，报
`line 86: syntax error near unexpected token '('`（那行本身完全合法）。

后果比报错更糟：conda 部分建好了、后面三步全没跑，而外层命令报 `EXIT=0`。
⇒ 要改脚本先 `TaskStop` + `pkill` 确认清干净，改完 `bash -n` 再重跑。
`bash -n` 对已损坏的运行没有诊断价值（它查磁盘当前版本，bash 读的是旧偏移片段）。

### 7.2 硬件与原机器不同（见 §1.5）

本机 **Xeon Gold 5218R / 20 物理核 / 31 GB**，不是 HANDOFF 早先记的
6 核 / 47 GB。`-n` 策略**必须重测**，详见 §1.5 的对照表。

---

## 8. 一句话给新会话

> 6.6.3 的失败点**不在 q 块，在 u 块**（续 22 推翻了「滑移态没有根」）。
> 交错路径第一层病因已定位：**方向 f 遗留的假弹簧** —— `_tang_jac_gate=1` 时
> u 的 Newton 跑满 200 次不降，置 0 后 **4 次迭代收敛**。但关掉它后完整周期
> 仍在 Uzawa 第 2 轮失败，**第二层（续 13 的 blocker B）尚未定位**。
> 单体路径不受该假弹簧污染，它的 `F_q` 残差地板（续 21）是**另一个独立的病**。
> 失败处理已修好（`27b4429`，旧文档误记为 `f187dfa`），现在会明确失败而不是
> 吐出 `alpha_bar = 3.29e31` 的假结果。
> **下一步：先撤/默认关闭假弹簧，再重测第二层，等用户拍板。
> 动手前先读 `DECISIONS.md` 的 D-68 问题 33 续 13~22（务必读到续 22）。**
>
> **2026-10-26 接手时**：环境没建起来（WSL 无网，见 §7.1），
> 代码与远程库都是好的。这轮做了「换机器路径自适应」：
> `scripts/*.sh` 不再硬编码 `/root`，统一 `scripts/_env.sh`。
