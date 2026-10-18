# HANDOFF — 6.5 线夹疲劳工况（D-68 问题 19：d ≡ 0 是正确的）

> 写给**完全没有上下文**的新会话。日期：2026-10-18。
> 上一会话的结论：**6.5 的 `max_d ≡ 0` 是方程组的正确解，不是 bug**。
> 卡点从「求解器不收敛」变成了「驱动差 0.06% 够不着门槛」——物理口径问题。

---

## 0. 先读这五行

- 代码树在 `C:\Users\Administrator\Desktop\疲劳寿命`（WSL `/root/pf-wire`）。
  ⚠️ 若环境显示工作目录是 `D:\导线疲劳` —— **那是陈旧副本**（git 停在 `e79372d`，
  没有任何 D-68 的工作）。**所有操作都在 Desktop 那个树里做。**
- 主分支 `master`；最新提交见 `git log --oneline -3`（本文档写作时为 `4b10155`，
  之后应已加上问题 19 的记录提交）。
- 全部结论、实测数字、被作废的假设都记在 `DECISIONS.md`（D-68 问题 1–19）。
  **动手前先读那一节**，否则很容易把已经证伪过的东西重新捡起来。
  **问题 19 是最新的一节，也是当前状态的唯一定论。**
- `pytest -q` 基线 **615 passed / 1 skip**（已核实，与 HANDOFF 记录一致）。
- 上一会话的探针 30–33 都在 `data/_probe_{r_at_n14352,r_masked,clusters_masked,
  cluster_profile}.py`，日志在 `data/probe_*.log`。

---

## 1. 我们在做什么任务

Phase 6.5：**线夹压紧铝单丝的微动疲劳寿命预测**（PLAN.md 的 6.5 节）。

模型链：AT1 相场断裂 + J2 弹塑性 + 罚接触 + Carrara 2020 疲劳累积
（`ᾱ += max(Δψ⁺, 0)`），用 cycle-jump 加速推进到起裂。

**验收判据**（`configs/clamp_base_c_1um.yaml` 的 `nucleation` 段）：
起裂 = `d ≥ 0.95` 且**从上表面连通贯穿到 6 µm**；
终止 = 裂纹走到 `turn_depth = 100 µm`。要报的是 `N_nuc`、`x_init`、`θ_init`。

**工况**：`configs/clamp_base_c_1um.yaml`
（W' = 1.75e6 N/m、σ_a = 45 MPa、T' = 1e4、ℓ = 2 µm、490529 单元、6 rank）

### 当前主故障的性质（**已变**）

| | 旧（HANDOFF v1） | 现在 |
|---|---|---|
| 现象 | N=94.28 那一 jump u–d 交错发散 | 发散早已消除（掩码），N 推到 14352 |
| 卡点 | d 的 SNES 报 -6 | SNES 解出的 d 就是 0，**解对了** |
| 停机原因 | ratcheting 判定 | 仍是 ratcheting，但**已不重要** |
| 真正的问题 | （未识别） | **驱动差 0.06% 够不着起裂门槛** |

---

## 2. 已经完成了什么

### 2.1 路线 A（损伤可信区掩码）—— 已实现并验证有效 ✅

**根因（问题 12/13）**：38 个「越过起裂门槛」的点几乎全在**粗网格区**
（`h/ℓ = 42~945`）。那里的 ψ⁺ 反推出等效应力 **451 MPa**，
是 σ_y0 = 167 MPa 的 2.7 倍 —— **超出 J2 屈服面上界 σ_y²/(6E) 达 7.3 倍**，
在 J2 模型里不可能存在，是离散误差。

**实现**（A-1 + A-b 口径）：

- `src/pfwire/mesh_h.py`：`cell_max_edge` / `damage_region_mask` /
  `damage_region_stats`
- `PhaseFieldModel.damage_region`：可选 DG0 掩码，**默认 None = 全域，逐位不变**
- `PhaseFieldSolver.psi_plus_drive()` = `psi_plus_expr() × 掩码`
- `_build_d_problem` 的驱动力改用 `psi_plus_drive()`
- `psi_plus_expr()` **保持未屏蔽**（诊断/ᾱ 累积仍看原始值）
- `PhaseFieldSolver.set_damage_region(mask)`：换掩码并重建 `_d_problem`
  （⚠ 掩码在装配时烘进 form，直接改 `model.damage_region` **静默无效**）
- 配置开关：`configs/clamp_base_c_1um.yaml` 的 `model.damage_region_h_max: 6.0e-06`
  （**不设此键 = 关闭**）

**验证结果（实测）**：

```
掩码覆盖率：480726 / 490529 cell（98.00%），阈值 1ℓ~5ℓ 只差 0.7 个百分点
掩码效果：  ∫drive / ∫ψ⁺ = 0.31%（切掉 99.7% 的驱动）
完整跑：    N_final 从 452.34 → 14352.39（31 倍），全程 max_d = 0，不再发散
```

**测试**：`tests/test_damage_region.py`（14 项）+ 全量 pytest 绿。

### 2.2 decay_jitter —— 修掉一处判据误判 ✅

`ShakedownConfig.decay_jitter`（默认 0 严格判据；本工况 0.01）。
只放宽「衰减」的**边界判定**，不放宽「不衰减」的定义 —— 真棘轮依然过不了。
修掉了问题 17 的误判（halving 序列末项超阈值 0.16%）。
**测试**：`tests/test_shakedown_jitter.py`（6 项，全绿）。

### 2.3 问题 19 的核查链（本会话所做）—— **这是当前状态的定论**

四个探针，从 N=14352.4 断点、**峰值相位 0.25** 逐层收窄：

| 探针 | 文件 | 量到什么 |
|---|---|---|
| 30 | `_probe_r_at_n14352.py` | 按坐标分盒：盒内 r_max = 1.81（⚠ 口径与掩码不一致，不可直接用） |
| 31 | `_probe_r_masked.py` | **按掩码分**：掩码内 r_max = 1.1441、214298 单元越门槛；直接解 d 得 0 |
| 32 | `_probe_clusters_masked.py` | 那 21 万单元 = **1 个连通团簇**，R_eq = 32.22 µm = 16.11 ℓ |
| 33 | `_probe_cluster_profile.py` | 团簇剖面 + **决定性对照**：塑性发展后 r_max = **0.9994** |

**问题 19 的最终结论：`max_d ≡ 0` 是正确的。**

塑性在第一周期就安定（`max|ε_p| = 1.2289e-2`，周期 2–6 逐位相同 —— 零增量，
与 D-66 的 `closed`/`decaying` 一致）。安定后掩码内
**r_max = 0.999373、中位 0.844、r > 1 的单元 0 个** —— 全场均匀地差
0.06% 够不着起裂门槛。d 方程的 VI 下界 `d_old = 0` 生效，d ≡ 0 正是解。

**⚠ 中途的误判（已撤回，记录以防重犯）**：探针 32 量到「16 ℓ 大连通团簇、
位置正确、形态完美」，我按问题 3 的判读标准报了「真 bug」。
**错因：没有先排除 `ep_n ≡ 0` 这个混淆源就去套判据。**
断点不存塑性历史（ε_p/p），而 `elastic_strain(u, g)` 读的是**已提交**的
`ep_n`（`plasticity.py:244-255`），于是探针 30–32 的 ψ⁺ 按**全量应变**
算、被高估 14%（1.1441 vs 0.9994）。

> **教训（重要）**：**任何用断点恢复现场量的 r，都必须先确认塑性历史是否在场。**
> 本项目已两次因「断点缺场」差点误判（本次、以及问题 17 的
> `plastic_increments` 口径）。判据本身没错，错在喂给它的数被污染了。

---

## 3. 当前卡在哪

**不是求解器问题，是物理口径问题：r_max = 0.9994，差 0.06%。**

```
塑性安定后掩码内：r_max = 0.999373   中位 = 0.844109   r>1 的单元 = 0
塑性：max|ε_p| = 1.228857e-02，第一周期即饱和，后续逐周期零增量
ᾱ_max = 4.2859e9，f(ᾱ_max) = 0.190，门槛 f·σ_c²/E = 9.4349e4 Pa
```

**HANDOFF v1 的路线 1/2/3 全部失去前提**：

- **路线 1**（d 的 SNES 为何在大 ΔN 下报 -6）：SNES 在起裂判定相位解出的 d
  就是 0，是对的；-6 只发生在外推步的 d 重解上，与「能否起裂」无关。
- **路线 2/3**（放宽 decay_ratio / 按 closed 处理）：棘轮判定只影响 N 能否
  继续涨，而 r 已饱和在 0.9994，再涨也不起裂。**顺序上应反过来**：
  先解决「驱动差 0.06%」，棘轮判定才有意义。

**N=14352.4 的棘轮停机**（增量比值 0.5035/0.5047/0.5156，均值 0.508 > 0.5）
仍是独立未解项，但**优先级低于驱动问题**。⚠ 不要再调大 `decay_jitter`
去盖 0.5156 —— 那是调参让跑通（硬约束 6 禁止），问题 18 已明确。

### 为什么「提高 σ_a 或 W'」不是默认答案

按 DECISIONS 2026-10-16 的论证：

- σ_a = 45 MPa 对应 Y_b = 1.45 mm，**已超 Lévesque 2011 实测区间上限 61%**
  （实测 Y_b ∈ [0.3, 0.9] mm → σ_a ∈ [9.3, 27.9] MPa）；
- W' = 1.75e6 对应 p ≈ 2.15σ_y = 359 MPa，**已在饱和压力上沿**
  （实测接触痕宽约 1 mm、p_avg ≈ 2.15σ_y）；
- 且 contact 侧 ψ⁺ 效率天然低：spectral 分解把三轴压缩的静水部分划给 ψ⁻，
  361 MPa 的钳口压力只贡献 ψ⁺ 门槛的 33%。

**提高任一输入都破坏与文献的可比性。**

---

## 4. 下一步计划（三条，用户尚未选定）

开工前先问用户，或按标记的倾向做。

### 路线 A：如实报「本工况不起裂」，并量化「差多少」（倾向）

r_max = 0.9994。ψ⁺ ∝ σ²、r ∝ ψ⁺，故需 ψ⁺ 提高 0.06%，
即 σ_a 或 p 提高约 **0.03%**。这个量级本身就是结论：
**该工况被设计在起裂门槛的边缘上，差一根头发。**

论文里这可以成为诚实且有信息量的结果（PLAN/D-43 的定位是
「报机理与相对敏感度，不报绝对寿命」），但必须同时报告：
r_max 的置信区间、以及它离门槛有多近。

### 路线 B：核查 ψ⁺ 的计算口径，看 0.06% 是不是口径/离散偏差

若 0.06% 来自某个可论证的离散误差（如 DG2 插值、求积阶、
`edge_ratio = 0.45` 那个实测换算），则「差一点」不是物理而是数值，
结论的强度会完全不同。要查的：

- ψ⁺ 的口径（spectral vs amor）在塑性安定态下的差别；
- 掩码阈值 3ℓ 的敏感性（1ℓ~5ℓ 只差 0.7 个百分点，已知不敏感）；
- 求积阶（`phasefield.py:739` 有塑性时钉死 `plastic.quad_degree = 4`）。

### 路线 C：换 σ_a 落到 Lévesque 区间的工况重跑，检验稳健性

若 σ_a = 20 MPa 时 r 显著更低（如 0.7），则「该构型不起裂」是**稳健结论**；
若仍贴 1.0，则说明驱动被某个与 σ_a 无关的项主导（最可能是钳口接触），
那 0.06% 的边缘性就是**结构性的**，结论反而更强。
⚠ 这一跑是数小时量级（101 min 跑到 N=14352），且要先解决棘轮停机
否则又是白跑 —— 故优先级最低。

---

## 5. 踩过的坑 —— 绝对不要再踩

### 5.1 MPI：不要写「按 rank 局部的判断」（**本项目第 4 次踩**）

```python
# ❌ 死锁。damage_region_stats 内部有 5 次 comm.allreduce，
#    只有 rank 0 进来 -> rank 0 永远等不到 rank 1..5
if comm.rank == 0:
    st = damage_region_stats(msh, h_max, comm)
    print(...)

# ✅ 统计所有 rank 都算，只有 print 留在 rank 0
st = damage_region_stats(msh, h_max, comm)
if comm.rank == 0:
    print(...)
```

**症状（很好认）**：100% CPU 空转、零输出、Python 的 SIGALRM 不触发
（阻塞在 MPI 原生代码）、各组件单独测都快。**完整跑卡在 build_clamp_case
>40 min 就是这么死的**（问题 15），而根因只是这两行。
历史踩点：探针 3、探针 26、探针 29、生产代码（clamp_case.py）。
**凡是「取最大 / 是否非空 / allreduce」一律先问是不是 rank 局部。**

### 5.2 **用断点恢复现场量的 r，必须先确认塑性历史在场**（问题 19 的新教训）

断点的 `checkpoint.FIELD_NAMES = ("u", "d", "d_old", "alpha_bar",
"psi_plus_prev")` **不含 ε_p/p**。于是 resume 后 `ep_n ≡ 0`，
而 `elastic_strain(u, g)` 读的是**已提交**的 `ep_n`
（`plasticity.py:244-255`）→ ψ⁺ 按全量应变算 → **r 被高估 14%**
（1.1441 vs 真实的 0.9994）。我因此误报了「真 bug」。

**正确做法**：要么先跑若干显式周期让塑性发展（探针 33 的做法，
`sol.commit_step()` 逐子步提交），要么明确在结论里标注
「本数在 ep_n ≡ 0 下量得，可能偏高」。

### 5.3 探针：解 d 之前必须先解一次 u

```
不解 u 直接解 d  -> SNES reason=+2, iters=0, max_d=0   ← 假的，空收敛
先解 u 再解 d    -> iters=1, max_d=0.0836             ← 真的
```

两次残差几乎相同（差 0.5%），**不是**「ψ⁺ 读旧值」。
真实推进（`solve_step`）总是 u/d 交替，不存在这条路径 ——
但**你写探针时一定会不小心走到这条路径**，然后误以为「掩码把 d 压成 0 了」。

### 5.4 探针：不要用清零当「重置状态」

把 `u` 也清零 → u 的 Newton 直接 `SNES -4`（自相矛盾的初值）。
**正确做法：保留 u，只清 d/d_old/ε_p/p**，然后重新解一次 u。

### 5.5 探针：`tabulate_dof_coordinates()` 与 DG2 的 dof 顺序**不对齐**

直接取会让 742104 个 dof 全落进 `|x| ≤ 3.5 mm`、且「1 mm 邻域」命中全部
本地 dof（两个都不可能）。**把 x[0]/x[1] 插值成场再取**，同序才有保证，
并加「坐标必须铺开到域宽」的自检断言。

### 5.6 r 的三种口径差 5 倍，必须选接近求解器的那个

```
A  ψ⁺ 投 DG2 节点 + numpy f  : ψ⁺max = 4.886e5, r_max = 3.434  ← 插值假象，别用
B  r 整条 UFL 投 DG0        : r_max = 0.668                    ← 接近求解器
C  ψ⁺/ᾱ 投 DG0 + numpy f    : r_max = 0.668（与 B 逐位一致）
```

d 方程在 `solver.dx`（`quadrature_degree = PlasticLaw.quad_degree = 4`）的
**求积点**上算，故 B/C 才是求解器看到的世界。**A 的 15 个 r>1 点是插值假象。**

### 5.7 配置：`raw.get("model", {})` 可能返回 None

YAML 里 `model:` 后面只有注释（没有实际键）会被解析成 **null**，
而 `dict.get` 的默认值只管**键不存在**的情形。已修成 `raw.get("model") or {}`。
同款隐患修过 `run_case.py:218`、`clamp_case.py:449/479`。

### 5.8 dolfinx 0.11：DG0 与 P1 的 basix family **都是 P**

区分 DG0 与 P1 **必须看 `basix_element.discontinuous`**，不能看 family
（`FiniteElement` 也没有 `family_name` 属性，会 AttributeError）。

### 5.9 环境：Bash 权限分类服务会长时间不可用

2026-10-17 中午起 `claude-sonnet-5[1m] is temporarily unavailable (timed out)`
持续约半小时，**连 `git commit` 都被挡**。简单命令（`echo`、`git add`）可用，
所以容易误以为是自己的命令写错。**等待 + 重试即可，不是代码问题。**

### 5.10 WSL 路径：`wsl -d Ubuntu-26 -- bash /root/...` 会被 Git Bash 改写

报 `bash: C:/Program Files/Git/root/pf-wire/...: No such file or directory`。
**用 `bash -lc '...'` 内联写法**，或写成脚本后用 `wsl -- bash -lc` 调。

### 5.11 输出缓冲：管道 + 重定向会吃掉几小时的日志

`... > log 2>&1 | tail` / `| grep` 都会块缓冲，跑几小时只留 3 行。
**直接 `> /root/pf-wire/data/xxx.log 2>&1`**，且脚本里 `PYTHONUNBUFFERED=1`。

---

## 6. 关键文件与入口

| 用途 | 文件 |
|---|---|
| 主工况配置 | `configs/clamp_base_c_1um.yaml` |
| 掩码实现 | `src/pfwire/mesh_h.py` |
| 求解器/掩码接入 | `src/pfwire/phasefield.py`（`psi_plus_drive`、`set_damage_region`） |
| 装配/配置读取 | `src/pfwire/clamp_case.py` |
| cycle-jump/棘轮判定 | `src/pfwire/cyclejump.py`（`ShakedownConfig`、`classify_plastic_history`） |
| 塑性（返回映射语义） | `src/pfwire/plasticity.py`（`update`/`commit`/`elastic_strain`，:244-255） |
| 掩码测试 | `tests/test_damage_region.py`（14 项） |
| jitter 测试 | `tests/test_shakedown_jitter.py`（6 项） |
| **所有结论与实测数字** | `DECISIONS.md`（D-68 问题 1–19） |
| 断点 | `data/clamp_base_c_1um/`（`ckpt_0.bp` / `ckpt_1.bp`，各 265 MB） |
| 运行日志 | `data/run_mask_jitter.log`（最后一次完整跑） |
| 问题 19 探针 | `data/_probe_{r_at_n14352,r_masked,clusters_masked,cluster_profile}.py` |
| 问题 19 日志 | `data/probe_{r_at_n14352,r_masked,clusters_masked,cluster_profile}.log` |

### 跑法

```bash
wsl -d Ubuntu-26 -- bash -lc 'source /root/miniforge3/etc/profile.d/conda.sh && conda activate pf-wire && cd /root/pf-wire && export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 && mpirun -n 6 python -u -m pfwire.run_clamp configs/clamp_base_c_1um.yaml --resume'
```

⚠️ **绝不用 `-n 12`**（超线程超订，实测慢 33 倍且不报错）。单工况一律 `-n 6`。

### 断点续跑须知

`--resume` 会读 `data/clamp_base_c_1um/checkpoint.json` 指向的 slot。
⚠ **断点里没有塑性历史 `ε_p`/`p`**（附带发现，见 DECISIONS.md）——
自己写探针时别假设能恢复它，**这是问题 19 误判的直接原因，见 §5.2**。

---

## 7. 开工前的 checklist

1. `git log --oneline -1` 确认在 Desktop 那个树里、且包含问题 19 的记录提交；
2. 读 `DECISIONS.md` 的 D-68 问题 **19**（当前定论）+ 问题 12/13（掩码依据）
   + 问题 17/18（棘轮判据的边界与真慢衰减）；
3. `pytest -q` 确认全绿（基线 615 项 + 1 skip）；
4. **问用户选路线 A / B / C**（§4，上一会话没等到答复）；
5. 写探针前重读 §5.1–5.6 —— 这六个坑每个都吃过几小时到几十小时。
