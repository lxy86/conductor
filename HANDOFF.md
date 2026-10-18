# HANDOFF — 6.5 线夹疲劳工况（D-68 排查链）

> 写给**完全没有上下文**的新会话。日期：2026-10-18。
> 上一会话在「D-68 问题 18」处停下，完整跑跑到 N=14352 后棘轮停机。

---

## 0. 先读这三行

- 代码树在 `C:\Users\Administrator\Desktop\疲劳寿命`（WSL `/root/pf-wire`）。
  ⚠️ 若环境显示工作目录是 `D:\导线疲劳` —— **那是陈旧副本**（git 停在 `e79372d`，
  没有任何 D-68 的工作）。**所有操作都在 Desktop 那个树里做。**
- 主分支 `master`；最新提交 `cd9bee1`（D-68 问题 18）。
- 全部结论、实测数字、被作废的假设都记在 `DECISIONS.md`（D-68 问题 1–18）。
  **动手前先读那一节**，否则很容易把已经证伪过的东西重新捡起来。

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

### 这个会话要解决的「主故障」

工况跑不远：原来在 **N=94.28 那一 jump 里 u–d 交错发散**（`max_d` 五轮
0.084→0.227→0.451→0.748→1.0，然后 d 的 SNES 报 −6），整条链子断在这里，
拿不到 `N_f`。

---

## 2. 已经完成了什么

### 2.1 路线 A（损伤可信区掩码）—— 已实现并验证有效

**根因（问题 12/13）**：38 个「越过起裂门槛」的点几乎全在**粗网格区**
（`h/ℓ = 42~945`）。那里的 ψ⁺ 反推出等效应力 **451 MPa**，
是 σ_y0 = 167 MPa 的 2.7 倍 —— **超出 J2 屈服面上界 σ_y²/(6E) 达 7.3 倍**，
在 J2 模型里不可能存在，是离散误差。加密区（`|x| ≤ 3.5 mm`）的真实驱动
`r_max` 只有 **1.0045**，只是勉强贴着门槛。

**实现**（A-1 + A-b 口径）：

- `src/pfwire/mesh_h.py`（新）：`cell_max_edge` / `damage_region_mask` /
  `damage_region_stats`
- `PhaseFieldModel.damage_region`：可选 DG0 掩码，**默认 None = 全域，逐位不变**
- `PhaseFieldSolver.psi_plus_drive()` = `psi_plus_expr() × 掩码`
- `_build_d_problem` 的驱动力改用 `psi_plus_drive()`
- `psi_plus_expr()` **保持未屏蔽**（诊断/ᾱ 累积仍看原始值）
- `PhaseFieldSolver.set_damage_region(mask)`：换掩码并重建 `_d_problem`
  （⚠ 掩码在装配时烘进 form，直接改 `model.damage_region` **静默无效**）
- 配置开关：`configs/clamp_base_c_1um.yaml` 的 `model.damage_region_h_max: 6.0e-06`
  （**不设此键 = 关闭**）

**A-1 口径**：只屏蔽 d 方程的驱动力，`ᾱ` 照旧累积 —— 语义是
「这里分辨不了，不判起裂」，**不是**「这里没有疲劳」。

**A-b 掩码**：按**实测单元尺寸**判定，不是按 `refine_boxes` 的矩形盒。

**验证结果（实测）**：

```
掩码覆盖率：480796 / 490529 cell（98.02%），阈值 1ℓ~5ℓ 只差 0.7 个百分点
掩码效果：  ∫drive / ∫ψ⁺ = 0.31%（切掉 99.7% 的驱动）
完整跑：    N_final 从 452.34 → 14352.39（31 倍），全程 max_d = 0，不再发散
W' 搜索：   δ* = 2.71863e-05，与参考跑**逐位相同**（证明没动力学解）
```

**测试**：`tests/test_damage_region.py`（14 项）+ 全量 pytest 绿。

### 2.2 decay_jitter —— 修掉一处判据误判

**根因（问题 17）**：N=452 处，塑性增量序列 `[5.156e-08, 2.527e-08,
1.263e-08, 6.325e-09]` 比值 `0.490 / 0.500 / 0.501`，是真 halving，
但第三项超 `decay_ratio=0.5` 阈值 **0.16%**，被误判成 not_decaying → 棘轮停机。

**修法**：`ShakedownConfig.decay_jitter`（默认 0 严格判据；本工况 0.01）。
只放宽「衰减」的**边界判定**，不放宽「不衰减」的定义 —— 真棘轮依然过不了。

**测试**：`tests/test_shakedown_jitter.py`（6 项，全绿）。

---

## 3. 当前卡在哪

### 卡点：N=14352.4 处棘轮停机，且 ΔN 被拒绝三次

```
status: ratcheting    N_final = 14352.39    n_jumps = 13
timing.solve = 6038 s（101 min）
每个 jump：max_d = 0.0000，深度 = 0 µm（未起裂）
```

**两件事叠在一起**：

**(a) 塑性增量是「真慢衰减」，不是抖动（问题 18）**

```
N=14352.4 增量 [3.176e-09, 1.599e-09, 8.070e-10, 4.161e-10]
逐步比值 = 0.5035, 0.5047, 0.5156     ← 全部 > 0.5，均值 0.508
```

对比问题 17 那组（均值 0.497）：这组**系统性**大于 0.5，最后一步 0.5156
连 1% jitter（上限 0.505）都盖不住。按 `decay_ratio=0.5` 判 not_decaying
**是判据本身的要求，不是误判**。⚠ **不要再调大 jitter —— 那就是调参让跑通，
硬约束 6 禁止。**

**(b) d 的 SNES 在大 ΔN 下报 −6，这是拦住外推的直接原因**

```
拒绝 ΔN=3325.26（d 的 SNES -6，残差 1.656e-07，45/200 次迭代）
拒绝 ΔN=2216.84（d 的 SNES -6，残差 1.809e-07，35/200 次迭代）
拒绝 ΔN=1477.89（d 的 SNES -6，残差 1.603e-07，52/200 次迭代）
→ ΔN 降到 dN_min 附近仍失败 → 报棘轮
```

残差 **1.6e-07 只比 atol=1.0e-08 大 16 倍**却报 −6，迭代 35~52 次。
与 `SolverConfig.snes_rtol_u` 注释里描述的症状一致：**d 的容差压到了
装配舍入地板**。不是迭代次数不够。

### 另一个尚未解决的保留

可信区 `r_max` 只有 **1.0045**（勉强贴着门槛）。掩码让它不被推爆了，
但**能不能真正起裂还没验证过** —— `max_d` 至今恒 0。
这与「掩码切掉假驱动后真实驱动就是弱的」自洽，但不是预期结果。

---

## 4. 下一步计划（三条，用户尚未选定）

上一会话结束时给了用户三个选项，**用户还没回答**。开工前先问，或按倾向做。

### 路线 1（上一会话的倾向）：查 d 的 SNES 为何在大 ΔN 下报 −6

这是拦住外推的**直接原因**。要查的：

- 残差 1.6e-07 贴 atol=1.0e-08（16 倍）就报 −6，是不是舍入地板？
  → 对照 `SolverConfig.snes_rtol_u` 的长注释（那里记录过同类症状与解法）
- 迭代 35~52 次就 −6，是否与 `vinewtonrsls` 的 line search 有关？
- 大 ΔN 下 `ᾱ` 外推量大、`f(ᾱ)` 变小、驱动力变强 —— d 问题是不是变陡了？

**探针入口**：仿 `data/_probe_dsnes.py` / `data/_probe_d_converge.py` 的写法，
在 N=14352 的断点上单独解 d，打残差历史。**注意 `iters=0` 陷阱，见 §5。**

### 路线 2：decay_ratio 从 0.5 放宽到 0.6

⚠ **需要 PLAN §2.4 的物理依据**，否则就是调参（硬约束 6）。别直接改。

### 路线 3：接受「真安定但衰减慢」，按 closed 处理

需要论证「比值 0.508 算不算安定」。

### 顺带：起裂能否发生

可信区 `r_max = 1.0045`。若 ΔN 能继续长大、ᾱ 继续累积，驱动可能越过门槛。
这一条要等路线 1 解决后才能观察。

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

### 5.2 探针：解 d 之前必须先解一次 u

```
不解 u 直接解 d  -> SNES reason=+2, iters=0, max_d=0   ← 假的，空收敛
先解 u 再解 d    -> iters=1, max_d=0.0836             ← 真的（复现问题 9）
```

两次残差几乎相同（1.929 vs 1.920，差 0.5%），**不是**「ψ⁺ 读旧值」。
真实推进（`solve_step`）总是 u/d 交替，不存在这条路径 ——
但**你写探针时一定会不小心走到这条路径**，然后误以为「掩码把 d 压成 0 了」。

### 5.3 探针：不要用清零当「重置状态」

把 `u` 也清零 → u 的 Newton 直接 `SNES -4`（自相矛盾的初值）。
**正确做法：保留 u，只清 d/d_old/ε_p/p**，然后重新解一次 u。

### 5.4 探针：`tabulate_dof_coordinates()` 与 DG2 的 dof 顺序**不对齐**

直接取会让 742104 个 dof 全落进 `|x| ≤ 3.5 mm`、且「1 mm 邻域」命中全部
本地 dof（两个都不可能）。**把 x[0]/x[1] 插值成场再取**，同序才有保证，
并加「坐标必须铺开到域宽」的自检断言。

### 5.5 配置：`raw.get("model", {})` 可能返回 None

YAML 里 `model:` 后面只有注释（没有实际键）会被解析成 **null**，
而 `dict.get` 的默认值只管**键不存在**的情形。已修成 `raw.get("model") or {}`。
同款隐患修过 `run_case.py:218`、`clamp_case.py:449/479`。

### 5.6 dolfinx 0.11：DG0 与 P1 的 basix family **都是 P**

区分 DG0 与 P1 **必须看 `basix_element.discontinuous`**，不能看 family
（`FiniteElement` 也没有 `family_name` 属性，会 AttributeError）。

### 5.7 环境：Bash 权限分类服务会长时间不可用

2026-10-17 中午起 `claude-sonnet-5[1m] is temporarily unavailable (timed out)`
持续约半小时，**连 `git commit` 都被挡**。简单命令（`echo`、`git add`）可用，
所以容易误以为是自己的命令写错。**等待 + 重试即可，不是代码问题。**

### 5.8 WSL 路径：`wsl -d Ubuntu-26 -- bash /root/...` 会被 Git Bash 改写

报 `bash: C:/Program Files/Git/root/pf-wire/...: No such file or directory`。
**用 `bash -lc '...'` 内联写法**，或写成脚本后用 `wsl -- bash -lc` 调。

### 5.9 输出缓冲：管道 + 重定向会吃掉几小时的日志

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
| 掩码测试 | `tests/test_damage_region.py`（14 项） |
| jitter 测试 | `tests/test_shakedown_jitter.py`（6 项） |
| **所有结论与实测数字** | `DECISIONS.md`（D-68 问题 1–18） |
| 断点 | `data/clamp_base_c_1um/`（`ckpt_0.bp` / `ckpt_1.bp`，各 265 MB） |
| 运行日志 | `data/run_mask_jitter.log`（最后一次完整跑） |

### 跑法

```bash
wsl -d Ubuntu-26 -- bash -lc 'source /root/miniforge3/etc/profile.d/conda.sh && conda activate pf-wire && cd /root/pf-wire && export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 && mpirun -n 6 python -u -m pfwire.run_clamp configs/clamp_base_c_1um.yaml --resume'
```

⚠️ **绝不用 `-n 12`**（超线程超订，实测慢 33 倍且不报错）。单工况一律 `-n 6`。

### 断点续跑须知

`--resume` 会读 `data/clamp_base_c_1um/checkpoint.json` 指向的 slot。
⚠ **断点里没有塑性历史 `ε_p`/`p`**（附带发现，见 DECISIONS.md）——
自己写探针时别假设能恢复它。

---

## 7. 开工前的 checklist

1. `git log --oneline -1` 确认是 `cd9bee1` 或更新，且**在 Desktop 那个树里**；
2. 读 `DECISIONS.md` 的 D-68 问题 9–18（尤其 12/13 的根因、15 的死锁、18 的慢衰减）；
3. `pytest -q` 确认全绿（基线约 615 项 + 1 skip）；
4. 问用户选路线 1 / 2 / 3（**上一会话没等到答复**）；
5. 写探针前重读 §5.1–5.4，这四个坑每个都吃过几小时。
