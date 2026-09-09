# PLAN.md — 基于 FEniCSx 相场疲劳模型的输电导线铝单丝微动疲劳寿命预测

> 目标：一篇 EI 会议论文（6–8 页）。
> 范围：仅 L3 尺度，2D 平面应变铝单丝微动疲劳；相场裂纹从接触表面贯穿到指定深度的
> 循环数定义为寿命 N_f。
> 不做：磨损几何更新、接触求解（用解析面力代替）、L1/L2 多股模型、3D、多丝、
> 弹塑性、代理模型、LLM。全部写入 Future Work。
>
> **对本文件的修订、以及 PLAN 未覆盖的决策，一律记在 `DECISIONS.md`**（含 ℓ 与失效
> 深度的尺度修正 D-1、G_c/ℓ/σ_c 三角的取舍 D-2、提速路线 D-4 等）。本文件中带
> 「见 D-x」标注的数值以 DECISIONS.md 为准。

---

## 0. 论文贡献点（只需两个）

1. 建立 FEniCSx 相场疲劳模型（AT1 + Carrara 疲劳退化 + cycle-jump），复现铝丝微动
   疲劳的裂纹萌生位置、扩展路径与寿命；
2. 给出单丝微动疲劳 S–N 曲线，与 CIGRE 安全边界线对比，并讨论接触压力/滑移状态
   对寿命的影响。

---

## 1. 物理模型

**几何**（2D 平面应变，圆柱–平面接触简化为矩形试样 + 表面面力）

```text
试样：矩形 W × H，W = 6a，H = 2 R_wire（默认 R_wire = 1.6 mm → H = 3.2 mm）
接触区中心在上表面 x = 0，半宽 a 由 Hertz 给出
边界：底边 u_y = 0；左边 u_x = 0；右边施加 σ_b(t)；上表面 |x| ≤ a 施加 p(x), q(x,t)
```
（W = 6a 与 ℓ = 40 µm 互不自洽，已按 D-1 改为 W 与 R_wire 挂钩、ℓ = 4 µm）


**载荷**（解析面力，Nowell–Hills 常规做法）

```text
Hertz（单位长度法向力 P，平面应变）：
E* = 1/[(1−ν₁²)/E₁ + (1−ν₂²)/E₂] 铝–铝：R = R_wire/2；铝–刚性：R = R_wire
a = sqrt(4PR/(πE*)), p0 = 2P/(πa), p(x) = p0 sqrt(1−(x/a)²)
Cattaneo–Mindlin 部分滑移（Q < μP）：
c/a = sqrt(1 − Q/(μP))
q(x) = μ p0 sqrt(1−(x/a)²) c ≤ |x| ≤ a
q(x) = μ p0 [sqrt(1−(x/a)²) − (c/a) sqrt(1−(x/c)²)] |x| < c
时间历程（比例加载简化，论文中说明）：
q(x,t) = q(x) sin(2πt/T), σ_b(t) = σ_a sin(2πt/T)
```


**与导线的对应**（只用于论文中把 σ_a 换算成弯曲振幅 Y_b，代码中一个函数即可）

```text
Poffenberger–Swart: σ_a = K·Y_b
K = E_al d_wire p²/(4(e^{−px} − 1 + px)), p = sqrt(T/EI_min), x = 89 mm
CIGRE 安全边界线（实现前对照 TB 332 核对系数）:
σ_a = 450 N^(−0.2) MPa (N ≤ 1.56e7); σ_a = 263 N^(−0.17) MPa (N > 1.56e7)
```
（该系数为多层导线用；单层导线另有一套，需按 TB 332 确认，见 D-6）

**材料**：Al 1350-H19，E = 69 GPa，ν = 0.33，σ_u ≈ 170 MPa。
G_c 由 K_IC 换算 G_c = K_IC²(1−ν²)/E；AT1 下 ℓ = 3EG_c/(8σ_c²)，取 σ_c ≈ σ_u。
（这三条与 §5 的 ell 过定且互相冲突，实际取法见 D-2：σ_c = σ_u 与 ℓ 为输入、G_c 导出）
参数与来源全部写入 `configs/material_al1350.yaml`。

---

## 2. 数值模型

**相场（AT1，Amor 体积–偏量分解）**

```text
E(u,d) = ∫ [g(d)ψ⁺ + ψ⁻] dV + ∫ f(ᾱ)(3G_c/8)[d/ℓ + ℓ|∇d|²] dV
g(d) = (1−d)² + k_res, k_res = 1e-6
```

- 不可逆：d 用 PETSc SNES `vinewtonrsls`，界 [d_old, 1]；
- 交错求解：u（LinearProblem, MUMPS）↔ d，收敛 ‖Δd‖_∞ < 1e-4，最多 50 次；
- 拉压分解为必需项（接触区为高压应力区），配置可切换 spectral 作对比。

**疲劳退化（Carrara et al. 2020）**

```text
ᾱ_{n+1} = ᾱ_n + max(ψ⁺_{n+1} − ψ⁺_n, 0)
f(ᾱ) = 1 (ᾱ ≤ α_T); f(ᾱ) = (2α_T/(ᾱ+α_T))² (ᾱ > α_T)
```

ᾱ、ψ⁺ 改存 DG-2（非 quadrature 元，为了 adios4dolfinx 断点续跑，见 D-7）。
α_T 为标定参数；若单参数拟不上 S–N，按 D-2/D-3 放开 G_c 作第二参数。

**cycle-jump**
- 显式循环：n_sub = 8 子步 / 周期，得逐点 Δᾱ_cyc；
- 外推 ΔN：ᾱ += ΔN·Δᾱ_cyc，随后重解 (u,d)；
- ΔN 自适应：目标 max Δd ≤ 0.02，ΔN ∈ [1, 1e5]，乘/除 2 调节。

**失效判据**
- d ≥ 0.95 的连通区域从上表面贯穿到深度 0.5·R_wire → N_f；
- N > N_max = 1e8 未失效 → runout。

---

## 3. 仓库结构（保持最小）

```text
pf-wire/
├── CLAUDE.md
├── PLAN.md
├── environment.yml
├── configs/
│ ├── material_al1350.yaml
│ ├── benchmark_sent.yaml # 验证 1：单边缺口拉伸
│ ├── benchmark_fatigue.yaml # 验证 2：Carrara 疲劳基准
│ ├── calib_tension.yaml # α_T 标定：无接触纯拉伸试样
│ ├── fretting_base.yaml # 基准微动工况
│ └── sweep/ # S–N 与参数扫描的 yaml（脚本自动生成）
├── src/pfwire/
│ ├── mesh.py # gmsh：SENT 试样 / 微动试样（接触区局部加密）
│ ├── material.py # 弹性张量、Amor/spectral 分解、参数换算
│ ├── phasefield.py # AT1 泛函、f(ᾱ)、交错求解器
│ ├── cyclejump.py # ᾱ 累积与 ΔN 自适应
│ ├── loads.py # Hertz、Mindlin、Poffenberger–Swart、CIGRE 边界线
│ ├── failure.py # 贯穿判据、裂纹长度提取
│ ├── run_case.py # 单工况入口：yaml → result.json + 场输出
│ └── postproc.py # a–N 曲线、S–N 图、d 场快照
├── scripts/
│ ├── calibrate_alphaT.py
│ ├── make_sweep.py # 生成 sweep/*.yaml
│ └── run_sweep.sh # 顺序/并行运行 sweep，跳过已完成工况
├── tests/
├── data/ # 每个工况：config.yaml, result.json, fields.bp
└── paper/figures/
```

（就地展开于当前目录，另加 `pyproject.toml`、`scripts/setup_env.sh`、`DECISIONS.md`，见 D-11）

---

## 4. 分阶段任务与验收

### Phase 0 — 环境（第 1 周）
- [x] environment.yml：dolfinx **0.11.0**（real PETSc）、gmsh、adios4dolfinx、pyvista；
      建环境用 `scripts/setup_env.sh`（adios4dolfinx 0.11 只在 PyPI 上，见 D-0）；
- [x] material.py 参数换算 + 单元测试（18 项，含 AT1 三角互逆与过定报错）；
- [x] 线弹性悬臂梁冒烟测试，端部挠度误差 **0.257%**（对 Timoshenko）< 1%；
- [x] 附加：gmsh→dolfinx 在中文路径下可用、adios4dolfinx checkpoint 往返、
      SNES `vinewtonrsls` 可用；`pytest tests/test_env.py` 在 1/2/4 进程均通过。

### Phase 1 — 相场疲劳求解器（第 2–5 周）
- [x] 1.1 静态 AT1 + Amor 分解 + 交错求解 + SNES VI 不可逆；
      `src/pfwire/phasefield.py`；36 项单测。均匀单轴算例与 AT1 闭式解
      d = max(0, 1-(σ_c/σ)²) 吻合到机器精度（相对差 ~1e-15）；
      amor/spectral 能量一致性、各向同性、不可逆、界约束、linear vs newton 一致性均测住；
- [x] 1.2 Carrara 疲劳退化，ᾱ 存 **DG-2**（非 quadrature，见 D-7）；
      `phasefield.fatigue_degradation` + `cyclejump.FatigueState`；12 项单测。
      循环算例与解析寿命 N = 2Eα_T(2/s−1)/(s²σ_c²) 吻合（起裂循环数与 d(N) 均对）；
      ᾱ/ψ⁺_prev 的 checkpoint 往返已测通（硬约束 5）。
      **发现并已解决**：asymptotic 形式（PLAN 原写的那个）均匀极限 S–N 斜率固定 ~1/3，
      单个 α_T 拟合不了 4 个标定点（平均 |Δlog10 N| = 0.434）。改用 Carrara 原文
      Eq. (38) 的**对数型** f(ᾱ)（多一个 κ 调斜率）后，两参数覆盖完整曲线：
      α_T = 3.407e6 J/m³、κ = 0.14888 时 4 点平均偏差 0.061、最大 0.077。见 D-14。
      另按原文 Eq. (13) 把 ᾱ 的累积量改为**退化后**的 ψ⁺（D-15）。
- [x] 1.3 cycle-jump；`cyclejump.CycleJumpDriver`；11 项单测。
      显式循环 Δᾱ_cyc 与闭式值 0.75·E·e_a²（n_sub=8/4 精确、n_sub=3 只得一半）吻合；
      端到端 d(N) 与闭式解吻合到 1e-13~1e-8；ΔN 自适应的增长/封顶/拒绝-回滚-减半、
      不越过 N_max、以及「降到下界仍不收敛则硬失败」均测住。
      按 D-4 子步只解 u（d 冻结）；jump 步数几乎与总寿命无关（ΔN 随寿命等比放大）。
- [x] 1.4 失效判据与裂纹长度提取（沿 d ≥ 0.95 区域最深点）；`failure.py`；19 项单测。
      连通性用「固定规则网格采样 + scipy 8 连通标记」而非网格拓扑连通域：
      前者天然 MPI 安全（各 rank 算自己的点再 allreduce），单元定位缓存复用，
      且 Phase 3 画裂纹路径正好要这张网格。只保留触及表面那一行的连通域，
      内部孤立损伤斑不计为裂纹。同时提取萌生位置 x_surface 与初始倾角
      （供 Phase 2.4 判「初始倾斜后转向」）。判据深度按 D-1 用绝对 100 µm，
      同时兼容 PLAN 原来的 depth_fraction 写法（两者只能给一个，否则报错）。
- [x] 验证 1：SENT 拉伸（Miehe 2010 几何），峰值载荷与文献偏差 < 5%，
      h = ℓ/4 与 ℓ/8 结果差 < 3%。**两条口径均满足**：h=ℓ/4 vs ℓ/8 差 1.34%；
      收敛网格上 ℓ₁ +1.46%（h=ℓ/8 时 +0.12%）、ℓ₂ +1.09%。
      关键发现：h = ℓ/2（原文规定的上限）欠解析、**高估**峰值 4–5%，
      三档 h 单调收到原文值，故此后一律 h ≤ ℓ/4。详见 D-17；
      图 paper/figures/verification1_sent.png。
- [ ] 验证 2：复现 Carrara 2020 一个疲劳算例（CT，a0/W = 0.22、ΔP = 20 N），
      画 **da/dN–ΔK** 曲线，Paris 斜率 m 与原文 3.42 同量级（D-10 定为 ±30%）。
      横坐标改用 ΔK 而非本行原写的 ΔG：原文 Tab. 2 就是 ΔK 口径，C 与 m 都能
      直接对表；需要 ΔG 时按 ΔG = ΔK²/E 换算（注意 m_G = m_K/2）。见 D-20。

**验收**：两张验证图入 `paper/figures/`；`pytest` 通过（分解对称性/能量一致性、
f(ᾱ) 单调性、ΔN 调节逻辑）；SENT 静态算例单核 < 10 min。

### Phase 2 — 微动疲劳算例（第 5–9 周）
- [ ] 2.1 mesh.py 微动试样：加密区 |x| ≤ 1.5a、深 1 mm、h = ℓ/4，外部渐变到 0.2 mm；
- [ ] 2.2 loads.py：Hertz、Mindlin（与解析值单元测试比对，误差 < 1e-6）、
      P–S、CIGRE 边界线；
- [ ] 2.3 α_T 标定：`calib_tension.yaml` 无接触纯拉伸，4 个应力水平，
      调 α_T 使 N_f 对齐 Al 1350-H19 单丝 S–N 文献数据（数据由用户提供）；
      结果写入 material yaml 并注明来源；
- [ ] 2.4 基准工况 `fretting_base.yaml`：p0 = 500 MPa，Q/μP = 0.6，σ_a = 35 MPa，
      μ = 0.7，R_wire = 1.6 mm，铝–铝对偶；
      **必须复现**：裂纹在接触边缘 x ≈ ±a 附近（滑移区）萌生，初始倾斜，
      随后转向垂直于本体应力方向。若不满足，停下报告，不得调参"凑"结果；
- [ ] 2.5 S–N 扫描：固定基准的 p0、Q/μP、μ、R，σ_a ∈ {20, 27, 35, 45, 55, 65} MPa，6 个工况；
- [ ] 2.6 参数影响（选一个）：固定 σ_a = 35 MPa，p0 ∈ {300, 500, 700} MPa，
      或 Q/μP ∈ {0.3, 0.6, 0.9}，3 个工况；
- [ ] 2.7 网格无关性：基准工况 h = ℓ/4 vs ℓ/6，N_f 差 < 10%（1 个工况）。

总计约 12–14 个微动工况 + 4 个标定工况。单工况目标 ≤ 45 min（单核，~4 万单元）；
超时则放宽 h = ℓ/3 或缩小加密区。`run_sweep.sh` 用 multiprocessing 并行、支持断点续跑。

**验收**：`data/` 下每个工况有 result.json（N_f、runout、x_init/a、初始倾角、a–N、耗时）；
基准工况经典现象复现；S–N 六点全部有效（runout ≤ 1 个）。

### Phase 3 — 图表与写作（第 10–12 周）
论文图（8–9 张）：
1. 问题示意：线夹口微动 → 2D 模型与边界条件
2. 验证 1：SENT 载荷–位移
3. 验证 2：Paris 曲线
4. 基准工况 d 场演化 4 帧（萌生 → 倾斜扩展 → 转向 → 贯穿）
5. 基准工况 a–N 曲线 + ᾱ 分布
6. S–N 曲线（横轴双标：σ_a 与换算的 Y_b），叠加 CIGRE 安全边界线
7. 参数影响：N_f 与萌生位置随 p0（或 Q/μP）变化
8. 网格无关性（可并入附录或一句话）

论文结构：Intro / Phase-field fatigue model / Fretting fatigue model of Al wire /
Verification / Results & discussion / Conclusions & future work。
Discussion 必须包含：(a) 为什么不做磨损——用 Archard 估算说明本工况磨损深度 ≪ 裂纹深度；
(b) "单丝首裂"与 CIGRE"导线断丝"定义差异。

---

## 5. 配置 schema（fretting_base.yaml）

```yaml
material: configs/material_al1350.yaml
model:   {type: AT1, split: amor, ell: 40e-6, k_res: 1e-6}
fatigue: {alpha_T: null, form: asymptotic}      # null → 读 material yaml 标定值
geometry: {R_wire: 1.6e-3, H: 3.2e-3, W_factor: 6.0}
mesh:    {h_fine_factor: 0.25, h_coarse: 2e-4, refine_depth: 1.0e-3, refine_halfwidth_factor: 1.5}
loads:   {p0: 500e6, Q_ratio: 0.6, sigma_a: 35e6, mu: 0.7, counterpart: aluminum}
cycle:   {n_sub: 8, dN_init: 100, dN_max: 100000, dd_max: 0.02, N_max: 1.0e8}
failure: {d_threshold: 0.95, depth_fraction: 0.5}
output:  {dir: data/fretting_base, save_fields_every: 5}
```

注：本 schema 中 `ell`、`W_factor`、`mesh.*`、`failure.depth_fraction` 的数值已按
D-1 修订（ℓ = 4e-6、失效改为绝对深度 100 µm、W 与 R_wire 挂钩、加密区随之调整），
实际取值以 Phase 2.1 落地时写入的 `configs/fretting_base.yaml` 为准。
另注：PyYAML 走 YAML 1.1，`500e6` / `1.0e8` 这类不带指数符号的写法会被解析成
**字符串**，配置载入必须做数值 coerce（Phase 0 已在 material.py 处理并加回归测试）。

---

## 6. 风险与应急

| 风险 | 表现 | 应急 |
|---|---|---|
| 接触区压应力产生假损伤 | d 在接触中心而非边缘增长 | 换 spectral 分解；接触面下 2 层单元 d 固定为 0（论文注明）。启用顺序见 D-8 |
| 单工况过慢 | > 1 h | h = ℓ/3；缩小加密区；n_sub = 6；N_max = 5e7。提速路线见 D-4 |
| cycle-jump 使裂纹"跳跃" | a–N 曲线阶梯状 | dd_max 降至 0.01；扩展阶段 dN_max 限 1e3 |
| 寿命数量级偏离 CIGRE | 差 > 1 个量级 | 复查 α_T 标定；论文中说明定义差异，不强行拟合 |
| dolfinx API 错误 | import/签名不符 | 以 0.11.0 官方 demo 为准，禁止 legacy 写法（0.11 已把 `dolfinx.io.gmshio` 改名为 `dolfinx.io.gmsh`，`LinearProblem` 多了必需的 `petsc_options_prefix`） |

---

## 7. 时间表

| 周 | 里程碑 |
|---|---|
| 1 | Phase 0 |
| 2–5 | Phase 1，两个验证通过 |
| 5–6 | 微动网格 + 载荷；基准工况复现经典现象 |
| 7 | α_T 标定 |
| 8–9 | S–N 扫描 + 参数影响 + 网格无关性（后台跑） |
| 10–12 | 图表、写作、投稿 |