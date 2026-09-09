# CLAUDE.md — 工作约定

## 项目
见 PLAN.md：FEniCSx 相场疲劳模型预测铝单丝微动疲劳寿命。只做 PLAN.md 中列出的内容，
按 Phase 顺序推进，每完成一个子任务在 PLAN.md 勾选。

## 环境
WSL2 `Ubuntu-26` 里的 conda 环境 `pf-wire`，dolfinx **0.11.0**（real PETSc）。
所有命令在此环境运行：

```bash
wsl -d Ubuntu-26
source /root/miniforge3/etc/profile.d/conda.sh && conda activate pf-wire
cd /root/pf-wire   # 指向 C:\Users\Administrator\Desktop\疲劳寿命 的 ASCII 软链接
```

建/更新环境用 `bash scripts/setup_env.sh`，**不要**裸跑 `conda env create`
（adios4dolfinx 0.11 只在 PyPI 上，须 `pip install --no-deps`，见 DECISIONS.md D-0）。
机器：12 核 / 47 GB。并行策略：单工况 MPI 4 核，sweep 同时 3 个工况（见 D-4）。

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