# conductor

**悬垂线夹口铝单丝的微动疲劳寿命预测** —— FE 摩擦接触 + 相场断裂 + 疲劳退化的
全耦合有限元模型。

基于 [FEniCSx](https://github.com/FEniCS/dolfinx)（dolfinx 0.11.0）的 2D 平面应变
模型，在**同一个变分框架**内同时求解接触、J2 弹塑性与相场损伤，
无需预设萌生裂纹长度、临界平面或裂纹扩展方向准则。
相场裂纹自接触表面长到指定深度的循环数定义为 N_f。

## 模型组成

| 组件 | 方法 |
|---|---|
| 接触 | 罚函数法向 + Coulomb 摩擦切向（`dolfinx_contact` 预研，生产路径为自建罚格式） |
| 摩擦 | 正则化 Coulomb 锥投影，Fischer–Burmeister 光滑化 + Fortin–Glowinski 阻尼 |
| 塑性 | J2 径向返回映射 + 一致切线模量 |
| 断裂 | AT1 相场（spectral 能量分裂） |
| 疲劳 | Carrara et al. 2020 的 `ᾱ += max(Δψ⁺, 0)` 累积律 |
| 加速 | cycle-jump（显式一周期 + ΔN 外推 + ΔN 自适应） |

## 快速开始

### 环境

```bash
bash scripts/setup_env.sh
```

⚠ **不要裸跑 `conda env create`**。两个包 conda-forge 上没有或版本不对：

- `adios4dolfinx==0.11.0` 只在 PyPI，必须 `pip install --no-deps`
- `dolfinx_contact` 需源码编译，且要打两个 stable-ABI 补丁（否则 `ContactProblem`
  构造时报 `incompatible function arguments ... MeshTags<int>`）

`setup_env.sh` 已处理这些，装完请验证：

```bash
python -c "import dolfinx_contact; print(dolfinx_contact.__file__)"
# 必须是 cpp.abi3.so；若是 cpp.cpython-*.so 则补丁没生效
```

### 跑算例

```bash
# 环境自检
pytest tests/test_env.py -q

# 解析解对拍（Hertz / Cattaneo–Mindlin / 弹塑性压入，11 项）
pytest tests/test_contact_verify.py tests/test_indent.py tests/test_loads.py -q

# 微动疲劳算例（**必须串行**：接触求解器目前并行不可用）
bash scripts/run_663_serial.sh
```

## 硬件要求

- 单工况 `mpirun -n 6` + `OMP_NUM_THREADS=1` 最快
- ⚠ **不要按超线程数定 rank 数**：按 6 物理核 / 12 线程取 `-n 12` 会慢 33 倍，
  **且不会报错、只是白等**
- 接触（`dolfinx_contact`）目前**只能串行**

## 文档

| 文件 | 内容 |
|---|---|
| [`PLAN.md`](PLAN.md) | 研究计划、物理模型、分阶段任务与验收判据 |
| [`DECISIONS.md`](DECISIONS.md) | **8400+ 行**决策日志：每一次假设、实测数字、被证伪的方向 |
| [`HANDOFF.md`](HANDOFF.md) | 当前进度、环境重建步骤、已证伪清单、测试基线 |
| [`CLAUDE.md`](CLAUDE.md) | 工作约定、硬约束、踩过的坑 |

**接手前先读 `HANDOFF.md`**，再读 `DECISIONS.md` 里当前工作的那一节。

## 仓库结构

```text
src/pfwire/     模型代码（接触、塑性、相场、疲劳累积、cycle-jump）
configs/        全部工况参数（YAML，代码里无魔数）
scripts/        环境搭建、标定、扫参、串行重跑
tests/          单元 + 解析解对拍 + 端到端
data/           算例输出（.bp 场数据不入库，可重算）
paper/figures/  论文插图
```

## 当前状态

微动疲劳的**法向接触链路已验证良好**（夹持力搜索偏差 < 0.1%、κ 敏感性 2.6%）。
当前卡点与下一步见 `HANDOFF.md` §5 —— **三条候选路线待用户拍板**，尚未定论。

## 许可

MIT，见 [`LICENSE`](LICENSE)。
