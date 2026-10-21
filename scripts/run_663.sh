#!/usr/bin/env bash
# 6.6.3 完整验证：粗网格 + 阻尼 Uzawa + 子步失败接入 ΔN 减半重试。
#
# 用法（WSL2 Ubuntu-26）：
#   bash scripts/run_663.sh            # 前台跑（会占用终端）
#   nohup bash scripts/run_663.sh > /root/pf-wire/data/_run663.log 2>&1 &
#
# 为什么写成脚本：多行命令经 `wsl -d ... -e bash -c "..."` 传递时换行会被
# 折叠成空格，`#` 注释会把后面的内容整行吃掉 —— 实测 2026-10-21 试了三次
# 都是起了个空 shell、什么都没跑（ps 里只剩 bash，无 mpirun/python）。
set -u

CONDA=/root/miniforge3/etc/profile.d/conda.sh
# shellcheck disable=SC1090
source "$CONDA"
conda activate pf-wire || exit 1

cd /root/pf-wire || exit 1

echo "=== 6.6.3 验证开始 $(date -Is) ==="
echo "host: $(hostname)  nproc=$(nproc)  OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
echo "config: ${CONFIG:-configs/clamp_fretting_coarse.yaml}"

# OMP_NUM_THREADS=1：纯 MPI 比 MPI+BLAS 线程混用更快、也不抢核（CLAUDE.md）。
export OMP_NUM_THREADS=1

# ⚠ **必须用粗网格 config（clamp_fretting_coarse.yaml），不是 clamp_fretting_uzawa.yaml**。
# 后者是细网格版（49 万节点），它的 `u_linesearch: basic` 在**延拓期**就报 SNES -5
# （实测 2026-10-21，跑到一半炸；DECISIONS.md 记「basic 在延拓上直接 -5」）。
# 粗网格版用 `bt`，已由无摩擦对照（clamp_fretting_coarse_mu0.yaml）验证能跑通
# 延拓 + W' 搜索 + cycle-jump 到 N=24.78。用户 2026-10-20 拍板「先用粗网格跑通全流程」。
CONFIG=${CONFIG:-configs/clamp_fretting_coarse.yaml}

# -n 6 = 6 物理核（一核一 rank）。**绝不用 -n 12**（超线程超订，MPICH 忙等
# 下会塌方，实测慢 33 倍，见 CLAUDE.md 与 D-21）。
mpirun -n 6 python -m pfwire.run_clamp "$CONFIG"
rc=$?

echo "=== 6.6.3 验证结束 $(date -Is)，exit=$rc ==="
exit "$rc"
