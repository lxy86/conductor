#!/usr/bin/env bash
# 6.6.3 串行重跑（D-68 问题 33 续 11 方向 v）。
#
# 为什么串行：续 11 的定量对照证明，**同一 config、同一相位**下
#   - 串行  max|psi+| = 7.06e5（alpha_T 的 4.0 倍）、max|p| = 1.44 sigma_y
#   - 并行  max|psi+| = 1.02e10（alpha_T 的 57933 倍）
# 而 u 与 delta 同量级/逐位相同。即并行特有数值问题，且三个并行探针都崩在
# sample_psi_plus()（psi+ 的 DG-2 采样）处。CLAUDE.md 已知陷阱 5 也早就明确
# 「接触（dolfinx_contact）目前只能串行」。
#
# 代价：串行慢。不设单工况耗时上限（CLAUDE.md D-4，用户 2026-09-09 指示）。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire || exit 1
cd /root/pf-wire || exit 1
export OMP_NUM_THREADS=1

CONFIG=${1:-configs/clamp_fretting_coarse.yaml}
OUT=$(python -c "
import yaml,sys
print(yaml.safe_load(open('$CONFIG'))['output']['dir'])
")
rm -rf "$OUT"
LOG=/root/pf-wire/data/_run663_serial.log
: > "$LOG"
echo "=== 6.6.3 串行开始 $(date -Is) config=$CONFIG ===" | tee -a "$LOG"
# 不加 timeout：串行慢，且 D-4 明确不设单工况耗时上限。
# 若需中止，用户直接 Ctrl-C（或 stop_terminal_tab）。
python -u -m pfwire.run_clamp "$CONFIG" >> "$LOG" 2>&1
echo "=== 结束 $(date -Is) rc=$? ===" | tee -a "$LOG"
