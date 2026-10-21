#!/usr/bin/env bash
# 切向幅值扫参探针（2026-10-21）：粗网格摩擦工况首跳 d-SNES -6 的定位。
# 用法：bash scripts/probe_amp.sh [幅值逗号列表]
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire || exit 1
cd /root/pf-wire || exit 1
export OMP_NUM_THREADS=1
AMPS=${1:-1e-8,3e-8,1e-7}
echo "=== 幅值扫参：$AMPS ==="
mpirun -n 6 python data/_probe_amp_sweep.py "$AMPS" 2>&1 | grep -v '^Info'
echo "=== done ==="
