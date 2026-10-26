#!/usr/bin/env bash
# 切向幅值扫参探针（2026-10-21）：粗网格摩擦工况首跳 d-SNES -6 的定位。
# 用法：bash scripts/probe_amp.sh [幅值逗号列表]
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
AMPS=${1:-1e-8,3e-8,1e-7}
echo "=== 幅值扫参：$AMPS ==="
mpirun -n 6 python data/_probe_amp_sweep.py "$AMPS" 2>&1 | grep -v '^Info'
echo "=== done ==="
