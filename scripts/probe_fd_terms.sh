#!/usr/bin/env bash
# 把 u 残差拆块，逐块比 FD 与 J·δ，定位并行下不一致的那一项。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire || exit 1
cd /root/pf-wire || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
timeout 900 mpirun -n 6 python -u data/_probe_fd_terms.py 2>&1 | grep -vE '^Info' | tail -30
