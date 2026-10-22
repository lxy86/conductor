#!/usr/bin/env bash
# 只跑 n=6，完整输出（不 tail 截断头部）。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire || exit 1
cd /root/pf-wire || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
timeout 600 mpirun -n 6 python -u data/_probe_fd_u0.py 2>&1 \
    | grep -vE '^Info|PETSC|Abort|^\s*$' | head -20
