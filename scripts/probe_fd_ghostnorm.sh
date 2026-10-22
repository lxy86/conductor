#!/usr/bin/env bash
# 最终定位：FD 范数是否把 ghost 自由度算进去了。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire || exit 1
cd /root/pf-wire || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
timeout 600 mpirun -n 6 python -u data/_probe_fd_ghostnorm.py 2>&1 \
    | grep -vE '^Info' | tail -25
