#!/usr/bin/env bash
# FD 切线检验扫 h（6 进程），判断失败是否只是步长对分区敏感。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire || exit 1
cd /root/pf-wire || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
timeout 900 mpirun -n 6 python -u data/_probe_fd_sweep.py 2>&1 | grep -vE '^Info' | tail -30
