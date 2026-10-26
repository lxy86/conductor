#!/usr/bin/env bash
# 最终定位：FD 范数是否把 ghost 自由度算进去了。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
timeout 600 mpirun -n 6 python -u data/_probe_fd_ghostnorm.py 2>&1 \
    | grep -vE '^Info' | tail -25
