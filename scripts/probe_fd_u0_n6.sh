#!/usr/bin/env bash
# 只跑 n=6，完整输出（不 tail 截断头部）。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
timeout 600 mpirun -n 6 python -u data/_probe_fd_u0.py 2>&1 \
    | grep -vE '^Info|PETSC|Abort|^\s*$' | head -20
