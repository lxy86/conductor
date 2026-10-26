#!/usr/bin/env bash
# FD/J 不一致是否来自状态本身（装配确定性、解可复现性、残差范数）。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
timeout 900 mpirun -n 6 python -u data/_probe_fd_state.py 2>&1 | grep -vE '^Info' | tail -30
