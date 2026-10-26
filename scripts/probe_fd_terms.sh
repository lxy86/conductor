#!/usr/bin/env bash
# 把 u 残差拆块，逐块比 FD 与 J·δ，定位并行下不一致的那一项。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
timeout 900 mpirun -n 6 python -u data/_probe_fd_terms.py 2>&1 | grep -vE '^Info' | tail -30
