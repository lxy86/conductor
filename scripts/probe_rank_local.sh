#!/usr/bin/env bash
# 验证 U8/U9 的 MPI 失败是否来自按 rank 局部的归约（CLAUDE.md 陷阱 3）。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
timeout 600 mpirun -n 6 python -u data/_probe_rank_local.py 2>&1 | grep -vE '^Info|^$' | tail -40
