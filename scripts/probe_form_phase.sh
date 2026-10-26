#!/usr/bin/env bash
# 定位 test_phase_changes_an_already_compiled_form 的 MPI 失败。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
timeout 400 mpirun -n 6 python -u data/_probe_form_phase.py 2>&1 | grep -vE '^Info' | tail -30
