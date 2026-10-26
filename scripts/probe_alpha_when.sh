#!/usr/bin/env bash
# ᾱ 爆炸时点定位探针（2026-10-21）。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
echo "=== ᾱ 时点定位 ==="
mpirun -n 6 python data/_probe_alpha_when.py 2>&1 | grep -v '^Info'
echo "=== done ==="
