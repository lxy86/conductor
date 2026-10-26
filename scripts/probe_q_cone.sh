#!/usr/bin/env bash
# Coulomb 锥检验（2026-10-21）：q 是否越出 |q| ≤ μp。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
echo "=== Coulomb 锥检验 ==="
mpirun -n 6 python data/_probe_q_cone.py 2>&1 | grep -v '^Info'
echo "=== done ==="
