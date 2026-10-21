#!/usr/bin/env bash
# Coulomb 锥检验（2026-10-21）：q 是否越出 |q| ≤ μp。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire || exit 1
cd /root/pf-wire || exit 1
export OMP_NUM_THREADS=1
echo "=== Coulomb 锥检验 ==="
mpirun -n 6 python data/_probe_q_cone.py 2>&1 | grep -v '^Info'
echo "=== done ==="
