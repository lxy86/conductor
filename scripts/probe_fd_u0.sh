#!/usr/bin/env bash
# 纯弹性配置下 ‖u‖≈0 却 ‖J·δ‖~1e12 的定位。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire || exit 1
cd /root/pf-wire || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
for n in 1 6; do
  echo "########## n=$n ##########"
  timeout 600 mpirun -n $n python -u data/_probe_fd_u0.py 2>&1 | grep -vE '^Info' | tail -20
done
