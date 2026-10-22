#!/usr/bin/env bash
# FD/J 不一致归因：串行 vs 并行 × 四种配置组合。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire || exit 1
cd /root/pf-wire || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
for n in 1 6; do
    echo "########## mpirun -n $n ##########"
    timeout 1200 mpirun -n $n python -u data/_probe_fd_attrib.py 2>&1 \
        | grep -vE '^Info' | grep -E '^\[|^  |^===|^  去|^  →' | tail -20
done
echo "=== done ==="
