#!/usr/bin/env bash
# FD/J 不一致归因：串行 vs 并行 × 四种配置组合。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
for n in 1 6; do
    echo "########## mpirun -n $n ##########"
    timeout 1200 mpirun -n $n python -u data/_probe_fd_attrib.py 2>&1 \
        | grep -vE '^Info' | grep -E '^\[|^  |^===|^  去|^  →' | tail -20
done
echo "=== done ==="
