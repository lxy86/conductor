#!/usr/bin/env bash
# 验证「测试漏调 commit_step()」假设：1 进程与 6 进程各跑一次。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire || exit 1
cd /root/pf-wire || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
for n in 1 6; do
    echo "########## mpirun -n $n ##########"
    timeout 900 mpirun -n $n python -u data/_probe_fd_commit.py 2>&1 \
        | grep -vE '^Info' | grep -E '^\[' | tail -10
done
echo "=== done ==="
