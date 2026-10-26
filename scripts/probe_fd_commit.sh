#!/usr/bin/env bash
# 验证「测试漏调 commit_step()」假设：1 进程与 6 进程各跑一次。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
for n in 1 6; do
    echo "########## mpirun -n $n ##########"
    timeout 900 mpirun -n $n python -u data/_probe_fd_commit.py 2>&1 \
        | grep -vE '^Info' | grep -E '^\[' | tail -10
done
echo "=== done ==="
