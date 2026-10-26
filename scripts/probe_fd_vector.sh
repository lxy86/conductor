#!/usr/bin/env bash
# 比较扰动向量 δ 在 n=1 与 n=6 下是否为同一个全局向量。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
for n in 1 6; do
  echo "########## n=$n ##########"
  timeout 400 mpirun -n $n python -u data/_probe_fd_vector.py 2>&1 \
      | grep -vE '^Info|PETSC|Abort|^\s*$' | head -30
done
