#!/usr/bin/env bash
# 取 6 个 MPI 失败的测试名。python -u + 落文件，跑完再单独 grep（管道 grep 会被 kill 吞缓冲）。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire || exit 1
cd /root/pf-wire || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
LOG=/root/pf-wire/data/_mpi_fail.log
: > "$LOG"
echo "start $(date -Is)"
timeout 1800 mpirun -n 6 python -u -m pytest tests/test_clamp_uzawa.py \
    tests/test_clamp_contact.py tests/test_clamp_loads.py \
    -v -rf --tb=no -p no:cacheprovider > "$LOG" 2>&1
echo "exit=$? lines=$(wc -l < "$LOG") $(date -Is)"
