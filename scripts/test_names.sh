#!/usr/bin/env bash
# 列出失败测试名（不带 mpirun，避免 MPI abort 截断汇总）。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire || exit 1
cd /root/pf-wire || exit 1
export OMP_NUM_THREADS=1
echo "=== 单进程跑，取完整失败清单 ==="
timeout 900 python -m pytest tests/test_clamp_uzawa.py tests/test_clamp_contact.py \
    tests/test_clamp_loads.py -q -rf 2>&1 | tail -30
echo "=== done ==="
