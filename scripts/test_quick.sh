#!/usr/bin/env bash
# 删 x_amplitude 后的回归：只跑**快**且与改动直接相关的测试。
#
# 为什么不跑 test_clamp_case.py / test_obstacle.py：它们要建真实网格，
# 2026-10-22 实测 mpirun -n 6 下 >58 min 仍未结束（单进程亦慢），
# 不适合当快速回归。这两个文件与 x_amplitude 无关（grep 无命中），
# 留待单独跑。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire || exit 1
cd /root/pf-wire || exit 1
export OMP_NUM_THREADS=1
echo "=== pytest: Uzawa / 接触 / 载荷 ==="
timeout 900 mpirun -n 6 python -m pytest tests/test_clamp_uzawa.py \
    tests/test_clamp_contact.py tests/test_clamp_loads.py -q 2>&1 | tail -25
echo "=== exit=$? ==="
