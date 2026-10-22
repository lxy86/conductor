#!/usr/bin/env bash
# 记录 MPI 测试失败排查（D-68 问题 29）。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire || exit 1
cd /root/pf-wire || exit 1
python data/_record_mpi_tests.py
