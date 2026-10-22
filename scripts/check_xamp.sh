#!/usr/bin/env bash
# 确认 x_amplitude 删除后幅值链路仍通。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire || exit 1
cd /root/pf-wire || exit 1
export OMP_NUM_THREADS=1
mpirun -n 6 python data/_check_xamp_gone.py 2>&1 | grep -v '^Info'
