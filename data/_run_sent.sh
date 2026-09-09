#!/usr/bin/env bash
# 验证 1：两个 ell 各跑一轮。纯 MPI（每 rank 单线程），12 进程。
set -u
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire
cd /root/pf-wire

NP=${NP:-12}
for cfg in configs/benchmark_sent.yaml configs/benchmark_sent_ell1.yaml; do
  echo "############ $cfg  (np=$NP) ############"
  start=$(date +%s)
  mpirun -n "$NP" python -m pfwire.run_case "$cfg" -v 2>&1 | tail -25
  echo "---- 墙钟 $(( $(date +%s) - start )) s ----"
done
echo "ALL_DONE"
