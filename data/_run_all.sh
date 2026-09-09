#!/usr/bin/env bash
# 验证 1 全部算例：两个 ell + 网格无关性两档。纯 MPI 12 进程。
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire
cd /root/pf-wire
for cfg in benchmark_sent benchmark_sent_ell1 benchmark_sent_h4 benchmark_sent_h8; do
  echo "############ $cfg ############"
  start=$(date +%s)
  mpirun -n 12 python -m pfwire.run_case "configs/$cfg.yaml" -v 2>&1 | grep -E "状态=|未收敛|max_d=0\.9" | tail -3
  echo "---- $cfg 墙钟 $(( $(date +%s) - start )) s ----"
done
echo ALL4_DONE
