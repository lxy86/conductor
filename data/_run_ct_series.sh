#!/usr/bin/env bash
# 验证 2 的 CT 系列：按代价从小到大跑，铺开 ΔK 跨度（原文 Fig. 13 的做法）。
# 本机 6 物理核 -> 一律 -n 6（见 D-21，绝不能用 -n 12）。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire
cd /root/pf-wire
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
for case in ct_a050_dp20 ct_a022_dp40 ct_a040_dp20; do
  echo "############ $case ############"
  t0=$SECONDS
  mpirun -n 6 python -m pfwire.run_case "configs/$case.yaml" -v \
      > "/root/${case}.log" 2>&1
  tail -2 "/root/${case}.log"
  echo "---- $case 墙钟 $((SECONDS-t0)) s ----"
done
echo CT_SERIES_DONE
