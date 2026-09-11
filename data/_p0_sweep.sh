#!/usr/bin/env bash
# 扫 p0，找「弹性解不越起裂门槛」的上限。
# ψ⁺ ∝ p0²，故 max ψ⁺/门槛 应按 p0² 变化 —— 这条标度关系本身也是一次自检。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire
cd /root/pf-wire
export OMP_NUM_THREADS=1
for p0 in 500e6 400e6 300e6 261e6 200e6 150e6; do
  echo "########## p0 = $p0 ##########"
  mpirun -n 6 python data/_fretting_elastic_probe.py "$p0" 2>&1 \
    | grep -E "p0 =|split = spectral|max ψ⁺|越门槛|弹性解"
done
echo P0_SWEEP_DONE
