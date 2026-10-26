#!/usr/bin/env bash
# 确认 x_amplitude 删除后幅值链路仍通。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
mpirun -n 6 python data/_check_xamp_gone.py 2>&1 | grep -v '^Info'
