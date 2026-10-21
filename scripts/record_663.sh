#!/usr/bin/env bash
# 把 6.6.3 诊断写进 DECISIONS.md（D-68 问题 28 续 5）。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire || exit 1
cd /root/pf-wire || exit 1
python data/_record_663_status.py
