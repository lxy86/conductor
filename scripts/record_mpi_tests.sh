#!/usr/bin/env bash
# 记录 MPI 测试失败排查（D-68 问题 29）。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
python data/_record_mpi_tests.py
