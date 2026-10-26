#!/usr/bin/env bash
# 把 6.6.3 诊断写进 DECISIONS.md（D-68 问题 28 续 5）。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
python data/_record_663_status.py
