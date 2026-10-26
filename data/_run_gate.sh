#!/usr/bin/env bash
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
for G in "$@"; do
  LOG="data/_gate_${G}.log"
  : > "$LOG"
  python -u data/_probe_gate.py "$G" >> "$LOG" 2>&1
  echo "--- gate=$G ---"
  grep -E "^(===|相位|  ✅|  ❌|    \[)" "$LOG" | tail -6
done
