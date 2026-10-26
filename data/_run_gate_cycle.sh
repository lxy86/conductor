#!/usr/bin/env bash
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
G=${1:-0.0}
LOG="data/_gate_cycle_${G}.log"
: > "$LOG"
python -u data/_probe_gate_cycle.py "$G" >> "$LOG" 2>&1
echo "--- gate=$G ---"
grep -aE "^(===|  ✅|  ❌|     )" "$LOG" | tail -10
