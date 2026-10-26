#!/usr/bin/env bash
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
ETA=${1:-0.05}
PHASE=${2:-0.05}
LOG="data/_eta_stag_${ETA}_${PHASE}.log"
: > "$LOG"
python -u data/_probe_eta_stag.py "$ETA" "$PHASE" >> "$LOG" 2>&1
echo "rc=$?"
echo "=== 结果 ==="
grep -E "^(===|延拓|求解|  ✅|  ❌)" "$LOG" | tail -12
