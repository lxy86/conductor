#!/usr/bin/env bash
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
LOG="data/_u_block.log"
: > "$LOG"
python -u data/_probe_u_block.py >> "$LOG" 2>&1
echo "rc=$?"
echo "=== 结果 ==="
grep -E "^(===|相位|  ✅|  ❌|    \[|  ⇒)" "$LOG" | tail -20
