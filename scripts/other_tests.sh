#!/usr/bin/env bash
# test_clamp_contact.py 与 test_clamp_loads.py 在 mpirun -n 6 下的表现。
set -u
source /root/miniforge3/etc/profile.d/conda.sh
conda activate pf-wire || exit 1
cd /root/pf-wire || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

for f in tests/test_clamp_contact.py tests/test_clamp_loads.py; do
    echo "=== $f ==="
    timeout 400 mpirun -n 6 python -u -m pytest "$f" -q --tb=line \
        -p no:cacheprovider --timeout=300 2>&1 \
        | grep -E 'passed|failed|AssertionError|Error|FAILED' | head -8
    echo "  (exit=$?)"
done
echo "=== done ==="
