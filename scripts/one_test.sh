#!/usr/bin/env bash
# 逐个跑 test_clamp_uzawa.py 的 15 个测试（mpirun -n 6），找出哪些失败、为什么。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

TESTS=$(grep -oE '^def (test_[A-Za-z0-9_]+)' tests/test_clamp_uzawa.py | sed 's/^def //')
echo "=== 共 $(echo "$TESTS" | wc -l) 个测试 ==="

for t in $TESTS; do
    timeout 150 mpirun -n 6 python -u -m pytest "tests/test_clamp_uzawa.py::$t" \
        -q --tb=line -p no:cacheprovider --timeout=120 > /tmp/_t.log 2>&1
    rc=$?
    summary=$(grep -oE '[0-9]+ (passed|failed|skipped)[^=]*' /tmp/_t.log | tail -1)
    if [ "$rc" -eq 0 ]; then
        echo "✅ $t   $summary"
    else
        echo "❌ $t   rc=$rc  $summary"
        # 打第一行错误/断言
        grep -E '^(/root|E  |FAILED|AssertionError|RuntimeError|[A-Za-z]*Error)' /tmp/_t.log \
            | head -2 | sed 's/^/       /'
    fi
done
echo "=== done ==="
