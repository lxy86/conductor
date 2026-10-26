#!/usr/bin/env bash
# Phase 2.5 S–N 扫描驱动（PLAN 2.5，硬约束 5：跳过已完成工况、崩了能续）。
#
# 设计要点（都是被这个项目的实际故障逼出来的）：
#  * **跳过已完成**：result.json 里是终态就不重跑；
#  * **崩了自动续**：有断点就 --resume，最多重试 $MAX_RETRY 次。
#    这台机器上 WSL 只要没有客户端挂着就关停，会把算例一起带走
#    （已经发生过一次，见 D-31/D-32 那一轮），断点 + 自动续跑是唯一的解；
#  * **事件写文件**：每次开始/结束/失败都带时间戳追加到 $EVENTS，
#    不依赖谁盯着终端，也不依赖哪个会话还活着；
#  * **不静默失败**：工况跑完立刻核对 result.json 的状态，不是终态就大声报。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
conda activate "${PFWIRE_ENV:-pf-wire}" || exit 1
cd "$PFWIRE_HOME" || exit 1
export OMP_NUM_THREADS=1

NP=${NP:-6}                 # 一核一 rank，见 CLAUDE.md：绝不要 -n 12
MAX_RETRY=${MAX_RETRY:-3}
EVENTS=${EVENTS:-$PFWIRE_HOME/data/_sweep_events.log}
ONLY=${ONLY:-}              # 只跑某个工况（调试用）

say () { echo "[$(date -u +%m-%d\ %H:%M:%S)] $*" | tee -a "$EVENTS"; }

terminal_status () {   # $1 = 工况名 -> 0 表示已是终态
  python3 - "$1" <<'PY'
import json, pathlib, sys
p = pathlib.Path("data")/sys.argv[1]/"result.json"
if not p.is_file():
    raise SystemExit(1)
try:
    st = json.loads(p.read_text(encoding="utf-8")).get("status")
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if st in ("failed", "failed_unstable", "runout", "completed") else 1)
PY
}

say "===== 扫描开始（NP=$NP，最多重试 $MAX_RETRY 次）====="
# **按 σ_a 从高到低跑**：低应力寿命长得多（可能到 N_max = 1e8 runout），
# 先跑快的能尽早暴露问题，也能早点拿到 S–N 曲线的多数点。
# 按文件名顺序会把最贵的 20 MPa 排在最前，等一天才知道机器有没有问题。
for cfg in $(ls configs/sweep/*.yaml | sort -r); do
  name=$(basename "$cfg" .yaml)
  [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && continue

  if terminal_status "$name"; then
    say "$name 已完成，跳过"
    continue
  fi

  for try in $(seq 0 "$MAX_RETRY"); do
    if [ -f "data/$name/checkpoint.json" ]; then
      FLAG="--resume"; WHAT="续跑"
    else
      FLAG=""; WHAT="新跑"
    fi
    say "$name $WHAT（第 $((try+1)) 次尝试）"
    PYTHONUNBUFFERED=1 mpirun -n "$NP" python -m pfwire.run_case "$cfg" $FLAG -v \
      >> "data/${name}_run.log" 2>&1
    rc=$?
    if terminal_status "$name"; then
      say "$name 完成：$(python3 -c "
import json,pathlib
r=json.loads(pathlib.Path('data/$name/result.json').read_text(encoding='utf-8'))
print('status=%s N_f=%s 关卡=%s' % (r['status'], r.get('N_f'),
      (r.get('gate') or {}).get('verdict')))
")"
      break
    fi
    say "$name 未达终态（exit=$rc），$( [ "$try" -lt "$MAX_RETRY" ] && echo '准备重试' || echo '**放弃，需要人看**' )"
  done
done
say "===== 扫描结束 ====="
python scripts/sweep_status.py | tee -a "$EVENTS"
echo SWEEP_DONE
