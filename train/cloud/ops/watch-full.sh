#!/bin/bash
# Poll one job every 5 minutes until it ends: status, progress read from the run bucket (GRPO groups and their
# rewards, or finished evaluation episodes), new console error lines, metric-file changes.
#   bash train/cloud/ops/watch-full.sh JOB_ID hf://buckets/<org>/<bucket>/<run> <run> <stage> <phase>
# Env: NAMESPACE (Attoloop), LAUNCH_DIR (train/cloud/data/launches). Needs HF_TOKEN.
JOB=${1:?job id}; PREFIX=${2:?artifact prefix}; RUN=${3:?run name}; STAGE=${4:?stage}; PHASE=${5:?phase}
export LC_ALL=C
cd "$(dirname "$0")/../../.." || exit 2
: "${HF_TOKEN:?set HF_TOKEN}"
NAMESPACE=${NAMESPACE:-Attoloop}
LAUNCH_DIR=${LAUNCH_DIR:-train/cloud/data/launches}
W=$LAUNCH_DIR/state/$RUN-$PHASE
mkdir -p "$W"; touch "$W/console.prev"
LOG=$LAUNCH_DIR/$RUN.watch.log
HF="uvx --from huggingface-hub==1.33.0 hf"
PY="uvx --from huggingface-hub==1.33.0 python"
n=0
while true; do
  n=$((n+1)); ts=$(date -u +%H:%M)
  status=$($PY -c "from huggingface_hub import HfApi; j=HfApi().inspect_job(job_id='$JOB', namespace='$NAMESPACE'); print(j.status.stage, (j.status.message or '').replace(chr(10),' ')[:100])" 2>/dev/null)
  $HF buckets cp "$PREFIX/$STAGE/console.log" "$W/console.cur" >/dev/null 2>&1 || true
  if [ -s "$W/console.cur" ]; then
    comm -13 <(sort -u "$W/console.prev") <(sort -u "$W/console.cur") | grep -E "Traceback|Error|error|failed|Killed|OOM|'loss'" | cut -c1-300 | sed "s/^/$ts [$PHASE] CONSOLE /" >>"$LOG"
    cp "$W/console.cur" "$W/console.prev"
  fi
  for f in $STAGE/metrics.json $STAGE/gpu.json; do
    b=$(basename "$f")
    $HF buckets cp "$PREFIX/$f" "$W/$b.cur" >/dev/null 2>&1 && \
      { cmp -s "$W/$b.cur" "$W/$b.prev" 2>/dev/null || { echo "$ts [$PHASE] FILE $f $(python3 -c "import json;d=json.load(open('$W/$b.cur'));print(json.dumps({k:d[k] for k in d if k in ('status','peak_device_bytes','optimizer_steps','seconds_per_step','ending_counts','worst_case','episodes')})[:400])")" >>"$LOG"; cp "$W/$b.cur" "$W/$b.prev"; }; }
  done
  prog=""
  if [ "$STAGE" = grpo ]; then
    $HF buckets cp "$PREFIX/grpo/episodes.jsonl" "$W/episodes.jsonl" >/dev/null 2>&1 || true
    prog=$(python3 - "$W/episodes.jsonl" <<'EOF'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if p.exists():
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    steps = sorted({r['optimizer_step'] for r in rows})
    last = [r for r in rows if r['optimizer_step'] == steps[-1]] if steps else []
    rewards = [round((r.get('final') or {}).get('reward', 0) or 0, 2) for r in last]
    ends = {}
    for r in last: ends[r['ended']] = ends.get(r['ended'], 0) + 1
    print(f"groups={len(steps)} rollouts={len(rows)} last_step={steps[-1]+1 if steps else 0} last_rewards={rewards} last_endings={ends}")
else:
    print('no groups yet')
EOF
)
  else
    prog="episodes_done=$($HF buckets ls -R "$PREFIX/$STAGE" 2>/dev/null | grep -c '/episode.json$') pending_grades=$($HF buckets ls -R "$PREFIX/$STAGE" 2>/dev/null | grep -c '/pending-grade.json$')"
  fi
  echo "$ts [$PHASE] STATUS ${status:-unknown} poll=$n $prog" >>"$LOG"
  case "$status" in
    COMPLETED*|ERROR*|CANCELED*|CANCELLED*|FAILED*) echo "$ts [$PHASE] TERMINAL $status" >>"$LOG"; exit 0;;
  esac
  sleep 300
done
