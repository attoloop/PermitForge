#!/bin/bash
# Poll a smoke job every 2 minutes: job log lines, console lines and metric files from its bucket, until it ends.
#   bash train/cloud/ops/watch-smoke.sh JOB_ID hf://buckets/<org>/<bucket>/<smoke-run>
# Env: NAMESPACE (Attoloop), LAUNCH_DIR (train/cloud/data/launches). Needs HF_TOKEN.
JOB=${1:?job id}; PREFIX=${2:?artifact prefix}
export LC_ALL=C
cd "$(dirname "$0")/../../.." || exit 2
: "${HF_TOKEN:?set HF_TOKEN}"
NAMESPACE=${NAMESPACE:-Attoloop}
LAUNCH_DIR=${LAUNCH_DIR:-train/cloud/data/launches}
W=$LAUNCH_DIR/state/smoke-$JOB
mkdir -p "$W"; touch "$W/console.prev" "$W/joblog.prev"
LOG=$LAUNCH_DIR/smoke-evalrules.watch.log
HF="uvx --from huggingface-hub==1.33.0 hf"
PY="uvx --from huggingface-hub==1.33.0 python"
n=0
while true; do
  n=$((n+1)); ts=$(date -u +%H:%M:%S)
  status=$($PY -c "from huggingface_hub import HfApi; j=HfApi().inspect_job(job_id='$JOB', namespace='$NAMESPACE'); print(j.status.stage, (j.status.message or '').replace(chr(10),' ')[:120])" 2>/dev/null)
  $HF jobs logs --namespace "$NAMESPACE" "$JOB" >"$W/joblog.cur" 2>/dev/null || true
  if [ -s "$W/joblog.cur" ]; then
    comm -13 <(sort -u "$W/joblog.prev") <(sort -u "$W/joblog.cur") | sed "s/^/$ts JOBLOG /" >>"$LOG"
    cp "$W/joblog.cur" "$W/joblog.prev"
  fi
  $HF buckets cp "$PREFIX/smoke/console.log" "$W/console.cur" >/dev/null 2>&1 || true
  if [ -s "$W/console.cur" ]; then
    comm -13 <(sort -u "$W/console.prev") <(sort -u "$W/console.cur") | sed "s/^/$ts CONSOLE /" >>"$LOG"
    cp "$W/console.cur" "$W/console.prev"
  fi
  for f in smoke/gpu.json smoke/grpo/metrics.json smoke/metrics.json smoke/base-eval/index.json; do
    b=$(basename "$f")
    $HF buckets cp "$PREFIX/$f" "$W/$b.cur" >/dev/null 2>&1 && \
      { cmp -s "$W/$b.cur" "$W/$b.prev" 2>/dev/null || { echo "$ts FILE $f $(python3 -c "import json;d=json.load(open('$W/$b.cur'));print(json.dumps({k:d[k] for k in d if k in ('status','peak_device_bytes','optimizer_steps','seconds_per_step','ending_counts','budget_cuts_per_step','groups','task_mean_saved_fraction','episodes','ready_for_grpo','worst_case')}))")" >>"$LOG"; cp "$W/$b.cur" "$W/$b.prev"; }; }
  done
  echo "$ts STATUS $status poll=$n console_lines=$(wc -l <"$W/console.cur" 2>/dev/null || echo 0)" >>"$LOG"
  case "$status" in
    COMPLETED*|ERROR*|CANCELED*|CANCELLED*|FAILED*) echo "$ts TERMINAL $status" >>"$LOG"; exit 0;;
  esac
  sleep 120
done
