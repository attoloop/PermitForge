#!/bin/bash
# Launch the evaluation-rules smoke on HF Jobs after a good one-second probe. Retry only the fast "no visible
# CUDA GPU" start failure (driver not injected), at most MAX attempts; any other failure stops. Once an attempt
# survives the guard, watch-smoke.sh polls it. Run from the repository root with HF_TOKEN set.
#   bash train/cloud/ops/smoke-attempts.sh d          # round letter; artifacts go to <bucket>/smoke-evalrules-d1
# Env: MAX (1), NAMESPACE (Attoloop), INPUTS, BUCKET, LAUNCH_DIR, BASE_EVAL_FROM (a passed smoke's base-eval to carry).
set -u
export LC_ALL=C
cd "$(dirname "$0")/../../.." || exit 2
: "${HF_TOKEN:?set HF_TOKEN}"
OPS=train/cloud/ops
ROUND=${1:?round letter}
MAX=${MAX:-1}
NAMESPACE=${NAMESPACE:-Attoloop}
INPUTS=${INPUTS:-hf://datasets/$NAMESPACE/permitforge-qwen38-inputs}
BUCKET=${BUCKET:-hf://buckets/$NAMESPACE/permitforge-qwen38-runs}
LAUNCH_DIR=${LAUNCH_DIR:-train/cloud/data/launches}
mkdir -p "$LAUNCH_DIR"
A=$LAUNCH_DIR/smoke-evalrules.attempts.log
REC=$LAUNCH_DIR/smoke-evalrules.json
HF="uvx --from huggingface-hub==1.33.0 hf"
SHA=$(sha256sum train/cloud/data/cloud-bundle.tar.gz | cut -d ' ' -f 1)
CARRY=${BASE_EVAL_FROM:+--base-eval-from $BASE_EVAL_FROM}
log(){ echo "$(date -u +%H:%M:%S) $*" >>"$A"; }
status(){ uvx --from huggingface-hub==1.33.0 python -c "from huggingface_hub import HfApi; print(HfApi().inspect_job(job_id='$1', namespace='$NAMESPACE').status.stage)" 2>/dev/null; }

for p in 1 2 3; do
  out=$($HF jobs run --namespace "$NAMESPACE" --flavor h200 --timeout 5m --name gpu-probe-$ROUND$p python:3.12-slim bash -c \
        'echo NVRM $(sed -n "s/.*Module for x86_64 *\([0-9.]*\).*/\1/p" /proc/driver/nvidia/version); test -e /usr/lib64/libcuda.so.1 && echo LIBS_OK || echo LIBS_MISSING' 2>/dev/null \
        | grep -E "^(NVRM|LIBS_)" | tr '\n' ' ')
  log "probe $ROUND$p: ${out:-no output}"
  case "$out" in *LIBS_OK*) break;; esac
  [ $p -lt 3 ] && sleep 180
done

for n in $(seq 1 "$MAX"); do
  PREFIX=$BUCKET/smoke-evalrules-$ROUND$n
  JOB=$($HF jobs uv run --namespace "$NAMESPACE" --flavor h200 --timeout 3h --secrets HF_TOKEN \
      --name permitforge-qwen38-smoke-evalrules-$ROUND$n --detach --quiet \
      --volume "$INPUTS:/inputs:ro" \
      train/cloud/hf_job.py --bundle /inputs/cloud-bundle.tar.gz --bundle-sha256 "$SHA" \
      --workspace /workspace/permit-cloud --job-seconds 10800 -- --execute --recipe smoke --hourly-usd 5.00 \
      --precision bf16 --output train/cloud/runs/smoke-evalrules --artifact-uri "$PREFIX" $CARRY 2>>"$A" \
      | grep -oE '[0-9a-f]{24}' | head -1)
  if [ -z "$JOB" ]; then log "attempt $n: launch returned no job id; stopping"; exit 1; fi
  log "attempt $n: launched job $JOB artifact $PREFIX"
  printf '{"job_id":"%s","attempt":%s,"bundle_sha256":"%s","commit":"%s","started":"%s","recipe":"smoke","artifact":"%s"}\n' \
      "$JOB" "$n" "$SHA" "$(git rev-parse --short HEAD)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PREFIX" >"$REC"
  t0=$(date +%s); st=""
  while :; do
    sleep 15
    st=$(status "$JOB"); el=$(( $(date +%s) - t0 ))
    case "$st" in ERROR*|CANCELED*|CANCELLED*|FAILED*|COMPLETED*) break;; esac
    [ $el -ge 300 ] && break
  done
  log "attempt $n: status '${st:-unknown}' after ${el}s"
  case "$st" in
    ERROR*)
      $HF buckets cp "$PREFIX/smoke/console.log" "$LAUNCH_DIR/smoke-$ROUND$n.console.log" >/dev/null 2>&1 || true
      if grep -q "visible CUDA GPU" "$LAUNCH_DIR/smoke-$ROUND$n.console.log" 2>/dev/null; then
        log "attempt $n: bad node; retrying"; continue
      fi
      log "attempt $n: ERROR for another reason; NOT retrying"; exit 1;;
    CANCELED*|CANCELLED*|FAILED*) log "attempt $n: ended '$st'; stopping"; exit 1;;
  esac
  log "attempt $n: survived the guard; handing over to the watcher"
  exec bash $OPS/watch-smoke.sh "$JOB" "$PREFIX"
done
log "all $MAX attempt(s) hit bad nodes; stopping"
exit 1
