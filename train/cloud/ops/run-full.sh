#!/bin/bash
# One phase of a run on HF Jobs. A one-second probe job finds a host whose container runtime injects the CUDA
# libraries; the real job is launched right after a good probe so it takes that slot. A start that fails with
# "no visible CUDA GPU" is retried up to MAX times; any other failure stops. Then watch-full.sh polls until the
# job ends. Run from the repository root with HF_TOKEN set.
#   STEPS=30 TIMEOUT=12h PHASE=1a STAGE=grpo bash train/cloud/ops/run-full.sh full-30 --stage grpo --stop-after-step 15
# Env: STEPS, TIMEOUT (12h), PHASE (a label), STAGE (grpo|base-eval|eval|export), MAX (3), NAMESPACE (Attoloop),
# INPUTS (dataset with the bundle), BUCKET (run bucket), QUAL (qualification smoke path inside /runs),
# LAUNCH_DIR (where logs and launch records go; default train/cloud/data/launches).
set -u
export LC_ALL=C
cd "$(dirname "$0")/../../.." || exit 2
: "${HF_TOKEN:?set HF_TOKEN}"
OPS=train/cloud/ops
RUN=${1:?run name}; shift
EXTRA="$*"
STEPS=${STEPS:?steps}; TIMEOUT=${TIMEOUT:?timeout e.g. 12h}; PHASE=${PHASE:?phase}; STAGE=${STAGE:?stage}; MAX=${MAX:-3}
NAMESPACE=${NAMESPACE:-Attoloop}
INPUTS=${INPUTS:-hf://datasets/$NAMESPACE/permitforge-qwen38-inputs}
BUCKET=${BUCKET:-hf://buckets/$NAMESPACE/permitforge-qwen38-runs}
QUAL=${QUAL:-/runs/smoke-evalrules-d1}
LAUNCH_DIR=${LAUNCH_DIR:-train/cloud/data/launches}
mkdir -p "$LAUNCH_DIR"
A=$LAUNCH_DIR/$RUN.attempts.log
REC=$LAUNCH_DIR/$RUN-$PHASE.json
HF="uvx --from huggingface-hub==1.33.0 hf"
SHA=$(sha256sum train/cloud/data/cloud-bundle.tar.gz | cut -d ' ' -f 1)
PREFIX=$BUCKET/$RUN
HOURS=$(echo "$TIMEOUT" | sed 's/h$//')
log(){ echo "$(date -u +%H:%M:%S) [$PHASE] $*" >>"$A"; }
status(){ uvx --from huggingface-hub==1.33.0 python -c "from huggingface_hub import HfApi; print(HfApi().inspect_job(job_id='$1', namespace='$NAMESPACE').status.stage)" 2>/dev/null; }

for p in 1 2 3; do
  out=$($HF jobs run --namespace "$NAMESPACE" --flavor h200 --timeout 5m --name gpu-probe-$RUN-$PHASE-$p python:3.12-slim bash -c \
        'echo NVRM $(sed -n "s/.*Module for x86_64 *\([0-9.]*\).*/\1/p" /proc/driver/nvidia/version); test -e /usr/lib64/libcuda.so.1 && echo LIBS_OK || echo LIBS_MISSING' 2>/dev/null \
        | grep -E "^(NVRM|LIBS_)" | tr '\n' ' ')
  log "probe $p: ${out:-no output}"
  case "$out" in *LIBS_OK*) break;; esac
  [ $p -lt 3 ] && sleep 180
done

for n in $(seq 1 "$MAX"); do
  JOB=$($HF jobs uv run --namespace "$NAMESPACE" --flavor h200 --timeout "$TIMEOUT" --secrets HF_TOKEN \
      --name permitforge-qwen38-$RUN-$PHASE-$n --detach --quiet \
      --volume "$INPUTS:/inputs:ro" --volume "$BUCKET:/runs:ro" \
      train/cloud/hf_job.py --bundle /inputs/cloud-bundle.tar.gz --bundle-sha256 "$SHA" \
      --workspace /workspace/permit-cloud --job-seconds $(( HOURS * 3600 )) -- \
      --execute --recipe grpo --grpo-steps "$STEPS" --hourly-usd 5.00 --precision bf16 \
      --qualification-run "$QUAL" --output train/cloud/runs/$RUN --artifact-uri "$PREFIX" $EXTRA 2>>"$A" \
      | grep -oE '[0-9a-f]{24}' | head -1)
  if [ -z "$JOB" ]; then log "attempt $n: launch returned no job id; stopping"; exit 1; fi
  log "attempt $n: launched job $JOB artifact $PREFIX steps $STEPS timeout $TIMEOUT args '$EXTRA'"
  printf '{"job_id":"%s","phase":"%s","attempt":%s,"bundle_sha256":"%s","commit":"%s","started":"%s","recipe":"grpo","steps":%s,"timeout":"%s","args":"%s","qualification":"%s","artifact":"%s"}\n' \
      "$JOB" "$PHASE" "$n" "$SHA" "$(git rev-parse --short HEAD)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$STEPS" "$TIMEOUT" "$EXTRA" "$QUAL" "$PREFIX" >"$REC"
  t0=$(date +%s); st=""
  while :; do
    sleep 20
    st=$(status "$JOB"); el=$(( $(date +%s) - t0 ))
    case "$st" in ERROR*|CANCELED*|CANCELLED*|FAILED*|COMPLETED*) break;; esac
    [ $el -ge 420 ] && break
  done
  log "attempt $n: status '${st:-unknown}' after ${el}s"
  case "$st" in
    ERROR*)
      $HF buckets cp "$PREFIX/$STAGE/console.log" "$LAUNCH_DIR/$RUN-$PHASE-$n.console.log" >/dev/null 2>&1 || true
      if grep -q "visible CUDA GPU" "$LAUNCH_DIR/$RUN-$PHASE-$n.console.log" 2>/dev/null || \
         $HF jobs logs --namespace "$NAMESPACE" "$JOB" 2>/dev/null | grep -q "visible CUDA GPU"; then
        log "attempt $n: bad node (driver not injected); retrying"; continue
      fi
      log "attempt $n: ERROR for another reason; NOT retrying"; exit 1;;
    CANCELED*|CANCELLED*|FAILED*) log "attempt $n: ended '$st'; stopping"; exit 1;;
  esac
  log "attempt $n: survived the guard; handing over to the watcher"
  exec bash $OPS/watch-full.sh "$JOB" "$PREFIX" "$RUN" "$STAGE" "$PHASE"
done
log "all $MAX attempt(s) hit bad nodes; stopping"
exit 1
