#!/bin/bash
# The phase sequence used for run full-30, as separate HF jobs: GRPO in chunks of five steps (each job resumes
# from the last complete checkpoint), then base-eval and eval as two concurrent jobs on the same run prefix
# (hf sync never deletes, so they do not disturb each other), then the adapter export.
#   STEPS=30 bash train/cloud/ops/chain-full.sh full-30
# Env: STEPS (30), CHUNK (5), NAMESPACE, BUCKET, LAUNCH_DIR as in run-full.sh. Needs HF_TOKEN.
set -u
cd "$(dirname "$0")/../../.." || exit 2
OPS=train/cloud/ops
RUN=${1:?run name}
STEPS=${STEPS:-30}; CHUNK=${CHUNK:-5}
LAUNCH_DIR=${LAUNCH_DIR:-train/cloud/data/launches}
LOG=$LAUNCH_DIR/$RUN.chain.log
WATCH=$LAUNCH_DIR/$RUN.watch.log
mkdir -p "$LAUNCH_DIR"
now(){ date -u +%H:%M; }
terminal(){ grep "\[$1\] TERMINAL" "$WATCH" 2>/dev/null | tail -1; }
phase(){ name=$1; stage=$2; timeout=$3; shift 3
  echo "$(now) phase $name START stage=$stage timeout=$timeout args: $*" >>"$LOG"
  STEPS=$STEPS TIMEOUT=$timeout PHASE=$name STAGE=$stage MAX=3 bash $OPS/run-full.sh "$RUN" "$@"
  rc=$?
  last=$(terminal "$name")
  echo "$(now) phase $name END rc=$rc ${last:-no terminal line}" >>"$LOG"
  case "$last" in *COMPLETED*) return 0;; esac
  return 1
}

echo "$(now) chain started: $STEPS steps in chunks of $CHUNK" >>"$LOG"
restore=""
for stop in $(seq "$CHUNK" "$CHUNK" "$STEPS"); do
  if [ "$stop" -ge "$STEPS" ]; then
    phase "g$stop" grpo 12h $restore --stage grpo || { echo "$(now) CHAIN STOPPED at step $stop" >>"$LOG"; exit 1; }
  else
    phase "g$stop" grpo 12h $restore --stage grpo --stop-after-step "$stop" || { echo "$(now) CHAIN STOPPED at step $stop" >>"$LOG"; exit 1; }
  fi
  restore="--restore"
done

echo "$(now) launching base-eval and eval in parallel" >>"$LOG"
phase base-eval base-eval 12h --restore --stage base-eval & p2=$!
sleep 240   # let the first launch finish its probe before the second probes, so they do not race for one slot
phase eval eval 12h --restore --stage eval & p3=$!
wait $p2; r2=$?; wait $p3; r3=$?
echo "$(now) evaluations ended: base-eval rc=$r2, eval rc=$r3" >>"$LOG"
if [ $r2 -ne 0 ] || [ $r3 -ne 0 ]; then echo "$(now) CHAIN STOPPED after the evaluations" >>"$LOG"; exit 1; fi

phase export export 1h --restore --stage export || { echo "$(now) CHAIN STOPPED after export" >>"$LOG"; exit 1; }
echo "$(now) ALL PHASES COMPLETE" >>"$LOG"
