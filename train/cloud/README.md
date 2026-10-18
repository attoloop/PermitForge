# Qwen3.8-27B on HF Jobs

WP2.16 uses standard TRL `GRPOTrainer` + `environment_factory` + PEFT LoRA,
with bf16 on one H200 under **Attoloop**. Transformers selects attention and
linear-attention implementations. TRL owns generation, tool turns, masks and
loss. No trainer subclass, persistent-cache implementation, FA3/Liger selection,
TileLang workaround or automatic retry/fallback remains in the cloud path.

Training follows the evaluation's rules: the same instructions, the same opening
line as the frontier Claude harness, the same four tool names and descriptions,
40 turns and 30 evaluations. See [DESIGN.md](DESIGN.md#training-under-the-evaluation-rules)
for every remaining interaction difference. The experiment compares the base and
trained Qwen on this unchanged cloud evaluation to test learnability; it is not
a comparison with the frontier panel.

## Smoke first

One bounded job performs these steps:

1. Verify the input archive and install the frozen cloud project. Trackio is
   installed only when explicitly requested.
2. Run one short bf16 linear-attention numerical check before loading the model.
   Compare outputs, final state and all input gradients with an independent fp32
   recurrence, using flash-linear-attention's own test tolerances. Any failure
   aborts and preserves `smoke/kernels.json` with every tensor's error.
3. Prepare fields for the selected two train and three eval tasks, with at most
   four native workers. Evaluate the **untrained base** on the three eval tasks,
   once each, using the public tools and native terminal verifier.
4. Free the base evaluation model, then take **two direct GRPO optimizer steps**
   on the two train tasks. Require finite loss/weights, changed adapter weights,
   both task identities and exact save/reload equality. Save complete checkpoints.
5. Write `smoke/REPORT.md`, `smoke/metrics.json`, memory measurements and projected
   costs for 20/50/100 steps from the observed step mean.

The provider timeout stays **3 hours**, a **$15 GPU-compute envelope at $5/hour**.
The stage limit stays 10,200 seconds ($14.17); the launcher subtracts setup from
its shared deadline and reserves final-upload time. Storage/taxes are separate.
A timeout or failure never authorizes another paid attempt automatically.

Longer GRPO requires a passed smoke with identical code/data/selection/settings,
a passed kernel sanity check and GPU guard, valid tools, submission, positive
base reward and nonzero training-group reward spread. Smoke readiness is not a
claim of learning or steady-state speed. A smoke with the earlier 8-turn settings
passed on the H200 (Job `Attoloop/6ad478f2c662881452271a2c`); the evaluation-rules
settings need their own passed smoke before any longer run.

## Standard stack

Python 3.12, torch **2.12.1+cu129** (CUDA 12.9 runtime, from the PyTorch cu129
index), Triton **3.7.1**, Transformers **5.17.0**, TRL **1.12.0**, PEFT **0.19.0**,
FLA/fla-core **0.5.2**, HF Hub **1.33.0** and uv **0.11.32** are pinned. Torch
requires that Triton version; it avoids the older compiler range rejected by FLA on
Hopper. The CUDA 12.9 build runs on CUDA 12 drivers: PyPI's CUDA 13 build saw no GPU
on an H200 host with an older driver. No override of Torch's dependencies or kernel
monkeypatch is used. No causal-conv1d wheel exists for torch 2.12, so Transformers
uses its own convolution implementation. See [DESIGN.md](DESIGN.md) for evidence.

Generation uses TRL's default Transformers path. The inspected TRL vLLM adapter
does not pass the pinned revision to its colocated engine, and the native vLLM
multimodal wrapper uses different weight names from this text-only trainer.
Supporting that combination would require custom integration, so this recipe
uses the default generation option permitted by WP2.15. vLLM is not installed.

The model revision remains `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` of
`Qwen/Qwen3.8-27B`. The text-only `Qwen3_5ForCausalLM` loader validates every
language weight and omits vision weights. Defaults are bf16, full-attention
q/k/v/o LoRA rank 16/alpha 32, four generations, accumulation four, 40 turns and
prompt/completion ceilings of 4,096/40,960 tokens (45,056 retained total). Thinking
is off. Replaying the three saved base episodes through installed TRL, including
the submission reply and tool feedback, takes 30,259 / 32,582 / 29,878 completion
tokens with a 983-token opening (the replay adds one synthetic EOS after submit).
The new completion allowance leaves at least 20% headroom. Each generated reply
is capped at the evaluation's 32,768 tokens; sampling matches the pinned model:
temperature 1.0, top_p 0.95, top_k 20. Reproduce the counts with the CPU replay:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=.:envs/permit/core:envs/permit/openenv \
  train/cloud/.venv-verify/bin/python -c \
  'import json; from train.cloud.tests.replay import replay_episodes; print(json.dumps(replay_episodes(), indent=2))'
```

TRL slices cumulative continuations **after** generation. Thus 45,056 bounds the
retained sequence, not transient generation memory: with the reply cap the
configured upper bound is 77,824 tokens. Loss computation still covers the full
retained completion, including masked tool positions. Both require a new GPU
smoke; the earlier memory measurement does not qualify these budgets.
`grpo/episodes.jsonl` records submission, termination reason, budget/turn flags
and completion tokens including tool feedback. `grpo/metrics.json` and the smoke
report count budget cuts per optimizer step; `trainer_metrics.json` retains TRL's
per-step model-generated lengths. Budget-cut rollouts remain in the loss.
The existing explicit `--precision nf4` option is retained,
but the H200 recipe uses bf16. The device guard aborts above 135,000,000,000 bytes;
this is a limit, not evidence of fit. Task identities, prompt, saved-fraction
reward and native evaluation verifier are unchanged. Local small-model `train/`
experiments and historical SFT utilities are separate and remain intact.

## Task selection and local bundle

`selection.py` chooses **30 train tasks**: three per station, ordered by
SHA256 of `WP2.12:` plus task ID, with no policy outcomes. Smoke train tasks are
the first two of these by `(source count, ID)`. Smoke eval tasks are the first
three distinct stations in the frozen ten-task cohort, ordered by the same
structural key. Full base/final evaluation uses that cohort, three repeats each,
as in the frontier panel. No eval task trains.

The archive includes code, the frozen stack, catalogs, the selection, native
binaries, and task/met inputs for the selected thirty train plus ten eval tasks.
Fields, model weights, credentials, rollouts and SFT data are excluded. This
README is excluded so it can record the archive digest without self-reference. All
archive members and converted task/met hashes are verified locally and again
by the job. Large archives remain ignored by git.

WP2.16 rebuilt and verified the retained archive against the working tree. Its
SHA256 (`sha256sum train/cloud/data/cloud-bundle.tar.gz`):

`21f792590a5a2ddfd91106fa3a4e1370f9a8c7088f82fa81a86ef973048a204f`

Run from the repository root (the existing `train` environment can run CPU checks
without installing a second CUDA stack):

```sh
uv lock --project train/cloud --check
uv sync --project train/cloud --frozen
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 \
  uv run --project train/cloud --no-sync python -m pytest \
  -c train/cloud/pyproject.toml train/cloud/tests train/tests -q
uv run --project train/cloud --frozen --no-sync ruff check train/cloud
uv run --project train/cloud --frozen --no-sync python -m train.cloud.bundle \
  --output train/cloud/data/cloud-bundle.tar.gz
BUNDLE_SHA="$(sha256sum train/cloud/data/cloud-bundle.tar.gz | cut -d ' ' -f 1)"
uv run --project train/cloud --frozen --no-sync python -m train.cloud.bundle --verify \
  --output train/cloud/data/cloud-bundle.tar.gz --sha256 "$BUNDLE_SHA"
```

## Plans, costs and continuation

These commands are CPU-only and write nothing:

```sh
uv run --project train/cloud --frozen --no-sync python -m train.cloud.run --plan --recipe smoke
uv run --project train/cloud --frozen --no-sync python -m train.cloud.run --plan --recipe grpo \
  --grpo-steps 100 --stop-after-step 20
uv run --project train/cloud --frozen --no-sync python -m train.cloud.run --plan --recipe grpo --grpo-steps 50
uv run --project train/cloud --frozen --no-sync python -m train.cloud.run --plan --recipe grpo --grpo-steps 100
```

Planning inputs at $5/hour, including 900 seconds of setup, reproduced by those
commands (`estimated_total_usd + setup_prior_usd`):

| Recipe | Estimated compute USD | Provider timeout seconds |
|---|---:|---:|
| Smoke | 9.21 | 10,800 |
| Stop at 20, target 100 | 75.25 | 97,200 |
| Fresh 50, base + final eval | 159.67 | 206,100 |
| Fresh 100, base + final eval | 243.00 | 326,100 |

Full plans are fields → base eval → GRPO → final eval → adapter export. The
20-step check omits final eval/export. Measured on the H200 smoke: 480 CPU-seconds
per field task divided by four workers, 816 seconds per complete 40-turn eval
episode including native grading (mean 815.54 s, max 996.94 s; reproduced by the
replay command above and `test_plan_caps_fit_the_measured_evaluation_timings`)
and 100 seconds for the kernel check. **Estimated until the
evaluation-rules smoke measures it:** 1,200 seconds per GRPO step (289 s was measured
with 8 turns; eval-length rollouts are about four times longer), capped at 2,400
seconds per step. Smoke save/reload is 180 seconds. Evaluation stages are capped at
30 × 1,200 seconds. The supervisor/provider enforce the hard wall/dollar limits,
including JIT. Use the evaluation-rules smoke's measurements before choosing caps or
launching longer work. `--wall-cap STAGE=SECONDS` and
`--usd-cap STAGE=USD` may tighten caps; resumed attempts consume the same cumulative
allowance. Unknown crash charges consume the full reservation.

For a check intended to continue to 100, set `--grpo-steps 100` from its first
invocation and add `--stop-after-step 20`. The callback saves a complete checkpoint
and stops at 20 while keeping the 100-step scheduler horizon. Resume the identical
run/output/artifact prefix with `--restore` and omit `--stop-after-step`. Optimizer,
RNG, scheduler and step are restored; partial checkpoints are ignored. Completed
fields/base episodes are skipped and training continues from 20. Do not change
`--grpo-steps` from 20 to 100: immutable run fingerprints reject that change.
A separate fresh 50-step run uses its own output and artifact prefix.

Longer jobs pass `--qualification-run /previous/<passed evaluation-rules smoke>`, with the run bucket
mounted read-only at `/previous`. Their `--job-seconds` must match the provider
`--timeout` printed by the plan. All jobs use the same workspace path. `run.json`
freezes code/data/settings, and payloads upload before completion markers. Failed
or interrupted native submissions remain pending for review and are not retried
selectively. An upload failure stops progression. The field cache is bounded at
128 GiB; misses regenerate within the requesting stage's allowance, four workers
maximum. Native grading runs in a small subprocess independent of model RAM; it
receives the worker's field-cache volume explicitly, because a fresh process does not
inherit the worker's in-memory cache binding.

## Operator commands: create, upload, launch, watch, cancel

**The commands below write to HF or spend money; they were not executed here.**
They use the inspected HF Hub CLI **1.33.0**, which supports `h200`, `--volume`
and `--namespace`. Set `HF_TOKEN` securely in your environment first. The named
input dataset and run bucket below are private and under Attoloop.

```sh
# Create destinations; uploads remain private.
uvx --from 'huggingface-hub==1.33.0' hf repos create \
  Attoloop/permitforge-qwen38-inputs --repo-type dataset --private
uvx --from 'huggingface-hub==1.33.0' hf buckets create \
  Attoloop/permitforge-qwen38-runs --private

# Upload the locally verified archive; retain this hash with the launch record.
BUNDLE_SHA="$(sha256sum train/cloud/data/cloud-bundle.tar.gz | cut -d ' ' -f 1)"
uvx --from 'huggingface-hub==1.33.0' hf upload \
  Attoloop/permitforge-qwen38-inputs train/cloud/data/cloud-bundle.tar.gz \
  cloud-bundle.tar.gz --repo-type dataset

# One smoke Job, including setup. Trackio is off.
JOB_ID="$(uvx --from 'huggingface-hub==1.33.0' hf jobs uv run \
  --namespace Attoloop --flavor h200 --timeout 3h --secrets HF_TOKEN \
  --name permitforge-qwen38-smoke-evalrules --detach --quiet \
  --volume hf://datasets/Attoloop/permitforge-qwen38-inputs:/inputs:ro \
  train/cloud/hf_job.py --bundle /inputs/cloud-bundle.tar.gz \
  --bundle-sha256 "$BUNDLE_SHA" --workspace /workspace/permit-cloud \
  --job-seconds 10800 -- --execute --recipe smoke --hourly-usd 5.00 \
  --precision bf16 --output train/cloud/runs/smoke-evalrules \
  --artifact-uri hf://buckets/Attoloop/permitforge-qwen38-runs/smoke-evalrules)"

# Keep a local log copy; inspect the final Job status after the stream ends.
uvx --from 'huggingface-hub==1.33.0' hf jobs logs \
  --namespace Attoloop "$JOB_ID" --follow | tee train/cloud/data/smoke-logs.txt
uvx --from 'huggingface-hub==1.33.0' hf jobs inspect --namespace Attoloop "$JOB_ID"

# Cancel immediately if needed (use a second terminal while following logs).
uvx --from 'huggingface-hub==1.33.0' hf jobs cancel --namespace Attoloop "$JOB_ID"

# Read back artifacts after completion; destination must be a fresh directory.
uvx --from 'huggingface-hub==1.33.0' hf sync \
  hf://buckets/Attoloop/permitforge-qwen38-runs/smoke-evalrules train/cloud/data/smoke-result
cat train/cloud/data/smoke-result/smoke/REPORT.md
```

Trackio is opt-in: after the operator names its Space, append
`--trackio-space Attoloop/CHOSEN_SPACE --trackio-bucket Attoloop/CHOSEN_METRICS_BUCKET`
to the **run arguments after `--`**. This installs the pinned `metrics` extra and
logs trainer metrics to that live Space, using Transformers' Trackio integration
as FineEnvs does. Static Space export is disabled. Trackio can create/update the
named Space/bucket; omitting the flag means no Trackio initialization or writes.
No Space name has been chosen by this work. The private run bucket independently
retains training logs, checkpoints, traces, GPU/kernel evidence and smoke reports.
