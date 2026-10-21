# Qwen3.8-27B on HF Jobs

GRPO on PermitForge with standard TRL: `GRPOTrainer` with `environment_factory`, PEFT LoRA,
bf16 on one H200. One method is overridden (`chunked_logps.py`, see [DESIGN.md](DESIGN.md#loss-memory)):
the loss computes per-token log-probabilities in 2,048-token chunks, because the full-completion
logits of a 248k-token vocabulary do not fit next to the model.

Training follows the evaluation's rules: the same instructions, the same opening line, the same
four tools, 40 turns and 30 evaluations. The question answered is learnability: does the trained
Qwen beat the base Qwen on the unchanged cloud evaluation? It is not a comparison with the
frontier panel in `results/eval-v2.json`; the harnesses differ
([DESIGN.md](DESIGN.md#training-under-the-evaluation-rules)).

**Result: no.** Base 0.4394 and trained 0.4390 mean saved fraction on the same 10 held-out tasks,
3 repeats each. Every number, the per-task table, the training curves, the cost ledger and the
failures are in [RESULTS.md](RESULTS.md); the card is `results/eval-v2-qwen.json`.

## The run

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, text-only loader, thinking off |
| Adapter | LoRA rank 16, alpha 32, dropout 0, on q/k/v/o of the full-attention layers |
| Steps | 30 optimizer steps, one task per step, 6 rollouts per task, gradient accumulation 6 |
| Optimizer | `paged_adamw_8bit`, learning rate 3e-5 with linear decay to 0 over the 30 steps, beta 0, group reward scaling |
| Episode | 40 tool turns, 30 evaluations, 32,768 tokens per reply, 40,960 completion tokens, 45,056 tokens kept in total |
| Sampling | temperature 1.0, top_p 0.95, top_k 20 (the model's pinned generation config) |
| Tasks | 30 train tasks, 3 per train station, chosen by hash (`selection.py`); the 10 eval tasks of the frontier card, 3 repeats |
| Reward | feasible saved fraction of the draft cost from the fast evaluator during training; native BPIP/AERMOD for every evaluation episode |
| Memory | 105 GB peak reserved by torch; the device guard aborts above 135 GB |
| Cost | $163.69 for every job of the project, $82 of it training; see RESULTS.md |

The run was `full-30` (commit `7e4caaf`, bundle `9696466c...`, qualified by smoke `smoke-evalrules-d1`),
as separate HF jobs so that a dead job loses little:

| Phase | Arguments after `--execute --recipe grpo --grpo-steps 30 ...` | Outcome |
|---|---|---|
| grpo to 15 | `--stage grpo --stop-after-step 15` | died at step 15 on the device watchdog; checkpoint 10 kept |
| grpo to 15, 20, 25 | `--restore --stage grpo --stop-after-step N` | resumed from the last checkpoint, 5 steps each |
| grpo to 30 | `--restore --stage grpo` | adapter and checkpoint 30 saved |
| base-eval, eval | `--restore --stage base-eval` and `--restore --stage eval`, in parallel | 30 episodes each, all submitted feasible designs |
| export | `--restore --stage export` | adapter, metrics and run manifest collected |

`train/cloud/ops/` holds the launch scripts, the probe that avoids hosts without CUDA libraries,
and the watcher. The run folder (episodes, trainer state, console logs, checkpoints, evaluation
episodes with native grades) is in the dataset under `training/runs/full-30`.

## Smoke first

A longer run needs a passed smoke with the same code, data, selection and settings. One 3-hour job:

1. Verify the input archive and install the frozen project.
2. Run a short bf16 linear-attention check against an fp32 recurrence with the kernel library's own tolerances; any failure aborts.
3. Prepare fields for two train and three eval tasks. Evaluate the untrained base on the three eval tasks with the native verifier (or carry a passed base evaluation with `--base-eval-from`).
4. Take two GRPO steps on the two train tasks: finite loss, changed adapter weights, exact save and reload.
5. Run one loss step at the full token budget (40,960 completion tokens, random ids) and record peak memory; the smoke qualifies a run only if it fits.
6. Write `smoke/REPORT.md` and `smoke/metrics.json` with step times and cost projections.

`smoke-evalrules-d1` passed: 12 of 12 rollouts submitted, 1,506 and 608 s per step, worst-case
step 105.4 GB reserved. The smokes before it, including the one that ran out of memory, are in
the dataset under `training/runs/smoke-*`.

## Stack

Python 3.12, torch 2.12.1+cu129, Triton 3.7.1, Transformers 5.17.0, TRL 1.12.0, PEFT 0.19.0,
flash-linear-attention 0.5.2, huggingface-hub 1.33.0, uv 0.11.32, all pinned in `uv.lock`. The
CUDA 12.9 build runs on CUDA 12 drivers; the CUDA 13 build saw no GPU on an older-driver host.
No causal-conv1d wheel exists for this torch, so Transformers uses its own convolution. Generation
is TRL's default Transformers path; vLLM is not installed (its TRL adapter does not pass the pinned
revision, and its Qwen weight names differ from this text-only loader).

## Task selection and bundle

`selection.py` picks 30 train tasks, three per station, ordered by a hash of the task id; no
policy outcome enters the choice. The smoke uses the first two of them and the first three
stations of the eval cohort. The archive sent to the job holds code, the lock, task and met data
for the 40 tasks, native binaries and the selection; fields, weights, credentials and rollouts are
not in it. Its digest is checked locally and again by the job:

`9696466c807d9991a9b21eaea0e52962d967a10d30f9295f9d2cd2db531a4c1a`

## Commands

CPU checks, lock, lint and bundle, from the repository root:

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

Plans (CPU only, write nothing): `python -m train.cloud.run --plan --recipe smoke` and
`--recipe grpo --grpo-steps 30`. They print the stage budgets at $5 per hour; the job supervisor
enforces them.

Hub setup, once, with `HF_TOKEN` set. The inputs dataset and the run bucket are private:

```sh
uvx --from 'huggingface-hub==1.33.0' hf repos create Attoloop/permitforge-qwen38-inputs --repo-type dataset --private
uvx --from 'huggingface-hub==1.33.0' hf buckets create Attoloop/permitforge-qwen38-runs --private
uvx --from 'huggingface-hub==1.33.0' hf upload Attoloop/permitforge-qwen38-inputs \
  train/cloud/data/cloud-bundle.tar.gz cloud-bundle.tar.gz --repo-type dataset
```

Launch. Each script probes for a good host first and watches the job until it ends:

```sh
bash train/cloud/ops/smoke-attempts.sh d                 # smoke; artifacts in <bucket>/smoke-evalrules-d1
STEPS=30 bash train/cloud/ops/chain-full.sh full-30      # GRPO in 5-step chunks, both evals, export
```

Read back and report:

```sh
uvx --from 'huggingface-hub==1.33.0' hf sync hf://buckets/Attoloop/permitforge-qwen38-runs/full-30 train/cloud/data/full-30
python train/cloud/ops/jobs_ledger.py Attoloop train/cloud/data/jobs.json
uv run --project envs/permit/openenv --no-sync python -m eval.learnability \
  --run train/cloud/data/full-30 --jobs train/cloud/data/jobs.json --write
```

Trackio is opt-in: add `--trackio-space <org>/<space> --trackio-bucket <org>/<bucket>` after `--`
to log trainer metrics to a Space. The run above did not use it; the bucket holds the logs.
