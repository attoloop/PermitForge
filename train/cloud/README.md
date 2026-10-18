# Qwen3.8-27B on HF Jobs

WP2.12 prepares an H200 job under **attoloop**, using **bf16 LoRA and direct
GRPO**. No job, upload, repository, bucket or Space has been created by this work.
SFT utilities remain available as historical code; no SFT data is required or
built by this workflow. GPU memory, speed and learning await the smoke.

## Smoke first

One job performs, in order:

1. Verify the input archive, install the frozen stack and attempt pinned fast kernels.
2. Prepare fields for exactly two selected train tasks and three selected eval tasks,
   with at most four native workers. No other task fields are warmed.
3. Run the **untrained base** on the three eval tasks, one repeat each, through the
   public tools and native terminal verifier. Preserve transcripts, valid calls,
   submissions, rewards and infrastructure failures.
4. Run **two direct GRPO optimizer steps**, on the two train tasks, without SFT.
   Require finite loss/weights, changed adapter weights, both task identities and
   exact save/reload equality. Save optimizer, scheduler and RNG state too.
5. Write `smoke/REPORT.md`, `smoke/metrics.json`, GPU/kernel evidence and projections
   for 20/50/100 steps, derived from the measured step mean.

The provider timeout is **2 hours**, a **$10 GPU-compute envelope at $5/hour**.
The stage limit is 6,600 seconds and $9.17; the launcher also subtracts setup
from a shared job deadline and reserves final-upload time. The provider timeout
covers setup and process termination. Storage, taxes and other account activity
are outside this compute envelope. A timeout may leave an incomplete smoke; it
never authorizes another paid attempt automatically.

Longer GRPO requires a passed smoke with matching code, input hashes, selection,
model and training settings. At least one valid tool call, one submission, one
positive base reward and nonzero training-group reward spread must be observed,
along with the gradient/reload checks. Failure produces a report and blocks the
longer workflow. A smoke pass establishes readiness to experiment, not learning.

## Pinned stack and kernels

The model revision is `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` of
`Qwen/Qwen3.8-27B`. The text-only `Qwen3_5ForCausalLM` loader checks every
language weight; it omits the vision encoder. Defaults are one H200, bf16,
full-attention q/k/v/o LoRA rank 16/alpha 32, HF generation microbatch one,
4 generations, accumulation 4, 8 tool turns, a 4,096-token prompt ceiling and
12,288-token completion ceiling. Thinking and vLLM are disabled. The completion
budget includes tool replies; its sizing follows `train/DESIGN.md`.
`--precision nf4` retains double-quantized NF4 with bf16 computation as an option.
The device guard aborts above 135,000,000,000 device bytes, reserving additional
allocator headroom. This is a guard, not evidence that the model fits.

`pyproject.toml` and `uv.lock` pin torch 2.10.0+cu128, Transformers 5.17.0,
TRL 1.12.0, PEFT 0.19.0 and HF Hub **1.33.0**. The bootstrap pins uv 0.11.32.
The optional `fast-kernels` extra pins FLA/fla-core **0.5.2**, and the **1.7.0**
causal-conv1d CPython 3.12/CUDA 12/torch 2.10/C++11 ABI wheel, including its SHA256.
It also pins **TileLang 0.1.15**, **TVM-FFI 0.1.12** and its CUDA compiler
components at **13.2.86** (runtime **13.2.86**). Torch retains CUDA 12.8 and its exact
Triton 3.6.0 dependency. FLA uses TileLang for the gated backward blocked on
Hopper with this Triton version. The CUDA 13 nvcc wheel supplies the compiler;
the CUDA 12 nvcc wheel supplies only ptxas. TileLang JIT compiles device code;
the convolution extension remains prebuilt. Normal attention uses PyTorch SDPA.
The base lock includes **nvidia-ml-py 13.580.82** for device memory monitoring.

`--kernels auto` attempts the extra for at most 300 seconds, then restores the
base lock on installation failure. Transformers automatically dispatches to the
installed local packages. Hub kernel downloads are disabled with
`USE_HUB_KERNELS=NO`. Before model loading, small bf16 probes compare outputs,
recurrent/cache states and gradients with the non-FLA Torch references. Tensor
errors and tolerances are recorded in `kernels.json`. Numerical mismatches and
runtime/JIT/gradient errors abort; `--kernels required` also fails if optimized
dispatch is missing. `--kernels off` skips the extra in the fresh job environment
and explicitly selects the reference Torch implementations even when extras
are already installed. Runtime failures do not trigger an automatic retry.

The complete frozen cloud project and fast extra installed in a CPU-only temporary
environment; the exact TileLang backend imports and Transformers resolves all four
functions to FLA/causal-conv1d. This does not establish Hopper numerical agreement,
driver compatibility or training speed. The reference implementation warns of more than an
order-of-magnitude GDN slowdown on H100; that is an upstream kernel observation,
not a measured H200 training multiplier. Fallback may exhaust the same $10 cap.
No numerical H200 fallback-cost prediction is available before a measured run.
See [DESIGN.md](DESIGN.md) for source links and local reproduction commands.

## Task selection and local bundle

`selection.py` chooses **30 train tasks**: three per station, ordered by
SHA256 of `WP2.12:` plus task ID, with no policy outcomes. Smoke train tasks are
the first two of these by `(source count, ID)`. Smoke eval tasks are the first
three distinct stations in the frozen ten-task cohort, ordered by the same
structural key. Full base/final evaluation uses that cohort, three repeats each;
this differs from the frontier panel's four repeats. No eval task trains.

The archive includes code, the frozen stack, catalogs, the selection, native
binaries, and task/met inputs for the selected thirty train plus ten eval tasks.
Fields, model weights, credentials, rollouts and SFT data are excluded. All
archive members and converted task/met hashes are verified locally and again
by the job. Large archives remain ignored by git.

Run from the repository root (the existing `train` environment can run CPU checks
without installing a second CUDA stack):

```sh
uv lock --project train/cloud --check
CUDA_VISIBLE_DEVICES='' uv run --project train --no-sync python -m pytest \
  -c train/cloud/pyproject.toml train/cloud/tests train/tests -q
uv run --project train --no-sync ruff check train/cloud
uv run --project train --no-sync python -m train.cloud.bundle \
  --output train/cloud/data/cloud-bundle.tar.gz
BUNDLE_SHA="$(sha256sum train/cloud/data/cloud-bundle.tar.gz | cut -d ' ' -f 1)"
uv run --project train --no-sync python -m train.cloud.bundle --verify \
  --output train/cloud/data/cloud-bundle.tar.gz --sha256 "$BUNDLE_SHA"
```

## Plans, costs and continuation

These commands are CPU-only and write nothing:

```sh
uv run --project train --no-sync python -m train.cloud.run --plan --recipe smoke
uv run --project train --no-sync python -m train.cloud.run --plan --recipe grpo \
  --grpo-steps 100 --stop-after-step 20
uv run --project train --no-sync python -m train.cloud.run --plan --recipe grpo --grpo-steps 50
uv run --project train --no-sync python -m train.cloud.run --plan --recipe grpo --grpo-steps 100
```

Unmeasured priors at $5/hour, including 900 seconds of setup, reproduced by those
commands (`estimated_total_usd + setup_prior_usd`):

| Recipe | Estimated compute USD | Provider timeout seconds | Provider compute envelope USD |
|---|---:|---:|---:|
| Smoke | 3.35 | 7,200 | 10.00 |
| Stop at 20, target 100 | 17.92 | 27,000 | 37.50 |
| Fresh 50, base + final eval | 33.33 | 49,500 | 68.75 |
| Fresh 100, base + final eval | 45.83 | 67,500 | 93.75 |

Full plans are fields → base eval → GRPO → final eval → adapter export. The
20-step check omits final eval/export. Priors assume 300 CPU-seconds per field
task divided by four workers, 180 seconds per eval episode, and 180 seconds per
GRPO step. They are planning inputs, not throughput claims. Use smoke measurements
before choosing new caps or launching longer work. `--wall-cap STAGE=SECONDS` and
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

Longer jobs pass `--qualification-run /previous/smoke-v1`, with the run bucket
mounted read-only at `/previous`. Their `--job-seconds` must match the provider
`--timeout` printed by the plan. All jobs use the same workspace path. `run.json`
freezes code/data/settings, and payloads upload before completion markers. Failed
or interrupted native submissions remain pending for review and are not retried
selectively. An upload failure stops progression. The field cache is bounded at
128 GiB; misses regenerate within the requesting stage's allowance, four workers
maximum. Native grading runs in a small subprocess independent of model RAM.

## Operator commands: create, upload, launch, watch, cancel

**The commands below write to HF or spend money; they were not executed here.**
They use the inspected HF Hub CLI **1.33.0**, which supports `h200`, `--volume`
and `--namespace`. Set `HF_TOKEN` securely in your environment first. The named
input dataset and run bucket below are private and under attoloop.

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
  --namespace attoloop --flavor h200 --timeout 2h --secrets HF_TOKEN \
  --name permitforge-qwen38-smoke --detach --quiet \
  --volume hf://datasets/Attoloop/permitforge-qwen38-inputs:/inputs:ro \
  train/cloud/hf_job.py --bundle /inputs/cloud-bundle.tar.gz \
  --bundle-sha256 "$BUNDLE_SHA" --workspace /workspace/permit-cloud \
  --job-seconds 7200 -- --execute --recipe smoke --hourly-usd 5.00 \
  --precision bf16 --kernels auto --output train/cloud/runs/smoke-v1 \
  --artifact-uri hf://buckets/Attoloop/permitforge-qwen38-runs/smoke-v1)"

# Keep a local log copy; inspect the final Job status after the stream ends.
uvx --from 'huggingface-hub==1.33.0' hf jobs logs \
  --namespace attoloop "$JOB_ID" --follow | tee train/cloud/data/smoke-logs.txt
uvx --from 'huggingface-hub==1.33.0' hf jobs inspect --namespace attoloop "$JOB_ID"

# Cancel immediately if needed (use a second terminal while following logs).
uvx --from 'huggingface-hub==1.33.0' hf jobs cancel --namespace attoloop "$JOB_ID"

# Read back artifacts after completion; destination must be a fresh directory.
uvx --from 'huggingface-hub==1.33.0' hf sync \
  hf://buckets/Attoloop/permitforge-qwen38-runs/smoke-v1 train/cloud/data/smoke-result
cat train/cloud/data/smoke-result/smoke/REPORT.md
```

Trackio is opt-in: after the operator names its Space, append
`--trackio-space attoloop/CHOSEN_SPACE --trackio-bucket attoloop/CHOSEN_METRICS_BUCKET`
to the **run arguments after `--`**. This installs the pinned `metrics` extra and
logs trainer metrics to that live Space, using Transformers' Trackio integration
as FineEnvs does. Static Space export is disabled. Trackio can create/update the
named Space/bucket; omitting the flag means no Trackio initialization or writes.
No Space name has been chosen by this work. The private run bucket independently
retains training logs, checkpoints, traces, GPU/kernel evidence and smoke reports.
