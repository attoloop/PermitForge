# Qwen3.8-27B cloud setup

This is a free, CPU-checked setup for one RTX PRO 6000 96 GB. It does not rent a
GPU or submit a Job. GPU fit, numerical stability, throughput and learning remain
unmeasured. Existing local training commands and defaults are unchanged.

`run.py` is the entrypoint for both platforms. `--plan` is read-only; execution
requires `--execute`, the actual `--hourly-usd`, and an existing `--artifact-uri`.
Artifacts accept `hf://buckets/OWNER/BUCKET/PREFIX`,
`hf://datasets/OWNER/REPO/PREFIX`, or `hf://models/OWNER/REPO/PREFIX`.
Use a dedicated private destination and an HF token in the environment.

## Pins and CPU checks

The model is `Qwen/Qwen3.8-27B`, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, from WP2.3a. The separate
[pyproject.toml](pyproject.toml) and [uv.lock](uv.lock) pin Python's cloud stack:
torch `2.10.0+cu128`, TRL `1.12.0`, Transformers `5.17.0`, PEFT `0.19.0`,
bitsandbytes `0.49.2`, Accelerate `1.12.0`, datasets `4.7.0`, HF Hub `1.33.0`.
The launch script pins uv `0.11.32`. Reproduce all pins with
`uv lock --project train/cloud --check` and `cat train/cloud/pyproject.toml`.

Transformers' explicit `Qwen3_5ForCausalLM` loader extracts the language backbone
from the multimodal checkpoint, using its supported `qwen3_5_text` prefix mapping.
A tiny CPU checkpoint test compares every text weight; unmatched language weights
abort loading. Vision weights are omitted. NF4, double quantization and bf16
compute are used; LoRA targets only language full-attention q/k/v/o projections.
There is one model, HF generation, generation microbatch one, and no vLLM.
Thinking is disabled in SFT, GRPO and evaluation. See [DESIGN.md](DESIGN.md).

Run from the repository root on a machine with sufficient installation disk:

```sh
uv sync --project train/cloud --frozen
CUDA_VISIBLE_DEVICES='' uv run --project train/cloud --frozen python -m pytest \
  train/tests train/cloud/tests -q
CUDA_VISIBLE_DEVICES='' uv run --project train/cloud --frozen python -m train.cloud.check_tokens \
  --records results/rollouts/wp2.0b-smoke --output train/cloud/data/token-check.json
uv run --project train/cloud --frozen ruff check train/cloud
```

The WP2.9 check used the existing `train` environment (`--project train --no-sync`):
a second complete CUDA installation ran out of local disk, and only its partial
`train/cloud/.venv` was removed. Dependency resolution passed; a fresh full cloud
installation was not completed locally. No weights, native simulations or GPU
training were used. Tokenizer files alone were downloaded at the pinned revision.

The recorded smoke episode is **10,422 tokens**, with a **2,226-token prompt** and
**585 assistant target tokens**; its **10 calls** round-trip through both the
Qwen XML parser and TRL. The **16,384-token** training context fits it. Reproduce
with the `check_tokens` command above. It also saves the complete rendered text.
Masks cover assistant completion bodies and end tokens, excluding system/user
text, tool results and the empty thinking prefix. Longer SFT episodes are counted
and excluded whole, never truncated before submission.

## Data hand-off and input bundle

After train-task evaluation trajectories exist, build the SFT data with:

```sh
uv run --project train/cloud --frozen python -m train.cloud.make_sft \
  --records results/rollouts/FINAL_RUN --output train/cloud/data/sft.jsonl
# Free check against the existing recorded smoke:
uv run --project train/cloud --frozen python -m train.cloud.make_sft \
  --records results/rollouts/wp2.0b-smoke --output train/cloud/data/smoke.jsonl
```

The builder reads canonical Episode records or explicit episode JSONL files. It
checks authoritative task IDs and station-disjoint splits, rejects contradictory
identities, deduplicates repeat identities, filters eval episodes, and retains
only complete, feasible, positive-savings trajectories ending in one submission.
Exact savings are checked against the task's final design. Synthetic examples are
excluded. The adjacent `.stats.json` records rejection counts and source hashes.
The smoke produces one accepted train trajectory; it is a formatting check, not
evidence that a useful training dataset exists.

Make a private input archive after reviewing the SFT data and current panel cohort:

```sh
uv run --project train/cloud --frozen python -m train.cloud.bundle \
  --sft-data train/cloud/data/sft.jsonl --cohort results/subsets.json \
  --output train/cloud/data/cloud-bundle.tar.gz
sha256sum train/cloud/data/cloud-bundle.tar.gz
```

`--inventory-only` inventories without creating the archive. The allowlist includes
converted task data/met, current code, the cloud lock, the cohort, SFT JSONL and
already-built static native binaries. It excludes credentials, rollouts, model
weights, field caches and other repositories. The bundle contains a hash manifest;
the launcher checks the whole archive hash before safe extraction. This setup
never uploads the bundle. The operator must place the reviewed bundle in an
existing private input repository or transfer it to the VM.

## Stage plan, caps and resume

```sh
uv run --project train/cloud --frozen python -m train.cloud.run --plan --hourly-usd 2.75
```

Default plan, reproduced by that command:

| Stage | Work | Estimated hours | Hard runtime cap, hours |
|---|---|---:|---:|
| fields | Selected train tasks plus panel eval tasks, four native workers | 5.000 | 8 |
| base-eval | Frozen ten eval tasks, three independent repeats | 1.500 | 16 |
| smoke | Two SFT steps, reload, two GRPO steps, reload | 0.500 | 1 |
| sft | 100 steps on accepted train trajectories | 0.556 | 2 |
| grpo | 100 steps, all eligible train tasks available | 5.000 | 8 |
| eval | Same tasks, seeds and sampling with trained adapter | 1.500 | 16 |
| export | Adapter, provenance and stage metrics | 0.250 | 1 |

These are **unmeasured planning priors**, not benchmark timings: field work assumes
300 CPU-seconds/task divided by four workers; evaluation 180 seconds/episode;
SFT 20 seconds/step; GRPO 180 seconds/step, including possible cold fields.
`--sft-steps`, `--grpo-steps` and `--train-tasks` change the plan explicitly.
The default train set contains all 230 eligible tasks. Actual throughput, reward
spread and stability must be established by the smoke before longer training.

Every stage has cumulative wall and compute-dollar limits. Override with, for
example, `--wall-cap grpo=14400 --usd-cap grpo=11`. The effective deadline is
`min(wall_cap, usd_cap / hourly_rate * 3600)`, less prior charged attempts.
A missing charge after a crash consumes the full reservation. The supervisor
stops the worker and tracks detached native descendants; periodic uploads and
final cleanup use the same allowance. An upload failure stops progress.

These are **process compute caps**, not an account-wide billing promise. Package
setup, HF Job Starting time, storage, tax and idle VM rental are outside them.
Set the provider timeout too and terminate the VM when the command exits. The
example HF timeout below permits setup beyond the 52-hour cumulative stage caps.

`run.json` freezes source/data/protocol hashes, cohort, settings and destination.
Resume uses the identical command and output directory; completed stages and
terminal episodes are skipped. Checkpoints have optimizer/RNG state and an
explicit completion marker. Uploads send payloads before completion markers.
Use `--restore` on an empty output directory to recover a remote run, keeping the
same workspace paths. A lost in-flight reservation remains charged. An interrupted
native submission is held for review, never selectively resubmitted. A stopped
non-submitting sampled episode is retained as infrastructure failure; it is not
counted as zero reward.

The field cache defaults to 128 GiB and evicts idle packs. The full selected field
estimate exceeds HF Jobs' local disk: reproduce with `estimated_bytes` in
[fields.py](fields.py) or the WP2.9 evidence command. Initial warming records
progress without requiring every pack to remain resident. A later miss is
computed in a small CPU subprocess with at most four native workers; the same
stage cap covers that work. This keeps all train tasks available. Native final
grading also runs in a small subprocess so the model's host RAM cannot trip the
published verifier's resource gate.

The smoke aborts if sampled device usage exceeds **90,000,000,000 bytes** and
sets a lower PyTorch allocator limit to reserve CUDA overhead. Metrics include
peak allocated/reserved/device memory, step times, changed weights, exact adapter
reload equality and GRPO reward spread. A zero-change or failed-reload smoke
blocks SFT/GRPO/eval. GPU allocation sampling is periodic; driver allocation
failures can abort sooner. Export is adapter-only; no quantized merged model is
produced.

Evaluation uses the current panel cohort and public OpenEnv tool schemas/prompt:
40 responses, 32,768 output tokens/response, 30 valid evaluations, native terminal
grading, 1,800 seconds/episode. WP2.9 explicitly requests **three** repeats; the
final frontier panel uses four. Qwen thinking is off and its pinned generation
configuration supplies temperature/sampling defaults. These differences are
recorded; cloud runs must not be pooled as identical frontier-panel runs.
Episode JSON, raw returned text/tool transcript, seeds, usage and per-task spread
are retained. Infrastructure failures are excluded from graded means and exposed
as incomplete coverage. Reward comes from saved fraction; no anchor is used.

## Platform prices and launch commands

Quotes captured by WP2.3a on 2026-10-17, rechecked against primary pricing pages.
Reproduce totals with `run --plan --hourly-usd RATE`; `config.py:PRICES` contains
these inputs. Installation/startup, storage and taxes are excluded.

| Platform, RTX PRO 6000 | USD/hour input | Estimated compute | Sum of stage compute caps |
|---|---:|---:|---:|
| [HF Jobs](https://huggingface.co/docs/hub/jobs-pricing) | 2.75 | 39.34 | 143.00 |
| [Jarvislabs](https://jarvislabs.ai/pricing) | 2.19 | 31.33 | 113.88 |
| [RunPod](https://www.runpod.io/pricing) | 2.49 | 35.62 | 129.48 |

Jarvislabs' VM page has advertised a lower quote than its pricing page; use the
higher quote until the dashboard confirms the actual offer. RunPod pricing
varies by offer. Recheck the selected rate and pass it explicitly. HF lists
475 GB ephemeral storage for this flavor; the bounded cache avoids needing the
entire field collection on disk. VM disk must also fit model downloads, the uv
cache, checkpoints and the chosen field-cache size.

The following are **operator launch commands; none was executed by this WP**.
`INPUT_REPO`, `BUNDLE_SHA` and `ARTIFACT_URI` are required operator values. The input
repository must contain the reviewed `cloud-bundle.tar.gz` at its root; `HF_TOKEN`
is supplied as a secret, never written into a command file.

HF Jobs (requires access to Jobs; the official [Jobs guide](https://huggingface.co/docs/huggingface_hub/guides/jobs)
documents the PEP 723 runner and mounted inputs):

```sh
hf jobs uv run --flavor rtx-pro-6000 --timeout 54h --secrets HF_TOKEN \
  --volume "hf://datasets/$INPUT_REPO:/inputs:ro" train/cloud/hf_job.py \
  --bundle /inputs/cloud-bundle.tar.gz --bundle-sha256 "$BUNDLE_SHA" \
  --workspace /workspace/permit-cloud -- --execute --stage all --hourly-usd 2.75 \
  --artifact-uri "$ARTIFACT_URI" --output train/cloud/runs/qwen38
```

The HF launcher checks and extracts the bundle, installs its frozen cloud project,
and invokes the same entrypoint as the VM. A 54-hour provider timeout at the quoted
rate is a **$148.50 compute envelope**, including setup time but excluding other
charges; reproduce with `python3 -c 'print(54*2.75)'`. The stage caps still apply.

Plain Linux VM (Jarvislabs or RunPod; NVIDIA driver supporting CUDA 12.8, one GPU
visible; uv 0.11.32 installed). Transfer the same bundle, then:

```sh
uv run train/cloud/hf_job.py --bundle /path/to/cloud-bundle.tar.gz \
  --bundle-sha256 "$BUNDLE_SHA" --workspace /workspace/permit-cloud -- --plan --hourly-usd 2.19
cd /workspace/permit-cloud/permitforge
CUDA_VISIBLE_DEVICES=0 timeout --kill-after=30s 54h sh train/cloud/vm.sh \
  --execute --stage all --hourly-usd 2.19 --artifact-uri "$ARTIFACT_URI" \
  --output train/cloud/runs/qwen38
# For RunPod, use the dashboard rate (planning quote: --hourly-usd 2.49).
```

The first VM command reuses the verified bundle bootstrap and prints the plan;
it performs package setup but starts no stage. Subsequent VM runs use the thin
`vm.sh` launcher. `timeout` ends the command, not the rental: the operator must
stop/delete the instance through the provider. No rental-management API is used.
