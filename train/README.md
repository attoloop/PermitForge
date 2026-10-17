# PermitForge training

The training adapter uses `permit_core.evaluate` for both evaluation and final
submission. Native verification remains the evaluation harness's responsibility.
Only eligible train tasks enter GRPO or regular SFT. Ordinary training requires
model-calibration anchors; `--smoke` alone permits a labelled provisional anchor
equal to the largest positive feasible saving in that task's recorded diagnostics.

Run from the repository root:

```bash
uv venv -p 3.12 train/.venv
uv sync --project train --locked
uv run --project train --locked pytest -c train/pyproject.toml train/tests -q
train/.venv/bin/python -m train.stack --check-tokenizer --output train/evidence/stack.json
```

`train/uv.lock` is the project lock. The fully pinned, hashed
`train/requirements.lock` is an export for environments using `uv pip`:

```bash
uv export --project train --locked --no-emit-project --no-emit-package permit-core --emit-index-url -o train/requirements.lock
uv pip sync --python train/.venv/bin/python --index-strategy unsafe-best-match train/requirements.lock
```

The export excludes the editable core package; the entry points import the local
core and lightweight OpenEnv task/prompt modules from this checkout. See
[DESIGN.md](DESIGN.md) for dependency and source decisions.

The bounded GPU smoke commands are:

```bash
smoke_args=(--smoke --tasks 074897d0f159-s47 01addc2a86f1-s47 --fields-dir convert/cache/fields --diagnostics-dir convert/cache/wp110-batch)
train/.venv/bin/python -m train.grpo "${smoke_args[@]}" --model Qwen/Qwen3-0.6B --output train/runs/grpo-06b
train/.venv/bin/python -m train.grpo "${smoke_args[@]}" --model Qwen/Qwen3-1.7B --output train/runs/grpo-17b
train/.venv/bin/python -m train.sft --smoke --synthetic --model Qwen/Qwen3-1.7B --output train/runs/sft-17b
```

Each smoke command requests exactly two optimizer updates. The GRPO commands
select two train tasks; the SFT command constructs four explicitly invented
tool-format episodes. Synthetic records never enter regular SFT. Models and
tokenizers go to `~/.cache/huggingface/hub`; adapters and traces go under the
ignored `train/runs/` tree. No command publishes, uses a hosted inference API,
or rents a GPU. Use the same checkout, task data, field packs and commands on an
already-provisioned cloud GPU. `--model`, `--revision`, `--no-4bit`, batch,
accumulation, generation, context and LoRA flags allow different model sizes.

Prerequisites: CUDA must be visible, and `--fields-dir` must already contain all
selected float64 fields (either `<task>.npz` or `<task>/<source>/<height>.npy`
with the core cache metadata). The adapter reads and hashes these inputs without
precomputing, editing or evicting them. At handoff the second selected task's
field directory was absent. Obtain it through the field work package before
running the GPU smoke. Do not interpret `--prepare-only` as field validation.

The no-GPU preparation commands used during this work were:

```bash
train/.venv/bin/python -m train.grpo "${smoke_args[@]}" --prepare-only --output train/evidence/grpo-preflight
train/.venv/bin/python -m train.sft --smoke --synthetic --prepare-only --output train/evidence/sft-preflight
```

`metrics.json` records actual step durations, peak allocated/reserved CUDA memory,
episodes/hour, each group's rewards and spread, and the LoRA checks. The final
adapter is saved and reloaded onto the same frozen base; the smoke fails if its
tensors differ after reload or if no LoRA tensor changed. A zero-spread group
can legitimately produce no update; rewards are never fabricated to pass this
check. An unavailable grade is NaN to TRL and `grading_failed: true` in JSON.
Missing submissions and invalid/infeasible submissions score zero.

Regular runs omit `--smoke` and `--diagnostics-dir`, supply calibrated tasks and
existing fields, and explicitly select `--steps`. SFT consumes saved episode
JSON/JSONL with `--records`; it rejects eval identities, missing submissions and
non-improvements, checks submitted design cost, and keeps complete tool calls.
SFT tokenizes before Arrow conversion and applies explicit assistant masks so
tool feedback is excluded from the loss. Overlong prompts/episodes fail before
training instead of silently truncating away the submission.

The default GRPO settings are defined in `config_values()`; reproduce them with
the preparation command's `config.json`. If GPU memory is insufficient, the
explicit smaller-group retry is `--num-generations 4 --accum 4` with a fresh
output directory. This has not been measured on the hidden GPU.
