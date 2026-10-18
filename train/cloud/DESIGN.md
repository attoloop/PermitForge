# Cloud training decisions

WP2.15 supersedes the cloud kernel/cache machinery from WP2.12–WP2.14. It uses
ordinary TRL `GRPOTrainer`, `GRPOConfig`, `environment_factory` and PEFT LoRA.
There is no custom trainer subclass or generation/loss override. Transformers
chooses its attention implementation and resolves installed FLA automatically.
The model, tasks, reward, native evaluation verifier and LoRA targets are unchanged;
training now follows the evaluation's rules (below). Local small-model experiments
in `train/` remain separate and keep their compact opening.

## Training under the evaluation rules

GRPO episodes use what the evaluation uses: the environment's instructions,
unchanged (30 evaluations); one opening line, `opening()` in
`permit_openenv/prompts.py`, shared by the frontier Claude driver, the cloud
evaluation and training; the same four tool names and descriptions; at most 40
executed tool rounds; and 30 evaluations. TRL appends `reset()`'s observation to
the dataset row's empty user message, so the rendered opening equals the
evaluation's; `tests/test_eval_rules.py` checks that, the tool descriptions against
the live MCP server, and that the rendered opening (983 tokens) fits.

Three differences remain, all disclosed:

- **Tool parameter schema.** TRL renders tools from the environment's methods
  through Transformers' `get_json_schema`, which has no override for an exact
  schema. Training therefore describes `design` as a mapping of source IDs to
  `{control, height_m}` without the evaluation schema's strictness metadata (no
  extra keys, at most six sources). The environment validates designs identically.
- **Final-turn reminder.** The evaluation adds "Last turn: submit your best
  feasible design now." before its 40th turn; TRL's tool loop cannot insert a user
  message mid-episode. Training is therefore no easier than evaluation at the limit.
- **Submission grading.** The model-facing text is the evaluation's ("independently
  verify"), but a training submission is graded by the fast evaluator, whose native
  agreement was measured on 240 designs (1,920 of 1,920 standard values to five
  decimals). Base and final evaluations use native AERMOD verification.

TRL's loop generates once and then at most `max_tool_calling_iterations` times
after executed tool calls, so 40 allows the evaluation's 40 executed tool rounds;
a 41st reply's tool calls are not executed and cannot change the reward.

## Versions and generation choice

`pyproject.toml` and `uv.lock` select torch 2.12.1+cu129 from the PyTorch cu129
index (CUDA 12.9 runtime) and exactly Triton 3.7.1, Torch's declared dependency, not
a resolver override. PyPI's torch 2.12.1 is a CUDA 13 build; on one H200 host its
driver could not run it and no GPU was visible, while another host's could. The
PyTorch cu128 index stops at torch 2.11, which pins the guarded Triton 3.6.0. The
lock covers Linux x86-64 only, the cloud target, and a test forbids CUDA 13 runtimes. FLA/fla-core 0.5.2 is required. Its Hopper guard excludes
Triton >=3.4.0,<3.7.1; the new compiler is outside that broken range. TileLang,
TVM-FFI, nvcc toolchain extras, FA3/Hub attention and Liger dependencies are
removed. Transformers 5.17.0, TRL 1.12.0, PEFT 0.19.0 and the existing optimizer
remain pinned. NVML (`nvidia-ml-py==13.580.82`) stays for the device memory guard;
Trackio 0.42.0 remains opt-in. No new training framework is introduced.

The old causal-conv1d 1.7.0 wheel targets torch 2.10/CUDA 12. It cannot be carried
into this binary stack. No publisher release (through v1.7.0) has a torch 2.11 or
2.12 wheel, so it is removed and Transformers' normal convolution implementation is
used; Transformers logs that this is correct but slower.
FLA accelerates linear attention independently of that optional extension.
No source build or custom runtime fallback is added.

vLLM 0.27.1 supports the Qwen3.5 architecture underlying the pinned Qwen3.8 model
and language LoRA, and falls within TRL 1.12's supported range. That alone does
not make this loader a clean stock integration: TRL's colocated constructor
passes `model.name_or_path` without `revision`, while its PEFT synchronization
streams text-only names (`model.layers.*`, `lm_head.*`) to the checkpoint's
conditional-generation wrapper (`language_model.*`). The inherited mapper handles
`lm_head.*` and `model.language_model.*`, but not this loader's `model.layers.*`.
The adapter merges and
unmerges LoRA; native vLLM LoRA support is a different interface. We therefore
use TRL's default Transformers generation, as allowed by the brief, without
writing a new synchronization or revision-mapping layer. This is a source-based
compatibility decision, not a measured vLLM failure or speed comparison.

Primary evidence, also captured by `internal/reports/WP2.15-evidence/sources.py`:

- [Torch 2.12.1 metadata](https://pypi.org/pypi/torch/2.12.1/json) declares the exact Triton dependency and CUDA runtime family.
- [Triton 3.7.1 release](https://github.com/triton-lang/triton/releases/tag/v3.7.1) and [FLA Hopper guard](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/common/chunk_o.py).
- [TRL vLLM construction and synchronization](https://github.com/huggingface/trl/blob/v1.12.0/trl/generation/vllm_generation.py) and [vLLM Qwen names/LoRA support](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/model_executor/models/qwen3_5.py), including its [inherited weight mapper](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/model_executor/models/qwen3_vl.py).
- [Convolution release assets](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0).
- [Transformers Qwen implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

## Numerical sanity and smoke

`kernels.py` runs one bf16 case of shape `(1,65,2,128)` with a nonzero initial
state, before any large model load. It calls the function selected by
Transformers and compares output, final state and q/k/v/g/beta/initial-state
gradients against an independent fp32 recurrence on the same quantized inputs.
Tolerances are the library's own: flash-linear-attention v0.5.2
`tests/ops/test_gdn.py::test_chunk` compares this kernel with a naive recurrence by
error ratio (RMS of the difference over RMS of the reference): output and state .005,
dq/dk/dv/dh0 .008, dg/dbeta .02. Every tensor is measured and recorded before the
verdict. Earlier ad-hoc bounds were wrong for bf16: an element-wise atol rejected one
bf16 rounding step on dk (ratio .0043), and a worst-element bound rejected dg at
ratio .0161, both within FLA's tolerances. This short
check crosses a chunk boundary and exercises the model's head dimension. It is
cheap insurance, not exhaustive qualification of every model shape or kernel.
It records versions, dispatch, errors and elapsed time. Failure aborts; it never
patches dispatch or tries another implementation. CPU tests inject finite wrong
gradients as well as wrong outputs/states and NaNs to verify rejection.

The smoke warms only its selected two train and three eval tasks. The untrained
base gets three native-verified evaluations before the training model is loaded.
Two GRPO steps must see both train identities, produce finite changed adapters,
and save/reload exactly. Training submission uses the unchanged fast evaluator;
base/final evaluation submission uses the unchanged isolated native verifier.
The report includes device/allocator memory, seconds per optimizer step, reward
spread and measured-step projections for 20/50/100 steps. It survives worker
failure or supervisor timeout. A longer run requires a passed smoke with matching
provenance and successful kernel/GPU checks. No candidate selection file exists.

Smoke history (HF Jobs under Attoloop, H200): FLA's guarded Hopper backward
stopped the first; a TileLang execution assertion and Torch fallback ended the
second; under the standard stack, an element-wise tolerance rejected one bf16
rounding step on dk; PyPI's CUDA 13 torch saw no GPU on an older-driver host; a
worst-element bound rejected dg at FLA-acceptable ratio .0161; then the native
grader failed because its fresh subprocess lacked the worker's field-cache binding
(now passed explicitly). Job `Attoloop/6ad478f2c662881452271a2c` then passed with
8-turn training: all eight kernel tensors within FLA's tolerances, three of three
untrained 40-turn evaluations feasible (58.2, 67.3 and 30.8% of draft cost saved),
peak device memory 120.4 GB, two GRPO steps with changed weights and exact reload,
289 s per step. In those training steps six of eight rollouts ran out of the 8 turns
before submitting, which led to training under the evaluation's rules. Those
settings need their own passed smoke.

## Preserved experiment and operations

The text-only loader checks all language weights against the pinned checkpoint;
a CPU test compares its mapping with a tiny conditional-generation checkpoint.
Only full-attention q/k/v/o LoRA trains (rank 16, alpha 32, zero dropout). Defaults
are bf16, group reward scaling, beta zero, one update per sampled group, four
generations, accumulation four, 40 turns and thinking off. The pinned Qwen
tokenizer is retained; TRL may not replace its template. Training context is 32,768
tokens, split into a 4,096 prompt ceiling and 28,672 completion ceiling, including
tool feedback; 40-turn base episodes measured about 19,000-23,000 tokens. Truncation masking and zero-spread diagnostics
stay unchanged. The existing explicit NF4 and historical SFT utilities remain
available but are not part of the standard H200 recipe.

This matches the FineEnvs examples' trainer/factory/PEFT structure. GeoGuesser uses
colocated vLLM; whitebox uses its server mode; OCR examples also use the default
generation route. We use the latter because it needs no custom integration here.
The named local FineEnvs checkout was read in place and never modified.

The frozen bundle, HF Jobs bootstrap, private artifacts bucket, opt-in Trackio,
complete-checkpoint resume and immutable run fingerprints remain. Payloads upload
before completion markers; interrupted native submissions cannot be selectively
retried. Field concurrency stays at most four, and cache limits are unchanged.
A single frozen install replaces independent optional-kernel attempts. Setup
failure stops the launcher. README commands use the `Attoloop` namespace.

The smoke's provider limit remains 10,800 seconds and $15 at the operator's
$5/hour rate, including setup. Its stage allowance remains 10,200 seconds and
$14.1667, with cleanup/upload time reserved. Planning inputs come from the passed
smoke (field preparation, 40-turn evaluation episodes, kernel check); the
evaluation-rules GRPO step (1,200 s, capped at 2,400 s) is an estimate until its
smoke measures it, and evaluation stages are capped at 30 × 1,200 s. Reproduce them
with `python -m train.cloud.run --plan --recipe smoke` in the cloud environment.

Reproduce CPU checks, lock validation, bundle rebuild and verification:

```sh
python3 internal/reports/WP2.15-evidence/reproduce.py
python3 internal/reports/WP2.15-evidence/reproduce.py --summary
```

No HF write, paid call, GPU job, commit or publication is part of this package.
