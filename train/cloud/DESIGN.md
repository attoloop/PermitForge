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

The operator's WP2.16 decision scopes this experiment to learnability: base Qwen
versus trained Qwen on the same unchanged cloud evaluation, with no frontier-panel
comparison. Standard TRL is retained. The remaining differences are:

- **Turn and retry prompts.** Evaluation appends `Turns left: …` to every tool
  result. It retries one empty/no-tool reply with a tool-use prompt and prompts for
  correction after malformed calls. Training has no turn-count suffix or retry
  prompts; TRL stops when a parsed reply has no tool calls.
- **Last-turn reminder.** Evaluation adds a submission reminder before its final
  turn. Training does not insert that reminder.
- **Termination and extra targets.** Evaluation stops immediately after submission
  and generates at most 40 replies. TRL permits 40 executed tool rounds and one
  additional reply. After submission it continues while the model calls tools;
  the environment returns an episode-over error. These extra replies have loss
  mask 1 even though they cannot change the grade. Multiple calls in a training
  reply are all dispatched; evaluation stops dispatching when submission ends
  the episode. Invalid submission ends the environment in either path.
- **Tool parameter schema.** Training's `design` schema is a mapping of source
  IDs to rows with arbitrary string-or-number values. It lacks explicit `control`
  and `height_m` properties, required row fields, their distinct types, and the
  evaluation's extra-key/source-count restrictions. Actual design validation is
  unchanged. The pinned template/parser is used by TRL; evaluation uses its own
  XML parser with format-correction prompts.
- **Tool JSON text and grading.** Training emits compact JSON and no per-receptor
  arrays; evaluation emits public JSON with the turn suffix. Training submit also
  includes `grading_failed` and `training_grade`. Training uses the fast evaluator
  at submission; evaluation uses the native verifier. These are different feedback
  strings and graders, even though tool names/descriptions and reward are shared.
- **Budgets and clocks.** Training has a cumulative completion allowance including
  tool feedback. Evaluation has a per-reply allowance and an episode wall-clock
  cap, with no matching cumulative allowance. Training is bounded by stage caps.
  TRL can execute tools before dropping overflowing feedback; that side effect
  (including submission) remains. The diagnostics below distinguish it.

Sampling is now matched through `GRPOConfig`: temperature 1.0, top_p 0.95 and
top_k 20, as recorded in the pinned model's `base-eval/sampling.json`. Both use
thinking off and a 32,768-token per-reply cap. TRL's tokenizer EOS/pad handling
stops generation at tokenizer EOS; evaluation's model config also accepts its pad
token as EOS. These stock stop-token defaults remain different. Training seeds
the trainer once and samples groups across updates; evaluation reseeds each task
and repeat. Training uses train-only groups and group-normalized advantages;
evaluation reports saved-fraction rewards on its frozen eval cohort.
`test_stock_trl_accepts_text_lora_and_existing_environment_factory` checks the
resolved trainer generation settings against that saved model configuration.

## Completion allowance and termination evidence

The retained `data/smoke-result-grade/smoke/base-eval/` transcripts replay through
installed `GRPOTrainer._tool_call_loop` and its `_get_tool_suffix_ids`, without
running any tools. In task order 4528c31805f3 / 660f38096246 / 9b1cffc36b20, the
prefixes before the submit reply are 30,207 / 32,472 / 29,829 total tokens. Through
the submit reply and compact feedback the completions are 30,259 / 32,582 / 29,878
tokens, including one synthetic EOS for TRL's unrecorded post-submit reply. The
opening adds 983 tokens. These are replay sizes, not evidence that the cloud
training feedback or future sampled rollouts have identical lengths.

`max_completion_length=40,960` gives at least 20% headroom over these complete
replays; `max_length=45,056` is the 4,096 prompt ceiling plus that allowance.
`tests/test_rollout_limits.py` compares unrestricted and budgeted replay, including
submit feedback; `test_eval_rules.py` retains the opening-fits check. Reproduce
all sizes with the replay command in README.

The installed trainer builds its `GenerationConfig` from GRPO options, then
updates it with `generation_kwargs`. Setting `max_new_tokens=32,768` there caps
**each** regeneration. `max_length` is not a GRPO generation-time limit: TRL
checks tool feedback against the cumulative allowance, generates a full reply,
then slices it to remaining space. The conservative configured transient bound
is 4,096 + 40,960 + 32,768 = 77,824 tokens, versus 45,056 retained. Loss logits
cover all retained completion positions, including masked tool feedback;
`logits_to_keep=1` only helps generation. Fit and speed need the new GPU smoke.

A zero-weight reward hook reads `completion_ids`, parsed completions and
`environments` before TRL's environment reward flushes the trace. Each rollout
records `ended` (`submitted`, `stopped_tools`, `budget`, `turn_limit`), `submitted`,
`budget_cut`, `turn_limit` and `completion_tokens` (retained IDs including tool
feedback). Submission takes precedence in `ended`; flags also expose a later
budget/round cut. Pending tool calls with dropped feedback identify budget cuts
that end in EOS below the allowance; non-EOS endings identify sliced replies or
the reply cap. The model context ceiling is above these configured limits.
Pooled resets clear the diagnostics.

`grpo/metrics.json` and `smoke/REPORT.md` report budget-cut counts per optimizer
step (one-based; episode records retain the trainer's zero-based sampling step).
`grpo/trainer_metrics.json` keeps TRL's log history: its completion lengths count
model tokens only, and its EOS-based clipped ratio can miss feedback overflow.
`generation_config.json` records the trainer's resolved sampling and reply cap.
**Budget-cut rollouts stay in the loss** (`mask_truncated_completions=False`);
only tool-result tokens retain TRL's usual zero loss mask. The installed-loop
regressions cover overflow after an executed tool, sliced replies, submission,
multiple/unknown calls and the round limit. A tiny CPU trainer verifies reward-hook
ordering and retained loss masks; there is no custom rollout implementation.

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

## Loss memory

TRL's loss asks the model for the logits of every completion token at once. With this
model's 248,320-token vocabulary, a 27,218-token completion is 13.5 GB of bf16 logits
and 27 GB once `selective_log_softmax` works in fp32, on top of 54 GB of weights and the
checkpointed activations; job `Attoloop/6ad4c41e5ccb47ba0f7c4004` needed about 153 GiB
of the H200's 139.8 GiB and stopped at the first optimizer step. TRL 1.12 chunks that
computation only across the batch, which is already one, and its Liger loss chunks the
same way. `chunked_logps.py` subclasses `GRPOTrainer` and overrides one method: the
backbone runs once (as TRL's Liger path does), then the output head and TRL's own
`selective_log_softmax` run per 2,048 completion tokens under activation checkpointing,
so a chunk's logits exist only while computed and are recomputed in backward. Generation,
rewards, advantages and the loss formula are TRL's. `tests/test_chunked_logps.py` shows
the values, entropies and LoRA gradients equal TRL's unchunked function on a tiny model
and that the head never receives more than one chunk. Because two smoke steps only show
the completions they happened to draw, the smoke then runs one loss forward and backward
at the full budget (40,960 completion tokens in a 45,056-token sequence, random ids)
through the same path and records peak memory and seconds as `worst_case`; a smoke
qualifies a run only if that fits under the device cap.

A smoke may carry a passed base evaluation instead of repeating it
(`--base-eval-from hf://buckets/.../smoke/base-eval`): the three episode records are
restored into `smoke/base-eval/` with `PROVENANCE.json` naming the source run, the report
says so, and the GRPO qualification gate reads them as before.

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
settings need their own passed smoke. Under them, job `Attoloop/6ad4c41e5ccb47ba0f7c4004`
(`smoke-evalrules-c1`) evaluated the base in three 40-turn episodes (all submitted feasible
designs: rewards .62, .67, .45; 13-17 min each including native grading), generated one
group of six rollouts in about 15 min (five submitted, one hit the turn limit, none cut by
the token budget; 2.5k-27.2k completion tokens; rewards .83, .83, .74, .41, .59, 0;
standard deviation .29; peak device memory 125.5 GB) and then ran out of memory in the
loss, which led to the chunked log-probabilities above. Two earlier starts that day found
H200 hosts on driver 580.178.04 whose container runtime injected no CUDA libraries; the
GPU guard now reports that condition and a start on such a host costs about a minute.

## Preserved experiment and operations

The text-only loader checks all language weights against the pinned checkpoint;
a CPU test compares its mapping with a tiny conditional-generation checkpoint.
Only full-attention q/k/v/o LoRA trains (rank 16, alpha 32, zero dropout). Defaults
are bf16, group reward scaling, beta zero, one update per sampled group, six
generations, accumulation six, 40 turns and thinking off. The pinned Qwen
tokenizer is retained; TRL may not replace its template. Completion limits,
measured replay sizes and budget-cut diagnostics are specified above. Zero-spread
diagnostics remain. The existing explicit NF4 and historical SFT utilities remain
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
smoke (field preparation, complete 40-turn evaluation episodes including native
grading, kernel check). Evaluation uses ceil(mean complete time) = 816 s (mean
815.54 s, max 996.94 s), checked from `episode.json` timestamps by
`test_plan_caps_fit_the_measured_evaluation_timings`; the
evaluation-rules GRPO step (1,700 s for six rollouts, capped at 3,400 s) is an estimate until its
smoke measures it, and evaluation stages are capped at 30 × 1,200 s. Reproduce them
with `python -m train.cloud.run --plan --recipe smoke` in the cloud environment.

Reproduce CPU checks, lock validation, bundle rebuild and verification:

```sh
python3 internal/reports/WP2.16-evidence/reproduce.py
python3 internal/reports/WP2.16-evidence/reproduce.py --summary
```

The WP2.15 reproducer now checks an explicit reviewed-source SHA256 manifest,
independent of git HEAD, and installed distribution versions against the lock
(including Torch’s CUDA local-version suffix). Its default environment is
`train/cloud/.venv-verify`. WP2.16 evidence is kept separately from the old smoke.

No HF write, paid call, GPU job, commit or publication is part of this package.
