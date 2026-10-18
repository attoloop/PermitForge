# Cloud implementation decisions

WP2.12 supersedes WP2.9's RTX PRO 6000/NF4/SFT workflow with one H200, bf16 LoRA,
and direct GRPO. Existing local `train/` defaults and SFT utilities are untouched.
Domain validation, exact cost, saved-fraction reward and native verification stay
in their existing modules. No domain logic or v1 code is copied here.

The pinned multimodal checkpoint is loaded with Transformers' explicit text-only
`Qwen3_5ForCausalLM` class and its `qwen3_5_text` prefix conversion. A tiny CPU test
compares every text tensor against the original multimodal checkpoint. Missing
or mismatched language keys abort loading. Vision weights are omitted. Only
language full-attention q/k/v/o LoRA weights train. Gated DeltaNet projections
are frozen. bf16 keeps the base in bf16; the explicit NF4 option uses double
quantization and restores large embedding/head tables to bf16 after k-bit setup.
Generation shares the base model, with no vLLM or reference copy. The baseline
uses microbatch one; WP2.14 qualifies persistent-cache microbatches one, two and
four against independent full-history execution before selecting a configuration.
The local Qwen3-specific chunked loss is disabled for this architecture.

GRPO uses TRL's environment factory, group-scaled rewards, beta zero, one update
per sampled group, four generations and four accumulation microsteps. Thinking
is off. The pinned tokenizer is unchanged; TRL may not substitute a new template.
The 16,384-token training context allocates 4,096 to the prompt and 12,288 to the
completion including tool feedback. This exceeds the eight-call sizing floor in
`train/DESIGN.md`; it cannot guarantee every sampled rollout fits. Truncated
completions are masked and zero-spread groups are exposed, never fabricated into
training signal.

The frozen project adds no training framework. Optional dependencies are:

- `flash-linear-attention==0.5.2` and its exact `fla-core==0.5.2`: Triton GDN kernels.
- `tilelang[nvcc]==0.1.15` and `apache-tvm-ffi==0.1.12`: FLA's gated-backward
  backend for Hopper and its compiler FFI. Torch 2.10.0+cu128 requires Triton
  3.6.0 exactly; FLA refuses gated backward with Triton >=3.4.0,<3.7.1 on Hopper.
  We keep Torch's dependency instead of overriding it. FLA automatically selects
  TileLang on Hopper only when an actual `nvcc` compiler is available.
- The compiler pins are `nvidia-cuda-nvcc`, `nvidia-cuda-crt`,
  and `nvidia-nvvm` **13.2.86**, with `nvidia-cuda-cccl==13.2.86` and
  `nvidia-cuda-runtime==13.2.86`. The CUDA 12 nvcc wheel contains
  ptxas, not nvcc, and cannot satisfy FLA's compiler check. The CUDA 13 compiler
  installs under `nvidia/cu13`; Torch still loads its pinned CUDA 12.8 runtime.
  TileLang JIT compiles device kernels. GPU driver/module-loading compatibility
  must still pass the H200 probe.
- `causal-conv1d==1.7.0`: a publisher release wheel for cp312, CUDA 12, torch 2.10,
  Linux x86_64, C++11 ABI. The URL and SHA256 are locked. This extension needs no
  local source build.
- `trackio==0.42.0`: optional live experiment metrics, enabled only with an
  explicitly supplied attoloop Space. Transformers passes the Space/bucket IDs
  and disables static Space export, matching FineEnvs' GeoGuesser recipe.
- `kernels==0.16.2`: the separately installable `attention` extra, required by
  Transformers' explicit pinned FA3 loader. Its locked transitive packages
  (`kernels-data`, `sigstore` and certificate/signature dependencies) support
  artifact loading and verification; no global kernel mapping is enabled.
- `liger-kernel==0.8.2`: the separately installable `loss` extra. Only
  `LigerFusedLinearGRPOLoss`, called through TRL's existing `compute_liger_loss`,
  is enabled after the unchanged objective passes comparison. The trainer's
  blanket Liger model patch is never requested.

The base project also pins `nvidia-ml-py==13.580.82`. The first H200 smoke's
memory-monitor thread exited because `torch.cuda.device_memory_used` could not
import this dependency. It is required even with fast kernels switched off.
TileLang also locks cloudpickle, ml-dtypes, torch-c-dlpack-ext and z3-solver.
Its nvcc extra locks Windows-only nvidia-cuda-nvrtc and nvidia-curand dependencies;
neither is installed by this Linux recipe.
TVM/TileLang are kernel compilers; no training framework is added.

The kernel extras also lock `einops==0.8.2` for tensor rearrangements and
`ninja==1.13.2` as a causal-conv1d package dependency; the pinned wheel avoids a
local source build.

The bootstrap installs the frozen base, attempts fast kernels for at most 300
seconds, then independently attempts attention and loss extras for at most 180
seconds each. An optional installation failure restores the last successful
frozen extras. `required` aborts on failed fast-kernel installation or qualification;
`off` explicitly unwraps all four GDN/convolution functions. `auto` records a
failed optimized probe and requires the Torch fallback to pass the same matrix.
There is no retry of a failed training rollout or optimizer step. Runtime
configuration selection happens entirely before the real training steps.
`USE_HUB_KERNELS=NO` disables implicit Hub replacement while the explicit pinned
FA3 attention loader remains usable. Local FLA/convolution dispatch is unchanged
when its probe passes.

Before model loading, `kernel_cases.py` compares GDN against an independent fp32
token recurrence on identical quantized inputs/cotangents. The real shape is
16 Q/K heads expanded to 48 value heads, each dimension 128; the convolution has
10,240 channels and width four. Seeds 42/212 cover lengths 63/64/65/129, nonzero
initial states and padded prefixes. The issue #640 regression adds B=1, T=1024,
H=4, D=100, fp16, external normalization and scale 1. Model loading independently
checks these constants against the pinned text config. Each chunk backward traces
and asserts execution of `chunk_bwd_dqkwg_tilelang`, not just an outer `fla.*`
wrapper or an imported backend. Outputs, final states, q/k/v/g/beta/initial-state
gradients, recurrent decode, convolution gradients and mutated caches are checked.
Every tensor must be finite and satisfy the unchanged elementwise
`rtol=.03, atol=.002` and relative L2 bound `.02`. No tolerance is fitted to a
candidate. Errors, geometry, seeds and executed inner calls are retained in
`kernels.json`, including a failed fast candidate when Torch fallback qualifies.
CPU tests retain the small original harness and test the independent recurrence,
fault rejection and backend tracing. A CUDA-unavailable probe cannot qualify GPU code.
The upstream Torch reference warns that GDN chunk execution is over an order of
magnitude slower on H100. That does not quantify H200 end-to-end cost. Fallback
keeps the same time/dollar cap and may time out.

WP2.12b installs the complete frozen cloud project with the fast-kernel extra into
`${TMPDIR:-/tmp}/permit-wp212b-cloud-env` (the initial workspace installation ran out of disk
and its incomplete environment was removed). It checks imports of the exact FLA
TileLang backend, nvcc discovery/version, NVML import, optimized dispatch, and the
full comparison harness on CPU references. Older TileLang 0.1.9 failed with the
resolver's TVM-FFI 0.1.14.post1 at import and with TVM-FFI 0.1.2 during lowering;
those diagnostics are retained in WP2.12b evidence. The selected release explicitly
requires TVM-FFI >=0.1.11,<0.1.13. CUDA compiler 13.2.86 fixes the rsqrt exception
declaration conflict seen with the older compiler and this host's glibc headers.
The installed FLA gated backward compiles offline for sm_90a with head dimensions
32 and 128. This checks lowering and compilation, not execution. GPU numerical
agreement, driver compatibility, memory and step times remain unmeasured.
Reproduce from the repository root:

```sh
export UV_PROJECT_ENVIRONMENT="${TMPDIR:-/tmp}/permit-wp212b-cloud-env"
export TILELANG_CACHE_DIR="${TMPDIR:-/tmp}/permit-wp212b-tilelang"
export TVM_FFI_CACHE_DIR="${TMPDIR:-/tmp}/permit-wp212b-tvm-ffi"
export USE_HUB_KERNELS=NO CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4
export PYTHONPATH=.:envs/permit/core:envs/permit/openenv
uv sync --project train/cloud --frozen --extra fast-kernels
uv run --project train/cloud --frozen --no-sync python \
  internal/reports/WP2.12b-evidence/check_stack.py
uv run --project train/cloud --frozen --no-sync python \
  internal/reports/WP2.12b-evidence/compile_hopper.py
```

The smoke warms exactly its five tasks, evaluates the untrained base on three
eval tasks once, frees that model, then takes two direct GRPO steps. Native
terminal grading is isolated from model RAM. A report is retained on worker
failure or supervisor timeout. A longer run requires matching provenance and a
passed smoke with valid tools, submission, positive reward, reward spread,
finite updates and changed/reloaded weights. The 135-billion-byte device guard
and lower allocator fraction are experimental limits, not fit claims.

The default thirty training identities are station-balanced deterministic hashes,
independent of rewards. The bundle contains task/met inputs only for those thirty
and the frozen ten eval tasks; full catalogs remain for identity validation.
Fields are regenerated within the job and use a bounded LRU cache. Concurrent
native work remains at most four processes.

The 20-step check retains a 100-step scheduler target and stops via a callback.
Complete checkpoints require adapter, trainer state, optimizer, scheduler, RNG and
completion marker. Subsequent invocations restore those states and continue;
finished base episodes are skipped. The stop point is an invocation control,
not an immutable hyperparameter. Model/source/data/config changes remain rejected.
Final eval/export require the fixed step target to be finished. Payload uploads
precede completion markers. Crash reservations remain charged and interrupted
native submissions cannot be selectively retried.

HF's three-hour provider timeout supplies the smoke's $15 compute envelope at the
operator's $5/hour input rate, including package setup. Process caps and an outer
job deadline reserve cleanup/upload time; storage, taxes and unrelated account
activity are separate. No GPU run, HF write or spend is part of this work package.
The README gives plans and exact operator commands. All timing estimates remain
priors until the smoke report contains measurements.

## WP2.14 qualification and selection

`attention.py` pins FA3 to
`kernels-community/flash-attn3@3c1451f803c146c54222305a8c350f7f46bb5135` and checks
the matching torch210/cu128 binary digest. Its variable-length kernel is tested
at 24 Q heads / 4 KV heads / dimension 256 for causal prefill, unequal padded
sequences and cached decode, including all input gradients against FP32 math
SDPA. Only full-attention layers change. The model gate compares selected-token
log probabilities, the actual TRL scalar losses, every adapter gradient and the
clipped optimizer update deltas against SDPA/existing loss, at both the initial
and a deterministic nonzero adapter. Tool and truncated-completion masks,
group-scaled advantages and accumulation/reduction are identical in both paths.
Synthetic diagnostic advantages are confined to this comparison; they never
replace actual environment rewards.

`generation.py` owns per-sequence attention KV, GDN recurrent and convolution
state. TRL supplies token histories rather than stable row IDs, so reuse claims
the longest exact token-prefix entry once from the active multiset. Identical
sibling histories are interchangeable at fixed weights but never share writable
tensor storage. Non-prefix edits/truncation cause a fresh prefill; unclaimed
entries are released. KV keys may be left-padded when merging different history
lengths, with unpadded position IDs and complete attention masks. Continuing
suffixes are bucketed by length so padding never advances a nonzero GDN state.
Prefill consumes all but the last prompt token. HF's returned cache likewise
trails the last sampled token, which remains in the next uncached suffix.
Stopping the batch at the first EOS permits cache splitting before any ended
row consumes padding. Remaining rows continue with the original sampling config.
The trainer keeps TRL's tool formatting, masks, budget and reward code; caches
are discarded at rollout-group boundaries, after optimizer updates and on
configuration/model changes. Checkpoint loading constructs a fresh generator.

The GPU state gate compares full histories with cached tool continuations and
single-token decode at candidate batch sizes, including every terminal KV/GDN/
convolution tensor. CPU tests also cover sibling ownership, active-row removal,
actual weight changes, reload, prefix edits and invalidation callbacks.

`qualification.py` runs before the two real smoke steps. The baseline must pass
first. Independent FA3, Liger and cache gates retain failures and select qualified
fallbacks. Fixed diagnostic traces use the selected smoke train tasks, unchanged
prompt/schema, a public draft validity/cost check and a final submit token sequence;
no native grade or synthetic learning occurs during replay. Forced tokens make
generation work identical across candidates. The warm comparison is generation
plus accumulated forward/backward/clipping/optimizer time on those traces; tool
execution, native verification and scheduler overhead are not in this ranking.
Cold/JIT times are recorded separately. After the independent gates, each passing
combination gets a sequential fixed-generation-plus-training benchmark; a combination
without that final timing pass cannot be selected. Caches are freed before loss
computation, as in real training. The fastest passing combination is
used for both smoke steps and persisted for longer jobs. RNG, adapters, gradients,
optimizer and trainer metrics are restored after diagnostic steps. The real smoke
still requires reward spread, finite changed weights and exact save/reload.

`qualification.json`, `qualification-gates.json`, `qualification-trajectories.json`
and real `generation-timings.json` distinguish diagnostic speed, chosen settings,
fallback reasons and actual time per turn. The prompt, cohort, group size,
objective, verifier, rollout budget and turn settings are unchanged. Future jobs
must match both source/data/config provenance and the smoke's kernel fallback.

The extra qualification prior is 2,400 seconds. The smoke plan totals 3,915 stage
seconds plus 900 setup seconds: 4,815 seconds and $6.6875 at $5/hour. Stage limits
are 10,200 seconds / $14.1667; the provider limit is 10,800 seconds / $15 including
setup. These are unmeasured allowances, not a completion or throughput guarantee.
Reproduce all arithmetic with `uv run --project train --no-sync python -m
train.cloud.run --plan --recipe smoke`. The optional loop checks its time allowance
between candidates; the existing stage/process/provider deadlines enforce the
hard cap even if a single compiler/model operation stalls. No new paid attempt
is authorized automatically after failure.

Reproduce the local frozen install, all CPU suites, lock/dependency/import checks,
bundle rebuild and bundle verification (no GPU execution or HF writes):

```sh
python3 internal/reports/WP2.14-evidence/reproduce.py
python3 internal/reports/WP2.14-evidence/reproduce.py --summary
```

## Later vLLM option

vLLM is deliberately absent: its compatible candidate changes the core binary
stack. A later isolated lock should use Python 3.12, `vllm==0.27.1`,
`torch==2.13.0`, `triton==3.7.1`, `tilelang==0.1.12`, `torchvision==0.28.0` and
FlashInfer `0.6.16.post3`, retaining Transformers 5.17.0 / TRL 1.12.0 / PEFT 0.19.0
and the exact model revision. Replace the cu128 index with the matching CUDA-13
family; relock TVM-FFI/compiler dependencies and FastAPI `<0.137.0`. Rebuild pinned
causal-conv1d 1.7.0 against that torch/CUDA family (the current torch2.10 wheel
cannot carry over). This candidate has not been resolved or installed here.

Integration must replace the HF generation override, keep one colocated engine
awake for the entire rollout group, and sleep only across the training boundary.
Pin the engine revision and verify text-only/multimodal weight-name mapping.
Qualify adapter synchronization, unchanged frozen base weights, teacher-forced
log probabilities, masks and updates. Invalidate every KV/GDN/conv/prefix cache
after policy changes; do not rely on TRL's per-generation sleep/sync behavior.
Start at tensor parallel one, context 16,384, bf16 KV, engine memory fraction
0.50 as an unqualified test point, `mamba_cache_mode='align'`, and test
`prefix_match_unit=16` with actual cache hits. Keep importance-sampling correction.
Measure total device memory, prefix reuse, graph behavior and complete optimizer
time in its own capped smoke. These settings are an exact proposed upgrade path,
not a fit, compatibility or speed claim. Sources and version metadata are retained
in `internal/reports/WP2.13-evidence/` and its report.

Primary references (implementation behavior was also checked in installed pins):

- [Pinned model config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)
- [Pinned Qwen implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Pinned kernel fallback dispatch](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/integrations/hub_kernels.py)
- [FLA package/release requirements](https://pypi.org/project/flash-linear-attention/0.5.2/)
- [FLA TileLang dispatch](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/common/backends/tilelang/__init__.py)
- [FLA compiler discovery](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/utils/_compat.py)
- [TileLang pinned wheel and compiler extra](https://pypi.org/pypi/tilelang/0.1.15/json)
- [Causal-convolution publisher releases](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0)
- [HF Jobs volumes, H200 and pricing](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)
- [Trackio Transformers integration](https://huggingface.co/docs/trackio/transformers_integration)

The FineEnvs checkout named in the brief was read in place, especially
`03-geoguesser/train/{README.md,grpo_geoguesser.py}` and the OCR job recipes under
`01-latex-ocr/train/` and `06-multilingual/ocr/train/`. They inform the job bootstrap,
checkpoint durability and live Trackio configuration; no files there were changed.
