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
Generation shares the base model, microbatch one, with no vLLM or reference copy.
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
- `causal-conv1d==1.7.0`: a publisher release wheel for cp312, CUDA 12, torch 2.10,
  Linux x86_64, C++11 ABI. The URL and SHA256 are locked. No source compile or nvcc
  dependency is introduced.
- `trackio==0.42.0`: optional live experiment metrics, enabled only with an
  explicitly supplied attoloop Space. Transformers passes the Space/bucket IDs
  and disables static Space export, matching FineEnvs' GeoGuesser recipe.

The kernel extras also lock `einops==0.8.2` for tensor rearrangements and
`ninja==1.13.2` as a causal-conv1d package dependency; the pinned wheel avoids a
local source build.

The bootstrap first installs the frozen base project, then attempts fast kernels
for at most 300 seconds. Installation failure in auto mode restores the base
project; required mode aborts. `USE_HUB_KERNELS=NO` prevents implicit Hub kernel
loads. Transformers 5.17.0's `use_kernel_func_from_hub_with_fallback` resolves local
FLA and causal-conv1d automatically. Runtime dispatch evidence and tiny
prefill/backward/recurrent/convolution probes precede model loading. Runtime
kernel failures abort instead of silently changing a training configuration.
The upstream Torch reference warns that GDN chunk execution is over an order of
magnitude slower on H100. That does not quantify H200 end-to-end cost. Fallback
keeps the same time/dollar cap and may time out.

The CPU installation probe installs only pinned optional packages and missing
Trackio dependencies into `/tmp`, using the existing locked torch environment.
All imports and four optimized dispatch targets resolve. CUDA is unavailable,
so Hopper JIT, numerical agreement, memory and real step times are unmeasured.
Reproduce the recorded package/dispatch checks from the repository root:

```sh
KERNEL_TARGET="${TMPDIR:-/tmp}/permit-wp212-kernels"
uv pip install --python train/.venv/bin/python --target "$KERNEL_TARGET" --no-deps \
  'flash-linear-attention==0.5.2' 'fla-core==0.5.2' 'einops==0.8.2' \
  'trackio==0.42.0' \
  'causal-conv1d @ https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=8e81f8c76435ad31aa41edc6c0c9d26de971a293b190183d2816da71547614d7'
CUDA_VISIBLE_DEVICES='' USE_HUB_KERNELS=NO \
  PYTHONPATH="$KERNEL_TARGET":.:envs/permit/core:envs/permit/openenv \
  train/.venv/bin/python internal/reports/WP2.12-evidence/check_installs.py
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

HF's two-hour provider timeout supplies the smoke's $10 compute envelope at the
operator's $5/hour input rate, including package setup. Process caps and an outer
job deadline reserve cleanup/upload time; storage, taxes and unrelated account
activity are separate. No GPU run, HF write or spend is part of this work package.
The README gives plans and exact operator commands. All timing estimates remain
priors until the smoke report contains measurements.

Primary references (implementation behavior was also checked in installed pins):

- [Pinned model config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)
- [Pinned Qwen implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Pinned kernel fallback dispatch](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/integrations/hub_kernels.py)
- [FLA package/release requirements](https://pypi.org/project/flash-linear-attention/0.5.2/)
- [Causal-convolution publisher releases](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0)
- [HF Jobs volumes, H200 and pricing](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)
- [Trackio Transformers integration](https://huggingface.co/docs/trackio/transformers_integration)

The FineEnvs checkout named in the brief was read in place, especially
`03-geoguesser/train/{README.md,grpo_geoguesser.py}` and the OCR job recipes under
`01-latex-ocr/train/` and `06-multilingual/ocr/train/`. They inform the job bootstrap,
checkpoint durability and live Trackio configuration; no files there were changed.
