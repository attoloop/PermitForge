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

The bootstrap first installs the frozen base project, then attempts fast kernels
for at most 300 seconds. Installation failure in auto mode restores the base
project; required mode aborts. `USE_HUB_KERNELS=NO` prevents implicit Hub kernel
loads. Transformers 5.17.0's `use_kernel_func_from_hub_with_fallback` resolves local
FLA and causal-conv1d automatically. `--kernels off` explicitly replaces all four
Qwen dispatch functions with their unwrapped Torch implementations before model
loading, even if optional packages are installed. Required mode rejects reference
dispatch. Auto mode falls back only on installation failure; runtime kernel or
numerical failures abort instead of changing a training configuration.

Before model loading, `kernels.qualify` compares installed dispatch against
`inspect.unwrap` of the pinned Transformers functions: the non-FLA Torch
references. Deterministic bf16 probes cover chunk lengths 64 and 65 (padding and
cross-chunk state), nonzero initial state, outputs/final states and gradients for
q/k/v/g/beta/initial state with identical random cotangents. Recurrent decode checks
output and state. Convolution checks output, input/weight/bias gradients, decode
output and mutated cache. Every tensor must be finite, satisfy elementwise
`rtol=0.03, atol=0.002`, and have relative L2 error <=0.02. These bf16 tolerances
are fixed guards, not a measured H200 error bound. `kernels.json` records errors,
tolerances, seed, dispatch and versions. CPU regression tests inject finite output,
state, gradient and cache corruption to verify rejection. CUDA-unavailable probes
return `gpu_unavailable` without claiming numerical qualification.
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
