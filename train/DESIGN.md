# Training implementation decisions

Domain validation, exact cost, field evaluation and calibration remain in
`permit_core`. `PermitTrainingEnv` reuses `permit_openenv.tasks` and the public
instructions. Its only public methods beyond TRL's `reset` and `get_reward`
lifecycle hooks are the four tools. Reset returns compact public tables; tools
return the established JSON payloads. Training instructions substitute the
trainer's tool-turn budget for the larger environment evaluation budget; the
environment itself still enforces invalid-is-free and submit-once accounting.

The implementation follows these pinned FineEnvs references because the checkout
path named in WP2.0 was unavailable in this sandbox:

- [GeoGuesser contract, reset, reward and GRPO configuration](https://github.com/adithya-s-k/FineEnvs/blob/410fb3454a903646f814bb752513c56fed3eae78/03-geoguesser/train/grpo_geoguesser.py).
- [Whitebox trainer](https://github.com/adithya-s-k/FineEnvs/blob/410fb3454a903646f814bb752513c56fed3eae78/05-multi-harness-rl/train/whitebox.py).
- [Whitebox NaN-on-grading-failure behavior](https://github.com/adithya-s-k/FineEnvs/blob/410fb3454a903646f814bb752513c56fed3eae78/05-multi-harness-rl/envs/whitebox/smoldataenv_whitebox/environment.py).

No v1 source code was lifted into this package. The adapter deliberately does
not import the OpenEnv HTTP server or fall back to native verification during
training.

Dependencies are confined to this work package. `pyproject.toml`, `uv.lock` and
the generated `requirements.lock` pin the complete resolved stack. PyTorch
provides CUDA/bfloat16 and gradient checkpointing; TRL provides SFT and
`GRPOTrainer(environment_factory=...)`; Transformers provides the pinned Qwen
tokenizer, tool parser and Hugging Face generation; PEFT provides LoRA;
bitsandbytes provides NF4 QLoRA and the paged optimizer; Accelerate is TRL's
single-process trainer backend; Datasets stores prepared examples. NumPy and
PyYAML are existing core dependencies; pytest and Ruff are development checks.
No additional training framework is introduced.

The selected execution path is the WP's TRL + bitsandbytes + HF-generation
fallback, with `use_vllm=False`. Unsloth's environment-factory integration could
not be established on the unavailable GPU, so it is not installed or represented
as tested. This is an uncertainty, not a claim that Unsloth is incompatible.
There is no colocated vLLM model copy. The lock selects the official CUDA wheel;
`python -m train.stack` records its compiled architecture flags separately from
actual CUDA availability.

Qwen model revisions are in `runtime.MODEL_REVISIONS`; other model IDs require
an explicit revision. Model loading uses no remote model code. Outputs contain
the chosen revision, arguments, resolved trainer configuration, traces, training
logs and metrics. The model tokenizer is checked for tool parsing; SFT uses
TRL's training chat template and masks assistant tokens before Arrow can alter
variable source dictionaries. The CPU regression initializes an SFT trainer
with a tiny random model only to verify labels; it performs no optimizer step.

Provisional anchors are never persisted into task packs. Their source and
`provisional` flag are recorded in selection and episode evidence, and they are
exposed to the policy only after submission. Normal training rejects missing,
unlabelled, baseline and search anchors. Provisional savings are taken only from
positive, finite, feasible diagnostics with matching task identity and split.
Search quality is not presented as a reward calibration result.

This WP does not generate fields. Read-only memory maps let the training process
reuse field inputs across rollouts without changing the concurrently owned
shared cache. A small shared cache bounds retained task packs. The selected
native fields, baseline files, actual CUDA execution and Unsloth integration
remain external prerequisites or unmeasured limitations, not silent fallbacks.

## WP2.3c: local memory controls and token measurements

The GPU run recorded in `internal/reports/gpu-smoke.log` failed inside SDPA
during GRPO generation. This work adds opt-in tensor scheduling controls to the
existing TRL recipe. `permit_env.py`, observations, tool schemas and physics
are unchanged. There are no new dependencies, model copies, or field writes.

### CPU tokenizer measurements

Reproduce every number in the following tables and the suggested limits with:

```bash
train/.venv/bin/python -m train.token_budget --output train/evidence/token-budget.json --markdown train/evidence/token-budget.md
```

The deterministic sample selects the first two eligible train IDs, sorted by
ID, for each source count from two through six. Both pinned Qwen tokenizers
in `runtime.MODEL_REVISIONS` produce the same counts; the JSON records each
model separately. Instructions and scenario columns count raw text separately.
Full prompt includes the actual Transformers-generated tool schemas, Qwen chat
template, initial user instruction and generation prefix. Result columns count
compact JSON payload text, excluding the surrounding chat delimiters.

These are constructed sizing episodes, not observed policy rollouts. Each uses
one `get_scenario`, two checks, the remaining evaluations, and a submission,
followed by a short acknowledgment. Calls reuse the draft design; evaluate and
submit payloads use stored ARM2 draft verification values from
`convert/cache/drafts-arm2/<id>.json`. Submission uses pending task anchors for
payload sizing only; this does not generate or persist training anchors. The
JSON includes input and draft-evidence hashes. These representative payloads
account for the unchanged tool reply sizes without running native models or
field computation. Longer reasoning, different values, errors and numeric
calibration anchors will change actual trajectory lengths.

| Task | Sources | Instructions | Scenario | Full prompt | get_scenario | check | evaluate | submit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 074897d0f159-s47 | 2 | 321 | 1035 | 1855 | 1239 | 29 | 763 | 812 |
| 0dcaf34d070e-s47 | 2 | 321 | 1031 | 1851 | 1235 | 29 | 748 | 797 |
| 00409ef904a8-s47 | 3 | 321 | 1296 | 2116 | 1588 | 29 | 759 | 808 |
| 0227870eb587-s47 | 3 | 321 | 1286 | 2106 | 1578 | 30 | 759 | 808 |
| 0c0f1dfdc277-s47 | 4 | 321 | 1541 | 2361 | 1921 | 30 | 746 | 795 |
| 120bc00dc814-s47 | 4 | 321 | 1553 | 2373 | 1933 | 30 | 768 | 817 |
| 003bd328a392-s47 | 5 | 321 | 1805 | 2625 | 2273 | 29 | 759 | 808 |
| 068cd7da58fd-s47 | 5 | 321 | 1800 | 2620 | 2268 | 30 | 750 | 799 |
| 01addc2a86f1-s47 | 6 | 321 | 2043 | 2863 | 2599 | 30 | 759 | 808 |
| 04ef47cb1279-s47 | 6 | 321 | 2015 | 2835 | 2571 | 29 | 752 | 801 |

Episode cells are **completion / total context** tokens.

| Task | 8 calls | 10 calls | 12 calls |
|---|---:|---:|---:|
| 074897d0f159-s47 | 5710 / 7565 | 7384 / 9240 | 9058 / 10914 |
| 0dcaf34d070e-s47 | 5631 / 7482 | 7275 / 9127 | 8919 / 10771 |
| 00409ef904a8-s47 | 6214 / 8330 | 7930 / 10047 | 9646 / 11763 |
| 0227870eb587-s47 | 6199 / 8305 | 7913 / 10020 | 9627 / 11734 |
| 0c0f1dfdc277-s47 | 6631 / 8992 | 8363 / 10725 | 10095 / 12457 |
| 120bc00dc814-s47 | 6767 / 9140 | 8547 / 10921 | 10327 / 12701 |
| 003bd328a392-s47 | 7221 / 9846 | 9029 / 11655 | 10837 / 13463 |
| 068cd7da58fd-s47 | 7173 / 9793 | 8963 / 11584 | 10753 / 13374 |
| 01addc2a86f1-s47 | 7710 / 10573 | 9564 / 12428 | 11418 / 14282 |
| 04ef47cb1279-s47 | 7603 / 10438 | 9431 / 12267 | 11259 / 14095 |

### Context and configuration choices

The pinned TRL `_tool_call_loop` counts both assistant tokens and tool reply
tokens against `max_completion_length`. The existing cloud default of 3072
cannot contain any of the scripted episodes above. Reducing it further would
truncate the task. The local sweep therefore reduces memory through batching
and kernels while sizing context from the measured conversations. It reserves
25% above the largest measured completion and rounds up to a multiple of 512:
9728 for eight calls, 12288 for ten, 14336 for twelve. The prompt ceiling is
3072. These are measured sizing recommendations, not a guarantee against
truncation or OOM. Regenerate the evidence after changes to prompts or tools.

Reproduce the ordered sweep's concrete configs and commands without GPU work:

```bash
train/.venv/bin/python -m train.memory_sweep --fields-dir convert/cache/fields --prepare-only --output train/evidence/memory-sweep-plan.json
```

| Preference | Generations / generation batch | HF generation microbatch | Tool turns | Completion cap | Gradient accumulation |
|---|---:|---:|---:|---:|---:|
| First | 4 / 4 | 4 | 12 | 14336 | 8 |
| Next | 4 / 4 | 2 | 12 | 14336 | 8 |
| Next | 4 / 4 | 1 | 12 | 14336 | 8 |
| Next | 4 / 4 | 1 | 10 | 12288 | 8 |
| Next | 4 / 4 | 1 | 8 | 9728 | 8 |
| Floor | 2 / 2 | 1 | 8 | 9728 | 4 |

Each configuration takes one optimizer step over two tasks, with one sample
per update microbatch. TRL's generation batch contains one complete reward
group; gradient accumulation spans both task groups. The CPU sampler test
checks the pinned TRL `RepeatSampler` behavior. The order is a capacity
preference from larger to smaller expected memory demand; only a GPU run can
establish measured memory ordering. Both models run in separate, sequential
processes; the first successful configuration is recorded per model.

The cloud/default arguments remain unchanged. Local flags enable:

- NF4 with double quantization and bf16 compute, existing LoRA and non-reentrant
  gradient checkpointing. `beta=0` avoids a reference model. The existing paged
  eight-bit optimizer only maintains trainable adapter state; explicit CPU
  optimizer offload is deferred until measurements show it is needed.
- Smaller `generation_batch_size`, keeping complete GRPO groups. HF generation
  can then be split into smaller batches without splitting reward computation
  or changing the tool loop. Generation enables its KV cache and requests only
  the last position's logits. bf16 autocast covers generation as well as update;
  the pinned TRL HF generation path otherwise uses `no_grad` without autocast.
- A checkpointed vocabulary projection (`--loss-chunk-size 64` in the sweep).
  The backbone runs once and token chunks compute the same selected log-probs
  and detached entropy using TRL's helpers. Backward recomputes each chunk
  rather than retaining full-sequence vocabulary logits. TRL still owns the
  loss, advantage normalization, tool mask and truncated-completion mask.
  CPU tests compare outputs and LoRA gradients with stock TRL on a tiny Qwen
  model. This implements chunked loss with the existing PyTorch dependency;
  explicit autocast covers the backbone and projection because this path
  bypasses Accelerate's forward wrapper. Each chunk's logits return to fp32
  before temperature scaling and log-softmax, matching that wrapper.
  Liger is not required or claimed tested. The chunked path rejects other
  model families, multimodal inputs, MoE auxiliary loss and entropy bonuses.
- `empty_cache` after `_generate` returns Python lists and before TRL builds
  update tensors. This releases allocator cache, not live tensors; it cannot
  make an intrinsically oversized configuration fit.

### Attention compatibility and GPU handoff

`--sdpa-backend fused` uses PyTorch's Flash or memory-efficient SDPA kernels
with the math fallback disabled during generation, update and backward. The
separate `flash` and `efficient` options allow diagnosis. This uses the
installed PyTorch kernels, not the external `flash-attn` extension. Unsupported
kernel/device/mask combinations become explicit failed trials. Backend
selection follows the [PyTorch API](https://docs.pytorch.org/docs/2.10/generated/torch.nn.attention.sdpa_kernel.html).
The general batching/checkpointing tradeoffs are described in the
[TRL memory guide](https://huggingface.co/docs/trl/reducing_memory_usage).

The installed wheel reports `sm_120` among its compiled targets, while CUDA is
unavailable in this sandbox. That establishes build support, not successful
attention dispatch on this GPU. Reproduce the distinction with:

```bash
train/.venv/bin/python -m train.stack --output train/evidence/wp23c-stack.json
```

On the GPU host, every trial writes `attention.json` checking each fused
backend with the selected model's head layout, bf16, causal GQA, masked
prefill and cached-decode shapes, including backward. These small checks do
not prove long-context fit; the actual training step also runs with math
fallback disabled. Source behavior was checked against the installed locked
packages, reproducible with:

```bash
rg -n 'generation_batch_size|steps_per_generation' train/.venv/lib/python3.12/site-packages/trl/trainer/grpo_config.py
rg -n 'def _generate_single_turn|def _tool_call_loop|def _get_last_hidden_state|def _get_per_token_logps_and_entropies|torch.no_grad|overlong' train/.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py
rg -n 'use_gqa_in_sdpa|repeat_kv|scaled_dot_product_attention' train/.venv/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py
```

Run the reviewer's actual sweep with the command in `README.md`. It records
model-loading OOMs as well as training failures, peak allocated/reserved bytes,
whole-trial and step timing, full completion lengths (including tool replies),
assistant-only lengths, and per-task reward spread. Failed trials do not stop
later configurations. A zero-spread group is recorded honestly and may still
demonstrate memory fit; unavailable grading fails the probe. Adapter saving
and reloading remain the normal smoke's responsibility, not the memory probe.

No GPU sweep, throughput claim or fitting configuration is asserted here.
Successful short or unproductive completions do not establish useful GRPO
training or worst-case context fit. GPU measurements and model learning remain
for the reviewer; the CPU evidence and runnable sweep are this work package's
handoff.
