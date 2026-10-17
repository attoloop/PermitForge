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
