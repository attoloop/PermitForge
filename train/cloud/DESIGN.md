# Cloud implementation decisions

This directory owns the cloud additions. It imports the existing task validation,
exact costs, saved-fraction reward, training environment, OpenEnv terminal grading
and HF generation microbatch helper; it changes none of their local defaults.
No domain logic is lifted or reimplemented here.

The pinned model has a vision encoder, but this experiment is text-only. The
explicit text-class loader uses Transformers 5.17.0's `qwen3_5_text` conversion
mapping. The CPU test saves a tiny multimodal checkpoint and compares every text
tensor after reloading. Missing/mismatched language keys abort. Frozen large
embedding/head weights are kept bf16 after k-bit preparation; only full-attention
q/k/v/o LoRA weights train. Gated DeltaNet and vision weights are not LoRA targets.
The normal HF sampler shares the training base. The existing Qwen3-only chunked
loss is disabled for this architecture.

Primary implementation references:

- [Pinned model config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)
- [Pinned Qwen chat template](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/chat_template.jinja)
- [Transformers text-class implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Transformers checkpoint conversion](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/conversion_mapping.py)
- [TRL Qwen3.8 parser/template support](https://github.com/huggingface/trl/blob/v1.12.0/trl/chat_template_utils.py)
- [bitsandbytes CUDA installation targets](https://huggingface.co/docs/bitsandbytes/installation)

The tokenizer remains byte-identical to the pinned inference template. Assistant
labels are derived from token offset mappings in complete rendered conversations.
They exclude the empty thinking prefix already supplied at generation time.
The audit verifies prefix preservation and parses each recorded call with TRL;
GRPO refuses an automatic template replacement.

The cloud project adds a separate frozen dependency lock. `psutil` is explicit
because the deadline supervisor must stop native subprocesses that start their
own sessions. `huggingface-hub` supplies artifact transfers and HF Jobs CLI
compatibility. Other named ML dependencies match the local recipe; OpenEnv is
installed here because the evaluation stage uses its actual public tool schemas
and native terminal grader. No vLLM, FlashAttention, FLA or custom CUDA kernels
are added. Gated DeltaNet speed/stability on the target GPU remains a smoke gate.

Native field preparation uses at most four workers. Core's cache-root function is
bound only inside each cloud worker to its selected cache volume. Initial warming
is resumable; evicted fields can be recomputed in a small CPU subprocess within
the requesting stage's limits. Training and base/adapter evaluation share the
same immutable task data. Native grading is isolated from the model's RAM because
the faithful verifier counts its own process RSS when checking resource limits.

Completion markers distinguish finished Trainer checkpoints from interrupted
writes. Artifact transfer sends payloads first, markers and run state last.
Stage reservations survive interruption. Transfer/setup time before the
entrypoint and provider idle/storage charges cannot be bounded by a training
process; the README distinguishes these from the process compute budget.

CPU tests do not establish the feasibility of NF4 Gated DeltaNet training. Only
the explicit cloud smoke can establish changed weights, correct reload, finite
training, measured memory and useful reward spread. A failed gate is retained,
not replaced by an invented successful measurement.
