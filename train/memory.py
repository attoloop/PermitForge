"""Opt-in memory controls for the pinned TRL text-only Qwen recipe.

CPU checks: uv run --project train pytest -c train/pyproject.toml train/tests/test_memory.py.
"""

from contextlib import nullcontext


def attention_context(backend: str):
    if backend == 'auto':
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    choices = {'flash': [SDPBackend.FLASH_ATTENTION], 'efficient': [SDPBackend.EFFICIENT_ATTENTION],
               'fused': [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]}
    return sdpa_kernel(choices[backend])


def chunked_token_stats(hidden, head, targets, temperature: float, chunk_size: int,
                        compute_entropy: bool):
    """Same selected log-probs as TRL, with recomputed rather than retained logits."""
    import torch
    from torch.utils.checkpoint import checkpoint
    from trl.trainer.utils import entropy_from_logits, selective_log_softmax

    def project(values, ids):
        # Accelerate's normal model.forward wrapper returns fp32 logits even
        # under bf16 autocast; preserve that log-softmax precision per chunk.
        logits = head(values).float() / temperature
        logps = selective_log_softmax(logits, ids)
        with torch.no_grad():
            entropy = entropy_from_logits(logits) if compute_entropy else torch.zeros_like(logps)
        return logps, entropy

    logps, entropies = [], []
    for start in range(0, hidden.shape[1], chunk_size):
        values, ids = hidden[:, start:start+chunk_size], targets[:, start:start+chunk_size]
        if torch.is_grad_enabled():
            logs, entropy = checkpoint(project, values, ids, use_reentrant=False)
        else:
            logs, entropy = project(values, ids)
        logps.append(logs)
        entropies.append(entropy)
    return torch.cat(logps, dim=1), torch.cat(entropies, dim=1) if compute_entropy else None


def memory_trainer_class(base, options):
    """Keep TRL's tool loop, masks, grouping and objective; change tensor scheduling."""
    import torch

    class MemoryTrainer(base):
        def _generate_single_turn(self, prompt_ids, images, multimodal_fields, has_tool_images=False):
            if images or multimodal_fields or has_tool_images:
                raise ValueError('Memory generation is text-only')
            size = options.generation_microbatch_size or len(prompt_ids)
            ids = []
            # k-bit preparation leaves some layers fp32. Generation in the pinned
            # TRL path lacks autocast; ensure attention sees bf16 on CUDA too.
            with torch.autocast(device_type=self.accelerator.device.type, dtype=torch.bfloat16,
                                enabled=self.accelerator.device.type == 'cuda'):
                for start in range(0, len(prompt_ids), size):
                    chunk, logps = super()._generate_single_turn(prompt_ids[start:start+size], None, {})
                    if logps is not None:
                        raise ValueError('Memory generation requires HF generate (use_vllm=False)')
                    ids.extend(chunk)
            return ids, None

        def _generate(self, prompts):
            result = super()._generate(prompts)
            prompt_ids, completion_ids, tool_masks = result[:3]
            if not hasattr(self, 'memory_lengths'):
                self.memory_lengths = []
            for i, (prompt, completion) in enumerate(zip(prompt_ids, completion_ids, strict=True)):
                self.memory_lengths.append(dict(prompt_tokens=len(prompt), completion_tokens=len(completion),
                    assistant_tokens=sum(tool_masks[i]) if tool_masks is not None else len(completion)))
            # _generate has returned Python lists: HF's per-turn KV cache and
            # tensors are out of scope, before TRL constructs update tensors.
            if options.empty_cache_after_generation:
                torch.cuda.empty_cache()
            return result

        def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask,
                                              logits_to_keep, batch_size=None, compute_entropy=False,
                                              compute_aux_loss=False, **kwargs):
            if not options.loss_chunk_size:
                return super()._get_per_token_logps_and_entropies(
                    model, input_ids, attention_mask, logits_to_keep, batch_size=batch_size,
                    compute_entropy=compute_entropy, compute_aux_loss=compute_aux_loss, **kwargs)
            if compute_aux_loss or self._entropy_bonus_enabled or any(v is not None for v in kwargs.values()):
                raise ValueError('Chunked loss supports text-only Qwen without auxiliary/entropy-bonus loss')
            unwrapped = self.accelerator.unwrap_model(model)
            if unwrapped.config.model_type != 'qwen3':
                raise ValueError('Chunked loss is validated only for Qwen3')
            # Calling the backbone directly bypasses Accelerate's model.forward
            # autocast wrapper. Keep this context active for checkpoint capture too.
            with torch.autocast(device_type=self.accelerator.device.type, dtype=torch.bfloat16,
                                enabled=self.accelerator.device.type == 'cuda'):
                hidden = self._get_last_hidden_state(unwrapped, input_ids, attention_mask, logits_to_keep)
                logps, entropies = chunked_token_stats(hidden, unwrapped.get_output_embeddings(),
                    input_ids[:, -logits_to_keep:], self.temperature, options.loss_chunk_size, compute_entropy)
            return logps, entropies, None

    return MemoryTrainer


def attention_probe(model_config) -> dict:
    """Check actual GPU execution, including backward and masked prefill; no model weights."""
    import torch
    import torch.nn.functional as functional
    result = {}
    for backend in ('flash', 'efficient'):
        checks = {}
        for case in ('causal_gqa', 'masked_prefill', 'decode_gqa'):
            try:
                with attention_context(backend):
                    nq, nk = model_config.num_attention_heads, model_config.num_key_value_heads
                    if case == 'masked_prefill':
                        nk = nq  # Transformers repeats KV for an explicit padding mask.
                    qlen, klen = (1, 128) if case == 'decode_gqa' else (128, 128)
                    shape = (1, nq, qlen, model_config.head_dim)
                    q = torch.randn(shape, dtype=torch.bfloat16, device='cuda', requires_grad=True)
                    k = torch.randn(1, nk, klen, model_config.head_dim,
                                    dtype=torch.bfloat16, device='cuda', requires_grad=True)
                    v = torch.randn_like(k, requires_grad=True)
                    mask = None
                    if case == 'masked_prefill':
                        mask = torch.ones(qlen, klen, dtype=torch.bool, device='cuda').tril()
                        mask[:, :3] = False
                    out = functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                        is_causal=case == 'causal_gqa', enable_gqa=nq != nk)
                    out.float().sum().backward()
                    torch.cuda.synchronize()
                    checks[case] = {'supported': True}
            except RuntimeError as exc:
                checks[case] = {'supported': False, 'error': str(exc)}
        result[backend] = checks
    return result
