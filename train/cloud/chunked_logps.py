"""GRPOTrainer whose per-token log-probabilities never hold a whole completion's logits at once.

Smoke `smoke-evalrules-c1` (2026-10-18, job Attoloop/6ad4c41e5ccb47ba0f7c4004) generated six 40-turn rollouts
and then ran out of memory in TRL's loss: the output head produced logits for the longest completion in one
piece, 27,218 tokens x 248,320 vocabulary, 13.5 GB in bf16 and 27 GB once `selective_log_softmax` upcast
them, on top of the weights and checkpointed activations. TRL 1.12 chunks that computation only across the
batch (already 1). Here the backbone runs once as TRL's own Liger path does, and the output head plus TRL's
`selective_log_softmax` run per chunk of completion tokens under activation checkpointing, so a chunk's
logits exist only while it is computed and are recomputed in backward. Values and gradients equal TRL's
(tests/test_chunked_logps.py)."""
import torch
from torch.utils.checkpoint import checkpoint
from accelerate.utils import is_peft_model
from trl import GRPOTrainer
from trl.models.utils import _ForwardRedirection
from trl.trainer.utils import entropy_from_logits, selective_log_softmax


class ChunkedLogpsTrainer(GRPOTrainer):
    logp_chunk_tokens = 2048

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep,
                                           batch_size=None, compute_entropy=False, compute_aux_loss=False,
                                           **multimodal):
        if compute_aux_loss or any(value is not None for value in multimodal.values()):
            return super()._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep, batch_size=batch_size,
                compute_entropy=compute_entropy, compute_aux_loss=compute_aux_loss, **multimodal)
        unwrapped = self.accelerator.unwrap_model(model)
        lm_head = (unwrapped.base_model.model if is_peft_model(unwrapped) else unwrapped).lm_head
        redirect = _ForwardRedirection()
        batch_size = batch_size or input_ids.size(0)
        logps, entropies = [], []
        for start in range(0, input_ids.size(0), batch_size):
            ids = input_ids[start:start+batch_size]
            mask = attention_mask[start:start+batch_size]
            hidden = redirect(model, unwrapped, self._get_last_hidden_state, unwrapped, ids, mask, logits_to_keep)
            targets = ids[:, -logits_to_keep:]
            rows, row_entropies = [], []
            for t in range(0, logits_to_keep, self.logp_chunk_tokens):
                chunk_logps, chunk_entropies = checkpoint(
                    self._chunk_logps, lm_head, hidden[:, t:t+self.logp_chunk_tokens],
                    targets[:, t:t+self.logp_chunk_tokens], compute_entropy, use_reentrant=False)
                rows.append(chunk_logps)
                row_entropies.append(chunk_entropies)
            logps.append(torch.cat(rows, dim=1))
            if compute_entropy:
                entropies.append(torch.cat(row_entropies, dim=1))
        return torch.cat(logps), (torch.cat(entropies) if compute_entropy else None), None

    def _chunk_logps(self, lm_head, hidden, targets, compute_entropy):
        logits = lm_head(hidden) / self.temperature  # the same ops, dtypes and order as TRL's path
        chunk_logps = selective_log_softmax(logits, targets)
        if not compute_entropy:
            return chunk_logps, None
        if self._entropy_bonus_enabled:
            return chunk_logps, entropy_from_logits(logits)
        with torch.no_grad():
            return chunk_logps, entropy_from_logits(logits)
