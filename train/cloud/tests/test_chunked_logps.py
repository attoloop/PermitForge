"""Smoke c1 (2026-10-18) ran out of memory in TRL's loss: fp32 logits for a 27k-token completion over the
248,320-token vocabulary are 27 GB. The chunked trainer must give TRL's own per-token log-probabilities,
entropies and gradients while never handing the output head more than one chunk of completion tokens."""
from functools import partial
from pathlib import Path

from train.cloud.config import CloudConfig
from train.cloud.training import training_options

ROOT = Path(__file__).resolve().parents[3]


def tiny_trainer(tmp_path, cls):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from trl import GRPOConfig
    from trl.chat_template_utils import add_response_schema
    from train.cloud.formatting import load_tokenizer
    from train.cloud.training import LORA_TARGET, rollout_diagnostics
    from train.permit_env import PermitTrainingEnv
    from permit_openenv.tasks import load_pack
    torch.manual_seed(0)
    tokenizer = add_response_schema(load_tokenizer(local_files_only=True))
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1, head_dim=16,
        vocab_size=len(tokenizer), layer_types=['full_attention'], partial_rotary_factor=.5))
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
                                             target_modules=LORA_TARGET, task_type='CAUSAL_LM'))
    opts = training_options(CloudConfig(), 'grpo', tmp_path, smoke=True)
    opts.update(use_cpu=True, bf16=False, optim='adamw_torch')
    return cls(model=model, processing_class=tokenizer, args=GRPOConfig(**opts, reward_weights=[0.0]),
        reward_funcs=partial(rollout_diagnostics, config=CloudConfig(), tokenizer=tokenizer),
        train_dataset=Dataset.from_list([{'prompt':[{'role':'user','content':'Test'}],
                                         'split':'train', 'task_id':'074897d0f159-s47'}]),
        environment_factory=partial(PermitTrainingEnv, pack=load_pack(ROOT/'envs/permit/tasks'),
                                    trace_sink=lambda record: None))


def test_chunked_logps_equal_trl_values_and_gradients_without_full_logits(tmp_path):
    import torch
    from trl import GRPOTrainer
    from train.cloud.chunked_logps import ChunkedLogpsTrainer
    trainer = tiny_trainer(tmp_path, ChunkedLogpsTrainer)
    trainer.logp_chunk_tokens = 3
    model = trainer.model
    ids = torch.randint(0, model.config.vocab_size, (2, 11))
    mask = torch.ones_like(ids)
    keep = 7  # chunks of 3, 3 and 1 completion tokens
    seen = []
    lm_head = model.base_model.model.lm_head
    hook = lm_head.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[1]))
    reference = GRPOTrainer._get_per_token_logps_and_entropies(trainer, model, ids, mask, keep, compute_entropy=True)
    full, seen[:] = list(seen), []
    chunked = trainer._get_per_token_logps_and_entropies(model, ids, mask, keep, compute_entropy=True)
    hook.remove()
    assert full == [keep+1]            # TRL: the whole completion through the head at once
    assert max(seen) <= 3 and seen == [3, 3, 1]
    assert chunked[0].shape == (2, keep) and torch.allclose(chunked[0], reference[0], atol=1e-6)
    assert torch.allclose(chunked[1], reference[1], atol=1e-6)
    assert chunked[2] is None and reference[2] is None
    params = [p for p in model.parameters() if p.requires_grad]
    ours = torch.autograd.grad(chunked[0].sum(), params, allow_unused=True)
    theirs = torch.autograd.grad(reference[0].sum(), params, allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in theirs)
    for mine, trl in zip(ours, theirs, strict=True):
        assert (mine is None and trl is None) or torch.allclose(mine, trl, atol=1e-6)
    logps, entropies, aux = trainer._get_per_token_logps_and_entropies(model, ids, mask, keep)
    assert entropies is None and aux is None and torch.allclose(logps, reference[0], atol=1e-6)


def test_chunked_trainer_overrides_only_the_logp_computation():
    from trl import GRPOTrainer
    from train.cloud.chunked_logps import ChunkedLogpsTrainer
    assert issubclass(ChunkedLogpsTrainer, GRPOTrainer)
    assert {name for name in vars(ChunkedLogpsTrainer) if not name.startswith('__')} == {
        'logp_chunk_tokens', '_get_per_token_logps_and_entropies', '_chunk_logps'}
    assert ChunkedLogpsTrainer.logp_chunk_tokens == 2048


def test_worst_case_step_measures_the_full_budget_through_the_chunked_loss(tmp_path):
    from types import SimpleNamespace
    from train.cloud.chunked_logps import ChunkedLogpsTrainer
    from train.cloud.training import worst_case_step
    trainer = tiny_trainer(tmp_path, ChunkedLogpsTrainer)
    trainer.logp_chunk_tokens = 3
    seen = []
    hook = trainer.model.base_model.model.lm_head.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].shape[1]))
    budget = SimpleNamespace(max_length=12, max_completion_length=7, max_vram_bytes=10**12)
    result = worst_case_step(trainer, budget)
    hook.remove()
    assert seen == [3, 3, 1, 1, 3, 3]  # whole budget through the chunked path; backward recomputes per chunk
    assert result['completion_tokens'] == 7 and result['total_tokens'] == 12 and result['seconds'] > 0
    assert result['fits'] is True and result['peak_reserved_bytes'] >= 0 and 'error' not in result
    assert all(p.grad is None for p in trainer.model.parameters())
