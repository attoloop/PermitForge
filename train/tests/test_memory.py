import argparse
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from train.grpo import config_values
from train.memory import attention_context, chunked_token_stats, memory_trainer_class
from train.runtime import add_arguments, validate_arguments


def args_for(*argv):
    parser = argparse.ArgumentParser()
    add_arguments(parser, 'grpo')
    args = parser.parse_args(argv)
    validate_arguments(args)
    return args


def test_opt_in_plumbing_and_unchanged_defaults():
    defaults = config_values(args_for())
    assert defaults['num_generations'] == 8
    assert defaults['max_completion_length'] == 3072
    assert defaults.get('generation_batch_size') is None
    args = args_for('--num-generations', '4', '--generation-batch-size', '4',
                    '--generation-microbatch-size', '1', '--loss-chunk-size', '64',
                    '--sdpa-backend', 'fused', '--empty-cache-after-generation', '--memory-probe')
    config = config_values(args)
    assert config['generation_batch_size'] == 4
    assert config['max_steps'] == 1
    assert config['beta'] == 0 and config['gradient_checkpointing']
    assert config['generation_kwargs']['use_cache'] is True
    assert config['generation_kwargs']['logits_to_keep'] == 1


def test_fused_attention_disables_math_and_restores_defaults():
    previous = (torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.flash_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled())
    with attention_context('fused'):
        assert not torch.backends.cuda.math_sdp_enabled()
        assert torch.backends.cuda.flash_sdp_enabled()
        assert torch.backends.cuda.mem_efficient_sdp_enabled()
    assert previous == (torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.flash_sdp_enabled(),
                        torch.backends.cuda.mem_efficient_sdp_enabled())


def test_chunked_backbone_and_projection_enter_cuda_autocast(monkeypatch):
    import train.memory as memory
    active = []
    @contextmanager
    def autocast(**kwargs):
        assert kwargs == {'device_type': 'cuda', 'dtype': torch.bfloat16, 'enabled': True}
        active.append(True)
        try:
            yield
        finally:
            active.pop()
    monkeypatch.setattr(torch, 'autocast', autocast)
    class Base:
        def _get_last_hidden_state(self, *args):
            assert active, 'Backbone bypassed autocast'
            return 'hidden'
    def project(*args):
        assert active, 'Projection bypassed autocast'
        return 'logps', 'entropy'
    monkeypatch.setattr(memory, 'chunked_token_stats', project)
    trainer = memory_trainer_class(Base, SimpleNamespace(loss_chunk_size=64))()
    trainer._entropy_bonus_enabled = False
    trainer.temperature = 1.
    trainer.accelerator = SimpleNamespace(device=torch.device('cuda'), unwrap_model=lambda model: model)
    model = SimpleNamespace(config=SimpleNamespace(model_type='qwen3'), get_output_embeddings=lambda: 'head')
    result = trainer._get_per_token_logps_and_entropies(model, torch.ones(1, 4), None, 2)
    assert result == ('logps', 'entropy', None)


@pytest.mark.parametrize('argv', [
    ('--generation-batch-size', '3'), ('--generation-microbatch-size', '0'),
    ('--loss-chunk-size', '-1'),
    ('--memory-probe', '--num-generations', '4', '--generation-batch-size', '4', '--accum', '4'),
])
def test_bad_memory_config_rejected(argv):
    with pytest.raises(ValueError):
        args_for(*argv)


@pytest.mark.parametrize('chunk', [1, 4, 64])
def test_chunked_logits_match_values_entropy_and_gradients(chunk):
    torch.manual_seed(7)
    hidden = torch.randn(2, 7, 5, requires_grad=True)
    head = torch.nn.Linear(5, 19)
    targets = torch.randint(0, 19, (2, 7))
    full = head(hidden) / .7
    logs = full.log_softmax(-1)
    expected = logs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = -(logs.exp() * logs).sum(-1)
    expected.square().sum().backward()
    grads = [t.grad.clone() for t in (hidden, head.weight, head.bias)]
    for t in (hidden, head.weight, head.bias):
        t.grad = None
    actual, actual_entropy = chunked_token_stats(hidden, head, targets, .7, chunk, True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_entropy, entropy.detach())
    actual.square().sum().backward()
    for t, grad in zip((hidden, head.weight, head.bias), grads):
        torch.testing.assert_close(t.grad, grad)


def test_chunked_logits_preserve_accelerate_float32_output_precision():
    hidden = torch.randn(1, 5, 4, requires_grad=True)
    head = torch.nn.Linear(4, 13)
    targets = torch.arange(5).unsqueeze(0)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        # Accelerate converts the prepared model's output logits back to fp32.
        logits = head(hidden).float() / .7
        expected = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        actual, _ = chunked_token_stats(hidden, head, targets, .7, 2, False)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_microbatches_preserve_order_and_release_cache_after_generation(monkeypatch):
    events = []
    class Base:
        def _generate_single_turn(self, ids, images, fields, has_tool_images=False):
            events.append(list(ids))
            return [[x[0] + 10] for x in ids], None

        def _generate(self, prompts):
            events.append('generate')
            return ([[1]], [[2, 3]], [[1, 0]], [], None, {}, None, [])

    args = SimpleNamespace(generation_microbatch_size=2, loss_chunk_size=0,
                           empty_cache_after_generation=True)
    cls = memory_trainer_class(Base, args)
    trainer = cls()
    trainer.accelerator = SimpleNamespace(device=torch.device('cpu'))
    monkeypatch.setattr(torch, 'autocast', lambda *a, **kw: nullcontext())
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: events.append('empty'))
    ids, logs = trainer._generate_single_turn([[i] for i in range(5)], None, {})
    assert ids == [[i + 10] for i in range(5)] and logs is None
    assert events == [[[0], [1]], [[2], [3]], [[4]]]
    trainer._generate([])
    assert events[-2:] == ['generate', 'empty']
    assert trainer.memory_lengths == [{'prompt_tokens': 1, 'completion_tokens': 2, 'assistant_tokens': 1}]


def test_pinned_trl_config_and_sampler_cover_two_tasks_in_one_step(tmp_path):
    from trl import GRPOConfig
    from trl.trainer.utils import RepeatSampler
    from train.memory_sweep import configurations

    for config in configurations({8: 9728, 10: 12288, 12: 14336}):
        argv = ['--memory-probe', '--output', str(tmp_path)]
        for key, value in config.items():
            argv.extend(['--'+key.replace('_', '-')] if value is True
                        else ['--'+key.replace('_', '-'), str(value)])
        args = args_for(*argv)
        values = config_values(args) | {'use_cpu': True, 'bf16': False}
        resolved = GRPOConfig(**values)
        assert resolved.generation_batch_size == args.num_generations
        sampler = RepeatSampler([0, 1], mini_repeat_count=args.num_generations,
                                batch_size=resolved.generation_batch_size//args.num_generations,
                                repeat_count=resolved.steps_per_generation, shuffle=True, seed=0)
        # TRL consumes a fresh generation at each steps_per_generation boundary.
        batches = list(sampler)
        seen = [batches[i*resolved.generation_batch_size]
                for i in (0, resolved.steps_per_generation)]
        assert set(seen) == {0, 1}


def test_chunked_trl_path_matches_qwen_lora_backward():
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from trl import GRPOTrainer

    torch.manual_seed(9)
    config = Qwen3Config(vocab_size=31, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    model = get_peft_model(Qwen3ForCausalLM(config), LoraConfig(
        r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    options = SimpleNamespace(loss_chunk_size=3)
    cls = memory_trainer_class(GRPOTrainer, options)
    trainer = cls.__new__(cls)
    trainer.accelerator = Accelerator(cpu=True)
    trainer.model_kwarg_keys = ['logits_to_keep']
    trainer._is_vlm = False
    trainer._entropy_bonus_enabled = False
    trainer.temperature = .9
    ids = torch.randint(0, 31, (2, 9))
    mask = torch.ones_like(ids)
    mask[0, -2:] = 0
    expected, entropy, _ = GRPOTrainer._get_per_token_logps_and_entropies(
        trainer, model, ids, mask, 6, compute_entropy=True)
    expected.square().sum().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    actual, actual_entropy, aux = trainer._get_per_token_logps_and_entropies(
        model, ids, mask, 6, compute_entropy=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_entropy, entropy)
    assert aux is None
    actual.square().sum().backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(param.grad, grads[name])


@pytest.mark.parametrize('grading_failed', [False, True])
def test_memory_probe_keeps_model_load_peak(monkeypatch, tmp_path, grading_failed):
    from train.runtime import run_training
    import peft
    events = []
    for name in ('synchronize', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, lambda name=name: events.append(name))
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 123)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 456)
    monkeypatch.setattr(peft, 'get_peft_model_state_dict', lambda model: {'adapter': torch.zeros(1)})
    args = args_for('--memory-probe', '--num-generations', '2', '--generation-batch-size', '2',
                    '--accum', '4', '--output', str(tmp_path))
    class Trainer:
        model = object()
        args = SimpleNamespace(to_json_string=lambda: '{}')
        state = SimpleNamespace(global_step=1, log_history=[])
        def add_callback(self, callback):
            pass
        def train(self):
            pass
    rows = [{'task_id': task, 'provisional': True, 'final': {'reward': 0.}, 'grading_failed': grading_failed}
            for task in ('a', 'a', 'b', 'b')]
    if grading_failed:
        with pytest.raises(AssertionError, match='grading'):
            run_training(Trainer(), None, args, rows)
    else:
        run_training(Trainer(), None, args, rows)
    assert 'reset_peak_memory_stats' not in events
