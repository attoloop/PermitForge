"""Shared local QLoRA setup and measured smoke evidence; no remote training."""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import statistics
import time

MODEL_REVISIONS = {
    'Qwen/Qwen3-0.6B': 'c1899de289a04d12100db370d81485cdf75e47ca',
    'Qwen/Qwen3-1.7B': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
}


def add_arguments(parser: argparse.ArgumentParser, kind: str) -> None:
    parser.set_defaults(kind=kind)
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--revision', help='Pinned model commit; defaults are pinned for the two Qwen3 sizes')
    parser.add_argument('--adapter', type=Path, help='Optional local SFT LoRA to continue training')
    parser.add_argument('--output', type=Path, default=Path('train/runs') / kind)
    parser.add_argument('--tasks-dir', type=Path, default=Path('envs/permit/tasks'))
    parser.add_argument('--smoke', action='store_true', help='Exactly two optimizer steps; permits provisional anchors')
    parser.add_argument('--prepare-only', action='store_true', help='Validate data/config without loading model or training')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accum', type=int, default=8)
    parser.add_argument('--num-generations', type=int, default=8)
    parser.add_argument('--max-turns', type=int, default=8, choices=range(8, 13))
    parser.add_argument('--max-completion-length', type=int, default=3072)
    parser.add_argument('--max-length', type=int, default=6144, help='SFT total context limit')
    parser.add_argument('--max-prompt-length', type=int, default=3000)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-4bit', action='store_true', help='Use bf16 on a larger local/cloud GPU')


def validate_arguments(args: argparse.Namespace) -> None:
    for name in ('steps', 'batch_size', 'accum', 'num_generations', 'max_completion_length',
                 'max_length', 'max_prompt_length', 'lora_rank', 'lora_alpha'):
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('learning_rate must be positive and finite')
    if args.kind == 'grpo' and (args.num_generations < 2 or
                              args.batch_size * args.accum % args.num_generations):
        raise ValueError('batch_size * accum must be divisible by num_generations >= 2')
    args.revision = args.revision or MODEL_REVISIONS.get(args.model)
    if args.revision is None:
        raise ValueError('Pass --revision for a model outside the pinned Qwen3 defaults')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This memory-bounded recipe uses one process on one GPU')


def load_tokenizer(args: argparse.Namespace, *, sft: bool = False):
    from transformers import AutoTokenizer
    from trl.chat_template_utils import add_response_schema, get_training_chat_template

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                             cache_dir=Path.home()/'.cache/huggingface/hub')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right' if sft else 'left'
    if sft:
        tokenizer.chat_template = get_training_chat_template(tokenizer) or tokenizer.chat_template
        if not tokenizer.chat_template:
            raise ValueError('This model requires an assistant-mask training chat template')
        tokenizer.chat_template = '{%- set enable_thinking = false -%}\n' + tokenizer.chat_template
    else:
        tokenizer = add_response_schema(tokenizer)
    return tokenizer


def prompt_ids(tokenizer, messages: list[dict], tools: list[dict]) -> list[int]:
    return tokenizer.apply_chat_template(messages, tools=tools, return_dict=True,
                                          add_generation_prompt=True, enable_thinking=False)['input_ids']


def load_model(args: argparse.Namespace, *, sft: bool = False):
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run CPU tests and use these commands on a GPU-visible host')
    if torch.cuda.get_device_capability()[0] >= 12 and 'sm_120' not in torch.cuda.get_arch_list():
        raise RuntimeError('Install the pinned CUDA build with sm_120 support')
    torch.set_num_threads(4)
    set_seed(args.seed)
    tokenizer = load_tokenizer(args, sft=sft)
    quantization = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, cache_dir=Path.home()/'.cache/huggingface/hub',
        dtype=torch.bfloat16, device_map={'': 0}, quantization_config=quantization,
        attn_implementation='sdpa',
    )
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={'use_reentrant': False})
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM'))
    model.config.use_cache = False
    return model, tokenizer


def reward_groups(records: list[dict], group_size: int) -> list[dict]:
    groups = []
    for start in range(0, len(records), group_size):
        rows = records[start:start+group_size]
        ids = {row['task_id'] for row in rows}
        if len(ids) != 1:
            raise ValueError('A GRPO reward group contains different tasks')
        rewards = [None if row.get('grading_failed') else (row.get('final') or {}).get('reward', 0.)
                   for row in rows]
        graded = [r for r in rewards if r is not None and math.isfinite(r)]
        groups.append({'group': start//group_size, 'task_id': rows[0]['task_id'],
                       'complete': len(rows) == group_size, 'rewards': rewards,
                       'ungraded': len(rows)-len(graded),
                       'reward_std': statistics.pstdev(graded) if graded else None,
                       'reward_min': min(graded) if graded else None,
                       'reward_max': max(graded) if graded else None})
    return groups


def run_training(trainer, tokenizer, args: argparse.Namespace,
                 records_per_step: list[dict] | None) -> None:
    """Persist measurements even when the gradient, reward spread or reload check fails."""
    import torch
    from peft import get_peft_model_state_dict
    from transformers import TrainerCallback

    class StepTimer(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.start = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            step_seconds.append(time.perf_counter()-self.start)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'arguments.json').write_text(json.dumps(vars(args), default=str, indent=2)+'\n')
    (args.output/'training_config.json').write_text(trainer.args.to_json_string())
    initial = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(trainer.model).items()}
    step_seconds = []
    trainer.add_callback(StepTimer())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    summary = {'kind': args.kind, 'model': args.model, 'revision': args.revision,
               'smoke': args.smoke, 'status': 'failed', 'weights_changed': False,
               'adapter_reload_equal': False}
    try:
        trainer.train()
        trainer.save_model(str(args.output/'adapter'))
        tokenizer.save_pretrained(args.output/'adapter')
        trainer.save_state()
        trained = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(trainer.model).items()}
        changed = [k for k in initial if not torch.equal(initial[k], trained[k])]
        summary['weights_changed'] = bool(changed)
        summary['changed_tensors'] = len(changed)
        # Reload only the small adapter onto the same frozen base; never duplicate the base model.
        trainer.model.load_adapter(args.output/'adapter', adapter_name='reload_check', is_trainable=False)
        reloaded = get_peft_model_state_dict(trainer.model, adapter_name='reload_check')
        summary['adapter_reload_equal'] = set(trained) == set(reloaded) and all(
            torch.equal(trained[k], reloaded[k].detach().cpu()) for k in trained)
        trainer.model.delete_adapter('reload_check')
        assert summary['adapter_reload_equal'], 'Saved/reloaded LoRA differs from trained weights'
        assert summary['weights_changed'], 'LoRA did not change; inspect zero-spread rewards and truncation'
        if args.smoke and records_per_step is not None:
            assert len({r['task_id'] for r in records_per_step}) == 2, 'Smoke must exercise two train tasks'
        summary['status'] = 'passed'
    except Exception as exc:
        summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        elapsed = time.perf_counter()-started
        summary.update(elapsed_seconds=elapsed, step_seconds=step_seconds,
                       seconds_per_step=statistics.mean(step_seconds) if step_seconds else None,
                       optimizer_steps=trainer.state.global_step,
                       peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_vram_reserved_bytes=torch.cuda.max_memory_reserved())
        if records_per_step is not None:
            summary.update(episodes=len(records_per_step), episodes_per_hour=len(records_per_step)*3600/elapsed,
                           provisional=any(r['provisional'] for r in records_per_step),
                           groups=reward_groups(records_per_step, args.num_generations))
        (args.output/'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        # Trainer logs may contain NaN for all-ungraded batches; encode them as null.
        logs = deepcopy(trainer.state.log_history)
        for row in logs:
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    row[key] = None
        (args.output/'trainer_metrics.json').write_text(json.dumps(logs, indent=2, allow_nan=False)+'\n')
