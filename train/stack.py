"""Inspect the pinned stack without training: python -m train.stack --output train/evidence/stack.json."""

import argparse
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import platform
import subprocess


def probe() -> dict:
    import torch
    from trl import GRPOTrainer

    native = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False)
    available = torch.cuda.is_available()
    return {
        'python': platform.python_version(),
        'versions': {name: version(name) for name in (
            'torch', 'trl', 'transformers', 'peft', 'bitsandbytes', 'accelerate', 'datasets')},
        'cuda_runtime': torch.version.cuda, 'cuda_available': available,
        # The public get_arch_list returns [] when the driver is hidden. Record
        # build flags separately; they establish wheel support, not GPU usability.
        'compiled_arch_flags': torch._C._cuda_getArchFlags(),
        'environment_factory': 'environment_factory' in inspect.signature(GRPOTrainer.__init__).parameters,
        'nvidia_smi_exit_code': native.returncode,
        'nvidia_smi_output': native.stdout + native.stderr,
        'device': torch.cuda.get_device_name() if available else None,
        'smoke_status': 'ready' if available else 'not_run_gpu_unavailable',
        'peak_vram_bytes': None, 'seconds_per_step': None, 'episodes_per_hour': None,
        'reward_spread': None, 'weights_changed': None, 'adapter_reload_equal': None,
    }


def tokenizer_probe() -> dict:
    """Download only the pinned tokenizer and exercise TRL's actual parser/masks."""
    from trl.chat_template_utils import parse_response
    from permit_openenv.tasks import load_pack
    from .permit_env import scenario_text, tool_schemas, training_instructions
    from .runtime import add_arguments, load_tokenizer, prompt_ids, validate_arguments
    from .sft import episode_to_chat, synthetic_records, tokenize_rows

    parser = argparse.ArgumentParser()
    add_arguments(parser, 'grpo')
    args = parser.parse_args(['--model', 'Qwen/Qwen3-0.6B'])
    validate_arguments(args)
    tokenizer = load_tokenizer(args)
    prompt = [{'role': 'system', 'content': training_instructions(8)},
              {'role': 'user', 'content': 'Submit a design.'}]
    prefix = prompt_ids(tokenizer, prompt, tool_schemas())
    sample = ('<tool_call>\n{"name":"submit_design","arguments":'
              '{"design":{"S1":{"control":"S1_moderate","height_m":10}}}}\n</tool_call><|im_end|>')
    parsed = parse_response(tokenizer, tokenizer.encode(sample, add_special_tokens=False), prefix=prefix)
    assert parsed['tool_calls'][0]['function']['name'] == 'submit_design'
    sft_tokenizer = load_tokenizer(args, sft=True)
    tokenized = tokenize_rows([episode_to_chat(r, smoke=True) for r in synthetic_records()], sft_tokenizer, 6144)
    lengths = {}
    pack = load_pack()
    for task_id in ('074897d0f159-s47', '01addc2a86f1-s47'):
        task_prompt = [prompt[0], {'role': 'user', 'content':
                                  'Optimize the plant and submit your design.\n' + scenario_text(pack.get(task_id))}]
        lengths[task_id] = len(prompt_ids(tokenizer, task_prompt, tool_schemas()))
    assert max(lengths.values()) <= args.max_prompt_length, lengths
    return {'model': args.model, 'revision': args.revision, 'parsed': parsed,
            'prompt_tokens': lengths,
            'sft_masks': [{'tokens': len(row['input_ids']), 'assistant_tokens': sum(row['assistant_masks'])}
                          for row in tokenized]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('train/evidence/stack.json'))
    parser.add_argument('--check-tokenizer', action='store_true')
    args = parser.parse_args()
    evidence = probe()
    if args.check_tokenizer:
        evidence['tokenizer'] = tokenizer_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
