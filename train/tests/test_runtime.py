import argparse
import json

import pytest

from train.grpo import config_values, select_tasks
from train.runtime import add_arguments, validate_arguments, reward_groups, prompt_ids


def arguments(argv=()):
    parser = argparse.ArgumentParser()
    add_arguments(parser, 'grpo')
    return parser.parse_args(argv)


def test_plan_configuration():
    args = arguments(['--smoke'])
    validate_arguments(args)
    config = config_values(args)
    assert config['num_generations'] == 8
    assert config['per_device_train_batch_size'] == 1
    assert config['gradient_accumulation_steps'] == 8
    assert config['max_tool_calling_iterations'] == 8
    assert config['max_completion_length'] == 3072
    assert config['scale_rewards'] == 'group'
    assert config['beta'] == 0
    assert config['mask_truncated_completions'] is True
    assert config['chat_template_kwargs'] == {'enable_thinking': False}
    assert config['max_steps'] == 2 and config['use_vllm'] is False


def test_selection_is_train_only(task_pack):
    pack = task_pack[0]
    assert [t.task_id for t in select_tasks(pack, ['synthetic-train'], 2)] == ['synthetic-train']
    with pytest.raises(ValueError, match='train'):
        select_tasks(pack, ['synthetic-eval'], 2)


def test_generation_batch_requires_whole_groups():
    with pytest.raises(ValueError, match='divisible'):
        validate_arguments(arguments(['--num-generations', '3']))


def test_group_statistics_keep_ungraded_separate():
    rows = [{'task_id': 'a', 'final': {'reward': r}, 'grading_failed': r is None}
            for r in [0, .4, None, .4, 0, 0, 0, 0]]
    groups = reward_groups(rows, 4)
    assert groups[0]['ungraded'] == 1
    assert groups[0]['reward_std'] > 0
    assert groups[1]['reward_std'] == 0
    json.dumps(groups, allow_nan=False)


def test_prompt_budget_counts_tokens_not_encoding_keys():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['return_dict'] is True
            assert kwargs['add_generation_prompt'] is True
            return {'input_ids': list(range(3100)), 'attention_mask': [1]*3100}
    assert len(prompt_ids(Tokenizer(), [], [])) == 3100
