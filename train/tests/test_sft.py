import copy

import pytest

from train.sft import episode_to_chat, synthetic_records, tokenize_rows


def test_episode_conversion_preserves_calls_and_hides_grade(episode, task_pack):
    original = copy.deepcopy(episode)
    row = episode_to_chat(episode, pack=task_pack[0])
    assert episode == original
    assert row['messages'][0]['role'] == 'system'
    assert row['tools']
    calls = [m for m in row['messages'] if m.get('tool_calls')]
    replies = [m for m in row['messages'] if m['role'] == 'tool']
    assert len(calls) == len(replies) == len(episode['steps'])
    assert calls[-1]['tool_calls'][0]['function']['name'] == 'submit_design'
    assert isinstance(calls[-1]['tool_calls'][0]['function']['arguments'], dict)
    assert 'anchors' not in replies[-1]['content']


@pytest.mark.parametrize('change', [
    {'split': 'eval'}, {'ended': 'no_submit'},
    {'final': {'feasible': False, 'prescore_usd_saved': 16}},
    {'final': {'feasible': True, 'prescore_usd_saved': 0}},
    {'final': {'feasible': True, 'prescore_usd_saved': float('nan')}},
    {'task_id': 'synthetic-eval'}, {'steps': []},
])
def test_bad_episode_filtered(episode, task_pack, change):
    assert episode_to_chat({**episode, **change}, pack=task_pack[0]) is None


def test_synthetic_only_smoke(task_pack):
    records = synthetic_records()
    assert len(records) == 4
    for record in records:
        assert record['synthetic'] is True
        assert episode_to_chat(record, pack=task_pack[0]) is None
        assert episode_to_chat(record, smoke=True)['messages']


def test_tokenize_before_arrow_preserves_dynamic_source_maps(episode, task_pack):
    row = episode_to_chat(episode, pack=task_pack[0])

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            final = next(m for m in reversed(messages) if m.get('tool_calls'))
            assert set(final['tool_calls'][0]['function']['arguments']['design']) == {'S1'}
            assert kwargs['return_assistant_tokens_mask'] is True
            return {'input_ids': [1, 2, 3], 'assistant_masks': [0, 1, 1]}
    assert tokenize_rows([row], Tokenizer(), 3) == [{'input_ids': [1, 2, 3], 'assistant_masks': [0, 1, 1]}]
    with pytest.raises(ValueError, match='exceeds'):
        tokenize_rows([row], Tokenizer(), 2)


def test_trl_precomputed_masks_exclude_tool_feedback(tmp_path):
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import SFTConfig, SFTTrainer

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {'[UNK]': 0, '[PAD]': 1, 'user': 2, 'assistant': 3}, unk_token='[UNK]')), pad_token='[PAD]')
    model = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_embd=8, n_layer=1, n_head=1))
    trainer = SFTTrainer(model=model, processing_class=tokenizer,
                         train_dataset=Dataset.from_list([{
                             'input_ids': [2, 3, 2, 3], 'assistant_masks': [0, 1, 0, 1]}]),
                         args=SFTConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False,
                                        report_to='none', max_length=8, assistant_only_loss=False))
    assert trainer.train_dataset[0]['labels'] == [-100, 3, -100, 3]
