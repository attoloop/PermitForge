from pathlib import Path

import pytest

from train.cloud.config import CloudConfig
from train.cloud.training import training_options, validate_loading_info
from train.cloud.storage import sync_command


def test_cloud_training_never_loads_a_second_generator():
    cfg = CloudConfig(hourly_usd=2.75)
    opts = training_options(cfg, 'grpo', Path('runs/output'), smoke=True)
    assert opts['max_steps'] == 2
    assert opts['use_vllm'] is False
    assert opts['num_generations'] == 6
    assert opts['generation_batch_size'] == 6
    assert opts['chat_template_kwargs'] == {'enable_thinking':False}
    assert opts['save_steps'] == 1
    assert opts['mask_truncated_completions'] is False


def test_missing_text_weights_fail_closed():
    validate_loading_info({'missing_keys':[], 'unexpected_keys':['model.visual.patch_embed.weight']})
    for info in [{'missing_keys':['model.layers.0.self_attn.q_proj.weight']},
                 {'unexpected_keys':['model.language_model.layers.0.weight']},
                 {'mismatched_keys':['lm_head.weight']}]:
        with pytest.raises(ValueError):
            validate_loading_info(info)


def test_storage_paths_are_explicit_and_scoped():
    assert sync_command(Path('runs/run'), 'hf://buckets/org/run/experiment') == [
        'hf','sync','runs/run','hf://buckets/org/run/experiment']
    command = sync_command(Path('runs/run'), 'hf://datasets/org/run/experiment')
    assert command[:6] == ['hf','upload','org/run','runs/run','experiment','--repo-type']
    for uri in ['org/run','hf://datasets/org/run','hf://buckets/org/run/../root']:
        with pytest.raises(ValueError):
            sync_command(Path('runs/run'), uri)


def test_text_backbone_mapping_matches_multimodal_checkpoint_on_cpu(tmp_path):
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM
    config = Qwen3_5Config(text_config=dict(hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        vocab_size=128, layer_types=['full_attention'], partial_rotary_factor=.5),
        vision_config=dict(hidden_size=32, intermediate_size=64, depth=1, num_heads=2,
                           out_hidden_size=32, num_position_embeddings=16))
    source = Qwen3_5ForConditionalGeneration(config)
    source.save_pretrained(tmp_path)
    text, info = Qwen3_5ForCausalLM.from_pretrained(tmp_path, config=config.text_config, output_loading_info=True)
    validate_loading_info(info)
    for name, weight in text.state_dict().items():
        source_name = name.replace('model.', 'model.language_model.', 1) if name.startswith('model.') else name
        assert torch.equal(weight, source.state_dict()[source_name])
    assert not any('visual' in name for name,_ in text.named_parameters())


def test_recorded_tokenizer_masks_and_trl_parser():
    from train.cloud.check_tokens import audit
    from train.cloud.formatting import load_tokenizer
    from train.cloud.make_sft import build_dataset
    root = Path(__file__).resolve().parents[3]
    rows, _ = build_dataset([root/'results/rollouts/wp2.0b-smoke'], root/'envs/permit/tasks')
    result = audit(rows, load_tokenizer(local_files_only=True))
    assert result['template_prefix_preserving'] is True
    assert result['episodes'][0]['trl_roundtrips'] == 10
    assert result['episodes'][0]['assistant_tokens'] < result['episodes'][0]['episode_tokens']
