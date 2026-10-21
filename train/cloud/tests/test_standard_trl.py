"""WP2.15: upstream trainer owns generation, kernels and tool-loop integration."""
import ast
from pathlib import Path
import tomllib

from train.cloud.config import CloudConfig
from train.cloud.training import model_load_kwargs, training_options
from train.cloud.tests.checkout import requires_smoke_base_eval  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def test_standard_grpo_trainer_owns_generation_and_loss():
    tree = ast.parse((ROOT/'train/cloud/training.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    constructors = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'ChunkedLogpsTrainer']
    assert len(constructors) == 1  # TRL's trainer with only the log-prob computation chunked (test_chunked_logps.py)
    assert 'environment_factory' in {k.arg for k in constructors[0].keywords}
    assert not any(isinstance(n.func, ast.Name) and n.func.id == 'GRPOTrainer' for n in calls)
    assert not any(isinstance(n.func, ast.Name) and n.func.id in (
        'memory_trainer_class', 'cached_trainer_class', 'qualify_stack') for n in calls)
    opts = training_options(CloudConfig(), 'grpo', Path('unused'), smoke=True)
    assert opts['use_vllm'] is False
    assert opts['max_steps'] == 2
    assert opts['num_generations'] == opts['gradient_accumulation_steps'] == 6
    assert opts['max_tool_calling_iterations'] == 40  # the evaluation's turn limit (test_eval_rules.py)
    assert opts['beta'] == 0 and opts['scale_rewards'] == 'group'
    assert opts['mask_truncated_completions'] is False
    assert not opts.get('use_liger_kernel')


def test_transformers_selects_attention_without_local_overrides():
    assert 'attn_implementation' not in model_load_kwargs(CloudConfig())
    for name in ('attention.py','generation.py','qualification.py','fixed_trajectories.py','kernel_cases.py'):
        assert not (ROOT/'train/cloud'/name).exists(), name


def test_fixed_upstream_compiler_is_required_without_workaround_extras():
    project = tomllib.loads((ROOT/'train/cloud/pyproject.toml').read_text())
    deps = project['project']['dependencies']
    assert 'torch==2.12.1' in deps
    assert 'triton==3.7.1' in deps
    assert 'flash-linear-attention==0.5.2' in deps
    assert 'nvidia-ml-py==13.580.82' in deps
    assert set(project['project']['optional-dependencies']) == {'metrics'}
    assert not any('tilelang' in dep or 'causal-conv1d' in dep for dep in deps)
    # PyPI's torch 2.12.1 is a CUDA 13 build; an H200 host with an older driver saw no GPU
    # (Attoloop/6ad46d435ccb47ba0f7c22a4). The CUDA 12.9 build runs on CUDA 12 drivers.
    assert project['tool']['uv']['sources']['torch'] == {'index': 'pytorch-cu129'}
    lock = (ROOT/'train/cloud/uv.lock').read_text()
    assert 'version = "2.12.1+cu129"' in lock
    assert 'name = "nvidia-cuda-runtime"\n' not in lock and '-cu13' not in lock


@requires_smoke_base_eval
def test_stock_trl_accepts_text_lora_and_existing_environment_factory(tmp_path):
    from functools import partial
    from unittest.mock import patch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from trl import GRPOConfig, GRPOTrainer
    from trl.chat_template_utils import add_response_schema
    from train.cloud.tests.checkout import cached_tokenizer
    from train.cloud.training import LORA_TARGET, rollout_diagnostics
    from train.permit_env import PermitTrainingEnv
    from permit_openenv.tasks import load_pack

    tokenizer = add_response_schema(cached_tokenizer())
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1, head_dim=16,
        vocab_size=len(tokenizer), layer_types=['full_attention'], partial_rotary_factor=.5))
    model = get_peft_model(model, LoraConfig(r=16,lora_alpha=32,lora_dropout=0,
                                            target_modules=LORA_TARGET,task_type='CAUSAL_LM'))
    opts = training_options(CloudConfig(), 'grpo', tmp_path, smoke=True)
    opts.update(use_cpu=True, bf16=False, optim='adamw_torch')
    records = []
    trainer = GRPOTrainer(model=model,processing_class=tokenizer,args=GRPOConfig(**opts, reward_weights=[0.0]),
        reward_funcs=partial(rollout_diagnostics,config=CloudConfig(),tokenizer=tokenizer),
        train_dataset=Dataset.from_list([{'prompt':[{'role':'user','content':'Test'}],
                                         'split':'train','task_id':'074897d0f159-s47'}]),
        environment_factory=partial(PermitTrainingEnv,pack=load_pack(ROOT/'envs/permit/tasks'),
                                    trace_sink=records.append))
    assert type(trainer) is GRPOTrainer
    assert trainer.chat_template is None
    assert trainer.generation_config.max_new_tokens == CloudConfig().eval_max_tokens
    assert trainer.generation_kwargs['max_new_tokens'] == CloudConfig().eval_max_tokens
    from transformers import GenerationConfig
    import json
    cfg = CloudConfig()
    base = ROOT/'train/cloud/data/smoke-result-grade/smoke/base-eval'
    sampling = json.loads((base/'sampling.json').read_text())
    assert all(json.loads(path.read_text())['model_revision'] == cfg.revision for path in base.rglob('episode.json'))
    pinned = GenerationConfig.from_dict(sampling['generation_config'])
    for key in ('top_p', 'top_k', 'temperature'):
        assert opts[key] == getattr(pinned, key) == getattr(trainer.generation_config, key)
    assert {tool.__name__ for tool in trainer.tools} == {
        'get_scenario','check_design','evaluate_design','submit_design'}
    assert trainer.reward_weights.tolist() == [0.0, 1.0]
    # Exercise installed TRL reward dispatch and loss masks with truncated scripted responses.
    n = CloudConfig().num_generations
    responses = [[{'role':'assistant','content':'unfinished'}] for _ in range(n)]
    generated = ([[1,2]]*n, [[7,8]]*n, [[1,1]]*n, responses, None, {}, None, [[]]*n)
    inputs = [{'prompt':[{'role':'user','content':'Test'}], 'split':'train',
               'task_id':'074897d0f159-s47'} for _ in range(n)]
    with patch.object(trainer, '_generate', return_value=generated):
        batch = trainer._generate_and_score_completions(inputs)
    assert batch['completion_mask'].sum().item() == 2*n
    assert batch['tool_mask'].sum().item() == 2*n
    assert len(records) == n
    # The hook's column is None for every rollout, so the environment's reward is the only score and a
    # rollout whose grading failed (environment NaN) stays unscorable, as without the hook.
    import torch
    trainer.environments[0]._grading_failed = True
    scored = trainer._calculate_rewards(inputs, [x['prompt'] for x in inputs], responses, [[7,8]]*n)
    assert torch.isnan(scored[:, 0]).all()
    assert torch.isnan(scored[0, 1]) and scored[1:, 1].tolist() == [0.0]*(n-1)
    assert all(r['ended'] == 'budget' and r['completion_tokens'] == 2 for r in records)


def test_smoke_caps_cannot_be_increased():
    import pytest
    from train.cloud.config import stage_plan
    cfg = CloudConfig()
    for caps in ({'wall_caps':{'smoke':10801}}, {'usd_caps':{'smoke':15.01}}):
        with pytest.raises(ValueError, match='Smoke caps'):
            stage_plan(cfg, **caps)
    plan = stage_plan(cfg)[0]
    assert plan['effective_seconds'] <= 10200 and plan['usd_cap'] <= 15
