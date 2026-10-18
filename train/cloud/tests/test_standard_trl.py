"""WP2.15: upstream trainer owns generation, kernels and tool-loop integration."""
import ast
from pathlib import Path
import tomllib

from train.cloud.config import CloudConfig
from train.cloud.training import model_load_kwargs, training_options

ROOT = Path(__file__).resolve().parents[3]


def test_standard_grpo_trainer_owns_generation_and_loss():
    tree = ast.parse((ROOT/'train/cloud/training.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    constructors = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'GRPOTrainer']
    assert len(constructors) == 1
    assert 'environment_factory' in {k.arg for k in constructors[0].keywords}
    assert not any(isinstance(n.func, ast.Name) and n.func.id in (
        'memory_trainer_class', 'cached_trainer_class', 'qualify_stack') for n in calls)
    opts = training_options(CloudConfig(), 'grpo', Path('unused'), smoke=True)
    assert opts['use_vllm'] is False
    assert opts['max_steps'] == 2
    assert opts['num_generations'] == opts['gradient_accumulation_steps'] == 4
    assert opts['max_tool_calling_iterations'] == 8
    assert opts['beta'] == 0 and opts['scale_rewards'] == 'group'
    assert opts['mask_truncated_completions'] is True
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


def test_stock_trl_accepts_text_lora_and_existing_environment_factory(tmp_path):
    from functools import partial
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from trl import GRPOConfig, GRPOTrainer
    from trl.chat_template_utils import add_response_schema
    from train.cloud.formatting import load_tokenizer
    from train.cloud.training import LORA_TARGET
    from train.permit_env import PermitTrainingEnv
    from permit_openenv.tasks import load_pack

    tokenizer = add_response_schema(load_tokenizer(local_files_only=True))
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1, head_dim=16,
        vocab_size=len(tokenizer), layer_types=['full_attention'], partial_rotary_factor=.5))
    model = get_peft_model(model, LoraConfig(r=16,lora_alpha=32,lora_dropout=0,
                                            target_modules=LORA_TARGET,task_type='CAUSAL_LM'))
    opts = training_options(CloudConfig(), 'grpo', tmp_path, smoke=True)
    opts.update(use_cpu=True, bf16=False, optim='adamw_torch')
    trainer = GRPOTrainer(model=model,processing_class=tokenizer,args=GRPOConfig(**opts),
        train_dataset=Dataset.from_list([{'prompt':[{'role':'user','content':'Test'}],
                                         'split':'train','task_id':'074897d0f159-s47'}]),
        environment_factory=partial(PermitTrainingEnv,pack=load_pack(ROOT/'envs/permit/tasks')))
    assert type(trainer) is GRPOTrainer
    assert trainer.chat_template is None
    assert {tool.__name__ for tool in trainer.tools} == {
        'get_scenario','check_design','evaluate_design','submit_design'}


def test_smoke_caps_cannot_be_increased():
    import pytest
    from train.cloud.config import stage_plan
    cfg = CloudConfig()
    for caps in ({'wall_caps':{'smoke':10801}}, {'usd_caps':{'smoke':15.01}}):
        with pytest.raises(ValueError, match='Smoke caps'):
            stage_plan(cfg, **caps)
    plan = stage_plan(cfg)[0]
    assert plan['effective_seconds'] <= 10200 and plan['usd_cap'] <= 15
