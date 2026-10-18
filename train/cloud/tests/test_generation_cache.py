import inspect

import pytest
import torch


@pytest.fixture
def hybrid():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    from train.cloud.kernels import NAMES
    for name in NAMES:
        setattr(qwen,name,inspect.unwrap(getattr(qwen,name)))
    torch.manual_seed(214)
    cfg = Qwen3_5TextConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=16,vocab_size=128,
        linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=8,
        linear_value_head_dim=8,linear_conv_kernel_dim=4,
        layer_types=['linear_attention','full_attention'],partial_rotary_factor=.5,
        pad_token_id=0,eos_token_id=127)
    cfg._attn_implementation = 'sdpa'
    return Qwen3_5ForCausalLM(cfg).eval()


def test_continued_hybrid_state_matches_full_history_batched_and_independent(hybrid):
    from train.cloud.generation import HybridGenerator
    generator = HybridGenerator(hybrid, pad_token_id=0, microbatch=2, persistent=True)
    prompts = [[4,5,6,7,8], [4,9,10]]
    with torch.no_grad():
        first = generator.teacher_force(prompts)
        for i,prompt in enumerate(prompts):
            ref = hybrid(torch.tensor([prompt]), use_cache=True)
            torch.testing.assert_close(first[i],ref.logits[0,-1],atol=1e-5,rtol=1e-5)
        longer = [prompts[0]+list(range(11,78)),prompts[1]+[31,32,33]]
        result = generator.teacher_force(longer)
        assert generator.stats['reused_tokens'] >= sum(map(len,prompts))
        for i,prompt in enumerate(longer):
            ref = hybrid(torch.tensor([prompt]), use_cache=True)
            torch.testing.assert_close(result[i],ref.logits[0,-1],atol=1e-5,rtol=1e-5)
            generator.assert_state(i,ref.past_key_values,atol=1e-5,rtol=1e-5)


def test_policy_updates_and_prefix_edits_clear_all_hybrid_state(hybrid):
    from train.cloud.generation import HybridGenerator
    generator = HybridGenerator(hybrid,pad_token_id=0,microbatch=2,persistent=True)
    generator.teacher_force([[4,5,6],[4,5,6]])
    generator.invalidate('optimizer_step')
    assert not generator.entries
    generator.teacher_force([[4,5,6]])
    before = generator.stats['reused_tokens']
    generator.teacher_force([[4,55,6,7]])
    assert generator.stats['reused_tokens'] == before
    assert len(generator.entries) == 1


def test_generate_retains_unconsumed_final_token_and_independent_siblings(hybrid):
    from transformers import GenerationConfig
    from train.cloud.generation import HybridGenerator
    cfg = GenerationConfig(max_new_tokens=3,do_sample=False,pad_token_id=0,eos_token_id=127,
                           use_cache=True)
    generator = HybridGenerator(hybrid,pad_token_id=0,microbatch=2,persistent=True)
    prompts = [[4,5,6],[4,5,6]]
    out = generator.generate(prompts,cfg)
    for entry, prompt, tokens in zip(generator.entries,prompts,out):
        assert entry.tokens == prompt+tokens
        assert entry.consumed == len(entry.tokens)-1
    # Identical prefixes must own distinct mutable recurrent/conv/KV tensors.
    a,b = generator.entries
    assert a.cache.layers[0].recurrent_states[0].data_ptr() != b.cache.layers[0].recurrent_states[0].data_ptr()
    continued = [prompts[1]+out[1]+[30,31]]  # First sibling has finished.
    cached = generator.generate(continued,cfg)
    independent = HybridGenerator(hybrid,pad_token_id=0,microbatch=1,persistent=False).generate(continued,cfg)
    assert cached == independent
    assert len(generator.entries) == 1


def test_eos_removal_and_unequal_lengths_preserve_each_rollout_state(hybrid):
    from transformers import GenerationConfig
    from train.cloud.generation import HybridGenerator
    cfg = GenerationConfig(max_new_tokens=5,do_sample=False,pad_token_id=0,eos_token_id=127,use_cache=True)
    generator = HybridGenerator(hybrid,pad_token_id=0,microbatch=2,persistent=True)
    prompts = [[4,5,6,7,8],[8,7,6]]
    forced = [[21,127],[22,23,24,127]]
    assert generator.generate(prompts,cfg,forced_tokens=forced) == forced
    continuation = [prompt+tokens+[33,34] for prompt,tokens in zip(prompts,forced)]
    out = generator.teacher_force(continuation)
    with torch.no_grad():
        for i,prompt in enumerate(continuation):
            ref = hybrid(torch.tensor([prompt]),use_cache=True)
            torch.testing.assert_close(out[i],ref.logits[0,-1],rtol=1e-5,atol=1e-5)
            generator.assert_state(i,ref.past_key_values,rtol=1e-5,atol=1e-5)


def test_trainer_invalidates_on_every_optimizer_step_and_failed_rollout_group():
    from types import SimpleNamespace
    from train.cloud.generation import cached_trainer_class
    class Base:
        def __init__(self):
            self.callbacks = []
        def add_callback(self, callback):
            self.callbacks.append(callback)
        def _generate(self,prompts):
            raise RuntimeError('interrupted tool loop')
    trainer = cached_trainer_class(Base,microbatch=2,persistent=True)()
    reasons = []
    trainer.hybrid_generator = SimpleNamespace(invalidate=reasons.append)
    for step in (1,2,3):
        trainer.callbacks[0].on_step_end(None,SimpleNamespace(global_step=step),None)
    assert reasons == ['optimizer_step']*3
    with pytest.raises(RuntimeError,match='interrupted'):
        trainer._generate([])
    assert reasons[-2:] == ['rollout_group_start','rollout_group_end']


def test_cache_after_actual_weight_update_and_reload_matches_fresh_policy(hybrid):
    from train.cloud.generation import HybridGenerator
    engine = HybridGenerator(hybrid,pad_token_id=0,microbatch=2,persistent=True)
    prompts = [[4,5,6],[7,8,9]]
    before = engine.teacher_force(prompts)
    state = {k:v.detach().clone() for k,v in hybrid.state_dict().items()}
    with torch.no_grad():
        hybrid.model.layers[1].self_attn.q_proj.weight.add_(.25)
    engine.invalidate('optimizer_step')
    after = engine.teacher_force([p+[10] for p in prompts])
    with torch.no_grad():
        for i,p in enumerate(prompts):
            reference = hybrid(torch.tensor([p+[10]]),use_cache=True)
            torch.testing.assert_close(after[i],reference.logits[0,-1],atol=1e-5,rtol=1e-5)
    hybrid.load_state_dict(state)
    engine.invalidate('checkpoint_load')
    reloaded = engine.teacher_force(prompts)
    for a,b in zip(before,reloaded):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_fixed_loss_probe_uses_trl_objective_and_restores_weights(hybrid,tmp_path):
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from trl import GRPOConfig, GRPOTrainer
    from train.cloud.fixed_trajectories import compare_steps,loss_step,restore,snapshot,enable_liger
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {str(i):i for i in range(128)},unk_token='1')),pad_token='0',eos_token='127',unk_token='1')
    hybrid.requires_grad_(False)
    hybrid.enable_input_require_grads()
    model = get_peft_model(hybrid,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],
                                           task_type='CAUSAL_LM'))
    trainer = GRPOTrainer(model=model,processing_class=tokenizer,
        args=GRPOConfig(output_dir=str(tmp_path),use_cpu=True,bf16=False,
            gradient_accumulation_steps=4,num_generations=4,generation_batch_size=4,
            per_device_train_batch_size=1,optim='adamw_torch',report_to='none',max_completion_length=8),
        reward_funcs=lambda completions,**kwargs:[0.]*len(completions),
        train_dataset=Dataset.from_list([dict(prompt='4 5 6')]))
    inputs = [dict(prompt_ids=torch.tensor([[4,5,6]]),prompt_mask=torch.ones(1,3,dtype=torch.long),
        completion_ids=torch.tensor([[8,9,10,127]]),completion_mask=torch.tensor([[1,1,1,1]]),
        tool_mask=torch.tensor([[1,0,1,1]]),advantages=torch.tensor([.5]),num_items_in_batch=torch.tensor(3.))]
    initial = snapshot(model)
    result = loss_step(trainer,inputs,initial)
    assert any(not torch.equal(result['updated'][k],v) for k,v in initial.items())
    repeated = loss_step(trainer,inputs,initial)
    compare_steps(repeated,result,{},initial)
    restore(model,initial)
    assert all(torch.equal(v,snapshot(model)[k]) for k,v in initial.items())
    try:
        enable_liger(trainer,True)
    except ImportError:
        pytest.skip('Optional Liger package absent in local training environment')
    assert trainer.use_liger_kernel and trainer.liger_loss is not None
    enable_liger(trainer,False)
    assert not trainer.use_liger_kernel


def test_diagnostic_forward_restores_original_even_on_failure(hybrid):
    from train.cloud.fixed_trajectories import training_forward
    original = hybrid.forward
    with pytest.raises(RuntimeError,match='failure'):
        with training_forward(hybrid):
            output = hybrid(torch.tensor([[4,5,6]]),use_cache=False)
            assert output.logits.dtype == torch.float32
            raise RuntimeError('failure')
    assert hybrid.forward == original
