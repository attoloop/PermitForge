"""Cloud text-backbone bf16/NF4 LoRA and save/reload qualification."""
from functools import partial
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from .config import CloudConfig
from .formatting import encode_row, load_tokenizer
from .run import write_json


LORA_TARGET = r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)'


def validate_loading_info(info: dict) -> None:
    unexpected = [key for key in info.get('unexpected_keys', [])
                  if not key.startswith(('model.visual.', 'mtp.'))]
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs') or unexpected:
        raise ValueError('Unmatched language weights: '+str(info))


def model_load_kwargs(config: CloudConfig) -> dict:
    import torch
    from transformers import BitsAndBytesConfig
    options = dict(dtype=torch.bfloat16, device_map={'':0}, attn_implementation='sdpa')
    if config.precision == 'nf4':
        options['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    return options


def load_model(config: CloudConfig, adapter: Path | None = None, *, trainable: bool = False):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, Qwen3_5ForCausalLM, set_seed
    set_seed(config.seed)
    original = AutoConfig.from_pretrained(config.model, revision=config.revision)
    if original.model_type != 'qwen3_5':
        raise ValueError('Unexpected checkpoint architecture')
    from .kernel_cases import check_model_geometry
    check_model_geometry(original.text_config)
    # Transformers 5.17's qwen3_5_text PrefixChange removes language_model.
    # CPU test compares every tensor with a tiny conditional-generation checkpoint.
    model, info = Qwen3_5ForCausalLM.from_pretrained(config.model, revision=config.revision,
        config=original.text_config, output_loading_info=True, **model_load_kwargs(config))
    validate_loading_info(info)
    if trainable and config.precision == 'nf4':
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={'use_reentrant':False})
        # k-bit preparation casts all nonquantized weights to fp32; retain bf16 large tables.
        model.get_input_embeddings().to(torch.bfloat16)
        model.get_output_embeddings().to(torch.bfloat16)
    elif trainable:
        model.requires_grad_(False)
        model.enable_input_require_grads()
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        model = get_peft_model(model, LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha,
            lora_dropout=0, target_modules=LORA_TARGET, task_type='CAUSAL_LM'))
    if trainable and any('visual' in n for n,p in model.named_parameters() if p.requires_grad):
        raise ValueError('Vision weights must never be trained')
    model.config.use_cache = not trainable
    return model, load_tokenizer()


def training_options(config: CloudConfig, kind: str, output: Path, *, smoke: bool = False) -> dict:
    common = dict(output_dir=str(output), max_steps=2 if smoke else getattr(config, kind+'_steps'),
        per_device_train_batch_size=1, gradient_accumulation_steps=config.accum,
        learning_rate=3e-5, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False}, optim='paged_adamw_8bit',
        logging_steps=1, save_strategy='steps', save_steps=1 if smoke else 10,
        save_total_limit=2, report_to='none', push_to_hub=False, seed=config.seed,
        dataloader_num_workers=0)
    if kind == 'sft':
        return {**common, 'max_length':config.max_length, 'packing':False, 'assistant_only_loss':False}
    if config.trackio_space:
        common.update(report_to='trackio', project='permitforge-qwen38', run_name=output.name,
            trackio_space_id=config.trackio_space, trackio_bucket_id=config.trackio_bucket,
            trackio_static_space_id=False)
    return {**common, 'num_generations':config.num_generations,
        'generation_batch_size':config.num_generations,
        'max_tool_calling_iterations':config.train_turns,
        'max_completion_length':config.max_completion_length, 'scale_rewards':'group',
        'beta':0, 'num_iterations':1, 'mask_truncated_completions':True,
        'chat_template_kwargs':{'enable_thinking':False}, 'use_vllm':False,
        'generation_kwargs':{'use_cache':True, 'logits_to_keep':1}}


def latest_checkpoint(output: Path) -> Path | None:
    candidates = [p for p in output.glob('checkpoint-*') if (p/'trainer_state.json').exists()
                  and (p/'optimizer.pt').exists() and (p/'adapter_model.safetensors').exists()
                  and (p/'scheduler.pt').exists() and (p/'rng_state.pth').exists()
                  and (p/'.complete').exists()]
    return max(candidates, key=lambda p:int(p.name.split('-')[-1]), default=None)


def stop_callback(step: int):
    from transformers import TrainerCallback
    class StopAtStep(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= step:
                control.should_save = True
                control.should_training_stop = True
    return StopAtStep()


def train(config: CloudConfig, kind: str, output: Path, *, data: Path, pack,
          tasks: list, fields_loader, adapter: Path | None = None, smoke: bool = False,
          stop_after_step: int | None = None, qualification_run: Path | None = None) -> dict:
    import torch
    from datasets import Dataset
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file, save_file
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
    from trl.chat_template_utils import add_response_schema
    from train.memory import memory_trainer_class
    from train.permit_env import PermitTrainingEnv, scenario_text, training_instructions, tool_schemas
    from train.runtime import reward_groups
    from .make_sft import validate_catalog

    output.mkdir(parents=True, exist_ok=True)
    if (output/'metrics.json').exists():
        previous = json.loads((output/'metrics.json').read_text())
        if (previous.get('status') == 'passed' and
                previous.get('optimizer_steps',0) >= (stop_after_step or (2 if smoke else getattr(config,kind+'_steps')))):
            return previous
    checkpoint = latest_checkpoint(output)
    model, tokenizer = load_model(config, checkpoint or adapter, trainable=True)
    options = training_options(config, kind, output, smoke=smoke)
    write_json(output/'training_options.json', options)
    completed_step = json.loads((checkpoint/'trainer_state.json').read_text())['global_step'] if checkpoint else 0
    traces = []
    if checkpoint and (output/'episodes.jsonl').exists():
        for line in (output/'episodes.jsonl').read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get('optimizer_step',completed_step) < completed_step:
                traces.append(record)
    trace = (output/'episodes.jsonl').open('a')
    if kind == 'sft':
        tokenizer.padding_side = 'right'
        rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
        validate_catalog([dict(task_id=t.task_id,split=t.split,station=t.record['station']) for t in pack.tasks])
        tokens, excluded = [], []
        for row in rows:
            task = pack.get(row['task_id'])
            if (task.split != 'train' or row['split'] != 'train' or
                    row['station'] != task.record['station'] or task.record.get('eligible') is not True):
                raise ValueError('SFT data has task/station leakage')
            encoded = encode_row(tokenizer, row)
            if len(encoded['input_ids']) > config.max_length:
                excluded.append(dict(task_id=task.task_id, tokens=len(encoded['input_ids'])))
            else:
                tokens.append(encoded)
        write_json(output/'data_lengths.json', dict(accepted=len(tokens), too_long=excluded))
        if not tokens:
            raise ValueError('No complete SFT episodes fit the context')
        trainer = SFTTrainer(model=model, processing_class=tokenizer, args=SFTConfig(**options),
                             train_dataset=Dataset.from_list(tokens))
    else:
        tokenizer.padding_side = 'left'
        tokenizer = add_response_schema(tokenizer)
        rows = []
        for task in tasks:
            messages = [{'role':'system','content':training_instructions(config.train_turns)},
                        {'role':'user','content':'Optimize the plant and submit your design.\n'}]
            expanded = [messages[0], {'role':'user','content':messages[1]['content']+scenario_text(task)}]
            ids = tokenizer.apply_chat_template(expanded,tools=tool_schemas(),enable_thinking=False,
                                                add_generation_prompt=True,return_dict=False)
            if len(ids) > config.max_prompt_length:
                raise ValueError(task.task_id+': prompt exceeds cloud limit')
            rows.append(dict(prompt=messages, split='train',task_id=task.task_id))
        def capture(record):
            record = {**record, 'optimizer_step':trainer.state.global_step}
            traces.append(record)
            trace.write(json.dumps(record, allow_nan=False)+'\n')
            trace.flush()
        # The local chunked loss is Qwen3-specific and deliberately disabled here.
        cls = memory_trainer_class(GRPOTrainer, SimpleNamespace(generation_microbatch_size=1,
                                    empty_cache_after_generation=True, loss_chunk_size=0))
        from .generation import cached_trainer_class
        cls = cached_trainer_class(cls,microbatch=1,persistent=False)
        trainer = cls(model=model, processing_class=tokenizer, args=GRPOConfig(**options),
            train_dataset=Dataset.from_list(rows), environment_factory=partial(PermitTrainingEnv,
                pack=pack, fields_loader=fields_loader, trace_sink=capture))
        if trainer.chat_template is not None:
            raise ValueError('TRL attempted to replace the qualified Qwen inference template')

        from .qualification import apply_configuration, load_selection, qualify_stack
        try:
            if smoke:
                qualify_stack(trainer,tasks,config,output)
            elif qualification_run is not None:
                selected = load_selection(qualification_run/'smoke/grpo/qualification.json')
                apply_configuration(trainer,selected)
            else:
                raise ValueError('Cloud GRPO requires a qualified stack from the smoke')
        except BaseException:
            trace.close()
            raise
        write_json(output/'selected-stack.json',trainer.stack_settings)

    initial_path = output/'initial-adapter.safetensors'
    if not initial_path.exists():
        save_file({k:v.detach().cpu().contiguous().clone() for k,v in get_peft_model_state_dict(model).items()}, str(initial_path))
    step_seconds = []
    class Timer(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            (output/f'checkpoint-{state.global_step}'/'.complete').write_text('complete\n')
        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.start = time.monotonic()
        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            step_seconds.append(time.monotonic()-self.start)
            if torch.cuda.max_memory_reserved() > config.max_vram_bytes:
                raise RuntimeError('VRAM exceeds configured device cap')
    trainer.add_callback(Timer())
    if stop_after_step:
        trainer.add_callback(stop_callback(stop_after_step))
    started = time.monotonic()
    metrics = dict(status='failed', kind=kind, smoke=smoke, resume_checkpoint=str(checkpoint) if checkpoint else None)
    try:
        if checkpoint and completed_step >= options['max_steps']:
            from transformers.trainer_callback import TrainerState
            trainer.state = TrainerState.load_from_json(str(checkpoint/'trainer_state.json'))
        else:
            trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        trainer.save_model(str(output/'adapter'))
        tokenizer.save_pretrained(output/'adapter')
        trainer.save_state()
        trained = {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
        initial = load_file(str(initial_path))
        if not all(torch.isfinite(v).all() for v in trained.values()):
            raise RuntimeError('Nonfinite adapter weights')
        metrics['weights_changed'] = any(not torch.equal(initial[k],v) for k,v in trained.items())
        model.load_adapter(output/'adapter', adapter_name='reload_check', is_trainable=False)
        loaded = get_peft_model_state_dict(model, adapter_name='reload_check')
        metrics['adapter_reload_equal'] = set(trained)==set(loaded) and all(
            torch.equal(v, loaded[k].detach().cpu()) for k,v in trained.items())
        model.delete_adapter('reload_check')
        if not metrics['weights_changed'] or not metrics['adapter_reload_equal']:
            raise RuntimeError('Smoke qualification failed: changed/reloaded weights')
        if smoke and trainer.state.global_step != 2:
            raise RuntimeError('Smoke requires exactly two optimizer steps')
        if any(r.get('grading_failed') for r in traces):
            raise RuntimeError('GRPO grading unavailable')
        if any(not math.isfinite(row[key]) for row in trainer.state.log_history
               for key in ('loss','grad_norm') if isinstance(row.get(key),(int,float))):
            raise RuntimeError('Nonfinite training loss/gradient norm')
        metrics['status'] = 'paused' if trainer.state.global_step < options['max_steps'] else 'passed'
    finally:
        trace.close()
        metrics.update(optimizer_steps=trainer.state.global_step, step_seconds=step_seconds,
            seconds_per_step=statistics.mean(step_seconds) if step_seconds else None,
            elapsed_seconds=time.monotonic()-started, peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(), groups=reward_groups(traces,config.num_generations),
            tasks_seen=sorted({r['task_id'] for r in traces}))
        if kind == 'grpo':
            metrics['stack'] = trainer.stack_settings
            metrics['generation_turns'] = trainer.generation_timings
            write_json(output/'generation-timings.json',trainer.generation_timings)
        write_json(output/'metrics.json', metrics)
        # NaN reward diagnostics are retained as null in JSON.
        logs = json.loads(json.dumps(trainer.state.log_history).replace('NaN','null').replace('Infinity','null'))
        write_json(output/'trainer_metrics.json', {'log_history':logs})
    return metrics
