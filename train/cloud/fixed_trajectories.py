"""Deterministic diagnostic tool traces and actual TRL loss/update comparisons."""
from copy import deepcopy
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import time
from types import SimpleNamespace

from .formatting import encode_row
from .kernel_cases import compare


@contextmanager
def training_forward(model):
    """Match Accelerator's bf16 forward/fp32 output wrapper before Trainer.prepare."""
    import torch
    from accelerate.utils import convert_to_fp32
    original = model.forward
    @wraps(original)
    def forward(*args,**kwargs):
        with torch.autocast(device_type=model.device.type,dtype=torch.bfloat16,
                            enabled=model.device.type == 'cuda'):
            return convert_to_fp32(original(*args,**kwargs))
    model.forward = forward
    try:
        yield
    finally:
        model.forward = original


def fixtures(trainer, tasks: list, config) -> dict:
    from train.permit_env import PermitTrainingEnv, scenario_text, training_instructions, tool_schemas
    tokenizer = trainer.processing_class
    rows = []
    for i in range(config.num_generations):
        task = tasks[i % len(tasks)]
        messages = [{'role':'system','content':training_instructions(config.train_turns)},
                    {'role':'user','content':'Optimize the plant and submit your design.\n'+scenario_text(task)}]
        prompt = tokenizer.apply_chat_template(messages,tools=tool_schemas(),enable_thinking=False,
                                               add_generation_prompt=True,return_dict=False)
        # Diagnostic fixed actions, not training examples or synthetic rewards for the real rollout.
        draft = {sid:dict(control=row['control'],height_m=float(row['stack_height_m']))
                 for sid,row in task.scenario.draft_design.items()}
        call = dict(type='function',function=dict(name='check_design',arguments=dict(design=draft)))
        messages.append(dict(role='assistant',content='',tool_calls=[call]))
        first = encode_row(tokenizer,dict(messages=messages,tools=tool_schemas()))['input_ids'][len(prompt):]
        # Use the existing public validation/cost path without fields or a native model launch.
        env = PermitTrainingEnv(pack=SimpleNamespace(get=lambda _: task),fields_loader=None)
        env.reset(task_id=task.task_id)
        response = env.check_design(draft)
        messages.append(dict(role='tool',name='check_design',content=str(response)))
        turn = tokenizer.apply_chat_template(messages,tools=tool_schemas(),enable_thinking=False,
                                             add_generation_prompt=True,return_dict=False)
        call = dict(type='function',function=dict(name='submit_design',arguments=dict(design=draft)))
        messages.append(dict(role='assistant',content='',tool_calls=[call]))
        encoded = encode_row(tokenizer,dict(messages=messages,tools=tool_schemas()))
        if encoded['input_ids'][:len(prompt)] != prompt or turn[:len(prompt)] != prompt:
            raise RuntimeError('Fixed trajectory template is not prefix preserving')
        if len(encoded['input_ids']) > config.max_length:
            raise RuntimeError('Fixed diagnostic trace exceeds unchanged context')
        second = encoded['input_ids'][len(turn):]
        forced = [seq[:seq.index(tokenizer.eos_token_id)+1] for seq in (first,second)]
        rows.append(dict(task_id=task.task_id,prompt_ids=prompt,turns=[prompt,turn],forced=forced,
            completion_ids=encoded['input_ids'][len(prompt):],
            tool_mask=encoded['assistant_masks'][len(prompt):],truncated=i == config.num_generations-1))
    result = dict(rows=rows,advantage_fixture='group-scaled [0, ..., n-1]; diagnostic only',
                  note='Fixed check-draft / submit-draft tokens; no native submission and no training signal.')
    result['sha256'] = hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result


def loss_inputs(fixed: dict, trainer) -> list[dict]:
    import torch
    device = trainer.accelerator.device
    rewards = torch.arange(len(fixed['rows']),dtype=torch.float32,device=device)
    advantages = (rewards-rewards.mean())/(rewards.std()+1e-4)
    rows = []
    count = sum(sum(r['tool_mask']) for r in fixed['rows'] if not r['truncated'])
    for i,row in enumerate(fixed['rows']):
        def ids(name):
            return torch.tensor([row[name]],device=device)
        completion = ids('completion_ids')
        rows.append(dict(prompt_ids=ids('prompt_ids'),prompt_mask=torch.ones_like(ids('prompt_ids')),
            completion_ids=completion,completion_mask=torch.full_like(completion,int(not row['truncated'])),
            tool_mask=ids('tool_mask'),advantages=advantages[i:i+1],
            num_items_in_batch=torch.tensor(float(count),device=device)))
    return rows


def enable_liger(trainer, enabled: bool) -> None:
    if enabled and not hasattr(trainer,'liger_loss'):
        from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
        from trl.models.utils import _ForwardRedirection
        args = trainer.args
        trainer._forward_redirection = _ForwardRedirection()
        trainer.liger_loss = LigerFusedLinearGRPOLoss(beta=trainer.beta,
            epsilon_low=trainer.epsilon_low,epsilon_high=trainer.epsilon_high,temperature=trainer.temperature,
            use_ref_model=trainer.beta != 0.,loss_type=trainer.loss_type,
            max_completion_length=trainer.max_completion_length,
            importance_sampling_level=trainer.importance_sampling_level,
            **{k:getattr(args,k) for k in ('delta','use_bias_correction_kl','sapo_temperature_pos',
                'sapo_temperature_neg','vespo_k_pos','vespo_lambda_pos','vespo_k_neg','vespo_lambda_neg')})
    trainer.use_liger_kernel = enabled


def snapshot(model) -> dict:
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def restore(model, weights: dict) -> None:
    import torch
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name in weights:
                p.copy_(weights[name])
    model.zero_grad(set_to_none=True)


def loss_step(trainer, inputs: list[dict], initial: dict, *, capture: bool = True) -> dict:
    """Exercise the same GRPO loss, accumulation, clipping and optimizer as training."""
    import torch
    model = trainer.model
    restore(model,initial)
    trainer.optimizer = None
    trainer.create_optimizer()
    trainer.current_gradient_accumulation_steps = trainer.args.gradient_accumulation_steps
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    logps = []
    if capture:
        with torch.no_grad(), torch.autocast(device_type=model.device.type,dtype=torch.bfloat16,
                                            enabled=model.device.type == 'cuda'):
            for row in inputs:
                ids = torch.cat([row['prompt_ids'],row['completion_ids']],dim=1)
                mask = torch.cat([row['prompt_mask'],row['completion_mask']],dim=1)
                values,_,_ = trainer._get_per_token_logps_and_entropies(model,ids,mask,
                    row['completion_ids'].shape[1],compute_entropy=False)
                logps.append(values.detach().cpu())
    torch.cuda.synchronize() if model.device.type == 'cuda' else None
    start = time.monotonic()
    losses = []
    for row in inputs:
        with torch.autocast(device_type=model.device.type,dtype=torch.bfloat16,
                            enabled=model.device.type == 'cuda'):
            loss = trainer.compute_loss(model,row)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite diagnostic GRPO loss')
        losses.append(float(loss.detach()))
        loss.backward()
    gradients = {}
    for name,p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError('Missing/nonfinite adapter gradient: '+name)
            if capture:
                gradients[name] = p.grad.detach().cpu().clone()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],trainer.args.max_grad_norm)
    trainer.optimizer.step()
    torch.cuda.synchronize() if model.device.type == 'cuda' else None
    elapsed = time.monotonic()-start
    updated = snapshot(model) if capture else {}
    trainer.optimizer = None
    model.zero_grad(set_to_none=True)
    return dict(losses=torch.tensor(losses),logps=logps,gradients=gradients,updated=updated,seconds=elapsed)


def compare_steps(candidate: dict, baseline: dict, evidence: dict, initial: dict) -> None:
    compare('loss',candidate['losses'],baseline['losses'],evidence)
    for i,(a,b) in enumerate(zip(candidate['logps'],baseline['logps'],strict=True)):
        compare(f'logps/{i}',a,b,evidence)
    for key in ('gradients','updated'):
        if candidate[key].keys() != baseline[key].keys():
            raise RuntimeError('Adapter parameter coverage differs')
        for name in baseline[key]:
            a,b = candidate[key][name],baseline[key][name]
            # Compare update deltas too: large starting weights must not hide a wrong step.
            if key == 'updated':
                a,b = a.float()-initial[name].float(),b.float()-initial[name].float()
            compare(key+'/'+name,a,b,evidence)


def nonzero_adapter(initial: dict) -> dict:
    import torch
    result = deepcopy(initial)
    rng = torch.Generator().manual_seed(214)
    for name,tensor in result.items():
        if 'lora_B' in name:
            tensor.add_(.001*torch.randn(tensor.shape,generator=rng,dtype=tensor.dtype))
    return result
