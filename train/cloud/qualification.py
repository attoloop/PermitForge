"""Bounded H200 gates and fixed-trajectory timing before the real GRPO steps."""
from copy import deepcopy
import gc
import math
from pathlib import Path
import time

from .attention import FA3, qualify_attention, set_attention
from .fixed_trajectories import (compare_steps, enable_liger, fixtures, loss_inputs, loss_step,
                                 nonzero_adapter, restore, snapshot, training_forward)
from .generation import HybridGenerator, cache_tensors
from .kernel_cases import TOLERANCES, compare
from .run import write_json

BASELINE = dict(attention='sdpa',persistent=False,microbatch=1,liger=False)


def gate(name: str, action, report: dict, path: Path):
    row = report[name] = dict(status='running')
    write_json(path,report)
    started = time.monotonic()
    try:
        value = action()
        row.update(status='passed',seconds=time.monotonic()-started)
        return value
    except Exception as exc:
        row.update(status='failed',error=type(exc).__name__+': '+str(exc),seconds=time.monotonic()-started,
                   fallback='qualified HF / SDPA / existing GRPO loss')
        # Tracebacks can retain a failed large allocation until this block exits.
        return None
    finally:
        write_json(path,report)


def select_configuration(rows: list[dict]) -> dict:
    passing = [r for r in rows if r.get('status') == 'passed' and all(
        isinstance(r.get(k),(int,float)) and math.isfinite(r[k]) and r[k] > 0
        for k in ('generation_seconds','training_seconds'))]
    if not any(r['name'] == 'baseline' for r in passing):
        raise RuntimeError('No qualified, timed baseline; training is forbidden')
    return min(passing,key=lambda r:(r['generation_seconds']+r['training_seconds'],r['name']))


def apply_configuration(trainer, settings: dict) -> dict:
    evidence = set_attention(trainer.model,settings['attention'])
    enable_liger(trainer,settings['liger'])
    trainer.stack_settings = dict(settings)
    if getattr(trainer,'hybrid_generator',None) is not None:
        trainer.hybrid_generator.invalidate('configuration_change')
        trainer.hybrid_generator = None
    return evidence


def qualify_cache(model, fixed: dict, microbatch: int, evidence: dict) -> None:
    import torch
    model.eval()
    engine = HybridGenerator(model,pad_token_id=model.config.pad_token_id or 0,
                             microbatch=microbatch,persistent=True)
    evidence['comparisons'] = {}
    for turn in range(3):
        prompts = [row['turns'][turn] if turn < 2 else row['turns'][1]+row['forced'][1][:1]
                   for row in fixed['rows']]
        actual = engine.teacher_force(prompts)
        for i,prompt in enumerate(prompts):
            with torch.no_grad():
                ref = model(torch.tensor([prompt],device=model.device),use_cache=True,logits_to_keep=1)
            compare(f'{turn}/{i}/logits',actual[i],ref.logits[0,-1],evidence['comparisons'])
            a,b = cache_tensors(engine.entries[i].cache),cache_tensors(ref.past_key_values)
            if a.keys() != b.keys():
                raise RuntimeError('Hybrid state components differ')
            for key in a:
                compare(f'{turn}/{i}/{key}',a[key],b[key],evidence['comparisons'])
            del ref,a,b
    if engine.stats['reused_tokens'] <= 0:
        raise RuntimeError('Cache gate did not reuse a prefix')
    evidence['reused_tokens'] = engine.stats['reused_tokens']
    engine.invalidate('qualification_policy_update')
    if engine.entries:
        raise RuntimeError('Policy update left stale hybrid state')


def replay_generation(trainer, fixed: dict, *, microbatch: int, persistent: bool) -> dict:
    """Time HF sampling machinery on identical forced tokens, including tool-turn prefill."""
    import torch
    from transformers import LogitsProcessorList
    model = trainer.model
    model.eval()
    config = deepcopy(trainer.generation_config)
    engine = HybridGenerator(model,pad_token_id=trainer._tokenizer.pad_token_id,
                             microbatch=microbatch,persistent=True) if persistent else None
    turns = []
    for turn in range(2):
        prompts = [row['turns'][turn] for row in fixed['rows']]
        forced = [row['forced'][turn] for row in fixed['rows']]
        config.max_new_tokens = max(map(len,forced))
        torch.cuda.synchronize() if model.device.type == 'cuda' else None
        started = time.monotonic()
        if engine is not None:
            actual = engine.generate(prompts,config,forced_tokens=forced)
        else:
            actual = []
            # The baseline is the existing independent full-history HF generation path.
            for prompt,tokens in zip(prompts,forced,strict=True):
                ids = torch.tensor([prompt],device=model.device)
                def force(input_ids,scores,prompt=prompt,tokens=tokens):
                    scores.fill_(-torch.inf)
                    scores[:,tokens[input_ids.shape[1]-len(prompt)]] = 0
                    return scores
                with torch.no_grad():
                    out = model.generate(ids,attention_mask=torch.ones_like(ids),generation_config=config,
                        logits_processor=LogitsProcessorList([force]),use_cache=True,logits_to_keep=1,
                        max_new_tokens=len(tokens),pad_token_id=trainer._tokenizer.pad_token_id)
                actual.append(out[0,len(prompt):].tolist())
        if actual != forced:
            raise RuntimeError('Fixed-trajectory replay changed tokens or termination')
        torch.cuda.synchronize() if model.device.type == 'cuda' else None
        turns.append(dict(turn=turn,seconds=time.monotonic()-started,input_tokens=sum(map(len,prompts)),
                          output_tokens=sum(map(len,forced)),batch_size=len(prompts)))
    if engine is not None:
        stats = engine.stats
        engine.invalidate('replay_end')
    else:
        stats = None
    return dict(seconds=sum(t['seconds'] for t in turns),turns=turns,cache=stats)


def qualify_stack(trainer, tasks: list, config, output: Path) -> dict:
    import torch
    if trainer.accelerator.device.type != 'cuda':
        raise RuntimeError('Stack qualification requires the H200; CPU tests do not qualify kernels')
    path = output/'qualification.json'
    report = dict(status='running',tolerances=dict(TOLERANCES),gates={},candidates=[],
        timing_definition='Same fixed tool trajectories and tokens; warm generation + accumulated GRPO '
                          'forward/backward/clipped optimizer step. Tool/native time excluded and unchanged.',
        budget_seconds=2400)
    write_json(path,report)
    started = time.monotonic()
    fixed = fixtures(trainer,tasks,config)
    write_json(output/'qualification-trajectories.json',fixed)
    report['trajectory_sha256'] = fixed['sha256']
    inputs = loss_inputs(fixed,trainer)
    initial = snapshot(trainer.model)
    policies = {'initial':initial,'nonzero':nonzero_adapter(initial)}
    metrics = deepcopy(trainer._metrics)
    was_training = trainer.model.training
    reference = {}
    generations,losses = {},{}

    def checkpoint():
        write_json(path,report)
        gc.collect()
        torch.cuda.empty_cache()

    def remaining():
        return time.monotonic()-started < report['budget_seconds']

    # This context restores the training RNG after every diagnostic update/replay.
    with torch.random.fork_rng(devices=[trainer.accelerator.device]), training_forward(trainer.model):
        try:
            apply_configuration(trainer,BASELINE)
            for policy,weights in policies.items():
                reference[policy] = loss_step(trainer,inputs,weights)
            restore(trainer.model,policies['nonzero'])
            cold = replay_generation(trainer,fixed,microbatch=1,persistent=False)
            warm = replay_generation(trainer,fixed,microbatch=1,persistent=False)
            generations[('sdpa',False,1)] = warm
            # Second nonzero step is warm; first includes kernel/JIT setup.
            step = loss_step(trainer,inputs,policies['nonzero'])
            check = {}
            compare_steps(step,reference['nonzero'],check,policies['nonzero'])
            losses[('sdpa',False)] = step['seconds']
            report['baseline'] = dict(status='passed',cold_generation=cold,warm_generation=warm,
                cold_training_seconds=reference['nonzero']['seconds'],warm_training_seconds=step['seconds'],
                comparisons=check)
            del step
            checkpoint()

            # Each optional gate records failure before proceeding with a qualified alternative.
            attention_evidence = report['attention'] = {}
            fa3 = gate('fa3_kernel',lambda:qualify_attention(attention_evidence),report['gates'],
                       output/'qualification-gates.json')
            attention_options = ['sdpa']+([FA3] if fa3 is not None else [])
            for attention in attention_options:
                attention_key = 'sdpa' if attention == 'sdpa' else 'fa3'
                switched = gate(attention_key+'/dispatch',
                    lambda:apply_configuration(trainer,{**BASELINE,'attention':attention}),
                    report['gates'],output/'qualification-gates.json')
                if switched is None:
                    apply_configuration(trainer,BASELINE)
                    continue
                report.setdefault('attention_dispatch',{})[attention_key] = switched
                for liger in (False,True):
                    if attention == 'sdpa' and not liger:
                        continue
                    name = attention_key+('/liger' if liger else '/existing_loss')
                    details = report.setdefault('loss_gates',{})[name] = {}
                    def loss_gate():
                        if not remaining():
                            raise TimeoutError('Qualification time allowance exhausted')
                        enable_liger(trainer,liger)
                        for policy,weights in policies.items():
                            candidate = loss_step(trainer,inputs,weights)
                            details[policy] = {}
                            compare_steps(candidate,reference[policy],details[policy],weights)
                        cold_seconds = candidate['seconds']
                        del candidate
                        warm_step = loss_step(trainer,inputs,policies['nonzero'],capture=False)
                        details.update(cold_seconds=cold_seconds,warm_seconds=warm_step['seconds'])
                        return warm_step['seconds']
                    measured = gate(name,loss_gate,report['gates'],output/'qualification-gates.json')
                    if measured is not None:
                        losses[(attention,liger)] = measured
                    enable_liger(trainer,False)
                    trainer.optimizer = None
                    restore(trainer.model,policies['nonzero'])
                    checkpoint()
                # Reject attention if it fails the unchanged-loss model/gradient/update gate.
                if (attention,False) not in losses:
                    continue
                if attention != 'sdpa':
                    details = report.setdefault('generation_gates',{})[attention_key+'/baseline'] = {}
                    def full_generation():
                        if not remaining():
                            raise TimeoutError('Qualification time allowance exhausted')
                        details['cold'] = replay_generation(trainer,fixed,microbatch=1,persistent=False)
                        return replay_generation(trainer,fixed,microbatch=1,persistent=False)
                    value = gate(attention_key+'/baseline',full_generation,report['gates'],output/'qualification-gates.json')
                    if value is not None:
                        generations[(attention,False,1)] = value
                for batch in (1,2,4):
                    if batch > config.num_generations:
                        continue
                    name = attention_key+f'/cache_batch{batch}'
                    details = report.setdefault('generation_gates',{})[name] = {}
                    def generation_gate():
                        if not remaining():
                            raise TimeoutError('Qualification time allowance exhausted')
                        for policy,weights in policies.items():
                            restore(trainer.model,weights)
                            details[policy] = {}
                            qualify_cache(trainer.model,fixed,batch,details[policy])
                        details['cold'] = replay_generation(trainer,fixed,microbatch=batch,persistent=True)
                        return replay_generation(trainer,fixed,microbatch=batch,persistent=True)
                    value = gate(name,generation_gate,report['gates'],output/'qualification-gates.json')
                    if value is not None:
                        generations[(attention,True,batch)] = value
                    restore(trainer.model,policies['nonzero'])
                    checkpoint()

            for (attention,persistent,batch),generation in generations.items():
                for (loss_attention,liger),seconds in losses.items():
                    if loss_attention != attention:
                        continue
                    settings = dict(attention=attention,persistent=persistent,microbatch=batch,liger=liger)
                    name = 'baseline' if settings == BASELINE else (
                        f'{attention}/cache={persistent}/batch={batch}/liger={liger}')
                    candidate = dict(name=name,status='passed',settings=settings,
                        generation_seconds=generation['seconds'],training_seconds=seconds,
                        turns=generation['turns'],cache=generation['cache'])
                    if settings != BASELINE:
                        def benchmark():
                            if not remaining():
                                raise TimeoutError('Qualification time allowance exhausted')
                            apply_configuration(trainer,settings)
                            restore(trainer.model,policies['nonzero'])
                            generated = replay_generation(trainer,fixed,microbatch=batch,persistent=persistent)
                            trained = loss_step(trainer,inputs,policies['nonzero'],capture=False)
                            return dict(generation_seconds=generated['seconds'],training_seconds=trained['seconds'],
                                        turns=generated['turns'],cache=generated['cache'])
                        measured = gate('combined/'+name,benchmark,report['gates'],output/'qualification-gates.json')
                        if measured is None:
                            candidate['status'] = 'failed'
                        else:
                            candidate.update(measured)
                        restore(trainer.model,policies['nonzero'])
                        trainer.optimizer = None
                    report['candidates'].append(candidate)
                    checkpoint()
            report['selected'] = select_configuration(report['candidates'])
            report['status'] = 'passed'
        except BaseException as exc:
            report.update(status='failed',error=type(exc).__name__+': '+str(exc))
            raise
        finally:
            restore(trainer.model,initial)
            trainer.optimizer = None
            trainer._metrics = metrics
            trainer.model.train(was_training)
            apply_configuration(trainer,report.get('selected',{}).get('settings',BASELINE))
            report['elapsed_seconds'] = time.monotonic()-started
            checkpoint()
    return report


def load_selection(path: Path) -> dict:
    import json
    report = json.loads(path.read_text())
    if report.get('status') != 'passed' or report.get('selected',{}).get('status') != 'passed':
        raise ValueError('Missing passed stack qualification')
    winner = select_configuration(report['candidates'])
    if winner != report['selected']:
        raise ValueError('Selected stack differs from passing measured winner')
    return winner['settings']


def check_kernel_selection(current: dict, smoke: dict) -> None:
    for key in ('status','dispatch','fallback_dispatch'):
        if current.get(key) != smoke.get(key):
            raise ValueError('Kernel configuration differs from the qualified smoke: '+key)
    if current.get('numerical_agreement',{}).get('status') != 'passed':
        raise ValueError('Current kernel configuration has not passed qualification')
