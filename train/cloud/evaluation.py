"""HF-generation panel episodes, unchanged public prompt and native submissions."""
import hashlib
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from .formatting import parse_calls, render
from .run import write_json


class EpisodeTimeout(BaseException):
    pass


def grading_command(tasks_dir: Path, task_id: str, design: Path, output: Path) -> list[str]:
    from permit_core import fields
    # The grader is a fresh process; hand it this worker's field-cache volume explicitly.
    return [sys.executable,'-m','train.cloud.verify',str(tasks_dir),task_id,str(design),str(output),
            str(fields.cache_root())]


def evaluate(config, model, tokenizer, pack, ids: list[str], output: Path, *, fields_loader,
             tasks_dir: Path, adapter: Path | None) -> None:
    import torch
    from transformers import set_seed
    from permit_core.evaluate import evaluate as fast_evaluate
    from permit_openenv.environment import PermitForgeEnvironment
    from permit_openenv.prompts import INSTRUCTIONS
    from permit_openenv.protocol import json_value

    class CachedEvaluator:
        def evaluate(self, task, design):
            return json_value(fast_evaluate(task.scenario, design, fields_loader(task), include_receptors=True))

    class IsolatedVerifier:
        def __init__(self, folder):
            self.folder = folder
        def evaluate(self, task, design):
            write_json(self.folder/'submitted-design.json', design)
            # Record before dispatch: interruptions here are held, never silently regraded.
            write_json(self.folder/'pending-grade.json', {'task_id':task.task_id})
            subprocess.run(grading_command(tasks_dir,task.task_id,self.folder/'submitted-design.json',
                self.folder/'native-grade.json'),check=True)
            result = json.loads((self.folder/'native-grade.json').read_text())
            (self.folder/'pending-grade.json').unlink()
            return result

    output.mkdir(parents=True, exist_ok=True)
    sampling = dict(max_turns=config.eval_turns,max_tokens=config.eval_max_tokens,
        wall_clock=config.eval_wall_seconds,evaluate_budget=30,repeat_count=config.eval_repeats,
        enable_thinking=False,temperature='pinned model generation_config',
        generation_config=model.generation_config.to_dict())
    write_json(output/'sampling.json', sampling)
    records = []
    for repeat in range(1,config.eval_repeats+1):
        for index, task_id in enumerate(ids):
            folder = output/'qwen3.8-27b'/task_id/f'r{repeat}'
            folder.mkdir(parents=True, exist_ok=True)
            final_path = folder/'episode.json'
            if (folder/'pending-grade.json').exists():
                raise RuntimeError('Unresolved native submission: '+str(folder)+'; recover authoritative grade before resume')
            if final_path.exists():
                records.append(json.loads(final_path.read_text()))
                continue
            if (folder/'started.json').exists():
                # No selective retry after a killed sampled episode.
                previous = json.loads((folder/'progress.json').read_text())
                previous.update(ended='error', error_type='infrastructure', grading_failed=True)
                write_json(final_path, previous)
                records.append(previous)
                continue
            task = pack.get(task_id)
            env = PermitForgeEnvironment(pack, evaluator=CachedEvaluator(), verifier=IsolatedVerifier(folder))
            env.reset(task_id=task_id)
            import asyncio
            specs = asyncio.run(env._require_mcp_server().list_tools())
            tools = [{'type':'function','function':{'name':t.name,'description':t.description,
                       'parameters':t.parameters}} for t in specs]
            seed = config.seed + repeat*1_000_003 + index
            set_seed(seed)
            model.eval()
            messages = [{'role':'system','content':INSTRUCTIONS}, {'role':'user','content':
                json.dumps(task.view())+f'\nYou have {config.eval_turns} turns. Submit before your final turn ends.'}]
            model_name = 'hf:'+config.model+(':adapter' if adapter else ':base')
            usage = {'input_tokens':0,'output_tokens':0,'usd':None}
            generation_turns = []
            started = time.monotonic()
            turn = -1
            def save_progress():
                record = env.episode.record(model_name,usage)
                record.update(repeat=repeat, seed=seed, settings=sampling,
                    turns=turn+1, adapter=str(adapter) if adapter else None, model_revision=config.revision,
                    prompt_sha256=hashlib.sha256(INSTRUCTIONS.encode()).hexdigest())
                record['generation_turns'] = generation_turns
                write_json(folder/'progress.json', record)
                return record
            save_progress()
            write_json(folder/'started.json', {'seed':seed})
            old_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(EpisodeTimeout()))
            signal.setitimer(signal.ITIMER_REAL, config.eval_wall_seconds)
            error = None
            turn = -1
            try:
                with (folder/'transcript.jsonl').open('a') as trace:
                    empty = 0
                    for turn in range(config.eval_turns):
                        if turn == config.eval_turns-1:
                            messages.append({'role':'user','content':'Last turn: submit your best feasible design now.'})
                        prompt = render(tokenizer,messages,tools,prompt=True)
                        inputs = tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to(model.device)
                        usage['input_tokens'] += inputs.input_ids.shape[1]
                        if inputs.input_ids.shape[1]+config.eval_max_tokens > model.config.max_position_embeddings:
                            env.episode.finish('context_limit')
                            break
                        torch.cuda.synchronize()
                        generation_started = time.monotonic()
                        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
                            generated = model.generate(**inputs, max_new_tokens=config.eval_max_tokens,
                                use_cache=True,logits_to_keep=1,pad_token_id=tokenizer.pad_token_id)
                        ids_out = generated[0,inputs.input_ids.shape[1]:].tolist()
                        torch.cuda.synchronize()
                        generation_turns.append(dict(turn=turn,seconds=time.monotonic()-generation_started,
                            input_tokens=inputs.input_ids.shape[1],output_tokens=len(ids_out)))
                        usage['output_tokens'] += len(ids_out)
                        response = tokenizer.decode(ids_out,skip_special_tokens=False)
                        trace.write(json.dumps(dict(turn=turn,response=response,elapsed_seconds=time.monotonic()-started))+'\n')
                        trace.flush()
                        try:
                            calls = parse_calls(response)
                        except ValueError as exc:
                            messages.extend([{'role':'assistant','content':response},
                                {'role':'user','content':'Invalid tool-call format: '+str(exc)}])
                            save_progress()
                            continue
                        content = response.split('<tool_call>',1)[0].replace('<|im_end|>','').strip()
                        messages.append({'role':'assistant','content':content,'tool_calls':calls})
                        if not calls:
                            empty += 1
                            if empty >= 2:
                                break
                            messages.append({'role':'user','content':'Use the tools; only submit_design submits an answer.'})
                        else:
                            empty = 0
                        for call in calls:
                            function = call['function']
                            result = json.loads(env._dispatch(function['name'],function['arguments']))
                            trace.write(json.dumps(dict(tool=function['name'],args=function['arguments'],result=result))+'\n')
                            trace.flush()
                            messages.append({'role':'tool','name':function['name'],
                                'content':json.dumps(result)+f'\nTurns left: {config.eval_turns-turn-1}'})
                            save_progress()
                            if env.episode.done:
                                break
                        if env.episode.done:
                            break
                    else:
                        env.episode.finish('turn_limit')
            except EpisodeTimeout:
                env.episode.finish('timeout')
                error = 'episode wall-clock limit'
            except Exception as exc:
                env.episode.finish('error')
                error = type(exc).__name__+': '+str(exc)
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                signal.signal(signal.SIGALRM,old_handler)
                if not env.episode.done:
                    env.episode.finish('no_submit')
                record = save_progress()
                if error:
                    record.update(error=error,error_type='timeout' if record['ended']=='timeout' else 'infrastructure')
                if any('E_INFRA' in e for e in (record.get('final') or {}).get('errors',[])):
                    record['grading_failed'] = True
                write_json(final_path,record)
                records.append(record)
                env.close()
            if (folder/'pending-grade.json').exists():
                raise RuntimeError('Submission interrupted; retained pending marker for review')
    write_json(output/'index.json', {'episodes':[{'task_id':r['task_id'],'repeat':r['repeat'],
        'ended':r['ended'],'reward':r['reward']} for r in records]})
    summary = summarize_records(records,ids,repeats=config.eval_repeats)
    write_json(output/'metrics.json', {'status':'passed' if summary['complete'] else 'incomplete', **summary})
    if not summary['complete']:
        raise RuntimeError('Evaluation contains infrastructure failures; records retained without selective reruns')


def summarize_records(records: list[dict], ids: list[str], *, repeats: int) -> dict:
    graded = [r for r in records if not r.get('grading_failed') and r.get('error_type') != 'infrastructure']
    per_task = []
    for tid in ids:
        values = [r['reward'] for r in graded if r['task_id']==tid]
        per_task.append(dict(task_id=tid,repeats=len(values),mean=statistics.mean(values) if values else None,
            std=statistics.stdev(values) if len(values)>1 else None,
            min=min(values) if values else None,max=max(values) if values else None))
    means = [r['mean'] for r in per_task if r['mean'] is not None]
    return dict(episodes=len(records),infrastructure_episodes=len(records)-len(graded),
        graded_tasks=len(means),complete=all(r['repeats']==repeats for r in per_task),per_task=per_task,
        task_mean_saved_fraction=statistics.mean(means) if means else None)
