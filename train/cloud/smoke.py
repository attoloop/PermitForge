"""Run the bounded smoke sequence and write its measured qualification report."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

from .run import write_json


def smoke_summary(records: list[dict], training: dict, gpu: dict, *,
                  elapsed_seconds: float, hourly_usd: float, kernels: dict | None = None) -> dict:
    valid_calls = sum(not (step.get('result') or {}).get('errors') and
        (step.get('result') or {}).get('valid') is not False
        for row in records for step in row.get('steps', row.get('history', [])))
    submitted = sum(r.get('ended') == 'submitted' for r in records)
    positive = sum(r.get('reward',0) > 0 for r in records)
    infrastructure = any(r.get('grading_failed') or r.get('error_type') == 'infrastructure' for r in records)
    spread = any((g.get('reward_std') or 0) > 0 for g in training.get('groups', []))
    ready = (len(records) == 3 and not infrastructure and valid_calls > 0 and submitted > 0 and positive > 0
        and training.get('status') == 'passed' and training.get('optimizer_steps') == 2
        and training.get('weights_changed') is True and training.get('adapter_reload_equal') is True and spread
        and kernels is not None and kernels.get('status') == 'passed')
    seconds = training.get('seconds_per_step')
    return dict(status='passed' if ready else 'failed', ready_for_grpo=bool(ready),
        base=dict(episodes=len(records), valid_tool_calls=valid_calls, submitted=submitted,
                  positive_reward=positive, infrastructure_failure=infrastructure),
        training=training, gpu=gpu, kernels=kernels, elapsed_seconds=elapsed_seconds,
        hourly_usd=hourly_usd,
        measured_compute_usd=elapsed_seconds/3600*hourly_usd,
        seconds_per_grpo_step=seconds, projections={str(n):dict(
            grpo_seconds=seconds*n if seconds else None,
            grpo_compute_usd=seconds*n/3600*hourly_usd if seconds else None) for n in (20,50,100)},
        projection_note='Measured step mean times steps; excludes setup, fields, model loading, base/final '
            'evaluation and export. Two steps do not establish steady-state throughput or learning.')


def smoke_gate(path: Path, current: dict) -> None:
    report = json.loads((path/'smoke/metrics.json').read_text())
    previous = json.loads((path/'run.json').read_text())
    keys = ('model','revision','precision','num_generations','accum','max_length',
            'max_completion_length','max_prompt_length','train_turns','lora_rank','lora_alpha')
    if (not report.get('ready_for_grpo') or report.get('status') != 'passed' or
            report.get('supervisor_status') != 'complete' or
            any(previous['config'][k] != current['config'][k] for k in keys) or
            previous['hashes'] != current['hashes'] or previous['selection'] != current['selection']):
        raise ValueError('A passed smoke with identical code, data, model and training settings is required')
    for name in ('kernels', 'gpu'):
        if json.loads((path/f'smoke/{name}.json').read_text()).get('status') != 'passed':
            raise ValueError('A passed smoke requires successful kernel sanity and GPU checks')


def run_smoke(config, record: dict, pack, output: Path) -> None:
    import torch
    from .fields import CloudFields, warm
    from .training import load_model, train
    from .evaluation import evaluate
    started = time.monotonic()
    root = Path(record['fields_dir'])
    tasks_dir = Path(record['tasks_dir'])
    train_tasks = [pack.get(t) for t in record['train_tasks']]
    eval_tasks = [pack.get(t) for t in record['eval_tasks']]
    training, records = {}, []
    try:
        warm(train_tasks+eval_tasks, root, output/'fields', config.field_jobs)
        fields = CloudFields(train_tasks+eval_tasks,root,tasks_dir=tasks_dir,
                             output=output/'field-misses',jobs=config.field_jobs)
        model, tokenizer = load_model(config)
        try:
            evaluate(replace(config, eval_repeats=1),model,tokenizer,pack,record['eval_tasks'],
                     output/'base-eval',fields_loader=fields,tasks_dir=tasks_dir,adapter=None)
        finally:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        training = train(config,'grpo',output/'grpo',data=Path(record['sft_data']),pack=pack,
                         tasks=train_tasks,fields_loader=fields,smoke=True)
        if set(training.get('tasks_seen',[])) != set(record['train_tasks']):
            raise RuntimeError('Smoke must exercise both selected training tasks')
    finally:
        records = [json.loads(p.read_text()) for p in sorted((output/'base-eval').rglob('episode.json'))]
        if (output/'grpo/metrics.json').exists():
            training = json.loads((output/'grpo/metrics.json').read_text())
        kernel_path = output/'kernels.json'
        kernels = json.loads(kernel_path.read_text()) if kernel_path.exists() else None
        report = smoke_summary(records,training,{},elapsed_seconds=time.monotonic()-started,
                               hourly_usd=config.hourly_usd,kernels=kernels)
        write_json(output/'metrics.json',report)
        write_report(output)
    if not report['ready_for_grpo']:
        raise RuntimeError('Smoke did not qualify direct GRPO; inspect retained behavior and reward spread')


def finalize_report(output: Path, status: str, elapsed: float, hourly: float) -> None:
    path = output/'metrics.json'
    if path.exists():
        report = json.loads(path.read_text())
    else:
        records = [json.loads(p.read_text()) for p in (output/'base-eval').rglob('episode.json')]
        training = output/'grpo/metrics.json'
        report = smoke_summary(records,json.loads(training.read_text()) if training.exists() else {},{},
                               elapsed_seconds=elapsed,hourly_usd=hourly)
    report['supervisor_status'] = status
    if status != 'complete':
        report.update(status='failed',ready_for_grpo=False)
    write_json(path,report)
    write_report(output)


def write_report(output: Path) -> None:
    path = output/'metrics.json'
    if not path.exists():
        return
    report = json.loads(path.read_text())
    setup = output.parent/'setup.json'
    if setup.exists():
        report['setup'] = json.loads(setup.read_text())
        report['setup_compute_usd'] = report['setup']['elapsed_seconds']/3600*report['hourly_usd']
    kernels = output/'kernels.json'
    if kernels.exists():
        report['kernels'] = json.loads(kernels.read_text())
    if (report.get('kernels') or {}).get('status') != 'passed':
        report.update(status='failed',ready_for_grpo=False)
    gpu = output/'gpu.json'
    if gpu.exists():
        report['gpu'] = json.loads(gpu.read_text())
        if report['gpu'].get('status') != 'passed':
            report.update(status='failed',ready_for_grpo=False)
    write_json(path,report)
    text = ['# Qwen3.8 H200 smoke', '', f"Ready for GRPO: {report['ready_for_grpo']}", '',
        'Base behavior: '+json.dumps(report['base']), '',
        'Memory: '+json.dumps(report['gpu']), '',
        'Linear-attention sanity: '+json.dumps(report.get('kernels',{})), '',
        f"Seconds per GRPO step: {report['seconds_per_grpo_step']}", '',
        '| Steps | Projected GRPO seconds | Projected GRPO USD |', '|---|---|---|']
    for n, row in report['projections'].items():
        text.append(f"| {n} | {row['grpo_seconds']} | {row['grpo_compute_usd']} |")
    text.extend(['', report['projection_note'], '',
        'Reproduce: read smoke/metrics.json, smoke/grpo/metrics.json and smoke/base-eval/**/episode.json.', ''])
    (output/'REPORT.md').write_text('\n'.join(text))
