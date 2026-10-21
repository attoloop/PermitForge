"""Learnability card and results page from a cloud training run.

python -m eval.learnability --run DIR [--jobs JOBS.json] [--write]

DIR is a downloaded run folder (the dataset keeps it under training/runs/<run>).
It holds base-eval/, eval/, grpo/ and run.json. JOBS.json is the HF Jobs ledger
for the project (train/cloud/ops/jobs_ledger.py). --write updates
results/eval-v2-qwen.json and train/cloud/RESULTS.md; without it the card is printed.
"""
from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'eval'

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics

from .card import GITHUB, HUB_DATASET, baseline_scores
from .common import ROOT, TOOLS, records, saved_fraction, write_json, write_text
from .report import cluster_ci, summarize

BASE, TRAINED = 'hf:Qwen/Qwen3.8-27B:base', 'hf:Qwen/Qwen3.8-27B:adapter'
NAMES = {BASE: 'Qwen3.8-27B', TRAINED: 'Qwen3.8-27B + GRPO LoRA'}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def episodes(run: Path, stage: str, model: str) -> list[dict]:
    rows = [r for _, r in records([run/stage]) if r['model'] == model]
    if not rows:
        raise ValueError(f'no {model} episodes under {run/stage}')
    return rows


def pair_table(base: list[dict], trained: list[dict]) -> dict:
    """Per-task means over repeats and their paired difference (trained minus base)."""
    by = {m: defaultdict(list) for m in ('base', 'trained')}
    for name, rows in (('base', base), ('trained', trained)):
        for r in sorted(rows, key=lambda r: r['repeat']):
            by[name][r['task_id']].append(r)
    if set(by['base']) != set(by['trained']):
        raise ValueError('base and trained evaluations cover different tasks')
    tasks = []
    for tid in sorted(by['base']):
        b = [saved_fraction(r) for r in by['base'][tid]]
        t = [saved_fraction(r) for r in by['trained'][tid]]
        tasks.append({'task_id': tid, 'station': by['base'][tid][0]['station'], 'base': b, 'trained': t,
                      'base_mean': statistics.mean(b), 'trained_mean': statistics.mean(t),
                      'diff': statistics.mean(t) - statistics.mean(b)})
    diffs = [t['diff'] for t in tasks]
    sd = statistics.stdev(diffs) if len(diffs) > 1 else None
    return {'tasks': tasks, 'n': len(tasks), 'mean_diff': statistics.mean(diffs),
            'sd': sd, 'se': None if sd is None else sd/math.sqrt(len(diffs)),
            'ci95': cluster_ci(diffs, [t['station'] for t in tasks]),
            'wins': sum(d > 0 for d in diffs), 'losses': sum(d < 0 for d in diffs), 'ties': sum(d == 0 for d in diffs)}


def training_summary(run: Path) -> dict:
    """Per-step trainer logs joined with the rollouts TRL generated for that step."""
    log = [row for row in read(run/'grpo/trainer_state.json')['log_history'] if 'reward' in row]
    rollouts = defaultdict(list)
    for line in (run/'grpo/episodes.jsonl').read_text().splitlines():
        row = json.loads(line)
        rollouts[row['optimizer_step'] + 1].append(row)
    steps = []
    for row in log:
        rs = rollouts.get(row['step'], [])
        steps.append({'step': row['step'], 'reward': row['reward'], 'reward_std': row['reward_std'],
                      'grad_norm': row['grad_norm'], 'entropy': row['entropy'], 'loss': row['loss'],
                      'learning_rate': row['learning_rate'], 'step_seconds': row['step_time'],
                      'completion_mean': row['completions/mean_length'],
                      'rollouts': len(rs), 'turn_limit': sum(r['turn_limit'] for r in rs),
                      'zero_reward': sum(((r['final'] or {}).get('reward') or 0) == 0 for r in rs),
                      'no_evaluation': sum(r['submitted'] and r['evaluations_used'] == 0 for r in rs),
                      'budget_cut': sum(r['budget_cut'] for r in rs)})
    thirds = []
    size = math.ceil(len(steps)/3)
    for i in range(0, len(steps), size):
        part = steps[i:i+size]
        thirds.append({'steps': f"{part[0]['step']}-{part[-1]['step']}",
                       'reward': statistics.mean(s['reward'] for s in part),
                       **{k: sum(s[k] for s in part) for k in ('rollouts', 'turn_limit', 'zero_reward', 'no_evaluation', 'budget_cut')}})
    total = sum(len(v) for v in rollouts.values())
    return {'steps': steps, 'thirds': thirds, 'rollouts': total,
            'extra_rollouts': total - sum(s['rollouts'] for s in steps if s['rollouts'] <= 6) - sum(6 for s in steps if s['rollouts'] > 6)}


def ledger(jobs: list[dict], run_name: str) -> dict:
    """Group the HF Jobs ledger by run and stage; cost at the hourly rate each job was billed."""
    groups = defaultdict(lambda: {'jobs': 0, 'seconds': 0.0, 'usd': 0.0, 'failed': 0})
    for job in jobs:
        args = job_args(job)
        if job.get('kind') == 'probe':
            key = 'node probes (1-second jobs)'
        else:
            name = args.split('--artifact-uri ')[-1].split(' ')[0].rsplit('/', 1)[-1] if '--artifact-uri' in args else 'other'
            stage = args.split('--stage ')[-1].split(' ')[0] if '--stage ' in args else ('smoke' if '--recipe smoke' in args else 'grpo')
            key = f'{name} {stage}' if name == run_name else f'smoke {name}' if name.startswith('smoke') else name
        g = groups[key]
        g['jobs'] += 1
        g['seconds'] += job.get('seconds') or 0
        g['usd'] += job.get('usd') or 0
        g['failed'] += job.get('stage') not in ('COMPLETED',)
    rows = [{'phase': k, **v} for k, v in groups.items()]
    return {'rows': rows, 'total_usd': sum(r['usd'] for r in rows), 'total_jobs': sum(r['jobs'] for r in rows)}


def job_args(job: dict) -> str:
    return (job.get('command') or '') + ' ' + (job.get('args') or '')


def stage_cost(jobs: list[dict], run_name: str, stage: str, hourly: float) -> float | None:
    pattern = re.compile(rf'/{re.escape(run_name)}(\s|$).*--stage {re.escape(stage)}(\s|$)')
    seconds = [j.get('seconds') or 0 for j in jobs if pattern.search(job_args(j)) and j.get('stage') == 'COMPLETED']
    return sum(seconds)*hourly/3600 if seconds else None


def build_card(run: Path, subsets: dict, jobs: list[dict] | None) -> dict:
    config = read(run/'run.json')['config']
    sampling = read(run/'eval/sampling.json')
    base, trained = episodes(run, 'base-eval', BASE), episodes(run, 'eval', TRAINED)
    tasks = {r['task_id'] for r in base}
    if tasks != {t['task_id'] for t in subsets['eval']}:
        raise ValueError('training evaluation does not use the frozen eval cohort')
    models = []
    for stage, rows in (('base-eval', base), ('eval', trained)):
        s = summarize(rows)
        cost = stage_cost(jobs or [], run.name, stage, config['hourly_usd'])
        models.append({'model': NAMES[rows[0]['model']], 'provider': 'HF Jobs, one H200, Transformers generation',
                       'runs': s['runs'], 'mean': round(s['mean'], 4), 'ci95': [round(x, 4) for x in s['ci95']],
                       'feasible': sum(r['ended'] == 'submitted' and r['final'].get('feasible') for r in rows),
                       'usd_per_episode': None if cost is None else round(cost/s['runs'], 4)})
    return {
        'env': 'permitforge', 'version': 'v2',
        'authors': [{'name': 'Attoloop', 'github': 'attoloop', 'affiliation': 'Attoloop'}],
        'framework': 'openenv',
        'tasks': {'split': 'eval', 'n': len(tasks),
                  'source': 'The same 10 held-out eval tasks as eval-v2.json (NOAA ISD + IGRA2 2020-2024 weather, '
                            'synthetic facilities, EPA AERMOD 26135 / BPIPPRM 04274, ARM2 for NO2, discrete height menus).',
                  'selection': 'The frozen eval cohort of eval-v2.json, three repeats per model and task, seeds fixed. '
                               'No training task is in it; train and eval are separated by weather station.'},
        'harness': {'agent': 'permitforge cloud eval harness (train/cloud/evaluation.py)',
                    'tools': list(TOOLS), 'max_turns': sampling['max_turns'],
                    'evaluate_budget': sampling['evaluate_budget'], 'repeats': sampling['repeat_count'],
                    'notes': f"{config['model']} at revision {config['revision'][:12]}, bf16, Transformers generation on one H200, "
                             f"thinking off, temperature {sampling['generation_config']['temperature']}, top_p "
                             f"{sampling['generation_config']['top_p']}, top_k {sampling['generation_config']['top_k']}, "
                             f"{sampling['max_tokens']:,} output tokens per reply, {sampling['wall_clock']} s wall clock per episode. "
                             'Tool results carry a turns-left line; one retry after an empty reply and a correction prompt after '
                             'a malformed call. Not the frontier harness of eval-v2.json; the two cards are not compared.'},
        'grading': {'reference': 'Same as eval-v2.json: reward is the feasible saved fraction of the draft design cost, '
                                 'verified with native BPIP/AERMOD; invalid, infeasible or no submission scores zero; '
                                 '1.0 would require zero cost and is unreachable.',
                    'reward': '(cost(draft) - cost(submission)) / cost(draft). Equal task weight after averaging repeats; '
                              '95% station-clustered percentile bootstrap (5 stations, 2000 draws, seed 0).',
                    'excused': 'nothing', 'judge': False},
        'baselines': baseline_scores(subsets['eval']),
        'models': models,
        'links': {'runs': f'{HUB_DATASET}/tree/main/training/runs/{run.name}',
                  'replay': f'{HUB_DATASET}/tree/main/training/runs/{run.name}/eval',
                  'code': f'{GITHUB}/tree/main/train/cloud'},
        'date': max(r['finished_at'] for r in trained)[:10],
    }


def results_markdown(run: Path, card: dict, jobs: list[dict] | None) -> str:
    config = read(run/'run.json')['config']
    base, trained = episodes(run, 'base-eval', BASE), episodes(run, 'eval', TRAINED)
    pairs = pair_table(base, trained)
    train = training_summary(run)
    grpo = read(run/'grpo/metrics.json')
    b, t = card['models']
    def f(x, d=3):
        return 'n/a' if x is None else f'{x:.{d}f}'
    ci = 'n/a' if pairs['ci95'] is None else f"{pairs['ci95'][0]:+.4f} to {pairs['ci95'][1]:+.4f}"
    lines = ['# Qwen3.8-27B GRPO on PermitForge: results', '',
             f"Run `{run.name}`: {config['grpo_steps']} GRPO steps, {config['num_generations']} rollouts per step, "
             f"LoRA rank {config['lora_rank']} on one H200. Base and trained model evaluated on the same 10 held-out "
             f"tasks, 3 repeats each, native verification. Everything here is computed from the run folder "
             f"(`python -m eval.learnability --run <folder> --write`).", '',
             '## Result', '',
             '| Model | Mean saved fraction | 95% CI | Feasible | USD per episode |', '|---|---:|---|---:|---:|',
             f"| {b['model']} | {b['mean']:.4f} | {b['ci95'][0]:.4f} to {b['ci95'][1]:.4f} | {b['feasible']}/{b['runs']} | {f(b['usd_per_episode'], 2)} |",
             f"| {t['model']} | {t['mean']:.4f} | {t['ci95'][0]:.4f} to {t['ci95'][1]:.4f} | {t['feasible']}/{t['runs']} | {f(t['usd_per_episode'], 2)} |", '',
             f"Paired difference (trained minus base) over {pairs['n']} tasks: mean {pairs['mean_diff']:+.4f}, "
             f"sd {f(pairs['sd'], 4)}, se {f(pairs['se'], 4)}, station-clustered 95% CI {ci}; "
             f"{pairs['wins']} tasks up, {pairs['losses']} down, {pairs['ties']} unchanged.", '',
             'No measurable change in the mean. The largest per-task moves are about four points either way, inside '
             'what three repeats can resolve for most tasks, and nothing in the pattern says the model learned to find '
             'cheaper designs.', '',
             '## Per task', '',
             '| Task | Station | Base r1 / r2 / r3 | Trained r1 / r2 / r3 | Base mean | Trained mean | Diff |',
             '|---|---|---|---|---:|---:|---:|']
    for row in pairs['tasks']:
        lines.append(f"| `{row['task_id']}` | {row['station']} | {' / '.join(f'{x:.3f}' for x in row['base'])} | "
                     f"{' / '.join(f'{x:.3f}' for x in row['trained'])} | {row['base_mean']:.4f} | {row['trained_mean']:.4f} | {row['diff']:+.4f} |")
    lines += ['', '## Training', '',
              f"{train['rollouts']} rollouts were generated for {len(train['steps'])} optimizer steps. "
              f"Steps whose row shows 12 rollouts were generated twice: the first job died at step 15 and training "
              f"resumed from checkpoint 10, so those steps were sampled again (only step 11's two groups come from "
              f"the same checkpoint). Rewards are the training grade (fast evaluator); zero means an infeasible, "
              f"invalid or missing submission.", '',
              '| Steps | Mean reward | Rollouts | Zero reward | Turn limit | Submitted without evaluating | Budget cut |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for third in train['thirds']:
        lines.append(f"| {third['steps']} | {third['reward']:.3f} | {third['rollouts']} | {third['zero_reward']} | "
                     f"{third['turn_limit']} | {third['no_evaluation']} | {third['budget_cut']} |")
    lines += ['', '| Step | Reward | Reward std | Grad norm | Entropy | Loss | LR | Mean completion tokens | Seconds | Rollouts |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in train['steps']:
        lines.append(f"| {s['step']} | {s['reward']:.3f} | {s['reward_std']:.3f} | {s['grad_norm']:.3f} | {s['entropy']:.3f} | "
                     f"{s['loss']:.4f} | {s['learning_rate']:.1e} | {s['completion_mean']:.0f} | {s['step_seconds']:.0f} | {s['rollouts']} |")
    lines += ['', f"Peak device memory {grpo['peak_vram_reserved_bytes']/1e9:.1f} GB reserved by torch; "
              f"mean {grpo['seconds_per_step']:.0f} s per step in the last chunk; endings over all rollouts: "
              f"{json.dumps(grpo['ending_counts'])}.", '']
    if jobs:
        led = ledger(jobs, run.name)
        unknown = sum(j.get('seconds') is None for j in jobs)
        lines += ['## Cost', '', f"Every HF job of this project, billed per second at $5 per H200 hour: "
                  f"{led['total_jobs']} jobs, ${led['total_usd']:.2f} in all. {unknown} job(s) have no recorded "
                  f"duration and count as zero; the frontier panel's API cost is separate.", '',
                  '| Phase | Jobs | Not completed | Hours | USD |', '|---|---:|---:|---:|---:|']
        for r in sorted(led['rows'], key=lambda r: -r['usd']):
            lines.append(f"| {r['phase']} | {r['jobs']} | {r['failed']} | {r['seconds']/3600:.2f} | {r['usd']:.2f} |")
        lines.append('')
    lines += ['## What went wrong, and the fixes', '',
              '- Some H200 hosts (driver 580.178.04) start the container without CUDA libraries. The GPU guard now '
              'reports it in a minute and every launch is gated on a one-second probe job landing on a good host '
              '(`train/cloud/ops/run-full.sh`).',
              '- TRL computes the logits of the whole completion at once. With a 248,320-token vocabulary a 27k-token '
              'completion needs 27 GB of fp32 logits and the first full-length step ran out of memory. '
              '`train/cloud/chunked_logps.py` computes log-probabilities in 2,048-token chunks under activation '
              'checkpointing; a CPU test shows the values and gradients equal TRL\'s.',
              '- The first training job died at step 15 on the device watchdog: 135 GB in use on the device while '
              'torch reported 118 GB reserved. The cause of the gap is not known. Training resumed from checkpoint 10 '
              'in 5-step jobs (`--restore --stop-after-step N`) and the gap never grew that large again.', '',
              '## Reading', '',
              'Three guesses, none established. The base model already submits a feasible design in every held-out '
              'episode, so the failure modes training could remove on the train tasks (turn-limit endings and '
              'submitting without evaluating) had nothing to fix on the eval tasks. The skill that would move the '
              'score, finding cheaper feasible designs in 30 evaluations, got a weak signal: little reward spread '
              'inside many groups, 180 rollouts in the final lineage, and a learning rate near zero by step 30. And '
              'the held-out stations have weather the model never trained on.', '',
              '## Next experiment', '',
              'If this is run again: start from a supervised fit on the frontier trajectories of the train split, keep '
              'the learning rate constant, use 8 to 16 rollouts per step for a usable spread, train for 100 or more '
              'steps on more than 30 tasks, and evaluate checkpoints along the way. At the measured pace that is '
              'roughly $400 to $600 of H200 time.', '']
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--jobs', type=Path)
    parser.add_argument('--subsets', type=Path, default=ROOT/'results/subsets.json')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    jobs = read(args.jobs) if args.jobs else None
    card = build_card(args.run, read(args.subsets), jobs)
    if args.write:
        write_json(ROOT/'results/eval-v2-qwen.json', card)
        write_text(ROOT/'train/cloud/RESULTS.md', results_markdown(args.run, card, jobs))
        print(ROOT/'results/eval-v2-qwen.json', ROOT/'train/cloud/RESULTS.md')
    else:
        print(json.dumps(card, indent=2))


if __name__ == '__main__':
    main()
