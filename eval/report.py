"""Regenerate tables and card models: python -m eval.report RUN [RUN ...]."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "eval"

import argparse
import csv
import io
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import random
import statistics

from .common import ROOT, failure_category, finite, saved_fraction, report_records, run_path, write_json, write_text


def cluster_ci(values: list[float], stations: list[str], *, n: int = 2000, seed: int = 0) -> list[float] | None:
    groups = defaultdict(list)
    for value, station in zip(values, stations, strict=True):
        groups[station].append(value)
    if len(groups) < 2 or not values:
        return None
    rng = random.Random(seed)
    names = sorted(groups)
    samples = sorted(statistics.mean(x for name in rng.choices(names, k=len(names)) for x in groups[name])
                     for _ in range(n))
    return [samples[int(.025*n)], samples[min(n-1, int(.975*n))]]


def metrics(record: dict) -> dict:
    final = record.get('final') or {}
    feasible = record.get('ended') == 'submitted' and final.get('valid', True) and bool(final.get('feasible'))
    saved = final.get('prescore_usd_saved', -1) if feasible else -1
    if not finite(saved):
        raise ValueError('nonfinite prescore in episode')
    score = saved_fraction(record)
    return {'usd_saved': saved, 'pct_saved': 100*score, 'reward': score, 'feasible': int(feasible),
            'evaluations': record.get('evaluations_used', 0), 'turns': record.get('turns', 0)}


def task_groups(items: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for record in items:
        groups[record['task_id']].append(record)
    for rows in groups.values():
        if len({r['station'] for r in rows}) != 1:
            raise ValueError('station mismatch in repeated task')
        drafts = {r.get('scenario', {}).get('draft_cost') for r in rows if r.get('scenario', {}).get('draft_cost') is not None}
        if len(drafts) > 1:
            raise ValueError('draft cost mismatch in repeated task')
    return dict(sorted(groups.items()))


def summarize(items: list[dict]) -> dict:
    if not items:
        raise ValueError('cannot summarize an empty cohort')
    groups = task_groups(items)
    per_task, values, stations = [], [], []
    for tid, rows in groups.items():
        measured = [metrics(r) for r in rows]
        scores = [v['reward'] for v in measured]
        means = {k: statistics.mean(v[k] for v in measured) for k in measured[0]}
        values.append(means)
        stations.append(rows[0]['station'])
        per_task.append({'task_id': tid, 'station': stations[-1], 'repeats': len(rows),
                         'mean': means['reward'], 'std': statistics.stdev(scores) if len(scores) > 1 else None,
                         'min': min(scores), 'max': max(scores), 'feasible_rate': means['feasible'],
                         'mean_pairwise_spread': statistics.mean(abs(a-b) for a,b in combinations(scores, 2))
                             if len(scores) > 1 else None})
    repeated = [r for r in per_task if r['repeats'] > 1]
    result = {'runs': len(items), 'tasks': len(groups), 'task_ids': list(groups), 'per_task': per_task,
              'station_clusters': len(set(stations)),
              'failure_categories': dict(Counter(failure_category(r) for r in items)),
              'ended_without_submit': sum(r.get('final') is None for r in items),
              'repeated_tasks': len(repeated),
              'grpo_signal': statistics.mean(r['max'] > r['min'] for r in repeated) if repeated else None,
              'mean_pairwise_spread': statistics.mean(r['mean_pairwise_spread'] for r in repeated) if repeated else None}
    for field, name in [('usd_saved', 'mean_usd_saved'), ('pct_saved', 'mean_pct_saved'),
                        ('feasible', 'feasible'), ('evaluations', 'mean_evaluations'),
                        ('turns', 'mean_turns'), ('reward', 'mean')]:
        xs = [v[field] for v in values]
        result[name] = statistics.mean(xs)
        result[name+'_ci95'] = cluster_ci(xs, stations)
    result['ci95'] = result['mean_ci95']
    costs = [r.get('usage', {}).get('usd') for r in items]
    result['usd_per_episode'] = statistics.mean(costs) if all(finite(c) for c in costs) else None
    result['usd_known_episodes'] = sum(finite(c) for c in costs)
    result['per_station'] = {s: {'n': stations.count(s), **{
        field: statistics.mean(v[field] for v,t in zip(values, stations) if t == s)
        for field in values[0]}} for s in sorted(set(stations))}
    return result


def paired(left: list[dict], right: list[dict]) -> dict:
    a, b = task_groups(left), task_groups(right)
    ids = sorted(a.keys() & b.keys())
    result = {'n': len(ids), 'task_ids': ids}
    for metric in ('usd_saved', 'pct_saved', 'reward'):
        deltas = []
        for task in ids:
            if a[task][0]['station'] != b[task][0]['station']:
                raise ValueError('station mismatch in paired task')
            x = statistics.mean(metrics(r)[metric] for r in a[task])
            y = statistics.mean(metrics(r)[metric] for r in b[task])
            deltas.append(x-y)
        result[f'mean_{metric}_delta'] = statistics.mean(deltas) if deltas else None
        result[f'{metric}_delta_ci95'] = cluster_ci(deltas, [a[t][0]['station'] for t in ids]) if deltas else None
    return result


def build_report(items: list[dict], baselines: list[dict] | None = None) -> dict:
    from .best_observed import best_observed
    by = defaultdict(list)
    seen = set()
    for record in items:
        repeat = record.get('repeat', 1)
        if type(repeat) is not int or repeat < 1:
            raise ValueError('repeat must be a positive integer')
        key = record['model'], record['task_id'], repeat
        if key in seen:
            raise ValueError(f'duplicate model/task/repeat across runs: {key}')
        seen.add(key)
        by[record['model']].append(record)
    splits = {r['split'] for r in items}
    if len(splits) > 1:
        raise ValueError('report train and eval separately')
    launch_efforts = set()
    for record in items:
        settings = record.get('settings', {})
        effort = settings.get('effort')
        launch_efforts.add(settings.get('launch_effort', 'xhigh' if effort == 'max' else effort))
    if len(launch_efforts) > 1:
        raise ValueError('cannot pool different efforts in one board')
    best = best_observed(items)
    from .settings import label
    models = []
    for m, rs in sorted(by.items()):
        efforts = {r.get('settings', {}).get('effort') for r in rs}
        if len(efforts) != 1:
            raise ValueError('cannot pool different efforts for '+m)
        models.append({'model': m, 'label': label(m, rs[0].get('settings', {})),
                       'provider': m.split(':')[0], **summarize(rs)})
    for row in models:
        fractions = []
        for task in row['per_task']:
            reference = best.get(task['task_id'], {}).get('saved_fraction', 0)
            task['fraction_of_best_observed'] = task['mean']/reference if reference > 0 else None
            if reference > 0:
                fractions.append(task['fraction_of_best_observed'])
        row['fraction_of_best_observed'] = statistics.mean(fractions) if fractions else None
        row['best_observed_tasks'] = len(fractions)
    return {'models': models, 'split': next(iter(splits), None), 'best_observed': best,
            'ci_method': 'equal task weight after averaging repeats; station-clustered percentile bootstrap; 2000 draws; seed 0; 95%; no CI below two clusters',
            'variance_method': 'sample std; mean absolute difference over unordered repeat pairs; GRPO signal denominator is tasks with at least two repeats; singleton spread is null',
            'secondary_metric': 'fraction of best observed uses model observations in this report only; zero-best tasks excluded with coverage; never reward',
            'raw_savings_policy': 'gate failures retain prescore -1 USD; saved fraction and pct saved are 0 for gate failures',
            'baselines': baselines or [],
            'paired': [{'left': a, 'right': b, **paired(by[a], by[b])} for a,b in combinations(sorted(by), 2)]}


def write_variance(stem: Path, report: dict) -> None:
    rows = [{'model': model['model'], **task} for model in report['models'] for task in model['per_task']]
    write_json(stem.with_suffix('.json'), rows)
    stream = io.StringIO()
    columns = ['model', 'task_id', 'station', 'repeats', 'mean', 'std', 'min', 'max',
               'feasible_rate', 'mean_pairwise_spread', 'fraction_of_best_observed']
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    write_text(stem.with_suffix('.csv'), stream.getvalue())


def markdown(report: dict) -> str:
    lines = ['# PermitForge evaluation', '', report['ci_method']+'.', '',
             'Reward is the saved fraction. Raw USD means include the -1 gate-failure sentinel.', '',
             '| Model | Tasks | Episodes | Mean saved fraction (95% CI) | Feasible | Fraction of best observed | USD saved | % saved | Evaluations | Turns | USD/episode |',
             '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):
        return 'n/a' if x is None else f'{x:.4f}'
    for row in report['models']+report.get('baselines', []):
        interval = row['ci95']
        reward = fmt(row['mean']) + (f' ({fmt(interval[0])}, {fmt(interval[1])})' if interval else '')
        lines.append(f"| {row.get('label', row.get('model', row.get('name')))} | {row['tasks']} | {row['runs']} | {reward} | {row['feasible']:.2%} | "
                     f"{fmt(row.get('fraction_of_best_observed'))} | {row['mean_usd_saved']:.2f} | {row['mean_pct_saved']:.2f} | {fmt(row['mean_evaluations'])} | "
                     f"{fmt(row['mean_turns'])} | {fmt(row['usd_per_episode'])} |")
    lines += ['', '| Model | Station | n | Reward | USD saved | Feasible |', '|---|---|---:|---:|---:|---:|']
    for row in report['models']:
        for station, s in row['per_station'].items():
            lines.append(f"| {row.get('label', row['model'])} | {station} | {s['n']} | {fmt(s['reward'])} | {s['usd_saved']:.2f} | {s['feasible']:.2%} |")
        lines += ['', f"{row.get('label', row['model'])}: ended without submission {row['ended_without_submit']}; "
                  f"categories `{json.dumps(row['failure_categories'], sort_keys=True)}`; "
                  f"coverage {row['station_clusters']} station clusters; GRPO signal {fmt(row['grpo_signal'])} "
                  f"over {row['repeated_tasks']} repeated tasks; mean pairwise spread {fmt(row['mean_pairwise_spread'])}; "
                  f"best-observed reference available for {row['best_observed_tasks']} tasks.", '']
    if report['paired']:
        lines += ['', '| Pair (left minus right) | Matched tasks | USD saved difference | Reward difference |', '|---|---:|---:|---:|']
        for p in report['paired']:
            lines.append(f"| {p['left']} minus {p['right']} | {p['n']} | {fmt(p['mean_usd_saved_delta'])} | {fmt(p['mean_reward_delta'])} |")
    if report.get('infrastructure_attempts'):
        lines += ['', 'Infrastructure attempts (excluded from graded metrics):', '',
                  '| Model | Task | Repeat | Attempt | Retry on resume |', '|---|---|---:|---:|---|']
        for attempt in report['infrastructure_attempts']:
            lines.append(f"| {attempt['model']} | {attempt['task_id']} | {attempt.get('repeat', 1)} | {attempt['attempt_number']} | "
                         f"{'yes' if attempt['retryable'] else 'no'} |")
    return '\n'.join(lines)+'\n'


def baseline_rows(items: list[dict], tasks_root: Path) -> tuple[list[dict], list[str]]:
    """Consume verified WP1.4 policy artifacts; never launch unapproved native work from reporting."""
    from permit_core.baselines import POLICIES
    from .baselines import load_baselines
    by_task = {r['task_id']: r for r in items}
    by_policy = defaultdict(list)
    missing = []
    for tid, episode in sorted(by_task.items()):
        candidates = list(tasks_root.glob(f'**/{tid}/baselines.json'))
        if (tasks_root/(tid+'.json')).exists():
            candidates.append(tasks_root/(tid+'.json'))
        if len(candidates) != 1:
            missing.append(tid)
            continue
        policies = load_baselines(candidates[0])
        for name in POLICIES:
            if name not in policies:
                missing.append(f'{tid}:{name}')
                continue
            result = policies[name]
            rec = {**episode, 'model': 'baseline:'+name, 'ended': 'submitted', 'turns': 0, 'steps': [],
                   'evaluations_used': result.get('evaluations_used', result.get('evaluations', 0)),
                   'usage': {'usd': None}, 'final': {'valid': True, 'feasible': result['feasible'],
                   'prescore_usd_saved': result['usd_saved']}}
            by_policy[name].append(rec)
    rows = [{'name': name, **summarize(rs)} for name,rs in sorted(by_policy.items()) if len(rs) == len(by_task)]
    return rows, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'board.md')
    parser.add_argument('--tasks-root', type=Path, default=ROOT/'envs/permit/tasks')
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--split', choices=['eval', 'train'])
    parser.add_argument('--compute-baselines', action='store_true', help='run core policies using existing cached fields, then verify submissions')
    args = parser.parse_args()
    items = [r for _,r in report_records([run_path(r) for r in args.runs])]
    if args.split:
        items = [r for r in items if r['split'] == args.split]
    baseline_root = args.baseline_root or run_path(args.runs[0])/'baselines'
    if args.compute_baselines:
        from .baselines import compute_cached
        from .transport import load_eval_pack
        pack = load_eval_pack(args.tasks_root)
        for tid in sorted({r['task_id'] for r in items}):
            destination = baseline_root/tid/'baselines.json'
            if not destination.exists():
                compute_cached(pack.get(tid), destination)
    baseline, missing = baseline_rows(items, baseline_root)
    report = build_report(items, baseline)
    from .attempts import infrastructure
    tasks = {r['task_id'] for r in items}
    report['infrastructure_attempts'] = [r for run in args.runs for r in infrastructure(run_path(run))
                                         if r['task_id'] in tasks]
    report['missing_baselines'] = missing
    write_json(args.output.with_suffix('.json'), report)
    write_variance(args.output.with_name(args.output.stem+'-variance'), report)
    write_text(args.output, markdown(report) + ('\nBaseline coverage incomplete; see JSON missing_baselines.\n' if missing else ''))
    print(args.output)


if __name__ == '__main__':
    main()
