"""Regenerate tables and card models: python -m eval.report RUN [RUN ...]."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "eval"

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import random
import statistics

from .common import ROOT, failure_category, finite, grade, records, run_path, write_json, write_text


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
    feasible = bool(final.get('feasible'))
    saved = final.get('prescore_usd_saved', -1) if feasible else -1
    if not finite(saved):
        raise ValueError('nonfinite prescore in episode')
    anchors = final.get('anchors') or record.get('anchors') or {}
    score = grade(saved, anchors) if anchors.get('source') == 'panel' else None
    draft = record.get('scenario', {}).get('draft_cost')
    pct = 100*saved/draft if feasible and finite(draft) and draft > 0 else 0
    return {'usd_saved': saved, 'pct_saved': pct, 'reward': score, 'feasible': int(feasible),
            'evaluations': record.get('evaluations_used', 0), 'turns': record.get('turns', 0)}


def summarize(items: list[dict]) -> dict:
    if not items:
        raise ValueError('cannot summarize an empty cohort')
    values = [metrics(r) for r in items]
    stations = [r['station'] for r in items]
    result = {'runs': len(items), 'task_ids': sorted({r['task_id'] for r in items}),
              'station_clusters': len(set(stations)),
              'failure_categories': dict(Counter(failure_category(r) for r in items)),
              'ended_without_submit': sum(r.get('final') is None for r in items),
              'calibrated_tasks': sum(v['reward'] is not None for v in values)}
    for field, name in [('usd_saved', 'mean_usd_saved'), ('pct_saved', 'mean_pct_saved'),
                        ('feasible', 'feasible'), ('evaluations', 'mean_evaluations'),
                        ('turns', 'mean_turns'), ('reward', 'mean')]:
        xs = [v[field] for v in values]
        result[name] = statistics.mean(xs) if all(x is not None for x in xs) else None
        result[name+'_ci95'] = cluster_ci(xs, stations) if result[name] is not None else None
    result['ci95'] = result['mean_ci95']
    costs = [r.get('usage', {}).get('usd') for r in items]
    result['usd_per_episode'] = statistics.mean(costs) if all(finite(c) for c in costs) else None
    result['usd_known_episodes'] = sum(finite(c) for c in costs)
    result['per_station'] = {s: {'n': stations.count(s), **{
        field: statistics.mean([v[field] for v,t in zip(values, stations) if t == s])
        if all(v[field] is not None for v,t in zip(values, stations) if t == s) else None
        for field in values[0]}} for s in sorted(set(stations))}
    return result


def paired(left: list[dict], right: list[dict]) -> dict:
    a, b = {r['task_id']: r for r in left}, {r['task_id']: r for r in right}
    ids = sorted(a.keys() & b.keys())
    result = {'n': len(ids), 'task_ids': ids}
    for metric in ('usd_saved', 'pct_saved', 'reward'):
        deltas = []
        for task in ids:
            if a[task]['station'] != b[task]['station']:
                raise ValueError('station mismatch in paired task')
            x, y = metrics(a[task])[metric], metrics(b[task])[metric]
            deltas.append(x-y if x is not None and y is not None else None)
        valid = bool(deltas) and all(x is not None for x in deltas)
        result[f'mean_{metric}_delta'] = statistics.mean(deltas) if valid else None
        result[f'{metric}_delta_ci95'] = cluster_ci(deltas, [a[t]['station'] for t in ids]) if valid else None
    return result


def build_report(items: list[dict], baselines: list[dict] | None = None) -> dict:
    by = defaultdict(list)
    seen = set()
    for record in items:
        key = record['model'], record['task_id']
        if key in seen:
            raise ValueError(f'duplicate model/task across runs: {key}; selective reruns are not pooled')
        seen.add(key)
        by[record['model']].append(record)
    splits = {r['split'] for r in items}
    if len(splits) > 1:
        raise ValueError('report train and eval separately')
    models = [{'model': m, 'provider': m.split(':')[0], **summarize(rs)} for m,rs in sorted(by.items())]
    return {'models': models, 'split': next(iter(splits), None),
            'ci_method': 'task-weighted station-clustered percentile bootstrap; 2000 draws; seed 0; 95%; no CI below two clusters',
            'raw_savings_policy': 'gate failures retain prescore -1 USD; pct saved is 0 for gate failures',
            'baselines': baselines or [],
            'paired': [{'left': a, 'right': b, **paired(by[a], by[b])} for a,b in combinations(sorted(by), 2)]}


def markdown(report: dict) -> str:
    lines = ['# PermitForge evaluation', '', report['ci_method']+'.', '',
             'Uncalibrated reward is unavailable. Raw USD means include the -1 gate-failure sentinel.', '',
             '| Model | n | Reward (95% CI) | Feasible | USD saved | % saved | Evaluations | Turns | USD/episode |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    def fmt(x):
        return '—' if x is None else f'{x:.4f}'
    for row in report['models']+report.get('baselines', []):
        interval = row['ci95']
        reward = fmt(row['mean']) + (f' ({fmt(interval[0])}, {fmt(interval[1])})' if interval else '')
        lines.append(f"| {row.get('model', row.get('name'))} | {row['runs']} | {reward} | {row['feasible']:.2%} | "
                     f"{row['mean_usd_saved']:.2f} | {row['mean_pct_saved']:.2f} | {fmt(row['mean_evaluations'])} | "
                     f"{fmt(row['mean_turns'])} | {fmt(row['usd_per_episode'])} |")
    lines += ['', '| Model | Station | n | Reward | USD saved | Feasible |', '|---|---|---:|---:|---:|---:|']
    for row in report['models']:
        for station, s in row['per_station'].items():
            lines.append(f"| {row['model']} | {station} | {s['n']} | {fmt(s['reward'])} | {s['usd_saved']:.2f} | {s['feasible']:.2%} |")
        lines += ['', f"{row['model']}: ended without submission {row['ended_without_submit']}; "
                  f"categories `{json.dumps(row['failure_categories'], sort_keys=True)}`; "
                  f"coverage {row['station_clusters']} station clusters.", '']
    if report['paired']:
        lines += ['', '| Pair (left − right) | Matched tasks | USD saved difference | Reward difference |', '|---|---:|---:|---:|']
        for p in report['paired']:
            lines.append(f"| {p['left']} − {p['right']} | {p['n']} | {fmt(p['mean_usd_saved_delta'])} | {fmt(p['mean_reward_delta'])} |")
    if report.get('infrastructure_attempts'):
        lines += ['', 'Infrastructure attempts (excluded from graded metrics):', '',
                  '| Model | Task | Attempt | Retry on resume |', '|---|---|---:|---|']
        for attempt in report['infrastructure_attempts']:
            lines.append(f"| {attempt['model']} | {attempt['task_id']} | {attempt['attempt_number']} | "
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
            rec = {**episode, 'model': 'baseline:'+name, 'turns': 0, 'steps': [],
                   'evaluations_used': result.get('evaluations_used', result.get('evaluations', 0)),
                   'usage': {'usd': None}, 'final': {'valid': True, 'feasible': result['feasible'],
                   'prescore_usd_saved': result['usd_saved'], 'anchors': (episode.get('final') or {}).get('anchors') or episode.get('anchors', {})}}
            by_policy[name].append(rec)
    rows = [{'name': name, **summarize(rs)} for name,rs in sorted(by_policy.items()) if len(rs) == len(by_task)]
    return rows, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'board.md')
    parser.add_argument('--tasks-root', type=Path, default=ROOT/'envs/permit/tasks')
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--compute-baselines', action='store_true', help='run core policies using existing cached fields, then verify submissions')
    args = parser.parse_args()
    items = [r for _,r in records([run_path(r) for r in args.runs])]
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
    report['infrastructure_attempts'] = [r for run in args.runs for r in infrastructure(run_path(run))]
    report['missing_baselines'] = missing
    write_json(args.output.with_suffix('.json'), report)
    write_text(args.output, markdown(report) + ('\nBaseline coverage incomplete; see JSON missing_baselines.\n' if missing else ''))
    print(args.output)


if __name__ == '__main__':
    main()
