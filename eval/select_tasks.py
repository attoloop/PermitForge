"""Model-free probes: python -m eval.select_tasks --field-jobs 12 --output results/subsets.json.

Resume uses immutable per-task probe files; --report-only recomputes selection without native work.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import fcntl
import json
from pathlib import Path
import statistics
import time
from typing import TYPE_CHECKING

from .common import ROOT, write_json

if TYPE_CHECKING:
    from permit_core.fields import FieldPack
    from permit_core.model import Scenario

SEEDS = tuple(range(8))
RULE = ('Station-stratified structural shortlist of 30 eligible tasks per split (6 per eval station, '
        '3 per train station), ordered by descending source count, near-binding draft standards '
        '(total/limit >= 0.8), uncontrolled-at-GEP upper-bound exceedance, then task ID. '
        'Probe draft, cheapest_first_greedy, v1_coordinate_descent, 60-second search, and random_30 '
        'with seeds 0..7 using ARM2 fast fields. Failed policies contribute zero savings. '
        'Hardness = 1 - greedy_saved/best_saved (zero if best_saved is zero); best includes every probe. '
        'Variance proxy is population standard deviation of the eight random saved fractions. '
        'Rank hard candidates by the product of within-split midrank percentiles of hardness and variance '
        '(then hardness, variance, ID). Reserve two easiest candidates from distinct stations first; '
        'fill eight hard slots subject to 2 tasks per eval station / 1 per train station. '
        'No model outcomes or pilot membership enter selection.')


def shortlist(rows: list[dict]) -> list[dict]:
    selected = []
    for split in ('eval', 'train'):
        pool = [r for r in rows if r['split'] == split and r.get('eligible') is True]
        stations = sorted({r['station'] for r in pool})
        quota = 30 // len(stations)
        for station in stations:
            candidates = sorted((r for r in pool if r['station'] == station),
                                key=lambda r: (-r['n_sources'], -r['near_binding'],
                                               -r['uncontrolled_exceedance'], r['task_id']))
            if len(candidates) < quota:
                raise ValueError('insufficient eligible tasks for structural shortlist')
            selected.extend(candidates[:quota])
    return selected


def metrics(draft_cost: float, policies: dict, random_results: list[dict]) -> dict:
    def saved(result: dict) -> float:
        return max(0., result['usd_saved']) if result['feasible'] else 0.
    best = max(saved(r) for r in [*policies.values(), *random_results])
    fractions = [saved(r)/draft_cost for r in random_results]
    return {'best_saved': best, 'greedy_saved': saved(policies['cheapest_first_greedy']),
            'best_saved_fraction': best/draft_cost,
            'hardness': 1-saved(policies['cheapest_first_greedy'])/best if best else 0.,
            'variance_proxy': statistics.pstdev(fractions), 'random_saved_fractions': fractions}


def choose(rows: list[dict]) -> list[dict]:
    if len({r['split'] for r in rows}) != 1:
        raise ValueError('choose one split at a time')
    quota = 2 if rows[0]['split'] == 'eval' else 1
    def percentile(value: float, field: str) -> float:
        return (sum(r[field] < value for r in rows) + .5*sum(r[field] == value for r in rows))/len(rows)
    ranked = [dict(r, hard_rank_score=percentile(r['hardness'], 'hardness') *
                   percentile(r['variance_proxy'], 'variance_proxy')) for r in rows]
    selected, counts = [], Counter()
    easy = sorted(ranked, key=lambda r: (r['hardness'], r['variance_proxy'], r['task_id']))
    for row in easy:
        if not counts[row['station']]:
            selected.append(dict(row, selection_group='easy'))
            counts[row['station']] += 1
            if len(selected) == 2:
                break
    ids = {r['task_id'] for r in selected}
    hard = sorted(ranked, key=lambda r: (-r['hard_rank_score'], -r['hardness'],
                                        -r['variance_proxy'], r['task_id']))
    for row in hard:
        if row['task_id'] not in ids and counts[row['station']] < quota:
            selected.append(dict(row, selection_group='hard'))
            counts[row['station']] += 1
            if len(selected) == 10:
                break
    if len(selected) != 10:
        raise ValueError('insufficient candidates for station quotas')
    return sorted(selected, key=lambda r: (r['station'], r['task_id']))


def structural_rows(tasks: Path, drafts: Path) -> list[dict]:
    """Native draft records supply binding statistics without new field work.

    The published manifest lacks exceedance/binding metrics. At GEP the uniform
    control ladder makes SO2/PM25 scale linearly; NO2 uses the conservative ARM2
    ratio bound .9/.5. This is a shortlist proxy, never a feasibility claim.
    """
    rows = []
    for path in sorted(tasks.glob('permit-v2-*/tasks.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get('eligible') is not True:
                continue
            draft_path = drafts/(row['id']+'.json')
            draft = json.loads(draft_path.read_text())
            data = json.loads((tasks/'data'/row['id']/'task.json').read_text())
            controls = [c for c in data['costs']['controls'] if c['id'].endswith('_maximum')]
            near, ratios = 0, {}
            for name, stat in draft['design_values'].items():
                pollutant = name.split('_')[0]
                efficiency = controls[0]['efficiency_by_pollutant'][pollutant]
                if any(c['efficiency_by_pollutant'][pollutant] != efficiency for c in controls):
                    raise ValueError('structural proxy requires the shared control ladder')
                value, background, limit = (float(stat[k]) for k in ('value', 'background', 'limit'))
                near += (value+background)/limit >= .8
                scale = (1.8 if pollutant == 'NO2' else 1)/(1-float(efficiency))
                ratios[name] = ((value+1e-5)*scale+background)/limit
            rows.append({**{k: row[k] for k in ('split', 'station', 'n_sources', 'archetype', 'eligible')},
                         'task_id': row['id'], 'near_binding': near,
                         'uncontrolled_exceedance': max(ratios.values())-1,
                         'uncontrolled_gep_upper_ratios': ratios,
                         'structure_source': str(draft_path.relative_to(ROOT)),
                         'structure_sha256': hashlib.sha256(draft_path.read_bytes()).hexdigest()})
    return rows


class ProbeEvaluator:
    """Cache identical designs and honor cancellation between evaluations."""
    def __init__(self, task: Scenario, pack: FieldPack, stop: Path | None = None) -> None:
        self.task, self.pack, self.stop = task, pack, stop
        self.evaluated: dict[str, dict] = {}

    def __call__(self, design: dict) -> dict:
        if self.stop is not None and self.stop.exists():
            raise InterruptedError('probe cancelled')
        from permit_core.baselines import canonical
        from permit_core.evaluate import evaluate
        key = canonical(design)
        if key not in self.evaluated:
            self.evaluated[key] = evaluate(self.task, design, self.pack)
        return self.evaluated[key]


def probe(tasks: Path, row: dict, cache: Path, *, field_jobs: int) -> dict:
    from permit_core.baselines import POLICIES, scenario_view
    from permit_core.fields import ensure_pack
    from permit_core.model import load_v2_instance
    from permit_core.search import search
    task = load_v2_instance(tasks, row['task_id'])
    start = time.monotonic()
    lease = ensure_pack(task, jobs=field_jobs)
    try:
        print(f'{task.id} fields ready', flush=True)
        view = scenario_view(task)
        fast = ProbeEvaluator(task, lease, cache/'.stop')
        policies = {}
        for name in ('draft', 'cheapest_first_greedy', 'v1_coordinate_descent'):
            policies[name] = POLICIES[name](view, fast)
            print(f'{task.id} {name} complete', flush=True)
        random_results = []
        for seed in SEEDS:
            random_results.append(dict(POLICIES['random_30'](view, fast, seed=seed), seed=seed))
            print(f'{task.id} random_30 seed {seed} complete', flush=True)
        # Selection is a fast-field diagnostic. The existing search's incumbent is
        # in this evaluation cache, so it needs no extra native final verification.
        policies['search'] = search(task, lease, seconds=60, evaluate_fn=fast, verified_results=fast.evaluated)
        policies['search'].update(verified=False, verification_source='fast-field selection probe')
        return {**row, 'draft_cost': view['draft_cost'], 'policies': policies,
                'random_30': random_results, **metrics(view['draft_cost'], policies, random_results),
                'field_identity': json.loads(lease.path.read_text())['identity'],
                'evaluated_designs': len(fast.evaluated),
                'seconds': time.monotonic()-start, 'grading': 'fast-field diagnostic, not model grade'}
    finally:
        lease.release(evict=True)


def validate_workers(jobs: int, field_jobs: int) -> None:
    if not 1 <= jobs <= 4 or not 1 <= field_jobs <= 12 or jobs*field_jobs > 12:
        raise ValueError('1..4 task workers; task workers times field jobs must be <=12')


def probe_cached(args: tuple) -> dict:
    tasks, row, cache, field_jobs, fingerprint, report_only = args
    if (cache/'.stop').exists():
        raise InterruptedError('probe cancelled')
    path = cache/(row['task_id']+'.json')
    if path.exists():
        result = json.loads(path.read_text())
        if result.get('fingerprint') != fingerprint:
            raise ValueError(f'stale probe {path}; use a fresh --cache')
        return result
    if report_only:
        raise ValueError(f'missing probe {path}')
    print('probe '+row['split']+' '+row['task_id'], flush=True)
    result = {**probe(tasks, row, cache, field_jobs=field_jobs), 'fingerprint': fingerprint}
    write_json(path, result)
    print(json.dumps({k: result[k] for k in ('task_id', 'hardness', 'variance_proxy', 'seconds')}), flush=True)
    return result


def run_probes(work: list[tuple], cache: Path, jobs: int) -> list[dict]:
    cache.mkdir(parents=True, exist_ok=True)
    with (cache/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (cache/'.stop').unlink(missing_ok=True)
        pool = ProcessPoolExecutor(max_workers=jobs)
        try:
            return list(pool.map(probe_cached, work))
        except BaseException:
            (cache/'.stop').touch()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks-root', type=Path, default=ROOT/'envs/permit/tasks')
    parser.add_argument('--drafts', type=Path, default=ROOT/'convert/cache/drafts-arm2')
    parser.add_argument('--cache', type=Path, default=ROOT/'internal/reports/WP2.8-evidence/difficulty-probes')
    parser.add_argument('--output', type=Path, default=ROOT/'results/subsets.json')
    parser.add_argument('--field-jobs', type=int, default=12)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    validate_workers(args.jobs, args.field_jobs)
    rows = structural_rows(args.tasks_root, args.drafts)
    candidates = shortlist(rows)
    inputs = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(args.tasks_root.glob('permit-v2-*/tasks.jsonl'))}
    code_paths = [Path(__file__), *sorted((ROOT/'envs/permit/core/permit_core').glob('*.py'))]
    fingerprint = hashlib.sha256(json.dumps({'inputs': inputs, 'structure': rows,
        'code': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}}, sort_keys=True).encode()).hexdigest()
    work = [(args.tasks_root, row, args.cache, args.field_jobs, fingerprint, args.report_only) for row in candidates]
    if args.report_only:
        results = list(map(probe_cached, work))
    else:
        results = run_probes(work, args.cache, args.jobs)
    output = {'protocol': 'WP2.8', 'repeats': 4, 'random_seeds': SEEDS, 'search_seconds': 60,
              'rule': RULE, 'inputs': inputs, 'fingerprint': fingerprint, 'structure': rows, 'probes': results,
              'structure_note': 'Manifest has no binding/exceedance metadata; native ARM2 draft records supply '
              'near-binding counts and a uniform-control uncontrolled-at-GEP bound (NO2 ratio bound .9/.5).',
              'selection_reproducibility': 'Selection is deterministic from frozen probe artifacts. '
              'A fresh wall-time-bounded search can visit a different number of nodes on a different host.'}
    for split in ('eval', 'train'):
        output[split] = choose([r for r in results if r['split'] == split])
    output['eval_card_selection'] = RULE
    write_json(args.output, output)


if __name__ == '__main__':
    main()
