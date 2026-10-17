"""Historical WP2.7 cohort only: python -m eval.subsets --output results/subsets-wp2.7.json.

Use eval.select_tasks --report-only to reproduce the active WP2.8 cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .common import ROOT, write_json

PILOT_TASKS = (
    'c210d1d5da6f-s47', 'e1375aee7217-s47', '83c7c4c84cc2-s47', '9a462007d534-s47',
    '80b9af4e9eac-s47', '77a5c043bcde-s47', 'b13becc39ba1-s47', 'c90f3d90f654-s47',
    'f815d234ea99-s47', '5d5f74b72e74-s47',
)


def select_subsets(rows: list[dict], *, seed: int = 27) -> dict:
    result = {'seed': seed, 'repeats': 4}
    for split, quota, count in [('eval', 4, 20), ('train', 1, 10)]:
        pool = {r.get('task_id', r.get('id')): r for r in rows if r['split'] == split and r.get('eligible') is True}
        chosen = list(PILOT_TASKS) if split == 'eval' else []
        if set(chosen)-pool.keys():
            raise ValueError('pilot tasks must be eligible eval members')
        stations = Counter(pool[t]['station'] for t in chosen)
        if any(v > quota for v in stations.values()):
            raise ValueError('pilot exceeds station quota')
        while len(chosen) < count:
            sizes = {pool[t]['n_sources'] for t in chosen}
            types = {pool[t]['archetype'] for t in chosen}
            candidates = [t for t in pool if t not in chosen and stations[pool[t]['station']] < quota]
            if not candidates:
                raise ValueError('insufficient tasks for station quotas')
            def priority(t: str) -> tuple:
                row = pool[t]
                diversity = int(row['n_sources'] not in sizes) + int(row['archetype'] not in types)
                tie = hashlib.sha256(f'{seed}:{split}:{t}'.encode()).hexdigest()
                return -diversity, stations[row['station']], tie, t
            tid = min(candidates, key=priority)
            chosen.append(tid)
            stations[pool[tid]['station']] += 1
        selected = [{'task_id': t, **{k: pool[t][k] for k in ('station', 'n_sources', 'archetype')}} for t in chosen]
        selected.sort(key=lambda r: (r['station'], r['task_id']))
        if ({r['n_sources'] for r in selected} != set(range(2, 7))
                or len({r['archetype'] for r in selected}) != 5
                or set(stations.values()) != {quota}):
            raise ValueError('selection does not span source counts, archetypes and station quotas')
        result[split] = selected
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks-root', type=Path, default=ROOT/'envs/permit/tasks')
    parser.add_argument('--output', type=Path, default=ROOT/'results/subsets-wp2.7.json')
    parser.add_argument('--seed', type=int, default=27)
    args = parser.parse_args()
    if args.output.resolve() == (ROOT/'results/subsets.json').resolve():
        parser.error('historical selector cannot write the active cohort; use eval.select_tasks --report-only')
    paths = sorted(args.tasks_root.glob('permit-v2-*/tasks.jsonl'))
    rows = [json.loads(s) for p in paths for s in p.read_text().splitlines() if s.strip()]
    result = select_subsets(rows, seed=args.seed)
    result['inputs'] = {p.parent.name+'/tasks.jsonl': hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write_json(args.output, result)
    print(f"eval: {len(result['eval'])}; train: {len(result['train'])}; repeats: {result['repeats']}")


if __name__ == '__main__':
    main()
