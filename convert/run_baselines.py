"""Task-major on-demand fields, policies and search; no publishing.

uv run --project envs/permit/core python convert/run_baselines.py \
    --tasks envs/permit/tasks --jobs 1 --only TASK_ID --fields convert/cache/fields
uv run --project envs/permit/core python convert/run_baselines.py --tasks envs/permit/tasks --report

Per-task checkpoints retain completed verified policies if interrupted. Resume
requires unchanged task hashes, field bytes, code, native binaries and settings.
Search limits default to 60 seconds / 10000 nodes; --unlimited removes both.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import resource
import tempfile
import time
from typing import Any

from permit_core.baselines import (POLICIES, canonical, outcome, scenario_view, verify_evaluation)
from permit_core.evaluate import evaluate
from permit_core.fields import ensure_pack, load_pack
from permit_core.model import load_v2_instance
from permit_core.reward import reward
from permit_core.search import search
from permit_core.verify import models_dir

if __package__:
    from .reverify_drafts import input_sha256, cached_result, DEFAULT_CACHE as DRAFT_CACHE
    from .v1_to_v2 import check_row_integrity, read_rows, sha256, write_json
else:
    from reverify_drafts import input_sha256, cached_result, DEFAULT_CACHE as DRAFT_CACHE
    from v1_to_v2 import check_row_integrity, read_rows, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / 'convert/cache/baselines'
DEFAULT_SEED = 1402026


def cached_draft(row: dict, directory: Path = DRAFT_CACHE) -> dict | None:
    """Reuse the matching ARM2 admission proof, including native-binary identity."""
    try:
        result = cached_result(directory/f"{row['id']}.json", row)
    except ValueError:
        return None
    if result is None or result['binary_sha256'] != {
            name: sha256(models_dir()/name) for name in ('aermod', 'bpipprm')}:
        return None
    return {'valid': True, 'errors': result['errors'], 'feasible': result['feasible'],
            'cost': float(result['cost']), 'verification_source': str(directory/f"{row['id']}.json"),
            'standards': [{'id': name, **{k: float(v[k]) for k in ('value', 'background', 'limit', 'margin')},
                           'binding_receptor': {'index': v['receptor_index'], 'x_m': float(v['x']), 'y_m': float(v['y'])}}
                          for name, v in result['design_values'].items()]}


def fingerprint(row: dict, field_path: Path, settings: dict) -> str:
    core = ROOT / 'envs/permit/core/permit_core'
    paths = [Path(__file__), *sorted(core.glob('*.py'))]
    paths += [models_dir() / name for name in ('aermod', 'bpipprm')]
    field_hash = sha256(field_path)
    if field_path.name == 'pack.json':
        manifest = json.loads(field_path.read_text())
        field_hash = {'identity': manifest['identity'], 'sha256': manifest['sha256']}
    content = {'task': input_sha256(row), 'fields': field_hash, 'settings': settings,
               'code_and_binaries': {str(p): sha256(p) if p.is_file() else None for p in paths}}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def read_result(path: Path, row: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if (data['schema_version'] != 1 or data['id'] != row['id'] or data['split'] != row['split'] or
                data['task_sha256'] != input_sha256(row) or type(data['complete']) is not bool):
            raise ValueError('identity or input fingerprint mismatch')
        if data['complete'] and (set(data['baselines']) != set(POLICIES) or 'search' not in data):
            raise ValueError('missing completed policies/search')
        return data
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'invalid/stale baseline cache {path}: {exc}') from exc


def compute_task(tasks: Path, row: dict, field_path: Path, cache: Path, *,
                 seed: int, max_nodes: int | None, seconds: float | None,
                 progress: dict, checkpoint: Any, pack: Any = None) -> dict:
    errors = check_row_integrity(tasks, row)
    if errors:
        raise ValueError('; '.join(errors))
    task = load_v2_instance(tasks, row['id'])
    if pack is None:
        pack = load_pack(field_path)
    view = scenario_view(task, split=row['split'])
    progress.update(draft_cost=view['draft_cost'])
    verified = {canonical(p['design']): p['verification'] for p in progress['baselines'].values()
                if p.get('design') is not None and p.get('verified') and 'verification' in p}
    draft = cached_draft(row)
    if draft is not None:
        verified[canonical(view['draft_design'])] = draft
    evaluated = {}
    def fast(design: dict) -> dict:
        key = canonical(design)
        if key not in evaluated:
            evaluated[key] = evaluate(task, design, pack)
        return evaluated[key]
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{row['id']}-", dir=cache) as work:
        for name, policy in POLICIES.items():
            if name in progress['baselines']:
                continue
            result = policy(view, fast, **({'seed': seed} if name == 'random_30' else {}))
            if result['design'] is not None:
                key = canonical(result['design'])
                if key not in verified:
                    verified[key] = verify_evaluation(task, result['design'], Path(work))
                checked = verified[key]
                fast_feasible = result['feasible']
                result = {**outcome(view, result['design'], checked, result['evaluations_used']),
                          'verified': True, 'verification': checked, 'fast_feasible': fast_feasible}
            else:
                result['verified'] = False
            progress['baselines'][name] = result
            checkpoint(progress)
        progress['search'] = search(task, pack, max_nodes=max_nodes, seconds=seconds, workdir=work,
                                    evaluate_fn=fast, verified_results=verified)
    progress['complete'] = True
    return progress


def run_one(tasks: Path, row: dict, field_path: Path | None, cache: Path, *,
            seed: int = DEFAULT_SEED, max_nodes: int | None = 10000,
            seconds: float | None = 60, field_jobs: int = 1, evict_after: bool = False,
            pack: Any = None) -> dict:
    errors = check_row_integrity(tasks, row)
    if errors:
        raise ValueError('; '.join(errors))
    if field_path is None:
        lease = ensure_pack(load_v2_instance(tasks, row['id']), jobs=field_jobs)
        try:
            return run_one(tasks, row, lease.path, cache, seed=seed, max_nodes=max_nodes,
                           seconds=seconds, pack=lease)
        finally:
            lease.release(evict=evict_after)
    settings = {'seed': seed, 'max_nodes': max_nodes, 'seconds': seconds}
    key = fingerprint(row, field_path, settings)
    path = cache / f"{row['id']}.json"
    progress = read_result(path, row)
    if progress is not None:
        if progress.get('input_sha256') != key:
            raise ValueError(f'stale baseline cache {path}: inputs/settings changed; use a new --cache directory')
        if progress['complete']:
            return progress
    else:
        progress = {'schema_version': 1, 'id': row['id'], 'split': row['split'],
                    'task_sha256': input_sha256(row), 'input_sha256': key, 'settings': settings,
                    'field_path': str(field_path.resolve()), 'baselines': {}, 'complete': False}
    def checkpoint(data: dict) -> None:
        data.update(input_sha256=key, settings=settings)
        write_json(path, data)
    result = compute_task(tasks, row, field_path, cache, seed=seed, max_nodes=max_nodes, seconds=seconds,
                          progress=progress, checkpoint=checkpoint, **({'pack': pack} if pack is not None else {}))
    checkpoint(result)
    return result


def field_path(tasks: Path, row: dict, fields: Path | None = None) -> Path | None:
    candidates = ([fields / f"{row['id']}.npz", fields / row['id'] / 'float64.npz'] if fields else
                  [tasks / f"permit-v2-{row['split']}" / 'fields' / f"{row['id']}.npz"])
    return next((p for p in candidates if p.is_file()), None)


def _worker(args: tuple) -> dict:
    tasks, row, field, cache, settings = args
    started = time.monotonic()
    def cpu() -> float:
        return sum(r.ru_utime+r.ru_stime for r in (
            resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))
    cpu_start = cpu()
    result = run_one(tasks, row, field, cache, **settings)
    result['run_timing'] = {'wall_seconds': time.monotonic()-started, 'cpu_seconds': cpu()-cpu_start}
    write_json(cache/f"{row['id']}.json", result)
    return {'id': row['id'], 'closed': result['search']['closed'], **result['run_timing']}


def run_tasks(tasks: Path, cache: Path, *, jobs: int = 1, only: list[str] | None = None,
              fields: Path | None = None, seed: int = DEFAULT_SEED,
              max_nodes: int | None = 10000, seconds: float | None = 60,
              field_jobs: int = 1, evict_after: bool = False) -> dict:
    if not 1 <= jobs <= 12 or not 1 <= field_jobs <= 12 or jobs*field_jobs > 12:
        raise ValueError('--jobs and --field-jobs must be positive with product at most 12')
    rows = read_rows(tasks)
    if only is not None and set(only)-{r['id'] for r in rows}:
        raise ValueError('unknown --only task ids: '+', '.join(sorted(set(only)-{r['id'] for r in rows})))
    rows = [r for r in rows if r.get('eligible') is True and (only is None or r['id'] in only)]
    pending, missing = [], []
    settings = {'seed': seed, 'max_nodes': max_nodes, 'seconds': seconds,
                'field_jobs': field_jobs, 'evict_after': evict_after}
    for row in rows:
        path = field_path(tasks, row, fields)
        if path is None and (fields is not None or not (tasks/'data'/row['id']/'task.json').exists()):
            missing.append(row['id'])
        else:
            pending.append((tasks, row, path, cache, settings))
    completed = []
    if pending:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for result in pool.map(_worker, pending):
                completed.append(result)
                print(json.dumps(result), flush=True)
    return {'selected': len(rows), 'completed': len(completed), 'missing_fields': missing}


def report(rows: list[dict], cache: Path, output: Path) -> None:
    lines = ['# PermitForge baselines', '',
             'Best-known search cost uses the supplied fast fields and a final ARM2 native check. '
             'Search and baselines are diagnostics, independent of admission and reward calibration.', '',
             'Means cover completed tasks only; missing tasks are excluded and counted. '
             'Infeasible/no-submit savings use sentinel -1 in JSON and contribute zero USD and zero % to means. '
             'Mean % saved is the mean of per-task percentages, not a ratio of sums. '
             'Evaluation counts exclude final native verification. Search seconds include it.', '',
             'Reward anchors come only from model calibration on each task in both splits; '
             'rewards remain pending until those anchors exist. Reward means require anchors for every included task.', '',
             'Reproduce: `uv run --project envs/permit/core python convert/run_baselines.py '
             '--tasks envs/permit/tasks --report`.', '']
    for split in ('train', 'eval'):
        selected = [r for r in rows if r['split'] == split]
        available = [(r, d) for r in selected if (d := read_result(cache / f"{r['id']}.json", r))
                     and d['complete']]
        lines += [f'## {split}', '', f'Completed tasks: {len(available)} / {len(selected)}; '
                  f'missing or incomplete: {len(selected)-len(available)}.', '',
                  '| policy | mean USD saved | mean % saved | feasible rate | mean reward | mean cost (feasible) |',
                  '|---|---:|---:|---:|---:|---:|']
        for name in [*POLICIES, 'search']:
            savings, percentages, feasible_costs, rewards = [], [], [], []
            for row, data in available:
                result = data['search'] if name == 'search' else data['baselines'][name]
                feasible = result['feasible'] and result.get('verified', False)
                saved = result['usd_saved'] if feasible else 0
                savings.append(saved)
                percentages.append(100*saved/data['draft_cost'])
                if feasible:
                    feasible_costs.append(result['cost'])
                anchors = row.get('anchors')
                if anchors and anchors.get('source') in {'panel', 'calibration'}:
                    rewards.append(reward(result['usd_saved'] if feasible else -1, anchors))
            def mean(values: list) -> str:
                return f'{sum(values)/len(values):.4f}' if values else 'pending'
            label = 'best-known search' if name == 'search' else name
            rate = f'{len(feasible_costs)/len(available):.4f}' if available else 'pending'
            calibrated = mean(rewards) if len(rewards) == len(available) else 'pending'
            lines.append(f'| {label} | {mean(savings)} | {mean(percentages)} | {rate} | '
                         f'{calibrated} | {mean(feasible_costs)} |')
        closed = sum(d['search']['closed'] for _, d in available)
        lines += ['', f'Search trees closed: {closed} / {len(available)}. '
                  'Closed describes the fast-field search; final feasibility is checked by the native verifier.', '']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines)+'\n')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, default=Path('envs/permit/tasks'))
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--fields', type=Path)
    parser.add_argument('--field-jobs', type=int, default=1)
    parser.add_argument('--evict-after', action='store_true', help='release and evict each task pack after policies/search')
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--max-nodes', type=int, default=10000)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--unlimited', action='store_true')
    parser.add_argument('--report', action='store_true', help='report existing caches only; launch no evaluations')
    parser.add_argument('--output', type=Path, default=Path('results/baselines.md'))
    args = parser.parse_args()
    try:
        if args.report:
            report(read_rows(args.tasks), args.cache, args.output)
        else:
            result = run_tasks(args.tasks, args.cache, jobs=args.jobs, only=args.only, fields=args.fields,
                               seed=args.seed, max_nodes=None if args.unlimited else args.max_nodes,
                               seconds=None if args.unlimited else args.seconds,
                               field_jobs=args.field_jobs, evict_after=args.evict_after)
            print(json.dumps(result, indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
