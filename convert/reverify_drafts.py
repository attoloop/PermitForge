"""Verify ARM2 drafts (run from the repository root, using permit-core's uv project).

uv run --project envs/permit/core python convert/reverify_drafts.py \
    --tasks envs/permit/tasks --jobs 3 --only ID ID ID
uv run --project envs/permit/core python convert/reverify_drafts.py \
    --tasks envs/permit/tasks --finalize
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, is_dataclass
from decimal import Decimal
import hashlib
import os
import json
from pathlib import Path
import tempfile
from typing import Any, Callable

if __package__:
    from .v1_to_v2 import (Row, check_row_integrity, read_rows, write_json, write_manifests,
                          write_rows)
else:
    from v1_to_v2 import (Row, check_row_integrity, read_rows, write_json, write_manifests,
                         write_rows)

DEFAULT_CACHE = Path(__file__).resolve().parent / 'cache/drafts-arm2'


def input_sha256(row: Row) -> str:
    inputs = {key: row[key] for key in ('id', 'station', 'sha256', 'met_sha256')}
    inputs['no2_tier'] = 'ARM2'
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def cached_result(path: Path, row: Row) -> Row | None:
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text())
        required = {'feasible', 'design_values', 'margins', 'native_seconds', 'errors',
                    'binary_sha256', 'failed_standards'}
        if (not required <= result.keys() or result['id'] != row['id'] or
                result['input_sha256'] != input_sha256(row) or result['no2_tier'] != 'ARM2' or
                type(result['feasible']) is not bool):
            raise ValueError('input fingerprint, identity or result schema mismatch')
        return result
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f'invalid/stale cache {path}: {exc}; remove it to re-run this task') from exc


def jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if is_dataclass(value):
        return jsonable(asdict(value))
    if hasattr(value, '__dict__'):
        return jsonable(vars(value))
    if isinstance(value, dict):
        return {key: jsonable(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def reverify_one(tasks: Path, row: Row, cache: Path, *, verify_fn: Callable | None = None) -> Row:
    errors = check_row_integrity(tasks, row)
    if errors:
        raise ValueError('; '.join(errors))
    path = cache / f"{row['id']}.json"
    existing = cached_result(path, row)
    if existing is not None:
        return existing
    from permit_core.model import load_v2_instance
    if verify_fn is None:
        from permit_core.verify import verify
        verify_fn = verify
    scenario = load_v2_instance(tasks, row['id'])
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{row['id']}-", dir=cache) as workdir:
        verification = verify_fn(scenario, scenario.draft_design, Path(workdir), no2_tier='ARM2')
    values = jsonable(verification.design_values)
    result = {'id': row['id'], 'no2_tier': 'ARM2', 'input_sha256': input_sha256(row),
              'feasible': verification.feasible, 'design_values': values,
              'margins': {s: v['margin'] for s, v in values.items()},
              'failed_standards': sorted(s for s, v in values.items() if Decimal(v['margin']) < 0),
              'native_seconds': verification.native_seconds, 'errors': verification.errors,
              'gep': jsonable(verification.gep), 'cost': jsonable(verification.cost),
              'binary_sha256': verification.binary_sha256}
    write_json(path, result)
    return result


def finalize(tasks: Path, cache: Path) -> Row:
    """Apply only completed, matching results; leave unverified tasks pending."""
    rows = read_rows(tasks)
    decisions = []
    for row in rows:
        errors = check_row_integrity(tasks, row)
        if errors:
            raise ValueError('; '.join(errors))
        result = cached_result(cache / f"{row['id']}.json", row)
        if result is None:
            continue
        feasible = result['feasible']
        reason = 'draft passes ARM2' if feasible else '; '.join([
            *result['failed_standards'],
            *(f"{e['code']}: {e['detail']}" for e in result['errors'])]) or 'draft verification failed'
        if row['eligible'] != feasible or row['eligibility_reason'] != reason:
            decisions.append({'id': row['id'], 'split': row['split'], 'decision': 'ARM2 draft verified',
                              'eligible': feasible, 'reason': reason,
                              'failed_standards': result['failed_standards'],
                              'input_sha256': result['input_sha256'],
                              'native_seconds': result['native_seconds']})
        row.update(eligible=feasible, eligibility_reason=reason,
                   failing_standards=result['failed_standards'])
    summary = write_manifests(tasks, rows)
    for path, selected in [(tasks / 'build_log.jsonl', decisions),
                           *((tasks / f'permit-v2-{s}/build_log.jsonl',
                              [d for d in decisions if d['split'] == s]) for s in ('train', 'eval'))]:
        old = [json.loads(s) for s in path.read_text().splitlines()]
        write_rows(path, old + selected)
    return summary


def run_drafts(tasks: Path, cache: Path, *, jobs: int, only: list[str] | None = None) -> Row:
    """Bounded process pool: each worker launches native cases sequentially."""
    if not 1 <= jobs <= (os.cpu_count() or 1):
        raise ValueError(f'--jobs must be between 1 and {os.cpu_count() or 1}')
    rows = read_rows(tasks)
    if only is not None and set(only) - {r['id'] for r in rows}:
        raise ValueError('unknown --only task ids: ' + ', '.join(sorted(set(only) - {r['id'] for r in rows})))
    selected = [r for r in rows if only is None or r['id'] in only]
    pending, completed = [], []
    for row in selected:
        errors = check_row_integrity(tasks, row)
        if errors:
            raise ValueError('; '.join(errors))
        existing = cached_result(cache / f"{row['id']}.json", row)
        if existing is None:
            pending.append(row)
        else:
            completed.append(existing)
    # Count against the whole original pack, including results outside --only.
    failed = sum((r := cached_result(cache / f"{row['id']}.json", row)) is not None
                 and not r['feasible'] for row in rows)
    decision_required = failed > len(rows) / 10
    iterator = iter(pending)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        active = {}
        def submit_next() -> None:
            row = next(iterator, None)
            if row is not None:
                active[pool.submit(reverify_one, tasks, row, cache)] = row['id']
        if not decision_required:
            for _ in range(jobs):
                submit_next()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                del active[future]
                result = future.result()
                completed.append(result)
                failed += not result['feasible']
                print(json.dumps({'id': result['id'], 'feasible': result['feasible'],
                                  'failed_standards': result['failed_standards'],
                                  'native_seconds': result['native_seconds']}), flush=True)
            decision_required = failed > len(rows) / 10
            if not decision_required:
                for _ in done:
                    submit_next()
    return {'selected': len(selected), 'cached': len(selected) - len(pending),
            'completed': len(completed), 'decision_required': decision_required}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, default=Path('envs/permit/tasks'))
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--finalize', action='store_true', help='update records from cache without launching models')
    args = parser.parse_args()
    if args.finalize and args.only:
        parser.error('--finalize applies all cached results; omit --only')
    try:
        result = (finalize(args.tasks, args.cache) if args.finalize else
                  run_drafts(args.tasks, args.cache, jobs=args.jobs, only=args.only))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')
    print(json.dumps(result, indent=2, sort_keys=True))
    if result['decision_required']:
        print('STOP: more than 10% of the pack failed ARM2 draft verification; operator decision required.')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
