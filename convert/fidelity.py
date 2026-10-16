"""Native fidelity gate: uv run --project envs/permit/core python convert/fidelity.py --jobs 4.

Resumes native comparisons under convert/cache/fields. Use --probe for one field,
--prepare for the six-task field precompute, --recheck 10 for the slow regression.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from decimal import Decimal
import json
import multiprocessing
import os
from pathlib import Path
import random
import statistics
import sys
import time

from permit_core.evaluate import evaluate
from permit_core.fields import (FieldPack, _cached_field, cache_root, cached_pack, height_key,
                                load_pack, precompute, certified_prune, save_pack)
from permit_core.menu import height_menu
from permit_core.model import extraction_root, load_v1_instance
from permit_core.verify import verify

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'results/evidence/fidelity.json'
SEED = 1302026


LATENCY_LIMIT_S = 1.0


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix('.tmp').write_text(json.dumps(data, indent=2, default=str, allow_nan=False)+'\n')
    path.with_suffix('.tmp').replace(path)


def inventory() -> list[dict]:
    rows = json.loads((extraction_root()/'family/manifest.json').read_text())['instances']
    result = []
    for row in rows:
        task = load_v1_instance(row['id'])
        result.append({'id': task.id, 'station': task.station,
                       'split': next(t.split(':')[1] for t in row['coverage_tags'] if t.startswith('split:')),
                       'archetype': task.task['scenario']['facility']['archetype'],
                       'sources': len(task.sources),
                       'fields': sum(len(height_menu(s.height_min_m, s.gep_height_m)) for s in task.sources)})
    return result


def select_tasks(rows: list[dict]) -> list[dict]:
    selected = []
    for archetype in sorted({r['archetype'] for r in rows}):
        candidates = [r for r in rows if r['archetype'] == archetype]
        selected.append(min(candidates, key=lambda r: (r['sources'], r['station'] in
                                                      {s['station'] for s in selected}, r['id'])))
    selected.append(min((r for r in rows if r['sources'] == 6 and r not in selected),
                        key=lambda r: (r['station'] in {s['station'] for s in selected}, r['id'])))
    if len({r['station'] for r in selected}) < 3:
        raise ValueError('selection needs at least three stations')
    return selected


def native_case(args: tuple[str, int, dict, bool]) -> dict:
    task_id, index, design, force = args
    path = cache_root()/task_id/'comparisons'/f'{index:02}.json'
    if path.exists() and not force:
        result = json.loads(path.read_text())
        if result['design'] != design:
            raise ValueError('cached fidelity design changed')
        return result
    task = load_v1_instance(task_id)
    ledger = {sid: {'control': row['control'], 'stack_height_m': row['height_m']} for sid, row in design.items()}
    result = verify(task, ledger, cache_root()/task_id/'verify', no2_tier='ARM2')
    data = {'task_id': task_id, 'index': index, 'design': design, 'verification': asdict(result)}
    write_json(path, data)
    return json.loads(path.read_text())


def public_draft(task) -> dict:
    return {sid: {'control': row['control'], 'height_m': float(row['stack_height_m'])}
            for sid, row in task.draft_design.items()}


def designs_for(task, pack: FieldPack) -> list[dict]:
    rng = random.Random(SEED + int(task.id.split('-')[0], 16))
    menus = {s.id: height_menu(s.height_min_m, s.gep_height_m) for s in task.sources}
    choices = {s.id: [c.id for c in task.controls if s.id in c.applies_to] for s in task.sources}
    designs = [public_draft(task),
               {s.id: {'control': 'none', 'height_m': float(s.gep_height_m)} for s in task.sources},
               {s.id: {'control': s.id+'_moderate', 'height_m': float(s.height_min_m)} for s in task.sources}]
    candidates = [{s.id: {'control': rng.choice(choices[s.id]), 'height_m': float(rng.choice(menus[s.id]))}
                   for s in task.sources} for _ in range(80)]
    # Explicitly include menu designs closest to either side of the feasibility boundary.
    ranked = []
    seen = {json.dumps(design, sort_keys=True) for design in designs}
    for candidate in candidates:
        key = json.dumps(candidate, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result = evaluate(task, candidate, pack)
        margin = min(s['margin']/s['limit'] for s in result['standards'])
        ranked.append((margin, candidate))
    boundary = (sorted((r for r in ranked if r[0] >= 0), key=lambda r: r[0])[:4] +
                sorted((r for r in ranked if r[0] < 0), key=lambda r: -r[0])[:3])
    for _, candidate in boundary + sorted(ranked, key=lambda r: abs(r[0])):
        if len(designs) >= 10:
            break
        if candidate not in designs:
            designs.append(candidate)
    for candidate in candidates:
        if candidate not in designs:
            designs.append(candidate)
        if len(designs) == 40:
            return designs
    raise ValueError('candidate pool lacks forty unique designs')


def prepare(selected: list[dict], jobs: int) -> None:
    with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
        checks = list(pool.map(native_case, [(r['id'], 0, public_draft(load_v1_instance(r['id'])), False)
                                            for r in selected]))
    for check in checks:
        if not check['verification']['feasible']:
            raise ValueError(f"selected draft ineligible: {check['task_id']}")
    precompute([r['id'] for r in selected], jobs)


def run_gate(selected: list[dict], jobs: int, *, recheck: int = 0) -> dict:
    report = {'seed': SEED, 'command': 'uv run --project envs/permit/core python convert/fidelity.py --jobs 4',
              'status': 'IN_PROGRESS', 'selected': selected, 'tasks': [], 'comparisons': [], 'failures': []}
    for row in selected:
        print(f"Comparing {row['id']}", flush=True)
        task = load_v1_instance(row['id'])
        full_pack = cached_pack(task)
        pack, pruning = certified_prune(task, full_pack)
        designs = designs_for(task, pack)
        paths = {dtype: cache_root()/task.id/f'{dtype}.npz' for dtype in ('float32', 'float64')}
        for dtype, path in paths.items():
            if not path.exists():
                save_pack(pack, path, dtype=dtype)
        print(f"Packs ready: {task.id}", flush=True)
        float32_pack = load_pack(paths['float32'])
        single_thread_cpu = min(os.sched_getaffinity(0))
        old_affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {single_thread_cpu})
        try:
            evaluate(task, designs[0], pack)
            times = []
            fast = []
            float32 = []
            for design in designs:
                start = time.perf_counter()
                fast.append(evaluate(task, design, pack))
                times.append(time.perf_counter()-start)
                float32.append(evaluate(task, design, float32_pack))
        finally:
            os.sched_setaffinity(0, old_affinity)
        chosen = list(enumerate(designs))[:recheck or 40]
        print(f"Native comparisons: {task.id}, count={len(chosen)}", flush=True)
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
            native = list(pool.map(native_case, [(task.id, i, d, bool(recheck)) for i, d in chosen]))
        changed = sum(a['value'] != b['value'] for f64, f32 in zip(fast, float32)
                      for a, b in zip(f64['standards'], f32['standards']))
        task_report = {**row, 'pruning': pruning, 'kept_receptors': len(pack.receptor_indices),
                       'pack_bytes': {k: p.stat().st_size for k, p in paths.items()},
                       'float32_changed_standard_values': changed, 'distribution_dtype': 'float64' if changed else 'float32',
                       'evaluation_seconds': {'median': statistics.median(times), 'max': max(times), 'samples': times},
                       'field_native_seconds': [json.loads(p.read_text())['native_seconds']
                                                for p in sorted((cache_root()/task.id).glob('S*/*.json'))]}
        report['tasks'].append(task_report)
        for case in native:
            index = case['index']
            verified = case['verification']
            errors = []
            stats = []
            for standard in fast[index]['standards']:
                name = standard['id']
                if name not in verified['design_values']:
                    errors.append(f'missing native {name}: {verified["errors"]}')
                    continue
                expected = Decimal(verified['design_values'][name]['value'])
                actual = Decimal(str(standard['value']))
                absolute = abs(actual-expected)
                relative = absolute/abs(expected) if expected else (Decimal(0) if not absolute else Decimal('Infinity'))
                stats.append({'id': name, 'fast': str(actual), 'native': str(expected),
                              'absolute_error': str(absolute), 'relative_error': str(relative)})
                if actual != expected and not (name.startswith('NO2') and relative <= Decimal('1e-5')):
                    errors.append(f'{name}: {actual} != {expected}')
            if fast[index]['feasible'] != verified['feasible']:
                errors.append('feasibility disagreement')
            report['comparisons'].append({'task_id': task.id, 'index': index, 'design': designs[index],
                                          'standards': stats, 'errors': errors,
                                          'fast_feasible': fast[index]['feasible'], 'native_feasible': verified['feasible']})
            report['failures'].extend(f'{task.id}/{index}: {e}' for e in errors)
        # Latency criterion: <= 1 s per evaluation is what agent tool calls and RL
        # need; The fidelity sample measured ~0.32 s (full AERMOD takes 60-150 s). See DESIGN.md §Fidelity.
        if max(times) >= LATENCY_LIMIT_S:
            report['failures'].append(f'{task.id}: evaluation exceeds {LATENCY_LIMIT_S} s')
        report['status'] = 'NO-GO' if report['failures'] else 'IN_PROGRESS'
        write_json(REPORT, report)
        if report['failures']:
            break
        if recheck:
            break
        del float32_pack, pack
        load_pack.cache_clear()
    report['status'] = 'NO-GO' if report['failures'] else ('RECHECK_PASS' if recheck else 'GO')
    report['summary'] = summarize(report)
    write_json(REPORT, report)
    return report


def summarize(report: dict) -> dict:
    standards = {}
    for case in report['comparisons']:
        for std in case['standards']:
            current = standards.setdefault(std['id'], {'comparisons': 0, 'exact': 0,
                                                       'max_absolute_error': 0.0, 'max_relative_error': 0.0})
            current['comparisons'] += 1
            current['exact'] += Decimal(std['fast']) == Decimal(std['native'])
            current['max_absolute_error'] = max(current['max_absolute_error'], float(std['absolute_error']))
            current['max_relative_error'] = max(current['max_relative_error'], float(std['relative_error']))
    inventory_rows = json.loads((cache_root()/'selection.json').read_text())['inventory']
    timings = [json.loads(p.read_text())['native_seconds'] for r in report['selected']
               for p in (cache_root()/r['id']).glob('S*/*.json')]
    bytes_per_field = [t['pack_bytes'][t['distribution_dtype']]/t['fields'] for t in report['tasks']]
    rows_path = ROOT/'envs/permit/tasks'
    eligibility = {}
    for path in rows_path.glob('permit-v2-*/tasks.jsonl'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            eligibility[row['id']] = row.get('eligible')
    projections = {}
    for split in ('eval', 'all'):
        rows = [r for r in inventory_rows if split == 'all' or r['split'] == split]
        known = [r for r in rows if eligibility.get(r['id']) is True]
        pending = [r for r in rows if eligibility.get(r['id']) is None]
        candidate = [r for r in rows if eligibility.get(r['id']) is not False]
        projections[split] = {}
        for label, subset in [('confirmed_eligible', known), ('including_pending_upper_bound', candidate)]:
            count = sum(r['fields'] for r in subset)
            projections[split][label] = {'tasks': len(subset), 'fields': count,
                'projected_serial_native_hours': count*statistics.mean(timings)/3600,
                'raw_field_bytes': count*43848*184*8,
                'projected_pack_bytes': count*statistics.mean(bytes_per_field)}
        projections[split]['pending_tasks'] = len(pending)
    return {'standards': standards, 'comparisons': len(report['comparisons']),
            'feasibility_agreements': sum(c['fast_feasible'] == c['native_feasible'] for c in report['comparisons']),
            'precomputed_fields': len(timings), 'native_field_seconds': {'mean': statistics.mean(timings),
                'min': min(timings), 'max': max(timings), 'sum': sum(timings)},
            'projections': projections,
            'projection_method': 'Native wall seconds per field (shared host), used as conservative serial CPU-hour proxy; '
                'compressed bytes per field extrapolated from measured packs. Pending admission is not counted as eligible.'}


def precision_report(report: dict) -> dict:
    """Expand already-measured float32 differences; no native launches or new designs."""
    result = {}
    for row in report['tasks']:
        task = load_v1_instance(row['id'])
        pack = load_pack(cache_root()/task.id/'float32.npz')
        standards = {}
        changes = []
        for case in report['comparisons']:
            if case['task_id'] != task.id:
                continue
            measured = {s['id']: s['value'] for s in evaluate(task, case['design'], pack)['standards']}
            for ref in case['standards']:
                std = standards.setdefault(ref['id'], {'compared': 0, 'changed': 0, 'max_absolute_error': 0.0})
                error = abs(Decimal(str(measured[ref['id']]))-Decimal(ref['fast']))
                std['compared'] += 1
                std['changed'] += bool(error)
                std['max_absolute_error'] = max(std['max_absolute_error'], float(error))
                if error:
                    changes.append({'design_index': case['index'], 'standard': ref['id'],
                                    'float64': ref['fast'], 'float32': str(measured[ref['id']])})
        result[task.id] = {'standards': standards, 'changes': changes}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--recheck', type=int, default=0)
    parser.add_argument('--inspect', metavar='TASK_ID')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--precision-report', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        parser.error('jobs must be between 1 and 8')
    if args.precision_report:
        report = json.loads(REPORT.read_text())
        report['precision'] = precision_report(report)
        write_json(REPORT, report)
        print(json.dumps(report['precision'], indent=2))
        return 0
    if args.summarize:
        report = json.loads(REPORT.read_text())
        report['summary'] = summarize(report)
        write_json(REPORT, report)
        print(json.dumps(report['summary'], indent=2))
        return 0
    rows = inventory()
    selected = select_tasks(rows)
    write_json(cache_root()/'selection.json', {'selected': selected, 'inventory': rows})
    print(json.dumps(selected), flush=True)
    if args.inspect:
        task = load_v1_instance(args.inspect)
        pack, pruning = certified_prune(task, cached_pack(task))
        start = time.perf_counter()
        evaluation = evaluate(task, public_draft(task), pack)
        print(json.dumps({'pruning': pruning, 'evaluation_seconds': time.perf_counter()-start,
                          'evaluation': evaluation}, default=str), flush=True)
        return 0
    if args.probe:
        task = load_v1_instance(selected[0]['id'])
        source = task.sources[0]
        print(json.dumps(_cached_field((task.id, source.id, height_key(source.gep_height_m)))), flush=True)
        return 0
    prepare(selected, args.jobs)
    if args.prepare:
        return 0
    report = run_gate(selected, args.jobs, recheck=args.recheck)
    print(json.dumps({'status': report['status'], 'failures': report['failures']}), flush=True)
    return int(report['status'] == 'NO-GO')


if __name__ == '__main__':
    sys.exit(main())
