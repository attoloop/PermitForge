"""Convert v1: uv run python convert/v1_to_v2.py --src convert/v1-src --out envs/permit/tasks.

Check all data/met hashes: uv run python convert/v1_to_v2.py --out envs/permit/tasks --check.
"""

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from permit_core.menu import height_menu

Row = dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def write_rows(path: Path, rows: list[Row]) -> None:
    write_text(path, ''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in rows))


def read_rows(root: Path) -> list[Row]:
    rows = [json.loads(line) for split in ('train', 'eval')
            for line in (root / f'permit-v2-{split}/tasks.jsonl').read_text().splitlines()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate task ids')
    return sorted(rows, key=lambda r: r['id'])


def counts(rows: list[Row]) -> Row:
    eligibility = Counter('pending' if r['eligible'] is None else
                          'eligible' if r['eligible'] else 'ineligible' for r in rows)
    return {'total': len(rows),
            **{key: dict(sorted(Counter(str(r[key]) for r in rows).items()))
               for key in ('split', 'station', 'archetype', 'n_sources')},
            'eligibility': {key: eligibility[key] for key in ('eligible', 'ineligible', 'pending')}}


def write_manifests(root: Path, rows: list[Row]) -> Row:
    rows = [{k: v for k, v in row.items() if k not in
             {'anchors', 'anchor_source', 'anchors_frozen_at', 'anchor_runs'}} for row in rows]
    failing = Counter(s for r in rows for s in r.get('failing_standards', []))
    summary = {'counts': counts(rows), 'failing_standards': dict(sorted(failing.items())),
               'eligibility_by_split': {s: counts([r for r in rows if r['split'] == s])['eligibility']
                                        for s in ('train', 'eval')},
               'decision_required': sum(r['eligible'] is False for r in rows) > len(rows) / 10}
    write_json(root / 'manifest.json', summary)
    for split in ('train', 'eval'):
        subset = [r for r in rows if r['split'] == split]
        write_rows(root / f'permit-v2-{split}/tasks.jsonl', subset)
        write_json(root / f'permit-v2-{split}/manifest.json', {'counts': counts(subset)})
    return summary


def file_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha256(p)
            for p in sorted(root.rglob('*')) if p.is_file()}


def check_row_integrity(root: Path, row: Row) -> list[str]:
    errors = []
    actual = file_hashes(root / 'data' / row['id'])
    for name in sorted(actual.keys() | row['sha256'].keys()):
        if actual.get(name) != row['sha256'].get(name):
            errors.append(f"{row['id']}/{name}: data sha256 mismatch or missing/unrecorded file")
    for name, expected in row['met_sha256'].items():
        path = root / 'met' / row['station'] / name
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"{row['station']}/{name}: met sha256 mismatch or missing file")
    if row.get('height_menu') is not None:
        task = json.loads((root/'data'/row['id']/'task.json').read_text(), parse_float=Decimal)
        expected = {u['id']: height_menu(u['stack_bounds']['height_min_m'], u['stack_bounds']['gep_height_m'])
                    for u in task['scenario']['units']}
        for label, menus in [('record', row['height_menu']), ('task', task['height_menu'])]:
            if {sid: [Decimal(str(h)) for h in heights] for sid, heights in menus.items()} != expected:
                errors.append(f"{row['id']}: {label} height menu differs from shared height_menu")
    return errors


def check_integrity(root: Path) -> list[str]:
    return [error for row in read_rows(root) for error in check_row_integrity(root, row)]


def convert(src: Path, out: Path) -> Row:
    """Materialize a new pack from the preserved v1 extraction."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'conversion output must be new or empty: {out}')
    extracted = json.loads((src / 'EXTRACT_MANIFEST.json').read_text())
    published = json.loads((src / 'family/manifest.json').read_text())['instances']
    if len({r['id'] for r in published}) != len(published):
        raise ValueError('duplicate ids in v1 manifest')
    rows, decisions, stations = [], [], {}
    for instance in sorted(published, key=lambda r: r['id']):
        ident = instance['id']
        public = src / 'instances' / ident / 'task'
        task = json.loads((public / 'task.json').read_text())
        station = instance['equivalence_key'].removeprefix('meteorology:')
        if extracted['instances'][ident]['met_station'] != station:
            raise ValueError(f'{ident}: station mismatch')
        splits = [t.removeprefix('split:') for t in instance['coverage_tags'] if t.startswith('split:')]
        if len(splits) != 1 or splits[0] not in {'train', 'eval'}:
            raise ValueError(f'{ident}: expected one train/eval split')
        split = splits[0]
        archetype = task['scenario']['facility']['archetype']
        menus = {u['id']: [float(h) for h in height_menu(u['stack_bounds']['height_min_m'],
                                                       u['stack_bounds']['gep_height_m'])]
                 for u in task['scenario']['units']}
        draft = json.loads((public / 'draft/design.yaml').read_text())
        for unit in task['scenario']['units']:
            expected = {'control': unit['id'] + '_maximum',
                        'stack_height_m': unit['stack_bounds']['gep_height_m']}
            if draft[unit['id']] != expected:
                raise ValueError(f'{ident}: published draft is not maximum at GEP')
        published_mode = next(r['fields'] for r in task['locked_keyword_table']['NO2']
                              if r['fields'][1] == 'MODELOPT')
        tier = 'PVMRM' if 'PVMRM' in published_mode else 'ARM2'
        task.update(height_menu=menus, no2_tier='ARM2', evaluate_budget=30, split=split,
                    station=station, archetype=archetype,
                    design_value_requirements=task['scenario']['naaqs_thresholds'],
                    gep_method=task['scenario']['gep_method'],
                    ozone={**task['background']['ozone'], 'usage': 'unused'},
                    native_templates_tier='as_published', draft_statistics_tier='as_published',
                    legacy={'instance_id': ident, 'v1_commit': extracted['source_git']['head'],
                            'v1_task_name': 'permit-forge'})
        destination = out / 'data' / ident
        destination.mkdir(parents=True)
        write_json(destination / 'task.json', task)
        for name in ('buildings.json', 'receptors.inp', 'background.json', 'cost_table.json', 'scenario.yaml'):
            shutil.copyfile(public / name, destination / name)
        shutil.copytree(public / 'draft', destination / 'draft')
        if station not in stations:
            shutil.copytree(src / 'met' / station, out / 'met' / station)
            stations[station] = file_hashes(out / 'met' / station)
        row = {'id': ident, 'split': split, 'station': station, 'archetype': archetype,
               'n_sources': len(task['scenario']['units']), 'sha256': file_hashes(destination),
               'met_sha256': stations[station], 'height_menu': menus, 'height_mode': 'menu',
               'no2_tier': 'ARM2', 'published_no2_tier': tier, 'eligible': None,
               'eligibility_reason': 'pending ARM2 draft verification', 'legacy': task['legacy']}
        rows.append(row)
        decisions.append({'id': ident, 'split': split, 'decision': 'converted', 'eligible': None,
                          'published_no2_tier': tier, 'no2_tier': 'ARM2',
                          'reason': 'preserved v1 physics inputs; menu heights; pending ARM2 draft verification'})
    summary = write_manifests(out, rows)
    write_rows(out / 'build_log.jsonl', decisions)
    for split in ('train', 'eval'):
        write_rows(out / f'permit-v2-{split}/build_log.jsonl', [d for d in decisions if d['split'] == split])
    legacy = [{**r, 'height_menu': None, 'height_mode': 'continuous', 'no2_tier': 'as_published',
               'eligible': None, 'eligibility_reason': 'legacy comparison only'} for r in rows]
    write_rows(out / 'permit-v1-legacy/tasks.jsonl', legacy)
    write_json(out / 'permit-v1-legacy/manifest.json', {'counts': counts(legacy)})
    write_rows(out / 'permit-v1-legacy/build_log.jsonl', [
        {'id': r['id'], 'decision': 'legacy pointer',
         'reason': 'published tier and continuous heights; data hashes identify preserved v2 input bundle'}
        for r in rows])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', type=Path, default=Path('convert/v1-src'))
    parser.add_argument('--out', type=Path, default=Path('envs/permit/tasks'))
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        errors = check_integrity(args.out)
        print(json.dumps({'integrity_errors': errors}, indent=2))
        return int(bool(errors))
    print(json.dumps(convert(args.src, args.out), indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
