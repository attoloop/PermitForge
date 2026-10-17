"""Build train-only conversations: python -m train.cloud.make_sft --help."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from permit_openenv.tasks import load_pack
from train.sft import episode_to_chat


def validate_catalog(rows: list[dict]) -> None:
    ids = [r['task_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate task identity across splits')
    stations = {split: {r['station'] for r in rows if r['split'] == split} for split in ('train', 'eval')}
    if stations['train'] & stations['eval']:
        raise ValueError('Train/eval station leakage in task catalog')


def episode_files(paths: list[Path]):
    seen = set()
    for path in paths:
        if path.is_dir():
            canonical = sorted(path.rglob('episode.json'))
            files = canonical or sorted(path.glob('*/*.json'))
        else:
            files = [path]
        for file in files:
            if file.resolve() not in seen:
                seen.add(file.resolve())
                yield file


def build_dataset(paths: list[Path], tasks_dir: Path) -> tuple[list[dict], dict]:
    pack = load_pack(tasks_dir)
    catalog = [{'task_id': t.task_id, 'split': t.split, 'station': t.record['station']} for t in pack.tasks]
    validate_catalog(catalog)
    by_id = {r['task_id']: r for r in catalog}
    rows, reasons, sources, seen = [], Counter(), {}, {}
    records_count = 0
    for path in episode_files(paths):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        records = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                   if path.suffix == '.jsonl' else [json.loads(path.read_text())])
        for record in records:
            if not isinstance(record, dict) or 'task_id' not in record or 'steps' not in record:
                continue
            identity = by_id.get(record['task_id'])
            if identity is None:
                raise ValueError('Unknown task identity: '+str(record['task_id']))
            if record.get('split') != identity['split']:
                raise ValueError('Record task/split identity disagrees with authoritative catalog')
            for view in (record, record.get('scenario') or {}):
                if view.get('task_id', identity['task_id']) != identity['task_id']:
                    raise ValueError('Scenario task identity disagrees with record')
                if view.get('station', identity['station']) != identity['station']:
                    raise ValueError('Record station identity disagrees with authoritative catalog')
            key = (record.get('model'), record['task_id'], record.get('repeat', 0))
            fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            if key in seen:
                if seen[key] != fingerprint:
                    raise ValueError('Conflicting duplicate episode identity')
                continue
            seen[key] = fingerprint
            records_count += 1
            if identity['split'] != 'train':
                reasons['eval_split'] += 1
                continue
            if record.get('steps_dropped', 0) or record.get('synthetic'):
                reasons['incomplete_or_synthetic'] += 1
                continue
            row = episode_to_chat(record, pack=pack)
            if row is None:
                reasons['not_feasible_improving_submission'] += 1
                continue
            rows.append({**identity, **row, 'source':str(path), 'source_sha256':sources[str(path)]})
    stats = dict(input_records=records_count, accepted=len(rows), rejected=sum(reasons.values()),
                 rejection_reasons=dict(reasons), sources_sha256=sources,
                 train_stations=sorted({r['station'] for r in catalog if r['split']=='train'}),
                 eval_stations=sorted({r['station'] for r in catalog if r['split']=='eval'}),
                 task_ids=sorted({r['task_id'] for r in rows}),
                 rule='authoritative train id and station; feasible; exact positive savings; terminal single submit')
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, nargs='+', required=True)
    parser.add_argument('--tasks-dir', type=Path, default=Path('envs/permit/tasks'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, stats = build_dataset(args.records, args.tasks_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    args.output.with_suffix('.stats.json').write_text(json.dumps(stats, indent=2)+'\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
