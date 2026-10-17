"""Freeze model-only anchors: python -m eval.calibrate --runs RUN ... --task-files PATH ... ."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "eval"

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .common import ROOT, finite, now, records, run_path, write_text


def calibrate(runs: list[Path], task_files: list[Path], *, frozen_at: str | None = None) -> None:
    best = defaultdict(float)
    seen = set()
    provenance = defaultdict(list)
    for path, record in records(runs):
        if not record['model'].startswith(('anthropic:', 'codex:')):
            continue
        key = record['model'], record['task_id']
        if key in seen:
            raise ValueError(f'duplicate model/task: {key}')
        seen.add(key)
        provenance[record['task_id']].append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        final = record.get('final') or {}
        saved = final.get('prescore_usd_saved')
        submitted = record.get('ended') == 'submitted' and any(s['tool'] == 'submit_design' for s in record.get('steps', []))
        if submitted and final.get('valid', True) and final.get('feasible') and finite(saved) and saved > 0:
            best[record['task_id']] = max(best[record['task_id']], saved)
    timestamp = frozen_at or now()
    writes = []
    for path in task_files:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            tid = row.get('task_id', row.get('id'))
            candidate = {'worst': 0, 'best': best[tid] or None,
                         'saturation': 4*best[tid] if best[tid] else None,
                         'source': 'panel' if best[tid] else 'none'}
            previous = row.get('anchors') or {}
            if previous.get('frozen_at') or row.get('anchors_frozen_at'):
                if any(previous.get(k) != v for k,v in candidate.items()):
                    raise ValueError(f'anchors already frozen for {tid}')
                continue
            if best[tid]:
                candidate['frozen_at'] = timestamp
                row['anchors_frozen_at'] = timestamp
                row['anchor_runs'] = provenance[tid]
            row['anchors'] = candidate
        writes.append((path, ''.join(json.dumps(r, allow_nan=False, separators=(',', ':'))+'\n' for r in rows)))
    # Validate every file before replacing any; unchanged files retain their bytes on repeat invocations.
    for path, text in writes:
        old_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if old_rows != [json.loads(line) for line in text.splitlines()]:
            write_text(path, text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--task-files', type=Path, nargs='+')
    args = parser.parse_args()
    paths = args.task_files or sorted((ROOT/'envs/permit/tasks').glob('permit-v2-*/tasks.jsonl'))
    if not paths:
        parser.error('no converted task records found')
    calibrate([run_path(p) for p in args.runs], paths)


if __name__ == '__main__':
    main()
