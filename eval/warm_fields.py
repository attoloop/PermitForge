"""Warm selected packs: python -m eval.warm_fields --subsets results/subsets.json --field-jobs 12.

All leases stay pinned until the final inventory check. Use the same cache cap
for warming and evaluation; the default 40 GiB may be too small for this cohort.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import json
from pathlib import Path
from typing import Iterator

from permit_core.fields import ensure_pack
from permit_core.model import load_v2_instance

from .common import ROOT, write_json


def warm(tasks_root: Path, task_ids: list[str], *, field_jobs: int) -> dict:
    if not 1 <= field_jobs <= 12 or len(set(task_ids)) != len(task_ids):
        raise ValueError('warm uses 1..12 native workers and unique task IDs')
    leases, result = [], []
    try:
        for tid in task_ids:
            lease = ensure_pack(load_v2_instance(tasks_root, tid), jobs=field_jobs)
            leases.append(lease)
            manifest = json.loads(lease.path.read_text())
            result.append({'task_id': tid, 'path': str(lease.path), 'identity': manifest['identity'],
                           'bytes': sum(v[0] for v in manifest['files'].values()), 'computed': lease.computed})
            print('warm '+tid, flush=True)
        return {'tasks': result, 'bytes': sum(r['bytes'] for r in result)}
    finally:
        for lease in leases:
            lease.release()


def check_warm(tasks_root: Path, task_ids: list[str]) -> list[dict]:
    from permit_core.fields import cache_root, pack_identity
    from permit_core.field_cache import ready
    result = []
    for tid in task_ids:
        identity = pack_identity(load_v2_instance(tasks_root, tid))
        path = cache_root()/tid/'pack.json'
        if not ready(path.parent, identity):
            raise ValueError('selected field pack missing/stale: '+tid+'; run eval.warm_fields')
        result.append({'task_id': tid, 'identity': identity, 'path': str(path)})
    return result


@contextmanager
def pin_warm(tasks_root: Path, task_ids: list[str]) -> Iterator[list[dict]]:
    """Keep existing packs from being evicted for the entire local panel."""
    from permit_core.field_cache import lock_file
    with ExitStack() as stack:
        for tid in sorted(task_ids):
            handle = stack.enter_context(lock_file(tid))
            fcntl.flock(handle, fcntl.LOCK_SH)
        yield check_warm(tasks_root, task_ids)


def main() -> None:
    from permit_core.field_cache import cache_limit
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subsets', type=Path, default=ROOT/'results/subsets.json')
    parser.add_argument('--tasks-root', type=Path, default=ROOT/'envs/permit/tasks')
    parser.add_argument('--field-jobs', type=int, default=12)
    parser.add_argument('--output', type=Path, default=ROOT/'internal/reports/WP2.8-evidence/warm.json')
    args = parser.parse_args()
    subsets = json.loads(args.subsets.read_text())
    ids = [r['task_id'] for split in ('eval', 'train') for r in subsets[split]]
    estimated = 0
    for tid in ids:
        task = load_v2_instance(args.tasks_root, tid)
        hours = sum(1 for _ in (task.met_path/f'{task.station}.sfc').open())-1
        estimated += sum(map(len, task.height_menu.values())) * (hours*len(task.receptors)*8+20000)
    if cache_limit() < estimated:
        parser.error(f'cache cap too small: need at least {estimated/1024**3:.2f} GiB; set PERMIT_FIELDS_CACHE_GB')
    result = warm(args.tasks_root, ids, field_jobs=args.field_jobs)
    result['checked'] = check_warm(args.tasks_root, ids)
    result['subsets'] = str(args.subsets)
    write_json(args.output, result)


if __name__ == '__main__':
    main()
