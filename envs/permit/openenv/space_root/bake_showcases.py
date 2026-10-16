"""Bake eval showcase packs: uv run python space_root/bake_showcases.py --jobs 4.

Uses the same ensure_pack path as live evaluation. No archive or field downloads.
"""

import argparse
import json
import time

import numpy as np

from permit_core.fields import ensure_pack
from permit_openenv.tasks import load_pack


SHOWCASES = ('040cd27878ef-s47', '0558a6a6c25d-s47')


def bake(task_ids: list[str], *, jobs: int) -> None:
    if not 1 <= jobs <= 4:
        raise ValueError('showcase preparation requires 1 to 4 jobs')
    catalog = load_pack()
    for task_id in task_ids:
        task = catalog.get(task_id)
        if task.split != 'eval':
            raise ValueError(f'{task_id}: showcase must come from eval')
        started = time.monotonic()
        with ensure_pack(task.scenario, jobs=jobs) as pack:
            if any(array.dtype != np.float64 for array in pack.fields.values()):
                raise ValueError(f'{task_id}: showcase fields must be float64')
            print(json.dumps({'task_id': task_id, 'fields': len(pack.fields),
                              'dtype': 'float64', 'seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', default=list(SHOWCASES))
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    bake(args.tasks, jobs=args.jobs)
