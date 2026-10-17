"""Frozen panel cohort and station-disjoint training identities."""
import json
from pathlib import Path

from .make_sft import validate_catalog


def select_train(pack, ids: list[str] | None = None) -> list:
    validate_catalog([dict(task_id=t.task_id, split=t.split, station=t.record['station']) for t in pack.tasks])
    tasks = [pack.get(tid) for tid in ids] if ids else [t for t in pack.tasks if t.split=='train']
    if not tasks or len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError('Need unique training task IDs')
    if any(t.split != 'train' or t.record.get('eligible') is not True for t in tasks):
        raise ValueError('Only eligible train tasks are available to training')
    return tasks


def select_cohort(pack, path: Path) -> list[str]:
    data = json.loads(path.read_text())
    rows = data['eval']
    ids = [r['task_id'] for r in rows]
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError('The cloud evaluation requires exactly ten distinct panel eval tasks')
    stations = []
    for row in rows:
        task = pack.get(row['task_id'])
        if task.split != 'eval' or task.record.get('eligible') is not True:
            raise ValueError('Panel evaluation requires eligible eval tasks')
        if row.get('station') != task.record['station']:
            raise ValueError('Cohort station disagrees with task catalog')
        stations.append(row['station'])
    if len(set(stations)) != 5 or any(stations.count(s) != 2 for s in stations):
        raise ValueError('Cohort requires two tasks per eval station')
    return ids
