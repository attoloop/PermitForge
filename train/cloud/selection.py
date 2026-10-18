"""Frozen panel cohort and station-disjoint training identities."""
import json
import hashlib
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


def selection_manifest(pack, cohort: Path) -> dict:
    """Three hash-ranked tasks per train station; no policy outcomes enter selection."""
    train = select_train(pack)
    chosen = []
    for station in sorted({t.record['station'] for t in train}):
        candidates = [t for t in train if t.record['station'] == station]
        chosen.extend(sorted(candidates, key=lambda t:hashlib.sha256(
            ('WP2.12:'+t.task_id).encode()).hexdigest())[:3])
    if len(chosen) != 30:
        raise ValueError('Expected three tasks from each of ten train stations')
    small = sorted(chosen, key=lambda t:(t.record['n_sources'], t.task_id))[:2]
    evaluation = select_cohort(pack, cohort)
    smoke_eval, stations = [], set()
    for task in sorted((pack.get(t) for t in evaluation), key=lambda t:(t.record['n_sources'], t.task_id)):
        if task.record['station'] not in stations:
            smoke_eval.append(task.task_id)
            stations.add(task.record['station'])
        if len(smoke_eval) == 3:
            break
    return dict(rule='Three eligible tasks per train station, ordered by SHA256(WP2.12:task_id); '
        'smoke train = first two by (source count, ID); smoke eval = first three distinct stations '
        'in the frozen ten-task cohort ordered by (source count, ID). No reward outcomes used.',
        train=[t.task_id for t in chosen], eval=evaluation,
        smoke_train=[t.task_id for t in small], smoke_eval=smoke_eval)


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
