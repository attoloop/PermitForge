"""Recover interrupted records without sampling: python -m eval.attempts RUN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .capture import EpisodeCapture
from .common import now, records, run_path, write_json

MAX_ATTEMPTS = 3  # Initial attempt and at most two infrastructure retries, on resume.


def history(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(root.glob('attempts/episode-*/attempt.json'))]


def infrastructure(run: Path) -> list[dict]:
    rows = [json.loads(p.read_text()) for p in sorted(run.glob('*/*/attempts/episode-*/attempt.json'))]
    return [row for row in rows if row.get('status') == 'infrastructure']


def publish(root: Path, record: dict, capture: EpisodeCapture) -> None:
    """The canonical episode is the commit point; the flat file is only a compatibility copy."""
    record.update(attempt_complete=True)
    capture.environment(record)
    for name in ('raw', 'transcript.jsonl', 'env.log'):
        source, target = capture.root/name, root/name
        if source == target or not source.exists():
            continue
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copyfile(source, target)
    write_json(root/'episode.json', record)
    write_json(root.parent/(root.name+'.json'), record)


def fail(root: Path, capture: EpisodeCapture, record: dict, error: str, trace: str) -> dict:
    number = record['attempt_number']
    submitted = record.get('final') is not None
    row = {'model': record['model'], 'task_id': record['task_id'], 'attempt_number': number,
           'status': 'infrastructure', 'started_at': record.get('started_at'), 'finished_at': now(),
           'error': error, 'traceback': trace, 'usage': record.get('usage'),
           'budget_charge_usd': record.get('budget_charge_usd'),
           'retryable': number < MAX_ATTEMPTS and not submitted and not record.get('submit_attempted'),
           'submission_preserved': submitted}
    write_json(capture.root/'checkpoint.json', record)
    write_json(capture.root/'attempt.json', row)
    capture.environment(record)
    if submitted:
        record.update(ended='submitted', infrastructure_error=error,
                      finished_at=record.get('finished_at') or now())
        publish(root, record, capture)
    return row


def recover(run: Path, *, error: str | None = None, trace: str | None = None) -> None:
    """Archive legacy unfinished finals and close abandoned attempts under the run lock."""
    for path, record in records([run], include_incomplete=True):
        if record.get('attempt_complete') is not False:
            continue
        root = path.parent if path.name == 'episode.json' else path.with_suffix('')
        number = len(history(root))+1
        capture = EpisodeCapture(root/'attempts'/f'episode-{number:04d}')
        # Preserve the exact old record and all captures before removing graded-looking paths.
        write_json(capture.root/'legacy-episode.json', record)
        for name in ('raw', 'transcript.jsonl', 'env.log'):
            source = root/name
            if source.is_dir():
                shutil.copytree(source, capture.root/name, dirs_exist_ok=True)
            elif source.exists():
                shutil.copyfile(source, capture.root/name)
        record.update(attempt_number=number, run=run.name)
        legacy_error = record.get('error') or 'interrupted legacy harness attempt'
        fail(root, capture, record, str(legacy_error), 'Traceback unavailable: recovered from an interrupted legacy run.')
        if record.get('final') is None:
            (root/'episode.json').unlink(missing_ok=True)
            (root.parent/(root.name+'.json')).unlink(missing_ok=True)
    for path in sorted(run.glob('*/*/attempts/episode-*/attempt.json')):
        attempt = json.loads(path.read_text())
        if attempt.get('status') not in ('running', 'infrastructure'):
            continue
        root = path.parents[2]
        # A crash after the canonical write must never trigger another model attempt.
        if (root/'episode.json').exists():
            if attempt['status'] == 'running':
                attempt.update(status='complete', finished_at=now())
                write_json(path, attempt)
            continue
        record = json.loads(path.with_name('checkpoint.json').read_text())
        record.update(attempt_number=attempt['attempt_number'])
        capture = EpisodeCapture(path.parent)
        if record.get('attempt_complete') is True:
            publish(root, record, capture)
            attempt.update(status='complete', finished_at=now(), retryable=False)
            write_json(path, attempt)
        elif attempt['status'] == 'running':
            fail(root, capture, record, error or 'harness interrupted', trace or
                 'Traceback unavailable: process ended before the collector finalized this attempt.')
        elif record.get('final') is not None:
            record.update(ended='submitted', infrastructure_error=attempt.get('error'))
            publish(root, record, capture)


def main() -> None:
    from .run_eval import run_lock, write_index
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args = parser.parse_args()
    run = run_path(args.run)
    if not run.is_dir():
        parser.error('run does not exist')
    with run_lock(run):
        recover(run)
        status = json.loads((run/'run.json').read_text()).get('status') if (run/'run.json').exists() else None
        write_index(run, status=status)
    print(f'Recovered records and rebuilt index/board: {run}')


if __name__ == '__main__':
    main()
