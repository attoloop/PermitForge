"""Report model best savings without changing tasks: python -m eval.best_observed RUN --output FILE."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import report_records, run_path, saved_fraction, write_json


def best_observed(items: list[dict]) -> dict[str, dict]:
    """Secondary reference only; never used by the reward function."""
    result = {}
    for record in items:
        if record['model'].startswith(('baseline:', 'search:')):
            continue
        tid = record['task_id']
        score = saved_fraction(record)
        row = result.setdefault(tid, {'saved_fraction': 0.0, 'observations': []})
        if score > row['saved_fraction']:
            row.update(saved_fraction=score, observations=[])
        if score > 0 and score == row['saved_fraction']:
            row['observations'].append({'model': record['model'], 'repeat': record.get('repeat', 1),
                                        'run': record.get('run')})
    return dict(sorted(result.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    items = [r for _, r in report_records([run_path(p) for p in args.runs])]
    from .report import build_report
    build_report(items)  # Reject duplicate repeat identities and mixed splits.
    write_json(args.output, {'metric': 'best observed saved fraction; secondary metric, never reward',
                             'tasks': best_observed(items)})


if __name__ == '__main__':
    main()
