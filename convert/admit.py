"""Apply ARM2 draft admission; no native work is launched.

uv run --project envs/permit/core python convert/admit.py --tasks envs/permit/tasks
Requires matching drafts-arm2 evidence. Search and baseline results are diagnostics only.
"""

import argparse
from collections import Counter
from pathlib import Path

if __package__:
    from .reverify_drafts import DEFAULT_CACHE as DRAFT_CACHE, cached_result
    from .run_baselines import DEFAULT_CACHE as BASELINE_CACHE
    from .v1_to_v2 import check_row_integrity, read_rows, write_json, write_manifests, write_rows
else:
    from reverify_drafts import DEFAULT_CACHE as DRAFT_CACHE, cached_result
    from run_baselines import DEFAULT_CACHE as BASELINE_CACHE
    from v1_to_v2 import check_row_integrity, read_rows, write_json, write_manifests, write_rows

def decision(draft: dict | None, result: dict | None = None) -> tuple[bool | None, list[str]]:
    """Admit a native-feasible draft; the legacy diagnostic argument has no effect."""
    if draft is None:
        return None, ['pending ARM2 draft verification']
    if draft.get('no2_tier') != 'ARM2':
        raise ValueError('admission requires ARM2 draft verification')
    if not draft['feasible']:
        return False, ['draft infeasible under ARM2']
    return True, []


def apply(tasks: Path, drafts: Path = DRAFT_CACHE, baselines: Path = BASELINE_CACHE) -> dict:
    """Validate draft evidence before writing; preserve data and existing calibration."""
    import json
    rows = read_rows(tasks)
    decisions = []
    tally: Counter = Counter()
    for row in rows:
        errors = check_row_integrity(tasks, row)
        if errors:
            raise ValueError('; '.join(errors))
        draft = cached_result(drafts / f"{row['id']}.json", row)
        tally['draft_failed'] += draft is not None and not draft['feasible']
        admitted, reasons = decision(draft)
        tally['pending' if admitted is None else 'admitted' if admitted else 'dropped'] += 1
        eligible = admitted if admitted is not None else draft['feasible'] if draft else None
        reason = '; '.join(reasons) if reasons else 'draft feasible under ARM2'
        changes = {'eligible': eligible, 'admitted': admitted, 'eligibility_reason': reason,
                   'admission_reasons': reasons}
        if any(row.get(k) != value for k, value in changes.items()):
            decisions.append({'id': row['id'], 'split': row['split'], 'decision': 'admission',
                              **changes})
        row.update(changes)
    # Preserve previous events and append only changed decisions.
    logs = {}
    for path, selected in [(tasks / 'build_log.jsonl', decisions),
                           *((tasks / f'permit-v2-{s}/build_log.jsonl',
                              [d for d in decisions if d['split'] == s]) for s in ('train', 'eval'))]:
        old = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        logs[path] = old+selected
    counts = {key: tally[key] for key in ('admitted', 'dropped', 'pending')}
    manifest = write_manifests(tasks, rows)
    manifest.update(admission_counts=counts, decision_required=tally['draft_failed'] > len(rows)/10)
    write_json(tasks / 'manifest.json', manifest)
    for path, events in logs.items():
        write_rows(path, events)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, default=Path('envs/permit/tasks'))
    parser.add_argument('--drafts', type=Path, default=DRAFT_CACHE)
    parser.add_argument('--baselines', type=Path, default=BASELINE_CACHE,
                        help='legacy compatibility argument; diagnostics do not affect admission')
    args = parser.parse_args()
    try:
        import json
        print(json.dumps(apply(args.tasks, args.drafts, args.baselines), indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
