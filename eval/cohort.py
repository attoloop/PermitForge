"""Validate a final selection before any provider work; used by eval.run_eval."""
from __future__ import annotations

from collections import Counter

from .select_tasks import choose


def validate_subsets(value: dict) -> None:
    if not isinstance(value, dict) or value.get('protocol') != 'WP2.8':
        raise ValueError('final cohort requires protocol WP2.8; reproduce with eval.select_tasks --report-only')
    try:
        for split, stations, quota in [('eval', 5, 2), ('train', 10, 1)]:
            selected = value[split]
            counts = Counter(r['station'] for r in selected)
            groups = Counter(r['selection_group'] for r in selected)
            probes = [r for r in value['probes'] if r['split'] == split]
            if (len(selected) != 10 or len({r['task_id'] for r in selected}) != 10
                    or len(counts) != stations or set(counts.values()) != {quota}
                    or groups != {'hard': 8, 'easy': 2} or selected != choose(probes)):
                raise ValueError('final cohort differs from WP2.8 frozen-probe selection')
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('invalid WP2.8 cohort; reproduce with eval.select_tasks --report-only') from exc
