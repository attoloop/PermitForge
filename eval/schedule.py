"""Deterministic episode order and cost projection; no model or native execution."""
from __future__ import annotations

from collections import Counter
import statistics


def phases(tasks: list[dict], models: list[str], repeats: int, order: str) -> list[list[dict]]:
    if order == 'task-major':
        return [[dict(task, model=m, repeat=r) for m in models for r in range(1, repeats+1)] for task in tasks]
    if order != 'round-robin':
        raise ValueError('unknown scheduling order')
    result = []
    for repeat in range(1, repeats+1):
        for split in ('eval', 'train'):
            cohort = sorted((t for t in tasks if t['split'] == split), key=lambda t: (t['station'], t['task_id']))
            paid = sorted((m for m in models if m.startswith('anthropic:')),
                          key=lambda m: (('opus' not in m) if split == 'eval' else ('sonnet' not in m), m))
            free = sorted(m for m in models if not m.startswith('anthropic:'))
            jobs = [dict(task, model=m, repeat=repeat) for m in paid+free for task in cohort]
            if jobs:
                result.append(jobs)
    return result


def projection(schedule: list[dict], records: list[dict], prior: float, *, after_round: int = 0) -> dict:
    completed = {(r['model'], r['task_id'], r.get('repeat', 1)) for r in records}
    remaining = Counter(j['model'] for j in schedule if (j['model'], j['task_id'], j['repeat']) not in completed)
    result = {}
    for model in sorted({j['model'] for j in schedule}):
        observed = [r['usage']['usd'] for r in records if r['model'] == model and
                    (r.get('usage') or {}).get('usd') is not None]
        paid = model.startswith('anthropic:')
        estimate = statistics.mean(observed) if observed else prior if paid else None
        result[model] = {'measured_episodes': len(observed), 'usd_per_episode': estimate,
                         'source': 'measured' if observed else 'prior' if paid else 'subscription; USD unknown',
                         'remaining_episodes': remaining[model],
                         'projected_remaining_usd': estimate*remaining[model] if estimate is not None else None}
    return {'after_round': after_round, 'models': result,
            'projected_remaining_paid_usd': sum(r['projected_remaining_usd'] or 0 for r in result.values()),
            'note': 'Projection is not authorization or a reservation. Request-level hard cap always applies.'}
