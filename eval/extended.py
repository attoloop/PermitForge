"""Extended experiment evidence: uv run --project envs/permit/openenv --no-sync python -m eval.extended."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from .common import ROOT, records

EXPERIMENT = 'extended-budget'
FABLE_API = 'anthropic:claude-fable-5-1'
FABLE_SUB = 'claudecode:claude-fable-5-1'
ASTRA = 'codex:gpt-6-astra'
PRIOR_USD = 60.0
PRIOR_NOTE = ('Conservative episode estimate, not an invoice ceiling: final-medium3 maximum Opus '
              'tokens repriced at Fable rates, scaled 100/30, doubled, rounded up to the next $10 '
              '(60 USD). Reproduce: python -m eval.extended. After the first completed API episode '
              'use the maximum measured episode cost. Admission reserves the whole estimate; '
              'no request-level budget can interrupt an admitted extended episode.')


def evidence() -> dict:
    from .anthropic_driver import request_cost
    rows = records([ROOT/'results/rollouts/final-medium3'])
    counts = Counter(r['model'] for _, r in rows)
    exhausted = Counter(r['model'] for _, r in rows if r['evaluations_used'] >= 30)
    candidates = [(request_cost('claude-fable-5-1', r['usage']), p, r) for p, r in rows
                  if r['model'] == 'anthropic:claude-opus-5-5']
    cost, path, record = max(candidates, key=lambda item: item[0])
    scaled = cost*100/30*2
    return {'command': 'PERMIT_FIELDS_CACHE_GB=64 uv run --project envs/permit/openenv --no-sync python -m eval.extended',
            'episodes': dict(counts), 'exhausted_30': {m: exhausted[m] for m in counts},
            'prior': {'source': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'usage': {k: record['usage'][k] for k in ('input_tokens', 'output_tokens')},
                      'fable_repriced_usd': cost, 'scale': '100/30 * 2',
                      'scaled_usd': scaled, 'rounded_up_10_usd': math.ceil(scaled/10)*10,
                      'launch_prior_usd': PRIOR_USD, 'note': PRIOR_NOTE}}


def remaining(run: Path, tasks: list[dict]) -> dict:
    completed = {r['task_id'] for _, r in records([run]) if r['model'] == FABLE_API}
    return {'experiment': EXPERIMENT, 'source_run': str(run), 'model': FABLE_SUB,
            'tasks': [t for t in tasks if t['task_id'] not in completed]}


def consume_remaining(path: Path, tasks: list[dict], *, preview: bool = False) -> list[dict]:
    value = json.loads(path.read_text())
    if value.get('experiment') != EXPERIMENT or value.get('model') != FABLE_SUB:
        raise ValueError('remaining tasks must come from the extended Fable credit leg')
    requested = value['tasks']
    ids = [t['task_id'] for t in requested]
    if len(set(ids)) != len(ids) or set(ids)-{t['task_id'] for t in tasks}:
        raise ValueError('remaining tasks must be unique members of the frozen cohort')
    if not value.get('source_run') and not preview:
        raise ValueError('subscription launch requires an authoritative credit-leg source_run')
    if value.get('source_run'):
        source = Path(value['source_run'])
        manifest = json.loads((source/'run.json').read_text())
        if manifest.get('status') not in ('budget_stopped', 'complete'):
            raise ValueError('credit leg must finish before the subscription leg')
        if remaining(source, tasks)['tasks'] != requested:
            raise ValueError('remaining tasks do not match authoritative credit-leg records')
    return [t for t in tasks if t['task_id'] in ids]


def main() -> None:
    print(json.dumps(evidence(), indent=2))


if __name__ == '__main__':
    main()
