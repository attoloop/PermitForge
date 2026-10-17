"""Verified naive policies on the report cohort: python -m eval.report RUN --compute-baselines."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any, Callable

from permit_core.baselines import POLICIES, outcome

from .common import write_json


def compute_policies(view: dict, evaluate: Callable, verify: Callable) -> dict:
    results = {}
    verified = {}
    for name, policy in POLICIES.items():
        result = policy(view, evaluate)
        if result['design'] is not None:
            key = json.dumps(result['design'], sort_keys=True)
            if key not in verified:
                verified[key] = verify(result['design'])
            check = verified[key]
            if not check.get('feasible') and check.get('valid', True) and check.get('errors') and not any(
                    code in str(check['errors']) for code in ('E_NAAQS', 'E_GEP', 'E_MENU')):
                raise RuntimeError(f'baseline verification infrastructure failure: {check["errors"]}')
            result = {**outcome(view, result['design'], check, result['evaluations_used']),
                      'verified': True, 'verification': check}
        results[name] = result
    return results


def load_baselines(path: Path) -> dict:
    data = json.loads(path.read_text())
    data = data.get('baselines', data.get('policies', data))
    for name in POLICIES:
        if name not in data:
            raise ValueError(f'incomplete baseline artifact: {path}')
        result = data[name]
        if result.get('design') is not None and not result.get('verified'):
            raise ValueError(f'unverified policy in {path}: {name}')
        if not result.get('feasible') and result.get('usd_saved') != -1:
            raise ValueError(f'policy failure lacks sentinel: {name}')
    return data


def compute_cached(task: Any, destination: Path) -> None:
    """Hold a cache lease; never build fields or silently use the fast evaluator as a grade."""
    from permit_core import field_cache, fields
    from permit_core.evaluate import evaluate
    from .rescore import native_verify
    scenario = task.scenario
    root = fields.cache_root()/task.task_id
    lease = field_cache.lock_file(task.task_id)
    try:
        fcntl.flock(lease, fcntl.LOCK_SH)
        if not field_cache.ready(root, fields.pack_identity(scenario)):
            raise FileNotFoundError(f'current cached fields unavailable for {task.task_id}; evaluate this task first')
        with field_cache.Pack(scenario, root, lease, False, scenario.height_menu) as pack:
            results = compute_policies(task.view(), lambda d: evaluate(scenario, d, pack),
                                       lambda d: native_verify(task, d))
            write_json(destination, {'task_id': task.task_id, 'baselines': results,
                                    'field_identity': fields.pack_identity(scenario)})
    finally:
        if not lease.closed:
            lease.close()
