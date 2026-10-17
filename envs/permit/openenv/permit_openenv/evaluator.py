"""Interchangeable evaluation, with the full native verifier reserved for grading."""
import importlib
from contextlib import nullcontext
import os
from pathlib import Path
import tempfile
import threading
from typing import Protocol

from permit_core.native_pool import native_slot

from .protocol import json_value
from .diagnostics import log
from .tasks import Task, to_ledger

# Each verification launches its native cases sequentially. Bound in-server native work.
_NATIVE_SLOTS = threading.BoundedSemaphore(1)


class Evaluator(Protocol):
    def evaluate(self, task: Task, design: dict) -> dict: ...


class VerifyEvaluator:
    def evaluate(self, task: Task, design: dict) -> dict:
        from permit_core.verify import verify
        log('native verification queued')
        with _NATIVE_SLOTS, native_slot(), tempfile.TemporaryDirectory(prefix='permit-openenv-') as work:
            result = verify(task.scenario, to_ledger(design), work, no2_tier='ARM2')
        log(f'native verification: feasible={result.feasible}; errors={result.errors}; standards={result.design_values}')
        return json_value({
            'valid': True, 'errors': [f"{e['code']}: {e['detail']}" for e in result.errors],
            'feasible': result.feasible, 'cost': result.cost,
            'standards': [{'id': name, 'value': v.value, 'background': v.background,
                           'limit': v.limit, 'margin': v.margin,
                           'binding_receptor': {'index': v.receptor_index, 'x_m': v.x, 'y_m': v.y}}
                          for name, v in result.design_values.items()],
            'receptor_values': {name: [float(r['modeled']) for r in sorted(rows, key=lambda r: r['receptor'])]
                               for name, rows in result.receptor_values.items()},
            'receptor_margins': {name: [float(r['margin']) for r in sorted(rows, key=lambda r: r['receptor'])]
                                for name, rows in result.receptor_values.items()},
            'backend': 'full AERMOD',
        })


class FastEvaluator:
    def __init__(self) -> None:
        self.module = importlib.import_module('permit_core.evaluate')
        self.fields_module = importlib.import_module('permit_core.fields')
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def evaluate(self, task: Task, design: dict) -> dict:
        log('field preparation requested')
        configured = os.environ.get('PERMIT_FIELDS_DIR')
        if configured:
            path = Path(configured)/f'{task.task_id}.npz'
            lease = nullcontext(self.fields_module.load_pack(path))
        else:
            lease = self.fields_module.ensure_pack(task.scenario, jobs=int(os.environ.get('PERMIT_FIELDS_JOBS', '4')))
        with lease as pack:
            log('field preparation complete; evaluating design')
            result = self.module.evaluate(task.scenario, design, pack, include_receptors=True)
        result['backend'] = 'fast evaluator'
        with self._lock:
            if task.task_id not in self._seen:
                result['note'] = 'Fields are cached on demand. The first evaluation may take minutes; later evaluations reuse the cache.'
                self._seen.add(task.task_id)
        return json_value(result)


def select_evaluator() -> Evaluator:
    name = os.environ.get('PERMIT_EVALUATOR', '').strip().lower()
    if name == 'verify':
        return VerifyEvaluator()
    if name not in ('', 'fast'):
        raise ValueError('PERMIT_EVALUATOR must be fast or verify')
    try:
        return FastEvaluator()
    except ImportError:
        if name == 'fast':
            raise RuntimeError('Fast evaluation dependencies are unavailable; use PERMIT_EVALUATOR=verify') from None
        return VerifyEvaluator()
