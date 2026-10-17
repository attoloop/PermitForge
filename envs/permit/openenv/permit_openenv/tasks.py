"""Load converted task packs, with a preserved-v1 fallback during conversion."""
from dataclasses import dataclass, replace
from decimal import Decimal
from functools import cached_property
import json
import os
from pathlib import Path
from typing import Any

from permit_core.cost import cost
from permit_core.menu import height_menu
from permit_core.model import Scenario, extraction_root, load_v1_instance, load_v2_instance
from permit_core.native import Reject
from permit_core.validate import validate_ledger

from .protocol import design_size_ok, json_value

DEFAULT_TASKS = Path(__file__).resolve().parents[2] / 'tasks'


def to_ledger(design: Any) -> dict:
    if not isinstance(design, dict):
        raise ValueError('design must be an object keyed by source ID')
    ledger = {}
    for sid, row in design.items():
        if not isinstance(row, dict) or set(row) != {'control', 'height_m'}:
            raise ValueError(f'{sid}: required keys control,height_m only')
        ledger[sid] = {'control': row['control'], 'stack_height_m': row['height_m']}
    return ledger


@dataclass
class Task:
    task_id: str
    split: str
    record: dict
    root: Path | None = None

    @cached_property
    def scenario(self) -> Scenario:
        scenario = (load_v2_instance(self.root, self.task_id) if self.root is not None
                    else load_v1_instance(self.task_id))
        menu = self.record.get('height_menu') or scenario.height_menu
        if menu is None:
            menu = {s.id: height_menu(s.height_min_m, s.gep_height_m) for s in scenario.sources}
        menu = {sid: tuple(Decimal(str(h)) for h in heights) for sid, heights in menu.items()}
        return replace(scenario, height_menu=menu)

    @property
    def draft_cost(self) -> Decimal:
        return cost(self.scenario.draft_design, self.scenario.task)

    def check(self, design: Any) -> dict:
        if not design_size_ok(design):
            return {'valid': False, 'errors': ['E_SIZE: design exceeds input limits']}
        try:
            ledger = validate_ledger(to_ledger(design), self.scenario.task,
                                     height_menu=self.scenario.height_menu)
            return {'valid': True, 'errors': [], 'cost': float(cost(ledger, self.scenario.task))}
        except (Reject, ValueError, TypeError, ArithmeticError):
            return {'valid': False, 'errors': ['E_MENU: use every source exactly once with a compatible control and listed height']}

    def view(self, evaluations_left: int = 30) -> dict:
        s = self.scenario
        return json_value({
            'task_id': self.task_id, 'split': self.split, 'station': s.station,
            'years': s.task['scenario']['locked_keywords']['meteorology_years'],
            'sources': [{'id': u.id, 'type': u.type, 'x_m': u.location['x_m'],
                         'y_m': u.location['y_m'], 'uncontrolled_g_s': u.uncontrolled_rates_g_s,
                         'no2_in_stack_ratio': u.no2_nox_in_stack_ratio,
                         'height_min_m': u.height_min_m, 'gep_m': u.gep_height_m,
                         'height_menu_m': s.height_menu[u.id]} for u in s.sources],
            'controls': [{'id': c.id, 'applies_to': c.applies_to, 'annualized_usd': c.annualized_usd,
                          'efficiency': c.efficiency_by_pollutant} for c in s.controls],
            'stack_usd_per_m': s.task['costs']['stack_usd_per_m'],
            'standards': [{'id': t.id, 'pollutant': t.pollutant, 'form': t.form,
                           'limit': t.limit, 'background': t.background} for t in s.standards],
            'draft_design': {sid: {'control': r['control'], 'height_m': r['stack_height_m']}
                             for sid, r in s.draft_design.items()},
            'draft_cost': self.draft_cost, 'evaluations_left': evaluations_left,
        })


class TaskPack:
    def __init__(self, tasks: list[Task]):
        self.tasks = [t for t in tasks if t.record.get('eligible') is not False]
        self._by_id = {t.task_id: t for t in self.tasks}
        if len(self._by_id) != len(self.tasks):
            raise ValueError('duplicate task IDs')

    def splits(self) -> list[str]:
        return sorted({t.split for t in self.tasks})

    def count(self, split: str) -> int:
        return sum(t.split == split for t in self.tasks)

    def get(self, task_id: str) -> Task:
        try:
            return self._by_id[task_id]
        except (KeyError, TypeError):
            raise ValueError('unknown task selection') from None

    def at(self, split: str, index: int) -> Task:
        tasks = [t for t in self.tasks if t.split == split]
        if index < 0 or index >= len(tasks):
            raise ValueError('unknown task selection')
        return tasks[index]

    def public(self, task: Task) -> dict:
        return {'task_id': task.task_id, 'split': task.split,
                'station': task.record.get('station', task.scenario.station),
                'eligible': task.record.get('eligible'),
                'eligibility_reason': task.record.get('eligibility_reason')}


def load_pack(root: str | Path | None = None) -> TaskPack:
    root = Path(root or os.environ.get('PERMIT_TASKS_DIR') or DEFAULT_TASKS).resolve()
    paths = sorted(root.glob('permit-v2-*/tasks.jsonl'))
    if paths:
        tasks = []
        for path in paths:
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    tasks.append(Task(row.get('task_id', row.get('id')), row['split'], row, root))
        return TaskPack(tasks)
    source = extraction_root()
    manifest = json.loads((source / 'family/manifest.json').read_text())
    tasks = []
    for row in manifest['instances']:
        split = next(tag.removeprefix('split:') for tag in row['coverage_tags'] if tag.startswith('split:'))
        station = row['equivalence_key'].split(':')[-1]
        tasks.append(Task(row['id'], split, {**row, 'station': station,
                                            'eligible': None, 'eligibility_reason': 'v1 fallback; ARM2 admission pending'}))
    return TaskPack(tasks)
