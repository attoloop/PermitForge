# v1 origin: tasks/permit-forge/instances/003bd328a392-s47/solution/trial.py::search
# v1_coordinate_descent adapts search's ordering/budget to one-index menu moves.
"""Policies using only the public scenario view and evaluate_design responses."""

from copy import deepcopy
from contextlib import nullcontext
from decimal import Decimal as D
import json
from pathlib import Path
import random
from typing import Any, Callable

from .cost import cost
from .model import Scenario

Design = dict[str, dict[str, Any]]
Evaluator = Callable[[Design], dict]


def scenario_view(task: Scenario, *, split: str | None = None, budget: int = 30) -> dict:
    """The SCHEMAS public view, without a dependency from core to OpenEnv."""
    if task.height_menu is None:
        raise ValueError('baselines require the converted height menu')
    return {
        'task_id': task.id, 'split': split or task.task.get('split'), 'station': task.station,
        'years': task.task['scenario']['locked_keywords']['meteorology_years'],
        'sources': [{'id': s.id, 'type': s.type, 'x_m': float(s.location['x_m']),
                     'y_m': float(s.location['y_m']),
                     'uncontrolled_g_s': {p: float(v) for p, v in s.uncontrolled_rates_g_s.items()},
                     'no2_in_stack_ratio': float(s.no2_nox_in_stack_ratio),
                     'height_min_m': float(s.height_min_m), 'gep_m': float(s.gep_height_m),
                     'height_menu_m': [float(h) for h in task.height_menu[s.id]]} for s in task.sources],
        'controls': [{'id': c.id, 'applies_to': list(c.applies_to), 'annualized_usd': float(c.annualized_usd),
                      'efficiency': {p: float(v) for p, v in c.efficiency_by_pollutant.items()}}
                     for c in task.controls],
        'stack_usd_per_m': float(task.task['costs']['stack_usd_per_m']),
        'standards': [{'id': s.id, 'pollutant': s.pollutant, 'form': s.form,
                       'limit': float(s.limit), 'background': float(s.background)} for s in task.standards],
        'draft_design': {sid: {'control': row['control'], 'height_m': float(row['stack_height_m'])}
                         for sid, row in task.draft_design.items()},
        'draft_cost': float(cost(task.draft_design, task.task)), 'evaluations_left': budget,
    }


def option_cost(view: dict, sid: str, row: dict) -> D:
    source = next(s for s in view['sources'] if s['id'] == sid)
    control = next((c for c in view['controls'] if c['id'] == row['control']), None)
    control_cost = D(0) if row['control'] == 'none' else D(str(control['annualized_usd']))
    return control_cost + D(str(view['stack_usd_per_m'])) * (
        D(str(row['height_m'])) - D(str(source['height_min_m'])))


def design_cost(view: dict, design: Design) -> D:
    return sum((option_cost(view, sid, row) for sid, row in design.items()), D(0))


def canonical(design: Design) -> str:
    return json.dumps(design, sort_keys=True, separators=(',', ':'))


def source_options(view: dict, sid: str) -> list[dict]:
    source = next(s for s in view['sources'] if s['id'] == sid)
    controls = [c['id'] for c in view['controls'] if sid in c['applies_to']]
    if sid+'_none' not in controls:
        controls.append('none')
    rows = [{'control': c, 'height_m': h} for c in controls for h in sorted(source['height_menu_m'])]
    return sorted(rows, key=lambda row: (option_cost(view, sid, row), row['control'], row['height_m']))


def outcome(view: dict, design: Design | None, evaluation: dict | None, used: int) -> dict:
    feasible = bool(evaluation and evaluation.get('valid', True) and evaluation.get('feasible'))
    value = design_cost(view, design) if design is not None else None
    return {'design': deepcopy(design), 'feasible': feasible,
            'cost': float(value) if value is not None else None,
            'usd_saved': float(D(str(view['draft_cost']))-value) if feasible else -1,
            'evaluations_used': used}


class _Run:
    def __init__(self, view: dict, evaluate: Evaluator, limit: int = 30):
        self.view, self.evaluate = view, evaluate
        self.limit = min(limit, int(view.get('evaluations_left', 30)))
        self.used = 0
        self.best: tuple[D, str, Design, dict] | None = None
        self.seen: set[str] = set()

    def run(self, design: Design) -> dict:
        if self.used >= self.limit:
            raise ValueError('policy evaluation budget exhausted')
        result = self.evaluate(deepcopy(design))
        if not result.get('valid', True):
            raise ValueError(f'policy generated an invalid design: {result.get("errors")}')
        self.used += 1
        key = canonical(design)
        self.seen.add(key)
        if result.get('feasible'):
            candidate = (design_cost(self.view, design), key, deepcopy(design), deepcopy(result))
            if self.best is None or candidate[:2] < self.best[:2]:
                self.best = candidate
        return result

    def result(self) -> dict:
        return outcome(self.view, self.best[2] if self.best else None,
                       self.best[3] if self.best else None, self.used)


def _uniform(view: dict, level: str, *, low: bool = False) -> Design:
    controls = {c['id'] for c in view['controls']}
    return {s['id']: {'control': ('none' if level == 'none' and s['id']+'_none' not in controls
                                 else s['id']+'_'+level),
                      'height_m': min(s['height_menu_m']) if low else s['gep_m']}
            for s in sorted(view['sources'], key=lambda s: s['id'])}


def _single(view: dict, evaluate: Evaluator, design: Design) -> dict:
    run = _Run(view, evaluate)
    if run.limit == 0:
        return run.result()
    return outcome(view, design, run.run(design), run.used)


def draft(view: dict, evaluate: Evaluator) -> dict:
    return _single(view, evaluate, view['draft_design'])


def all_none_at_gep(view: dict, evaluate: Evaluator) -> dict:
    return _single(view, evaluate, _uniform(view, 'none'))


def uniform_moderate_at_gep(view: dict, evaluate: Evaluator) -> dict:
    return _single(view, evaluate, _uniform(view, 'moderate'))


def cheapest_first_greedy(view: dict, evaluate: Evaluator) -> dict:
    """Start at draft; visit sources by ID, accepting their first feasible cheaper option."""
    run = _Run(view, evaluate)
    if run.limit == 0:
        return run.result()
    if not run.run(view['draft_design']).get('feasible'):
        return run.result()
    for source in sorted(view['sources'], key=lambda s: s['id']):
        sid = source['id']
        for row in source_options(view, sid):
            if run.used == run.limit:
                return run.result()
            if option_cost(view, sid, row) >= option_cost(view, sid, run.best[2][sid]):
                break
            candidate = deepcopy(run.best[2])
            candidate[sid] = row
            if run.run(candidate).get('feasible'):
                break
    return run.result()


def random_30(view: dict, evaluate: Evaluator, *, seed: int = 1402026) -> dict:
    """Independent uniform menu draws with replacement; no free draft evaluation."""
    rng = random.Random(seed)
    run = _Run(view, evaluate)
    options = {s['id']: source_options(view, s['id']) for s in sorted(view['sources'], key=lambda s: s['id'])}
    for _ in range(run.limit):
        run.run({sid: rng.choice(rows) for sid, rows in options.items()})
    return run.result()


def v1_coordinate_descent(view: dict, evaluate: Evaluator) -> dict:
    """v1 trial.search order, with one menu-index height step and at most 25 calls.

    Draft; none/moderate/maximum at GEP/minimum; sorted-source height then control
    proposals; restart after each accepted cheaper move; fresh final evaluation.
    The continuous 10/1/.1/.01 m passes become one discrete-index pass.
    """
    run = _Run(view, evaluate, 25)
    if run.limit == 0:
        return run.result()
    if not run.run(view['draft_design']).get('feasible'):
        return run.result()
    levels = ['none', 'moderate', 'maximum']
    for level in levels:
        for low in (False, True):
            if run.used < run.limit-1:
                run.run(_uniform(view, level, low=low))
    while run.used < run.limit-1:
        moved = False
        for source in sorted(view['sources'], key=lambda s: s['id']):
            sid = source['id']
            row = run.best[2][sid]
            heights = sorted(source['height_menu_m'])
            index = heights.index(row['height_m'])
            level = row['control'].removeprefix(sid+'_')
            proposals = [('height_m', heights[max(0, index-1)])]
            if level in levels[1:]:
                lower = sid+'_'+levels[levels.index(level)-1]
                if lower == sid+'_none' and lower not in {c['id'] for c in view['controls']}:
                    lower = 'none'
                proposals.append(('control', lower))
            for field, value in proposals:
                candidate = deepcopy(run.best[2])
                candidate[sid][field] = value
                if canonical(candidate) in run.seen or design_cost(view, candidate) >= run.best[0]:
                    continue
                if run.used == run.limit-1:
                    break
                if run.run(candidate).get('feasible'):
                    moved = True
                    break
            if moved or run.used == run.limit-1:
                break
        if not moved:
            break
    if run.used < run.limit:
        retained = run.best
        final = run.run(retained[2])
        keys = ('valid', 'feasible', 'cost', 'standards', 'errors')
        if any(final.get(k) != retained[3].get(k) for k in keys):
            raise ValueError('Final fresh evaluation disagrees with retained candidate')
    return run.result()


def verify_evaluation(task: Scenario, design: Design, workdir: Path, *, verify_fn: Callable | None = None) -> dict:
    """Adapt the full ARM2 verifier to the public evaluation contract."""
    guard = nullcontext()
    if verify_fn is None:
        from .verify import verify
        from .native_pool import native_slot
        verify_fn = verify
        guard = native_slot()
    ledger = {sid: {'control': row['control'], 'stack_height_m': row['height_m']}
              for sid, row in design.items()}
    with guard:
        result = verify_fn(task, ledger, workdir, no2_tier='ARM2')
    infrastructure = [e for e in result.errors if e['code'] not in {'E_NAAQS', 'E_GEP', 'E_MENU'}]
    if infrastructure:
        raise ValueError(f'native verification failed: {infrastructure}')
    return {'valid': not any(e['code'] == 'E_MENU' for e in result.errors),
            'errors': result.errors, 'feasible': result.feasible,
            'cost': float(result.cost) if result.cost is not None else None,
            'standards': [{'id': name, 'value': float(v.value), 'background': float(v.background),
                           'limit': float(v.limit), 'margin': float(v.margin),
                           'binding_receptor': {'index': v.receptor_index, 'x_m': float(v.x), 'y_m': float(v.y)}}
                          for name, v in result.design_values.items()]}


POLICIES = {fn.__name__: fn for fn in (draft, all_none_at_gep, uniform_moderate_at_gep,
                                    cheapest_first_greedy, random_30, v1_coordinate_descent)}
