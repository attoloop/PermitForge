"""FineEnvs environment_factory adapter. Test: python -m pytest train/tests.

Both training evaluation and submission use permit_core.evaluate. Native grading
belongs to the separate evaluation harness, never this adapter.
"""

import copy
from decimal import Decimal
import json
import math
from pathlib import Path
from typing import Any, Callable

from permit_core.cost import cost
from permit_core.evaluate import evaluate
from permit_core.reward import Anchors, reward
from permit_openenv.prompts import INSTRUCTIONS, PROMPT_VERSION
from permit_openenv.protocol import DESIGN_SCHEMA, json_value
from permit_openenv.tasks import Task, TaskPack, load_pack, to_ledger

TOOL_NAMES = ('get_scenario', 'check_design', 'evaluate_design', 'submit_design')


def training_instructions(max_turns: int | None = None) -> str:
    """Keep the environment instructions, substituting only the trainer budget."""
    if max_turns is None:
        return INSTRUCTIONS
    return INSTRUCTIONS.replace('you have 30 evaluations', f'you have {max_turns} tool turns')


def resolve_anchors(task: Task, *, smoke: bool = False,
                    diagnostics_dir: Path | None = None) -> dict:
    raw = task.record.get('anchors') or task.scenario.task.get('anchors')
    if raw and raw.get('best') is not None:
        source = str(raw.get('source') or raw.get('anchor_source') or task.record.get('anchor_source') or '').lower()
        if (source in ('', 'none') or raw.get('provisional')
                or any(word in source for word in ('search', 'baseline', 'provisional'))):
            raise ValueError('Regular task anchors must come from model calibration')
        anchors = Anchors.from_dict({**raw, 'source': source})
        return {k: (float(v) if k != 'source' else v) for k, v in anchors.to_dict().items()} | {'provisional': False}
    if not smoke:
        raise ValueError(f'{task.task_id}: missing model calibration anchors; provisional anchors are smoke-only')
    candidates = []
    recorded = dict(task.record)
    if diagnostics_dir is not None:
        path = diagnostics_dir / f'{task.task_id}.json'
        recorded = json.loads(path.read_text())
        if recorded.get('id', recorded.get('task_id')) != task.task_id or recorded.get('split') != task.split:
            raise ValueError('diagnostic task identity mismatch')
    for entry in [*recorded.get('baselines', {}).values(), recorded.get('search', {})]:
        if entry.get('feasible') is not True or isinstance(entry.get('usd_saved'), bool):
            continue
        try:
            saved = Decimal(str(entry['usd_saved']))
        except (KeyError, ArithmeticError):
            continue
        if saved.is_finite() and saved > 0:
            candidates.append(saved)
    if not candidates:
        raise ValueError(f'{task.task_id}: no positive feasible recorded baseline/search savings for smoke anchor')
    anchors = Anchors(Decimal(0), max(candidates), source='provisional_baseline_search_smoke')
    return {k: (float(v) if k != 'source' else v) for k, v in anchors.to_dict().items()} | {'provisional': True}


def tool_schemas() -> list[dict]:
    return [{'type': 'function', 'function': {
        'name': name, 'description': getattr(PermitTrainingEnv, name).__doc__.split('\n')[0],
        'parameters': {'type': 'object', 'properties': {} if name == 'get_scenario' else {
            'design': copy.deepcopy(DESIGN_SCHEMA)},
            'required': [] if name == 'get_scenario' else ['design']},
    }} for name in TOOL_NAMES]


def scenario_text(task: Task) -> str:
    """Compact public tables at reset; the unchanged JSON remains a tool away."""
    view = task.view()
    pollutants = ('NO2', 'PM25', 'SO2')
    lines = [f"Task {task.task_id}; weather {view['station']}; years {view['years']}.",
             f"Stack price: {view['stack_usd_per_m']} USD/year/m above height_min.",
             'Sources: id | type | x,y (m) | uncontrolled NO2,PM25,SO2 (g/s) | '
             'in-stack NO2 ratio | height_min,GEP (m) | height menu (m)']
    for source in view['sources']:
        lines.append(' | '.join(map(str, [source['id'], source['type'], [source['x_m'], source['y_m']],
            [source['uncontrolled_g_s'][p] for p in pollutants], source['no2_in_stack_ratio'],
            [source['height_min_m'], source['gep_m']], source['height_menu_m']])))
    lines.append('Controls: id | applies_to | annualized USD | efficiencies NO2,PM25,SO2')
    for control in view['controls']:
        lines.append(' | '.join(map(str, [control['id'], ','.join(control['applies_to']),
            control['annualized_usd'], [control['efficiency'][p] for p in pollutants]])))
    lines.append('Standards: id | form | limit | background (concentrations in ug/m3)')
    for standard in view['standards']:
        lines.append(' | '.join(str(standard[k]) for k in ('id', 'form', 'limit', 'background')))
    lines.append(f"Draft annualized USD: {view['draft_cost']}; draft design: " +
                 json.dumps(view['draft_design'], separators=(',', ':')))
    return '\n' + '\n'.join(lines)


class PermitTrainingEnv:
    """Four tool methods plus TRL's reset/get_reward lifecycle hooks."""

    def __init__(self, *, pack: TaskPack | None = None,
                 fields_loader: Callable[[Task], Any] | None = None,
                 smoke: bool = False, diagnostics_dir: Path | None = None,
                 trace_sink: Callable[[dict], None] | None = None):
        self._pack = pack if pack is not None else load_pack()
        self._fields_loader = fields_loader
        self._smoke, self._diagnostics_dir = smoke, diagnostics_dir
        self._trace_sink = trace_sink
        self._task: Task | None = None
        self._done = False

    def reset(self, split: str = 'train', index: int | None = None,
              task_id: str | None = None, **row: Any) -> str:
        """Start the dataset row's task and return its public scenario text."""
        if split != 'train':
            raise ValueError('Training accepts the train split only')
        task = self._pack.get(task_id) if task_id is not None else self._pack.at(split, 0 if index is None else index)
        if task.split != 'train' or task.record.get('eligible') is not True:
            raise ValueError('Training requires an eligible train task')
        anchors = resolve_anchors(task, smoke=self._smoke, diagnostics_dir=self._diagnostics_dir)
        self._task, self._anchors = task, anchors
        self._done, self._grading_failed, self._flushed = False, False, False
        self._evaluations_used = 0
        self._steps: list[dict] = []
        self._final: dict | None = None
        return scenario_text(task)

    def get_reward(self) -> float:
        """Return a calibrated grade; unavailable grading is NaN, no submission is zero."""
        if self._task is None:
            raise RuntimeError('reset must precede get_reward')
        score = math.nan if self._grading_failed else (self._final['reward'] if self._final else 0.0)
        if self._trace_sink is not None and not self._flushed:
            self._trace_sink(self._record())
            self._flushed = True
        return score

    def get_scenario(self) -> str:
        """Read sources, controls, heights, standards, draft and remaining evaluations."""
        return self._dispatch('get_scenario')

    def check_design(self, design: dict[str, dict[str, str | float]]) -> str:
        """Check design validity and exact cost for free, without testing air quality.

        Args:
            design: Every source ID mapped to a control ID and a menu height_m.
        """
        return self._dispatch('check_design', design)

    def evaluate_design(self, design: dict[str, dict[str, str | float]]) -> str:
        """Return feasibility and binding standards; valid designs spend one evaluation.

        Args:
            design: Every source ID mapped to a control ID and a menu height_m.
        """
        return self._dispatch('evaluate_design', design)

    def submit_design(self, design: dict[str, dict[str, str | float]]) -> str:
        """Submit the final design once and end the episode.

        Args:
            design: Every source ID mapped to a control ID and a menu height_m.
        """
        return self._dispatch('submit_design', design)

    def _evaluate(self, design: dict) -> dict:
        if self._fields_loader is None:
            raise RuntimeError('An existing field pack loader is required for training')
        result = evaluate(self._task.scenario, design, self._fields_loader(self._task))
        if not result.get('valid') or type(result.get('feasible')) is not bool:
            raise RuntimeError('Fast evaluator did not return a grade')
        return result

    def _dispatch(self, name: str, design: dict | None = None) -> str:
        if self._task is None:
            raise RuntimeError('reset must precede tools')
        left = 30 - self._evaluations_used
        if self._done:
            return json.dumps({'error': 'Episode is over. End your response.', 'evaluations_left': left})
        if name == 'get_scenario':
            result = self._task.view(left)
        else:
            checked = self._task.check(design)
            result = {**checked, 'evaluations_left': left}
            if name == 'submit_design':
                self._done = True
                result.update(feasible=False, standards=[])
                if checked['valid']:
                    try:
                        result.update(self._evaluate(design))
                    except Exception:
                        self._grading_failed = True
                        result['error'] = 'Grading unavailable'
                exact = cost(to_ledger(design), self._task.scenario.task) if checked['valid'] else None
                saved = self._task.draft_cost - exact if result['feasible'] else Decimal(-1)
                result.update(reward=None if self._grading_failed else reward(saved, self._anchors),
                              prescore_usd_saved=float(saved), anchors=self._anchors,
                              grading_failed=self._grading_failed, training_grade='fast')
                self._final = copy.deepcopy(result)
            elif checked['valid'] and name == 'evaluate_design':
                if left == 0:
                    result = {'error': 'No evaluations left; submit your design', 'evaluations_left': 0}
                else:
                    self._evaluations_used += 1
                    try:
                        result = self._evaluate(design)
                    except Exception:
                        self._grading_failed = True
                        result = {'error': 'Evaluation unavailable'}
                    result['evaluations_left'] = left - 1
        result = json_value(result)
        self._steps.append(copy.deepcopy({'tool': name, 'args': {} if name == 'get_scenario' else {'design': design},
                                         'result': result}))
        return json.dumps(result, separators=(',', ':'), allow_nan=False)

    def _record(self) -> dict:
        return copy.deepcopy(json_value({
            'schema_version': 1, 'task_id': self._task.task_id, 'split': self._task.split,
            'prompt_version': PROMPT_VERSION, 'scenario': self._task.view(),
            'steps': self._steps, 'final': self._final,
            'ended': 'submitted' if self._done else 'no_submit',
            'evaluations_used': self._evaluations_used, 'grading_failed': self._grading_failed,
            'anchors': self._anchors, 'provisional': self._anchors['provisional'],
        }))
