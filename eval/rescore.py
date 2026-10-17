"""Reverify saved submissions: python -m eval.rescore RUN ... --tasks-root PATH."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "eval"

import argparse
import copy
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from permit_core.reward import reward

from .common import ROOT, now, records, run_path, write_json


def native_verify(task: Any, design: dict) -> dict:
    from permit_openenv.evaluator import VerifyEvaluator
    checked = task.check(design)
    if not checked['valid']:
        return {**checked, 'feasible': False, 'cost': None, 'standards': []}
    # VerifyEvaluator invokes permit_core.verify.verify with ARM2 and the global native slot.
    return VerifyEvaluator().evaluate(task, design)


def rescore_record(record: dict, task: Any, *, verify_fn: Callable = native_verify) -> dict:
    out = copy.deepcopy(record)
    out.pop('anchors', None)
    submission = next((s for s in out['steps'] if s['tool'] == 'submit_design'), None)
    if submission is None:
        out['final'] = None
        out['reward'] = 0.0
        return out
    previous = {'rescored_at': now(), 'reward': out.get('reward'), 'final': copy.deepcopy(out.get('final')),
                'submission': copy.deepcopy(submission)}
    evaluation = verify_fn(task, submission['args'].get('design'))
    errors = evaluation.get('errors', [])
    # Preserve the old grade on infrastructure failure, rather than freezing a false gate failure.
    if evaluation.get('valid', True) and not evaluation.get('feasible') and not any(
            code in str(errors) for code in ('E_NAAQS', 'E_GEP', 'E_MENU')):
        raise RuntimeError(f'verification did not produce a grade: {errors}')
    feasible = bool(evaluation.get('feasible'))
    saved = Decimal(str(task.draft_cost))-Decimal(str(evaluation['cost'])) if feasible else Decimal(-1)
    final = {k:v for k,v in evaluation.items() if k not in ('receptor_values', 'receptor_margins')}
    final.update(prescore_usd_saved=float(saved),
                 pct_saved=float(100*saved/Decimal(str(task.draft_cost))) if feasible else 0.0,
                 reward=reward(saved, task.draft_cost),
                 evaluations_left=30-out.get('evaluations_used', 0))
    out.setdefault('rescore_history', []).append(previous)
    out.update(final=final, reward=final['reward'], ended='submitted')
    submission['result'] = final
    for key in ('receptor_values', 'receptor_margins'):
        submission.pop(key, None)
        if key in evaluation:
            submission[key] = evaluation[key]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--tasks-root', type=Path, default=ROOT/'envs/permit/tasks')
    args = parser.parse_args()
    from .transport import load_eval_pack
    from .run_eval import write_index
    pack = load_eval_pack(args.tasks_root)
    runs = [run_path(p) for p in args.runs]
    for path, record in records(runs):
        write_json(path, rescore_record(record, pack.get(record['task_id'])))
    for run in runs:
        write_index(run)


if __name__ == '__main__':
    main()
