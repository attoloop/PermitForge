"""Run a resumable panel: python -m eval.run_eval --run NAME --models codex:gpt-6.1-sol --split train --limit 1."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "eval"

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import secrets
import threading
from typing import Any, Callable

from .common import ROOT, component, finite, model_slug, now, records, write_json, write_text


class Budget:
    """Launch guard; reservations cover concurrent in-flight episodes, not a hard invoice cap."""
    def __init__(self, limit: float, *, concurrency: int, prior: float = 5.0, spent: float = 0,
                 costs: list[float] | None = None):
        if not finite(limit) or limit < 0 or not finite(prior) or prior <= 0 or not 1 <= concurrency <= 4:
            raise ValueError('budget and prior must be finite and nonnegative; prior positive; concurrency 1..4')
        self.limit, self.concurrency, self.prior, self.spent = limit, concurrency, prior, spent
        self.costs = list(costs or [])
        self.active: dict[str, float] = {}
        self.lock = threading.Lock()

    @property
    def estimate(self) -> float:
        if not self.costs:
            return self.prior
        ordered = sorted(self.costs)
        return ordered[max(0, math.ceil(.95*len(ordered))-1)]

    def reserve(self, key: str, paid: bool) -> bool:
        with self.lock:
            estimate = self.estimate if paid else 0
            if paid and self.spent+max(self.concurrency*estimate, sum(self.active.values())+estimate) > self.limit:
                return False
            self.active[key] = estimate
            return True

    def finish(self, key: str, cost: float | None) -> float:
        with self.lock:
            reservation = self.active.pop(key)
            charged = cost if cost is not None else reservation
            self.spent += charged
            if reservation or cost is not None:
                self.costs.append(charged)
            return charged


def write_index(run: Path, *, status: str | None = None) -> None:
    from .report import build_report, markdown
    items = [r for _,r in records([run])]
    index = {'run': run.name, 'episodes': [{k:r.get(k) for k in (
        'model', 'task_id', 'split', 'station', 'ended', 'turns', 'evaluations_used', 'usage', 'final', 'error')}
        for r in items]}
    if status:
        index['status'] = status
    write_json(run/'index.json', index)
    report = build_report(items)
    write_json(run/'board.json', report)
    write_text(run/'board.md', markdown(report))


@contextmanager
def run_lock(run: Path):
    run.mkdir(parents=True, exist_ok=True)
    with (run/'.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('another process owns this run') from None
        yield


def run_panel(run: Path, tasks: list[dict], models: list[str], driver: Callable,
              *, concurrency: int = 1, budget_usd: float = 0, prior: float = 5.0,
              force: bool = False, settings: dict | None = None) -> None:
    if not 1 <= concurrency <= 4:
        raise ValueError('concurrency must be between 1 and 4')
    if len(models) != len(set(models)):
        raise ValueError('duplicate models')
    if len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('duplicate tasks')
    with run_lock(run):
        existing = [r for _,r in records([run])]
        if force and existing:
            raise ValueError('--force cannot rerun a recorded attempt; use a separately disclosed run')
        manifest = {'models': models, 'tasks': tasks, 'settings': settings or {}}
        path = run/'run.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError('resume settings or task/model cohort changed')
        write_json(path, manifest)
        costs = [r.get('budget_charge_usd', r.get('usage', {}).get('usd')) for r in existing
                 if r['model'].startswith('anthropic:')]
        costs = [c if finite(c) else prior for c in costs]
        budget = Budget(budget_usd, concurrency=concurrency, prior=prior, spent=sum(costs), costs=costs)
        stop = False
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for task in tasks:  # Finish every model for this task before moving to the next pack.
                jobs = []
                for model in models:
                    destination = run/model_slug(model)/(component(task['task_id'])+'.json')
                    if destination.exists():
                        continue  # Includes no-submit, timeout, error, invalid and infeasible terminal records.
                    key = model+'|'+task['task_id']
                    if not budget.reserve(key, model.startswith('anthropic:')):
                        stop = True
                        continue
                    # Never queue more episodes than the reservation's configured concurrency.
                    if len(jobs) == concurrency:
                        collect(jobs, run, budget)
                        jobs.clear()
                    initial = {'schema_version': 1, **task, 'model': model, 'scenario': {}, 'geometry': {},
                               'steps': [], 'final': None, 'evaluations_used': 0, 'turns': 0,
                               'prompt_version': 1, 'started_at': now(), 'finished_at': None,
                               'ended': 'error', 'error_type': 'interrupted', 'attempt_complete': False,
                               'usage': {'input_tokens': 0, 'output_tokens': 0, 'usd': None},
                               'budget_charge_usd': budget.active[key]}
                    write_json(destination, initial)
                    def checkpoint(partial, destination=destination, model=model):
                        current = json.loads(destination.read_text())
                        write_json(destination, {**current, **partial, 'model': model,
                                                'attempt_complete': False, 'finished_at': None})
                    kwargs = {'checkpoint': checkpoint} if 'checkpoint' in inspect.signature(driver).parameters else {}
                    jobs.append((pool.submit(driver, model, task, **kwargs), key, destination, model, task))
                collect(jobs, run, budget)
                if stop:
                    break
        write_index(run, status='budget_stopped' if stop else 'complete')


def collect(jobs: list, run: Path, budget: Budget) -> None:
    mapping = {job[0]: job[1:] for job in jobs}
    for future in as_completed(mapping):
        key, path, model, task = mapping[future]
        # A driver bug is not an infrastructure retry; surface it instead of synthesizing a grade.
        try:
            record = future.result()
        except BaseException:
            write_index(run, status='interrupted')
            raise
        record.update(run=run.name, attempt_complete=True)
        record['budget_charge_usd'] = budget.finish(key, record.get('usage', {}).get('usd'))
        write_json(path, record)
        write_index(run)
        print(f"{model} {task['task_id']}: {record['ended']}", flush=True)


def select_tasks(pack: Any, split: str, ids: list[str] | None, limit: int | None) -> list[dict]:
    tasks = [t for t in pack.tasks if t.split == split and t.record.get('eligible') is True]
    if ids is not None:
        allowed = {t.task_id:t for t in tasks}
        if len(ids) != len(set(ids)) or set(ids)-allowed.keys():
            raise ValueError('tasks must be unique eligible members of the requested split')
        tasks = [allowed[t] for t in ids]
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        tasks = tasks[:limit]
    return [{'task_id': t.task_id, 'split': t.split, 'station': t.record['station']} for t in tasks]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--split', choices=['eval', 'train'], default='eval')
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--max-turns', type=int, default=40)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--wall-clock', type=float, default=7200)
    parser.add_argument('--budget-usd', type=float, default=0)
    parser.add_argument('--prior-episode-usd', type=float, default=5)
    parser.add_argument('--server')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--tasks-root', type=Path, default=ROOT/'envs/permit/tasks')
    parser.add_argument('--output-root', type=Path, default=ROOT/'results/rollouts')
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 4 or args.max_turns < 1 or args.max_turns > 40 or args.max_tokens < 1 or not finite(args.wall_clock) or args.wall_clock <= 0:
        parser.error('invalid concurrency, turn, token, or wall-clock limit')
    Budget(args.budget_usd, concurrency=args.concurrency, prior=args.prior_episode_usd)
    for model in args.models:
        provider, _, name = model.partition(':')
        if provider not in ('codex', 'anthropic') or not name:
            parser.error('supported drivers: anthropic:<model>, codex:<model>')
    paid = any(m.startswith('anthropic:') for m in args.models)
    if paid and (args.budget_usd <= 0 or not os.environ.get('ANTHROPIC_API_KEY')):
        parser.error('Anthropic requires ANTHROPIC_API_KEY and a positive --budget-usd')
    from .transport import Session, load_eval_pack, recording_app, serve
    from .anthropic_driver import run_anthropic
    from .codex_driver import run_codex
    from permit_openenv.prompts import INSTRUCTIONS, PROMPT_VERSION
    pack = load_eval_pack(args.tasks_root)
    tasks = select_tasks(pack, args.split, args.tasks, args.limit)
    if not tasks:
        parser.error('no eligible tasks selected')
    token = os.environ.get('PERMIT_EVAL_TOKEN') if args.server else secrets.token_urlsafe(32)
    if not token:
        parser.error('--server requires PERMIT_EVAL_TOKEN and eval.transport recorder routes')

    def run(url):
        def driver(model, task, checkpoint):
            with Session(url, token, task['task_id'], timeout=args.wall_clock, checkpoint=checkpoint) as session:
                fn = run_anthropic if model.startswith('anthropic:') else run_codex
                return fn(session, model, max_turns=args.max_turns,
                          max_tokens=args.max_tokens, wall_clock=args.wall_clock)
        settings = {'prompt_version': PROMPT_VERSION, 'prompt_sha256': hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
                    'max_turns': args.max_turns, 'max_tokens': args.max_tokens,
                    'wall_clock': args.wall_clock, 'temperature': 'provider-default',
                    'evaluate_budget': 30, 'concurrency': args.concurrency,
                    'budget_usd': args.budget_usd, 'prior_episode_usd': args.prior_episode_usd}
        run_panel(args.output_root/component(args.run), tasks, args.models, driver,
                  concurrency=args.concurrency, budget_usd=args.budget_usd,
                  prior=args.prior_episode_usd, force=args.force, settings=settings)
    if args.server:
        run(args.server)
    else:
        with serve(recording_app(token, pack=pack)) as url:
            run(url)


if __name__ == '__main__':
    main()
