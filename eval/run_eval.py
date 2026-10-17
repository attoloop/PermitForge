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
import signal
import platform
import subprocess
import threading
from typing import Any, Callable

from .common import ROOT, component, finite, model_slug, now, records, write_json, write_text
from .capture import EpisodeCapture, append_json, append_log


def provenance(models: list[str], settings: dict, concurrency: int, budget: float) -> dict:
    from permit_openenv.prompts import INSTRUCTIONS, PROMPT_VERSION
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return {'git_commit': commit, 'git_dirty': bool(dirty), 'prompt_version': PROMPT_VERSION,
            'prompt_sha256': hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
            'protocol_sha256': hashlib.sha256(Path(__file__).with_name('PROTOCOL.md').read_bytes()).hexdigest(),
            'models': models, 'sampling': {key: settings.get(key, default) for key, default in
                [('max_turns', 40), ('max_tokens', 4096), ('wall_clock', 1800), ('temperature', 'provider-default')]},
            'concurrency': concurrency, 'budget_usd': budget, 'started_at': now(), 'finished_at': None,
            'host': {'name': platform.node(), 'platform': platform.platform(), 'python': platform.python_version()}}


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
        try:
            yield
        except BaseException as exc:
            manifest = json.loads((run/'run.json').read_text()) if (run/'run.json').exists() else {}
            if manifest.get('status') == 'running':
                manifest.update(finished_at=now(), status='interrupted')
                write_json(run/'run.json', manifest)
                append_log(run/'harness.log', now()+f' run interrupted: {type(exc).__name__}')
                charged = {(row['model'], row['task_id']) for row in
                           (json.loads(line) for line in (run/'costs.jsonl').read_text().splitlines())}
                for path, record in records([run]):
                    if (record['model'], record['task_id']) in charged:
                        continue
                    record.update(finished_at=record.get('finished_at') or now(),
                                  ended='submitted' if record.get('final') is not None else 'no_submit',
                                  error=record.get('error') or 'harness terminated')
                    write_json(path, record)
                    if path.name == 'episode.json':
                        write_json(path.parent.parent/(path.parent.name+'.json'), record)
                        capture = EpisodeCapture(path.parent)
                        capture.last_diagnostic = float('inf')
                        capture.finish(record)
                    append_json(run/'costs.jsonl', {'at': now(), 'model': record['model'],
                                'task_id': record['task_id'], 'usage': record.get('usage'),
                                'budget_charge_usd': record.get('budget_charge_usd')})
                write_index(run, status='interrupted')
            raise


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
        selection = {'models': models, 'tasks': tasks, 'settings': settings or {}}
        path = run/'run.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            if any(manifest.get(k) != v for k, v in selection.items()):
                raise ValueError('resume settings or task/model cohort changed')
        else:
            manifest = {**provenance(models, settings or {}, concurrency, budget_usd), **selection}
        manifest.update(status='running', finished_at=None)
        write_json(path, manifest)
        append_log(run/'harness.log', now()+' run started/resumed')
        (run/'costs.jsonl').touch(exist_ok=True)
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
                    if destination.exists() or (destination.with_suffix('')/'episode.json').exists():
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
                               'prompt_version': manifest['prompt_version'], 'started_at': now(), 'finished_at': None,
                               'ended': 'error', 'error_type': 'interrupted', 'attempt_complete': False,
                               'usage': {'input_tokens': 0, 'output_tokens': 0, 'usd': None},
                               'budget_charge_usd': budget.active[key]}
                    write_json(destination, initial)
                    capture = EpisodeCapture(destination.with_suffix(''))
                    write_json(capture.root/'episode.json', initial)
                    append_log(run/'harness.log', now()+f' episode started {model} {task["task_id"]}')
                    def checkpoint(partial, destination=destination, model=model, capture=capture):
                        current = json.loads(destination.read_text())
                        updated = {**current, **partial, 'model': model, 'attempt_complete': False}
                        write_json(destination, updated)
                        write_json(capture.root/'episode.json', updated)
                        capture.environment(updated)
                    kwargs = {'checkpoint': checkpoint} if 'checkpoint' in inspect.signature(driver).parameters else {}
                    if 'capture' in inspect.signature(driver).parameters:
                        kwargs['capture'] = capture
                    jobs.append((pool.submit(driver, model, task, **kwargs), key, destination, model, task, capture))
                collect(jobs, run, budget)
                if stop:
                    break
        write_index(run, status='budget_stopped' if stop else 'complete')
        manifest.update(finished_at=now(), status='budget_stopped' if stop else 'complete')
        write_json(path, manifest)
        append_log(run/'harness.log', now()+' run '+manifest['status'])


def collect(jobs: list, run: Path, budget: Budget) -> None:
    mapping = {job[0]: job[1:] for job in jobs}
    for future in as_completed(mapping):
        key, path, model, task, capture = mapping[future]
        # A driver bug is not an infrastructure retry; surface it instead of synthesizing a grade.
        try:
            record = future.result()
        except BaseException as exc:
            partial = json.loads(path.read_text())
            partial.update(finished_at=partial.get('finished_at') or now(), error=str(exc),
                           ended='submitted' if partial.get('final') is not None else 'error')
            capture.finish(partial)
            write_json(path, partial)
            append_json(run/'costs.jsonl', {'at': now(), 'model': model, 'task_id': task['task_id'],
                        'usage': partial.get('usage'), 'budget_charge_usd': partial.get('budget_charge_usd')})
            append_log(run/'harness.log', now()+f' episode interrupted {model} {task["task_id"]}: {exc}')
            manifest = json.loads((run/'run.json').read_text())
            manifest.update(finished_at=now(), status='interrupted')
            write_json(run/'run.json', manifest)
            write_index(run, status='interrupted')
            raise
        record.update(run=run.name, attempt_complete=True)
        record['budget_charge_usd'] = budget.finish(key, record.get('usage', {}).get('usd'))
        write_json(path, record)
        capture.finish(record)
        append_json(run/'costs.jsonl', {'at': now(), 'model': model, 'task_id': task['task_id'],
                    'usage': record.get('usage'), 'budget_charge_usd': record['budget_charge_usd']})
        append_log(run/'harness.log', now()+f' episode ended {model} {task["task_id"]}: {record["ended"]}')
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
    from .transport import terminate_sessions
    def terminate(signum, frame):
        terminate_sessions()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--split', choices=['eval', 'train'], default='eval')
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--max-turns', type=int, default=40)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--wall-clock', type=float, default=1800)
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
        def driver(model, task, checkpoint, capture):
            with Session(url, token, task['task_id'], timeout=args.wall_clock, checkpoint=checkpoint, capture=capture) as session:
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
