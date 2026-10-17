"""Run a resumable panel: python -m eval.run_eval --run NAME --models codex:gpt-6.1-sol --split train --limit 1."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "eval"

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextlib import contextmanager, nullcontext
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
import traceback
from typing import Any, Callable

from .common import ROOT, component, finite, model_slug, now, records, write_json, write_text
from .capture import EpisodeCapture, append_json, append_log
from . import attempts, preflight
from .settings import EFFORTS, EPISODE_PRIORS, MAX_TOKENS, MODELS, PRIOR_ASSUMPTION, model_settings
from .schedule import phases, projection
from .budget import HardBudget, request_bound


def provenance(models: list[str], settings: dict, concurrency: int, budget: float) -> dict:
    from permit_openenv.prompts import INSTRUCTIONS, PROMPT_VERSION
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    effort = settings.get('effort', 'xhigh')
    sampling = {key: settings.get(key, default) for key, default in
                [('max_turns', 40), ('max_tokens', MAX_TOKENS), ('wall_clock', 1800),
                 ('temperature', 'provider-default'), ('effort', effort)]}
    sampling['models'] = {m: model_settings(m, effort) for m in models if m in MODELS}
    protocol = Path(__file__).with_name('PROTOCOL.md').read_bytes()
    digest = hashlib.sha256(protocol + b'\n' + json.dumps(sampling, sort_keys=True).encode()).hexdigest()
    return {'git_commit': commit, 'git_dirty': bool(dirty), 'prompt_version': PROMPT_VERSION,
            'prompt_sha256': hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
            'protocol_sha256': digest,
            'protocol_document_sha256': hashlib.sha256(protocol).hexdigest(),
            'reporting_sha256': hashlib.sha256(Path(__file__).with_name('REPEATS.md').read_bytes()).hexdigest(),
            'reward_definition': 'saved_fraction',
            'models': models, 'sampling': sampling,
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
    from .report import build_report, markdown, write_variance
    items = [r for _,r in records([run])]
    index = {'run': run.name, 'episodes': [{k:r.get(k) for k in (
        'model', 'task_id', 'repeat', 'split', 'station', 'ended', 'turns', 'evaluations_used', 'usage', 'final', 'error')}
        for r in items]}
    index['infrastructure_attempts'] = attempts.infrastructure(run)
    if status:
        index['status'] = status
    write_json(run/'index.json', index)
    splits = sorted({r['split'] for r in items})
    if len(splits) > 1:
        reports = {s: build_report([r for r in items if r['split'] == s]) for s in splits}
        task_splits = {r['task_id']: r['split'] for r in items}
        if (run/'run.json').exists():
            task_splits.update({t['task_id']: t['split'] for t in json.loads((run/'run.json').read_text()).get('tasks', [])})
        for split, report in reports.items():
            report['infrastructure_attempts'] = [r for r in index['infrastructure_attempts']
                                                  if task_splits.get(r['task_id']) == split]
            write_json(run/f'board-{split}.json', report)
            write_text(run/f'board-{split}.md', markdown(report))
            write_variance(run/f'variance-{split}', report)
        write_json(run/'board.json', reports)
        write_text(run/'board.md', '\n\n'.join(f'## {s}\n\n'+markdown(r) for s, r in reports.items()))
        return
    report = build_report(items)
    report['infrastructure_attempts'] = index['infrastructure_attempts']
    write_json(run/'board.json', report)
    write_variance(run/'variance', report)
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
                attempts.recover(run, error=str(exc), trace=traceback.format_exc())
                write_index(run, status='interrupted')
            raise


def run_panel(run: Path, tasks: list[dict], models: list[str], driver: Callable,
              *, concurrency: int = 1, budget_usd: float = 0, prior: float = 5.0,
              force: bool = False, settings: dict | None = None,
              repeats: int = 1, import_runs: list[Path] | None = None,
              order: str = 'task-major', codex_concurrency: int | None = None,
              anthropic_concurrency: int | None = None) -> None:
    if type(repeats) is not int or repeats < 1:
        raise ValueError('repeats must be positive')
    if not 1 <= concurrency <= 4:
        raise ValueError('concurrency must be between 1 and 4')
    if len(models) != len(set(models)):
        raise ValueError('duplicate models')
    if len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('duplicate tasks')
    limits = {'codex': codex_concurrency or concurrency, 'anthropic': anthropic_concurrency or concurrency}
    if any(not 1 <= v <= 8 for v in limits.values()):
        raise ValueError('provider concurrency must be 1..8')
    if codex_concurrency is not None or anthropic_concurrency is not None:
        concurrency = sum(limits[p] for p in limits if any(m.startswith(p+':') for m in models))
        if concurrency > 12:  # episodes mostly wait on provider APIs; native submits are single-core
            raise ValueError('combined concurrency must be at most 12')
    schedule_phases = phases(tasks, models, repeats, order)
    schedule = [job for phase in schedule_phases for job in phase]
    with run_lock(run):
        attempts.recover(run)
        existing = [r for _,r in records([run])]
        if force and (existing or attempts.infrastructure(run)):
            raise ValueError('--force cannot rerun a recorded attempt; use a separately disclosed run')
        selection = {'models': models, 'tasks': tasks, 'settings': settings or {},
                     'order': order, 'provider_concurrency': limits}
        current = provenance(models, settings or {}, concurrency, budget_usd)
        path = run/'run.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            if any(manifest.get(k) != v for k, v in selection.items()):
                raise ValueError('resume settings or task/model cohort changed')
            if manifest.get('protocol_sha256') != current['protocol_sha256']:
                raise ValueError('resume protocol hash changed')
            if repeats < manifest.get('repeats', 1):
                raise ValueError('cannot reduce repeats on resume')
        else:
            manifest = {**current, **selection}
        from .import_runs import import_completed
        import_completed(run, import_runs or [], manifest, repeats)
        existing = [r for _,r in records([run])]
        manifest.update(status='running', finished_at=None, repeats=repeats)
        write_json(path, manifest)
        write_json(run/'schedule.json', schedule)
        append_log(run/'harness.log', now()+' run started/resumed')
        (run/'costs.jsonl').touch(exist_ok=True)
        # Count each paid attempt, including failed work, once on resume.
        failed = attempts.infrastructure(run)
        billed = existing + [r for r in failed if not r.get('submission_preserved')]
        costs = [r.get('budget_charge_usd', (r.get('usage') or {}).get('usd')) for r in billed
                 if r['model'].startswith('anthropic:')]
        costs = [c if finite(c) else prior for c in costs]
        budget = Budget(budget_usd, concurrency=limits['anthropic'], prior=prior, spent=sum(costs), costs=costs)
        hard_budget = HardBudget(run/'request-budget.json', budget_usd)
        stop = False
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for phase_index, phase in enumerate(schedule_phases):
                jobs = []
                pending = list(phase)
                while pending:
                    occupied = {p: sum(j[3].startswith(p+':') for j in jobs) for p in limits}
                    available = next((i for i, e in enumerate(pending)
                                      if occupied[e['model'].partition(':')[0]] < limits[e['model'].partition(':')[0]]), None)
                    if jobs and (len(jobs) == concurrency or available is None):
                        done, _ = wait([j[0] for j in jobs], return_when=FIRST_COMPLETED)
                        collect([j for j in jobs if j[0] in done], run, budget)
                        jobs = [j for j in jobs if j[0] not in done]
                        continue
                    entry = pending.pop(available)
                    model, repeat = entry['model'], entry['repeat']
                    task = {k: v for k, v in entry.items() if k != 'model'}
                    task_root = run/model_slug(model)/component(task['task_id'])
                    if repeat == 1 and ((task_root/'episode.json').exists() or task_root.with_suffix('.json').exists()):
                        continue
                    destination = task_root/f'r{repeat}.json'
                    if repeat == 1 and attempts.history(task_root):
                        destination = task_root.with_suffix('.json')
                    if destination.exists() or (destination.with_suffix('')/'episode.json').exists():
                        continue  # Includes no-submit, timeout, error, invalid and infeasible terminal records.
                    root = destination.with_suffix('')
                    history = attempts.history(root)
                    if history and (len(history) >= attempts.MAX_ATTEMPTS or not history[-1].get('retryable')):
                        continue
                    key = model+'|'+task['task_id']+'|'+str(repeat)
                    if model in MODELS[:2] and not hard_budget.can_reserve(
                            request_bound(model.partition(':')[2], (settings or {}).get('max_tokens', MAX_TOKENS))):
                        stop = True
                        continue
                    if not budget.reserve(key, model.startswith('anthropic:')):
                        stop = True
                        continue
                    number = len(history)+1
                    initial = {'schema_version': 1, **task, 'model': model, 'scenario': {}, 'geometry': {},
                               'steps': [], 'final': None, 'evaluations_used': 0, 'turns': 0,
                               'prompt_version': manifest['prompt_version'], 'started_at': now(), 'finished_at': None,
                               'ended': 'error', 'error_type': 'interrupted', 'attempt_complete': False,
                               'attempt_number': number, 'run': run.name,
                               'settings': {**(settings or {}), **(model_settings(
                                   model, (settings or {}).get('effort', 'xhigh')) if model in MODELS else {})},
                               'usage': {'input_tokens': 0, 'output_tokens': 0, 'usd': None},
                               'budget_charge_usd': budget.active[key]}
                    capture = EpisodeCapture(root/'attempts'/f'episode-{number:04d}')
                    checkpoint_path = capture.root/'checkpoint.json'
                    write_json(checkpoint_path, initial)
                    write_json(capture.root/'attempt.json', {
                        'model': model, 'task_id': task['task_id'], 'repeat': repeat, 'attempt_number': number,
                        'status': 'running', 'started_at': initial['started_at']})
                    append_log(run/'harness.log', now()+f' episode started {model} {task["task_id"]} r{repeat}')
                    def checkpoint(partial, destination=checkpoint_path, model=model, capture=capture):
                        current = json.loads(destination.read_text())
                        updated = {**current, **partial, 'model': model, 'attempt_complete': False}
                        write_json(destination, updated)
                        capture.environment(updated)
                    kwargs = {'checkpoint': checkpoint} if 'checkpoint' in inspect.signature(driver).parameters else {}
                    if 'capture' in inspect.signature(driver).parameters:
                        kwargs['capture'] = capture
                    if 'hard_budget' in inspect.signature(driver).parameters:
                        kwargs['hard_budget'] = hard_budget
                    jobs.append((pool.submit(driver, model, task, **kwargs), key, destination, model, task, capture))
                collect(jobs, run, budget)
                next_repeat = schedule_phases[phase_index+1][0]['repeat'] if phase_index+1 < len(schedule_phases) else None
                if order == 'round-robin' and phase[0]['repeat'] == 1 and next_repeat != 1:
                    projected = projection(schedule, [r for _, r in records([run])], prior, after_round=1)
                    if not (run/'projection.json').exists():
                        write_json(run/'projection.json', projected)
                        append_log(run/'harness.log', now()+' round 1 projection '+json.dumps(projected, sort_keys=True))
        completed = {(r['model'], r['task_id'], r.get('repeat', 1)) for _, r in records([run])}
        missing = any((model, task['task_id'], k) not in completed
                      for model in models for task in tasks for k in range(1, repeats+1))
        status = 'budget_stopped' if stop else 'infrastructure_failed' if missing else 'complete'
        write_index(run, status=status)
        manifest.update(finished_at=now(), status=status)
        write_json(path, manifest)
        append_log(run/'harness.log', now()+' run '+manifest['status'])


def collect(jobs: list, run: Path, budget: Budget) -> None:
    mapping = {job[0]: job[1:] for job in jobs}
    for future in as_completed(mapping):
        key, path, model, task, capture = mapping[future]
        error, trace = None, ''
        try:
            record = future.result()
            if not isinstance(record, dict) or record.get('ended') not in ('submitted', 'no_submit', 'turn_limit', 'error'):
                raise ValueError('driver did not return an episode record')
            usage = record.get('usage', {})
            if not isinstance(usage, dict) or (usage.get('usd') is not None and (
                    not finite(usage['usd']) or usage['usd'] < 0)):
                raise ValueError('driver returned invalid usage')
            if record['ended'] == 'submitted' and not isinstance(record.get('final'), dict):
                raise ValueError('submitted episode has no final grade')
            json.dumps(record, allow_nan=False)
        except Exception as exc:
            error, trace = str(exc), traceback.format_exc()
            record = {}
        partial = json.loads((capture.root/'checkpoint.json').read_text())
        record = {**partial, **record, 'model': model, **task}
        if error is None and (record.get('infrastructure_error') or (
                record.get('ended') == 'error' and record.get('error_type') != 'timeout')):
            error = str(record.get('error') or record.get('error_type') or 'driver returned an infrastructure error')
            trace = record.get('traceback') or 'No Python traceback: driver reported an external process/provider failure.'
        record['budget_charge_usd'] = budget.finish(key, record.get('usage', {}).get('usd'))
        root = path.with_suffix('')
        if error is not None:
            attempts.fail(root, capture, record, error, trace)
            ended = 'infrastructure'
        else:
            record.update(run=run.name, attempt_complete=True)
            write_json(capture.root/'checkpoint.json', record)
            capture.finish(record)
            attempts.publish(root, record, capture)
            write_json(capture.root/'attempt.json', {
                'model': model, 'task_id': task['task_id'], 'repeat': task['repeat'], 'attempt_number': record['attempt_number'],
                'status': 'complete', 'finished_at': now(), 'retryable': False})
            ended = record['ended']
        append_json(run/'costs.jsonl', {'at': now(), 'model': model, 'task_id': task['task_id'], 'repeat': task['repeat'],
                    'attempt_number': record['attempt_number'], 'status': ended,
                    'usage': record.get('usage'), 'budget_charge_usd': record['budget_charge_usd']})
        append_log(run/'harness.log', now()+f' episode ended {model} {task["task_id"]} r{task["repeat"]}: {ended}')
        write_index(run)
        print(f"{model} {task['task_id']} r{task['repeat']}: {ended}", flush=True)


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
    parser.add_argument('--split', choices=['eval', 'train', 'both'], default='eval')
    parser.add_argument('--subsets', type=Path)
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--import-run', action='append', default=[])
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--codex-concurrency', type=int, default=1)
    parser.add_argument('--anthropic-concurrency', type=int, default=1)
    parser.add_argument('--order', choices=['task-major', 'round-robin'], default='round-robin')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--effort', choices=EFFORTS, default='xhigh')
    parser.add_argument('--max-turns', type=int, default=40)
    parser.add_argument('--max-tokens', type=int, default=MAX_TOKENS)
    parser.add_argument('--wall-clock', type=float, default=1800)
    parser.add_argument('--budget-usd', type=float, default=0)
    parser.add_argument('--prior-episode-usd', type=float)
    parser.add_argument('--server')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--tasks-root', type=Path, default=ROOT/'envs/permit/tasks')
    parser.add_argument('--output-root', type=Path, default=ROOT/'results/rollouts')
    args = parser.parse_args()
    if args.prior_episode_usd is None:
        args.prior_episode_usd = EPISODE_PRIORS[args.effort]
    if args.max_tokens != MAX_TOKENS:
        parser.error('final protocol requires --max-tokens 32768')
    if not 1 <= args.concurrency <= 4 or args.max_turns < 1 or args.max_turns > 40 or args.max_tokens < 1 or not finite(args.wall_clock) or args.wall_clock <= 0:
        parser.error('invalid concurrency, turn, token, or wall-clock limit')
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if any(not 1 <= c <= 8 for c in (args.codex_concurrency, args.anthropic_concurrency)):
        parser.error('provider concurrency must be 1..8')
    used_concurrency = sum(c for p, c in [('codex', args.codex_concurrency), ('anthropic', args.anthropic_concurrency)]
                           if any(m.startswith(p+':') for m in args.models))
    if used_concurrency > 12:
        parser.error('combined provider concurrency must be at most 12')
    if args.budget_usd > 190:
        parser.error('Anthropic hard cap is 190 USD')
    Budget(args.budget_usd, concurrency=args.concurrency, prior=args.prior_episode_usd)
    for model in args.models:
        provider, _, name = model.partition(':')
        if provider not in ('codex', 'anthropic') or not name:
            parser.error('supported drivers: anthropic:<model>, codex:<model>')
        if model not in MODELS:
            parser.error('WP2.8 models are: '+', '.join(MODELS))
    paid = any(m.startswith('anthropic:') for m in args.models)
    if paid and args.budget_usd <= 0:
        parser.error('Anthropic requires a positive --budget-usd')
    from .transport import Session, load_eval_pack, recording_app, serve
    from permit_openenv.prompts import INSTRUCTIONS, PROMPT_VERSION
    try:
        pack = load_eval_pack(args.tasks_root)
        if args.subsets:
            if args.tasks or args.limit:
                raise ValueError('--subsets cannot be combined with --tasks or --limit')
            subsets = json.loads(args.subsets.read_text())
            from .cohort import validate_subsets
            validate_subsets(subsets)
            splits = ('eval', 'train') if args.split == 'both' else (args.split,)
            tasks = [t for split in splits for t in select_tasks(pack, split,
                     [r['task_id'] for r in subsets[split]], None)]
        else:
            if args.split == 'both':
                raise ValueError('--split both requires --subsets')
            tasks = select_tasks(pack, args.split, args.tasks, args.limit)
    except (OSError, ValueError) as exc:
        parser.error('preflight: task loading/selection failed: '+str(exc))
    if not tasks:
        parser.error('no eligible tasks selected')
    settings = {'prompt_version': PROMPT_VERSION, 'prompt_sha256': hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
                'max_turns': args.max_turns, 'max_tokens': args.max_tokens,
                'wall_clock': args.wall_clock, 'temperature': 'provider-default',
                'evaluate_budget': 30, 'concurrency': used_concurrency,
                'codex_concurrency': args.codex_concurrency, 'anthropic_concurrency': args.anthropic_concurrency,
                'budget_usd': args.budget_usd, 'prior_episode_usd': args.prior_episode_usd,
                'order': args.order, 'effort': args.effort,
                'models': {m: model_settings(m, args.effort) for m in args.models},
                'subsets_sha256': hashlib.sha256(args.subsets.read_bytes()).hexdigest() if args.subsets else None}
    schedule = [j for phase in phases(tasks, args.models, args.repeats, args.order) for j in phase]
    if args.dry_run:
        projected = projection(schedule, [], args.prior_episode_usd)
        projected['note'] += ' '+PRIOR_ASSUMPTION
        print(json.dumps({'schedule': schedule, 'settings': settings,
                          'projection_prior': projected}, indent=2))
        return
    try:
        loaded_drivers = preflight.drivers(args.models)
        if args.subsets:
            from .warm_fields import check_warm
            check_warm(args.tasks_root, [t['task_id'] for t in tasks])
    except (preflight.PreflightError, ValueError) as exc:
        parser.error('preflight: '+str(exc))
    token = os.environ.get('PERMIT_EVAL_TOKEN') if args.server else secrets.token_urlsafe(32)
    if not token:
        parser.error('--server requires PERMIT_EVAL_TOKEN and eval.transport recorder routes')

    def run(url):
        try:
            preflight.server(url, token, tasks)
        except preflight.PreflightError as exc:
            parser.error('preflight: '+str(exc))
        def driver(model, task, checkpoint, capture, hard_budget):
            with Session(url, token, task['task_id'], timeout=args.wall_clock, checkpoint=checkpoint, capture=capture) as session:
                fn = loaded_drivers[model.partition(':')[0]]
                return fn(session, model, max_turns=args.max_turns,
                          max_tokens=args.max_tokens, wall_clock=args.wall_clock, effort=args.effort,
                          **({'hard_budget': hard_budget} if model.startswith('anthropic:') else {}))
        run_panel(args.output_root/component(args.run), tasks, args.models, driver,
                  concurrency=args.concurrency, budget_usd=args.budget_usd,
                  prior=args.prior_episode_usd, force=args.force, settings=settings,
                  repeats=args.repeats, import_runs=[args.output_root/component(r) for r in args.import_run],
                  order=args.order, codex_concurrency=args.codex_concurrency,
                  anthropic_concurrency=args.anthropic_concurrency)
    from .warm_fields import pin_warm
    with pin_warm(args.tasks_root, [t['task_id'] for t in tasks]) if args.subsets else nullcontext():
        if args.server:
            run(args.server)
        else:
            with serve(recording_app(token, pack=pack)) as url:
                run(url)


if __name__ == '__main__':
    main()
