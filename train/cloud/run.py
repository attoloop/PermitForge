"""Single cloud entrypoint: python -m train.cloud.run --plan --hourly-usd 2.75.

Execution is opt-in with --execute; --plan imports no GPU libraries and does no I/O.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .config import CloudConfig, STAGES, stage_plan, remaining_seconds

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def terminate(process: subprocess.Popen, descendants: dict | None = None) -> None:
    import psutil
    for child in (descendants or {}).values():
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=.25)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_guarded(command: list[str], output: Path, *, seconds: float, sync=None) -> dict:
    """Bound the entire process group, including native children and hung GPU calls."""
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process = None
    descendants = {}
    status = 'failed'
    try:
        with (output/'console.log').open('a') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            next_sync = started + 60
            while process.poll() is None:
                import psutil
                try:
                    for child in psutil.Process(process.pid).children(recursive=True):
                        descendants[(child.pid, child.create_time())] = child
                except psutil.NoSuchProcess:
                    pass
                now = time.monotonic()
                if now-started >= seconds:
                    status = 'cap_exceeded'
                    break
                if sync and now >= next_sync:
                    # A sync failure stops training, so checkpoints never silently stay ephemeral.
                    sync(min(120, max(.01, seconds-(now-started))))
                    next_sync = time.monotonic()+60
                time.sleep(min(.1, max(.001, seconds-(time.monotonic()-started))))
            else:
                status = 'complete' if process.returncode == 0 else 'failed'
    finally:
        if process is not None:
            terminate(process, descendants)
    return dict(status=status, exit_code=process.returncode, elapsed_seconds=time.monotonic()-started)


def parse_caps(values: list[str]) -> dict:
    result = {}
    for value in values:
        stage, number = value.split('=', 1)
        if stage not in STAGES or stage in result:
            raise ValueError('Caps require unique stage=value entries')
        result[stage] = float(number)
    return result


def fingerprint(paths: list[Path]) -> dict:
    return {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--stage', choices=(*STAGES, 'all'), default='all')
    p.add_argument('--hourly-usd', type=float, required=True)
    p.add_argument('--wall-cap', action='append', default=[], help='stage=seconds, cumulative across resumes')
    p.add_argument('--usd-cap', action='append', default=[], help='stage=USD of compute at --hourly-usd')
    p.add_argument('--output', type=Path, default=Path('train/cloud/runs/qwen38'))
    p.add_argument('--tasks-dir', type=Path, default=Path('envs/permit/tasks'))
    p.add_argument('--cohort', type=Path, default=Path('results/subsets.json'))
    p.add_argument('--sft-data', type=Path, default=Path('train/cloud/data/sft.jsonl'))
    p.add_argument('--artifact-uri', help='Existing HF bucket/repository run prefix')
    p.add_argument('--restore', action='store_true')
    p.add_argument('--field-jobs', type=int, default=4)
    p.add_argument('--fields-dir', type=Path, default=Path('train/cloud/cache/fields'))
    p.add_argument('--field-cache-gb', type=float, default=128)
    p.add_argument('--train-tasks', nargs='+', help='Optional explicit train-only subset; default all eligible train tasks')
    p.add_argument('--sft-steps', type=int, default=100)
    p.add_argument('--grpo-steps', type=int, default=100)
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    config = CloudConfig(hourly_usd=args.hourly_usd, field_jobs=args.field_jobs,
                         sft_steps=args.sft_steps, grpo_steps=args.grpo_steps)
    plan = stage_plan(config, len(args.train_tasks) if args.train_tasks else 230,
                      parse_caps(args.wall_cap), parse_caps(args.usd_cap))
    if args.plan:
        print(json.dumps(dict(config=config.record(), stages=plan,
            estimated_total_usd=sum(s['estimated_usd'] for s in plan),
            cap_total_usd=sum(s['effective_seconds']*config.hourly_usd/3600 for s in plan),
            priors='Unmeasured: 300 CPU-s/task / workers; 180 s/eval episode; 20 s/SFT step; '
                   '180 s/GRPO step; smoke 1800 s; export 900 s. Includes no startup/storage/tax/idle costs.',
            eval_difference='WP2.9 requires three repeats; final panel uses four. Thinking disabled for this model.'), indent=2))
        return
    if not args.execute or not args.artifact_uri:
        raise ValueError('Execution requires --execute and an existing --artifact-uri')
    if int(os.environ.get('WORLD_SIZE','1')) != 1:
        raise ValueError('One process/GPU only')
    if not 0 < args.field_cache_gb < float('inf'):
        raise ValueError('field-cache-gb must be positive and finite')
    from .storage import restore, sync, sync_command
    sync_command(args.output, args.artifact_uri)  # validate before compute
    args.output = args.output.resolve()
    if args.restore:
        restore(args.output, args.artifact_uri, 120)
    args.output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, config, plan, sync)


def run(args, config: CloudConfig, plan: list[dict], sync) -> None:
    from permit_openenv.tasks import load_pack
    from .selection import select_cohort, select_train
    pack = load_pack(args.tasks_dir)
    train = select_train(pack, args.train_tasks)
    cohort = select_cohort(pack, args.cohort)
    files = [args.cohort.resolve(), args.sft_data.resolve(), ROOT/'eval/PROTOCOL.md',
             ROOT/'envs/permit/openenv/permit_openenv/prompts.py',
             *ROOT.glob('train/cloud/*.py'), *ROOT.glob('train/*.py'),
             *ROOT.glob('envs/permit/core/permit_core/*.py'),
             *ROOT.glob('envs/permit/openenv/permit_openenv/*.py'),
             *args.tasks_dir.resolve().glob('permit-v2-*/tasks.jsonl'),
             ROOT/'train/cloud/uv.lock']
    frozen = dict(config=config.record(), plan=plan, hashes=fingerprint(files),
                  train_tasks=[t.task_id for t in train], eval_tasks=cohort,
                  tasks_dir=str(args.tasks_dir.resolve()), sft_data=str(args.sft_data.resolve()),
                  fields_dir=str(args.fields_dir.resolve()), field_cache_gb=args.field_cache_gb,
                  artifact_uri=args.artifact_uri)
    manifest = args.output/'run.json'
    if manifest.exists() and json.loads(manifest.read_text()) != frozen:
        raise ValueError('Resume refuses changed config, cohort, data, source or protocol')
    write_json(manifest, frozen)
    state_path = args.output/'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    os.environ.update(PERMIT_FIELDS_CACHE_GB=str(args.field_cache_gb),
                      PERMIT_FIELDS_JOBS=str(config.field_jobs),
                      OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                      PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for spec in plan:
        stage = spec['stage']
        if args.stage not in ('all', stage):
            continue
        entry = state.setdefault(stage, {'attempts':[]})
        if entry.get('status') == 'complete':
            if entry.get('upload_complete'):
                continue
            allowance = remaining_seconds(spec['wall_seconds'], spec['usd_cap'], config.hourly_usd, entry['attempts'])
            if allowance <= 0:
                raise RuntimeError(stage+': cap exhausted before artifact upload')
            start_sync = time.monotonic()
            sync(args.output,args.artifact_uri,min(120,allowance))
            entry['attempts'].append({'charged_seconds':time.monotonic()-start_sync})
            entry['upload_complete'] = True
            write_json(state_path,state)
            continue
        seconds = remaining_seconds(spec['wall_seconds'], spec['usd_cap'], config.hourly_usd, entry['attempts'])
        if seconds <= 150:
            raise RuntimeError(stage+': cumulative cap exhausted')
        attempt = {'reserved_seconds':seconds, 'started_unix':time.time()}
        entry['attempts'].append(attempt)
        entry['status'] = 'running'
        write_json(state_path, state)
        start = time.monotonic()
        try:
            sync(args.output, args.artifact_uri, min(30, seconds/4))
            command = [sys.executable, '-m', 'train.cloud.worker', '--run', str(args.output), '--stage', stage]
            result = run_guarded(command, args.output/stage, seconds=seconds-(time.monotonic()-start)-120,
                                 sync=lambda timeout: sync(args.output, args.artifact_uri, timeout))
            entry.update(status=result['status'], exit_code=result['exit_code'])
        except BaseException:
            entry['status'] = 'interrupted'
            raise
        finally:
            # Reserve the final upload allowance before syncing state. A crash keeps the full reservation.
            attempt['charged_seconds'] = min(seconds, time.monotonic()-start+120)
            write_json(state_path, state)
            sync(args.output, args.artifact_uri, max(.01, min(120, seconds-(time.monotonic()-start))))
            entry['upload_complete'] = True
            write_json(state_path,state)
        if entry['status'] != 'complete':
            raise RuntimeError(stage+': '+entry['status']+'; see console.log')


if __name__ == '__main__':
    main()
