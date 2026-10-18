"""Single cloud entrypoint: python -m train.cloud.run --plan --recipe smoke.

Execution is opt-in with --execute; --plan reads local identities and writes nothing.
"""
import argparse
import fcntl
import hashlib
import json
import math
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
    p.add_argument('--hourly-usd', type=float, default=5.00)
    p.add_argument('--recipe', choices=('smoke','grpo'), default='smoke')
    p.add_argument('--precision', choices=('bf16','nf4'), default='bf16')
    p.add_argument('--trackio-space', help='Explicit Attoloop/Space; omitted means no Trackio writes')
    p.add_argument('--trackio-bucket')
    p.add_argument('--qualification-run', type=Path, help='Previously passed smoke run (local or mounted read-only)')
    p.add_argument('--stop-after-step',type=int,help='Stop and save at this step, keeping --grpo-steps scheduler target')
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
    p.add_argument('--train-tasks', nargs='+', help='Optional train-only override; default frozen station-balanced thirty')
    p.add_argument('--sft-steps', type=int, default=100)
    p.add_argument('--grpo-steps', type=int, default=100)
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    config = CloudConfig(hourly_usd=args.hourly_usd, field_jobs=args.field_jobs,
                         sft_steps=args.sft_steps, grpo_steps=args.grpo_steps,recipe=args.recipe,
                         precision=args.precision,
                         trackio_space=args.trackio_space,trackio_bucket=args.trackio_bucket)
    if args.stop_after_step is not None and (config.recipe != 'grpo' or not 0 < args.stop_after_step <= config.grpo_steps):
        raise ValueError('stop-after-step must be within the fixed GRPO scheduler target')
    from permit_openenv.tasks import load_pack
    from .selection import selection_manifest
    selection = selection_manifest(load_pack(args.tasks_dir),args.cohort)
    plan = stage_plan(config, len(args.train_tasks) if args.train_tasks else 30,
                      parse_caps(args.wall_cap), parse_caps(args.usd_cap),stop_after_step=args.stop_after_step)
    if args.stage != 'all':
        plan = [s for s in plan if s['stage'] == args.stage]
        if not plan:
            raise ValueError('Stage is outside this recipe')
    if args.plan:
        print(json.dumps(dict(config=config.record(), stages=plan,
            selection=selection, stop_after_step=args.stop_after_step,
            provider_timeout_seconds=min(10800,math.floor(15*3600/config.hourly_usd)) if args.recipe == 'smoke' else
                math.ceil(sum(s['effective_seconds'] for s in plan)+1800),
            estimated_total_usd=sum(s['estimated_usd'] for s in plan),
            cap_total_usd=sum(s['effective_seconds']*config.hourly_usd/3600 for s in plan),
            setup_prior_seconds=900, setup_prior_usd=900*config.hourly_usd/3600,
            priors='Unmeasured: 300 CPU-s/task / workers; 180 s/eval episode; '
                   '180 s/GRPO step; smoke reload 180 s; kernel sanity 60 s; setup 900 s. No storage/tax/idle costs.',
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
        run(args, config, stage_plan(config,len(args.train_tasks) if args.train_tasks else 30,
            parse_caps(args.wall_cap),parse_caps(args.usd_cap)), sync)


def run(args, config: CloudConfig, plan: list[dict], sync) -> None:
    from permit_openenv.tasks import load_pack
    from .selection import selection_manifest, select_train
    pack = load_pack(args.tasks_dir)
    selection = selection_manifest(pack,args.cohort)
    train = select_train(pack, selection['smoke_train'] if config.recipe == 'smoke' else
                         args.train_tasks or selection['train'])
    cohort = selection['smoke_eval'] if config.recipe == 'smoke' else selection['eval']
    files = [args.cohort.resolve(), ROOT/'eval/PROTOCOL.md',
             ROOT/'envs/permit/openenv/permit_openenv/prompts.py',
             *ROOT.glob('train/cloud/*.py'), *ROOT.glob('train/*.py'),
             *ROOT.glob('envs/permit/core/permit_core/*.py'),
             *ROOT.glob('envs/permit/openenv/permit_openenv/*.py'),
             *args.tasks_dir.resolve().glob('permit-v2-*/tasks.jsonl'),
             ROOT/'train/cloud/uv.lock', ROOT/'train/cloud/pyproject.toml']
    input_manifest = ROOT/'train/cloud/input-manifest.json'
    if input_manifest.exists():
        files.append(input_manifest)
    frozen = dict(config=config.record(), plan=plan, hashes=fingerprint(files),
                  selection=selection,
                  train_tasks=[t.task_id for t in train], eval_tasks=cohort,
                  tasks_dir=str(args.tasks_dir.resolve()), sft_data=str(args.sft_data.resolve()),
                  fields_dir=str(args.fields_dir.resolve()), field_cache_gb=args.field_cache_gb,
                  artifact_uri=args.artifact_uri,
                  qualification_run=str(args.qualification_run.resolve()) if args.qualification_run else None)
    if config.recipe == 'grpo':
        if not args.qualification_run:
            raise ValueError('GRPO requires --qualification-run from a passed smoke')
        from .smoke import smoke_gate
        smoke_gate(args.qualification_run, frozen)
    manifest = args.output/'run.json'
    if manifest.exists() and json.loads(manifest.read_text()) != frozen:
        raise ValueError('Resume refuses changed config, cohort, data, source or protocol')
    write_json(manifest, frozen)
    if os.environ.get('PERMIT_BOOTSTRAP_REPORT'):
        write_json(args.output/'setup.json',json.loads(Path(os.environ['PERMIT_BOOTSTRAP_REPORT']).read_text()))
    state_path = args.output/'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    os.environ.update(PERMIT_FIELDS_CACHE_GB=str(args.field_cache_gb),
                      PERMIT_FIELDS_JOBS=str(config.field_jobs),
                      OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                      PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for spec in plan:
        stage = spec['stage']
        if args.stop_after_step and args.stop_after_step < config.grpo_steps and stage in ('eval','export'):
            continue
        if args.stage not in ('all', stage):
            continue
        entry = state.setdefault(stage, {'attempts':[]})
        if entry.get('status') == 'complete' and not (stage == 'grpo' and
                entry.get('optimizer_steps',0) < (args.stop_after_step or config.grpo_steps)):
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
        if stage == 'grpo' and args.stop_after_step:
            seconds = min(seconds,args.stop_after_step*360+1800)
        if os.environ.get('PERMIT_JOB_DEADLINE_UNIX'):
            seconds = min(seconds,float(os.environ['PERMIT_JOB_DEADLINE_UNIX'])-time.time())
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
            if stage == 'grpo' and args.stop_after_step:
                command += ['--stop-after-step',str(args.stop_after_step)]
            result = run_guarded(command, args.output/stage, seconds=seconds-(time.monotonic()-start)-120,
                                 sync=lambda timeout: sync(args.output, args.artifact_uri, timeout))
            entry.update(status=result['status'], exit_code=result['exit_code'])
            if stage == 'grpo' and (args.output/'grpo/metrics.json').exists():
                entry['optimizer_steps'] = json.loads((args.output/'grpo/metrics.json').read_text()).get('optimizer_steps',0)
        except BaseException:
            entry['status'] = 'interrupted'
            raise
        finally:
            # Reserve the final upload allowance before syncing state. A crash keeps the full reservation.
            attempt['charged_seconds'] = min(seconds, time.monotonic()-start+120)
            if stage == 'smoke':
                from .smoke import finalize_report
                finalize_report(args.output/stage,entry['status'],time.monotonic()-start,config.hourly_usd)
            write_json(state_path, state)
            sync(args.output, args.artifact_uri, max(.01, min(120, seconds-(time.monotonic()-start))))
            entry['upload_complete'] = True
            write_json(state_path,state)
        if entry['status'] != 'complete':
            raise RuntimeError(stage+': '+entry['status']+'; see console.log')


if __name__ == '__main__':
    main()
