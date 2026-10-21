"""Sequential GPU memory trials: python -m train.memory_sweep --help.

Each isolated child runs one optimizer step over two train tasks. OOM and other
trial failures are recorded and do not abort the sweep. No fields are generated.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .runtime import MODEL_REVISIONS


def configurations(limits: dict[int, int]) -> list[dict]:
    """Largest group/batch/context first; order expresses preference, not measured VRAM."""
    return [dict(num_generations=generations, generation_batch_size=generations,
                 generation_microbatch_size=micro, accum=2*generations, batch_size=1,
                 max_turns=turns, max_completion_length=limits[turns], loss_chunk_size=64,
                 sdpa_backend='fused', empty_cache_after_generation=True)
            for generations, micro, turns in ((4, 4, 12), (4, 2, 12), (4, 1, 12),
                                               (4, 1, 10), (4, 1, 8), (2, 1, 8))]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run_sweep(*, models: list[str], configs: list[dict], common: list[str], output: Path,
              runs_dir: Path, runner=subprocess.run, timeout: float = 1800,
              prepare_only: bool = False, stop_after_success: bool = False) -> dict:
    result = dict(schema_version=1, status='planned' if prepare_only else 'running',
                  preference='group 4 before 2; generation microbatch 4/2/1; turns 12/10/8',
                  first_success={model: None for model in models}, results=[])
    for model in models:
        for number, config in enumerate(configs):
            config_id = f'{model.split("/")[-1]}-{number:02d}'
            directory = runs_dir/config_id
            directory.mkdir(parents=True, exist_ok=False)
            command = [sys.executable, '-m', 'train.memory_sweep', '--worker', '--model', model,
                       '--memory-probe', '--output', str(directory), *common]
            for name, value in config.items():
                flag = '--'+name.replace('_', '-')
                command.extend([flag] if value is True else [flag, str(value)])
            row = dict(model=model, config_id=config_id, config=config, command=command,
                       directory=str(directory), status='planned', oom=None,
                       peak_vram_allocated_bytes=None, peak_vram_reserved_bytes=None,
                       seconds=None, completion_lengths=[], groups=[], optimizer_steps=0,
                       tasks_seen=[])
            result['results'].append(row)
            write_json(output, result)
            if prepare_only:
                continue
            started = time.perf_counter()
            env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                   'OPENBLAS_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
                   'HF_HUB_DISABLE_TELEMETRY': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
            with (directory/'worker.log').open('w') as log:
                try:
                    process = runner(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                     timeout=timeout, check=False)
                    row['exit_code'] = process.returncode
                    path = directory/'probe.json'
                    if path.is_file():
                        row.update(json.loads(path.read_text()))
                    else:
                        row.update(status='error', error='Worker exited without probe.json', oom=None)
                    if row['status'] == 'passed' and (process.returncode != 0 or
                            row.get('optimizer_steps') != 1 or len(set(row.get('tasks_seen', []))) != 2):
                        row.update(status='error', error='Incomplete optimizer step or task coverage')
                except subprocess.TimeoutExpired:
                    row.update(status='timeout', error=f'Trial exceeded {timeout} seconds', oom=None)
                except OSError as exc:
                    row.update(status='error', error=str(exc), oom=None)
            row['seconds'] = time.perf_counter()-started
            if row['status'] == 'passed' and result['first_success'][model] is None:
                result['first_success'][model] = config_id
            write_json(output, result)
            print(f'{config_id}: {row["status"]}', flush=True)
            if row['status'] == 'passed' and stop_after_success:
                break
    result['status'] = 'planned' if prepare_only else 'complete'
    write_json(output, result)
    return result


def worker(argv: list[str]) -> int:
    """Catch even model-load OOM; process exit releases the CUDA context per trial."""
    import torch
    from . import grpo

    output = Path(argv[argv.index('--output')+1])
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(status='error', oom=False, peak_vram_allocated_bytes=None,
                   peak_vram_reserved_bytes=None, completion_lengths=[], groups=[], tasks_seen=[],
                   optimizer_steps=0)
    available = torch.cuda.is_available()
    summary['cuda_available'] = available
    summary['torch_version'] = torch.__version__
    summary['compiled_arch_flags'] = torch._C._cuda_getArchFlags()
    if available:
        summary['device'] = torch.cuda.get_device_name()
        summary['capability'] = list(torch.cuda.get_device_capability())
        free, total = torch.cuda.mem_get_info()
        summary.update(initial_free_bytes=free, total_bytes=total)
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    code = 1
    try:
        grpo.main(argv)
        summary['status'] = 'passed'
        code = 0
    except Exception as exc:
        summary['oom'] = isinstance(exc, torch.OutOfMemoryError)
        summary['status'] = 'oom' if summary['oom'] else 'error'
        summary['error'] = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    finally:
        path = output/'metrics.json'
        if path.is_file():
            metrics = json.loads(path.read_text())
            summary.update({k: v for k, v in metrics.items() if k not in ('status', 'error')})
        if available:
            summary.update(peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_vram_reserved_bytes=torch.cuda.max_memory_reserved())
        summary['seconds'] = time.perf_counter()-started
        write_json(output/'probe.json', summary)
    return code


def main() -> int:
    if '--worker' in sys.argv[1:]:
        return worker([arg for arg in sys.argv[1:] if arg != '--worker'])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(MODEL_REVISIONS), default=list(MODEL_REVISIONS))
    parser.add_argument('--tasks', nargs=2, default=['074897d0f159-s47', '01addc2a86f1-s47'])
    parser.add_argument('--fields-dir', type=Path, required=True)
    parser.add_argument('--tasks-dir', type=Path, default=Path('envs/permit/tasks'))
    parser.add_argument('--diagnostics-dir', type=Path, default=Path('convert/cache/local-batch'))
    parser.add_argument('--token-budget', type=Path, default=Path('train/evidence/token-budget.json'))
    parser.add_argument('--output', type=Path, default=Path('train/evidence/memory-sweep.json'))
    parser.add_argument('--runs-dir', type=Path, default=Path('train/runs/memory-sweep') /
                        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    parser.add_argument('--prepare-only', action='store_true', help='Write commands only; no models or GPU work')
    parser.add_argument('--stop-after-success', action='store_true', help='Stop at first fitting config per model')
    parser.add_argument('--timeout', type=float, default=1800, help='Per-child wall clock seconds')
    args = parser.parse_args()
    if args.timeout <= 0 or len(set(args.tasks)) != 2:
        parser.error('Timeout must be positive and tasks distinct')
    from .grpo import select_tasks
    from permit_openenv.tasks import load_pack
    select_tasks(load_pack(args.tasks_dir), args.tasks, None)
    budgets = json.loads(args.token_budget.read_text())['models']
    limits = {n: max(budgets[m]['recommended_completion_limits'][str(n)] for m in args.models)
              for n in (8, 10, 12)}
    prompt_limit = max(budgets[m]['recommended_prompt_limit'] for m in args.models)
    common = ['--tasks', *args.tasks, '--fields-dir', str(args.fields_dir),
              '--tasks-dir', str(args.tasks_dir), '--diagnostics-dir', str(args.diagnostics_dir),
              '--max-prompt-length', str(prompt_limit)]
    run_sweep(models=args.models, configs=configurations(limits), common=common, output=args.output,
              runs_dir=args.runs_dir, timeout=args.timeout, prepare_only=args.prepare_only,
              stop_after_success=args.stop_after_success)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
