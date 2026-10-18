"""One bounded stage; invoked by train.cloud.run, never as a local training default."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import threading

from .config import CloudConfig
from .run import write_json


@contextmanager
def gpu_guard(config: CloudConfig, output: Path):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        import ctypes.util
        import subprocess
        try:  # the job image has no nvidia-smi unless the host injects the driver
            smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                                 capture_output=True, text=True).stdout.strip()
        except FileNotFoundError:
            smi = ''
        raise RuntimeError(f'Cloud training requires exactly one visible CUDA GPU (torch CUDA {torch.version.cuda}, '
                           f'devices {torch.cuda.device_count()}, libcuda {ctypes.util.find_library("cuda") or "not found"}, '
                           f'nvidia-smi: {smi or "unavailable"})')
    properties = torch.cuda.get_device_properties(0)
    if (torch.cuda.get_device_capability() != (9,0) or 'sm_90' not in torch.cuda.get_arch_list()
            or properties.total_memory < config.max_vram_bytes or 'H200' not in properties.name):
        raise RuntimeError('This recipe requires one H200, Hopper sm_90')
    # Reserve nonallocator CUDA overhead below the device-wide cap.
    torch.cuda.set_per_process_memory_fraction((config.max_vram_bytes-2_000_000_000)/properties.total_memory)
    torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(4)
    stop = threading.Event()
    evidence = dict(device=properties.name,total_bytes=properties.total_memory,
                    capability=list(torch.cuda.get_device_capability()),peak_device_bytes=0,
                    abort_bytes=config.max_vram_bytes,status='running')
    def watch():
        while not stop.is_set():
            used = torch.cuda.device_memory_used(0)
            evidence['peak_device_bytes'] = max(evidence['peak_device_bytes'],used)
            if used > config.max_vram_bytes:
                evidence['status'] = 'vram_cap_exceeded'
                write_json(output/'gpu.json', evidence)
                os.kill(os.getpid(),signal.SIGTERM)
                return
            stop.wait(.25)
    thread = threading.Thread(target=watch,daemon=True)
    thread.start()
    try:
        yield
        evidence['status'] = 'passed'
    except BaseException:
        if evidence['status'] == 'running':
            evidence['status'] = 'failed'
        raise
    finally:
        stop.set()
        thread.join(timeout=1)
        evidence.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(output/'gpu.json',evidence)


def export(run: Path, output: Path) -> None:
    adapter = run/'grpo/adapter'
    if not (adapter/'adapter_model.safetensors').exists():
        raise ValueError('GRPO adapter missing; export cannot substitute an earlier checkpoint')
    shutil.copytree(adapter,output/'adapter',dirs_exist_ok=True)
    summaries = {str(p.relative_to(run)):json.loads(p.read_text()) for p in run.rglob('metrics.json')
                 if output not in p.parents}
    write_json(output/'metrics.json',dict(status='passed',stages=summaries,merged=False,
        merge_note='Adapter-only export; precision and kernel provenance are in run.json.'))
    shutil.copyfile(run/'run.json',output/'run.json')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--stage',required=True)
    parser.add_argument('--stop-after-step',type=int)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    record = json.loads((args.run/'run.json').read_text())
    config = CloudConfig(**record['config'])
    output = args.run/args.stage
    output.mkdir(parents=True,exist_ok=True)
    from permit_openenv.tasks import load_pack
    from .fields import configure, warm, CloudFields
    from .training import train, load_model
    pack = load_pack(Path(record['tasks_dir']))
    train_tasks = [pack.get(tid) for tid in record['train_tasks']]
    eval_tasks = [pack.get(tid) for tid in record['eval_tasks']]
    root = Path(record['fields_dir'])
    configure(root)
    if args.stage == 'fields':
        warm(train_tasks+eval_tasks,root,output,config.field_jobs)
        return
    if args.stage == 'export':
        if json.loads((args.run/'grpo/metrics.json').read_text()).get('optimizer_steps',0) < config.grpo_steps:
            raise ValueError('Finish the fixed GRPO step target before export')
        export(args.run,output)
        return
    if args.stage in ('grpo','eval'):
        from .smoke import smoke_gate
        smoke_gate(Path(record['qualification_run']), record)
    if args.stage == 'eval' and json.loads((args.run/'grpo/metrics.json').read_text()).get('optimizer_steps',0) < config.grpo_steps:
        raise ValueError('Finish the fixed GRPO step target before final evaluation')
    fields = CloudFields(eval_tasks if args.stage in ('base-eval','eval') else train_tasks, root,
        tasks_dir=Path(record['tasks_dir']),output=output/'field-misses',jobs=config.field_jobs)
    try:
        with gpu_guard(config,output):
            from .kernels import probe
            kernels = probe(output=output/'kernels.json')
            if kernels['status'] != 'passed':
                raise RuntimeError('Linear-attention sanity check failed; see kernels.json')
            if args.stage in ('grpo','smoke'):
                data = Path(record['sft_data'])
                if args.stage == 'smoke':
                    from .smoke import run_smoke
                    run_smoke(config,record,pack,output)
                else:
                    train(config,args.stage,output,data=data,pack=pack,tasks=train_tasks,
                          fields_loader=fields,stop_after_step=args.stop_after_step)
            else:
                from .evaluation import evaluate
                adapter = args.run/'grpo/adapter' if args.stage == 'eval' else None
                if adapter and not adapter.exists():
                    raise ValueError('GRPO adapter missing')
                model,tokenizer = load_model(config,adapter)
                evaluate(config,model,tokenizer,pack,record['eval_tasks'],output,
                         fields_loader=fields,tasks_dir=Path(record['tasks_dir']),adapter=adapter)
    finally:
        if args.stage == 'smoke':
            from .smoke import write_report
            write_report(output)


if __name__ == '__main__':
    main()
