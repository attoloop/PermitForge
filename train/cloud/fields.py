"""Bounded cloud CPU field cache: python -m train.cloud.fields --help."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading

from .run import write_json


def configure(root: Path) -> None:
    from permit_core import fields
    root.mkdir(parents=True, exist_ok=True)
    # Core fixes its cache relative to the v1 extraction. Bind only this cloud
    # process to the requested volume; no core or local-default edits.
    fields.cache_root = lambda: root


def estimated_bytes(tasks: list) -> int:
    return sum(sum(len(h) for h in t.scenario.height_menu.values()) *
        (43848 * len(t.scenario.receptors) * 8 + 20000) for t in tasks)


def check_capacity(sizes: list[int], *, cache_bytes: int, free_bytes: int) -> None:
    if not sizes or min(cache_bytes,free_bytes) < max(sizes):
        raise ValueError('Field volume/cache must hold at least the largest task working set')


def warm(tasks: list, root: Path, output: Path, jobs: int, *, resume: bool = True) -> None:
    from permit_core.fields import ensure_pack
    from permit_core.field_cache import cache_limit, evict_packs
    if not 1 <= jobs <= 4:
        raise ValueError('At most four native workers')
    sizes = [estimated_bytes([t]) for t in tasks]
    # Leave room for one incoming pack before native work; the remainder is LRU.
    evict_packs(max_bytes=max(0,cache_limit()-max(sizes)))
    check_capacity(sizes,cache_bytes=cache_limit(),free_bytes=shutil.disk_usage(root).free)
    prior = json.loads((output/'inventory.json').read_text()).get('tasks',[]) if resume and (output/'inventory.json').exists() else []
    previous = {r['task_id']:r for r in prior}
    inventory = []
    for task,size in zip(tasks,sizes):
        from permit_core.fields import pack_identity
        if task.task_id in previous and previous[task.task_id]['identity'] == pack_identity(task.scenario):
            inventory.append(previous[task.task_id])
            continue
        evict_packs(max_bytes=max(0,cache_limit()-size))
        with ensure_pack(task.scenario, jobs=jobs) as lease:
            manifest = json.loads(lease.path.read_text())
            inventory.append(dict(task_id=task.task_id, identity=manifest['identity'],
                bytes=sum(row[0] for row in manifest['files'].values()), computed=lease.computed))
            write_json(output/'inventory.json', dict(tasks=inventory,estimated_bytes=sum(sizes),
                note='All selected tasks warmed sequentially; LRU eviction may remove early tasks.'))
    write_json(output/'metrics.json', dict(status='passed',tasks=len(inventory),
        computed_bytes=sum(r['bytes'] for r in inventory),cache_cap_bytes=cache_limit()))


class CloudFields:
    """Borrowed immutable mmap packs; cache misses run in a small CPU process.

    GRPO uses one task per generation group. Concurrent calls are serialized at
    cache acquisition; leases protect active arrays against eviction.
    """
    def __init__(self, tasks: list, root: Path, *, tasks_dir: Path, output: Path, jobs: int):
        self.allowed = {t.task_id for t in tasks}
        self.root, self.tasks_dir, self.output, self.jobs = root,tasks_dir,output,jobs
        self.lock = threading.Lock()

    def __call__(self, task):
        from permit_core.field_cache import lock_file, ready, Pack
        from permit_core.fields import pack_identity
        if task.task_id not in self.allowed:
            raise ValueError('Field request outside stage task selection')
        with self.lock:
            root = self.root/task.task_id
            identity = pack_identity(task.scenario)
            if not ready(root,identity):
                subprocess.run([sys.executable,'-m','train.cloud.fields','--tasks-dir',str(self.tasks_dir),
                    '--task-id',task.task_id,'--root',str(self.root),'--output',str(self.output/task.task_id),
                    '--jobs',str(self.jobs)],check=True)
            handle = lock_file(task.task_id)
            try:
                fcntl.flock(handle,fcntl.LOCK_SH)
                if not ready(root,identity):
                    raise ValueError('Field cache changed before acquisition')
                return Pack(task.scenario,root,handle,False,task.scenario.height_menu)
            except BaseException:
                handle.close()
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks-dir',type=Path,required=True)
    parser.add_argument('--task-id',required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=4,choices=range(1,5))
    args = parser.parse_args()
    from permit_openenv.tasks import load_pack
    configure(args.root)
    warm([load_pack(args.tasks_dir).get(args.task_id)],args.root,args.output,args.jobs,resume=False)


if __name__ == '__main__':
    main()
