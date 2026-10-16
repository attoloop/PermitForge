"""Process-safe leases for the on-demand float64, memory-mapped field packs."""

from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import BinaryIO

import numpy as np

from . import fields
from .model import Scenario
from .native_pool import native_slot


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def pack_identity(task: Scenario) -> str:
    inputs = [task.task_root/'task.json', task.task_root/'buildings.json', task.task_root/'receptors.inp',
              *(task.met_path/f'{task.station}.{ext}' for ext in ('sfc', 'pfl')),
              *(fields.models_dir()/name for name in ('aermod', 'bpipprm'))]
    payload = {'files': [digest(p) for p in inputs],
               'menus': task.height_menu, 'code': digest(Path(fields.__file__))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def lock_file(task_id: str) -> BinaryIO:
    root = fields.cache_root()/'.locks'
    root.mkdir(parents=True, exist_ok=True)
    return (root/f'{task_id}.lock').open('a+b')


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, sort_keys=True)+'\n')
    temporary.replace(path)


def file_stat(path: Path) -> list[int]:
    info = path.stat()
    return [info.st_size, info.st_mtime_ns]


def ready(root: Path, identity: str) -> bool:
    try:
        manifest = json.loads((root/'pack.json').read_text())
        return (manifest['identity'] == identity and 'sha256' in manifest and
                all(file_stat(root/p) == info for p, info in manifest['files'].items()))
    except (OSError, ValueError, KeyError):
        return False


def ensure_field(task: Scenario, sid: str, height: str, identity: str) -> None:
    path = fields.cache_root()/task.id/sid/f'{height}.npy'
    meta_path = path.with_suffix('.json')
    try:
        meta = json.loads(meta_path.read_text())
        # Adopt the immutable fields after checking their content hashes.
        if (meta.get('identity', identity) == identity and meta['task_id'] == task.id and
                digest(path) == meta['sha256']):
            values = np.load(path, mmap_mode='r', allow_pickle=False)
            if values.dtype == np.float64 and values.shape[1] == len(task.receptors):
                if 'identity' not in meta:
                    write_json(meta_path, {**meta, 'identity': identity})
                return
    except (OSError, ValueError, KeyError, IndexError):
        pass
    with native_slot():
        result = fields.compute_field(task, sid, height)
    # RUSAGE_CHILDREN is process-wide and overlaps when native runs use threads.
    # Keep the accurate per-run wall time; do not attribute another thread's CPU.
    result.metadata.pop('native_cpu_seconds', None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as handle:
        np.save(handle, np.asarray(result.values, dtype=np.float64), allow_pickle=False)
    path.with_suffix('.tmp').replace(path)
    write_json(meta_path, {**result.metadata, 'identity': identity, 'sha256': digest(path)})


class Pack(fields.FieldPack):
    """Use as a context manager, or call release(); the shared flock pins disk files."""

    def __init__(self, task: Scenario, root: Path, lease, computed: bool, menus: dict):
        arrays = {(sid, fields.height_key(h)): np.load(root/sid/f'{fields.height_key(h)}.npy',
                                                     mmap_mode='r', allow_pickle=False)
                  for sid, heights in menus.items() for h in heights}
        dates, valid = fields.met_calendar(task)
        if any(a.shape != (len(valid), len(task.receptors)) for a in arrays.values()):
            raise ValueError('cached field/calendar shape mismatch')
        offsets = (0, *(np.flatnonzero(np.diff(dates//1000000))+1).tolist(), len(dates))
        super().__init__(task.id, arrays, valid, offsets, np.arange(len(task.receptors)))
        self.path, self.computed, self._lease = root/'pack.json', computed, lease

    def __enter__(self) -> 'Pack':
        return self

    def __exit__(self, *args) -> None:
        self.release()

    def release(self, *, evict: bool = False) -> None:
        lease = getattr(self, '_lease', None)
        if lease is None:
            return
        self.fields.clear()
        self._lease = None
        lease.close()
        evict_packs(task_id=self.task_id if evict else None)

    def __del__(self) -> None:
        lease = getattr(self, '_lease', None)
        if lease is not None:
            lease.close()


def ensure_pack(task: Scenario, *, jobs: int) -> Pack:
    if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= 12:
        raise ValueError('jobs must be between 1 and 12')
    if Path(task.id).name != task.id or task.id.startswith('.'):
        raise ValueError('invalid cache task id')
    cache_limit()  # Reject invalid configuration before starting native work.
    root = fields.cache_root()/task.id
    identity = fields.pack_identity(task)
    menus = task.height_menu or {s.id: fields.height_menu(s.height_min_m, s.gep_height_m) for s in task.sources}
    lease = lock_file(task.id)
    computed = False
    try:
        fcntl.flock(lease, fcntl.LOCK_SH)
        if not ready(root, identity):
            fcntl.flock(lease, fcntl.LOCK_UN)
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not ready(root, identity):
                root.mkdir(parents=True, exist_ok=True)
                (root/'access').touch()
                (root/'pack.json').unlink(missing_ok=True)
                work = [(sid, fields.height_key(h)) for sid, heights in menus.items() for h in heights]
                with ThreadPoolExecutor(max_workers=jobs) as pool:
                    list(pool.map(lambda pair: ensure_field(task, *pair, identity), work))
                paths = [root/sid/f'{h}.{ext}' for sid, h in work for ext in ('npy', 'json')]
                write_json(root/'pack.json', {'identity': identity, 'dtype': 'float64',
                                             'files': {str(p.relative_to(root)): file_stat(p) for p in paths},
                                             'sha256': {str(p.relative_to(root)): json.loads(p.read_text())['sha256']
                                                        for p in paths if p.suffix == '.json'}})
                computed = True
            fcntl.flock(lease, fcntl.LOCK_SH)
        os.utime(root/'pack.json', None)
        (root/'access').touch()
        pack = Pack(task, root, lease, computed, menus)
        evict_packs()
        return pack
    except BaseException:
        lease.close()
        evict_packs()
        raise


def cache_limit() -> int:
    gb = float(os.environ.get('PERMIT_FIELDS_CACHE_GB', '40'))
    if not math.isfinite(gb) or gb < 0:
        raise ValueError('PERMIT_FIELDS_CACHE_GB must be finite and nonnegative')
    return int(gb*1024**3)


def evict_packs(*, max_bytes: int | None = None, task_id: str | None = None) -> None:
    """Evict least recently acquired idle packs; active leases may temporarily exceed the cap.

    Legacy gate directories count against the cap but remain pinned until acquired.
    Their independent precompute/fidelity jobs do not participate in this locking protocol.
    """
    limit = cache_limit() if max_bytes is None else max_bytes
    root = fields.cache_root()
    lease = lock_file('_eviction')
    with lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        candidates = []
        for directory in {p.parent for pattern in ('*/pack.json', '*/access') for p in root.glob(pattern)}:
            try:
                marker = directory/'access' if (directory/'access').exists() else directory/'pack.json'
                size = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
                candidates.append((marker.stat().st_mtime_ns, directory, size))
            except FileNotFoundError:
                continue
        managed = {directory for _, directory, _ in candidates}
        legacy = sum(p.stat().st_size for directory in root.iterdir()
                     if directory.is_dir() and directory not in managed and
                     not directory.name.startswith(('.', 'native-'))
                     for p in directory.rglob('*') if p.is_file())
        total = legacy + sum(size for _, _, size in candidates)
        for _, directory, size in sorted(candidates):
            if total <= limit and directory.name != task_id:
                continue
            with lock_file(directory.name) as guard:
                try:
                    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if directory.exists():
                    shutil.rmtree(directory)
                    total -= size
