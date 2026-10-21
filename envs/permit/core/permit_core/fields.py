"""Native unit fields and resumable packs.

uv run python -m permit_core.fields --tasks TASK_ID ... --jobs 4
Cache data stays under convert/cache/fields; the verifier binary is never patched.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from decimal import Decimal as D
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import resource
import tempfile
import time
from typing import TYPE_CHECKING

import numpy as np

from .menu import height_menu
from .model import Scenario, extraction_root, load_v1_instance
from .native import run_native
from .render import render
from .verify import models_dir

if TYPE_CHECKING:
    from .field_cache import Pack


def height_key(height: object) -> str:
    return format(D(str(height)).normalize(), 'f')


def cache_root() -> Path:
    try:
        return extraction_root().parent / 'cache/fields'
    except FileNotFoundError:
        for parent in Path(__file__).resolve().parents:
            if (parent / 'convert').is_dir():
                return parent / 'convert/cache/fields'
        raise


def pack_identity(task: Scenario) -> str:
    from .field_cache import pack_identity as identity
    return identity(task)


def ensure_pack(task: Scenario, *, jobs: int) -> 'Pack':
    """Acquire an on-demand float64 pack; release it with a context manager."""
    from .field_cache import ensure_pack as ensure
    return ensure(task, jobs=jobs)


def evict_packs(*, max_bytes: int | None = None, task_id: str | None = None) -> None:
    from .field_cache import evict_packs as evict
    evict(max_bytes=max_bytes, task_id=task_id)


@dataclass
class Field:
    values: np.ndarray
    valid_hours: np.ndarray
    dates: np.ndarray
    native_seconds: float
    metadata: dict


@dataclass
class FieldPack:
    task_id: str
    fields: dict[tuple[str, str], np.ndarray]
    valid_hours: np.ndarray
    year_offsets: tuple[int, ...]
    receptor_indices: np.ndarray
    binding_certified: bool = False


def read_postfile(path: Path, receptors: int) -> tuple[np.ndarray, np.ndarray]:
    # calc2.f:2952–2954; modules.f:1455. gfortran sequential records, real*8 AVEVAL.
    dtype = np.dtype([('start', '<i4'), ('date', '<i4'), ('period', '<i4'),
                      ('group', 'S8'), ('values', '<f8', (receptors,)), ('end', '<i4')])
    if path.stat().st_size % dtype.itemsize:
        raise ValueError('truncated POSTFILE record')
    data = np.fromfile(path, dtype=dtype)
    size = 16 + 8*receptors
    if not len(data) or not (np.all(data['start'] == size) and np.all(data['end'] == size)
                            and np.all(data['period'] == 1) and np.all(data['group'] == b'ALL     ')):
        raise ValueError('unexpected POSTFILE precision, group, or record markers')
    values = data['values'].copy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('invalid native concentrations')
    return data['date'].copy(), values


def met_calendar(task: Scenario) -> tuple[np.ndarray, np.ndarray]:
    # metext.f:1499–1516,1959–1966,2006–2037 (CHKCLM and CHKMSG).
    dates, valid = [], []
    for line in (task.met_path / f'{task.station}.sfc').read_text().splitlines()[1:]:
        row = line.split()
        year, month, day, _, hour = map(int, row[:5])
        _, ustar, wstar, _, ziconv, zimech, obulen, _, _, _, wind, direction, _, temp = map(float, row[5:19])
        missing = (wind >= 90 or wind < 0 or direction > 900 or direction <= -9 or
                   temp > 900 or temp <= 0 or obulen < -99990 or
                   (obulen < 0 and (ziconv > 90000 or ziconv < 0)) or
                   zimech > 90000 or zimech < 0 or ustar < 0 or ustar >= 9 or
                   (wstar < 0 and -99990 < obulen < 0))
        dates.append((year % 100)*1000000 + month*10000 + day*100 + hour)
        valid.append(wind != 0 and not missing)
    return np.asarray(dates, dtype=np.int32), np.asarray(valid, dtype=bool)


def validity_from_messages(path: Path, dates: np.ndarray) -> np.ndarray:
    """Native I440/I460 messages expose the same calm/missing flags as counters."""
    invalid = set()
    for line in path.read_text().splitlines():
        if re.search(r'\bI(?:440|460)\b', line):
            matches = re.findall(r'\b\d{10}\b', line)
            if not matches:
                raise ValueError('invalid native validity message')
            invalid.add(int(matches[-1]) % 100000000)
    if not invalid.issubset(set(map(int, dates))):
        raise ValueError('validity dates outside POSTFILE')
    return np.array([int(date) not in invalid for date in dates])


def field_input(task: Scenario, source_id: str, height: D, directory: Path) -> str:
    ledger = {s.id: {'control': 'none', 'stack_height_m': s.gep_height_m} for s in task.sources}
    ledger[source_id]['stack_height_m'] = height
    render(task.task_root, ledger, directory / 'rendered')
    text = (directory / 'rendered/aermod.inp').read_text().split('** PERMIT_CASE PM25\n')[1].split('** PERMIT_CASE ')[0]
    output = []
    for line in text.splitlines():
        row = line.split()
        if not row:
            continue
        if row[0] == 'OU':
            continue
        if row[1] == 'MODELOPT':
            row = ['CO', 'MODELOPT', 'DFAULT', 'CONC']
        elif row[1] == 'AVERTIME':
            row = ['CO', 'AVERTIME', '1']
        elif row[1] == 'POLLUTID':
            row[2] = 'OTHER'
        elif row[1] == 'SRCPARAM':
            row[3] = '1.0' if row[2] == source_id else '0.0'
        elif row[1] == 'ERRORFIL':
            continue
        if row == ['CO', 'FINISHED']:
            output.append('CO ERRORFIL messages.out')
        output.append(f'{row[0]} {row[1]:<8} ' + ' '.join(row[2:]))
    output += ['OU STARTING', 'OU POSTFILE 1 ALL UNFORM hourly.bin', 'OU FINISHED']
    return ('\n'.join(output)+'\n').replace('/task/met/', 'met/').replace('/task/', str(task.task_root)+'/')


def compute_field(task: Scenario, source_id: str, height: object) -> Field:
    """One unit-rate AERMOD run; BPIP always sees the full stack set."""
    source = next(s for s in task.sources if s.id == source_id)
    height = D(str(height))
    if height not in height_menu(source.height_min_m, source.gep_height_m):
        raise ValueError('off-menu field height')
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='native-', dir=root) as tmp:
        work = Path(tmp)
        text = field_input(task, source_id, height, work)
        (work / 'aermod.inp').write_text(text)
        (work / 'met').symlink_to(task.met_path, target_is_directory=True)
        start = time.perf_counter()
        cpu_start = resource.getrusage(resource.RUSAGE_CHILDREN)
        run_native(models_dir() / 'aermod', work, task.task['guards']['native_seconds'],
                   {'resources': {'native_launches': 0}})
        seconds = time.perf_counter()-start
        cpu_end = resource.getrusage(resource.RUSAGE_CHILDREN)
        output = (work / 'aermod.out').read_text()
        if 'AERMOD Finishes Successfully' not in output:
            raise ValueError('AERMOD field failed: '+output[-1500:])
        dates, values = read_postfile(work / 'hourly.bin', len(task.receptors))
        expected_dates, expected_valid = met_calendar(task)
        if not np.array_equal(dates, expected_dates):
            raise ValueError('POSTFILE dates differ from meteorology')
        valid = validity_from_messages(work / 'messages.out', dates)
        if not np.array_equal(valid, expected_valid):
            raise ValueError('native calm/missing flags differ from met-driven flags')
        if np.any(values[~valid] != 0):
            raise ValueError('nonzero concentration in invalid hour')
        metadata = {'task_id': task.id, 'source_id': source_id, 'height': height_key(height),
                    'input_sha256': hashlib.sha256(text.encode()).hexdigest(),
                    'binary_sha256': hashlib.sha256((models_dir()/'aermod').read_bytes()).hexdigest(),
                    'native_seconds': seconds, 'shape': list(values.shape),
                    'native_cpu_seconds': cpu_end.ru_utime + cpu_end.ru_stime - cpu_start.ru_utime - cpu_start.ru_stime,
                    'valid_hours': int(valid.sum()), 'input': text}
        return Field(values, valid, dates, seconds, metadata)


def _cached_field(args: tuple[str, str, str]) -> dict:
    task_id, source_id, height = args
    task = load_v1_instance(task_id)
    path = cache_root() / task_id / source_id / f'{height}.npy'
    metadata = path.with_suffix('.json')
    if metadata.exists() and path.exists():
        result = json.loads(metadata.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != result['sha256']:
            raise ValueError(f'corrupt cached field: {path}')
        return result
    result = compute_field(task, source_id, height)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.save(handle, result.values, allow_pickle=False)
    temporary.replace(path)
    result.metadata['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata.write_text(json.dumps(result.metadata, indent=2)+'\n')
    return result.metadata


def precompute(task_ids: list[str], jobs: int = 4) -> list[dict]:
    if not 1 <= jobs <= 8:
        raise ValueError('jobs must be between 1 and 8')
    tasks = [load_v1_instance(task_id) for task_id in task_ids]
    work = [(t.id, s.id, height_key(h)) for t in tasks for s in t.sources
            for h in height_menu(s.height_min_m, s.gep_height_m)]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_cached_field, work))


def cached_pack(task: Scenario) -> FieldPack:
    fields = {(s.id, height_key(h)): np.load(cache_root()/task.id/s.id/f'{height_key(h)}.npy', mmap_mode='r')
              for s in task.sources for h in height_menu(s.height_min_m, s.gep_height_m)}
    dates, valid = met_calendar(task)
    offsets = tuple([0, *(np.flatnonzero(np.diff(dates//1000000)) + 1).tolist(), len(dates)])
    return FieldPack(task.id, fields, valid, offsets, np.arange(len(task.receptors)))


def prune_candidates(task: Scenario, pack: FieldPack) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    maxima = {s.id: np.maximum.reduce([v.max(axis=0) for (sid, _), v in pack.fields.items() if sid == s.id])
              for s in task.sources}
    bounds = {std.id: sum(float(s.uncontrolled_rates_g_s[std.pollutant])*maxima[s.id] for s in task.sources)
              for std in task.standards}
    # Include a final-printed-value allowance and round outward for the certificate.
    candidates = np.logical_and.reduce([
        np.nextafter(bounds[std.id], np.inf) + 1e-5 + float(std.background) < float(std.limit)
        for std in task.standards])
    return candidates, bounds


def certified_prune(task: Scenario, pack: FieldPack) -> tuple[FieldPack, dict]:
    """Keep reported maxima exact, in addition to proving all omitted receptors pass."""
    from .evaluate import annual_statistics
    candidates, bounds = prune_candidates(task, pack)
    minima = {}
    for source in task.sources:
        arrays = [v for (sid, _), v in pack.fields.items() if sid == source.id]
        minimum = np.array(arrays[0], dtype=np.float64)
        for values in arrays[1:]:
            np.minimum(minimum, values, out=minimum)
        minima[source.id] = minimum
    lower_bounds = {}
    for pollutant in ('NO2', 'PM25', 'SO2'):
        lower = np.zeros_like(next(iter(minima.values())))
        for source in task.sources:
            efficiency = max(c.efficiency_by_pollutant[pollutant] for c in task.controls if source.id in c.applies_to)
            rate = (D(str(source.uncontrolled_rates_g_s[pollutant]))*(1-efficiency)).quantize(D('1e-9'))
            lower += float(rate)*minima[source.id]
        if pollutant == 'NO2':
            lower *= .5  # A bound on every ARM2 ratio, independent of polynomial monotonicity.
        stats = annual_statistics(lower, pack.valid_hours, pack.year_offsets, pollutant)
        for standard in task.standards:
            if standard.pollutant != pollutant:
                continue
            kind = 'annual' if standard.averaging_time == 'annual' else ('three' if standard.averaging_time == '3h' else 'daily')
            max_form = standard.form in {'max_receptor_max_annual_mean',
                                        'max_receptor_max_annual_second_highest_3h_block'}
            values = stats[kind].max(axis=0) if max_form else stats[kind].mean(axis=0)
            lower_bounds[standard.id] = float(np.nextafter(values.max(), -np.inf))-1e-5
    prunable = candidates.copy()
    for name, bound in bounds.items():
        prunable &= np.nextafter(bound, np.inf)+1e-5 < lower_bounds[name]
    keep = ~prunable
    if not keep.any():
        raise ValueError('invalid pruning certificate: no witness receptor')
    fields = {key: values[:, keep].copy(order='C') for key, values in pack.fields.items()}
    return (FieldPack(pack.task_id, fields, pack.valid_hours, pack.year_offsets,
                      pack.receptor_indices[keep], True),
            {'feasibility_candidates': int(candidates.sum()), 'pruned': int(prunable.sum()),
             'binding_lower_bounds': lower_bounds, 'kept_indices': pack.receptor_indices[keep].tolist()})


def save_pack(pack: FieldPack, path: Path, *, dtype: str = 'float64') -> None:
    if dtype not in {'float32', 'float64'}:
        raise ValueError('expected float32 or float64')
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(pack.fields)
    metadata = {'task_id': pack.task_id, 'keys': keys, 'year_offsets': pack.year_offsets,
                'binding_certified': pack.binding_certified}
    with path.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle, metadata=json.dumps(metadata), valid_hours=pack.valid_hours,
                            receptor_indices=pack.receptor_indices,
                            **{f'f{i}': np.asarray(pack.fields[key], dtype=dtype) for i, key in enumerate(keys)})
    path.with_suffix('.tmp').replace(path)
    load_pack.cache_clear()


@lru_cache(maxsize=1)
def load_pack(path: str | Path) -> FieldPack:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        fields = {tuple(key): data[f'f{i}'] for i, key in enumerate(meta['keys'])}
        for values in fields.values():
            values.flags.writeable = False
        return FieldPack(meta['task_id'], fields, data['valid_hours'], tuple(meta['year_offsets']),
                         data['receptor_indices'], meta['binding_certified'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--tasks', nargs='+')
    selection.add_argument('--manifest', nargs='+', type=Path, help='precompute only eligible=true rows')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    task_ids = args.tasks or [r['id'] for path in args.manifest for line in path.read_text().splitlines()
                             if (r := json.loads(line)).get('eligible') is True]
    print(json.dumps(precompute(task_ids, args.jobs), indent=2))
