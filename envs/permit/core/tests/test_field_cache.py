"""On-demand cache lifecycle, real interprocess locking, and failure recovery."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import multiprocessing
from pathlib import Path
import time

import numpy as np
import pytest

from permit_core import fields
from wp14_fixtures import synthetic


@pytest.fixture
def cache(tmp_path, monkeypatch):
    task, data = synthetic()
    monkeypatch.setattr(fields, 'cache_root', lambda: tmp_path)
    monkeypatch.setattr(fields, 'pack_identity', lambda t: t.id)
    monkeypatch.setattr(fields, 'met_calendar', lambda t: (np.full(240, 20010101), data.valid_hours))
    calls = []
    def compute(t, sid, height):
        calls.append((t.id, sid, fields.height_key(height)))
        time.sleep(.02)
        return fields.Field(data.fields[sid, fields.height_key(height)], data.valid_hours,
                            np.full(240, 20010101), .02, {'task_id': t.id})
    monkeypatch.setattr(fields, 'compute_field', compute)
    return task, data, calls


def test_same_task_threads_compute_once_and_release(cache):
    task, expected, calls = cache
    def use(_):
        with fields.ensure_pack(task, jobs=2) as pack:
            assert pack.task_id == task.id
            for key, value in pack.fields.items():
                assert value.dtype == np.float64
                np.testing.assert_array_equal(value, expected.fields[key])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(use, range(4)))
    assert len(calls) == len(expected.fields)
    with fields.ensure_pack(task, jobs=1):
        pass
    assert len(calls) == len(expected.fields)


def test_process_waits_for_builder_and_active_pack_survives_eviction(cache):
    task, expected, _ = cache
    context = multiprocessing.get_context('fork')
    ready, release = context.Event(), context.Event()
    def hold():
        with fields.ensure_pack(task, jobs=2):
            ready.set()
            assert release.wait(10)
    process = context.Process(target=hold)
    process.start()
    try:
        assert ready.wait(10)
        fields.evict_packs(max_bytes=0)
        assert (fields.cache_root()/task.id).exists()
        with fields.ensure_pack(task, jobs=1) as pack:
            assert len(pack.fields) == len(expected.fields)
    finally:
        release.set()
        process.join(10)
        if process.is_alive():
            process.terminate()
    assert process.exitcode == 0
    fields.evict_packs(max_bytes=0)
    assert not (fields.cache_root()/task.id).exists()


def test_same_task_processes_compute_once(cache, monkeypatch):
    task, data, _ = cache
    original = fields.compute_field
    log = fields.cache_root()/'calls.txt'
    def compute(*args):
        with log.open('a') as handle:
            handle.write('computed\n')
        return original(*args)
    monkeypatch.setattr(fields, 'compute_field', compute)
    context = multiprocessing.get_context('fork')
    start = context.Event()
    def use():
        assert start.wait(10)
        with fields.ensure_pack(task, jobs=2) as pack:
            assert len(pack.fields) == len(data.fields)
    processes = [context.Process(target=use) for _ in range(3)]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(10)
        if process.is_alive():
            process.terminate()
        assert process.exitcode == 0
    assert len(log.read_text().splitlines()) == len(data.fields)


def test_distinct_tasks_share_twelve_native_slots(cache, monkeypatch):
    task, _, _ = cache
    context = multiprocessing.get_context('fork')
    active, peak = context.Value('i', 0), context.Value('i', 0)
    guard, start = context.Lock(), context.Event()
    original = fields.compute_field
    def compute(*args):
        with guard:
            active.value += 1
            peak.value = max(peak.value, active.value)
        try:
            time.sleep(.2)
            return original(*args)
        finally:
            with guard:
                active.value -= 1
    monkeypatch.setattr(fields, 'compute_field', compute)
    def use(index):
        assert start.wait(10)
        with fields.ensure_pack(replace(task, id=f'task-{index}'), jobs=4):
            pass
    processes = [context.Process(target=use, args=(i,)) for i in range(4)]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(10)
        if process.is_alive():
            process.terminate()
        assert process.exitcode == 0
    assert 1 < peak.value <= 12


def test_lru_cap_and_explicit_evict_after(cache, monkeypatch):
    task, _, _ = cache
    other = replace(task, id='other')
    with fields.ensure_pack(task, jobs=1):
        pass
    with fields.ensure_pack(other, jobs=1):
        pass
    with fields.ensure_pack(task, jobs=1):
        pass
    size = sum(p.stat().st_size for p in (fields.cache_root()/task.id).rglob('*') if p.is_file())
    fields.evict_packs(max_bytes=size+100)
    assert (fields.cache_root()/task.id).exists()
    assert not (fields.cache_root()/other.id).exists()
    monkeypatch.setenv('PERMIT_FIELDS_CACHE_GB', '0')
    with fields.ensure_pack(task, jobs=1) as pack:
        assert Path(pack.path).exists()
    assert not (fields.cache_root()/task.id).exists()


def test_failed_build_can_resume_and_corruption_is_detected(cache, monkeypatch):
    task, _, calls = cache
    original = fields.compute_field
    def fail(t, sid, height):
        if sid == 'S2':
            raise RuntimeError('interrupted')
        return original(t, sid, height)
    monkeypatch.setattr(fields, 'compute_field', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        fields.ensure_pack(task, jobs=1)
    monkeypatch.setattr(fields, 'compute_field', original)
    with fields.ensure_pack(task, jobs=1):
        pass
    assert len(calls) == 4
    path = fields.cache_root()/task.id/'S1/10.npy'
    path.write_bytes(b'broken')
    with fields.ensure_pack(task, jobs=1) as pack:
        assert pack.fields['S1', '10'].shape == (240, 2)
    assert len(calls) == 5


def test_failed_partial_build_is_subject_to_cap(cache, monkeypatch):
    task, _, _ = cache
    monkeypatch.setenv('PERMIT_FIELDS_CACHE_GB', '0')
    original = fields.compute_field
    def fail(t, sid, height):
        if sid == 'S2':
            raise RuntimeError('interrupted')
        return original(t, sid, height)
    monkeypatch.setattr(fields, 'compute_field', fail)
    with pytest.raises(RuntimeError):
        fields.ensure_pack(task, jobs=1)
    assert not (fields.cache_root()/task.id).exists()


@pytest.mark.parametrize('jobs', [0, 13, True])
def test_invalid_parallelism(cache, jobs):
    with pytest.raises(ValueError, match='jobs'):
        fields.ensure_pack(cache[0], jobs=jobs)


def test_cache_root_without_v1_extraction(monkeypatch):
    def missing():
        raise FileNotFoundError('Set PERMIT_V1_DIR to the extracted v1 source directory')
    monkeypatch.setattr(fields, 'extraction_root', missing)
    assert fields.cache_root() == Path(fields.__file__).resolve().parents[4]/'convert/cache/fields'
