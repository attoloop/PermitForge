"""Space build contract: uv run --directory envs/permit/openenv pytest tests/test_space_container.py -q."""

from pathlib import Path
import runpy

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
SPACE = ROOT / 'envs/permit/openenv'


def instructions(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith('#')]


def test_native_stage_matches_reference():
    reference = instructions(ROOT / 'envs/permit/core/native_build/Dockerfile')
    reference = reference[:reference.index('FROM scratch')]
    actual = instructions(SPACE / 'Dockerfile')
    actual = actual[:next(i for i, line in enumerate(actual[1:], 1) if line.startswith('FROM '))]
    # Only the repository-root COPY prefixes may differ. Using the identical
    # build script also fixes the compiler flags and source order, not just apt.
    actual = [line.replace('envs/permit/core/native_build/', '') for line in actual]
    assert actual == reference


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_bake_requires_float64_and_eval_tasks(monkeypatch, dtype):
    from permit_core import fields
    from permit_openenv import tasks

    calls = []

    class Task:
        task_id = 'showcase'
        split = 'eval'
        scenario = object()

    class Catalog:
        def get(self, task_id):
            assert task_id == Task.task_id
            return Task()

    class Pack:
        fields = {('S1', '65'): np.zeros((2, 2), dtype=dtype)}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append('released')

    def ensure(task, *, jobs):
        assert task is Task.scenario
        calls.append(jobs)
        return Pack()

    monkeypatch.setattr(tasks, 'load_pack', lambda: Catalog())
    monkeypatch.setattr(fields, 'ensure_pack', ensure)
    bake = runpy.run_path(str(SPACE / 'space_root/bake_showcases.py'))['bake']
    if dtype == np.float32:
        with pytest.raises(ValueError, match='float64'):
            bake(['showcase'], jobs=1)
    else:
        bake(['showcase'], jobs=1)
    assert calls == [1, 'released']
    Task.split = 'train'
    with pytest.raises(ValueError, match='eval'):
        bake(['showcase'], jobs=1)
