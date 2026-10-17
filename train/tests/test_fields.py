import json

import numpy as np
import pytest

from train.grpo import FieldStore


@pytest.mark.parametrize('task_id,dtype,error', [
    ('wrong-task', np.float64, 'identity'),
    ('synthetic-train', np.float32, 'float64'),
    ('synthetic-train', np.float64, None),
])
def test_readonly_fields_identity_and_precision(task_pack, tmp_path, task_id, dtype, error):
    task, pack = task_pack[0].at('train', 0), task_pack[1]
    keys = list(pack.fields)
    path = tmp_path/f'{task.task_id}.npz'
    np.savez(path, metadata=json.dumps({'keys': keys, 'task_id': task_id,
                                      'year_offsets': pack.year_offsets, 'binding_certified': False}),
             valid_hours=pack.valid_hours, receptor_indices=pack.receptor_indices,
             **{f'f{i}': pack.fields[key].astype(dtype) for i, key in enumerate(keys)})
    before = path.read_bytes()
    store = FieldStore(tmp_path)
    if error:
        with pytest.raises(ValueError, match=error):
            store(task)
    else:
        loaded = store(task)
        assert loaded is store(task)
        assert not loaded.fields[keys[0]].flags.writeable
    assert path.read_bytes() == before


def test_missing_pack_never_computes_fields(task_pack, tmp_path, monkeypatch):
    import permit_core.fields
    monkeypatch.setattr('train.grpo.met_calendar', lambda scenario: (np.array([2020010101]), np.array([True])))
    monkeypatch.setattr(permit_core.fields, 'compute_field', lambda *a: pytest.fail('native work is forbidden'))
    with pytest.raises(FileNotFoundError):
        FieldStore(tmp_path)(task_pack[0].at('train', 0))
