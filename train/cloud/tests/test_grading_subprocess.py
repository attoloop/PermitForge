"""The native grading subprocess must use the worker's field-cache volume, not the local v1 default."""
import json
import sys

from train.cloud import evaluation, verify


def test_grading_command_carries_the_configured_cache_root(tmp_path, monkeypatch):
    from permit_core import fields
    monkeypatch.setattr(fields, 'cache_root', lambda: tmp_path/'cloud-fields')
    command = evaluation.grading_command(tmp_path/'tasks', 'T1', tmp_path/'design.json', tmp_path/'grade.json')
    assert command[:3] == [sys.executable, '-m', 'train.cloud.verify']
    assert command[-1] == str(tmp_path/'cloud-fields')


def test_grader_binds_the_cache_root_before_native_slots(tmp_path, monkeypatch):
    # The first H200 base evaluation failed here: a fresh process has no binding and no PERMIT_V1_DIR.
    import permit_core.fields
    import permit_openenv.evaluator as evaluator
    import permit_openenv.tasks as tasks
    from permit_core.native_pool import native_slot
    monkeypatch.delenv('PERMIT_V1_DIR', raising=False)
    monkeypatch.setattr(permit_core.fields, 'cache_root', permit_core.fields.cache_root)
    class Grader:
        def evaluate(self, task, design):
            with native_slot():
                return {'valid': True, 'feasible': True, 'task': task, 'design': design}
    class Pack:
        def get(self, task_id):
            return task_id
    monkeypatch.setattr(evaluator, 'VerifyEvaluator', Grader)
    monkeypatch.setattr(tasks, 'load_pack', lambda path: Pack())
    design, grade, root = tmp_path/'design.json', tmp_path/'grade.json', tmp_path/'cloud-fields'
    design.write_text(json.dumps({'S1': {'control': 'none', 'height_m': 10}}))
    monkeypatch.setattr(sys, 'argv', ['verify', str(tmp_path), 'T1', str(design), str(grade), str(root)])
    verify.main()
    assert json.loads(grade.read_text())['feasible'] is True
    assert (root/'.native-slots').is_dir()
