from dataclasses import replace
from decimal import Decimal

from permit_core.verify import Verification, DesignValue
from permit_openenv.evaluator import VerifyEvaluator, select_evaluator
from permit_openenv.geometry import wind_rose
from permit_openenv.tasks import load_pack, height_menu


def test_v1_fallback_and_menu(tmp_path):
    pack = load_pack(tmp_path)
    task = pack.get('003bd328a392-s47')
    assert task.split == 'train' and not hasattr(task, 'anchors')
    source = task.scenario.sources[0]
    menu = height_menu(source.height_min_m, source.gep_height_m)
    assert menu[0] == source.height_min_m and menu[-1] == source.gep_height_m
    assert len(menu) <= 8 and list(menu) == sorted(set(menu))
    assert all(h % Decimal('.5') == 0 for h in menu[1:-1])


def test_verify_adapter_forces_arm2(pack, monkeypatch):
    import permit_core.verify as module
    calls = []
    def verify(scenario, design, workdir, *, no2_tier):
        calls.append((scenario, design, no2_tier))
        return Verification(feasible=True, cost=Decimal('100.12'), design_values={
            'NO2_1h': DesignValue(Decimal('2'), Decimal('3'), Decimal('10'), Decimal('5'),
                                  1, Decimal('-250'), Decimal('250'))})
    monkeypatch.setattr(module, 'verify', verify)
    task = pack.get('003bd328a392-s47')
    result = VerifyEvaluator().evaluate(task, task.view()['draft_design'])
    assert calls[0][2] == 'ARM2'
    assert 'stack_height_m' in calls[0][1]['S1']
    assert result['cost'] == 100.12 and result['feasible']
    assert result['standards'][0]['binding_receptor'] == {'index': 1, 'x_m': -250, 'y_m': 250}


def test_evaluator_selection_is_explicit(monkeypatch):
    monkeypatch.setenv('PERMIT_EVALUATOR', 'verify')
    assert isinstance(select_evaluator(), VerifyEvaluator)
    monkeypatch.setenv('PERMIT_EVALUATOR', 'typo')
    import pytest
    with pytest.raises(ValueError):
        select_evaluator()


def test_wind_rose_bins_missing_calm_and_north(tmp_path):
    path = tmp_path / 'test.sfc'
    lines = ['header']
    for speed, direction in [(0, 0), (.49, 100), (.5, 0), (2, 22.5), (4, 90),
                              (6, 180), (8, 270), (10, 359), (12, 360), (-99, -999)]:
        fields = ['2020', '1', '1', '1', '1'] + ['0'] * 10 + [str(speed), str(direction)]
        lines.append(' '.join(fields))
    path.write_text('\n'.join(lines))
    rose = wind_rose(path)
    assert rose['calm_hours'] == 2
    assert sum(map(sum, rose['counts'])) == 7
    assert rose['counts'][0] == [1, 0, 0, 0, 0, 2]
    assert rose['counts'][1][1] == 1
    assert rose['counts'][4][2] == 1


def test_pack_filters_ineligible(pack):
    from permit_openenv.tasks import TaskPack
    task = pack.get('003bd328a392-s47')
    excluded = replace(task, record=dict(task.record, eligible=False))
    assert TaskPack([excluded]).count('train') == 0


def test_fast_adapter_uses_shared_contract_and_npz(pack, tmp_path, monkeypatch):
    import numpy as np
    from permit_core.fields import FieldPack, height_key, save_pack
    from permit_openenv.evaluator import FastEvaluator
    task = pack.get('003bd328a392-s47')
    scenario = task.scenario
    fields = {(s.id, height_key(h)): np.full((8*24, len(scenario.receptors)), 1e-5)
              for s in scenario.sources for h in scenario.height_menu[s.id]}
    data = FieldPack(task.task_id, fields, np.ones(8*24, dtype=bool), (0, 8*24),
                     np.arange(len(scenario.receptors)))
    save_pack(data, tmp_path / f'{task.task_id}.npz')
    monkeypatch.setenv('PERMIT_FIELDS_DIR', str(tmp_path))
    result = FastEvaluator().evaluate(task, task.view()['draft_design'])
    assert result['valid'] and result['feasible']
    assert len(result['standards']) == len(scenario.standards)


def test_default_evaluator_graceful_when_fast_module_absent(monkeypatch):
    import permit_openenv.evaluator as module
    real_import = module.importlib.import_module
    def missing(name):
        if name == 'permit_core.evaluate':
            raise ImportError(name)
        return real_import(name)
    monkeypatch.setattr(module.importlib, 'import_module', missing)
    monkeypatch.delenv('PERMIT_EVALUATOR', raising=False)
    assert isinstance(select_evaluator(), VerifyEvaluator)
