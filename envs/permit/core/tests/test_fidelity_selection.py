import importlib.util
import json
from pathlib import Path

from conftest import requires_v1
from permit_core.model import load_v1_instance


@requires_v1
def test_seeded_unique_designs_include_both_boundary_sides(monkeypatch):
    path = Path(__file__).resolve().parents[4] / 'convert/fidelity.py'
    spec = importlib.util.spec_from_file_location('fidelity_selection', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task = load_v1_instance('074897d0f159-s47')

    def fake_evaluate(task, design, pack):
        margin = sum(r['height_m'] for r in design.values()) - 70
        return {'standards': [{'margin': margin, 'limit': 100}]}

    monkeypatch.setattr(module, 'evaluate', fake_evaluate)
    designs = module.designs_for(task, None)
    assert len(designs) == len({json.dumps(d, sort_keys=True) for d in designs}) == 40
    assert designs == module.designs_for(task, None)
    margins = [fake_evaluate(task, design, None)['standards'][0]['margin'] for design in designs[3:10]]
    assert any(m < 0 for m in margins) and any(m >= 0 for m in margins)
