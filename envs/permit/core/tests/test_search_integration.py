"""Sound search acceleration: early rejection, safe receptor subsets, prefix reuse."""
from dataclasses import replace
from decimal import Decimal

from permit_core.baselines import scenario_view
from permit_core.search import LowerBound
from wp14_fixtures import synthetic


def test_bound_stops_after_first_violating_pollutant(monkeypatch):
    import permit_core.search as module
    task, pack = synthetic(2)
    for values in pack.fields.values():
        values *= 1000
    calls = []
    original = module.annual_statistics
    def statistics(*args):
        calls.append(args[-1])
        return original(*args)
    monkeypatch.setattr(module, 'annual_statistics', statistics)
    bounds = LowerBound(task, pack)
    calls.clear()
    assert bounds.infeasible({})
    assert calls == ['NO2']


def test_bounds_skip_certifiably_safe_receptors(monkeypatch):
    import permit_core.search as module
    task, pack = synthetic(2)
    for values in pack.fields.values():
        values[:, 1] = 1e-8
    task = replace(task, standards=tuple(replace(s, limit=Decimal(2)) for s in task.standards))
    columns = []
    original = module.annual_statistics
    def statistics(*args):
        columns.append(args[0].shape[1])
        return original(*args)
    monkeypatch.setattr(module, 'annual_statistics', statistics)
    bounds = LowerBound(task, pack)
    columns.clear()
    bounds.values({})
    assert columns and set(columns) == {1}


def test_extending_a_prefix_reuses_its_assigned_contribution():
    task, pack = synthetic(3)
    accessed = []
    class Tracked(dict):
        def __getitem__(self, key):
            accessed.append(key)
            return super().__getitem__(key)
    pack.fields = Tracked(pack.fields)
    bounds = LowerBound(task, pack)
    draft = scenario_view(task)['draft_design']
    bounds.values({'S1': draft['S1']})
    accessed.clear()
    bounds.values({'S1': draft['S1'], 'S2': draft['S2']})
    assert all(sid != 'S1' for sid, _ in accessed)
