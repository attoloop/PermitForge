"""Receptor arrays reuse the exact binding-value statistics and rounding."""
from dataclasses import replace
from decimal import Decimal

import pytest

from permit_core.baselines import scenario_view
from permit_core.evaluate import evaluate
from wp14_fixtures import synthetic


def test_receptors_match_independent_columns_and_binding_values():
    task, pack = synthetic(3, seed=42)
    design = scenario_view(task)['draft_design']
    public = evaluate(task, design, pack)
    result = evaluate(task, design, pack, include_receptors=True)
    assert {k: v for k, v in result.items() if not k.startswith('receptor_')} == public
    for column in range(2):
        single = replace(pack, fields={k: v[:, column:column+1] for k, v in pack.fields.items()},
                         receptor_indices=pack.receptor_indices[column:column+1], binding_certified=True)
        expected = evaluate(task, design, single)
        for standard in expected['standards']:
            assert result['receptor_values'][standard['id']][column] == standard['value']
            assert result['receptor_margins'][standard['id']][column] == standard['margin']
    for standard in result['standards']:
        values = result['receptor_values'][standard['id']]
        assert max(values) == standard['value']
        assert min(result['receptor_margins'][standard['id']]) == standard['margin']
        assert all(Decimal(str(v)).is_finite() for v in values)


def test_full_receptor_request_rejects_a_pruned_pack():
    task, pack = synthetic()
    pack = replace(pack, fields={k: v[:, :1] for k, v in pack.fields.items()},
                   receptor_indices=pack.receptor_indices[:1], binding_certified=True)
    with pytest.raises(ValueError, match='full receptor'):
        evaluate(task, scenario_view(task)['draft_design'], pack, include_receptors=True)
