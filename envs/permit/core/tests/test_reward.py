from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from permit_core.reward import prescore, reward
from permit_core.cost import cost


def test_saved_fraction():
    values = [reward(D(x), D(100)) for x in (-1, 0, 1, 50, 99, 100)]
    assert values == [0, 0, .01, .5, .99, 1]
    assert reward(D("0.01"), D("0.03")) == pytest.approx(1/3)


@pytest.mark.parametrize("value", [None, True, "100", D("NaN"), D("sNaN"), D("Infinity"), float("nan"), -1])
def test_bad_prescores_are_zero(value):
    assert reward(value, D(100)) == 0


def test_prescore_gates(scenario):
    draft_cost = cost(scenario.draft_design, scenario.task)
    assert prescore(SimpleNamespace(feasible=True, cost=draft_cost), scenario) == D(0)
    assert prescore(SimpleNamespace(feasible=True, cost=draft_cost-D("0.01")), scenario) == D("0.01")
    assert prescore(SimpleNamespace(feasible=False, cost=D(0)), scenario) == D(-1)
    assert prescore(None, scenario) == D(-1)
