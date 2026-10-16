from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from permit_core.reward import Anchors, hyperbolic_calibration, prescore, reward
from permit_core.cost import cost


def test_anchors_and_curve():
    anchors = Anchors(worst=D(0), best=D(100))
    assert Anchors.from_dict(anchors.to_dict()) == anchors
    values = [reward(D(x), anchors) for x in (-1, 0, 1, 50, 99, 100, 101, 400, 1000, 1000000)]
    assert values == sorted(values)
    assert values[0] == values[1] == 0
    assert reward(D(100), anchors) == pytest.approx(0.4)
    assert reward(D(400), anchors) == pytest.approx(0.8)
    assert values[-1] < 1
    with pytest.raises(ValueError):
        Anchors(worst=D(0), best=D(0))
    with pytest.raises(ValueError):
        hyperbolic_calibration(0, 1, 1)


@pytest.mark.parametrize("value", [None, True, "100", D("NaN"), D("sNaN"), D("Infinity"), float("nan"), -1])
def test_bad_prescores_are_zero(value):
    assert reward(value, Anchors(worst=D(0), best=D(100))) == 0


def test_prescore_gates(scenario):
    draft_cost = cost(scenario.draft_design, scenario.task)
    assert prescore(SimpleNamespace(feasible=True, cost=draft_cost), scenario) == D(0)
    assert prescore(SimpleNamespace(feasible=True, cost=draft_cost-D("0.01")), scenario) == D("0.01")
    assert prescore(SimpleNamespace(feasible=False, cost=D(0)), scenario) == D(-1)
    assert prescore(None, scenario) == D(-1)
