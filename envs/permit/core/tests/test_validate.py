from copy import deepcopy
from decimal import Decimal as D

import pytest

from permit_core.validate import Reject, design, validate_ledger
from permit_core.verify import verify


@pytest.mark.parametrize("value", [True, False, "65", None, [], {}, float("nan"),
                                  float("inf"), D("NaN"), -1, 10**100, D("65.001")])
def test_hostile_heights(scenario, value, tmp_path):
    ledger = deepcopy(scenario.draft_design)
    ledger["S1"]["stack_height_m"] = value
    result = verify(scenario, ledger, tmp_path)
    assert not result.feasible
    assert result.errors
    assert result.native_seconds == {}


@pytest.mark.parametrize("value", [None, [], True, {}, {"unknown": {}}, {1: {}},
                                  {"S1": {"control": [], "stack_height_m": 65}}])
def test_hostile_shapes(scenario, value, tmp_path):
    assert not verify(scenario, value, tmp_path).feasible


def test_wrong_control_and_extra_keys(scenario, tmp_path):
    for control in ("S2_maximum", "unknown", [], True):
        ledger = deepcopy(scenario.draft_design)
        ledger["S1"]["control"] = control
        assert not verify(scenario, ledger, tmp_path).feasible
    ledger = deepcopy(scenario.draft_design)
    ledger["S1"]["extra"] = 1
    assert not verify(scenario, ledger, tmp_path).feasible


@pytest.mark.parametrize("text", [
    "S1: 1\nS1: 2", "S1: &x {}\nS2: *x", "!!python/object:builtins.object {}",
    "---\n{}\n---\n{}", "? [S1, S2]\n: {}", "S1: {}\n" * 50,
])
def test_strict_yaml(scenario, text):
    with pytest.raises(Reject):
        design(text, scenario.task)


def test_optional_height_menu_preserves_continuous_v1(scenario):
    ledger = deepcopy(scenario.draft_design)
    ledger["S1"]["stack_height_m"] -= D("0.01")
    assert validate_ledger(ledger, scenario.task) == ledger
    menu = {s.id: [s.height_min_m, s.height_max_m] for s in scenario.sources}
    with pytest.raises(Reject, match="menu"):
        validate_ledger(ledger, scenario.task, height_menu=menu)


def test_exact_input_is_not_mutated(scenario):
    ledger = deepcopy(scenario.draft_design)
    assert validate_ledger(ledger, scenario.task) == ledger == scenario.draft_design
