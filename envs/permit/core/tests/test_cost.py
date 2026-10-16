from copy import deepcopy
from decimal import Decimal as D

from permit_core.cost import cost, costs


def test_draft_maximum_and_floor(scenario):
    cd, cm, floor = costs(scenario.task, scenario.draft_design)
    expected = sum(
        D(str(c["annualized_usd"]))
        for c in scenario.task["costs"]["controls"]
        if c["id"].endswith("_maximum")
    ) + sum(
        D(str(scenario.task["costs"]["stack_usd_per_m"]))
        * (s.height_max_m - s.height_min_m)
        for s in scenario.sources
    )
    assert cd == cm == expected
    assert floor == D(-1)
    assert isinstance(cd, D)


def test_exact_decimal_and_per_source_price(scenario):
    task = deepcopy(scenario.task)
    task["costs"]["stack_usd_per_m"] = {s.id: D("0.07") for s in scenario.sources}
    ledger = {s.id: {"control": "none", "stack_height_m": s.height_min_m + D("0.01")}
              for s in scenario.sources}
    assert cost(ledger, task) == D("0.0007") * len(scenario.sources)
