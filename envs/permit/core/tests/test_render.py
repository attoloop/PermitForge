from copy import deepcopy

import pytest

from permit_core.native import bpip_records, cards, equivalent, parse_blocks
from permit_core.render import render, with_no2_tier
from permit_core.verify import run_bpip


def test_complete_draft_render(scenario, tmp_path):
    out = tmp_path / "permit"
    render(scenario.task_root, scenario.draft_design, out,
           run_bpip=lambda inp, dest: run_bpip(inp, dest, scenario.task["guards"]["native_seconds"]))
    for name, parser in (("bpip.inp", bpip_records), ("aermod.inp", cards)):
        got = parser((out / name).read_text())
        want = parser((scenario.verifier_root / "draft" / name).read_text())
        assert len(got) == len(want)
        for a, b in zip(got, want, strict=True):
            assert len(a) == len(b)
            assert all(equivalent(x, y) for x, y in zip(a, b, strict=True))


def test_tier_cards(scenario):
    task = scenario.task
    original = deepcopy(task)
    assert with_no2_tier(task, "as_published") == task
    switched = with_no2_tier(task, "ARM2")
    rows = [r["fields"] for r in switched["locked_keyword_table"]["NO2"]]
    assert ["CO", "MODELOPT", "DFAULT", "CONC", "ARM2"] in rows
    assert ["SO", "SRCGROUP", "ALL"] in rows
    assert not {r[1] for r in rows} & {"OZONEVAL", "NO2RATIO", "ARMRATIO"}
    restored = with_no2_tier(switched, "PVMRM")
    assert restored["locked_keyword_table"]["NO2"] == task["locked_keyword_table"]["NO2"]
    assert switched["locked_keyword_table"]["SO2"] == task["locked_keyword_table"]["SO2"]
    assert switched["locked_keyword_table"]["PM25"] == task["locked_keyword_table"]["PM25"]
    assert task == original
    with pytest.raises(ValueError):
        with_no2_tier(task, "unknown")
    blocks, profiles = parse_blocks((scenario.verifier_root / "draft/aermod.inp").read_text(), task)
    assert [p for p, _ in blocks] == task["scenario"]["pollutants"]
    assert not profiles
