from decimal import Decimal

import pytest

from conftest import EXTRACT
from permit_core.model import ControlRow, Scenario, Source, Standard, load_v1_instance


def test_loaded_scenario(scenario):
    assert isinstance(scenario, Scenario)
    assert all(isinstance(s, Source) for s in scenario.sources)
    assert all(isinstance(c, ControlRow) for c in scenario.controls)
    assert all(isinstance(s, Standard) for s in scenario.standards)
    assert scenario.met_path == EXTRACT / "met" / scenario.station
    assert scenario.buildings["buildings"]
    assert len(scenario.receptors) == scenario.task["screening"]["NO2"]["receptors"]
    assert all(isinstance(s.limit, Decimal) for s in scenario.standards)


@pytest.mark.parametrize("id", ["../anything", "/task", "", "unknown"])
def test_invalid_instance_ids(id):
    with pytest.raises(ValueError):
        load_v1_instance(id)
