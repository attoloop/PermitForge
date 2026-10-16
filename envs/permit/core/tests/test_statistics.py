from copy import deepcopy
from decimal import Decimal as D

import pytest

from permit_core.validate import Reject, admit_air_quality, statistics


@pytest.mark.parametrize("pollutant", ["NO2", "PM25", "SO2"])
def test_statistical_forms_and_half_even(scenario, tmp_path, pollutant):
    task = deepcopy(scenario.task)
    task["screening"][pollutant]["receptors"] = 1
    task["locked_keyword_table"][pollutant] = [{"fields": ["RE", "DISCCART", "0", "0"]}]
    for period in task["scenario"]["averaging_times"][pollutant]:
        filename = "annual.plt" if period == "annual" else f"period_{period[:-1]}.plt"
        label = "ANNUAL" if period == "annual" else f"{period[:-1]}-HR"
        (tmp_path / filename).write_text(f"0 0 7.12345 0 0 0 {label} ALL 1\n")
    kinds = ["ANNUAL", "DAILY"] + (["THREE"] if pollutant == "SO2" else [])
    path = tmp_path / "pf_stats.txt"
    path.write_text("".join(f"{kind} {year} 1 {value}\n"
                            for kind in kinds
                            for year, value in enumerate(["1", "2", "3", "4", "5.000025"], 1)))
    values = statistics(path, pollutant, task)
    for name, rows in values.items():
        value = D(rows[0]["modeled"])
        if name == "SO2_1h":
            assert value == D("3.00000")  # Five-year mean is exactly 3.000005.
        elif name in {"NO2_annual", "SO2_3h"}:
            assert value == D("5.00002")
        else:
            assert value == D("7.12345")  # Native F13.5 means, not recomputed means.
    with path.open("a") as file:
        file.write("ANNUAL 1 1 1\n")
    with pytest.raises(Reject, match="duplicate"):
        statistics(path, pollutant, task)


def test_every_receptor_and_exact_zero_margin():
    table = {"standard": [{"receptor": 1, "margin": "0", "total": "1", "limit": "1"}]}
    admit_air_quality(table)
    table["standard"].append({"receptor": 2, "margin": "-0.0000000001", "total": "1.0000000001", "limit": "1"})
    with pytest.raises(Reject, match="receptor 2"):
        admit_air_quality(table)
