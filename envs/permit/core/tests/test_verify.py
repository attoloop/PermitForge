from decimal import Decimal as D
import importlib

from permit_core.verify import verify


def test_gep_gate_before_aermod(scenario, tmp_path, monkeypatch):
    module = importlib.import_module("permit_core.verify")
    monkeypatch.setattr(module, "run_bpip", lambda *args: None)
    monkeypatch.setattr(module, "bpip_results", lambda *args: ({}, {s.id: D(65) for s in scenario.sources}))
    result = verify(scenario, scenario.draft_design, tmp_path)
    assert not result.feasible
    assert result.errors[0]["code"] == "E_GEP"
    assert set(result.native_seconds) == {"bpipprm"}


def test_missing_binary_is_reported(scenario, tmp_path, monkeypatch):
    monkeypatch.setenv("PERMIT_MODELS_DIR", str(tmp_path))
    result = verify(scenario, scenario.draft_design, tmp_path)
    assert not result.feasible
    assert result.errors[0]["code"] == "E_OUTPUT"
    assert "FileNotFoundError" in result.errors[0]["detail"]
    assert result.native_seconds == {}
