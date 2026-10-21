import hashlib
import json
from decimal import Decimal as D

import pytest

from conftest import EXTRACT, requires_v1
from permit_core.model import load_v1_instance
from permit_core.verify import verify


pytestmark = requires_v1


def evidence_cases():
    if not EXTRACT.exists():
        return []
    found = {}
    for prefix in ("bind_verify/7369", "mechanical_matrix/7355"):
        for report in sorted((EXTRACT / "evidence" / prefix).rglob("report.json")):
            if "oracle" not in report.parts:
                continue
            instance = report.parent.parent.name.split("__")[0]
            found.setdefault(instance, report)
    assert len(found) >= 5
    return list(found.items())


@pytest.mark.native
@pytest.mark.parametrize("instance,report_path", evidence_cases(), ids=lambda x: str(x) if isinstance(x, str) else "v1")
def test_published_verifier(instance, report_path, tmp_path, evidence_dir):
    scenario = load_v1_instance(instance)
    submitted = report_path.parent.parent / "artifacts/submission/permit/design.yaml"
    ledger = json.loads(submitted.read_text(), parse_float=D)
    expected = json.loads(report_path.read_text())
    result = verify(scenario, ledger, tmp_path, no2_tier="as_published")
    rows = []
    for standard, values in expected["design_values"].items():
        ours = result.receptor_values.get(standard, [])
        assert len(ours) == len(values), result.errors
        for old, new in zip(values, ours, strict=True):
            assert new["receptor"] == old["receptor"]
            for key in ("modeled", "background", "limit", "margin", "total"):
                assert D(new[key]).quantize(D(".00001")) == D(old[key]).quantize(D(".00001"))
        peak = max(values, key=lambda row: D(row["modeled"]))
        summary = result.design_values[standard]
        assert summary.receptor_index == peak["receptor"]
        assert (summary.x, summary.y) == scenario.receptors[peak["receptor"]-1]
        assert summary.margin == summary.limit - summary.value - summary.background
        rows.append({"standard": standard, "v1": peak["modeled"], "ours": str(summary.value),
                     "match": D(peak["modeled"]) == summary.value,
                     "receptors_compared": len(values)})
    assert result.feasible == bool(expected["gate_pass"]), result.errors
    assert result.cost == D(expected["costs"]["submission"])
    hashes = {name: {"v1": digest, "ours": result.binary_sha256[name],
                     "match": digest == result.binary_sha256[name]}
              for name, digest in expected.get("binary_sha256", {}).items()}
    evidence = {"instance": instance, "report": str(report_path.relative_to(EXTRACT)),
                "submission": str(submitted.relative_to(EXTRACT)),
                "submission_sha256": hashlib.sha256(submitted.read_bytes()).hexdigest(),
                "feasible": result.feasible, "standards": rows, "binary_sha256": hashes,
                "native_seconds": result.native_seconds}
    (evidence_dir / f"{instance}.json").write_text(json.dumps(evidence, indent=2)+"\n")


@pytest.mark.native
def test_arm2_native_draft(tmp_path, evidence_dir):
    scenario = load_v1_instance("00409ef904a8-s47")
    result = verify(scenario, scenario.draft_design, tmp_path, no2_tier="ARM2")
    assert result.feasible, result.errors
    assert set(result.design_values) == {s.id for s in scenario.standards}
    (evidence_dir / "arm2-smoke.json").write_text(json.dumps({
        "instance": scenario.id, "feasible": result.feasible,
        "native_seconds": result.native_seconds,
        "design_values": {k: str(v.value) for k, v in result.design_values.items()},
    }, indent=2)+"\n")
