# PermitForge measurement evidence

These receipts support the public design and provenance claims. Historical
measurements remain historical: they are not model calibration or fresh runtime
benchmarks. Files were relocated from the development archive; command and source
path strings were made repository-relative. Numerical measurements are preserved.

| Artifact | Contents | Reproduce or inspect |
|---|---|---|
| `conversion.json` | Original scenario inventory and split counts | `python3 envs/permit/openenv/space_root/report_evidence.py` |
| `fidelity.json` | Saved menu designs, native/fast comparisons, timings and precision checks | `python3 envs/permit/openenv/space_root/report_evidence.py`; native rerun: `uv run --project envs/permit/core python convert/fidelity.py --jobs 4` |
| `faithful-lift/`, `faithful-lift.md` | Published v1 submissions compared with the lifted verifier; binding values and binary hashes | `uv run --directory envs/permit/core pytest tests/test_lift_faithful.py --evidence-dir ../../../results/evidence/faithful-lift -q` |
| `search/`, `pipeline-budget.json` | Before/after search timings and estimated CPU budgets | `python3 convert/report_search.py`; native commands are recorded in each receipt |
| `space-container-smoke.json` | Historical image endpoint and native/fast tool measurements | `python3 -m json.tool results/evidence/space-container-smoke.json`; current smoke: `uv run --project envs/permit/openenv python envs/permit/openenv/space_root/smoke.py --url http://127.0.0.1:7860 --output results/evidence/space-smoke.json` |
| `design-claims.json` | Public numerical claims mapped to source files and commands | `python3 -m json.tool results/evidence/design-claims.json` |
| `provenance/` | Preserved generator and source declarations supporting the acquisition tables | `python3 -m json.tool results/evidence/provenance/origins.json` |

All commands run from the repository root. Native reruns require preserved inputs,
converted task data and the pinned binaries. Search measurements are diagnostics;
reward anchors will come only from model calibration on every train and eval task.

`search/reference.py.txt` preserves the historical search implementation byte for
byte; its hash matches the before-measurement receipts. Their commands use
`--reference-file`, so reruns do not depend on retaining old Git commits. The
original relative source path and revision are recorded in each receipt.
