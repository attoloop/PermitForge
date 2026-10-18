# PermitForge core

The physical verifier, lifted from the published v1 grader. It loads the preserved v1
extraction, renders a design, runs BPIP, enforces GEP, runs AERMOD for each
pollutant, and checks every receptor with the published statistical forms.

From this directory:

```sh
uv run pytest -q
uv run ruff check .
```

The default suite includes the tests marked `native`. They use the supplied
executables and run sequentially. `uv run pytest -q -m 'not native'` runs the
short checks, including a BPIP rendering comparison and a simulated timeout.
No tests download data or invoke a model API.

```python
from pathlib import Path
from permit_core.model import load_v1_instance
from permit_core.verify import verify
from permit_core.reward import prescore, reward
from permit_core.cost import cost

scenario = load_v1_instance("003bd328a392-s47")
result = verify(scenario, scenario.draft_design, Path("scratch/permit-check"))
saved = prescore(result, scenario)
score = reward(saved, cost(scenario.draft_design, scenario.task))
```

Designs retain v1's `{source_id: {control, stack_height_m}}` schema. Heights
must be finite numbers, within the published bounds and exactly representable
at the native height precision. The dictionary interface retains Decimal
values without passing through binary floats. `validate.design(text, task)`
is the unchanged StrictLoader entry point for legacy YAML. `Scenario.height_menu`
is optional: v1 loading leaves it unset; conversion can supply it later.

`verify(..., no2_tier="as_published")` retains the published NO2 cards.
`"ARM2"` selects the default AERMOD ratios and removes PVMRM-only cards;
`"PVMRM"` restores ozone and per-source in-stack ratios from the task data.
The source citations and mathematical formula are in the root `DESIGN.md`.

The returned `Verification` contains exact Decimal cost (or `None` if the
ledger is invalid), GEP heights, structured errors, binary hashes, and elapsed
native seconds keyed by `bpipprm` and pollutant. `design_values` contains a
`DesignValue` per standard at the binding receptor. `receptor_index` is
one-based, matching v1; `x` and `y` are local Cartesian metres. The full v1
per-receptor tables are retained in `receptor_values`, including on an air
quality failure. `feasible` is true only after every gate succeeds. Each
verification uses private scratch directories, removed on completion.

`PERMIT_MODELS_DIR` overrides the default `native_build/bin` directory;
`PERMIT_V1_DIR` overrides automatic discovery of `convert/v1-src`. Set these
when installing outside this checkout. The extraction and binaries are not
part of the Python wheel.

The lift origins below are relative to
`tasks/permit-forge/instances/003bd328a392-s47/` in the v1 repository, preserved
under `convert/v1-src/shared/`. Each lifted module also names its origin in
its header. Function bodies listed as unchanged are checked against the
extraction by `tests/test_source_lift.py`.

| Module | Lifted functions/classes | v1 origin | Adaptation |
| --- | --- | --- | --- |
| `native.py` | `Reject`, `require`, `dec`, `numeric`, `equivalent`, `native_lines`, `bpip_records`, `bpip_input`, `cards`, `cases`, `child_limits`, `run_native`, `bpip_results` | `environment/public_native.py` | Unchanged bodies, except that `run_native`'s disk scan skips files the model deletes mid-scan (`case_bytes`); unused imports removed |
| `native.py` | `calendar`, `parse_blocks` | `environment/service.py` | Unchanged bodies; service is never imported |
| `validate.py` | `StrictLoader`, `design`, `plot_values`, `statistics`, `admit_air_quality` | `solution/validation.py` | Unchanged bodies; common helpers imported from `native` |
| `cost.py` | `cost`, `costs` | `tests/grade.py` | Unchanged bodies |
| `render.py` | `bpip_template`, `is_number` | `environment/render.py` | Unchanged bodies |
| `render.py` | `render` | `environment/render.py` | Explicit root and dictionary input; optional copied task grammar; native BPIP callback |
| `render.py` | `design_text` | `solution/materialize.py` | Decimal import alias only |
| `verify.py` | physical flow of `grade` | `tests/grade.py` | Function interface, internally rendered inputs, local met symlink, optional tier switch, result dataclasses |
| `native_build/instrument.py` | patch script | `tests/instrument.py` | Supplied build input; no rebuild |
| `native_build/build-models.sh` | compiler script | `environment/build-models.sh` | Supplied build input; no rebuild |

Rendering equivalences: comments, whitespace, BPIP string quoting, numeric
spelling with equal Decimal values, and continuation lines for directional
arrays do not change the records. The complete draft comparison checks these
through v1's `bpip_records`, `cards`, and `equivalent`. At runtime only the
trusted met paths change, to a relative symlink into the extracted station.

The fidelity test uses the submitted oracle designs for the distinct instances
in `bind_verify/7369`, plus the additional instance in
`mechanical_matrix/7355`. The nop reports failed the old file-format gate
before recording physical values; an automatically rendered draft cannot be
compared to those failures. No fallback or reconstruction is needed for the
selected oracle designs. The test compares all recorded receptors, and writes
summary tables, source paths, submission hashes, binary comparisons, and
timings when `--evidence-dir` is supplied.

```sh
uv run pytest -q --evidence-dir ../../../results/evidence/faithful-lift \
  --junitxml=../../../results/rollouts/core-tests.xml
```

## On-demand fields

The native lift above is unchanged apart from that one scan. The integrated fast evaluator uses full
float64 memory maps, acquired and released with an on-demand cache lease:

```python
from permit_core.fields import ensure_pack
from permit_core.evaluate import evaluate
from permit_core.baselines import scenario_view
from permit_core.model import load_v2_instance

task = load_v2_instance("envs/permit/tasks", "074897d0f159-s47")
with ensure_pack(task, jobs=4) as pack:
    result = evaluate(task, scenario_view(task)["draft_design"], pack, include_receptors=True)
# Do not retain or use pack arrays after releasing the lease.
```

`ensure_pack` computes missing fields, serializes concurrent builders, and pins
the pack against LRU eviction while acquired. `PERMIT_FIELDS_CACHE_GB` defaults
to 40 GiB. Active leases and unadopted legacy gate data may exceed the cap; idle
managed packs are removed on acquisition/release. For batch work, retain the
lease through all policies and search, then `pack.release(evict=True)` if the
pack should be removed immediately. See `convert/README.md` for the runner.

Per-receptor arrays are optional in the core result; OpenEnv records them but
keeps agent-visible tool results compact. Search retains unlimited defaults in
the library; the batch runner applies explicit time/node limits and reports
whether the tree closed. Native verification still grades submissions.
