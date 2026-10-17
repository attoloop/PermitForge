# PermitForge v2: design notes

The details behind the [README](README.md): task data, physics, rules, reward,
search and tests. [CONTRACT.md](CONTRACT.md) defines the public
interface; [SOURCES.md](SOURCES.md) records input provenance. Model results remain **TODO**.
All tasks are admitted after native ARM2 draft verification; baseline/search
scores do not filter the release.

## Packs

The existing synthetic facilities are converted without regeneration. The source
inventory contains **360 tasks**, **230 train / 130 eval**, with **15 weather
stations** and **5 facility archetypes**. Every source task is admitted after native ARM2 draft verification.

| Split | Stations (source tasks) | Total |
|---|---|---:|
| train | KALB 21, KAMA 20, KCRP 29, KELP 21, KGRB 26, KIAD 19, KMSP 21, KPIT 17, KSHV 28, KTUS 28 | 230 |
| eval | KABQ 22, KDDC 33, KJAX 23, KOMA 20, KTOP 32 | 130 |

| Archetype | Source tasks |
|---|---:|
| gas_boilers | 74 |
| simple_cycle_turbine | 63 |
| process_heaters | 64 |
| kiln | 78 |
| small_refinery_mix | 81 |

Source counts and distributions come from
[conversion evidence](results/evidence/conversion.json); station membership
comes from `envs/permit/tasks/permit-v2-{train,eval}/tasks.jsonl`.
Reproduce these tables without modifying data:

```sh
python3 envs/permit/openenv/space_root/report_evidence.py
```

Each facility has **2–6 sources**, **1–4 buildings** and **184 receptors**, using
**2020–2024** hourly weather at its station. Terrain is flat. Source locations,
stack diameter/temperature/velocity, buildings, weather, uncontrolled emissions,
backgrounds and control rows remain physically unchanged from v1. Train and eval
stations are disjoint. “Held out” means station-disjoint; the original scenarios
were published as v1, so it does not imply unseen task data.

Preserved converted inputs live under `envs/permit/tasks/data/` and `met/`. The following command
recounts geometry and weather directly from those files:

```sh
python3 - <<'PY'
import json
from pathlib import Path
root = Path('envs/permit/tasks/data')
sources, buildings, receptors, years = set(), set(), set(), set()
for path in root.glob('*/task.json'):
    task = json.loads(path.read_text())
    sources.add(len(task['scenario']['units']))
    years.add(tuple(task['scenario']['locked_keywords']['meteorology_years']))
    buildings.add(len(json.loads((path.parent/'buildings.json').read_text())['buildings']))
    receptors.add(sum(line.strip().startswith('RE DISCCART') for line in
                      (path.parent/'receptors.inp').read_text().splitlines()))
print('sources:', sorted(sources), 'buildings:', sorted(buildings))
print('receptors:', receptors, 'years:', years)
PY
```

Heights are a finite menu: exact `h_min`, six interior seventh-points between
`h_min` and GEP rounded half-even to **0.5 m**, and exact GEP, with duplicates and
out-of-interval interior values removed. That gives at most **8 heights** per
source. Controls retain the none/moderate/maximum ladder. Menus make unit-field
reuse possible; they do not define a reward ceiling.

Menu constants and rounding are implemented once in
`envs/permit/core/permit_core/menu.py`; inspect with
`cat envs/permit/core/permit_core/menu.py`. Conversion is resumable, preserves
input hashes and records the original published NO2 tier separately from v2.
`uv run python convert/v1_to_v2.py --out envs/permit/tasks --check` validates
hashes and menus without changing task records.

## The task

Choose one compatible control and one listed height for every source. Heights
must be finite numeric values, within physical bounds and no higher than GEP.
The verifier runs BPIP-PRIME and independently enforces its GEP result. Every
standard must satisfy `modeled + background <= limit` at every receptor.
**Nothing is excused.** Native model failure or invalid output cannot pass.

Annualized cost, calculated with Decimal, is:

```text
sum(control annualized_usd)
  + sum(stack_usd_per_m * (height_m - height_min_m))
```

`none` has zero cost and zero control efficiency. The draft selects maximum
controls at GEP. Rules/cost sources:
`permit_core/{validate,cost,verify}.py`; inspect with
`cat envs/permit/core/permit_core/{cost,verify}.py`.

The four tools are `get_scenario`, free/unlimited `check_design`,
`evaluate_design` with **30 valid evaluations**, and terminal `submit_design`.
Invalid evaluations spend none; a valid evaluation spends its token even if
infrastructure subsequently fails. Exhaustion does not prevent submission.
Only the first submission produces a grade. These constants and transitions are
in `permit_openenv/environment.py`; reproduce with
`sed -n '20,220p' envs/permit/openenv/permit_openenv/environment.py`.

## NO2 and the fast evaluator

SO2 and primary PM2.5 are linear sums of per-source unit-rate fields for these
rural, flat-terrain `DFAULT CONC` point sources without deposition. This claim
is restricted to the bundled scenario options. Urban SO2 decay and other model
options are outside it. Plume rise and downwash depend on the source's own stack
and buildings. Each unit-field run keeps the full stack set for BPIP.

PVMRM combines plume chemistry across sources, so independent NO2 fields cannot
be added. V2 selects **ARM2 for every task**: sum hourly NOx fields first, then
apply ARM2 to the total at each receptor/hour. This changes the feasible set
relative to v1's mixed tiers and is why drafts are reverified.

With hourly NOx concentration `x` expressed as NO2 mass, ARM2 uses:

```text
p(x) = -1.1723e-17*x^6 + 4.2795e-14*x^5 - 5.8345e-11*x^4
       + 3.4555e-8*x^3 - 5.6062e-6*x^2 - 2.7383e-3*x + 1.2441
NO2(x) = x * clip(p(x), 0.50, 0.90)
```

The coefficients, source summation and ratio multiplication are in bundled
`aermod/calc1.f:10813–10858`; default clipping is in `coset.f:657–663`.
The renderer removes PVMRM-only ozone/in-stack-ratio cards and retains
`SRCGROUP ALL`. Source paths below are relative to
`envs/permit/core/native_build/model-source/`. Reproduce the formula and its
supporting physical assumptions with:

```sh
sed -n '10810,10860p;7003,7006p;7298,7312p' envs/permit/core/native_build/model-source/aermod/calc1.f
sed -n '302,312p;657,665p;1865,1871p' envs/permit/core/native_build/model-source/aermod/coset.f
sed -n '455,471p;1834,1844p' envs/permit/core/native_build/model-source/aermod/soset.f
sed -n '1615,1638p;2436,2460p;2952,2954p' envs/permit/core/native_build/model-source/aermod/calc2.f
sed -n '1413,1440p' envs/permit/core/native_build/model-source/bpipprm/Bpipprm.for
```

`permit_core/fields.py` obtains double-precision hourly unit fields using native
unformatted POSTFILE output, validates record markers/calendar/valid-hour flags,
and checks invalid-hour zeros. `permit_core/evaluate.py` scales each field by
the controlled rate rounded to **9 decimal places**, sums it, applies ARM2 to
NOx, and calculates the native annual/rank statistics. Horner evaluation changes
floating-point operation order; the fidelity gate measures that difference.

Short-term averaging uses `max(valid_hours, round(hours*0.75+0.4))`; annual means
use valid hours without that floor. Daily ranks and annual statistics are
calculated before averaging across years. Mean forms follow the native printed
**5-decimal F13.5** precision; max forms use Decimal half-even quantization, and
SO2 daily-rank means retain the verifier's Decimal exception. The exact forms
and limits are in [CONTRACT.md](CONTRACT.md#physical-grading-and-naaqs-forms), copied from task definitions.
Reproduce the rounding and denominator constants with
`cat envs/permit/core/permit_core/evaluate.py` and
`sed -n '80,150p' envs/permit/core/permit_core/{render,validate}.py`.

### Fidelity

The completed fidelity receipt is **GO**: **1,920/1,920 binding
standard values** match at five decimal places, and **240/240 feasibility
outcomes** agree. It covers **40 menu designs on each of 6 tasks**, spanning
all archetypes and a six-source case. This is sampled agreement, not a proof
for every possible menu design.

| Standard | Comparisons | Exact at five decimals | Max absolute error | Max relative error |
|---|---:|---:|---:|---:|
| NO2_1h | 240 | 240 | 0 | 0 |
| NO2_annual | 240 | 240 | 0 | 0 |
| PM25_24h | 240 | 240 | 0 | 0 |
| PM25_annual_primary | 240 | 240 | 0 | 0 |
| PM25_annual_secondary | 240 | 240 | 0 | 0 |
| SO2_1h | 240 | 240 | 0 | 0 |
| SO2_3h | 240 | 240 | 0 | 0 |
| SO2_annual | 240 | 240 | 0 | 0 |

| Task / sources | Median ms | Maximum ms | Float64 compressed bytes | Float32 changed values |
|---|---:|---:|---:|---:|
| `074897d0f159-s47` / 2 | 316.178 | 376.987 | 753791040 | 11 |
| `206fa4d0075d-s47` / 2 | 293.335 | 440.681 | 929276533 | 18 |
| `0558a6a6c25d-s47` / 2 | 314.652 | 353.746 | 774431087 | 19 |
| `040cd27878ef-s47` / 2 | 307.494 | 484.980 | 753618096 | 9 |
| `0dcaf34d070e-s47` / 2 | 298.390 | 356.443 | 768355702 | 9 |
| `01addc2a86f1-s47` / 6 | 536.838 | 572.024 | 2656512135 | 34 |

Source: [fidelity evidence](results/evidence/fidelity.json), including saved
designs and timing samples. Timings measure warmed, loaded-pack evaluation,
not cold field construction or HTTP latency. The revised latency limit is
**1 second** (`convert/fidelity.py::LATENCY_LIMIT_S`). Float32 changed **100 rounded values** in this sample, so runtime
packs retain float64. These compressed sizes describe the gate experiments;
current runtime storage uses full-receptor memory-mapped arrays.

Reprint the tables and totals without native work:

```sh
python3 envs/permit/openenv/space_root/report_evidence.py
rg -n LATENCY_LIMIT_S convert/fidelity.py
```

The original measurement command is recorded in the JSON's `command` field:
`uv run --project envs/permit/core python convert/fidelity.py --jobs 4`.
It is a heavy gate rerun, not part of documentation generation.

### On-demand cache

`permit_core.fields.ensure_pack(task, jobs=...)` returns a context-managed lease
over float64 `.npy` memory maps. Shared/exclusive locks serialize builders and
protect active packs; idle packs are evicted in LRU order. Input, meteorology,
code and native-binary identities invalidate stale fields. Active leases can
briefly exceed the configured cap. Unadopted legacy gate directories remain
pinned because the older gate workflow does not hold these leases.

The default cap is **40 GiB**, and interactive preparation defaults to **4 jobs**
(`field_cache.py::cache_limit`, `evaluator.py::FastEvaluator.evaluate`). Inspect
with `rg -n 'PERMIT_FIELDS|jobs' envs/permit/core/permit_core/field_cache.py envs/permit/openenv/permit_openenv/evaluator.py`.
Batch work is task-major so each pack can be reused across policies and evicted
when finished. Field packs are reproducible local cache data, never standalone
published dataset artifacts.

The conservative pruning certificate must preserve both feasibility and the
reported binding receptor/value. An hourly upper bound alone only proves that
a receptor cannot violate a standard; it need not preserve a feasible design's
maximum. Current on-demand packs keep the complete receptor network. Full
receptor values and margins go to saved episode records;
MCP results retain binding summaries. Schema: [docs/SCHEMAS.md](docs/SCHEMAS.md).

The Space recipe bakes **2 eval showcases**, `040cd27878ef-s47` (KTOP) and
`0558a6a6c25d-s47` (KABQ), using `ensure_pack`. It uses **8 GiB** for the idle
cache, **1 field job** and **1 server worker** at runtime; build preparation
uses at most **4 native processes**. These deployment constants are in
`space_root/bake_showcases.py` and the OpenEnv Dockerfile; reproduce with
`cat envs/permit/openenv/{Dockerfile,space_root/bake_showcases.py}`.
Showcases can later be evicted; their next evaluation then rebuilds them.

Other tasks also prepare fields on demand. Cold calls may take minutes on a CPU
Space; submission always runs queued native verification. Local batch evaluation
is preferred for sustained work. Historical container tool measurements are in
[the smoke receipt](results/evidence/space-container-smoke.json). A rebuild of the
current image remains TODO; the source-copy and native-stage contracts are tested.

## The reward

A feasible graded submission earns the fraction of draft cost saved:
`reward = (cost(draft) - cost(submission)) / cost(draft)`.
Prescore retains exact USD/year savings; gate failure uses sentinel −1 and
reward zero. The draft, invalid/infeasible submissions and no submission earn
zero. A score of 1.0 means zero annualized cost: the all-uncontrolled minimum-height
design, which fails a standard in the admitted tasks. It is therefore unreachable.
Every non-submission action has zero reward.

Saved fraction is directly interpretable and comparable across different plant
costs. It needs no model calibration, frozen anchors or train-split calibration
runs. Search and baselines remain diagnostics. The old hyperbolic helper and
anchor-writing command have been removed; no dependency was added.

For GRPO with per-task group normalization, dividing all rewards in a group by
the same positive draft cost leaves `(reward - group_mean) / group_std` unchanged
apart from numerical stabilizers. Thus per-task cost scale is irrelevant to the
normalized training advantage; within-task variation supplies the learning signal.
Zero-variance groups still provide no relative signal. Training uses fast grading;
the evaluation harness independently verifies submissions with native models.

Repeat selection and reporting are defined in [eval/REPEATS.md](eval/REPEATS.md).
Within each task, report mean, sample standard deviation, minimum, maximum,
feasible rate and mean absolute pairwise difference across repeats. Average task
means with equal task weight before station-clustered bootstrap. The secondary
fraction of best observed divides a task's mean savings fraction by its best
observed feasible model fraction in the reported cohort. Zero-best tasks have
no ratio and coverage is explicit; this comparison never changes the reward.

Reproduce the formula and subset selection without model/native computation:

```sh
cat envs/permit/core/permit_core/reward.py
uv run --no-sync python -m eval.select_tasks --report-only --output results/subsets.json
uv run --project envs/permit/core python -c 'from permit_core.reward import reward; print([reward(x, 100) for x in (-1, 0, 20, 100)])'
```

## Baselines and search

Public-tool policies are draft, all-none at GEP, uniform moderate at GEP,
cheapest-first greedy, random-30 and the lifted v1 coordinate descent. Random
uses **30 evaluations**; v1 coordinate descent uses **25**. Final proposed
designs receive full native verification. A failed policy is retained with an
infeasible/sentinel outcome. Source and limits:
`cat envs/permit/core/permit_core/baselines.py`.

Search uses cost bounds and conservative concentration lower bounds for partial
assignments. ARM2 monotonicity has an exact-rational certificate; source/option
ordering changes traversal, not soundness. Prefix sums, remaining-source minima
and certified-safe receptor checks accelerate bounds. Full evaluation and final
native verification retain every receptor.

`closed: true` means traversal completed with the implemented sound bounds;
`closed: false` means a time/node limit interrupted it. This flag does not replace
native feasibility. Results are always called **best-known search cost**, and
are not presented as an optimum. The library has no default limit. The batch
runner's provisional limits are **60 seconds / 10000 nodes**, checked between
operations; final native verification occurs outside the deadline.

Inspect these constants and closure semantics with
`rg -n 'closed|seconds|max_nodes|unlimited' envs/permit/core/permit_core/search.py convert/run_baselines.py`.

The recorded search measurements use a **1200-second search cap**, followed by native
verification. Times include that verification; costs are USD/year. Shared-host
measurements do not establish isolated machine performance.

| Sources / task | Run | Nodes | Evaluations | Seconds incl. verify | Closed | Best cost | Greedy | v1-CD |
|---|---|---:|---:|---:|---|---:|---:|---:|
| 2 / `074897d0f159-s47` | before | 97 | 12 | 85.128 | true | 115841.9976 | 115841.9976 | 115841.9976 |
| 2 / `074897d0f159-s47` | after | 97 | 12 | 55.357 | true | 115841.9976 | 115841.9976 | 115841.9976 |
| 4 / `05efc240e839-s47` | before | 6097 | 32 | 1304.520 | true | 724392.4354 | 838792.5540 | 1234016.7500 |
| 4 / `05efc240e839-s47` | after | 5185 | 31 | 473.960 | true | 724392.4354 | 838792.5540 | 1234016.7500 |
| 6 / `01addc2a86f1-s47` | before | 1536 | 34 | 1399.887 | false | 1884725.1576 | 2759891.3224 | 2937453.8472 |
| 6 / `01addc2a86f1-s47` | after | 12055 | 39 | 1382.049 | false | 1743547.5576 | 2759891.3224 | 2937453.8472 |

The four-source replacement `05efc240e839-s47` was prepared before either run
because the fidelity set had no four-source case. The six-source tree remains
open. Receipts and original rerun commands are in `results/evidence/search/`.
Reproduce the table without native work: `python3 convert/report_search.py`.

The cold full-pipeline projection was **8.099 hours on 12 cores**, compared with
the **6-hour** target; even removing search and its final check projected
**6.918 hours**. These are sample/interpolation estimates assuming perfect core
utilization, not a completed full run. Sources:
[pipeline-budget.json](results/evidence/pipeline-budget.json); command: `python3 convert/report_search.py`.
These timings are diagnostics, not model evaluation results.

### Admission

Admission requires a native-feasible ARM2 draft. All source tasks meet it;
search savings and naive-policy performance do not filter tasks.

| Final outcome | train | eval |
|---|---:|---:|
| Native ARM2 draft admission | 230 / 230 | 130 / 130 |
| Dropped tasks | 0 | 0 |

Reproduce from `eligible` in the task indices with
`python3 envs/permit/openenv/space_root/report_evidence.py`.
`convert/reverify_drafts.py --finalize` applies cached native checks;
`convert/admit.py` uses those same draft checks without model-dependent grading.
The release's model and baseline results remain **TODO**.

## How it is kept honest

The independent native submission path is authoritative. Tests compare lifted
functions and preserved v1 receptor statistics, exercise hostile designs, check
exact cost/reward behavior, validate fast statistics and native fidelity, and
compare bounded search with exhaustive synthetic cases. Cache tests cover
concurrent builders, leases, invalidation and eviction. Server tests cover schemas,
budget exhaustion, invalid-is-free, submit-once, rubrics and episode records.
The Space regression compares its native stage with `native_build/Dockerfile`
and rejects non-eval or non-float64 showcase packs. Release checks validate
source-copy paths, public-document links, file sizes and portable paths.

The [faithful-lift table](results/evidence/faithful-lift.md) and
[receipts](results/evidence/faithful-lift/) preserve binding
values, receptor comparison counts, source submissions and binary hashes.
Reprint those measurements with
`python3 envs/permit/openenv/space_root/report_evidence.py`.
Run all suites, including native tests, using the commands in the root README.


## What differs from v1

| Area | Published v1 | v2 |
|---|---|---|
| NO2 | Published PVMRM/ARM2 mixture | ARM2 for every task; drafts reverified |
| Heights | Continuous within native precision/bounds | Explicit menu |
| Interface | Agent-authored native files and CLI launches | Four tools; environment renders native inputs |
| Evaluation budget | 75 launches, equivalent to 25 complete pollutant evaluations | 30 valid evaluations, invalid checks free |
| Reward | Raw USD saved, gate-failure floor −1 | Same savings prescore, reward = saved fraction; gate failure zero |
| Evaluation | Native physics calls | Cached float64 fields; full native submission |

Budget and historical score sources: `convert/v1-src/family/bind.py`,
`convert/v1-src/shared/tests/grade.py` and `permit_openenv/environment.py`.
Reproduce with `rg -n 'query_tokens|floor|EVALUATE_BUDGET' convert/v1-src/family/bind.py convert/v1-src/shared/tests/grade.py envs/permit/openenv/permit_openenv/environment.py`.
The legacy index preserves published tiers
and continuous-height semantics for comparisons; v1 and v2 score scales are
never combined. Flat terrain, a single control ladder per source, a common
stack price within each facility and synthetic industrial inventories remain
limitations. Chemistry/menu changes are disclosed, not excused.

## Native build and pins

`envs/permit/core/native_build/Dockerfile` builds the v1 verifier binaries from
bundled **AERMOD 26135** and **BPIPPRM 04274** sources. Meteorology was processed
with **AERMET 26135**; AERMET is not rebuilt in the Space. Source inventories,
NOAA/EPA acquisition hashes and provenance are in SOURCES.md. Inspect version
headers with:

```sh
sed -n '1,24p' envs/permit/core/native_build/model-source/aermod/aermod.f
sed -n '1,16p' envs/permit/core/native_build/model-source/bpipprm/Bpipprm.for
rg -n 'AERMET|26135' SOURCES.md
```

The native stage pins Debian `bookworm-20250224-slim` at digest
`sha256:12c396bd585df7ec21d5679bb6a83d4878bc4415ce926c9e5ea6426d23c60bdc`
and snapshot `20250224T000000Z`. It applies `instrument.py` for annual/rank
statistics and does not use `screening_patch.py`. AERMOD compiles with
`-O2 -static -fbounds-check -Wuninitialized`; BPIP uses
`-O2 -static -std=legacy`, removing the DOS SUB byte only in compiler scratch
input. The exact source order and linking flags come from `build-models.sh`.
This is the verifier's pinned build, not a reconstruction from generator prose.

```sh
cat envs/permit/core/native_build/{Dockerfile,build-models.sh}
uv run --directory envs/permit/openenv pytest tests/test_space_container.py -q
sh envs/permit/core/native_build/build.sh
sha256sum -c envs/permit/core/native_build/bin/SHA256SUMS
```

The Space uses that identical stage with repository-root COPY paths, copies the
binaries into a separate runtime stage, installs both local Python packages with
the OpenEnv frozen lock, copies converted task/met inputs and bakes showcases
as a non-root user. It serves **port 7860**. An allowlisted `.dockerignore`
excludes host archives, caches, environments, credentials and native binaries.
`PERMIT_V1_DIR` points at an empty marker directory in the image solely to locate
the sibling field cache; converted tasks supply every runtime physical input.
The local Python packages are installed in editable mode. The build recipe is
`cat envs/permit/openenv/{Dockerfile,openenv.yaml,space_root/README.md}`.

Dependency choices are unchanged: Python **>=3.11**; PyYAML **6.0.3** for the
faithful StrictLoader; NumPy **2.3.4** for fields/statistics; OpenEnv **0.7.0**,
FastMCP **>=3,<4**, Gradio **>=5** and Uvicorn **>=0.30** for the existing server.
Setuptools **80.9.0** builds packages; pytest **8.4.2** and Ruff **0.14.1** are
development tools. The container bootstraps existing uv tooling at **0.11.32**
and resolves no dependencies outside the frozen lock during sync.  Reproduce all declared pins with
`cat pyproject.toml envs/permit/{core,openenv}/pyproject.toml`; deployment uv pin:
`rg -n 'uv==' envs/permit/openenv/Dockerfile`. Transitive pins are in the package
`uv.lock` files.
