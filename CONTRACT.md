# PermitForge v2 agent contract

Choose a control and a stack height for every emission source. Minimize annualized
control and stack-extension cost while meeting every task standard at every
receptor. The facility is synthetic; the weather is NOAA meteorology. The returned
concentrations are NAAQS design values across the task years, not an instantaneous
plume. The canonical wire shapes are [docs/SCHEMAS.md](docs/SCHEMAS.md); implementation
is in `envs/permit/openenv/permit_openenv/` and physical grading in `permit_core`.

## Episode and tools

Connect to the local OpenEnv server as shown in [README.md](README.md#connect-an-agent).
`reset(task_id=...)` selects a task, or use `reset(split="train", index=0)`.
Reset metadata includes `instructions`, scenario data, admission status and the
evaluator name. A new reset starts a new budget. Tasks explicitly marked ineligible
are excluded; pending admission is exposed in metadata, not treated as completed
admission. Task discovery is available through `list_splits`, `num_tasks`,
`get_task` and `list_tasks` on the environment.

| Tool | Arguments | Effect |
|---|---|---|
| `get_scenario()` | `{}` | Read task data and remaining evaluations; free |
| `check_design(design)` | `{"design": DESIGN}` | Validate the design and compute its exact cost; free and unlimited |
| `evaluate_design(design)` | `{"design": DESIGN}` | Evaluate air quality; a valid design spends one evaluation |
| `submit_design(design)` | `{"design": DESIGN}` | End the episode; independently verify and grade once |

The episode budget is **30 evaluations**. Invalid designs spend none. A valid
evaluation spends its token before computation, including if the evaluator fails.
Exhaustion returns an error without additional spend. Submission remains available
after exhaustion and does not consume an evaluation. Every submission is terminal,
including an invalid one. Calls after termination return an episode-over error;
they cannot change the submission or regrade. Only submission produces a grade;
other actions and an episode ending without submission earn zero reward.

Reproduce the budget, terminal behavior and reset semantics:

```bash
nl -ba envs/permit/openenv/permit_openenv/environment.py | sed -n '28,225p'
```

## Design validity and cost

The public design is a JSON object keyed by source ID. This is a shape example;
use the actual source IDs, controls and menu values returned by `get_scenario`,
and include every source rather than copying this single row:

```json
{"S1": {"control": "S1_moderate", "height_m": 45.0}}
```

- Include every source exactly once and no unknown source. Each row has exactly
  `control` and `height_m`; extra or missing fields are invalid.
- `control` is a returned control ID whose `applies_to` contains that source, or
  the literal `"none"` for zero cost and zero efficiency. A returned source-specific
  none row is also allowed.
- `height_m` is a finite JSON number from that source's `height_menu_m`, within
  its physical bounds. Booleans, strings, NaN and infinity are invalid numbers.
  Heights must be representable at two decimal places for native rendering.
- The native grade recomputes BPIP GEP and rejects any stack above it. All other
  plant data, receptors, weather, stack diameter, temperature and exit velocity
  remain fixed.

The published menu contains the minimum, six interior divisions of the span to
GEP (sevenths, rounded to the nearest half metre with half-even ties), and GEP,
then bounds filtering, sorting and deduplication. Use the returned menu, rather
than reconstructing it from these rules.

Annualized cost in USD/year is
`sum(control.annualized_usd + stack_usd_per_m * (height_m - height_min_m))` over
sources. Core arithmetic uses Decimal; JSON exposes numeric values. Cost checking
alone establishes no air-quality feasibility.

Reproduce the validation, menu and cost rules:

```bash
nl -ba docs/SCHEMAS.md | sed -n '1,42p'
nl -ba envs/permit/openenv/permit_openenv/tasks.py | sed -n '21,110p'
nl -ba envs/permit/core/permit_core/validate.py | sed -n '8,31p'
nl -ba envs/permit/core/permit_core/native.py | sed -n '30,37p'
nl -ba convert/v1_to_v2.py | sed -n '1,100p'
nl -ba envs/permit/core/permit_core/cost.py | sed -n '5,15p'
```

## JSON results

All lengths are metres, emissions g/s, concentrations µg/m³ and costs USD/year.
The following numeric examples describe field shapes, not measured results or a
complete real task. Reproduce the examples and authoritative field names with
`cat docs/SCHEMAS.md`; the scenario exporter also returns `no2_in_stack_ratio`.
`cost` is the wire key, rather than `cost_usd_per_year`.

`get_scenario()` returns the task identity and split, station, years, sources,
controls, stack price, standards, draft design, draft cost and remaining evaluations:

```json
{"task_id": "003bd328a392-s47", "split": "train", "station": "KMSP", "years": [2020,2021,2022,2023,2024],
 "sources": [{"id": "S1", "type": "simple_cycle_turbine", "x_m": -55.35, "y_m": -82.87,
              "uncontrolled_g_s": {"NO2": 94.14, "PM25": 0.356, "SO2": 0.349},
              "height_min_m": 14.72, "gep_m": 82.27, "height_menu_m": [14.72,26.0,37.0,48.5,59.5,71.0,82.27]}],
 "controls": [{"id": "S1_moderate", "applies_to": ["S1"], "annualized_usd": 260357.61,
               "efficiency": {"NO2": 0.7125, "PM25": 0.74625, "SO2": 0.735}}],
 "stack_usd_per_m": 5390.47,
 "standards": [{"id": "NO2_1h", "pollutant": "NO2", "form": "5-year mean of the yearly 8th-highest daily max 1-hour",
                "limit": 188.16155, "background": 47.534383}],
 "draft_design": {"S1": {"control": "S1_maximum", "height_m": 82.27}},
 "draft_cost": 3123456.78, "evaluations_left": 30}
```

`form` describes the statistic; the implementation can return the preserved task's
machine-readable form string. Actual `limit` and `background` values are taken
from the task at full stored precision; the example is not a rounded grading limit.

`check_design()` returns `{valid, errors, cost, evaluations_left}` for a valid
design. An invalid design omits `cost` and returns errors with the unchanged budget:

```json
{"valid": false, "errors": ["S3: height 50.0 not on menu"], "evaluations_left": 30}
```

`evaluate_design()` returns:

```json
{"valid": true, "errors": [], "feasible": false, "cost": 1234567.89, "evaluations_left": 29,
 "standards": [{"id": "SO2_1h", "value": 151.2, "background": 25.6, "limit": 196.5, "margin": 19.7,
                "binding_receptor": {"index": 17, "x_m": 250.0, "y_m": -100.0}}]}
```

`margin = limit - value - background`. For each standard the binding receptor is
the receptor with the smallest margin; its index is one-based in the preserved
receptor order. All receptors are checked, including those not selected for the
summary. The example's displayed standard has positive margin; a different standard
can still make the design infeasible. Error responses use `errors`, and an `error`
message for exhaustion, failed evaluation or calls after termination. A failed
evaluation may have an empty `standards` list.

`submit_design()` returns the evaluation fields from full verification plus
`reward`, `prescore_usd_saved`, `pct_saved`, `anchors` and
`best_known_search_cost` (nullable when no search result is recorded). For example,
an unchanged feasible draft without calibration adds:

```json
{"reward": 0.0, "prescore_usd_saved": 0, "pct_saved": 0.0,
 "anchors": {"worst": 0, "best": null, "saturation": null, "source": "none"},
 "best_known_search_cost": null}
```

With calibration, `anchors` holds numeric `worst`, `best`, `saturation` and a
`source` identifying model calibration, such as `"panel"`. Invalid submission still returns terminal
grade fields with `valid: false`, `feasible: false` and no standards. Errors may be
validation strings or native `{code, detail}` records.

Reproduce export fields, binding-receptor selection and submission additions:

```bash
nl -ba envs/permit/openenv/permit_openenv/tasks.py | sed -n '66,110p'
nl -ba envs/permit/openenv/permit_openenv/evaluator.py | sed -n '19,37p'
nl -ba envs/permit/openenv/permit_openenv/environment.py | sed -n '138,198p'
nl -ba envs/permit/core/permit_core/verify.py | sed -n '105,131p'
```

## Physical grading and NAAQS forms

Submission validates the menu and cost, renders native inputs, runs BPIP-PRIME,
recomputes GEP/downwash, then runs full AERMOD cases for NO2, PM25 and SO2. NO2
uses **ARM2 for every v2 task**. Native success and complete valid output are
required. Every task standard must satisfy `modeled + background <= limit` at
every receptor. There is no tolerance added to that Decimal comparison.
Native failures and malformed output fail the grade. Nothing is excused.
Submission may take minutes; use a local server for repeated native evaluations.

These are fixed benchmark forms, over the task's five calendar years. Limits
come from each task; ppb conversions below use 24.45 L/mol. This table describes
the benchmark, not advice on current permit law.

| Standard | Limit (µg/m³; expression preserves the conversion) | Per-receptor statistic, then checked at every receptor |
|---|---|---|
| `NO2_1h` | `100 * 46.0055 / 24.45` | Mean of yearly 8th-highest daily maximum 1-hour values |
| `NO2_annual` | `53 * 46.0055 / 24.45` | Maximum over years of the annual mean |
| `PM25_24h` | `35` | Mean of yearly 8th-highest daily mean 24-hour values |
| `PM25_annual_primary` | `9` | Mean of yearly annual means |
| `PM25_annual_secondary` | `15` | Mean of yearly annual means |
| `SO2_1h` | `75 * 64.066 / 24.45` | Mean of yearly 4th-highest daily maximum 1-hour values |
| `SO2_3h` | `500 * 64.066 / 24.45` | Maximum over years of the yearly 2nd-highest 3-hour block |
| `SO2_annual` | `10 * 64.066 / 24.45` | Mean of yearly annual means |

The native model handles calm/missing hours; short-term averages use a denominator
of `max(valid_hours, NINT(0.75 * period_hours + 0.4))`. Periods with no valid hours
are missing. The grade reads the native statistics and plot files rather than
averaging the tool's already-aggregated standard values.

Precision is part of the grade: emissions are quantized to nine decimal places
with `ROUND_HALF_EVEN`; heights are rendered to two decimals. Plot-derived
multi-year means use AERMOD's printed `F13.5` concentrations. Max-over-years forms
are Decimal half-even quantized to five decimal places. SO2 1-hour is a special
case: the mean of the collected yearly daily ranks is also Decimal half-even
quantized to five places. Backgrounds and limits keep their task precision, and
the final margin comparison uses Decimal at precision 28.

`evaluate_design` uses the configured evaluator, while submission always uses
full verification. `PERMIT_EVALUATOR=verify` selects full native evaluation and
works without field packs. Fast evaluation prepares missing packs on demand; its fidelity evidence is in
[DESIGN.md](DESIGN.md). Module availability alone does not establish fidelity. This contract makes no fast timing or accuracy claim.

Reproduce the forms, conversion constants, rounding and physical grade:

```bash
nl -ba results/evidence/provenance/scenario-generator.py.txt | sed -n '35,49p'
nl -ba envs/permit/core/permit_core/validate.py | sed -n '102,161p'
nl -ba envs/permit/core/permit_core/render.py | sed -n '19,46p;85,105p'
nl -ba envs/permit/core/permit_core/verify.py | sed -n '58,145p'
rg -n '0.75|F13.5|NINT' envs/permit/core/native_build/model-source/aermod/{calc2,output}.f
```

## Prescore and reward

For a feasible submission, `prescore_usd_saved = cost(draft) - cost(submitted)`
in USD/year. An invalid, infeasible or absent submission has prescore sentinel
−1 and reward zero; a feasible unchanged draft has prescore zero and reward zero.
`pct_saved` reports `100 * prescore / draft_cost` for feasible submissions and
zero for a failed submission. Best-known search cost is contextual, never a
claim of a proven optimum.

For prescore `x`, per-task anchors `w = worst = 0`, `b = best > w`, and
`s = saturation = b + 3 * (b - w)`, the hyperbolic calibration is:

```text
x <= w:       0
w < x < b:    0.4 * (x - w) / (b - w)
x >= b:       0.4 + 1.2 * (x - b) / (s - b + 2 * (x - b))
```

The best anchor scores 0.4, saturation scores 0.8, and reward tends toward 1.0 as
savings grow; the formula has no attainable 1.0 target. Later improvements beyond
the anchor still earn more. Core maps missing/non-finite prescores to zero and
clips the numeric result to [0,1]. Missing anchors are exposed as null `best` and
`saturation` with `source: "none"`: all rewards are then zero, while measured USD
saved remains available. No unmeasured anchor is filled in.

Both train and eval anchors come only from model calibration runs on each task.
`best` is the highest feasible model savings; search and baselines never supply
anchors. TODO: run calibration on every task, rescore and freeze anchors, and
publish model results. A task without a positive model-calibration result remains
uncalibrated with `source: "none"`; there is no search fallback.

Reproduce the formula, sentinel and anchor handling:

```bash
nl -ba envs/permit/core/permit_core/reward.py
nl -ba envs/permit/openenv/permit_openenv/tasks.py | sed -n '66,76p'
nl -ba envs/permit/openenv/permit_openenv/environment.py | sed -n '178,197p'
```

## What differs from v1

| Aspect | Published v1 | v2 |
|---|---|---|
| Interface | Agent writes design YAML and native input files, invokes CLI shims | Four MCP tools; environment renders files |
| NO2 | Published mix of PVMRM and ARM2 | ARM2 throughout; feasible sets can differ and draft admission is rechecked |
| Height choice | Continuous within physical/GEP bounds | Returned discrete menu |
| Evaluation budget | 75 native launches, equivalent to 25 three-pollutant evaluations | 30 evaluations; checks free; submission separate |
| Score | Raw USD/year saved; failure floor −1 | USD savings prescore plus per-task hyperbolic reward; failed grade and draft reward zero |

Facilities, costs, backgrounds, weather and receptors are reused. Limitations:
one control ladder per source, one stack price per facility, flat terrain,
synthetic facilities, a fixed station set and archetypes. The task menu and ARM2
change are disclosed rather than described as a faithful reproduction of v1's
feasible set. Legacy and v2 scores must be kept separate.

Reproduce the conversion rules with `cat convert/v1_to_v2.py`; see [SOURCES.md](SOURCES.md) for provenance and draw ranges.
