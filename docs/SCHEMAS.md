# Shared data contracts (v2)

These JSON shapes define the tool interface and saved analysis records. All numbers are JSON numbers (costs in USD/year, lengths in metres, concentrations in µg/m³).
Examples illustrate shapes, not complete tasks; use the actual menus and values returned by the tools.

## Design
`{"S1": {"control": "S1_moderate", "height_m": 45.0}, ...}` — every source id exactly once; `control` is a control id
whose `applies_to` contains the source (or `"none"` meaning zero efficiency, zero cost); `height_m` is a value from that
source's `height_menu_m`.

## Scenario view — `get_scenario()` result
```json
{"task_id": "003bd328a392-s47", "split": "train", "station": "KMSP", "years": [2020,2021,2022,2023,2024],
 "sources": [{"id": "S1", "type": "simple_cycle_turbine", "x_m": -55.35, "y_m": -82.87,
              "uncontrolled_g_s": {"NO2": 94.14, "PM25": 0.356, "SO2": 0.349},
              "height_min_m": 14.72, "gep_m": 82.27, "height_menu_m": [14.72, 26.0, 37.0, 48.5, 59.5, 71.0, 82.27]}],
 "controls": [{"id": "S1_moderate", "applies_to": ["S1"], "annualized_usd": 260357.61,
               "efficiency": {"NO2": 0.7125, "PM25": 0.74625, "SO2": 0.735}}],
 "stack_usd_per_m": 5390.47,
 "standards": [{"id": "NO2_1h", "pollutant": "NO2", "form": "5-year mean of the yearly 8th-highest daily max 1-hour",
                "limit": 188.16155, "background": 47.534383}],
 "draft_design": {"S1": {"control": "S1_maximum", "height_m": 82.27}},
 "draft_cost": 3123456.78, "evaluations_left": 30}
```

## Evaluation — `evaluate_design()` result (and `submit_design()` adds the grade fields)
```json
{"valid": true, "errors": [], "feasible": false, "cost": 1234567.89, "evaluations_left": 29,
 "standards": [{"id": "SO2_1h", "value": 151.2, "background": 25.6, "limit": 196.5, "margin": 19.7,
                "binding_receptor": {"index": 17, "x_m": 250.0, "y_m": -100.0}}]}
```
`submit_design()` = the same shape (verified by full AERMOD) plus
`{"reward": 0.26, "prescore_usd_saved": 26.0, "pct_saved": 26.0}` for an illustrative draft cost of 100.
`reward` is feasible USD saved divided by draft cost; failed/absent submissions score zero.
There is no `anchors` field. Task records also omit `anchors`, `anchor_source`,
`anchors_frozen_at` and `anchor_runs`. Historical records may retain old fields;
reporting recomputes saved fraction from the recorded savings and draft cost.
Invalid designs: `{"valid": false, "errors": ["E_MENU: use every source exactly once with a compatible control and listed height"], "evaluations_left": <unchanged>}`.
The first fast evaluation of a task may also return a `note` explaining on-demand field preparation.
Subsequent evaluations reuse the disk cache. Field preparation spends no additional evaluation tokens.

## Geometry — per-task analysis data
Geometry is private record data; reset and the four public tools never return it.
```json
{"task_id": "...", "station": "KMSP",
 "buildings": [{"id": "B1", "corners": [[x,y],[x,y],[x,y],[x,y]], "height_m": 32.91}],
 "receptors": [[x_m, y_m], ...],
 "stacks": [{"id": "S1", "x_m": -55.35, "y_m": -82.87, "height_min_m": 14.72, "gep_m": 82.27, "diameter_m": 2.06}],
 "wind_rose": {"sectors": 16, "speed_bins_m_s": [0.5, 2, 4, 6, 8, 10], "counts": [[...16 sectors × 6 bins...]], "calm_hours": 123}}
```

## Episode record — `results/rollouts/<run>/<model_slug>/<task_id>/r<k>/episode.json`
```json
{"schema_version": 1, "repeat": 1, "task_id": "...", "split": "eval", "station": "KTOP", "model": "anthropic:claude-sonnet-5-5",
 "prompt_version": 2, "started_at": "ISO-8601", "finished_at": "ISO-8601 or null while active",
 "scenario": {<scenario view>}, "geometry": {<geometry>},
 "steps": [{"i": 0, "tool": "get_scenario|check_design|evaluate_design|submit_design", "args": {...}, "result": {...},
            "receptor_values": {"NO2_1h": [<one number per geometry.receptors entry>], "...": []},
            "receptor_margins": {"NO2_1h": [<one number per geometry.receptors entry>], "...": []},
            "t_s": 1.2, "assistant_text": "optional, truncated to 2,000 chars"}],
 "evaluations_used": 7, "ended": "submitted|no_submit|turn_limit|error",
 "final": {<submit_design result> | null},
 "usage": {"input_tokens": 0, "output_tokens": 0, "usd": null}}
```

Successful evaluate/submit steps carry `receptor_values` and `receptor_margins` alongside `result`,
for every standard and every receptor, in `geometry.receptors` order (184 entries for existing tasks).
Values are modeled concentrations; margins are `limit − background − value`, using the same rounding
as the binding values. These arrays are record-only: MCP tool results, state history, and the public
final grade exclude them. Invalid designs and infrastructure failures have no computed arrays.
Legacy records may omit arrays; unrecorded values must not be inferred. Native binding-receptor indices are one-based.
Live retention is bounded and records `steps_dropped` and `diagnostics_dropped`.
The eval harness also writes full `transcript.jsonl`, provider `raw/`, private `env.log`
and failed `attempts/`; see `eval/PROTOCOL.md`. The repeat-local `r<k>.json` is a compatibility snapshot. Older `<task_id>/episode.json`
and flat `<task_id>.json` records count as repeat 1; readers prefer canonical records.

Repeat identities are `(model, task_id, repeat)`, with positive one-based integers.
Run manifests record the requested repeats and the immutable protocol hash.
Import requires the same protocol hash, prompt version/hash and sampling settings.
Reports expose per-task repeat statistics in `variance.json` and `variance.csv`,
and task-weighted model metrics in the board. See [eval/REPEATS.md](../eval/REPEATS.md).

Wind speed bins are lower edges: `[0.5,2)`, `[2,4)`, `[4,6)`, `[6,8)`, `[8,10)`, `[10,∞)` m/s;
valid speeds below 0.5 m/s count as calm, and missing observations are excluded.
