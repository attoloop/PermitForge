# PermitForge v2 on OpenEnv

Choose a control and a stack height for each plant source, minimize annualized cost,
and meet every air-quality standard at every receptor. This package exposes the
shared contract in `docs/SCHEMAS.md` through OpenEnv. Domain validation, cost,
physical verification and reward remain in `permit_core`.

From this directory:

```sh
uv sync --python 3.12
uv run uvicorn permit_openenv.server:app --port 8012
```

`/healthz` reports task counts. `/web/` serves the standard OpenEnv Gradio
playground with the task guide. Episode records retain geometry, wind counts,
and per-receptor values and margins for analysis.
The shared playground is a public demo, not a scored evaluation endpoint.
The eval harness forces `ENABLE_WEB_INTERFACE=false`; use `eval/run_eval.py` for
scored runs and complete private capture. Reset returns only the public scenario;
geometry, wind counts and receptor arrays stay in the private episode record.

Connect an agent to the running local server (no model API required):

```python
from openenv.core.mcp_client import MCPToolClient
from openenv.core.env_server.mcp_types import CallToolAction

with MCPToolClient("http://127.0.0.1:8012", message_timeout_s=1800).sync() as env:
    observation = env.reset(task_id="003bd328a392-s47")
    print(observation.observation.metadata["instructions"])
    step = env.step(CallToolAction(tool_name="get_scenario", arguments={}))
    print(step.observation.result)
```

| Tool | Effect |
|---|---|
| `get_scenario()` | Sources, menus, standards, draft and evaluations remaining |
| `check_design(design)` | Free validation and exact annualized cost |
| `evaluate_design(design)` | Air-quality evaluation; a valid call spends one evaluation |
| `submit_design(design)` | Ends the episode and grades once with full ARM2 BPIP/AERMOD verification |

The evaluation budget is 30. Invalid designs are free; exhausted calls return an
error without spending. Submission remains available after exhaustion and does
not consume an evaluation. Any submission, including an invalid one, is terminal.
Only the first submission produces a reward; subsequent calls cannot regrade.
Use `{ "S1": { "control": "S1_moderate", "height_m": 45.0 }, ... }` with the actual
menus returned for the task. Public costs are JSON numbers; core cost arithmetic
and savings are Decimal. Rubric children are `feasible` and `cost_quality`.

The reward calls `permit_core.reward` with exact savings and draft cost, returning
the saved fraction. Invalid or infeasible submissions have prescore -1 and reward
zero; a feasible unchanged draft has prescore and reward zero. No anchors are
returned or required. USD/year savings and `pct_saved` remain available.

Run the complete suite, including the native smoke (no key or model):

```sh
uv run pytest -q
uv run python rollout.py --smoke \
  --task-id 003bd328a392-s47 --output ../../../results/rollouts/smoke.json
uv run ruff check permit_openenv tests rollout.py space_root
```

The smoke uses the default FastEvaluator for evaluation and VerifyEvaluator for
submission. Missing field packs are computed on demand. It prints
`get_scenario → check_design → evaluate_design → submit_design`
and writes the shared Episode record, including geometry and the wind rose.
Cold field preparation and native submission may take minutes. Each verifier runs its cases
sequentially, and the server serializes native verifications across sessions.
Use one server worker for live episode state. Integrated field and verification
calls share process-safe native slots; independent legacy jobs still require
coordinating the repository's native-process limit. The tests execute no model API.

`agent.run_episode(client, model, max_turns=40)` implements an Anthropic tool-use
demo loop, records token usage and truncated assistant text, and handles missing
submission, turn limit and errors. Run it only after authorizing API spend:
Its public-client record has no geometry; complete private records come from the
eval harness's authenticated recording sidecar.

```sh
uv sync --extra agents
uv run python rollout.py --model anthropic:MODEL --base-url http://127.0.0.1:8012 \
  --task-id 003bd328a392-s47 --output ../../../results/rollouts/model-episode.json
```

Configuration:

- `PERMIT_TASKS_DIR`: converted tasks root (default `../tasks`). Tasks explicitly
  marked ineligible are excluded; pending records retain their admission status
  in the Task API. If converted manifests are absent, load the preserved
  v1 extraction through `permit_core.model.load_v1_instance` and its manifest splits.
- `PERMIT_V1_DIR`: core's extraction-root override.
- `PERMIT_EVALUATOR=fast|verify`: selects evaluation, never final grading. When
  unset, fast is selected if its modules import; otherwise verification is selected.
  Explicit `fast` without those modules fails with a clear setup error.
- `PERMIT_FIELDS_DIR`: optional directory containing full-receptor `<task_id>.npz`
  packs. By default fast evaluation uses on-demand float64 packs under
  `convert/cache/fields/`. The first call per task includes a cold-cache note.
- `PERMIT_FIELDS_JOBS`: field preparation parallelism (default 4).
- `PERMIT_FIELDS_CACHE_GB`: LRU size cap (default 40 GiB). Active leases and legacy
  gate directories can exceed it; idle managed packs are evicted on release.
- `PERMIT_MODELS_DIR`: core's native-binary override.
- `ENABLE_WEB_INTERFACE=false`: disables Gradio; static `/web` remains available.
- `MAX_CONCURRENT_ENVS`: session limit (default 4, at most 16).

`openenv.yaml` and `space_root/` follow the PortSim structure. The Dockerfile
uses the repository root as its build context and compiles the native binaries
with the core's pinned build recipe. Converted task data and met files must be
present. The same image runs the public Space.

The wind rose uses north-centered clockwise sectors, lower-bound speed bins
`[0.5, 2, 4, 6, 8, 10]` m/s with an open final bin, and calm below 0.5 m/s. Missing
wind records are omitted. Geometry retains the published receptor order and
building corners; binding receptor indices follow core's one-based convention.
Episode steps retain every receptor's values and margins beside the compact
tool result. These arrays are excluded from MCP results and agent state; the
saved records support analysis across the full receptor network.

Package implementation notes and dependencies: [DESIGN.md](DESIGN.md).
Measurement receipts: [results/evidence](../../../results/evidence/).
