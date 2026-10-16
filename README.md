# PermitForge

Choose emissions controls and stack heights for a synthetic industrial plant,
using real NOAA weather and EPA dispersion models. An OpenEnv environment for
cutting annualized cost while meeting every task air-quality standard.

`Space: recipe prepared` · `Eval: TODO` · `Dataset: TODO` · `Framework: OpenEnv`

## The task

Choose a compatible control and one menu height for every source. Meet each NO2,
SO2 and PM2.5 standard at every receptor, including background concentrations,
across the task's weather years. Taller stacks and stronger controls cost more.
The draft uses maximum controls at GEP height.

| Tool | What it does |
|---|---|
| `get_scenario()` | Sources, controls, menus, standards, draft and remaining evaluations |
| `check_design(design)` | Validity and exact annualized cost; free and unlimited |
| `evaluate_design(design)` | Air-quality values, margins and binding receptors; 30 valid evaluations per episode |
| `submit_design(design)` | Ends the episode and grades once with full BPIP/AERMOD verification |

Reward calibrates USD/year saved against the draft: the draft, infeasible designs
and no submission score zero. Each task's best feasible model-calibration result
anchors 0.4, with further savings approaching 1.0; missing calibration yields
zero reward while preserving measured savings.

All **360 tasks are admitted: 230 train and 130 eval**, separated by weather
station, with every draft verified feasible under ARM2. V2 uses ARM2 for all NO2
cases and discrete heights, so its feasible set differs from published v1.
Search and baselines are diagnostics, independent of admission and calibration.

Reproduce inventory and admission counts from the task records and
[conversion evidence](results/evidence/conversion.json):

```sh
python3 envs/permit/openenv/space_root/report_evidence.py
rg -n 'EVALUATE_BUDGET|evaluations_left' envs/permit/openenv/permit_openenv/environment.py
```

## Results

**Pending — TODO.** Model-calibration runs on both splits, frozen anchors,
model scores, confidence intervals and calibrated baselines are pending.
Measured physics agreement is in [DESIGN.md](DESIGN.md#fidelity).

## Start here

| You want to… | Open |
|---|---|
| Try the standard OpenEnv playground | Local [/web/](http://127.0.0.1:8012/web/) after starting the server |
| Connect an agent | Client snippet below; MCP endpoint `/mcp` |
| Read the rules and reward | [CONTRACT.md](CONTRACT.md) |
| Understand physics, fidelity and search | [DESIGN.md](DESIGN.md) |
| Read the data contract | [docs/SCHEMAS.md](docs/SCHEMAS.md) |
| Trace inputs and licences | [SOURCES.md](SOURCES.md) |
| Prepare local data | [convert/README.md](convert/README.md) |
| Build the Space image | [Space card](envs/permit/openenv/space_root/README.md) |

## Connect an agent

Start the server below, then run this with `uv run python` from
`envs/permit/openenv`. It reads and validates the actual draft without calling a
model API. `space_root/smoke.py` exercises this snippet against a live server.

```python
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.mcp_client import MCPToolClient
from permit_openenv.protocol import tool_result

with MCPToolClient("http://127.0.0.1:8012", message_timeout_s=1800).sync() as env:
    obs = env.reset(task_id="040cd27878ef-s47")  # or reset(split="train", index=0)
    print(obs.observation.metadata["instructions"])
    scenario = tool_result(env.step(CallToolAction(tool_name="get_scenario", arguments={})).observation)
    design = scenario["draft_design"]
    checked = env.step(CallToolAction(tool_name="check_design", arguments={"design": design}))
    print(tool_result(checked.observation))
```

Use the same `CallToolAction` shape for `evaluate_design` and `submit_design`,
with `arguments={"design": design}`. Keep the client connected during native
submission, which may take minutes.

## Run it locally

A code-only checkout excludes task/met data, extracted v1 inputs and binaries.
Prepare the data using [convert/README.md](convert/README.md). Core fidelity tests
also need the preserved `convert/v1-src/` evidence. Native binaries live in
`envs/permit/core/native_build/bin/`, or in `PERMIT_MODELS_DIR`.

From the repository root:

```sh
sh envs/permit/core/native_build/build.sh  # requires working podman
uv sync --directory envs/permit/openenv --python 3.12 --frozen
uv run --directory envs/permit/openenv uvicorn permit_openenv.server:app \
  --host 127.0.0.1 --port 8012 --workers 1 --ws-ping-timeout 1800
```

The standard OpenEnv Gradio playground is at **[/web/](http://127.0.0.1:8012/web/)**;
**[/healthz](http://127.0.0.1:8012/healthz)** reports health and task counts.
Float64 fields are cached on demand: an uncached task may take minutes to prepare.
`PERMIT_FIELDS_JOBS` controls preparation parallelism and `PERMIT_FIELDS_CACHE_GB`
caps idle storage. `PERMIT_EVALUATOR=verify` selects full native evaluation.
Submission always uses native verification. Use a local server for heavy runs.

Run the model-free smoke and all suites from the repository root:

```sh
uv run --directory envs/permit/openenv python rollout.py --smoke \
  --task-id 040cd27878ef-s47 --output ../../../results/rollouts/smoke.json
uv run pytest -q
uv run --directory envs/permit/core pytest -q
uv run --directory envs/permit/openenv pytest -q
uv run python convert/v1_to_v2.py --out envs/permit/tasks --check
```

Build the Space image with the converted task/met inputs present:

```sh
podman build -f envs/permit/openenv/Dockerfile -t permitforge-space .
podman run --rm --name permitforge-space -p 127.0.0.1:7860:7860 permitforge-space
# In another terminal:
uv run --project envs/permit/openenv python envs/permit/openenv/space_root/smoke.py \
  --url http://127.0.0.1:7860 --output results/evidence/space-smoke.json
```

The recipe bakes eval showcases with `space_root/bake_showcases.py`; other tasks
build fields on demand. [The historical container smoke](results/evidence/space-container-smoke.json)
records native and fast tool results from the earlier image. A container rebuild
of the current recipe remains TODO; source-copy and native-stage contracts are tested.
No model credentials are needed for these commands.

## Layout

```text
README.md, DESIGN.md, CONTRACT.md, SOURCES.md   task, implementation, rules, provenance
docs/SCHEMAS.md                                public data contract
convert/                                      extraction, conversion and diagnostics
envs/permit/
  core/                                       verifier, evaluator, cache, cost, reward, search
  openenv/                                    OpenEnv server, tools, rollout harness, Space recipe
  tasks/                                      train/eval and legacy records; local task/met data
results/evidence/                             reproducible measurement receipts
```

## Data and licence

Meteorology is NOAA ISD hourly observations and IGRA2 soundings, processed by
AERMET. EPA SCRAM supplies AERMOD and BPIP-PRIME. Facilities, emissions, controls,
costs, backgrounds and ozone are synthetic. Flat terrain, one control ladder per
source and a common stack-extension price within each facility limit coverage.
Station-disjoint tasks do not represent real industrial inventories.

[SOURCES.md](SOURCES.md) records acquisition URLs, hashes, build pins and
U.S. Government-work terms. Code is licensed under [Apache-2.0](LICENSE).
Use [CITATION.cff](CITATION.cff) to cite PermitForge.
