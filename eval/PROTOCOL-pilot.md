# PermitForge pilot evaluation protocol (historical)

This is the pilot protocol, kept for the record. The final protocol is [PROTOCOL.md](PROTOCOL.md).

Pilot protocol, 2026-10-17. Freeze this file and each run's `run.json` before model sampling.

## Cohort and attempts

Use every eligible converted `permit-v2-eval` task, station-disjoint from train, with one attempt per model and task.
The plan's upper bound is 130 eval tasks at five stations; actual coverage comes from the selected task records and is
saved in `run.json` and the board. Train calibration uses the same rules. `--tasks` must name eligible members of the
requested split. `--limit` selects a deterministic prefix, not a representative subset; explicitly record a
station-stratified selection for a pilot or reduced panel. Report paired comparisons only on intersecting task IDs,
and disclose each model's full coverage. Do not pool repeated model/task attempts across run names.

The requested Anthropic models are `claude-opus-5-5`, `claude-sonnet-5-5`, and `claude-haiku-5-5`.
The subscription driver accepts locally catalogued Codex models, including `gpt-6.1-sol` and `gpt-6-astra`.
HF router execution is outside this work package. No API calls, paid pilots, or publication are authorized by this file.
Spending must be authorized by hand before a paid driver runs, including the five-task cost pilot.

## Locked interaction

The task is v2; `permit_openenv.prompts.INSTRUCTIONS` currently has `PROMPT_VERSION = 2`.
Every run records that integer and the SHA-256 of the exact text. Both drivers use that text as the system instruction.
Both expose exactly `get_scenario`, `check_design`, `evaluate_design`, and `submit_design`.
The environment owns the evaluation budget (30), validity, costs, and terminal native verification.
The harness does not approximate a grade. There are at most 40 model responses, provider-default temperature,
and a default cap of 4096 output tokens per response. Overrides below the turn ceiling are diagnostic runs and must
be disclosed; token caps, wall-clock seconds, provider price snapshot and concurrency are recorded.
The default wall-clock allowance is 1800 seconds (30 minutes). A timeout never becomes a fresh graded attempt.

Anthropic uses the synchronous Messages API, with no requested caching, batch mode, fast mode or regional surcharge.
SDK retries are disabled. Input, cached input, output and per-response cost are retained; aggregate input tokens include
cache reads and writes. `prices.toml` records standard prices, their source and verification date, including Haiku's
prompt-length tier and cache TTL categories. Sources: [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
and the [Haiku pricing description](https://www.anthropic.com/claude/haiku), checked 2026-10-17.
Unknown models are rejected before a request. Prices were re-verified for the final panel.

## Isolation and recording

Run the OpenEnv server in-process by default. The harness adds an authenticated `/eval/records/{episode_id}` sidecar
for full Episode records. It requires a private bearer token and is never registered as a model tool.
An external `--server` must run `python -m eval.transport` with `PERMIT_EVAL_TOKEN`; an ordinary server without the
recording route is rejected before sampling. This avoids silently omitting SCHEMAS' record-only receptor arrays.
Before opening any episode, startup imports every requested driver (including the Anthropic SDK), checks the
Anthropic key's presence, checks Codex CLI availability and ChatGPT login, resolves eligible tasks, and verifies
server health plus its authenticated `/eval/preflight` task inventory. Failures exit before creating run artifacts
or resetting an environment; credentials and login output are never printed.
Each episode gets its own OpenEnv session. A separate streamable HTTP MCP gateway exposes only that session's four tools;
there is no reset, task-selection, filesystem, record-reading, or arbitrary HTTP tool on that gateway.
The recording server forces `ENABLE_WEB_INTERFACE=false`, including when the caller sets it true. Scored runs
never serve the shared Gradio playground. The public Space may keep `/web/` as a demo; it is not a scored endpoint.
Reset carries the same public scenario as `get_scenario`; geometry and full receptor arrays remain private.

Codex runs `codex exec -m MODEL --json` in a fresh temporary working directory, with user configuration ignored,
project instructions disabled, shell/file-image tools and apps/plugins/browser/agents disabled. Authentication must
be ChatGPT subscription login; API-key billing is rejected. The installed CLI's local model metadata is copied to a
temporary catalog to select explicit function tools rather than Code Mode, deferred tool discovery, or Responses Lite.
Model identity and reasoning defaults are retained. This is a disclosed harness adaptation, not a stock Codex coding session.

Disabling shell alone does not remove every built-in tool. Therefore the local subscription boundary also removes all
non-PermitForge tool schemas, strips host/system/developer/skill context, installs the locked prompt, and forwards only
the episode's assistant/tool conversation. It buffers each provider response and rejects unadvertised function calls
before Codex can execute them. It counts model requests for the turn ceiling and sets the output-token cap.
Only the fixed ChatGPT subscription endpoint is reachable through this boundary. Authorization headers are forwarded
in memory and never saved. No general-purpose network tool is available to the evaluated model.
Unsupported protocol/tool shapes fail closed. A changed CLI must pass the isolation suite again.

The implementation is grounded in the installed `codex features list`, `codex exec --help`, its cached model metadata,
and the [official configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
CLI event capture follows the [official non-interactive output documentation](https://learn.chatgpt.com/docs/non-interactive-mode)
and [CLI reference](https://learn.chatgpt.com/docs/cli/reference), checked 2026-10-17.
The tests run the installed CLI against a fake provider with a prompt requesting local file listing and
`envs/permit/tasks` access, exercise an actual MCP tool, inspect the outgoing tool inventory, and separately inject
forbidden file/shell calls to prove they never reach execution. No external model is used in those tests.

Each episode directory is `results/rollouts/<run>/<model_slug>/<task_id>/`:

- `episode.json`: SCHEMAS record with private geometry, retained tool steps, receptor arrays, grade and usage.
- `transcript.jsonl`: ordered system/user/assistant messages, tool calls/results, timestamps and per-response usage.
  Text and provider-returned thinking/reasoning blocks or summaries are retained in full; nothing unreturned is inferred.
- `raw/`: Anthropic wire request/response JSON; Codex provider request/SSE responses, full CLI JSONL and stderr,
  exact argv/config, stdin and the temporary model catalog. Request options are also retained before dispatch.
- `env.log`: episode-specific receipt, field-preparation, evaluation timing and native-verification diagnostic lines.
- `attempts/episode-NNNN/`: each infrastructure attempt's checkpoint, raw capture, transcript and environment log.
  `attempt.json` records status, attempt number, error/traceback, usage, budget charge and whether resume may retry.
  Request-level failures live inside that attempt's `attempts/` directory. Successful attempts retain their capture too.

The live environment bounds retained steps/diagnostics; the append-only harness transcript is the complete conversation.
Assistant text on each Episode step remains capped at 2000 characters; transcript and raw content are not truncated.
The flat `<task_id>.json` remains a compatibility snapshot; readers prefer the canonical `episode.json` and never double count.
Active checkpoints exist only inside an attempt directory. The collector publishes a canonical episode only when the
driver completes normally or a checkpoint already contains an authoritative submission. Reports and calibration ignore
unfinished legacy checkpoints (`attempt_complete:false`). Infrastructure attempts appear separately on the board and
in the index; they do not count as failed designs or no-submissions. Every future is collected even if another driver fails.
Per run, `run.json` records git commit/dirty status, prompt version/hash, protocol hash, models, sampling settings,
concurrency, budget, host and start/end timestamps. `harness.log`, `costs.jsonl`, `index.json` and boards are retained.
Writes redact credential keys, known credential environment values and token/header patterns. Authentication headers
are never captured. Redacted wire text otherwise preserves original content. Capture tests scan every generated file.

Session close and graceful SIGINT/SIGTERM finalize unfinished private environment records as `no_submit`, zero reward,
with a fixed terminal timestamp. That cleanup alone does not establish a model outcome: a driver/harness exception
is archived as infrastructure, with its traceback. An in-flight native submission keeps its authority to finish the grade.
SIGKILL or host loss cannot execute cleanup; resume archives abandoned checkpoints as infrastructure (explicitly noting
that no traceback was available). Recovery preserves an existing native grade and never resubmits it. A saved submission
intent without a returned grade is held for investigation and cannot be retried automatically.
`ended` stays within `submitted|no_submit|turn_limit|error`; timeout and infrastructure detail use `error_type`.

Both the environment Task and eval's `EvalTask` accept SCHEMAS' `source:none, best:null` anchor shape.
Native evaluation and reward logic remain in the existing core package.

## Retries, resume and budget

Retry only transient transport/provider infrastructure errors, up to two retries of the same request, logged with
attempt number and timestamp. Never restart the environment to get a better grade. Invalid, infeasible, no-submit,
turn-limit, and timed-out episodes are retained. A submit is marked attempted before dispatch; a lost grade response
cannot trigger another submit. A partially streamed Codex response is not replayed because it may already be billed.
Resume skips every graded terminal episode. A driver/harness infrastructure failure may start one new attempt per
resume, up to two retries (three total attempts), unless a submission was attempted. All attempts and costs are retained;
the original failed attempt never becomes a graded `no_submit`. Exhausted retries remain explicitly ungraded.
Provider connection/read failures are infrastructure; exhausting the episode's own wall-clock budget remains a
terminal timeout. `--force` refuses to overwrite any recorded attempt. A separate diagnostic run must be disclosed
and cannot be silently pooled into the panel.
Atomic record/index writes and a per-run process lock protect resume. Task/model selection and settings cannot change
under the same run name. Scheduling is task-major, with at most four simultaneous episodes and the native pool's own limit.

Paid runs require `ANTHROPIC_API_KEY` and positive `--budget-usd`. Before launching an episode, require:

`spent + max(concurrency × estimated_episode_cost, outstanding_reservations + estimated_episode_cost) <= budget`.

The estimate is the observed nearest-rank p95 episode cost, or a conservative USD 5 prior before observations
(`--prior-episode-usd` is recorded). Persisted episode costs are included on resume. This is a launch guard, not a hard
invoice cap: an already-running episode can exceed the estimate. Use the measured pilot to set panel size and prior.
Unknown billed usage must be investigated before resuming a paid run. Subscription USD remains null, not a fabricated zero.

## Calibration, reports and regrading

The calibration step described here was retired before the final panel; no `calibrate.py` exists. It used only feasible submitted model episodes from the declared runs, for both splits. Per task:
`worst = 0`, `best = maximum positive feasible USD saved`, `saturation = 4 × best`, `source = panel`, `frozen_at = UTC`.
No positive saving means `worst:0, best:null, saturation:null, source:none`: an all-draft panel cannot define a nonzero
calibration span. Search and naive policies never supply anchors. Existing frozen anchors cannot change; identical
replays are idempotent. The model-only decision superseded the older search fallback.
Calibration writes only the explicitly selected task record files (or both converted v2 files by default).

`rescore.py` re-verifies the first stored submission using the full ARM2 verifier and current panel anchors. It preserves
the old grade and submission in `rescore_history`, updates final/submission results and arrays, and rebuilds indices
and run boards. No-submit remains no-submit. Infrastructure failure leaves the previous grade intact and exits with
an error; it is not an infeasible design. Regrading is sequential and may take minutes per submission.

`report.py` regenerates Markdown and JSON with the FineEnvs card's `models[]` fields, per-station panels, paired task
differences, feasible rate, raw USD and percent savings, calibrated reward, evaluation/turn counts, end categories,
and measured USD/episode. Raw USD means retain the -1 failure sentinel; percent savings are zero for failures.
A missing anchor makes the overall reward mean unavailable, rather than treating uncalibrated tasks as zero or dropping
them. Coverage and available anchors are explicit. Bootstrap whole stations with replacement, retain all tasks in each
sampled station, and take the task-weighted mean: 2000 replicates, seed 0, percentile 95% CI. Report the cluster count
(five for the full eval cohort); fewer than two clusters yields no CI. Paired CIs bootstrap the matched stations.

Naive rows call the policies in `permit_core.baselines` on exactly the reported task cohort and independently verify
submitted designs. `report.py --compute-baselines` uses already cached fields under a shared lease; it refuses to build
missing packs. Records are saved under the run's `baselines/`. Later reports reuse them. Missing artifacts are explicitly
listed; partial baseline cohorts are not presented as comparable. Search is never included as a model or reward anchor.

## Commands

Run from the repository root, using the existing OpenEnv uv project with its locked `agents` extra. No new dependency is introduced.

```sh
# Full WP suite, all offline (includes local sockets and installed CLI with fake provider).
uv sync --project envs/permit/openenv --extra agents --locked
uv run --project envs/permit/openenv --no-sync pytest eval/tests -q
uv run --project envs/permit/openenv --no-sync ruff check eval

# Subscription smoke: one eligible TRAIN task; never the eval panel.
uv run --project envs/permit/openenv --extra agents --locked python eval/run_eval.py --run wp2.1-smoke \
  --models codex:gpt-6.1-sol --split train --tasks 003bd328a392-s47 --concurrency 1 --max-turns 40

# Only after paid use is authorized; install the existing agents extra first.
uv run --project envs/permit/openenv --extra agents --locked python eval/run_eval.py --run approved-pilot \
  --models anthropic:claude-sonnet-5-5 --split eval --tasks TASK_IDS --budget-usd 25

# Offline recovery of a crashed/legacy run and its index/board; never launches an episode.
uv run --project envs/permit/openenv --no-sync python -m eval.attempts RUN

uv run --project envs/permit/openenv --no-sync python eval/report.py RUN --compute-baselines
uv run --project envs/permit/openenv --no-sync python eval/rescore.py RUN
uv run --project envs/permit/openenv --no-sync python eval/report.py RUN

# Local, redacted HF dataset configs; does not upload. Use an empty destination.
uv run --project envs/permit/openenv --no-sync python eval/export_hf.py RUN \
  --out results/rollouts/RUN/hf-export
```
