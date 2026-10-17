# PermitForge final evaluation protocol

Version WP2.8b, updated 2026-10-17. Freeze this file and `run.json` before sampling.
This supersedes the earlier cohort, effort, output limit and scheduling decisions.
The byte-identical pilot protocol is preserved as [PROTOCOL-pilot.md](PROTOCOL-pilot.md).
Existing pilot rollouts remain untouched and cannot be imported into this protocol:
the protocol hash and sampling settings differ. The operator launches all model work.

## Models and launch effort

| Launch flag (one value for all models) | Codex request | Anthropic request |
|---|---|---|
| `--effort medium` | `model_reasoning_effort="medium"`; wire `reasoning.effort="medium"` | `output_config={"effort":"medium"}` |
| `--effort high` | `model_reasoning_effort="high"`; wire `reasoning.effort="high"` | `output_config={"effort":"high"}` |
| `--effort xhigh` (default) | `model_reasoning_effort="xhigh"`; wire `reasoning.effort="xhigh"` | `output_config={"effort":"max"}` |

The Codex models are `codex:gpt-6.1-sol` and `codex:gpt-6-astra`; Claude models are
`anthropic:claude-opus-5-5` and `anthropic:claude-sonnet-5-5`. Anthropic names its
highest level `max`; the launch value `xhigh` maps to `max` on Claude. At every
effort level both Claude requests use
`thinking={"type":"adaptive","display":"summarized"}`. Manual `enabled` thinking
with `budget_tokens` is unsupported for these models. No beta header is needed.
Effort is fixed across every request and repeat; temperature is provider-default.
Sources checked 2026-10-17: [effort parameter and levels](https://platform.claude.com/docs/en/build-with-claude/effort),
[manual-thinking migration](https://platform.claude.com/docs/en/build-with-claude/extended-thinking).

Keep the entire returned assistant content array unchanged when appending tool
results, including thinking signatures and `redacted_thinking` blocks. Capture
all returned summaries and opaque blocks; unreturned internal reasoning is not
inferred. Explicit summarized display is necessary because these models otherwise
omit readable thinking. Sources checked 2026-10-17:
[thinking/display](https://platform.claude.com/docs/en/build-with-claude/thinking),
[tool-use round trips](https://platform.claude.com/docs/en/build-with-claude/thinking-tool-workflows).

Each provider response is capped at **32,768 output tokens**, **40 responses per
episode**, **30 valid evaluations**, and **1,800 seconds wall clock**. Anthropic
thinking counts inside `max_tokens`, including thinking not displayed; there is
no separate thinking budget. Both Claude models support 128K output and a 1M
context window; the evaluation deliberately sets the smaller common output cap.
Sources checked 2026-10-17: [Opus specifications](https://platform.claude.com/docs/en/models/opus-5-5/overview),
[Sonnet specifications](https://platform.claude.com/docs/en/models/sonnet-5-5/overview),
[thinking token accounting](https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost).
Reproduce settings: `uv run --no-sync python -c 'from eval.settings import EFFORTS,MODELS,model_settings,MAX_TOKENS; print(MAX_TOKENS,{e:{m:model_settings(m,e) for m in MODELS} for e in EFFORTS})'`.
The final launcher rejects output-token limits other than 32,768.

The Codex driver sets effort through CLI config and checks the outgoing request
before forwarding it. Installed-CLI tests capture `reasoning.effort` against a
local fake provider. Per-episode temporary CLI state avoids writes to the host's
user-state directory; subscription authentication is a read-only symlink to the
existing auth file. No model is used by those tests. Effort appears in raw
requests/config, episode settings, manifest sampling, the protocol hash, and
board labels (`gpt-6.1-sol@high`, `claude-opus-5-5@high`, or
`claude-opus-5-5@max` for launch effort `xhigh`). `launch_effort` records the
shared flag and `effort` records the provider's name in episode settings.
`run.json` records the flag and per-model mapping in settings and sampling.
`protocol_sha256` hashes these protocol bytes, a newline, and the sorted-key JSON
sampling settings; `protocol_document_sha256` separately hashes this file alone.
Changing effort changes the protocol hash and prevents resume/import pooling.

## Model-free cohort

`eval/select_tasks.py` writes `results/subsets.json`: 10 eval and 10 train tasks,
with four independent repeats per model/task. It shortlists 30 tasks per split,
stratified by station, using source count, native-draft standards near their
limit, and an uncontrolled-at-GEP exceedance bound. The preserved manifest has
no binding/exceedance metadata. The bound scales the uniform control ladder;
NO2 uses the conservative ARM2 ratio range, rather than claiming linearity.
Every structural feature, source file hash, candidate probe and selection group
is retained. The exact rule and a card-ready paragraph are in `subsets.json`.

For each candidate, on-demand float64 fields feed draft, cheapest-first greedy,
v1 coordinate descent, search with a 60-second traversal allowance, and random_30
with seeds 0 through 7. These are fast-field selection diagnostics, not native
submission grades. The search reuses fast evaluations for its final candidate;
its output is explicitly marked unverified. No model results or pilot membership
are read. Failure savings are zero when computing hardness and variance.

Hardness is `1 - greedy_saved/best_saved`, with zero when best_saved is zero.
Best includes all policies, search and all random seeds. Variance is population
standard deviation of random_30 saved fractions. Reserve two easiest tasks from
distinct stations; select eight hard tasks using the product of hardness and
variance midrank percentiles, then hardness, variance and task ID as tie breaks.
Quotas are two per eval station and one per train station. Selection from frozen
probe artifacts is deterministic and independent of input order. Fresh timed
searches may visit different node counts under different host load; retained
probe artifacts are the reproducible selection inputs. No search is called an optimum.
Only `eval.select_tasks --report-only` reproduces the active selection from the
frozen probes. The historical `eval.subsets` defaults to
`results/subsets-wp2.7.json` and refuses the active destination. The launcher
checks the WP2.8 cohort marker, quotas and frozen-probe selection before any
provider preflight, including during a dry run.

```sh
uv run --no-sync python -m eval.select_tasks --jobs 4 --field-jobs 3 \
  --cache internal/reports/WP2.8-evidence/difficulty-probes --output results/subsets.json
# Rebuild selection from frozen probes, without native work:
uv run --no-sync python -m eval.select_tasks --report-only \
  --cache internal/reports/WP2.8-evidence/difficulty-probes --output results/subsets.json
PERMIT_FIELDS_CACHE_GB=72 uv run --no-sync python -m eval.warm_fields \
  --subsets results/subsets.json --field-jobs 12
```

To cancel a probe cleanly, create `internal/reports/WP2.8-evidence/difficulty-probes/.stop`
and wait for its workers to release their leases. Resume with the same command;
it clears the marker only after acquiring the exclusive probe-run lock.

The probe has at most four task workers and twelve native processes, and evicts
packs after use. Warming uses the same twelve-native-process ceiling. Warming retains all
selected packs and verifies the final inventory; use the same sufficiently large
cache cap when launching the panel. Startup refuses missing/stale selected packs and pins all selected local packs for the run.

## Scheduling, repeats and spend

`--repeats 4 --order round-robin` finishes repeat r of the cohort before repeat
r+1. Each round finishes eval before train. Paid dispatch order is Opus-eval,
Sonnet-eval, Sonnet-train, Opus-train, preserving eval and Sonnet training coverage
when the cap trims the tail. Codex and Anthropic have separate concurrency flags,
with combined episode concurrency at most four. Codex continues if paid launches
are denied. After round one, `projection.json` and `harness.log` record measured
USD/episode per model multiplied by its remaining episodes; subscription USD is
unknown. Before sampling, `--dry-run` prints the full schedule, per-model settings
and an explicit prior projection without provider login, server startup or episodes.

The unmeasured paid cost prior is $2.00/episode at `medium`, $3.50 at `high`,
and $5.00 at `xhigh`, for both Claude models. This planning assumption uses
40%, 70%, and 100% of the earlier $5.00 prior; it is not a measured effort-cost
relationship or a provider quote. `--prior-episode-usd` overrides it. The dry
run prints the assumption, and round-one measurements replace it per model.
Reproduce the prior: `uv run --no-sync python -c 'from eval.settings import EPISODE_PRIORS,PRIOR_ASSUMPTION; print(EPISODE_PRIORS,PRIOR_ASSUMPTION)'`.

Anthropic has a **hard $190 maximum**, optionally lowered by `--budget-usd`.
The existing p95/prior episode launch guard remains conservative, but is not the
hard cap. A shared request ledger atomically persists a worst-case reservation
before each dispatch: a full context (rounded up to 2^20 input tokens) plus the
entire output limit at the pinned standard prices. Successful responses settle
to measured cost. Missing responses, interrupted requests and request retries
retain the full reservation unless usage is known. These charges survive resume.
A request is never dispatched if committed charges plus its reservation exceed
the cap. This may stop early with unused credit. A budget stop during an episode
is recorded explicitly as `no_submit` with `error_type: budget`, never rerun for
an improved score. The cap covers this harness/run, not unrelated account usage.

No caching, batch, fast mode, server tools or regional premium is requested.
`prices.toml` pins standard prices and cache-accounting categories; Sonnet's
cache-read rate is corrected to $0.10/MTok. Source checked 2026-10-17:
[Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing).
Provider SDK retries are disabled. Transient failures have at most two explicit
request retries, each reserved separately; unknown usage remains conservatively
charged. Infrastructure episode retries and all attempts stay visible.

## Isolation, capture and grades

The system prompt text/version remains unchanged. Both drivers expose exactly
`get_scenario`, `check_design`, `evaluate_design`, `submit_design`. OpenEnv owns
validity, evaluation accounting and final BPIP/AERMOD verification. The model
cannot reset/select tasks, read files/private records, or access arbitrary HTTP.
The Codex boundary retains only the public tool conversation and rejects
unadvertised tool calls before execution. Subscription login is mandatory;
API-key fallback is prohibited. The authenticated recording sidecar is never a
model tool, and the scored server disables the shared web playground.

Canonical records live at `<run>/<model_slug>/<task_id>/r<repeat>/episode.json`.
Each repeat retains `transcript.jsonl`, `raw/` request/response bodies, Codex
JSONL/stderr/argv/config/catalog, `env.log`, and every infrastructure attempt.
Credential keys and known secrets are redacted; authorization headers are never
captured. Captured provider-returned thinking summaries are complete. The
append-only transcript is authoritative when the live step view is bounded.
`run.json`, `schedule.json`, `request-budget.json`, `costs.jsonl`, `index.json`,
boards and projections record provenance. Resume checks cohort, order, concurrency,
prompt, protocol and sampling; terminal repeats are skipped. No selective reruns.
A pending submission without an authoritative grade remains held for investigation.
Imports refuse every occupied repeat, including legacy snapshots, pending
submissions and attempts. Only an identical previously completed import may be
resumed, without writing its evidence again.

Reward is feasible `(draft_cost - submitted_cost)/draft_cost`, otherwise zero;
prescore keeps the -1 failure sentinel. No calibration or reward anchors exist.
Reports separate eval/train; task means average repeats before station-clustered
confidence intervals. Within-task spread and task-weighted CIs follow
[REPEATS.md](REPEATS.md). Efforts cannot be silently pooled. Full regrading uses
`eval/rescore.py` and the native verifier, and is not part of this preparation WP.

## Operator launch

The following command spends model credit and is for the operator only. Add
`--dry-run` to inspect it without any model episodes.

```sh
PERMIT_FIELDS_CACHE_GB=72 uv run --project envs/permit/openenv --no-sync python -m eval.run_eval \
  --run final-xhigh --models anthropic:claude-opus-5-5 anthropic:claude-sonnet-5-5 \
  codex:gpt-6.1-sol codex:gpt-6-astra --subsets results/subsets.json --split both \
  --effort xhigh --repeats 4 --order round-robin --codex-concurrency 2 --anthropic-concurrency 1 \
  --max-tokens 32768 --max-turns 40 --wall-clock 1800 --budget-usd 190
```

Regenerate the combined panel's separate boards with `python -m eval.report final-xhigh
--split eval --output results/final-eval.md` and the corresponding `--split train`
command, using the OpenEnv uv project. The runner also writes these split boards
automatically. For an external recorder, warm its host separately; local cache
pins do not protect a different host's files.

Offline suite: `uv run --project envs/permit/openenv --no-sync python -m pytest eval/tests -q`.
