# PermitForge final evaluation protocol

Final version, 2026-10-17. Freeze this file and `run.json` before sampling.
This supersedes the earlier cohort, effort, output limit and scheduling decisions.
The panel ran with the version of this file whose SHA-256 `run.json` records
(`1edab505616c5ecc02d4e29a202441fc9e75d483e0934f22a486795a4c7684be`). The edits since
are wording, the probe cache path, the repeat-count note below and the appended
extended-budget section. No sampling rule changed.
The pilot protocol is preserved as [PROTOCOL-pilot.md](PROTOCOL-pilot.md), with later wording edits.
Existing pilot rollouts remain untouched and cannot be imported into this protocol:
the protocol hash and sampling settings differ. Model runs are launched by hand, never by tests.

## Models and launch effort

| Launch flag (one value for all models) | Codex request | Anthropic request |
|---|---|---|
| `--effort medium` | `model_reasoning_effort="medium"`; wire `reasoning.effort="medium"` | `output_config={"effort":"medium"}` |
| `--effort high` | `model_reasoning_effort="high"`; wire `reasoning.effort="high"` | `output_config={"effort":"high"}` |
| `--effort xhigh` (default) | `model_reasoning_effort="xhigh"`; wire `reasoning.effort="xhigh"` | `output_config={"effort":"max"}` |

The Codex models are `codex:gpt-6.1-sol` and `codex:gpt-5.6-sol` (GPT-6 Astra was dropped on 2026-10-17 to stay within plan usage limits); Claude models are
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
with three independent repeats per model/task (the file still says four; the run manifest sets three). It shortlists 30 tasks per split,
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
checks the cohort marker, quotas and frozen-probe selection before any
provider preflight, including during a dry run.

```sh
# Fresh probes go to a new cache; the tracked probes are frozen at the code they were made with:
uv run --no-sync python -m eval.select_tasks --jobs 4 --field-jobs 3 \
  --cache results/selection-probes-new --output results/subsets-new.json
# Rebuild selection from the frozen probes, without native work:
uv run --no-sync python -m eval.select_tasks --report-only \
  --cache eval/selection-probes --output results/subsets.json
PERMIT_FIELDS_CACHE_GB=64 uv run --no-sync python -m eval.warm_fields \
  --subsets results/subsets.json --field-jobs 12
```

To cancel a probe cleanly, create `eval/selection-probes/.stop`
and wait for its workers to release their leases. Resume with the same command;
it clears the marker only after acquiring the exclusive probe-run lock.

The probe has at most four task workers and twelve native processes, and evicts
packs after use. Warming uses the same twelve-native-process ceiling. Warming retains all
selected packs and verifies the final inventory; use the same sufficiently large
cache cap when launching the panel. Startup refuses missing/stale selected packs and pins all selected local packs for the run.

## Scheduling, repeats and spend

`--repeats 3 --order round-robin` finishes repeat r of the cohort before repeat
r+1. Each round finishes eval before train. Paid dispatch order is Opus-eval,
Sonnet-eval, Sonnet-train, Opus-train, preserving eval and Sonnet training coverage
when the cap trims the tail. Codex and Anthropic have separate concurrency flags,
with combined episode concurrency at most twelve. Codex continues if paid launches
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
`eval/rescore.py` and the native verifier, and is not part of this preparation.

## Launch

The following command spends model credit. Add
`--dry-run` to inspect it without any model episodes.

```sh
ENABLE_WEB_INTERFACE=false PERMIT_FIELDS_JOBS=4 PERMIT_FIELDS_CACHE_GB=64 MAX_CONCURRENT_ENVS=12 \
PERMIT_NATIVE_VERIFY_SLOTS=8 uv run --project envs/permit/openenv --no-sync python eval/run_eval.py \
  --run final-medium3 --models codex:gpt-6.1-sol codex:gpt-5.6-sol \
  anthropic:claude-opus-5-5 anthropic:claude-sonnet-5-5 --subsets results/subsets.json --split both \
  --effort medium --repeats 3 --order round-robin --codex-concurrency 6 --anthropic-concurrency 4 \
  --max-tokens 32768 --max-turns 40 --wall-clock 1800 --budget-usd 150
```

Regenerate the combined panel's separate boards with `python -m eval.report final-medium3
--split eval --output results/final-eval.md` and the corresponding `--split train`
command, using the OpenEnv uv project. The runner also writes these split boards
automatically. For an external recorder, warm its host separately; local cache
pins do not protect a different host's files.

Offline suite: `uv run --project envs/permit/openenv --no-sync python -m pytest eval/tests -q`.

## Extended-budget experiment

Extended-budget experiment, 2026-10-18: a separate headroom experiment asks whether the evaluation
budget constrains performance. In `results/rollouts/final-medium3`, neither Claude
model exhausted its budget (0/120 episodes); the GPT models exhausted it in
27/120 episodes (16 GPT-6.1 Sol, 11 GPT-5.6 Sol). Recompute the counts and the
cost prior without sampling:

```sh
export PERMIT_FIELDS_CACHE_GB=64
uv run --project envs/permit/openenv --no-sync python -m eval.extended
```

Use the same frozen `results/subsets.json` cohort: 10 eval tasks followed by
10 train tasks. Run one attempt per task/model, effort **medium**, **100 valid
evaluations**, **150 turns**, **7,200 seconds**, and 32,768 output tokens per
response. Infrastructure retries follow the existing rules and retain their
attempts; terminal outcomes are never selectively rerun. No result enters
`results/eval-v2.json`, `results/final-eval.md`, `results/final-train.md`, or the
main board. Report the extended results beside the 30-evaluation results; never
pool them. `eval.card` rejects extended runs, and protocol/sampling hashes prevent
cross-budget imports and pooling. The public Space still defaults to 30.
At 30, prompt text and SHA-256 remain exactly
`75fa2efb3393976c7c85209945f0ffbd97a18449cd75f2f00fdf54d2a8ce0f13`.
The chosen evaluation budget, prompt caching and experiment marker enter the
protocol hash and `run.json`; resume rejects changes.

The credit leg is `extended-budget`: `anthropic:claude-fable-5-1` first uses the
remaining Anthropic console credit, passed as `--budget-usd`;
`codex:gpt-6-astra` uses the ChatGPT subscription. Fable's verified standard
USD/MTok prices are input 10, output 50, cache write 5m 12.50, write 1h 20,
read 0.25. Source checked 2026-10-18:
[official Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing).
`--prompt-cache` opts this experiment into explicit breakpoints on the system
prompt, final tool definition and latest turn. Previous-turn breakpoints are not
retained; signed thinking blocks remain unchanged. Caching remains off by default
for the main protocol. Raw request bodies and categorized usage show the policy.

Unlike the main run's per-request hard cap, this leg reserves an episode cost
estimate before launch and never interrupts an admitted episode for budget.
Committed charges plus concurrent episode reservations plus the next episode's
estimate must fit. The initial conservative estimate is $60: reprice the
maximum-cost Opus token record from final-medium3 at Fable rates ($7.59872),
scale by 100/30 and double ($50.6581333), then round upward to the next $10.
The command above prints the source path/hash and arithmetic.
After the first completed API episode, use the maximum measured episode cost;
unknown charges retain their full launch reservation, including after resume.
This is an empirical cost estimate, **not a mathematical invoice ceiling**;
an unusually expensive admitted episode can exceed its reservation. No request
receives the main protocol's budget cutoff in this experiment. An unaffordable
task is left unstarted for the subscription leg; it is never recorded as a
budget-truncated episode. Astra continues independently.

Once the credit leg finishes with no affordable Fable launches left, it writes
`extended-budget/remaining-tasks.json`. Only then run `extended-budget-sub` with
`claudecode:claude-fable-5-1` and that file. The launcher checks its source run is
finished and its tasks match the authoritative credit records. It may leave a
small credit balance that cannot admit another episode. Each Fable result carries
its provider/harness prefix; API and Claude Code results appear side by side,
never silently pooled. Subscription harnesses only are used for Astra and the
Fable subscription leg; Fable's explicit initial console-credit leg is the sole
API exception in this experiment. Subscription USD is null.

Claude Code 2.1.302 is launched with the flags verified in `claude --help` on this
host (the driver and its CLI test are tied to that version; with another version the
test skips and the driver must be re-checked): `-p --model --effort --system-prompt --tools "" --mcp-config
--strict-mcp-config --allowedTools --output-format stream-json --verbose
--max-turns --no-session-persistence --disable-slash-commands --setting-sources
--settings`. The working directory is empty. `CLAUDE_CONFIG_DIR` points to an
isolated directory initially containing only a mode-0600 copy of the subscription
OAuth credential. No user/project settings, hooks, memory, installed plugins,
skills or other MCP configurations are copied; hooks and memory are disabled.
The child strips API keys, auth tokens and inherited provider configuration.
An isolated `claude auth status --json` must report `claude.ai`, first-party,
no API-key source. The recording proxy also refuses every API-key-authenticated
request, filters host context, pins the system prompt, effort and four tools,
and validates returned calls before releasing them. Authorization headers are
never captured. The MCP gateway/session recording is the same as Codex's.

`ANTHROPIC_BASE_URL` works with OAuth on this host: the installed CLI test uses
fake subscription credentials and a local fake provider, captures the first
filtered request and executes `get_scenario` then `submit_design` through the
real MCP gateway with fake physics. That request contains only our system prompt
and the four public tools. Stream JSON records returned thinking and tool calls;
raw requests, SSE, stderr, argv and MCP config are retained. `--bare` cannot be
used because it disables OAuth. `--safe-mode` cannot be used because it disables
explicit MCP too. Two compiled-in plugins are still listed by this CLI; the
individual controls and provider boundary remove their context and tool effects.
They are not user-installed plugins. These limitations are exposed in the
saved init event, not described as an entirely plugin-free binary.
Sources checked 2026-10-18: [CLI flags](https://code.claude.com/docs/en/cli-reference),
[configuration isolation](https://code.claude.com/docs/en/authentication),
[settings](https://code.claude.com/docs/en/settings).

These commands spend model credit. Set
`PERMIT_CONSOLE_CREDIT_USD` to the remaining console credit before the credit leg;
the existing maximum of $190 still applies. Episode/native concurrency is kept
within four. The second command is only for the finished credit leg's remainder:

```sh
export PERMIT_FIELDS_CACHE_GB=64
ENABLE_WEB_INTERFACE=false PERMIT_FIELDS_JOBS=4 MAX_CONCURRENT_ENVS=4 \
PERMIT_NATIVE_VERIFY_SLOTS=4 uv run --project envs/permit/openenv --no-sync python -m eval.run_eval \
  --run extended-budget --models codex:gpt-6-astra anthropic:claude-fable-5-1 \
  --subsets results/subsets.json --split both --repeats 1 --order round-robin \
  --effort medium --evaluate-budget 100 --max-turns 150 --wall-clock 7200 \
  --max-tokens 32768 --prompt-cache --codex-concurrency 1 --anthropic-concurrency 1 \
  --budget-usd "${PERMIT_CONSOLE_CREDIT_USD:?set remaining console credit}"

ENABLE_WEB_INTERFACE=false PERMIT_FIELDS_JOBS=4 MAX_CONCURRENT_ENVS=4 \
PERMIT_NATIVE_VERIFY_SLOTS=4 uv run --project envs/permit/openenv --no-sync python -m eval.run_eval \
  --run extended-budget-sub --models claudecode:claude-fable-5-1 \
  --subsets results/subsets.json --split both --repeats 1 --order round-robin \
  --effort medium --evaluate-budget 100 --max-turns 150 --wall-clock 7200 \
  --max-tokens 32768 --claudecode-concurrency 1 \
  --remaining-tasks results/rollouts/extended-budget/remaining-tasks.json
```

Add `--dry-run` to either command to print schedules/settings without provider
preflight, server startup, cache computation or model episodes. For a preparation
preview before credit sampling, the subscription dry run alone may consume a
clearly labelled preview remainder without a source run. Actual subscription
launches require the authoritative remainder file. Offline tests use fake
binaries/providers, including an installed-CLI check against a fake endpoint.
