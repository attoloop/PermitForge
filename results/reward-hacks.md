# Reward-hack ledger

The archived build record contains **61 defects: 54 closed, 4 deferred and 3 waived**.
Its red-team index contains **1,632 probe occurrences, 591 distinct probe names,
across 56 attempts with indexed probe directories**. The database records **107
red-team attempts** in total. These are different measures: a directory name is
not proof that a probe completed or succeeded. Probe outcomes are not present in
the index. Repeated names across attempts are retained.

Reproduce every inventory count below from the [public evidence](evidence/reward-hacks.json):

```sh
uv run python convert/report_reward_hacks.py --summary
```

| Archived category | Closed | Deferred | Waived |
|---|---:|---:|---:|
| alignment | 1 | 0 | 0 |
| ambiguous-falsifier-condition | 0 | 0 | 1 |
| ambiguous-gate-order | 1 | 0 | 0 |
| authoring_gate | 2 | 0 | 0 |
| binding-contract | 1 | 0 | 0 |
| contract-constant-mismatch | 1 | 0 | 0 |
| contract-visibility | 1 | 0 | 0 |
| contradictory-instruction | 1 | 0 | 0 |
| contradictory-public-contract | 1 | 0 | 0 |
| data_provenance | 1 | 0 | 0 |
| degenerate-difficulty | 1 | 0 | 0 |
| derived-budget-composition | 2 | 0 | 0 |
| exception-suppression | 1 | 0 | 0 |
| exploitability | 1 | 0 | 0 |
| falsifier-fidelity | 0 | 1 | 0 |
| falsifier-metric-drift | 2 | 0 | 0 |
| hidden-requirement | 3 | 0 | 0 |
| instruction-environment-mismatch | 1 | 0 | 0 |
| intended-vs-accepted-behaviour | 6 | 1 | 0 |
| internal-contradiction | 0 | 2 | 1 |
| materialization-feasibility | 2 | 0 | 0 |
| mechanical | 2 | 0 | 0 |
| reported resource-accounting fault | 1 | 0 | 0 |
| platform-primitive-feasibility | 1 | 0 | 0 |
| proposal-falsifier-drop | 0 | 0 | 1 |
| proposal-selector-drop | 1 | 0 | 0 |
| protected-control-weakening | 1 | 0 | 0 |
| reference-definition | 1 | 0 | 0 |
| reward-manipulation | 1 | 0 | 0 |
| reward-manipulation / baseline substitution | 1 | 0 | 0 |
| reward_integrity | 5 | 0 | 0 |
| scale-dependent-threshold | 1 | 0 | 0 |
| specification | 2 | 0 | 0 |
| stated-graded-gap | 1 | 0 | 0 |
| stated-requirement-contradicts-grader | 1 | 0 | 0 |
| stated-vs-enforced-diagnostic-contract | 1 | 0 | 0 |
| underpowered-falsifier | 1 | 0 | 0 |
| unobservable-graded-quantity | 1 | 0 | 0 |
| unsourced-mandatory-artifact | 1 | 0 | 0 |
| unstated-grader-condition | 1 | 0 | 0 |
| withheld-material-published | 1 | 0 | 0 |

“Closed” maps to the database's `RESOLVED` status. It does not mean every early
report described an actual exploit or that every fix was independently replayed
for this ledger. Some reports show false rejection, misleading diagnostics, weak
reference policies, or inconsistent experimental claims. A feasible cheap design
beating a weak reference is valid engineering, not evidence of bypassing physics.
V1 paid raw annual savings and used a negative failure floor; v2 uses the feasible saved fraction of draft cost and zero for failures. Scores on those scales must not be compared directly.

All defect IDs below refer to entries in the public evidence JSON, including
first/last observation, discovery and repair stages, event history, closure
checks and waiver text. The database has no separate resolution narrative:
`resolution.summary` preserves its `root_cause` field. When that field is absent,
the table says so instead of treating a suggested remedy as an implemented fix.
Paths in exported historical text are normalized to portable labels.

The tables cover every archived defect, including supporting contract issues, so
the selection does not hide deferred or waived findings. “Review” means a source
or contract review; “audit” means a change audit; “red-team probe” means an indexed
adversarial check. Discovery stages can be checked in `first_seen.node`; repair
stages are separate. V2 test links establish the particular assertions named,
not a general certification of network or deployment security.

## Submission-format exploits

An agent could try to replace a required artifact with a file that blocks reads,
redirects the verifier, cannot be read, or exceeds its resource allowance. These
probe names are present in the JSON index but have **no separately identified
defect record or per-probe outcome**. They must not be counted as extra defects
or automatically described as successful attacks.

| Probe evidence | Intended bypass or disruption | V1 boundary observed in source | V2 status |
|---|---|---|---|
| `P-FIFO-ARTIFACT`, `FIFO-AERMOD` | A pipe or other special file could prevent ordinary artifact snapshotting. | `read_submission` requires a regular file before opening it. | Closed by the tool interface: submissions contain design values, not agent-selected filesystem objects. Transport resource limits remain a separate open audit item. |
| `P-UNREADABLE-MODE`, `P-MODE-000-FILES` | Unreadable artifacts could turn invalid work into an infrastructure fault or incomplete grade. | Opening unreadable files fails; the grade writes a failure result before replay and reports unexpected I/O faults separately. This is source evidence, not a claimed successful replay of these probes. | Closed by the tool interface for agent submission files; backend I/O failures still apply. |
| `P-SYMLINK-DRAFT`, `artifact-symlink` | Redirect required files or reuse linked artifacts outside the submission snapshot. | `read_submission` rejects symbolic links and extra hard links, uses no-follow opening, and compares snapshot identity. | Closed by the tool interface for submission paths. |
| `AERMOD-OVERSIZE`, `OVERSIZE-SPARSE-FILE` | Exhaust verifier memory or disk through an oversized artifact. | `read_submission` bounds stat size and bytes actually read. | Closed by the tool interface for native input files; large JSON requests/history remain untested at the network boundary. |

The preserved implementations are `convert/v1-src/shared/tests/grade.py::read_submission`
and `grade`, with file-type coverage in `shared/tests/selfcheck.py::test_file_types`.
Those historical sources are local preservation inputs, not distributed public
links. Reproduce this source inspection without executing a probe:

```sh
sed -n '82,111p;537,617p' convert/v1-src/shared/tests/grade.py
sed -n '186,201p' convert/v1-src/shared/tests/selfcheck.py
```

## Grader and verifier evasion

| Finding | What could go wrong / how found | V1 closure | Status | V2 status |
|---|---|---|---|---|
| `DEF-f0fa72b721282f5b` | Red-team probes found normalization could admit native text that the native reader would reject, including structural tabs and malformed sign spelling. A passing draft grade could avoid the stated native-input obligation. | Closure narrative says structural tabs and repeated plus signs are rejected while valid scientific notation remains allowed. | Closed | Closed by the tool interface: the environment authors native text. Renderer coverage: [test_render.py](../envs/permit/core/tests/test_render.py). |
| `DEF-16cfe36bad0e182c` | Red-team probes found BPIP comment normalization erased native read boundaries; height comparison also hid extra nonzero precision. | Closure narrative records preserved BPIP read boundaries and exact height comparison. | Closed | Native-comment authorship is closed by the tool interface; height validity is guarded by [test_validate.py](../envs/permit/core/tests/test_validate.py), `test_hostile_heights`. |
| `DEF-896d8260d049e607` | Audit found verifier-owned I/O faults were reported as ordinary submission resource failures, concealing infrastructure failure from measurement. This was not a demonstrated feasible-design bypass. | Final v1 `grade` distinguishes unexpected exceptions with an infrastructure flag and nonzero exit; `selfcheck` checks injected I/O routing. | Closed | Still applies as an infrastructure-accounting concern. [test_verify.py](../envs/permit/core/tests/test_verify.py), `test_missing_binary_is_reported`, checks failure reporting; v2 does not preserve v1's process exit distinction. |
| `DEF-1b81788b1a32bd81` | Review found valid native record reorderings were rejected by an undisclosed positional comparison. False rejection, rather than paid evasion. | Closure narrative records explicit order/count requirements in the instructions. | Closed | Closed by the tool interface: agents no longer choose record ordering. |
| `DEF-a21009d6a1c53f48` | Review found instructions required a meteorology-period card absent from locked templates. Following that instruction caused rejection. | Instructions were corrected to preserve full weather coverage without adding the absent card. | Closed | Closed by the tool interface: locked templates and rendering choose the native cards. |
| `DEF-076a20e781394b52` | Review found instructions demanded preserving incomplete public-template counts while also requiring missing records to be added. | Count/order references changed to the complete locked table and BPIP record inventory. | Closed | Closed by the tool interface. |
| `DEF-32de6ed00ea419e4` | Review found competing expected error codes for copied incomplete templates. No reward difference between failures. | Archived closed; no closure narrative is recorded. Final v1 `grade` checks BPIP before AERMOD consistency. | Closed | Closed by the tool interface for incomplete native files; invalid designs are covered in [test_environment.py](../envs/permit/openenv/tests/test_environment.py). |
| `DEF-92991d7946d41a48` | Review found BPIP, locked-card and downwash errors assigned to inconsistent diagnostic stages. All failed grades already had the failure floor. | Closure narrative identifies the stage-order problem; final v1 helpers use the contracted stage codes. | Closed | Closed by the tool interface for agent-authored native text. |
| `DEF-e5cab6c503158340` | Review found ordinary Unicode comments and reserved-marker mentions rejected despite the documented equivalence rules. | Parser now distinguishes comment bodies from structural records and marker positions. | Closed | Closed by the tool interface for native comments. |
| `DEF-87a8526a51f9e414` | Review found the interactive BPIP parser treated comments as stack tokens although the grader accepted them. | Service switched to the shared comment-aware parser. | Closed | Closed by the tool interface: no interactive BPIP text entry. |
| `DEF-b26132af209326a6` | Review found incomplete source records returned the wrong interactive error code. Grading and token debit were unaffected. | Service now checks record completeness with the design-mismatch code. | Closed | Closed by the tool interface; missing source rows are guarded by [test_validate.py](../envs/permit/core/tests/test_validate.py), `test_hostile_shapes`. |
| `DEF-e85a87b34ec12870` | Review found bad BPIP source IDs and malformed heights returned unrelated interactive error codes. | Service now reports BPIP mismatch for both. | Closed | Closed by the tool interface for BPIP text; malformed design rows are guarded by [test_environment.py](../envs/permit/openenv/tests/test_environment.py), `test_hostile_rows_are_free`. |
| `DEF-3ef0c6b8615bba90` | Review found unchecked height conversion blamed service output instead of malformed input. | Closure narrative records syntax guards exercised through the real client without native launch or debit. | Closed | Closed by the tool interface for source cards; finite-value checks are guarded by [test_validate.py](../envs/permit/core/tests/test_validate.py). |
| `DEF-2140ce2d4b4ffdb5` | Review found interactive validation checked mutable source data before locked structure, unlike the terminal grader. No paid bypass; diagnostics hid the first error. | Unfixed in archive; deferred to later discovery/review. No specific deferral rationale is recorded beyond that scope. | Deferred | Closed by the tool interface for competing native-input stages. General error-detail disclosure remains an open v2 concern. |

## Constant or degenerate designs

| Finding | What could go wrong / how found | V1 closure or reason retained | Status | V2 status |
|---|---|---|---|---|
| `DEF-31d40c0d2e75803d` | Red-team constant policies beat a reference that returned the unchanged draft. The report did not establish that native feasibility was bypassed. | Archived closed after probe replay; no closure narrative recorded. The final reference performs trial-time search rather than treating draft cost as the frontier. | Closed | Still applies as a benchmark-strength question. Feasible constant designs may legitimately save money. [test_baselines.py](../envs/permit/core/tests/test_baselines.py) covers policies; it is not a proof that they save only a small fraction of draft cost. |
| `DEF-fe02a17df96e5338` | Red-team probe earned savings from a small change to a public starting design. The historical report called this leakage/reward shaping but supplied no proof of a physics bypass. | Archived closed after replay; no closure narrative recorded. Final v1 pays exact feasible savings, without normalizing to a claimed optimum. | Closed | Still applies as a headroom measurement, not an automatic exploit. [test_reward.py](../envs/permit/core/tests/test_reward.py) guards failure gating and saved-fraction arithmetic. |
| `DEF-805f428aad176769` | Full-bind replay found a fixed maximum-control/minimum-height policy beat a reference that omitted that starting corner. | Reference now evaluates both height corners independently before descent and fresh replay. | Closed | Still applies to reference quality. Search and baselines are diagnostics; they never define reward anchors. [test_baselines.py](../envs/permit/core/tests/test_baselines.py) covers the lifted search. |
| `DEF-275e58939979b3ce`, `DEF-278f20c7716820dd` | Review and change audit found the nominal search reference silently fell back to the unchanged draft because a prepared ledger was absent. | Closure narratives record replacement with a trial-time search using the public metered service; the older prepared-ledger remedy was superseded. | Closed | Still applies to claims about reference quality. V2 uses full native verification and saved fraction, with failure gating and arithmetic guarded by [test_reward.py](../envs/permit/core/tests/test_reward.py). |
| `DEF-9e35581212cb8e94`, `DEF-bd91d5719db4a484` | Mechanical tests found the reference and no-op both failed; reference materialization expected an absent prepared ledger. A broken reference erased meaningful comparisons. | Archived closed; no individual closure narrative recorded. Final reference entrypoint performs trial-time search. | Closed | Still applies to baseline reliability; [test_baselines.py](../envs/permit/core/tests/test_baselines.py) exercises policy behavior, not all historical trial infrastructure. |
| `DEF-a921d60239a00356` | Review found unreachable models reduced an ostensibly interactive task to blind edits of an expensive draft. | Packaged local transport restored model access and persistent accounting. | Closed | Guarded for ordinary tool flow by [test_integration_records.py](../envs/permit/openenv/tests/test_integration_records.py); deployment availability still matters. |
| `DEF-0edaac74fa140fa7` | Repair/review found a valid corner policy conflicted with requirements that the fixed reference dominate probes. This was a reference-definition conflict, not evidence of invalid physics admission. | Archived closed at specification review; no closure narrative recorded. | Closed | Still applies when interpreting model and naive-policy comparisons. V2 promises no optimum or dominance ceiling. |
| `DEF-a818ae726243895d` | Review found a proposed draft-headroom threshold missing from the acceptance checks. A near-optimal draft could make the task trivial. | Waiver treats richness as a diagnostic rather than a gate, citing strong savings in the recorded pilot matrix. The waiver is evidence of the decision, not a newly reproduced result. | Waived | Still applies as a diagnostic limitation. Current admission and saved-fraction grading must not be described as a richness guarantee. |
| `DEF-262add11c7d465cd` | Later review repeated the missing headroom-threshold concern. | Deferred to discovery/review; no specific reason recorded. It remains deferred even though the related earlier finding was waived. | Deferred | Still applies; no new headroom gate was added by this ledger. |
| `DEF-81845201ffc2bbdc` | Review found the search assumed source-specific none/moderate/maximum IDs that the generic control schema did not guarantee. | Waiver says the actual trial search requires these IDs and fails loudly if absent, with pilot and full-bind checks as protection. It does not generalize the search to arbitrary menus. | Waived | Still applies to reuse beyond the fixed control ladder. [test_baselines.py](../envs/permit/core/tests/test_baselines.py) covers this dataset's policies. |
| `DEF-2f2c931ca743cb23` | Later review repeated the named-control/schema mismatch. | Deferred to discovery/review; no specific reason recorded. Kept separately from the earlier waiver. | Deferred | Still applies as a generic-menu limitation; the current benchmark discloses its single control ladder. |
| `DEF-6a726d396d400830` | Review found incompatible requirements for a trial-time reference versus an author-time prepared ledger and comparator dominance. | Deferred to discovery/review; no specific reason recorded. Final implementation uses trial-time search, but the archived specification conflict was not closed. | Deferred | Still applies as historical specification debt. V2's saved-fraction grading is independent of search diagnostics. |
| `DEF-356400b8b8ee5b28` | Review found pilot saturation could pool models and budgets, overstating what one agent achieved under the normal allowance. | Waived because no grade or reward depends on that pilot statistic; the reason requires reporting a single-model panel at the normal budget when analysis is written. | Waived | Still applies to evaluation reporting. Model results are not replaced by pooled best-case diagnostic claims. |
| `DEF-6fbe8d6498c3bc6c` | Review found fixed dollar diagnostic thresholds changed meaning with facility cost. | Archived closed; no closure narrative recorded. No numerical threshold repair is inferred here. | Closed | Still applies when comparing diagnostics; saved fraction and raw USD savings must both be disclosed. |
| `DEF-8306c051a5edc8f5`, `DEF-c7c56f27aff2b515` | Review found diagnostic metrics and thresholds drifted from the proposed excess-cost and budget-sensitivity tests. This could make a weak benchmark look informative. | Both archived closed; no closure narratives recorded. | Closed | Still applies to honest evaluation claims; no new model panel or diagnostic measurement is supplied here. |
| `DEF-cc775ee04bc049a7` | Review found a coarse response-library search was compared against a much finer search, making the proposed nonadditivity check too weak. | Archived closed; no closure narrative recorded. | Closed | Still applies to numerical/search claims. V2 explicitly changes NO2 to ARM2; it does not claim v1 PVMRM is separable. See [DESIGN.md](../DESIGN.md). |

## Information leaks

| Finding | What could go wrong / how found | V1 closure | Status | V2 status |
|---|---|---|---|---|
| `DEF-380ccb063bc68770` | Review found withheld complete native draft input/output copied into public provenance files, bypassing the intended reconstruction work. | Archived closed without a narrative. Final preserved public instance omits that provenance directory and exposes draft statistics instead. | Closed | Closed by the tool interface for native-file reconstruction. Confidentiality of actual search results and backend paths remains a separate v2 audit concern. |
| `DEF-8bb3ff5f9aee2263` | Review found complete public draft files and mutable values allowed a verbatim copy to pass despite an explicit reconstruction requirement. Passing the unchanged draft earned no savings. | Archived closed without a narrative. Final v1 public templates omit stack/source/downwash records; verifier copies remain complete. | Closed | Closed by the tool interface: draft reconstruction is intentionally no longer an agent obligation. V2's unchanged feasible draft earns zero; [test_environment.py](../envs/permit/openenv/tests/test_environment.py), `test_budget_exhaustion_and_submit_once`. |
| `DEF-204672c5b5edb08b` | Review found a global withholding rule would hide necessary scenario inputs, making the task impossible rather than protecting a solution. | Archived closed; no closure narrative recorded. | Closed | Still applies as an input-visibility requirement. The public scenario allowlist is in [tasks.py](../envs/permit/openenv/permit_openenv/tasks.py); reset geometry currently exceeds the narrower review boundary described below. |

## Budget and accounting exploits

| Finding | What could go wrong / how found | V1 closure | Status | V2 status |
|---|---|---|---|---|
| `DEF-4928fffa8b309816` | Review found undefined interactive grammar could permit more modeled work per token than intended, invalidating the experiment budget. | Archived closed without a narrative. Final shared native parser restricts locked inputs; unit-response requests have an explicit exception rather than arbitrary cases. | Closed | Closed by the tool interface: evaluations accept a full design, not arbitrary native jobs. Spending/terminal rules are guarded by [test_environment.py](../envs/permit/openenv/tests/test_environment.py), `test_budget_exhaustion_and_submit_once`. |
| `DEF-36d9a15a1fc51868` | Review found binary access, network access and process identity were unspecified, so a launch meter could not reliably constrain native computation. | Archived closed without a narrative. Final service/bootstrap source provides separated local execution. This ledger does not certify the historical platform sandbox. | Closed | Native-launch meter removed by the tool interface. Remote evaluation accounting is server-owned; local unrestricted source access is not a sandboxed scored episode. Hostile session behavior remains untested. |
| `DEF-1ef1744673eb4433` | Review found promised clients, service and persistent metering were absent from the agent image. | Closure narrative records packaged Unix-only transport with serialized handlers and persistent meter. | Closed | Closed by the tool interface; ordinary lifecycle/budget behavior is guarded by [test_environment.py](../envs/permit/openenv/tests/test_environment.py). |
| `DEF-3d24e158d524e06a` | Review found serialized launch allowances could not fit the stated episode timeout, making host timing the real budget. | Archived closed without a narrative; final binding publishes a composed timeout. | Closed | Still applies to harness wall-clock limits. Tool budget is tested; no deployment latency guarantee is inferred. |
| `DEF-5b104f25ac3f5326` | Review found rendered episode time omitted BPIP and workflow overhead required by the resource formula. | Binding now composes timeout from pollutant count, native guard and frozen overhead. | Closed | Still applies to external harness timing, which is separate from evaluation count. |
| `DEF-19f44c09ec584ec6`, `DEF-e79e065adf924c40` | Review found hourly-output storage was unspecified, then incorrectly assumed every pollutant used hourly periods. Unbounded or mis-sized output could invalidate the budget. | Archived closed without narratives. Final task metadata specifies per-pollutant screening dimensions; terminal and interactive output profiles are separate. | Closed | Closed by the tool interface: agent-selected screening files are absent. Server cache/request storage remains a deployment concern. |
| `DEF-73bf987bb03b651d` | A reported resource fault showed unrelated host disk growth could deny queries to an episode that had not used its allowance. | Accounting changed from host growth to service-owned files, retaining scratch/free-space guards. Later audit found that repair alone was incomplete; see the next row. The initial observation has no attempt ID. | Closed | Legacy retained-screening accounting is closed by the tool interface. Shared server cache capacity and host availability still apply. |
| `DEF-446cbd1c83f7fdd8`, `DEF-85ab2652e1b5db57` | Change audit and review found service-only accounting omitted delivered files, allowing retained results to exceed the promised allocation. | Closure narratives record retained-inode accounting through Unix descriptor transfer, including recovery and whole-request reservation checks. | Closed | Closed by the tool interface for delivered screening files; server resource limits still need their own audit. |
| `DEF-91411a2cf9bb3d7a` | Source check rejected general-purpose socket imports in an offline harness. This was an isolation/authoring rule, not a demonstrated paid bypass. | A compiled Unix-only transport replaced Python socket imports. | Closed | The legacy offline transport is absent. V2 intentionally uses HTTP/MCP; security depends on deployment/session boundaries. |

## Precision and rounding edge cases

| Finding | What could go wrong / how found | V1 closure | Status | V2 status |
|---|---|---|---|---|
| `DEF-a137ec51f5bca855`, `DEF-8e878f2a90268e4e` | Red-team probes and an audit found exact ledger cost could describe a shorter stack than the rounded native model, paying for a modeled change that never happened. | Closure narrative on the probe finding records rejection of unrepresentable heights. The related audit is archived closed without a narrative; final native checks require exact height consistency and representability. | Closed | Guarded by [test_validate.py](../envs/permit/core/tests/test_validate.py), `test_hostile_heights` and `test_optional_height_menu_preserves_continuous_v1`; tool validation additionally requires menu membership. |
| `DEF-99a89d8411197410` | Review found emission ties rounded half-even without disclosure, causing otherwise plausible designs to fail consistency checks. | Instructions/metadata publish the convention; shared validation and service explicitly use it. | Closed | Renderer-owned rate quantization closes agent spelling ambiguity; [test_render.py](../envs/permit/core/tests/test_render.py) and [test_statistics.py](../envs/permit/core/tests/test_statistics.py) cover rendering/statistical rounding. |
| `DEF-e3aa3082828280f7` | Review found float spelling dropped trailing zeros, making some native templates too coarse for valid published minimum heights. | Templates render heights to fixed two-decimal precision with matching metadata/instructions. | Closed | Guarded by [test_render.py](../envs/permit/core/tests/test_render.py), `test_complete_draft_render`, and core height validation. |
| `DEF-032651974e6e61c8` | Review found metadata still promised rounded height equality after the grader changed to exact equality. | Metadata now requires exact heights and limits half-even rounding to rates/downwash. | Closed | Closed by the tool interface for native field spelling; exact validation remains in [validate.py](../envs/permit/core/permit_core/validate.py). |
| `DEF-bc74d123991df9f1` | Review found the verifier graded annual/short-period statistics unavailable from the interactive binary, leaving the agent unable to observe the graded quantity. | Both builds now use the collector; the service requires and delivers the documented output. | Closed | Guarded for forms/all receptors by [test_statistics.py](../envs/permit/core/tests/test_statistics.py); native submission remains authoritative. |

## Supporting task-contract findings

These are included for completeness; none independently demonstrates an agent
earning reward without a feasible design.

| Finding | What could go wrong / how found | V1 closure | Status | V2 status |
|---|---|---|---|---|
| `DEF-21eab99d58d50646` | Schema review found nonrelative required paths and missing required baseline declarations. | Archived closed; no closure narrative recorded. | Closed | Closed by the tool interface for submission path declarations; baseline reporting remains a requirement. |
| `DEF-042848bd3c394de2` | Review found the graded stack-cost formula hidden from the agent. | Archived closed without a narrative. Final v1 instructions publish the cost formula. | Closed | Public scenario and free cost checks expose pricing; [test_cost.py](../envs/permit/core/tests/test_cost.py) guards Decimal cost. |
| `DEF-7a36902312e8378d` | Review found mandatory regulatory documents had no declared source and could block all releases. | Archived closed; no closure narrative recorded. | Closed | Still applies as a provenance/documentation obligation. [CONTRACT.md](../CONTRACT.md) states benchmark forms; it makes no promise of a bundled legal library. |
| `DEF-d2271d76670ea35b` | Review found proposed source-count/admission/tier selection predicates missing from selection rules. | Archived closed; no closure narrative recorded. | Closed | Still applies to interpreting the fixed dataset. ARM2 throughout is an explicit v2 task change, not preservation of the old tier quota. |
| `DEF-21f1424769d1dfec` | Review found the reference required preexisting pollutant cases unavailable in the source product. | Archived closed without a narrative. Final binding/reference derive cases from trusted task data. | Closed | Closed by the tool interface and environment renderer; [test_render.py](../envs/permit/core/tests/test_render.py) covers complete case generation. |
| `DEF-1402eb48cc924e64` | Review found required chemistry inputs absent from the visible inventory. | Archived closed without a narrative. Final v1 task data carries chemistry provenance and locked inputs. | Closed | Still applies to physical-definition clarity. V2 uses ARM2 and explicitly treats ozone as unused provenance. |
| `DEF-5c4cb13aa4531e9c` | Review found ARM2 chemistry provenance presented as though it were modeled ozone input. | Instructions now distinguish modeled chemistry from provenance. | Closed | Guarded for tier cards by [test_render.py](../envs/permit/core/tests/test_render.py), `test_tier_cards`; provenance distinction remains documented. |
| `DEF-445b47eb17ffc576` | Build review found an arbitrary seed-zero guard broke the binding protocol. | Seeds now remain provenance without varying physical task content. | Closed | Still applies as provenance correctness; v2 reuses existing tasks without regeneration. |

## V2 audit

The completed coherence review found ordinary tool-based optimization working in
its saved sample, including native submission, invalid-is-free checks, budget
exhaustion and terminal submission behavior. It did **not** certify adversarial
transport or deployment safety. This ledger records its conclusions; it does not
rerun native models or claim the findings were repaired. The reward row is
updated to the current saved-fraction contract; the other rows remain historical.

| Surface | Review finding | Disposition recorded by that review |
|---|---|---|
| Reset disclosure | Reset returns full geometry, including receptor coordinates, beyond the review's narrower allowed initial view. Coordinates are not concentration fields or a hidden best design. | Closed: reset returns only the public scenario; geometry stays in the private record (CONTRACT.md, `test_public_boundary.py`). |
| Ordinary successful tool results | No observed pre-submit baseline/search or private field-array disclosure. | Limited historical observation and static allowlist review; not a fresh boundary audit. |
| Error details | Raw native/exception details may reveal backend paths or implementation information. | Open static risk; not dynamically induced. |
| Sessions and shared playground | Ordinary independent MCP sessions worked; that did not establish shared playground isolation or resistance to hostile session access. | Untested deployment boundary. |
| Request/history size | No application-level bound was established for request/record retention. | Open defensive resource risk; no exhaustion probe was run. |
| Reset and disconnect | Reset starts a new budget as documented; scored-attempt harnesses must own reset. At that time disconnect did not finalize the private no-submit record. | Closed: disconnect now finalizes `no_submit` with zero reward (CONTRACT.md). |
| Native authority | Sampled submitted fast/native standard summaries agreed; final grades used native verification. | Sample evidence, not a proof over all designs or a new fidelity measurement. |
| Reward (current saved-fraction rule) | A feasible graded submission earns `(draft_cost - submitted_cost) / draft_cost`; invalid, infeasible and missing submissions earn zero. Raw savings retain the failure sentinel. | The old coherence receipts predate this rule. Current arithmetic and failure gating are covered by [test_reward.py](../envs/permit/core/tests/test_reward.py) and [repeat/report tests](../eval/tests/test_saved_fraction_repeats.py). |

The review source is the local coherence report and its saved tool receipts.
This is a review snapshot. Concurrent application changes and newly added
regressions have not been revalidated here; this table must not be read as a
fresh audit of those changes. Current reward checks (no model episodes):

```sh
uv run --directory envs/permit/core --no-sync pytest tests/test_reward.py -q
uv run --project envs/permit/openenv --no-sync python -m pytest eval/tests/test_saved_fraction_repeats.py -q
```

## Evaluation audit: final-medium3

Canonical completed episodes and their append-only transcripts were scanned once each; compatibility snapshots and successful-attempt capture copies are not double-counted. Counts describe observed calls and responses, not inferred intent or proof against untried shortcuts. Reproduce every count in this section (including per-model totals and finding locations) with:

```sh
export PERMIT_FIELDS_CACHE_GB=64
uv run --project envs/permit/openenv --no-sync python -m eval.card final-medium3 --audit
```

Scanned **240 episodes / 240 transcripts**. All completed episodes submitted feasible designs under native verification.

| Outcome | Episodes |
|---|---:|
| submitted | 240 |
| invalid | 0 |
| infeasible | 0 |
| no_submit | 0 |
| turn_limit | 0 |
| error | 0 |
| timeout | 0 |

| Transcript/raw-evidence scan | Count |
|---|---:|
| Tool calls | 5870 |
| Calls outside the four tools | 0 |
| Invalid designs returned by check/evaluate/submit | 10 |
| Draft submissions | 0 |
| Evaluate calls after exhaustion | 0 |
| Repeated identical evaluations (beyond the first per episode) | 6 |
| Proxy rejection markers in GPT CLI events/stderr | 0 |

Design comparisons ignore mapping order, numeric spelling and the `none` control alias. Invalid-design counts include free checks, not just terminal submissions; infeasible exploratory evaluations are ordinary optimization. Repeated evaluations consume budget without improving information and are not evidence of an exploit. Exhaustion is detected from the preceding tool result’s evaluations_left. The proxy scan matches explicit rejection/unadvertised-tool markers; ordinary unstable-feature and temporary-directory warnings are not proxy rejections. Raw GPT CLI execution and MCP events are also checked for calls absent from the transcript. All invalid design findings were E_MENU responses to evaluate_design; none was a terminal submission. Per-event locations and errors are in the command’s findings output.

Lazy-policy rewards on the selected eval tasks (equal task means):

| Model-free policy | Mean saved fraction |
|---|---:|
| draft | 0.000000 |
| cheapest_first_greedy | 0.330772 |
| v1_coordinate_descent | 0.396380 |
| random_30 (mean over 8 seeds) | 0.313358 |
| best of 8 random 30-evaluation runs | 0.455739 |
| search (60 seconds; not proven optimal) | 0.475663 |

These use fast-field selection probes, verified by the saved search `verification_source: fast-field selection probe` and task `grading` labels; they are not native grades. Random failures count as zero, random_30 averages eight seeds within each task, and best-of-eight uses eight times that budget. The 60-second search is not proven optimal. Train comparisons are in final-train.md.

Measurement issues and handling:

- The discarded final-medium2 run’s verifier counted long-lived server RSS, including resident memory-mapped fields. Nine environment logs record E_RESOURCE; public final results sanitize this to E_INFRA, so the logs establish the cause. The contemporaneous snapshot was described as 9 of 75; the retained directory contains 80 submissions and the same nine resource failures. None contributes to final-medium3. Commit `348404f` moved verification into a fresh spawned process. The adapter regression `test_verify_adapter_forces_arm2` asserts that spawn context and the ARM2 call.
- A missing/stale selected field pack was caught before episodes at launch. The preflight error named 07b4e4050840-s47; no episode ran in that failed launch.
- Two GPT-6.1 Sol r1 train attempts (6b3a997f0ce8-s47 and 8e62c1a7ddb2-s47) failed before submission. CLI events record incomplete responses/502; upstream SSE names service_unavailable_error. The protocol permitted fresh attempts on resume. Failed attempts remain preserved and are excluded from grades, not silently zeroed.

Reproduce the measurement checks without new physics:

```sh
export PERMIT_FIELDS_CACHE_GB=64
uv run --project envs/permit/openenv --no-sync python -m eval.card final-medium3 --issues
git show --stat 348404f
uv run --project envs/permit/openenv --no-sync python -m pytest envs/permit/openenv/tests/test_adapters.py::test_verify_adapter_forces_arm2 -q
```

This is an observational evaluation audit. It does not close the historical deployment/resource findings above, re-execute adversarial probes, or establish that unobserved shortcut attempts are impossible.

## Regeneration and limits

The extraction command reads the archived environment database and preserved
probe index, checks source hashes, and writes only the public JSON. It refuses
nonempty SQLite journals/WAL and verifies the archive stays unchanged while read.
The machine-specific archive location can be supplied with `--database`; the
default is resolved from the current user's home, without embedding it in output.

```sh
uv run python convert/report_reward_hacks.py
uv run python convert/report_reward_hacks.py --check
uv run pytest tests/test_report_reward_hacks.py -q
```

An extracted source manifest identifies the published v1 revision; it does not
include complete intermediate git history. Defect status/events and final source
support the closure descriptions above. Missing narratives, missing initial
attempt IDs, related findings under distinct fingerprints and unresolved specification conflicts
are retained as recorded. File-shape probes are indexed coverage, not invented
defect IDs. Full historical probe replays and new adversarial execution are outside
this ledger; the model evaluation audit is the section above.
