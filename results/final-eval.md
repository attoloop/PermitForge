# PermitForge evaluation

equal task weight after averaging repeats; station-clustered percentile bootstrap; 2000 draws; seed 0; 95%; no CI below two clusters.

Reward is the saved fraction. Raw USD means include the -1 gate-failure sentinel.

| Model | Tasks | Episodes | Mean saved fraction (95% CI) | Feasible | Fraction of best observed | USD saved | % saved | Evaluations | Turns | USD/episode |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| anthropic:claude-opus-5-5@medium | 10 | 30 | 0.5755 (0.5005, 0.6602) | 100.00% | 0.9354 | 1860804.21 | 57.55 | 21.4000 | 21.8333 | 1.7119 |
| anthropic:claude-sonnet-5-5@medium | 10 | 30 | 0.5071 (0.4239, 0.6000) | 100.00% | 0.8151 | 1617124.33 | 50.71 | 16.0667 | 12.7000 | 0.3670 |
| codex:gpt-5.6-sol@medium | 10 | 30 | 0.5858 (0.5186, 0.6634) | 100.00% | 0.9563 | 1889132.04 | 58.58 | 26.4667 | 28.8333 | — |
| codex:gpt-6.1-sol@medium | 10 | 30 | 0.6065 (0.5423, 0.6768) | 100.00% | 0.9942 | 1957808.60 | 60.65 | 24.9667 | 26.9667 | — |

| Model | Station | n | Reward | USD saved | Feasible |
|---|---|---:|---:|---:|---:|
| anthropic:claude-opus-5-5@medium | KABQ | 2 | 0.4719 | 1577898.60 | 100.00% |
| anthropic:claude-opus-5-5@medium | KDDC | 2 | 0.5830 | 2166858.40 | 100.00% |
| anthropic:claude-opus-5-5@medium | KJAX | 2 | 0.7118 | 2206272.26 | 100.00% |
| anthropic:claude-opus-5-5@medium | KOMA | 2 | 0.4826 | 1313881.60 | 100.00% |
| anthropic:claude-opus-5-5@medium | KTOP | 2 | 0.6284 | 2039110.21 | 100.00% |

anthropic:claude-opus-5-5@medium: ended without submission 0; categories `{"submitted": 30}`; coverage 5 station clusters; GRPO signal 0.9000 over 10 repeated tasks; mean pairwise spread 0.0291; best-observed reference available for 10 tasks.

| anthropic:claude-sonnet-5-5@medium | KABQ | 2 | 0.3746 | 1254090.25 | 100.00% |
| anthropic:claude-sonnet-5-5@medium | KDDC | 2 | 0.4979 | 1761365.05 | 100.00% |
| anthropic:claude-sonnet-5-5@medium | KJAX | 2 | 0.6601 | 2040636.12 | 100.00% |
| anthropic:claude-sonnet-5-5@medium | KOMA | 2 | 0.4181 | 1138217.47 | 100.00% |
| anthropic:claude-sonnet-5-5@medium | KTOP | 2 | 0.5849 | 1891312.77 | 100.00% |

anthropic:claude-sonnet-5-5@medium: ended without submission 0; categories `{"submitted": 30}`; coverage 5 station clusters; GRPO signal 1.0000 over 10 repeated tasks; mean pairwise spread 0.0362; best-observed reference available for 10 tasks.

| codex:gpt-5.6-sol@medium | KABQ | 2 | 0.5005 | 1662825.09 | 100.00% |
| codex:gpt-5.6-sol@medium | KDDC | 2 | 0.5941 | 2196403.11 | 100.00% |
| codex:gpt-5.6-sol@medium | KJAX | 2 | 0.7096 | 2197878.25 | 100.00% |
| codex:gpt-5.6-sol@medium | KOMA | 2 | 0.4976 | 1354818.10 | 100.00% |
| codex:gpt-5.6-sol@medium | KTOP | 2 | 0.6269 | 2033735.66 | 100.00% |

codex:gpt-5.6-sol@medium: ended without submission 0; categories `{"submitted": 30}`; coverage 5 station clusters; GRPO signal 1.0000 over 10 repeated tasks; mean pairwise spread 0.0220; best-observed reference available for 10 tasks.

| codex:gpt-6.1-sol@medium | KABQ | 2 | 0.5245 | 1743387.59 | 100.00% |
| codex:gpt-6.1-sol@medium | KDDC | 2 | 0.6272 | 2312492.68 | 100.00% |
| codex:gpt-6.1-sol@medium | KJAX | 2 | 0.7170 | 2220132.20 | 100.00% |
| codex:gpt-6.1-sol@medium | KOMA | 2 | 0.5159 | 1404783.42 | 100.00% |
| codex:gpt-6.1-sol@medium | KTOP | 2 | 0.6478 | 2108247.11 | 100.00% |

codex:gpt-6.1-sol@medium: ended without submission 0; categories `{"submitted": 30}`; coverage 5 station clusters; GRPO signal 0.5000 over 10 repeated tasks; mean pairwise spread 0.0036; best-observed reference available for 10 tasks.


| Pair (left − right) | Matched tasks | USD saved difference | Reward difference |
|---|---:|---:|---:|
| anthropic:claude-opus-5-5 − anthropic:claude-sonnet-5-5 | 10 | 243679.8831 | 0.0684 |
| anthropic:claude-opus-5-5 − codex:gpt-5.6-sol | 10 | -28327.8252 | -0.0102 |
| anthropic:claude-opus-5-5 − codex:gpt-6.1-sol | 10 | -97004.3837 | -0.0310 |
| anthropic:claude-sonnet-5-5 − codex:gpt-5.6-sol | 10 | -272007.7083 | -0.0786 |
| anthropic:claude-sonnet-5-5 − codex:gpt-6.1-sol | 10 | -340684.2668 | -0.0994 |
| codex:gpt-5.6-sol − codex:gpt-6.1-sol | 10 | -68676.5585 | -0.0207 |

Infrastructure attempts (excluded from graded metrics):

| Model | Task | Repeat | Attempt | Retry on resume |
|---|---|---:|---:|---|
| codex:gpt-6.1-sol | 6b3a997f0ce8-s47 | 1 | 1 | yes |
| codex:gpt-6.1-sol | 8e62c1a7ddb2-s47 | 1 | 1 | yes |

Baseline coverage incomplete; see JSON missing_baselines.

## Per-task saved percentages

Each model cell lists r1 / r2 / r3 (% of draft cost saved). Baselines use frozen fast-field selection probes (`verification_source: fast-field selection probe`), not native submission grades. Failed probes score zero. Best-of-eight random uses eight separate 30-evaluation runs (240 evaluations total).

| Task | Station | Sources | Archetype | anthropic:claude-opus-5-5 | anthropic:claude-sonnet-5-5 | codex:gpt-5.6-sol | codex:gpt-6.1-sol | Greedy % | Best-of-8 random % | Best observed model % |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `d50e1051dcec-s47` | KABQ | 6 | kiln | 57.89 / 59.52 / 56.20 | 52.19 / 39.25 / 48.14 | 55.02 / 60.44 / 57.82 | 61.44 / 61.44 / 59.52 | 23.98 | 45.39 | [61.44 (codex:gpt-6.1-sol r1)](rollouts/final-medium3/codex_gpt-6.1-sol-cbf796da1b7a/d50e1051dcec-s47/r1/episode.json) |
| `e1375aee7217-s47` | KABQ | 5 | process_heaters | 36.37 / 34.57 / 38.60 | 28.33 / 28.44 / 28.39 | 44.40 / 43.57 / 39.04 | 43.67 / 44.21 / 44.40 | 17.03 | 26.30 | [44.40 (codex:gpt-6.1-sol r3)](rollouts/final-medium3/codex_gpt-6.1-sol-cbf796da1b7a/e1375aee7217-s47/r3/episode.json) |
| `93b53359f60e-s47` | KDDC | 6 | gas_boilers | 59.52 / 64.48 / 59.65 | 44.67 / 48.36 / 44.86 | 63.22 / 59.65 / 61.76 | 64.37 / 64.61 / 64.61 | 30.93 | 48.11 | [64.61 (codex:gpt-6.1-sol r2)](rollouts/final-medium3/codex_gpt-6.1-sol-cbf796da1b7a/93b53359f60e-s47/r2/episode.json) |
| `9b1cffc36b20-s47` | KDDC | 5 | kiln | 54.71 / 55.71 / 55.71 | 52.71 / 54.13 / 54.01 | 56.67 / 57.53 / 57.66 | 60.96 / 60.96 / 60.84 | 18.80 | 44.82 | [60.96 (codex:gpt-6.1-sol r1)](rollouts/final-medium3/codex_gpt-6.1-sol-cbf796da1b7a/9b1cffc36b20-s47/r1/episode.json) |
| `4528c31805f3-s47` | KJAX | 5 | process_heaters | 72.53 / 72.53 / 72.53 | 63.10 / 67.50 / 67.41 | 72.53 / 72.53 / 70.70 | 72.53 / 72.53 / 72.53 | 72.53 | 59.41 | [72.53 (anthropic:claude-opus-5-5 r1)](rollouts/final-medium3/anthropic_claude-opus-5-5-d6e321e50001/4528c31805f3-s47/r1/episode.json) |
| `5a86424e2861-s47` | KJAX | 6 | simple_cycle_turbine | 69.37 / 70.87 / 69.22 | 60.97 / 68.60 / 68.50 | 70.87 / 69.93 / 69.21 | 70.87 / 70.87 / 70.87 | 29.47 | 56.76 | [70.87 (anthropic:claude-opus-5-5 r2)](rollouts/final-medium3/anthropic_claude-opus-5-5-d6e321e50001/5a86424e2861-s47/r2/episode.json) |
| `07b4e4050840-s47` | KOMA | 6 | simple_cycle_turbine | 63.25 / 57.73 / 64.24 | 64.24 / 57.28 / 54.85 | 64.24 / 64.24 / 63.25 | 63.25 / 63.25 / 63.25 | 34.70 | 43.21 | [64.24 (anthropic:claude-opus-5-5 r3)](rollouts/final-medium3/anthropic_claude-opus-5-5-d6e321e50001/07b4e4050840-s47/r3/episode.json) |
| `c90f3d90f654-s47` | KOMA | 5 | gas_boilers | 35.31 / 40.30 / 28.72 | 27.62 / 23.45 / 23.45 | 35.15 / 36.12 / 35.58 | 40.39 / 38.53 / 40.88 | 15.60 | 31.78 | [40.88 (codex:gpt-6.1-sol r3)](rollouts/final-medium3/codex_gpt-6.1-sol-cbf796da1b7a/c90f3d90f654-s47/r3/episode.json) |
| `660f38096246-s47` | KTOP | 5 | small_refinery_mix | 67.77 / 70.89 / 70.89 | 66.31 / 66.24 / 66.31 | 70.48 / 68.01 / 70.89 | 70.89 / 70.89 / 70.89 | 64.40 | 54.91 | [70.89 (anthropic:claude-opus-5-5 r2)](rollouts/final-medium3/anthropic_claude-opus-5-5-d6e321e50001/660f38096246-s47/r2/episode.json) |
| `6d3f0a5005fd-s47` | KTOP | 6 | kiln | 58.68 / 51.14 / 57.67 | 56.38 / 45.97 / 49.74 | 58.68 / 49.40 / 58.68 | 58.68 / 58.68 / 58.68 | 23.33 | 45.04 | [58.68 (anthropic:claude-opus-5-5 r1)](rollouts/final-medium3/anthropic_claude-opus-5-5-d6e321e50001/6d3f0a5005fd-s47/r1/episode.json) |

Best observed is the highest feasible model result across these repeats, never a proven optimum or a reward ceiling. Each best-observed cell links to its design ledger in a canonical episode; the first recorded observation is shown for ties.

## Model-free comparisons

| Model-free policy | Mean saved fraction |
|---|---:|
| draft | 0.000000 |
| cheapest_first_greedy | 0.330772 |
| v1_coordinate_descent | 0.396380 |
| random_30 (mean over 8 seeds) | 0.313358 |
| best of 8 random 30-evaluation runs | 0.455739 |
| search (60 seconds; not proven optimal) | 0.475663 |

The generated baseline-coverage warning refers to absent separate native baseline artifacts; the complete frozen fast-field comparisons are shown above.

## How this was run

The complete final-medium3 run used the frozen model-free cohort from results/subsets.json, four models, medium effort, three repeats, round-robin scheduling, 40 responses per episode, 32,768 output tokens per response, 30 valid evaluations and 1,800 seconds wall clock. The subset file’s four-repeat planning field is superseded by run.json’s three repeats. Claude Opus 5.5 and Sonnet 5.5 used the Anthropic Messages API and our tool loop, output_config.effort=medium and adaptive summarized thinking passed back unchanged. GPT-6.1 Sol and GPT-5.6 Sol used Codex CLI (`codex exec`) on a ChatGPT subscription, model_reasoning_effort=medium, base instructions replaced by the task prompt, only the four PermitForge MCP tools and text.verbosity=low. Claude’s first user message was “Use the permitforge tools. You have 40 turns; submit before the last turn ends.” GPT received the full prompt again plus “Use the permitforge tools. You have 40 turns.” GPT used one tool call per response (parallel_tool_calls=false); Claude could make several. Raw requests, CLI config and argv preserve these differences; this is not an identical-prompt or identical-harness comparison. Codex USD is unknown. Two GPT-6.1 Sol train episodes (6b3a997f0ce8-s47 r1 and 8e62c1a7ddb2-s47 r1) encountered provider service_unavailable_error before submission. Their CLI events report an incomplete response/502; the upstream SSE bodies name the provider error. Both restarted from scratch on resume under the protocol; failed attempts remain under attempts/ and are excluded from grades. Before this run, the v1 verifier’s memory guard counted the whole long-lived eval-server process, including resident memory-mapped fields. Nine submissions in the discarded final-medium2 run failed E_RESOURCE rather than physics (the contemporaneous report said 9 of 75; the retained directory now has 80 submissions, with the same nine failures). Final verification was moved to a fresh spawned process in commit 348404f; regression: envs/permit/openenv/tests/test_adapters.py::test_verify_adapter_forces_arm2. That discarded run is excluded. A field-cache eviction was also caught by preflight before any episode in the failed launch; the preflight error named the missing/stale 07b4e4050840-s47 pack. Final submissions use full native verification; baseline probes use fast fields. V2 changes NO2 to ARM2 and uses menu heights; published v1 scenarios have been public since 2026-10-16.

Anthropic request-budget.json and costs.jsonl agree on $120.510718 for the complete run across both splits; subscription USD is unknown.

Reproduce the generated tables with `PERMIT_FIELDS_CACHE_GB=64 uv run --project envs/permit/openenv --no-sync python -m eval.report final-medium3 --split eval --output results/final-eval.md`, then append these supplements with `python -m eval.card final-medium3 --write` under the same environment. The card command is offline and idempotent.
