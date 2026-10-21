# PermitForge evaluation

equal task weight after averaging repeats; station-clustered percentile bootstrap; 2000 draws; seed 0; 95%; no CI below two clusters.

Reward is the saved fraction. Raw USD means include the -1 gate-failure sentinel.

| Model | Tasks | Episodes | Mean saved fraction (95% CI) | Feasible | Fraction of best observed | USD saved | % saved | Evaluations | Turns | USD/episode |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| anthropic:claude-opus-5-5@medium | 10 | 30 | 0.5445 (0.5024, 0.5928) | 100.00% | 0.9457 | 1743551.09 | 54.45 | 19.9667 | 19.9333 | 1.5638 |
| anthropic:claude-sonnet-5-5@medium | 10 | 30 | 0.4886 (0.4338, 0.5441) | 100.00% | 0.8455 | 1559382.60 | 48.86 | 15.0333 | 12.5667 | 0.3743 |
| codex:gpt-5.6-sol@medium | 10 | 30 | 0.5443 (0.4993, 0.5980) | 100.00% | 0.9444 | 1733795.68 | 54.43 | 26.6000 | 29.0667 | n/a |
| codex:gpt-6.1-sol@medium | 10 | 30 | 0.5704 (0.5250, 0.6199) | 100.00% | 0.9898 | 1828112.48 | 57.04 | 27.9667 | 30.0333 | n/a |

| Model | Station | n | Reward | USD saved | Feasible |
|---|---|---:|---:|---:|---:|
| anthropic:claude-opus-5-5@medium | KALB | 1 | 0.4563 | 1395234.35 | 100.00% |
| anthropic:claude-opus-5-5@medium | KAMA | 1 | 0.4591 | 1485504.72 | 100.00% |
| anthropic:claude-opus-5-5@medium | KCRP | 1 | 0.5539 | 2494381.54 | 100.00% |
| anthropic:claude-opus-5-5@medium | KELP | 1 | 0.5420 | 1492494.30 | 100.00% |
| anthropic:claude-opus-5-5@medium | KGRB | 1 | 0.6150 | 2005048.32 | 100.00% |
| anthropic:claude-opus-5-5@medium | KIAD | 1 | 0.5426 | 1490018.26 | 100.00% |
| anthropic:claude-opus-5-5@medium | KMSP | 1 | 0.7223 | 2110431.53 | 100.00% |
| anthropic:claude-opus-5-5@medium | KPIT | 1 | 0.5293 | 1382135.32 | 100.00% |
| anthropic:claude-opus-5-5@medium | KSHV | 1 | 0.4968 | 1574363.50 | 100.00% |
| anthropic:claude-opus-5-5@medium | KTUS | 1 | 0.5281 | 2005899.05 | 100.00% |
| anthropic:claude-sonnet-5-5@medium | KALB | 1 | 0.3255 | 995150.88 | 100.00% |
| anthropic:claude-sonnet-5-5@medium | KAMA | 1 | 0.4615 | 1493437.78 | 100.00% |
| anthropic:claude-sonnet-5-5@medium | KCRP | 1 | 0.4908 | 2210128.90 | 100.00% |
| anthropic:claude-sonnet-5-5@medium | KELP | 1 | 0.4776 | 1315220.89 | 100.00% |
| anthropic:claude-sonnet-5-5@medium | KGRB | 1 | 0.5220 | 1701938.10 | 100.00% |
| anthropic:claude-sonnet-5-5@medium | KIAD | 1 | 0.5018 | 1377897.58 | 100.00% |
| anthropic:claude-sonnet-5-5@medium | KMSP | 1 | 0.6971 | 2036702.30 | 100.00% |
| anthropic:claude-sonnet-5-5@medium | KPIT | 1 | 0.5147 | 1344057.42 | 100.00% |
| anthropic:claude-sonnet-5-5@medium | KSHV | 1 | 0.4427 | 1403094.11 | 100.00% |
| anthropic:claude-sonnet-5-5@medium | KTUS | 1 | 0.4518 | 1716198.09 | 100.00% |
| codex:gpt-5.6-sol@medium | KALB | 1 | 0.4408 | 1347583.03 | 100.00% |
| codex:gpt-5.6-sol@medium | KAMA | 1 | 0.4733 | 1531612.09 | 100.00% |
| codex:gpt-5.6-sol@medium | KCRP | 1 | 0.5042 | 2270903.88 | 100.00% |
| codex:gpt-5.6-sol@medium | KELP | 1 | 0.5320 | 1465071.83 | 100.00% |
| codex:gpt-5.6-sol@medium | KGRB | 1 | 0.6175 | 2013091.67 | 100.00% |
| codex:gpt-5.6-sol@medium | KIAD | 1 | 0.5590 | 1534992.65 | 100.00% |
| codex:gpt-5.6-sol@medium | KMSP | 1 | 0.7363 | 2151156.99 | 100.00% |
| codex:gpt-5.6-sol@medium | KPIT | 1 | 0.5283 | 1379382.75 | 100.00% |
| codex:gpt-5.6-sol@medium | KSHV | 1 | 0.5565 | 1763586.86 | 100.00% |
| codex:gpt-5.6-sol@medium | KTUS | 1 | 0.4951 | 1880575.07 | 100.00% |
| codex:gpt-6.1-sol@medium | KALB | 1 | 0.4646 | 1420397.84 | 100.00% |
| codex:gpt-6.1-sol@medium | KAMA | 1 | 0.4694 | 1518745.20 | 100.00% |
| codex:gpt-6.1-sol@medium | KCRP | 1 | 0.5957 | 2682601.87 | 100.00% |
| codex:gpt-6.1-sol@medium | KELP | 1 | 0.5716 | 1573971.11 | 100.00% |
| codex:gpt-6.1-sol@medium | KGRB | 1 | 0.6362 | 2074263.03 | 100.00% |
| codex:gpt-6.1-sol@medium | KIAD | 1 | 0.5794 | 1591051.92 | 100.00% |
| codex:gpt-6.1-sol@medium | KMSP | 1 | 0.7421 | 2168345.14 | 100.00% |
| codex:gpt-6.1-sol@medium | KPIT | 1 | 0.5377 | 1403974.73 | 100.00% |
| codex:gpt-6.1-sol@medium | KSHV | 1 | 0.5681 | 1800308.58 | 100.00% |
| codex:gpt-6.1-sol@medium | KTUS | 1 | 0.5391 | 2047465.34 | 100.00% |

anthropic:claude-opus-5-5@medium: ended without submission 0; categories `{"submitted": 30}`; coverage 10 station clusters; GRPO signal 1.0000 over 10 repeated tasks; mean pairwise spread 0.0321; best-observed reference available for 10 tasks.


anthropic:claude-sonnet-5-5@medium: ended without submission 0; categories `{"submitted": 30}`; coverage 10 station clusters; GRPO signal 1.0000 over 10 repeated tasks; mean pairwise spread 0.0335; best-observed reference available for 10 tasks.


codex:gpt-5.6-sol@medium: ended without submission 0; categories `{"submitted": 30}`; coverage 10 station clusters; GRPO signal 1.0000 over 10 repeated tasks; mean pairwise spread 0.0254; best-observed reference available for 10 tasks.


codex:gpt-6.1-sol@medium: ended without submission 0; categories `{"submitted": 30}`; coverage 10 station clusters; GRPO signal 0.9000 over 10 repeated tasks; mean pairwise spread 0.0076; best-observed reference available for 10 tasks.


| Pair (left minus right) | Matched tasks | USD saved difference | Reward difference |
|---|---:|---:|---:|
| anthropic:claude-opus-5-5 minus anthropic:claude-sonnet-5-5 | 10 | 184168.4856 | 0.0560 |
| anthropic:claude-opus-5-5 minus codex:gpt-5.6-sol | 10 | 9755.4072 | 0.0002 |
| anthropic:claude-opus-5-5 minus codex:gpt-6.1-sol | 10 | -84561.3858 | -0.0258 |
| anthropic:claude-sonnet-5-5 minus codex:gpt-5.6-sol | 10 | -174413.0785 | -0.0557 |
| anthropic:claude-sonnet-5-5 minus codex:gpt-6.1-sol | 10 | -268729.8715 | -0.0818 |
| codex:gpt-5.6-sol minus codex:gpt-6.1-sol | 10 | -94316.7930 | -0.0261 |

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
| `6b3a997f0ce8-s47` | KALB | 6 | gas_boilers | 46.41 / 44.36 / 46.13 | 31.67 / 31.77 / 34.20 | 46.09 / 42.52 / 43.62 | 46.58 / 46.58 / 46.21 | 18.84 | 29.20 | [46.58 (codex:gpt-6.1-sol r1)](https://huggingface.co/datasets/Attoloop/PermitForge/blob/main/rollouts/final-medium3/codex_gpt-6.1-sol-cbf796da1b7a/6b3a997f0ce8-s47/r1/episode.json) |
| `2298ee8bf485-s47` | KAMA | 6 | process_heaters | 45.19 / 45.93 / 46.61 | 45.93 / 45.93 / 46.61 | 47.36 / 48.03 / 46.61 | 46.61 / 46.66 / 47.54 | 18.24 | 38.15 | [48.03 (codex:gpt-5.6-sol r2)](https://huggingface.co/datasets/Attoloop/PermitForge/blob/main/rollouts/final-medium3/codex_gpt-5.6-sol-5c332fea6027/2298ee8bf485-s47/r2/episode.json) |
| `8e62c1a7ddb2-s47` | KCRP | 6 | kiln | 47.37 / 58.89 / 59.90 | 46.74 / 50.26 / 50.23 | 52.07 / 48.85 / 50.35 | 59.01 / 59.90 / 59.79 | 32.85 | 51.98 | [59.90 (anthropic:claude-opus-5-5 r3)](https://huggingface.co/datasets/Attoloop/PermitForge/blob/main/rollouts/final-medium3/anthropic_claude-opus-5-5-d6e321e50001/8e62c1a7ddb2-s47/r3/episode.json) |
| `b78cb1b23628-s47` | KELP | 6 | small_refinery_mix | 49.72 / 57.13 / 55.75 | 45.99 / 52.32 / 44.97 | 52.52 / 55.85 / 51.24 | 57.41 / 57.41 / 56.65 | 42.71 | 43.93 | [57.41 (codex:gpt-6.1-sol r1)](https://huggingface.co/datasets/Attoloop/PermitForge/blob/main/rollouts/final-medium3/codex_gpt-6.1-sol-cbf796da1b7a/b78cb1b23628-s47/r1/episode.json) |
| `659ceaa85370-s47` | KGRB | 6 | gas_boilers | 60.44 / 63.62 / 60.44 | 49.84 / 63.53 / 43.24 | 61.19 / 63.62 / 60.43 | 63.66 / 63.60 / 63.62 | 31.34 | 48.92 | [63.66 (codex:gpt-6.1-sol r1)](https://huggingface.co/datasets/Attoloop/PermitForge/blob/main/rollouts/final-medium3/codex_gpt-6.1-sol-cbf796da1b7a/659ceaa85370-s47/r1/episode.json) |
| `71191a7c395b-s47` | KIAD | 6 | kiln | 52.83 / 52.90 / 57.05 | 47.08 / 51.72 / 51.72 | 59.20 / 57.12 / 51.37 | 57.49 / 59.20 / 57.12 | 33.10 | 49.56 | [59.20 (codex:gpt-5.6-sol r1)](https://huggingface.co/datasets/Attoloop/PermitForge/blob/main/rollouts/final-medium3/codex_gpt-5.6-sol-5c332fea6027/71191a7c395b-s47/r1/episode.json) |
| `2639cbaab750-s47` | KMSP | 6 | process_heaters | 68.63 / 74.21 / 73.85 | 68.21 / 72.70 / 68.21 | 73.30 / 74.21 / 73.37 | 74.21 / 74.21 / 74.21 | 53.94 | 53.22 | [74.21 (anthropic:claude-opus-5-5 r2)](https://huggingface.co/datasets/Attoloop/PermitForge/blob/main/rollouts/final-medium3/anthropic_claude-opus-5-5-d6e321e50001/2639cbaab750-s47/r2/episode.json) |
| `939fa0273671-s47` | KPIT | 6 | small_refinery_mix | 54.13 / 51.84 / 52.82 | 51.84 / 51.84 / 50.74 | 54.16 / 52.39 / 51.93 | 52.39 / 54.46 / 54.46 | 22.85 | 36.72 | [54.46 (codex:gpt-6.1-sol r2)](https://huggingface.co/datasets/Attoloop/PermitForge/blob/main/rollouts/final-medium3/codex_gpt-6.1-sol-cbf796da1b7a/939fa0273671-s47/r2/episode.json) |
| `fb0cb3a39692-s47` | KSHV | 6 | kiln | 51.57 / 46.18 / 51.29 | 43.19 / 42.73 / 46.90 | 57.44 / 54.21 / 55.30 | 57.76 / 57.76 / 54.90 | 16.23 | 43.96 | [57.76 (codex:gpt-6.1-sol r1)](https://huggingface.co/datasets/Attoloop/PermitForge/blob/main/rollouts/final-medium3/codex_gpt-6.1-sol-cbf796da1b7a/fb0cb3a39692-s47/r1/episode.json) |
| `84c41ee337e2-s47` | KTUS | 6 | process_heaters | 53.33 / 54.56 / 50.54 | 44.23 / 45.66 / 45.66 | 45.61 / 49.43 / 53.49 | 53.55 / 54.76 / 53.41 | 30.23 | 39.58 | [54.76 (codex:gpt-6.1-sol r2)](https://huggingface.co/datasets/Attoloop/PermitForge/blob/main/rollouts/final-medium3/codex_gpt-6.1-sol-cbf796da1b7a/84c41ee337e2-s47/r2/episode.json) |

Best observed is the highest feasible model result across these repeats, never a proven optimum or a reward ceiling. Each best-observed cell links to its design ledger in a canonical episode; the first recorded observation is shown for ties.

## Model-free comparisons

| Model-free policy | Mean saved fraction |
|---|---:|
| draft | 0.000000 |
| cheapest_first_greedy | 0.300338 |
| v1_coordinate_descent | 0.340854 |
| random_30 (mean over 8 seeds) | 0.261913 |
| best of 8 random 30-evaluation runs | 0.435212 |
| search (60 seconds; not proven optimal) | 0.354249 |

The generated baseline-coverage warning refers to absent separate native baseline artifacts; the complete frozen fast-field comparisons are shown above.

## How this was run

The complete final-medium3 run used the frozen model-free cohort from results/subsets.json, four models, medium effort, three repeats, round-robin scheduling, 40 responses per episode, 32,768 output tokens per response, 30 valid evaluations and 1,800 seconds wall clock. The subset file’s four-repeat planning field is superseded by run.json’s three repeats. Claude Opus 5.5 and Sonnet 5.5 used the Anthropic Messages API and our tool loop, output_config.effort=medium and adaptive summarized thinking passed back unchanged. GPT-6.1 Sol and GPT-5.6 Sol used Codex CLI (`codex exec`) on a ChatGPT subscription, model_reasoning_effort=medium, base instructions replaced by the task prompt, only the four PermitForge MCP tools and text.verbosity=low. Claude’s first user message was “Use the permitforge tools. You have 40 turns; submit before the last turn ends.” GPT received the full prompt again plus “Use the permitforge tools. You have 40 turns.” GPT used one tool call per response (parallel_tool_calls=false); Claude could make several. Raw requests, CLI config and argv preserve these differences; this is not an identical-prompt or identical-harness comparison. Codex USD is unknown. Two GPT-6.1 Sol train episodes (6b3a997f0ce8-s47 r1 and 8e62c1a7ddb2-s47 r1) encountered provider service_unavailable_error before submission. Their CLI events report an incomplete response/502; the upstream SSE bodies name the provider error. Both restarted from scratch on resume under the protocol; failed attempts remain under attempts/ and are excluded from grades. Before this run, the v1 verifier’s memory guard counted the whole long-lived eval-server process, including resident memory-mapped fields. Nine submissions in the discarded final-medium2 run failed E_RESOURCE rather than physics (the contemporaneous report said 9 of 75; the retained directory now has 80 submissions, with the same nine failures). Final verification was moved to a fresh spawned process in commit 348404f; regression: envs/permit/openenv/tests/test_adapters.py::test_verify_adapter_forces_arm2. That discarded run is excluded. A field-cache eviction was also caught by preflight before any episode in the failed launch; the preflight error named the missing/stale 07b4e4050840-s47 pack. Final submissions use full native verification; baseline probes use fast fields. V2 changes NO2 to ARM2 and uses menu heights; published v1 scenarios have been public since 2026-10-16.

Anthropic request-budget.json and costs.jsonl agree on $120.510718 for the complete run across both splits; subscription USD is unknown.

The run folder is `rollouts/final-medium3` in the dataset: `uvx --from 'huggingface-hub==1.33.0' hf download Attoloop/PermitForge --repo-type dataset --include 'rollouts/final-medium3/*' --local-dir results` puts it under `results/rollouts/`. Reproduce the generated tables with `PERMIT_FIELDS_CACHE_GB=64 uv run --project envs/permit/openenv --no-sync python -m eval.report final-medium3 --split train --output results/final-train.md`, then append these supplements with `python -m eval.card final-medium3 --write` under the same environment. The card command is offline and idempotent.
