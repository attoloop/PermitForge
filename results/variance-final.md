# Final repeat variance

Saved-fraction units; sample standard deviation (ddof=1) and mean absolute difference over unordered repeat pairs. Models use medium effort. Tasks have equal weight.

| Split | Model | Tasks with 3 repeats | Mean std | Mean pairwise spread | Any spread / repeated tasks (GRPO signal) | Identical repeats |
|---|---|---:|---:|---:|---:|---:|
| eval | anthropic:claude-opus-5-5 | 10 | 0.023196 | 0.029129 | 9/10 (90.0%) | 1 |
| eval | anthropic:claude-sonnet-5-5 | 10 | 0.029027 | 0.036159 | 10/10 (100.0%) | 0 |
| eval | codex:gpt-5.6-sol | 10 | 0.017781 | 0.021953 | 10/10 (100.0%) | 0 |
| eval | codex:gpt-6.1-sol | 10 | 0.002940 | 0.003578 | 5/10 (50.0%) | 5 |
| train | anthropic:claude-opus-5-5 | 10 | 0.026343 | 0.032061 | 10/10 (100.0%) | 0 |
| train | anthropic:claude-sonnet-5-5 | 10 | 0.027204 | 0.033472 | 10/10 (100.0%) | 0 |
| train | codex:gpt-5.6-sol | 10 | 0.019536 | 0.025414 | 10/10 (100.0%) | 0 |
| train | codex:gpt-6.1-sol | 10 | 0.006392 | 0.007582 | 9/10 (90.0%) | 1 |

Any nonzero spread provides a relative reward signal under the repeat definition; its presence does not establish that GRPO will improve a trained model. Identical repeats provide no within-task relative signal. These are three frontier-model repeats, not measurements of a small model’s training groups. Train results describe the selected train cohort; they are separate from held-out station evaluation.

Reproduce: `PERMIT_FIELDS_CACHE_GB=64 uv run --project envs/permit/openenv --no-sync python -m eval.card final-medium3 --write`. Values are recomputed from canonical episodes, matching `variance-eval.json` and `variance-train.json`. The combined `variance.json`/`variance.csv` inside the run are stale round-one outputs and are excluded.
