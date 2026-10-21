# Launch scripts

The shell scripts that ran the Qwen3.8-27B training on HF Jobs. They are small wrappers around
`hf jobs`; the training code itself is in `train/cloud/`. All of them run from the repository root
and need `HF_TOKEN` in the environment. Logs and launch records go to `train/cloud/data/launches/`
(git-ignored); the ones from the published run are in the dataset under `training/logs/`.

| Script | What it does |
|---|---|
| `smoke-attempts.sh ROUND` | Probe for a good host, launch one smoke job, retry only a bad-host start, then watch it |
| `watch-smoke.sh JOB PREFIX` | Poll a smoke job every 2 minutes and append its log and metric changes |
| `run-full.sh RUN ARGS` | One phase of a run: probe, launch with the given `--stage` arguments, guard the start, watch |
| `watch-full.sh JOB PREFIX RUN STAGE PHASE` | Poll a phase job every 5 minutes with progress read from the run bucket |
| `chain-full.sh RUN` | The whole sequence: GRPO in 5-step chunks, base-eval and eval in parallel, export |
| `jobs_ledger.py ORG OUT.json` | Every job of the org with timings and cost, for `eval/learnability.py` |

Why the probe: some H200 hosts start the container without CUDA libraries. A one-second
`python:3.12-slim` job prints whether `libcuda.so.1` is present; a real job launched right after a
good probe takes the same slot. A bad host still costs about a minute, which the start guard
recognises and retries.

Why the chunks: the first 30-step job died at step 15 on the device memory watchdog. Each chunk is
a fresh process that resumes from the last complete checkpoint, so a dead job loses at most five
steps.
