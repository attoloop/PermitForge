# Saved-fraction evaluation and repeats

[PROTOCOL.md](PROTOCOL.md) owns cohort, effort, scheduling and budget.
The historical pilot protocol is preserved in [PROTOCOL-pilot.md](PROTOCOL-pilot.md).
The prompt stays unchanged. The metrics below remain authoritative.

Reward is feasible USD/year saved divided by draft cost; draft, invalid,
infeasible and no submission score zero. There are no calibration runs or
anchors. USD/year savings retain the historical -1 sentinel for gate failures.
`rescore.py` independently re-verifies stored submissions and applies saved
fraction. It preserves previous grades; infrastructure failures retain the old
grade. This is native work and is not required to regenerate reports.

## Cohorts and storage

`results/subsets.json` contains the model-free difficulty probes, selection
rule, and 10 tasks per split. `eval/select_tasks.py` rebuilds the selection from
frozen probes with `--report-only`. The earlier selector in `eval/subsets.py` is
kept for the record and refuses the active output path. The launcher rejects historical or malformed
cohorts before provider preflight.

New episodes use `<run>/<model_slug>/<task_id>/r<k>/episode.json`, with full capture
and attempts beside each repeat. `r<k>.json` remains a compatibility snapshot;
readers prefer the canonical record. Round-robin scheduling finishes every task
in a repeat before advancing, with eval before train. Resume skips all terminal
outcomes and rejects changed sampling, effort, protocol, cohort or scheduling.

Historical imports require identical protocol hash, prompt hash/version and
sampling settings. Pilot episodes therefore cannot enter the final panel.
Existing pilot artifacts remain read-only. Regrading historical records retains
the original grade and provenance; it never changes their original sampling.

## Metrics and plotting

Per model and task: repeat count, mean, sample standard deviation (ddof=1),
minimum, maximum and mean absolute difference over all unordered repeat pairs
of saved fraction, plus feasible rate. Singleton standard deviation and pairwise
spread are null. “GRPO signal” is the share of tasks with at least two completed
repeats whose maximum exceeds their minimum; its denominator is reported.
Failures contribute zero saved fraction. Missing/infrastructure-only episodes
are not silently graded zero; repeat counts expose incomplete coverage.

Headline mean saved fraction and feasible rate average repeats within each task,
then average task means equally. The 95% CI resamples whole stations over those
task means (2000 draws, seed 0); repeated episodes are never treated as independent
tasks. Fewer than two stations yields no CI. Paired comparisons also use matched
task means. USD/episode averages measured episode charges; any unknown charge
makes that mean null, with known-charge coverage reported.

`best_observed.py` reports each task's highest feasible saved fraction across the
provided model runs. It never writes task records. Fraction of best observed is
`task mean / task best observed`, averaged over tasks with a positive reference;
zero-best tasks have a null ratio and are excluded with explicit coverage. Search
and baseline results do not set this reference. It is a cohort-relative secondary
metric, never a reward, and may change as additional model observations are added.

`run_eval` writes `board.json`, `board.md`, `variance.json` and `variance.csv` beside
`index.json`. Standalone reporting writes `<output-stem>-variance.json/.csv` with
one row per model/task for plotting. JSON has numeric values; CSV leaves nulls
blank. Both scripts are read-only with respect to input rollouts and task packs:

```sh
uv run --project envs/permit/openenv --no-sync python -m eval.report repeated-eval \
  --output results/repeated-eval-board.md
uv run --project envs/permit/openenv --no-sync python -m eval.best_observed repeated-eval \
  --output results/repeated-eval-best.json
```

Report train and eval separately. Duplicate `(model, task_id, repeat)` identities
are rejected rather than pooled as selective reruns. A full native rescore remains
available through `python -m eval.rescore RUN`; this command launches verification.
