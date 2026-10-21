# Lossless v1 extraction

Set `V1_FAMILY` to the preserved v1 family directory and `V1_RUNS` to its run evidence directory.
Run from the repository root. Extraction uses the Python standard library, reads
both inputs, and requires a new or empty output directory. It never modifies the
source or run directories. Existing archives are deliberately not overwritten;
use a different `--out` path to repeat extraction.

```bash
uv run python convert/extract_v1.py \
  --src "$V1_FAMILY" \
  --runs "$V1_RUNS" \
  --out convert/v1-src
uv run python convert/verify_extract.py \
  --out convert/v1-src \
  --src "$V1_FAMILY"
```

After the original source is evicted, destination integrity remains checkable:

```bash
uv run python convert/verify_extract.py --out convert/v1-src
```

`EXTRACT_MANIFEST.json` records the source git HEAD, origin and branch, each
regular source file's relative path, size, SHA-256 and destination, copied run
evidence, generated probe-index integrity, variant membership and skipped paths
with reasons. `root` distinguishes `src` from `runs`. Verification with `--src`
independently inventories the source and detects missing mappings as well as
missing or changed source and destination files. An unreadable regular source
file prevents a successful lossless check.

Archive layout:

- `family/`: root files plus additional family content such as `template/`.
- `shared/`: identical instance code at its original relative path, and shared
  documentation under `shared/docs/`.
- `variants/<relative path>/<sha12>`: every content variant when code differs;
  instance membership is in the manifest's `variants` mapping.
- `instances/<id>/`: instance metadata, public `task/` and `verifier-task/`,
  without their deduplicated met and docs. Unexpected instance files are kept
  at their original instance-relative paths too.
- `met/<STATION>/`: station files, checked for identical names and bytes across
  all instances and both public and verifier tasks.
- `evidence/`: only the requested run subtrees and suffixes within the size
  limit; `adversarial_probes.json` holds names only. Probe contents are never
  traversed. Run paths outside the requested subtrees are not inventoried.

Traversal uses `lstat`/`scandir` without following symlinks. Files are opened
only after regular-file and readability checks, with `O_NOFOLLOW` and
`O_NONBLOCK`, followed by an `fstat` identity check. Special files, symlinks,
unreadable entries, excluded suffixes and oversized evidence files are listed
in `skipped`. Hardlink identity caches avoid repeatedly hashing the same
unchanged inode; output files are independent copies, with their bytes checked
immediately. Extracted code remains byte-for-byte archival data; the extraction does
not lift it into an importable package or alter its headers.

Reproduce the archive counts, split counts directly from the copied family
manifest, and the total regular-file size of OUT (including the extraction
manifest):

```bash
uv run python - <<'PY'
from collections import Counter
import json
from pathlib import Path
out = Path('convert/v1-src')
m = json.loads((out / 'EXTRACT_MANIFEST.json').read_text())
rows = json.loads((out / 'family/manifest.json').read_text())['instances']
print(json.dumps(m['summary'], sort_keys=True))
print('family instances:', len(rows))
print('family stations:', len({r['equivalence_key'] for r in rows}))
print('family splits:', dict(Counter(t.removeprefix('split:') for r in rows
    for t in r['coverage_tags'] if t.startswith('split:'))))
print('OUT bytes:', sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
print('skip reasons:', dict(Counter(r['reason'] for r in m['skipped'])))
print('skipped source files:', sum(r['root'] == 'src' for r in m['skipped']))
print('source git:', m['source_git'])
PY
```

Run the full root-package test suite and lint the tooling:

```bash
uv run pytest -q
uv run --project envs/permit/core ruff check convert tests
```

`convert/v1-src/` is intentionally gitignored. Keep the archive on disk after
source eviction; the tooling and report alone do not contain the extracted data.

## V2 task conversion and ARM2 draft verification

From the repository root, convert the preserved archive in one command. The
output folder must be new or empty, so existing eligibility records are never
reset; the tracked records in `envs/permit/tasks` were produced this way:

```sh
uv run python convert/v1_to_v2.py --src convert/v1-src --out convert/cache/rebuilt-tasks
```

`tasks/data/<instance_id>/` contains the original scenario, costs, backgrounds,
geometry, receptors and draft files, with v2 metadata added to `task.json`.
`design_value_requirements` aliases the published `scenario.naaqs_thresholds`;
`gep_method` is copied from the scenario. Facility archetype comes from
`scenario.facility.archetype`, including refineries with mixed source types.
Native locked tables, draft templates and historical draft statistics stay as
published and are labelled accordingly. The v2 physical check explicitly calls
`verify(..., no2_tier="ARM2")`; preserved ozone is marked unused by ARM2.
`permit_core.model.load_v2_instance(tasks_root, id)` loads this bundle and its
menu with exact Decimal values, using the shared station met directory.

Height menus contain the published endpoints plus six interior seventh-points,
rounded to the nearest half metre using Decimal half-even ties. Interior points
outside the open interval or repeated after rounding are dropped; endpoints
are retained exactly. Degenerate intervals contain just the endpoint.

Data and shared met are gitignored. The root and split manifests, JSONL indices
and decision logs are reviewable repository artifacts; rows record all data-file
and met-file SHA-256 hashes. The legacy index points to the same preserved
input bundle, selects the published NO2 cards and continuous heights, and stays
independent of v2 admission. It is only for the later before/after comparison;
it is not a separate copy of the old Harbor task or a verified v1 result.

Check every task's complete file inventory and shared met hashes:

```sh
uv run python convert/v1_to_v2.py --out envs/permit/tasks --check
```

Run a native sample, then apply its cached results to the indices:

```sh
uv run --project envs/permit/core python convert/reverify_drafts.py \
  --tasks envs/permit/tasks --jobs 3 \
  --only 003bd328a392-s47 01addc2a86f1-s47 0227870eb587-s47
uv run --project envs/permit/core python convert/reverify_drafts.py \
  --tasks envs/permit/tasks --finalize
```

The sample covers published PVMRM and ARM2 cards; `01addc2a86f1-s47` has six
sources. Per-task results, including binding design values, margins, native wall
seconds, errors and binary hashes, live at `convert/cache/drafts-arm2/<id>.json`.
Each process-pool worker calls the verifier sequentially in a private temporary
directory: at most `--jobs` native processes run at once; accepted jobs are 1-4.
Results are written atomically, and matching completed tasks are skipped.
Changed inputs or corrupt/stale cache entries fail explicitly; remove only the
affected result to request a rerun. A failed interrupted task has no result and
is retried. `--cache PATH` permits isolated runs and tests. Do not run two
writers against the same task pack/cache concurrently.

`--finalize` launches no models, applies all available matching results, leaves
missing results pending, updates counts/reasons/failing standards and appends
decisions only when eligibility changes. Repeating it is idempotent. It also
checks current data integrity. A result with a native/verification error is
ineligible with that error recorded; inspect infrastructure errors before
interpreting any failure as a physics change. The global failure fraction uses
all original tasks as its denominator. More than ten percent ineligible stops
further task scheduling and returns exit code 2 for inspection (work
already running is allowed to finish). Finalize prints the same decision flag.

To recheck every draft, run:

```sh
uv run --project envs/permit/core python convert/reverify_drafts.py \
  --tasks envs/permit/tasks --jobs 3
uv run --project envs/permit/core python convert/reverify_drafts.py \
  --tasks envs/permit/tasks --finalize
```

Run the complete tooling and core suites (native core tests included):

```sh
uv run pytest -q
uv run --directory envs/permit/core pytest -q
uv run --project envs/permit/core ruff check convert tests
uv run --directory envs/permit/core ruff check .
```

## Task-major baselines and search

Run from the repository root. Without `--fields`, each worker acquires a full
float64 pack through `ensure_pack`, runs the public policies and search, then
releases it. `--evict-after` also requests immediate deletion after release;
another active lease prevents deletion. `PERMIT_FIELDS_CACHE_GB` sets the LRU
cap (default 40 GiB); active packs can temporarily exceed it. Existing legacy
gate directories count against this cap and remain pinned until adopted.

```sh
uv run --project envs/permit/core python convert/run_baselines.py \
  --tasks envs/permit/tasks --jobs 4 --field-jobs 1 --evict-after \
  --seconds 60 --max-nodes 10000 --cache convert/cache/baselines
uv run --project envs/permit/core python convert/run_baselines.py \
  --tasks envs/permit/tasks --cache convert/cache/baselines --report
```

The defaults are a search deadline of 60 seconds and 10000 nodes. Final native
verification is outside that search deadline; `closed: false` remains explicit.
The library retains unlimited search by default, and the runner's `--unlimited`
removes both limits. Worker count times field parallelism must not exceed twelve.
Use fewer workers when unrelated native jobs are running. `--fields PATH` keeps
support for explicitly supplied `.npz` packs and reports missing packs instead
of computing them.

Checkpoints resume only with matching inputs, implementation, binaries and
settings; use a fresh cache directory after changing any of these. Matching ARM2
draft admission checks and exact designs already verified within the task are
reused; other submitted baseline/search designs still receive native checks.

The full cold run is **not demonstrated to fit six hours**. See `DESIGN.md`
and `results/evidence/pipeline-budget.json` for measured CPU projections, limitations and the
before/after search table. Regenerate the report tables without running models:

```sh
python3 convert/report_search.py
```

Admission uses only native ARM2 draft feasibility; all source tasks are admitted.
Search and baselines are diagnostics. Reward is feasible savings divided by draft
cost in both splits, with no anchors or calibration runs.
