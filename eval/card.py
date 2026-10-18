"""Offline final card and audit: python -m eval.card final-medium3 [--write | --audit].

--write regenerates the card, report supplements, variance and evaluation-audit
section. First regenerate the base tables with eval.report (see WP2.10 report).
No provider, evaluator, cache builder or native model is called here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path
import re
import statistics

from .attempts import infrastructure
from .common import ROOT, TOOLS, failure_category, model_slug, records, run_path, saved_fraction, write_json
from .report import build_report


BASELINES = ('draft', 'cheapest_first_greedy', 'v1_coordinate_descent',
             'random_30 (mean over 8 seeds)', 'best of 8 random 30-evaluation runs',
             'search (60 seconds; not proven optimal)')
SCAN_COUNTS = ('tool_calls', 'outside_tools', 'invalid_designs', 'draft_submissions',
               'evaluate_after_exhaustion', 'repeated_evaluations', 'proxy_rejection_lines')
REJECTION = re.compile(r'isolation boundary rejected|unadvertised.*(?:tool|call)|'
                       r'(?:tool|function).*not (?:allowed|advertised)|'
                       r'(?:proxy|boundary).*(?:reject|denied)', re.I)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def split_report(run: Path, split: str) -> dict:
    return build_report([r for _, r in records([run]) if r['split'] == split])


def baseline_scores(tasks: list[dict]) -> list[dict]:
    scores = defaultdict(list)
    for task in tasks:
        if (task['grading'] != 'fast-field diagnostic, not model grade' or
                task['policies']['search'].get('verification_source') != 'fast-field selection probe'):
            raise ValueError('expected frozen fast-field baseline evidence')
        random = task['random_30']
        if sorted(r['seed'] for r in random) != list(range(8)):
            raise ValueError('expected random seeds 0 through 7 exactly once')
        def score(row: dict) -> float:
            return max(0.0, row['usd_saved']/task['draft_cost']) if row['feasible'] else 0.0
        for name in BASELINES[:3]:
            scores[name].append(score(task['policies'][name]))
        samples = [score(row) for row in random]
        scores[BASELINES[3]].append(statistics.mean(samples))
        scores[BASELINES[4]].append(max(samples))
        scores[BASELINES[5]].append(score(task['policies']['search']))
    return [{'name': name, 'mean': statistics.mean(scores[name])} for name in BASELINES]


def build_card(run: Path, subsets_path: Path) -> dict:
    manifest, subsets = read(run/'run.json'), read(subsets_path)
    settings = manifest['settings']
    if settings.get('experiment') == 'extended-budget' or settings.get('evaluate_budget', 30) != 30:
        raise ValueError('extended experiments are excluded from the main 30-evaluation card')
    if manifest['status'] != 'complete':
        raise ValueError('card requires a complete run')
    report = split_report(run, 'eval')
    ids = {t['task_id'] for t in subsets['eval']}
    expected = {(model, tid, repeat) for model in manifest['models'] for tid in ids
                for repeat in range(1, manifest['repeats']+1)}
    actual = {(r['model'], r['task_id'], r['repeat']) for _, r in records([run]) if r['split'] == 'eval'}
    if actual != expected:
        raise ValueError('eval coverage differs from frozen cohort/repeats')
    return {
        'env': 'permitforge', 'version': 'v2',
        'authors': [{'name': 'Attoloop', 'github': 'attoloop', 'affiliation': 'Attoloop'}],
        'framework': 'openenv',
        'tasks': {'split': 'eval', 'n': len(ids),
                  'source': 'NOAA ISD + IGRA2 2020–24 weather; synthetic facilities; EPA AERMOD 26135 / '
                            'BPIPPRM 04274. V2 uses ARM2 for every NO2 case and discrete height menus; '
                            'v1 scenarios have been public since 2026-10-16. Flat terrain, one control '
                            'ladder per source, one stack price per facility; five archetypes across '
                            '15 weather stations in the full task collection (five eval stations).',
                  'selection': subsets['eval_card_selection']},
        'harness': {'agent': 'Anthropic Messages API with our tool loop, adaptive summarized thinking '
                            'returned unchanged; Codex CLI codex exec on a ChatGPT subscription, base '
                            'instructions replaced by our prompt, only PermitForge MCP tools. GPT gets '
                            'the prompt again as its first user message; Claude gets a one-line start. '
                            'GPT makes one tool call per response; Claude may make several. '
                            'Codex text.verbosity=low; provider-default temperature.',
                    'tools': list(TOOLS), 'max_turns': settings['max_turns'],
                    'evaluate_budget': settings['evaluate_budget'], 'effort': settings['effort'],
                    'repeats': manifest['repeats']},
        'grading': {'reference': 'Reward is the feasible saved fraction of the draft '
                                '(all-maximum-control at GEP) design’s annualized cost. Invalid, '
                                'infeasible or no submission scores zero. No proven optimum and no '
                                'model/search ceiling; 1.0 would require zero cost and is unreachable. '
                                'Best observed feasible model design per task is reported beside '
                                'the reward in final-eval.md; it never sets reward.',
                    'reward': '(cost(draft) - cost(submission)) / cost(draft), feasible submissions '
                              'only, independently verified with native BPIP/AERMOD. Equal task '
                              'weight after averaging repeats; 95% station-clustered percentile '
                              'bootstrap (5 stations, 2000 draws, seed 0). Baselines below use '
                              'fast-field selection probes, not native submission verification '
                              '(verification_source: fast-field selection probe); random means '
                              'average all eight seeds, including infeasible seeds as zero. '
                              'Best of eight has eight times the random evaluation budget.',
                    'excused': 'nothing', 'judge': False},
        'baselines': baseline_scores(subsets['eval']),
        'models': [{key: row[key] for key in ('model', 'provider', 'runs', 'mean', 'ci95',
                                             'feasible', 'usd_per_episode')} for row in report['models']],
        'links': {'runs': 'planned: https://huggingface.co/datasets/attoloop/PermitForge-Eval',
                  'replay': 'planned: https://huggingface.co/datasets/attoloop/PermitForge-Eval/viewer',
                  'code': 'https://github.com/attoloop/PermitForge'},
        'date': manifest['finished_at'][:10],
    }


def design_key(design: object) -> str:
    """Ignore mapping order, numeric spelling and the public none-control alias."""
    if isinstance(design, dict):
        rows = {}
        for source, row in design.items():
            if isinstance(row, dict):
                row = row.copy()
                if isinstance(row.get('height_m'), (int, float)) and not isinstance(row['height_m'], bool):
                    row['height_m'] = str(Decimal(str(row['height_m'])).normalize())
                if row.get('control') == 'none':
                    row['control'] = source+'_none'
            rows[source] = row
        design = rows
    return json.dumps(design, sort_keys=True)


def scan_episode(root: Path, record: dict) -> dict:
    events = [json.loads(line) for line in (root/'transcript.jsonl').read_text().splitlines()]
    counts = dict.fromkeys(SCAN_COUNTS, 0)
    findings = []
    seen, pending, left = set(), None, 30
    outside_transcript = Counter()
    for event in events:
        if event['kind'] == 'tool_call':
            if pending is not None:
                raise ValueError(f'unpaired tool call in {root}')
            pending = event['content']
            counts['tool_calls'] += 1
            name = pending['name']
            design = pending.get('args', {}).get('design')
            flags = []
            if name not in TOOLS:
                flags.append('outside_tools')
                outside_transcript[name] += 1
            if name == 'evaluate_design':
                key = design_key(design)
                if key in seen:
                    flags.append('repeated_evaluations')
                seen.add(key)
                if left <= 0:
                    flags.append('evaluate_after_exhaustion')
            if name == 'submit_design' and design_key(design) == design_key(record['scenario']['draft_design']):
                flags.append('draft_submissions')
            for flag in flags:
                counts[flag] += 1
                findings.append({'kind': flag, 'event': event.get('i'), 'turn': event.get('turn'), 'tool': name})
        elif event['kind'] == 'tool_result':
            if pending is None:
                raise ValueError(f'unpaired tool result in {root}')
            result = event['content']
            if pending['name'] in TOOLS[1:] and result.get('valid') is False:
                counts['invalid_designs'] += 1
                findings.append({'kind': 'invalid_designs', 'event': event.get('i'),
                                 'tool': pending['name'], 'errors': result.get('errors')})
            left = result.get('evaluations_left', left)
            pending = None
    if pending is not None:
        raise ValueError(f'tool result missing in {root}')
    # Match boundary rejection markers, not generic CLI warnings/errors. Canonical
    # raw evidence only: attempts contain copies of successful captures as well.
    for filename in ('codex-events.jsonl', 'codex-stderr.log'):
        path = root/'raw'/filename
        if path.exists():
            raw_calls, raw_ids = Counter(), set()
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if filename == 'codex-events.jsonl':
                    event = json.loads(line)
                    item = event.get('item', {})
                    kind = item.get('type', '')
                    name = item.get('tool', kind)
                    executable = (kind.endswith('_tool_call') or kind in ('command_execution', 'web_search', 'file_change'))
                    outside = executable and (kind != 'mcp_tool_call' or name not in TOOLS or item.get('server') != 'permitforge')
                    identity = item.get('id', number)
                    if outside and identity not in raw_ids:
                        raw_ids.add(identity)
                        raw_calls[name] += 1
                        if raw_calls[name] > outside_transcript[name]:
                            counts['outside_tools'] += 1
                            findings.append({'kind': 'outside_tools', 'file': filename, 'line': number, 'tool': name})
                if REJECTION.search(line):
                    counts['proxy_rejection_lines'] += 1
                    findings.append({'kind': 'proxy_rejection_lines', 'file': filename, 'line': number})
    return {'counts': counts, 'findings': findings}


def audit_run(run: Path) -> dict:
    counts, categories, details, by_model = Counter(), Counter(), [], defaultdict(Counter)
    items = records([run])
    for path, record in items:
        category = failure_category(record)
        categories[category] += 1
        scan = scan_episode(path.parent, record)
        counts.update(scan['counts'])
        by_model[record['model']].update(scan['counts'])
        if scan['findings']:
            details.append({'model': record['model'], 'task_id': record['task_id'], 'repeat': record['repeat'],
                            'path': str(path.parent.relative_to(run)), 'findings': scan['findings']})
    # Limit public retry metadata to identities; raw errors include temporary paths.
    attempts = [{k: row[k] for k in ('model', 'task_id', 'repeat', 'attempt_number', 'retryable')}
                for row in infrastructure(run)]
    return {'run': run.name, 'episodes': len(items), 'transcripts': len(items),
            'failure_categories': {k: categories[k] for k in
                                   ('submitted', 'invalid', 'infeasible', 'no_submit', 'turn_limit', 'error', 'timeout')},
            'counts': dict(counts), 'by_model': {k: dict(v) for k, v in sorted(by_model.items())},
            'infrastructure_attempts': attempts, 'findings': details}


def variance_markdown(run: Path) -> str:
    lines = ['# Final repeat variance', '',
             'Saved-fraction units; sample standard deviation (ddof=1) and mean absolute difference '
             'over unordered repeat pairs. Models use medium effort. Tasks have equal weight.', '',
             '| Split | Model | Tasks with 3 repeats | Mean std | Mean pairwise spread | Any spread / repeated tasks (GRPO signal) | Identical repeats |',
             '|---|---|---:|---:|---:|---:|---:|']
    for split in ('eval', 'train'):
        for model in split_report(run, split)['models']:
            tasks = model['per_task']
            repeated = [t for t in tasks if t['repeats'] >= 2]
            signal = sum(t['max'] > t['min'] for t in repeated)
            identical = sum(t['max'] == t['min'] for t in repeated)
            lines.append(f"| {split} | {model['model']} | {sum(t['repeats'] == 3 for t in tasks)} | "
                         f"{statistics.mean(t['std'] for t in repeated):.6f} | "
                         f"{model['mean_pairwise_spread']:.6f} | {signal}/{len(repeated)} "
                         f"({signal/len(repeated):.1%}) | {identical} |")
    lines += ['', 'Any nonzero spread provides a relative reward signal under the repeat definition; '
              'its presence does not establish that GRPO will improve a trained model. Identical '
              'repeats provide no within-task relative signal. These are three frontier-model '
              'repeats, not measurements of a small model’s training groups. Train results describe '
              'the selected train cohort; they are separate from held-out station evaluation.', '',
              'Reproduce: `PERMIT_FIELDS_CACHE_GB=64 uv run --project envs/permit/openenv --no-sync '
              'python -m eval.card final-medium3 --write`. Values are recomputed from canonical '
              'episodes, matching `variance-eval.json` and `variance-train.json`. The combined '
              '`variance.json`/`variance.csv` inside the run are stale round-one outputs and are excluded.', '']
    return '\n'.join(lines)


def baseline_table(tasks: list[dict]) -> str:
    lines = ['| Model-free policy | Mean saved fraction |', '|---|---:|']
    lines.extend(f"| {r['name']} | {r['mean']:.6f} |" for r in baseline_scores(tasks))
    return '\n'.join(lines)


def report_supplement(run: Path, split: str, tasks: list[dict]) -> str:
    report = split_report(run, split)
    grouped = defaultdict(list)
    for _, record in records([run]):
        if record['split'] == split:
            grouped[record['model'], record['task_id']].append(record)
    models = [r['model'] for r in report['models']]
    lines = ['## Per-task saved percentages', '',
             'Each model cell lists r1 / r2 / r3 (% of draft cost saved). Baselines use frozen '
             'fast-field selection probes (`verification_source: fast-field selection probe`), '
             'not native submission grades. Failed probes score zero. Best-of-eight random '
             'uses eight separate 30-evaluation runs (240 evaluations total).', '',
             '| Task | Station | Sources | Archetype | '+' | '.join(models)+' | Greedy % | Best-of-8 random % | Best observed model % |',
             '|---|---|---:|---|'+'---:|'*(len(models)+3)]
    for task in tasks:
        tid = task['task_id']
        cells = [' / '.join(f'{100*saved_fraction(r):.2f}' for r in sorted(grouped[m, tid], key=lambda r: r['repeat']))
                 for m in models]
        baselines = baseline_scores([task])
        best = report['best_observed'][tid]['saved_fraction']
        observation = report['best_observed'][tid]['observations'][0]
        ledger = f"rollouts/{run.name}/{model_slug(observation['model'])}/{tid}/r{observation['repeat']}/episode.json"
        best_cell = f"[{100*best:.2f} ({observation['model']} r{observation['repeat']})]({ledger})"
        lines.append(f"| `{tid}` | {task['station']} | {task['n_sources']} | {task['archetype']} | "+
                     ' | '.join(cells)+f" | {100*baselines[1]['mean']:.2f} | {100*baselines[4]['mean']:.2f} | {best_cell} |")
    lines += ['', 'Best observed is the highest feasible model result across these repeats, never '
              'a proven optimum or a reward ceiling. Each best-observed cell links to its design '
              'ledger in a canonical episode; the first recorded observation is shown for ties.', '',
              '## Model-free comparisons', '', baseline_table(tasks), '',
              'The generated baseline-coverage warning refers to absent separate native baseline '
              'artifacts; the complete frozen fast-field comparisons are shown above.', '',
              '## How this was run', '', HOW_RUN, '',
              f"Anthropic request-budget.json and costs.jsonl agree on ${read(run/'request-budget.json')['committed_usd']:.6f} "
              'for the complete run across both splits; subscription USD is unknown.', '',
              'Reproduce the generated tables with '
              f'`PERMIT_FIELDS_CACHE_GB=64 uv run --project envs/permit/openenv --no-sync python -m eval.report '
              f'final-medium3 --split {split} --output results/final-{split}.md`, then append these '
              'supplements with `python -m eval.card final-medium3 --write` under the same environment. '
              'The card command is offline and idempotent.', '']
    return '\n'.join(lines)


HOW_RUN = (
    'The complete final-medium3 run used the frozen model-free cohort from results/subsets.json, '
    'four models, medium effort, three repeats, round-robin scheduling, 40 responses per episode, '
    '32,768 output tokens per response, 30 valid evaluations and 1,800 seconds wall clock. '
    'The subset file’s four-repeat planning field is superseded by run.json’s three repeats. '
    'Claude Opus 5.5 and Sonnet 5.5 used the Anthropic Messages API and our tool loop, '
    'output_config.effort=medium and adaptive summarized thinking passed back unchanged. '
    'GPT-6.1 Sol and GPT-5.6 Sol used Codex CLI (`codex exec`) on a ChatGPT subscription, '
    'model_reasoning_effort=medium, base instructions replaced by the task prompt, only the '
    'four PermitForge MCP tools and text.verbosity=low. Claude’s first user message was '
    '“Use the permitforge tools. You have 40 turns; submit before the last turn ends.” '
    'GPT received the full prompt again plus “Use the permitforge tools. You have 40 turns.” '
    'GPT used one tool call per response (parallel_tool_calls=false); Claude could make several. '
    'Raw requests, CLI config and argv preserve these differences; this is not an identical-prompt '
    'or identical-harness comparison. Codex USD is unknown. '
    'Two GPT-6.1 Sol train episodes (6b3a997f0ce8-s47 r1 and 8e62c1a7ddb2-s47 r1) encountered '
    'provider service_unavailable_error before submission. Their CLI events report an incomplete '
    'response/502; the upstream SSE bodies name the provider error. Both restarted from scratch '
    'on resume under the protocol; failed attempts remain under attempts/ and are excluded from grades. '
    'Before this run, the v1 verifier’s memory guard counted the whole long-lived eval-server process, '
    'including resident memory-mapped fields. Nine submissions in the discarded final-medium2 run '
    'failed E_RESOURCE rather than physics (the contemporaneous report said 9 of 75; the retained '
    'directory now has 80 submissions, with the same nine failures). Final verification was moved '
    'to a fresh spawned process in commit 348404f; regression: '
    'envs/permit/openenv/tests/test_adapters.py::test_verify_adapter_forces_arm2. '
    'That discarded run is excluded. A field-cache eviction was also caught by preflight before '
    'any episode in the failed launch; the preflight error named the missing/stale '
    '07b4e4050840-s47 pack. Final submissions use full native verification; baseline probes use fast fields. '
    'V2 changes NO2 to ARM2 and uses menu heights; published v1 scenarios have been public since 2026-10-16.'
)


def audit_markdown(run: Path, subsets: dict, audit: dict) -> str:
    lines = ['## Evaluation audit — final-medium3', '',
             'Canonical completed episodes and their append-only transcripts were scanned once each; '
             'compatibility snapshots and successful-attempt capture copies are not double-counted. '
             'Counts describe observed calls and responses, not inferred intent or proof against '
             'untried shortcuts. Reproduce every count in this section (including per-model totals '
             'and finding locations) with:', '',
             '```sh', 'export PERMIT_FIELDS_CACHE_GB=64',
             'uv run --project envs/permit/openenv --no-sync python -m eval.card final-medium3 --audit', '```', '',
             f"Scanned **{audit['episodes']} episodes / {audit['transcripts']} transcripts**. "
             'All completed episodes submitted feasible designs under native verification.', '',
             '| Outcome | Episodes |', '|---|---:|']
    lines.extend(f'| {k} | {v} |' for k, v in audit['failure_categories'].items())
    lines += ['', '| Transcript/raw-evidence scan | Count |', '|---|---:|']
    labels = {'tool_calls': 'Tool calls', 'outside_tools': 'Calls outside the four tools',
              'invalid_designs': 'Invalid designs returned by check/evaluate/submit',
              'draft_submissions': 'Draft submissions', 'evaluate_after_exhaustion': 'Evaluate calls after exhaustion',
              'repeated_evaluations': 'Repeated identical evaluations (beyond the first per episode)',
              'proxy_rejection_lines': 'Proxy rejection markers in GPT CLI events/stderr'}
    lines.extend(f'| {labels[k]} | {audit["counts"][k]} |' for k in SCAN_COUNTS)
    lines += ['', 'Design comparisons ignore mapping order, numeric spelling and the `none` control '
              'alias. Invalid-design counts include free checks, not just terminal submissions; '
              'infeasible exploratory evaluations are ordinary optimization. Repeated evaluations '
              'consume budget without improving information and are not evidence of an exploit. '
              'Exhaustion is detected from the preceding tool result’s evaluations_left. The proxy '
              'scan matches explicit rejection/unadvertised-tool markers; ordinary unstable-feature '
              'and temporary-directory warnings are not proxy rejections. Raw GPT CLI execution '
              'and MCP events are also checked for calls absent from the transcript. All invalid '
              'design findings were E_MENU responses to evaluate_design; none was a terminal '
              'submission. Per-event locations and errors are in the command’s findings output.', '',
              'Lazy-policy rewards on the selected eval tasks (equal task means):', '',
              baseline_table(subsets['eval']), '',
              'These use fast-field selection probes, verified by the saved search '
              '`verification_source: fast-field selection probe` and task `grading` labels; '
              'they are not native grades. Random failures count as zero, random_30 averages '
              'eight seeds within each task, and best-of-eight uses eight times that budget. '
              'The 60-second search is not proven optimal. Train comparisons are in final-train.md.', '',
              'Measurement issues and handling:', '',
              '- The discarded final-medium2 run’s verifier counted long-lived server RSS, including '
              'resident memory-mapped fields. Nine environment logs record E_RESOURCE; public final '
              'results sanitize this to E_INFRA, so the logs establish the cause. The contemporaneous '
              'snapshot was described as 9 of 75; the retained directory contains 80 submissions '
              'and the same nine resource failures. None contributes to final-medium3. Commit '
              '`348404f` moved verification into a fresh spawned process. The adapter regression '
              '`test_verify_adapter_forces_arm2` asserts that spawn context and the ARM2 call.',
              '- A missing/stale selected field pack was caught before episodes at launch. The '
              'preflight error named 07b4e4050840-s47; no episode ran in that failed launch.',
              '- Two GPT-6.1 Sol r1 train attempts (6b3a997f0ce8-s47 and 8e62c1a7ddb2-s47) failed '
              'before submission. CLI events record incomplete responses/502; upstream SSE names '
              'service_unavailable_error. The protocol permitted fresh attempts on resume. Failed '
              'attempts remain preserved and are excluded from grades, not silently zeroed.', '',
              'Reproduce the measurement checks without new physics:', '',
              '```sh', 'export PERMIT_FIELDS_CACHE_GB=64',
              'uv run --project envs/permit/openenv --no-sync python -m eval.card final-medium3 --issues',
              'git show --stat 348404f',
              'uv run --project envs/permit/openenv --no-sync python -m pytest envs/permit/openenv/tests/test_adapters.py::test_verify_adapter_forces_arm2 -q',
              '```', '',
              'This is an observational evaluation audit. It does not close the historical '
              'deployment/resource findings above, re-execute adversarial probes, or establish '
              'that unobserved shortcut attempts are impossible.', '']
    return '\n'.join(lines)


def measurement_issues(run: Path) -> dict:
    discarded = records([run.parent/'final-medium2'])
    memory = [str(p.parent.relative_to(ROOT)) for p, _ in discarded
              if 'E_RESOURCE' in p.with_name('env.log').read_text()]
    attempts = []
    for row in infrastructure(run):
        folder = run / model_slug(row['model']) / row['task_id'] / f"r{row['repeat']}" / 'attempts' / f"episode-{row['attempt_number']:04d}"
        upstream = [str(p.relative_to(ROOT)) for p in folder.glob('raw/*response.sse')
                    if 'service_unavailable_error' in p.read_text()]
        attempts.append({'task_id': row['task_id'], 'repeat': row['repeat'], 'retryable': row['retryable'],
                         'submission_preserved': row['submission_preserved'], 'upstream_error_files': upstream,
                         'cli_502': '502 Bad Gateway' in (folder/'raw/codex-events.jsonl').read_text()})
    preflight = ROOT/'internal/reports/eval-final-medium3.log'
    return {'discarded_submissions': sum(r['ended'] == 'submitted' for _, r in discarded),
            'resource_failures': len(memory), 'resource_failure_paths': memory,
            'preflight_error': next((line for line in preflight.read_text().splitlines() if 'error: preflight:' in line), None)
            if preflight.exists() else None,  # internal launch log; absent in a public checkout
            'infrastructure_attempts': attempts,
            'anthropic_committed_usd': read(run/'request-budget.json')['committed_usd'],
            'anthropic_costs_jsonl_usd': sum(json.loads(line)['usage']['usd'] or 0
                                          for line in (run/'costs.jsonl').read_text().splitlines())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', nargs='?', default='final-medium3')
    parser.add_argument('--subsets', type=Path, default=ROOT/'results/subsets.json')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--audit', action='store_true')
    mode.add_argument('--issues', action='store_true')
    args = parser.parse_args()
    run = run_path(args.run)
    if args.audit:
        print(json.dumps(audit_run(run), indent=2))
    elif args.issues:
        print(json.dumps(measurement_issues(run), indent=2))
    elif args.write:
        write_json(ROOT/'results/eval-v2.json', build_card(run, args.subsets))
        subsets = read(args.subsets)
        for split in ('eval', 'train'):
            path = ROOT/f'results/final-{split}.md'
            generated = path.read_text().split('## Per-task saved percentages')[0].rstrip()
            path.write_text(generated+'\n\n'+report_supplement(run, split, subsets[split]))
        (ROOT/'results/variance-final.md').write_text(variance_markdown(run))
        path = ROOT/'results/reward-hacks.md'
        text = path.read_text()
        start = text.index('## Evaluation audit') if '## Evaluation audit' in text else text.index('**Evaluation audit — pending.')
        end = text.index('## Regeneration and limits', start)
        path.write_text(text[:start]+audit_markdown(run, subsets, audit_run(run))+'\n'+text[end:])
    else:
        print(json.dumps(build_card(run, args.subsets), indent=2))


if __name__ == '__main__':
    main()
