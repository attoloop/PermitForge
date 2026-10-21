"""Card/report acceptance: offline records only; never run models or native code."""
import copy
import json
import statistics

import pytest

from eval.card import DISPLAY, audit_markdown, audit_run, baseline_scores, build_card, scan_episode, variance_markdown
from eval.common import ROOT


RUN = ROOT / 'results/rollouts/final-medium3'
SUBSETS = ROOT / 'results/subsets.json'
requires_run = pytest.mark.skipif(not (RUN / 'run.json').exists(), reason='local rollout evidence absent')


def test_baselines_zero_failures_and_average_seeds_before_tasks():
    def task(cost, saved):
        policy = {'feasible': True, 'usd_saved': saved}
        return {'draft_cost': cost, 'grading': 'fast-field diagnostic, not model grade',
                'policies': {**{n: policy for n in ('draft', 'cheapest_first_greedy', 'v1_coordinate_descent')},
                             'search': {**policy, 'verification_source': 'fast-field selection probe'}},
                'random_30': [{'seed': i, 'feasible': i == 0, 'usd_saved': cost/2 if i == 0 else -1}
                              for i in range(8)]}
    tasks = [task(100, 20), task(1000, 600)]
    tasks[1]['policies']['cheapest_first_greedy'] = {'feasible': False, 'usd_saved': 600}
    means = {r['name']: r['mean'] for r in baseline_scores(tasks)}
    assert means['cheapest_first_greedy'] == pytest.approx(.1)
    assert means['random_30 (mean over 8 seeds)'] == pytest.approx(.0625)
    assert means['best of 8 random 30-evaluation runs'] == pytest.approx(.5)
    assert means['search (60 seconds; not proven optimal)'] == pytest.approx(.4)
    bad = copy.deepcopy(tasks)
    bad[0]['policies']['search']['verification_source'] = 'native'
    with pytest.raises(ValueError, match='fast-field'):
        baseline_scores(bad)
    bad = copy.deepcopy(tasks)
    bad[0]['random_30'].pop()
    with pytest.raises(ValueError, match='seed'):
        baseline_scores(bad)


def test_audit_counts_attempts_not_only_final_grades(tmp_path):
    draft = {'S1': {'control': 'S1_maximum', 'height_m': 65.0}}
    events = []
    def call(name, design, result):
        events.extend([{'kind': 'tool_call', 'turn': len(events)+1,
                        'content': {'name': name, 'args': {'design': design}}},
                       {'kind': 'tool_result', 'content': result}])
    call('evaluate_design', draft, {'valid': True, 'evaluations_left': 0})
    call('evaluate_design', {'S1': {'height_m': 65, 'control': 'S1_maximum'}},
         {'valid': True, 'errors': ['evaluation budget exhausted'], 'evaluations_left': 0})
    call('check_design', {}, {'valid': False})
    call('shell', {}, {'errors': ['unadvertised tool']})
    call('submit_design', draft, {'valid': True, 'feasible': True})
    (tmp_path / 'transcript.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
    raw = tmp_path / 'raw'
    raw.mkdir()
    (raw / 'codex-events.jsonl').write_text(json.dumps({'type': 'error', 'message':
                                                    'isolation boundary rejected provider tool call'})+'\n')
    (raw / 'codex-stderr.log').write_text('unadvertised tool call rejected\n')
    result = scan_episode(tmp_path, {'scenario': {'draft_design': draft}})
    assert result['counts']['outside_tools'] == 1
    assert result['counts']['invalid_designs'] == 1
    assert result['counts']['draft_submissions'] == 1
    assert result['counts']['evaluate_after_exhaustion'] == 1
    assert result['counts']['repeated_evaluations'] == 1
    assert result['counts']['proxy_rejection_lines'] == 2


def test_audit_finds_raw_calls_that_never_reach_transcript(tmp_path):
    (tmp_path / 'transcript.jsonl').write_text('')
    raw = tmp_path / 'raw'
    raw.mkdir()
    events = [{'type': 'item.completed', 'item': {'type': 'mcp_tool_call', 'tool': 'read_file',
                                               'server': 'other'}},
              {'type': 'item.completed', 'item': {'type': 'command_execution', 'command': 'ls'}}]
    (raw / 'codex-events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    result = scan_episode(tmp_path, {'scenario': {'draft_design': {}}})
    assert result['counts']['outside_tools'] == 2


@requires_run
def test_card_matches_board_and_schema_deterministically():
    card = build_card(RUN, SUBSETS)
    assert card == build_card(RUN, SUBSETS)
    # FineEnvs card schema, as in 07-simulation-environments/portsim-live/results/eval-v4.json.
    assert list(card) == ['env', 'version', 'authors', 'framework', 'tasks', 'harness', 'grading',
                          'baselines', 'models', 'links', 'date']
    assert card['tasks']['n'] == 10 and card['tasks']['split'] == 'eval'
    assert card['tasks']['selection'] == json.loads(SUBSETS.read_text())['eval_card_selection']
    assert card['harness']['repeats'] == 3  # Run overrides stale subsets.repeats=4.
    assert card['harness']['evaluate_budget'] == 30
    assert list(card['harness'])[:3] == ['agent', 'tools', 'max_turns'] and len(card['harness']['agent']) < 60
    board = json.loads((RUN / 'board-eval.json').read_text())
    for row, expected in zip(card['models'], board['models'], strict=True):
        assert list(row) == ['model', 'provider', 'runs', 'mean', 'ci95', 'feasible', 'usd_per_episode']
        assert (row['model'], row['provider']) == DISPLAY[expected['model']]
        assert row['runs'] == 30 and row['feasible'] == 30  # a count of feasible episodes, as upstream
        assert row['mean'] == round(expected['mean'], 4) and len(str(row['mean']).split('.')[1]) <= 4
        assert row['ci95'] == [round(x, 4) for x in expected['ci95']]
        assert row['usd_per_episode'] == (None if expected['usd_per_episode'] is None
                                          else round(expected['usd_per_episode'], 4))
    assert all(card['links'][k].startswith('https://') for k in ('runs', 'replay', 'code'))
    assert json.loads((ROOT / 'results/eval-v2.json').read_text()) == card


@requires_run
def test_variance_recomputed_from_all_repeats_and_audit_coverage():
    text = variance_markdown(RUN)
    assert (ROOT / 'results/variance-final.md').read_text() == text
    for split in ('eval', 'train'):
        board = json.loads((RUN / f'board-{split}.json').read_text())
        for model in board['models']:
            std = statistics.mean(t['std'] for t in model['per_task'])
            assert f'{std:.6f}' in text
    audit = audit_run(RUN)
    assert audit['episodes'] == 240 and audit['transcripts'] == 240
    assert audit['failure_categories'] == {'submitted': 240, 'invalid': 0, 'infeasible': 0,
                                           'no_submit': 0, 'turn_limit': 0, 'error': 0, 'timeout': 0}
    assert audit['counts']['outside_tools'] == 0
    assert audit['counts']['draft_submissions'] == 0
    assert audit['counts']['invalid_designs'] == 10
    assert audit['counts']['repeated_evaluations'] == 6
    assert audit['counts']['evaluate_after_exhaustion'] == 0
    assert audit['counts']['proxy_rejection_lines'] == 0
    assert len(audit['infrastructure_attempts']) == 2
    text = (ROOT / 'results/reward-hacks.md').read_text()
    section = text[text.index('## Evaluation audit'):text.index('## Regeneration and limits')]
    assert section == audit_markdown(RUN, json.loads(SUBSETS.read_text()), audit)+'\n'
