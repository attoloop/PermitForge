"""The learnability card and results page come from the run folder alone; offline, no models."""
import json

import pytest

from eval.learnability import BASE, TRAINED, build_card, ledger, pair_table, results_markdown, training_summary


def episode(model, task, station, repeat, reward, cost=100.0):
    return {'schema_version': 1, 'task_id': task, 'split': 'eval', 'station': station, 'model': model,
            'repeat': repeat, 'ended': 'submitted', 'finished_at': f'2026-10-19T1{repeat}:00:00+00:00',
            'scenario': {'draft_cost': cost}, 'turns': 5, 'evaluations_used': 3, 'usage': {'usd': None},
            'final': {'valid': True, 'feasible': True, 'prescore_usd_saved': reward*cost, 'reward': reward}}


def rollout(step, reward, *, turn_limit=False, evaluations=2):
    return {'task_id': 't', 'optimizer_step': step, 'ended': 'turn_limit' if turn_limit else 'submitted',
            'submitted': not turn_limit, 'turn_limit': turn_limit, 'budget_cut': False,
            'evaluations_used': evaluations, 'completion_tokens': 100,
            'final': None if turn_limit else {'reward': reward}}


@pytest.fixture
def run(tmp_path):
    run = tmp_path/'full-2'
    scores = {BASE: {'a': [.4, .5], 'b': [.2, .2]}, TRAINED: {'a': [.5, .5], 'b': [.1, .2]}}
    for model, stage in ((BASE, 'base-eval'), (TRAINED, 'eval')):
        for task, (s1, s2) in scores[model].items():
            for repeat, score in ((1, s1), (2, s2)):
                path = run/stage/'qwen'/task/f'r{repeat}'/'episode.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(episode(model, task, 'K'+task.upper(), repeat, score)))
    (run/'eval/sampling.json').write_text(json.dumps({'max_turns': 40, 'evaluate_budget': 30, 'repeat_count': 2,
        'max_tokens': 32768, 'wall_clock': 1800, 'generation_config': {'temperature': 1.0, 'top_p': .95, 'top_k': 20}}))
    (run/'run.json').write_text(json.dumps({'config': {'hourly_usd': 5.0, 'model': 'Qwen/Qwen3.8-27B',
        'revision': 'abcdef0123456789', 'grpo_steps': 3, 'num_generations': 2, 'lora_rank': 16}}))
    (run/'grpo').mkdir()
    log = [{'step': s, 'reward': r, 'reward_std': .1, 'grad_norm': .05, 'entropy': .2, 'loss': .1,
            'learning_rate': 3e-5, 'step_time': 100.0, 'completions/mean_length': 1000.0} for s, r in ((1, .3), (2, .5), (3, .6))]
    (run/'grpo/trainer_state.json').write_text(json.dumps({'log_history': log + [{'step': 3, 'train_loss': 0}]}))
    rows = [rollout(0, .3), rollout(0, 0.0, evaluations=0), rollout(1, .5), rollout(1, .5, turn_limit=True),
            rollout(1, .4), rollout(1, .6), rollout(2, .6), rollout(2, .6)]  # step 2 generated twice
    (run/'grpo/episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (run/'grpo/metrics.json').write_text(json.dumps({'peak_vram_reserved_bytes': 1e11, 'seconds_per_step': 100,
                                                     'ending_counts': {'submitted': 7, 'turn_limit': 1}}))
    return run


def subsets():
    policy = {'feasible': True, 'usd_saved': 20}
    return {'eval': [{'task_id': t, 'draft_cost': 100, 'grading': 'fast-field diagnostic, not model grade',
                      'policies': {**{n: policy for n in ('draft', 'cheapest_first_greedy', 'v1_coordinate_descent')},
                                   'search': {**policy, 'verification_source': 'fast-field selection probe'}},
                      'random_30': [{'seed': i, 'feasible': True, 'usd_saved': 10} for i in range(8)]} for t in 'ab']}


JOBS = [{'id': '1', 'stage': 'COMPLETED', 'seconds': 3600, 'usd': 5.0, 'kind': 'job',
         'command': 'uv run hf_job.py -- --recipe grpo --artifact-uri hf://b/x/full-2 --stage base-eval'},
        {'id': '2', 'stage': 'COMPLETED', 'seconds': 7200, 'usd': 10.0, 'kind': 'job',
         'command': 'uv run hf_job.py -- --recipe grpo --artifact-uri hf://b/x/full-2 --stage eval'},
        {'id': '3', 'stage': 'ERROR', 'seconds': 60, 'usd': .083, 'kind': 'job',
         'command': 'uv run hf_job.py -- --recipe smoke --artifact-uri hf://b/x/smoke-z1'},
        {'id': '4', 'stage': 'COMPLETED', 'seconds': 1, 'usd': .001, 'kind': 'probe', 'command': 'bash -c echo'}]


def test_paired_difference_and_training_thirds(run):
    base = [json.loads(p.read_text()) for p in sorted((run/'base-eval').rglob('episode.json'))]
    trained = [json.loads(p.read_text()) for p in sorted((run/'eval').rglob('episode.json'))]
    pairs = pair_table(base, trained)
    assert [t['diff'] for t in pairs['tasks']] == pytest.approx([.05, -.05])
    assert pairs['mean_diff'] == pytest.approx(0) and pairs['wins'] == 1 and pairs['losses'] == 1
    assert pairs['se'] == pytest.approx(pairs['sd']/2**.5)
    train = training_summary(run)
    assert [s['rollouts'] for s in train['steps']] == [2, 4, 2] and train['rollouts'] == 8
    assert train['steps'][0]['no_evaluation'] == 1 and train['steps'][0]['zero_reward'] == 1
    assert train['steps'][1]['turn_limit'] == 1 and train['steps'][1]['zero_reward'] == 1
    assert [t['steps'] for t in train['thirds']] == ['1-1', '2-2', '3-3']


def test_card_follows_the_fineenvs_shape_and_prices_episodes_from_the_ledger(run):
    card = build_card(run, subsets(), JOBS)
    assert list(card) == ['env', 'version', 'authors', 'framework', 'tasks', 'harness', 'grading',
                          'baselines', 'models', 'links', 'date']
    assert card['tasks']['n'] == 2 and card['harness']['max_turns'] == 40 and card['date'] == '2026-10-19'
    base, trained = card['models']
    assert (base['model'], trained['model']) == ('Qwen3.8-27B', 'Qwen3.8-27B + GRPO LoRA')
    assert base['mean'] == pytest.approx(.325) and trained['mean'] == pytest.approx(.325)
    assert base['feasible'] == 4 and base['runs'] == 4 and len(base['ci95']) == 2
    assert base['usd_per_episode'] == pytest.approx(5/4) and trained['usd_per_episode'] == pytest.approx(10/4)
    assert all(v.startswith('https://') for v in card['links'].values())
    bad = subsets()
    bad['eval'][0]['task_id'] = 'zz'
    with pytest.raises(ValueError, match='cohort'):
        build_card(run, bad, JOBS)
    text = results_markdown(run, card, JOBS)
    for needle in ('| `a` | KA |', '## Cost', 'full-2 base-eval', 'smoke smoke-z1', 'node probes',
                   '| 1 | 0.300 | 0.100 |', '| 2-2 | 0.500 | 4 | 1 | 1 | 0 | 0 |'):
        assert needle in text, needle
    assert '—' not in text and '–' not in text


def test_ledger_groups_by_run_and_stage():
    led = ledger(JOBS, 'full-2')
    rows = {r['phase']: r for r in led['rows']}
    assert rows['full-2 base-eval']['usd'] == 5.0 and rows['full-2 eval']['jobs'] == 1
    assert rows['smoke smoke-z1']['failed'] == 1 and led['total_jobs'] == 4
    assert led['total_usd'] == pytest.approx(15.084)
