from collections import Counter
from types import SimpleNamespace

from train.token_budget import episode_calls, select_measurement_tasks, recommended_limit


def test_stratification_is_train_only_and_deterministic():
    tasks = [SimpleNamespace(task_id=f'{split}-{n}-{i}', split=split,
                             record={'eligible': True, 'n_sources': n})
             for split in ('eval', 'train') for n in range(2, 7) for i in range(3)]
    chosen = select_measurement_tasks(SimpleNamespace(tasks=list(reversed(tasks))))
    assert len(chosen) == 10
    assert all(t.split == 'train' for t in chosen)
    assert Counter(t.record['n_sources'] for t in chosen) == {n: 2 for n in range(2, 7)}
    assert [t.task_id for t in chosen] == [f'train-{n}-{i}' for n in range(2, 7) for i in range(2)]


def test_episode_budget_includes_replies_and_final_submission():
    for calls in (8, 10, 12):
        names = episode_calls(calls)
        assert len(names) == calls
        assert names[0] == 'get_scenario' and names[-1] == 'submit_design'
        assert names.count('check_design') == 2
        assert names.count('evaluate_design') == calls - 4
    assert recommended_limit([1000, 2000]) == 2560
