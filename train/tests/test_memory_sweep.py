import json
import subprocess

from train.memory_sweep import configurations, run_sweep, worker


def test_sweep_decreases_capacity_and_includes_floor():
    configs = configurations({8: 8192, 10: 10240, 12: 12288})
    assert configs[0]['num_generations'] == 4
    assert configs[-1]['num_generations'] == 2
    assert [c['max_completion_length'] for c in configs] == sorted(
        [c['max_completion_length'] for c in configs], reverse=True)
    for c in configs:
        assert c['generation_batch_size'] == c['num_generations']
        assert c['accum'] == 2*c['num_generations']


def test_oom_and_other_errors_do_not_abort_sweep(tmp_path):
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        output = command[command.index('--output')+1]
        from pathlib import Path
        path = Path(output)
        if len(calls) == 1:
            row = {'status': 'oom', 'oom': True, 'peak_vram_allocated_bytes': 123}
        elif len(calls) == 2:
            row = {'status': 'passed', 'oom': False, 'optimizer_steps': 1,
                   'tasks_seen': ['a', 'b'], 'completion_lengths': [100], 'groups': []}
        else:
            return subprocess.CompletedProcess(command, 1)
        (path/'probe.json').write_text(json.dumps(row))
        return subprocess.CompletedProcess(command, int(row['oom']))
    result = run_sweep(models=['Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B'],
                       configs=configurations({8: 8192, 10: 10240, 12: 12288})[:2],
                       common=['--tasks', 'a', 'b'], output=tmp_path/'sweep.json',
                       runs_dir=tmp_path/'runs', runner=runner)
    assert len(result['results']) == 4
    assert result['results'][0]['oom'] is True
    assert result['results'][2]['status'] == 'error'
    assert result['first_success']['Qwen/Qwen3-0.6B'] == result['results'][1]['config_id']
    assert result['first_success']['Qwen/Qwen3-1.7B'] is None
    assert json.loads((tmp_path/'sweep.json').read_text()) == result


def test_worker_persists_model_load_oom(monkeypatch, tmp_path):
    import torch
    from train import grpo
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    def fail(argv):
        raise torch.OutOfMemoryError('synthetic model-load allocation failure')
    monkeypatch.setattr(grpo, 'main', fail)
    assert worker(['--output', str(tmp_path)]) == 1
    result = json.loads((tmp_path/'probe.json').read_text())
    assert result['status'] == 'oom' and result['oom'] is True
    assert result['optimizer_steps'] == 0
    assert result['peak_vram_allocated_bytes'] is None


def test_timeout_continues_and_is_not_mislabeled_as_oom(tmp_path):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 1)
    result = run_sweep(models=['Qwen/Qwen3-0.6B'],
                       configs=configurations({8: 9728, 10: 12288, 12: 14336})[:2], common=[],
                       output=tmp_path/'sweep.json', runs_dir=tmp_path/'runs', runner=timeout)
    assert [r['status'] for r in result['results']] == ['timeout', 'timeout']
    assert all(r['oom'] is None for r in result['results'])
