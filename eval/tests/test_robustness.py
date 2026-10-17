"""Pilot regressions: offline startup checks and durable infrastructure attempts."""
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from eval.common import model_slug, records, write_json
from eval.run_eval import run_panel, write_index
from test_contract import episode


TASKS = [{'task_id': 'a', 'split': 'eval', 'station': 'X'}]


def test_preflight_imports_sdk_before_any_episode(monkeypatch, tmp_path, capsys):
    from eval import preflight, run_eval
    imported = []
    def missing(name):
        imported.append(name)
        if name == 'anthropic':
            raise ModuleNotFoundError("No module named 'anthropic'")
        return SimpleNamespace()
    monkeypatch.setattr(preflight, 'import_module', missing)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'private-preflight-key')
    monkeypatch.setattr('sys.argv', ['run_eval', '--run', 'test', '--models',
        'anthropic:claude-sonnet-5-5', '--budget-usd', '10', '--output-root', str(tmp_path)])
    monkeypatch.setattr(run_eval, 'run_panel', lambda *a, **kw: pytest.fail('episode launched'))
    with pytest.raises(SystemExit) as exc:
        run_eval.main()
    assert exc.value.code == 2 and 'anthropic' in imported
    error = capsys.readouterr().err
    assert 'agents' in error and 'preflight' in error and 'private-preflight-key' not in error
    assert not list(tmp_path.iterdir())


def test_preflight_checks_all_drivers_credentials_and_login(monkeypatch):
    from eval import preflight
    imported = []
    monkeypatch.setattr(preflight, 'import_module', lambda name: imported.append(name) or SimpleNamespace())
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    with pytest.raises(preflight.PreflightError, match='ANTHROPIC_API_KEY'):
        preflight.drivers(['anthropic:claude-sonnet-5-5'])
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'do-not-print-this')
    monkeypatch.setattr(preflight.shutil, 'which', lambda name: None)
    with pytest.raises(preflight.PreflightError, match='codex.*CLI'):
        preflight.drivers(['codex:gpt-6.1-sol'])
    monkeypatch.setattr(preflight.shutil, 'which', lambda name: '/test/codex')
    calls = []
    def login(*args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(returncode=0, stdout='Logged in using an API key', stderr='')
    monkeypatch.setattr(preflight.subprocess, 'run', login)
    with pytest.raises(preflight.PreflightError, match='ChatGPT'):
        preflight.drivers(['codex:gpt-6.1-sol'])
    monkeypatch.setattr(preflight.subprocess, 'run', lambda *a, **kw:
        SimpleNamespace(returncode=0, stdout='', stderr='Logged in using ChatGPT'))
    preflight.drivers(['anthropic:claude-sonnet-5-5', 'codex:gpt-6.1-sol'])
    assert {'eval.anthropic_driver', 'eval.codex_driver', 'anthropic'} <= set(imported)
    assert calls == [['codex', 'login', 'status']]


@pytest.mark.parametrize('fault', ['health', 'recorder', 'task'])
def test_server_preflight_fails_without_opening_session(monkeypatch, fault):
    from eval import preflight
    calls = []
    def respond(request):
        calls.append(request.url.path)
        if request.url.path == '/healthz':
            return httpx.Response(503 if fault == 'health' else 200, json={'status': 'ok'})
        return httpx.Response(403 if fault == 'recorder' else 200,
            json={'status': 'ok', 'tasks': [] if fault == 'task' else TASKS})
    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(preflight.httpx, 'Client', lambda **kw: client)
    with pytest.raises(preflight.PreflightError):
        preflight.server('http://test.invalid', 'private-token', TASKS)
    assert all('reset' not in path for path in calls)


def test_recording_preflight_auth_and_tasks_without_episodes(monkeypatch):
    from eval.transport import load_eval_pack, recording_app
    from permit_openenv.episode import Episode
    from permit_openenv.tasks import TaskPack
    task = load_eval_pack().tasks[0]
    monkeypatch.setattr(Episode, '__init__', lambda *a, **kw: pytest.fail('episode created'))
    with TestClient(recording_app('private-token', pack=TaskPack([task]))) as client:
        assert client.get('/eval/preflight').status_code == 403
        response = client.get('/eval/preflight', headers={'Authorization': 'Bearer private-token'})
        assert response.status_code == 200
        assert response.json()['tasks'] == [
            {'task_id': task.task_id, 'split': task.split, 'station': task.record['station']}]


def test_failed_future_does_not_hide_later_submission_or_abort_panel(tmp_path, monkeypatch):
    import eval.run_eval as runner
    monkeypatch.setattr(runner, 'as_completed', lambda futures: iter(futures))
    def driver(model, task, checkpoint, capture):
        if model == 'codex:bad':
            capture.raw_text('partial.txt', 'retained diagnostic')
            checkpoint({'ended': 'no_submit', 'finished_at': 'cleanup', 'usage': {'usd': None}})
            raise RuntimeError('driver broke')
        return episode(model, task['task_id'])
    tasks = TASKS + [{'task_id': 'b', 'split': 'eval', 'station': 'Y'}]
    run_panel(tmp_path, tasks, ['codex:bad', 'codex:good'], driver, concurrency=2)
    assert [(r['model'], r['task_id']) for _, r in records([tmp_path])] == [
        ('codex:good', 'a'), ('codex:good', 'b')]
    index = json.loads((tmp_path/'index.json').read_text())
    assert all(r['ended'] == 'submitted' for r in index['episodes'])
    assert len(index['infrastructure_attempts']) == 2
    failures = list(tmp_path.glob('*/a/attempts/episode-*/attempt.json'))
    failed = next(json.loads(p.read_text()) for p in failures if 'bad' in str(p))
    assert failed['status'] == 'infrastructure' and 'RuntimeError: driver broke' in failed['traceback']
    assert not (tmp_path/model_slug('codex:bad')/'a.json').exists()
    assert (tmp_path/model_slug('codex:bad')/'a/attempts/episode-0001/raw/partial.txt').exists()


def test_resume_retries_infrastructure_at_most_twice(tmp_path):
    calls = []
    def broken(model, task):
        calls.append(task['task_id'])
        raise RuntimeError('always broken')
    for _ in range(4):
        run_panel(tmp_path, TASKS, ['codex:bad'], broken)
    assert calls == ['a', 'a', 'a']
    assert records([tmp_path]) == []
    assert len(list(tmp_path.glob('*/*/attempts/episode-*/attempt.json'))) == 3
    assert json.loads((tmp_path/'index.json').read_text())['infrastructure_attempts'][-1]['retryable'] is False


def test_resume_succeeds_and_does_not_retry_grade(tmp_path):
    def broken(model, task):
        raise RuntimeError('transient harness bug')
    run_panel(tmp_path, TASKS, ['codex:m'], broken)
    run_panel(tmp_path, TASKS, ['codex:m'], lambda m, t: episode(m, t['task_id']))
    run_panel(tmp_path, TASKS, ['codex:m'], lambda *a: pytest.fail('repeated grade'))
    assert len(records([tmp_path])) == 1
    assert records([tmp_path])[0][1]['attempt_number'] == 2


def test_checkpoint_after_submission_survives_driver_exception(tmp_path):
    def interrupted(model, task, checkpoint):
        checkpoint(episode(model, task['task_id']))
        raise RuntimeError('after grade')
    run_panel(tmp_path, TASKS, ['codex:m'], interrupted)
    run_panel(tmp_path, TASKS, ['codex:m'], lambda *a: pytest.fail('repeated submission'))
    assert records([tmp_path])[0][1]['final']['feasible']
    assert 'after grade' in next(tmp_path.glob('*/*/attempts/episode-*/attempt.json')).read_text()


def test_records_ignore_incomplete_cleanup_no_submit(tmp_path):
    value = episode()
    value.update(final=None, ended='no_submit', attempt_complete=False)
    write_json(tmp_path/'model/a/episode.json', value)
    write_index(tmp_path)
    assert records([tmp_path]) == []
    assert json.loads((tmp_path/'board.json').read_text())['models'] == []


def test_repair_pilot_ordering_preserves_grade_and_archives_failures(tmp_path):
    from eval.attempts import recover
    good, bad = episode(), episode('anthropic:m')
    good.update(attempt_complete=False, ended='submitted', error_type='interrupted')
    bad.update(attempt_complete=False, ended='no_submit', final=None, error_type='interrupted')
    for item in [good, bad]:
        write_json(tmp_path/model_slug(item['model'])/'a/episode.json', item)
        write_json(tmp_path/model_slug(item['model'])/'a.json', item)
    write_json(tmp_path/'index.json', {'episodes': [{**good, 'final': None, 'ended': 'no_submit'}]})
    recover(tmp_path)
    write_index(tmp_path)
    assert len(records([tmp_path])) == 1
    canonical, rec = records([tmp_path])[0]
    assert rec['attempt_complete'] and rec['ended'] == 'submitted'
    assert rec == json.loads(canonical.parent.parent.joinpath('a.json').read_text())
    assert json.loads((tmp_path/'index.json').read_text())['episodes'][0]['final'] == good['final']
    assert json.loads((tmp_path/'board.json').read_text())['models'][0]['mean_usd_saved'] == 20
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    recover(tmp_path)
    assert all(p.read_bytes() == content for p, content in before.items())


def test_hard_crash_checkpoint_is_not_a_grade_and_resume_recovers(tmp_path):
    from eval.attempts import recover
    root = tmp_path/model_slug('codex:m')/'a/attempts/episode-0001'
    value = episode()
    value.update(final=None, ended='no_submit', attempt_complete=False)
    write_json(root/'checkpoint.json', value)
    write_json(root/'attempt.json', {'status': 'running', 'attempt_number': 1, 'model': 'codex:m', 'task_id': 'a'})
    assert records([tmp_path]) == []
    recover(tmp_path)
    assert json.loads((root/'attempt.json').read_text())['status'] == 'infrastructure'
    run_panel(tmp_path, TASKS, ['codex:m'], lambda m, t: episode(m, t['task_id']))
    assert records([tmp_path])[0][1]['attempt_number'] == 2


def test_submit_intent_without_grade_is_infrastructure_without_replay(tmp_path):
    def lost(model, task, checkpoint):
        checkpoint({'submit_attempted': True, 'pending_submission': {'design': {}}})
        raise ConnectionError('lost grade response')
    run_panel(tmp_path, TASKS, ['codex:m'], lost)
    run_panel(tmp_path, TASKS, ['codex:m'], lambda *a: pytest.fail('duplicate submission'))
    assert records([tmp_path]) == []
    assert not json.loads((tmp_path/'index.json').read_text())['infrastructure_attempts'][0]['retryable']


def test_anthropic_caught_exception_is_infrastructure_with_traceback(tmp_path):
    from eval.anthropic_driver import run_anthropic
    from test_contract import Session
    def crash(**kwargs):
        raise ValueError('provider driver bug')
    def driver(model, task):
        return run_anthropic(Session('a'), model,
            api=SimpleNamespace(messages=SimpleNamespace(create=crash)))
    run_panel(tmp_path, TASKS, ['anthropic:claude-sonnet-5-5'], driver, budget_usd=10)
    assert records([tmp_path]) == []
    failure = json.loads(next(tmp_path.glob('*/*/attempts/episode-*/attempt.json')).read_text())
    assert 'ValueError: provider driver bug' in failure['traceback']


@pytest.mark.parametrize('ended,error_type', [('no_submit', None), ('turn_limit', None), ('error', 'timeout')])
def test_completed_model_failures_are_never_retried(tmp_path, ended, error_type):
    def driver(model, task):
        return {**episode(model, task['task_id']), 'ended': ended, 'error_type': error_type, 'final': None}
    run_panel(tmp_path, TASKS, ['codex:m'], driver)
    run_panel(tmp_path, TASKS, ['codex:m'], lambda *a: pytest.fail('retried a model outcome'))
    assert len(records([tmp_path])) == 1


def test_paid_infrastructure_costs_survive_resume_without_double_charge(tmp_path):
    calls = []
    def broken(model, task, checkpoint):
        calls.append(task['task_id'])
        checkpoint({'usage': {'usd': 2.0}})
        raise RuntimeError('after billed response')
    for _ in range(3):
        run_panel(tmp_path, TASKS, ['anthropic:m'], broken, budget_usd=5, prior=2)
    assert calls == ['a', 'a']
    assert len((tmp_path/'costs.jsonl').read_text().splitlines()) == 2


def test_cli_bad_server_and_missing_tasks_never_launch(monkeypatch, tmp_path, capsys):
    from eval import preflight, run_eval
    monkeypatch.setattr(preflight, 'drivers', lambda models: {})
    monkeypatch.setenv('PERMIT_EVAL_TOKEN', 'private-test-token')
    monkeypatch.setattr(run_eval, 'run_panel', lambda *a, **kw: pytest.fail('episode launched'))
    arguments = ['run_eval', '--run', 'test', '--models', 'codex:m', '--server', 'http://test.invalid',
                 '--output-root', str(tmp_path)]
    monkeypatch.setattr('sys.argv', arguments + ['--tasks', 'missing-task'])
    with pytest.raises(SystemExit) as exc:
        run_eval.main()
    assert exc.value.code == 2 and 'task' in capsys.readouterr().err
    def unhealthy(*args):
        raise preflight.PreflightError('test server is unhealthy')
    monkeypatch.setattr(preflight, 'server', unhealthy)
    monkeypatch.setattr('sys.argv', arguments)
    with pytest.raises(SystemExit) as exc:
        run_eval.main()
    assert exc.value.code == 2 and 'unhealthy' in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('result', [None, {}, {'ended': 'unexpected'}, {'ended': 'no_submit', 'usage': None},
                                   {'ended': 'no_submit', 'usage': {'usd': float('nan')}}])
def test_malformed_driver_results_are_infrastructure_and_other_tasks_continue(tmp_path, result):
    tasks = TASKS + [{'task_id': 'b', 'split': 'eval', 'station': 'X'}]
    def driver(model, task):
        return result if task['task_id'] == 'a' else episode(model, task['task_id'])
    run_panel(tmp_path, tasks, ['codex:m'], driver)
    assert [r['task_id'] for _, r in records([tmp_path])] == ['b']
    attempt = json.loads(next(tmp_path.glob('*/a/attempts/episode-*/attempt.json')).read_text())
    assert attempt['status'] == 'infrastructure' and 'Traceback' in attempt['traceback']


def test_harness_interruption_saves_traceback_and_all_completed_checkpoints(tmp_path):
    def interrupted(model, task, checkpoint):
        checkpoint(episode(model, task['task_id']))
        raise KeyboardInterrupt('harness stopped')
    with pytest.raises(KeyboardInterrupt):
        run_panel(tmp_path, TASKS, ['codex:m'], interrupted)
    attempt = json.loads(next(tmp_path.glob('*/*/attempts/episode-*/attempt.json')).read_text())
    assert 'KeyboardInterrupt: harness stopped' in attempt['traceback']
    assert records([tmp_path])[0][1]['ended'] == 'submitted'


@pytest.mark.parametrize('status,finalized', [('running', True), ('infrastructure', False)])
def test_recovery_finishes_interrupted_publication_without_retry(tmp_path, status, finalized):
    from eval.attempts import recover
    root = tmp_path/model_slug('codex:m')/'a/attempts/episode-0001'
    value = episode()
    if finalized:
        value.update(final=None, ended='no_submit')
    value.update(attempt_complete=finalized, attempt_number=1)
    write_json(root/'checkpoint.json', value)
    write_json(root/'attempt.json', {'status': status, 'attempt_number': 1, 'model': 'codex:m',
               'task_id': 'a', 'retryable': False, 'submission_preserved': not finalized})
    recover(tmp_path)
    assert len(records([tmp_path])) == 1
    run_panel(tmp_path, TASKS, ['codex:m'], lambda *a: pytest.fail('publication failure replayed'))


def test_exception_tracebacks_are_redacted(tmp_path, monkeypatch):
    secret = 'test-api-secret-not-for-recording'
    monkeypatch.setenv('ANTHROPIC_API_KEY', secret)
    def driver(*args):
        raise RuntimeError(secret)
    run_panel(tmp_path, TASKS, ['codex:m'], driver)
    for path in tmp_path.rglob('*'):
        if path.is_file():
            assert secret not in path.read_text()
