"""Review regressions and launch effort; only cached data and local fake providers."""
import copy
from contextlib import nullcontext
import json
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from eval.common import ROOT, model_slug, records, write_json
from eval.run_eval import run_panel
from eval.tests.test_saved_fraction_repeats import episode


@pytest.mark.parametrize('occupied', [
    'r1/attempts/episode-0001/attempt.json', 'r1/checkpoint.json', 'r1/episode.json',
    'r1.json', 'attempts/episode-0001/attempt.json', 'episode.json', '../a.json',
    'pending_submission.json', 'raw/request.json',
])
def test_import_refuses_any_occupied_repeat_without_writes(tmp_path, occupied):
    from eval.import_runs import import_completed
    tasks = [{'task_id': t, 'split': 'eval', 'station': 'X'} for t in ('a', 'b')]
    source, target = tmp_path/'source', tmp_path/'target'
    run_panel(source, tasks, ['codex:test'], lambda m, t: episode(model=m, task=t['task_id']))
    path = target/model_slug('codex:test')/'a'/occupied
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"submit_attempted": true, "retryable": false}')
    before = {p.relative_to(target): p.read_bytes() for p in target.rglob('*') if p.is_file()}
    manifest = json.loads((source/'run.json').read_text())
    with pytest.raises(ValueError, match='occupied.*repeat'):
        import_completed(target, [source], manifest, 1)
    assert before == {p.relative_to(target): p.read_bytes() for p in target.rglob('*') if p.is_file()}


def test_import_preserves_held_submission_and_allows_identical_import_resume(tmp_path):
    tasks = [{'task_id': 'a', 'split': 'eval', 'station': 'X'}]
    source, target = tmp_path/'source', tmp_path/'target'
    run_panel(source, tasks, ['codex:test'], lambda m, t: episode(model=m))
    def lost(model, task, checkpoint):
        checkpoint({'submit_attempted': True, 'pending_submission': {'design': {}}})
        raise ConnectionError('held submission')
    def never(*args):
        pytest.fail('import must not run a driver')
    run_panel(target, tasks, ['codex:test'], lost)
    root = target/model_slug('codex:test')/'a/r1'
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='occupied.*repeat'):
        run_panel(target, tasks, ['codex:test'], never, import_runs=[source])
    assert not records([target])
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    imported = tmp_path/'imported'
    run_panel(imported, tasks, ['codex:test'], never, import_runs=[source])
    before = {p: p.read_bytes() for p in imported.rglob('*') if p.is_file() and 'a' in p.parts}
    run_panel(imported, tasks, ['codex:test'], never, import_runs=[source])
    assert all(p.read_bytes() == data for p, data in before.items())


def test_documented_selection_reproduces_active_bytes(tmp_path):
    text = (ROOT/'DESIGN.md').read_text().replace('\\\n', ' ')
    command = re.search(r'uv run[^\n]*python -m eval\.(?:subsets|select_tasks)[^\n]*', text)[0]
    argv = shlex.split(command)
    output = tmp_path/'selection.json'
    argv[argv.index('--output')+1] = str(output)
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == (ROOT/'results/subsets.json').read_bytes()


def test_retired_selector_has_safe_default_and_refuses_active_path(tmp_path, monkeypatch):
    from eval import subsets
    monkeypatch.setattr(subsets, 'ROOT', tmp_path)
    monkeypatch.setattr(subsets, 'select_subsets', lambda *a, **kw: {'eval': [], 'train': [], 'repeats': 4})
    monkeypatch.setattr(sys, 'argv', ['subsets'])
    subsets.main()
    assert not (tmp_path/'results/subsets.json').exists()
    assert (tmp_path/'results/subsets-wp2.7.json').exists()
    monkeypatch.setattr(sys, 'argv', ['subsets', '--output', str(tmp_path/'results/subsets.json')])
    with pytest.raises(SystemExit):
        subsets.main()


@pytest.mark.parametrize('mutation', ['historical', 'unversioned', 'truncated'])
def test_launch_rejects_nonfinal_cohort_before_provider_work(tmp_path, monkeypatch, mutation):
    from eval import run_eval
    value = json.loads((ROOT/'results/subsets.json').read_text())
    if mutation == 'historical':
        value['protocol'] = 'WP2.7'
    elif mutation == 'unversioned':
        value.pop('protocol')
    else:
        value['eval'].pop()
    path = tmp_path/'subsets.json'
    write_json(path, value)
    monkeypatch.setattr(run_eval.preflight, 'drivers', lambda *a: pytest.fail('provider preflight'))
    monkeypatch.setattr(sys, 'argv', ['run_eval', '--run', 'bad', '--models', 'codex:gpt-6.1-sol',
                                    '--subsets', str(path), '--dry-run'])
    with pytest.raises(SystemExit):
        run_eval.main()


@pytest.mark.parametrize('effort,native,prior', [('medium', 'medium', 2.), ('high', 'high', 3.5), ('xhigh', 'max', 5.)])
def test_launch_effort_dry_run_manifest_hash_and_episode(tmp_path, monkeypatch, capsys, effort, native, prior):
    from eval import run_eval
    from eval.settings import MODELS
    monkeypatch.setattr(run_eval.preflight, 'drivers', lambda *a: pytest.fail('provider preflight'))
    monkeypatch.setattr(sys, 'argv', ['run_eval', '--run', 'dry', '--models', *MODELS, '--effort', effort,
        '--subsets', str(ROOT/'results/subsets.json'), '--split', 'both', '--repeats', '4',
        '--budget-usd', '190', '--dry-run', '--output-root', str(tmp_path)])
    run_eval.main()
    result = json.loads(capsys.readouterr().out)
    settings = result['settings']
    assert settings['effort'] == effort and settings['max_tokens'] == 32768
    assert settings['prior_episode_usd'] == prior
    assert settings['models'][MODELS[0]]['effort'] == native
    assert settings['models'][MODELS[-1]]['effort'] == effort
    assert 'assumption' in result['projection_prior']['note'].lower()
    assert result['projection_prior']['projected_remaining_paid_usd'] == 80*sum(m.startswith('anthropic:') for m in MODELS)*prior
    assert not list(tmp_path.iterdir())
    hashes = {run_eval.provenance(list(MODELS), {'effort': level}, 1, 190)['protocol_sha256']
              for level in ('medium', 'high', 'xhigh')}
    assert len(hashes) == 3
    run_panel(tmp_path/'run', [{'task_id': 'a', 'station': 'X', 'split': 'eval'}],
              [MODELS[-1]], lambda m, t: episode(model=m), settings={'effort': effort})
    manifest = json.loads((tmp_path/'run/run.json').read_text())
    assert manifest['sampling']['effort'] == effort
    assert manifest['sampling']['models'][MODELS[-1]]['effort'] == effort
    assert records([tmp_path/'run'])[0][1]['settings']['effort'] == effort
    assert MODELS[-1]+'@'+effort in (tmp_path/'run/board.md').read_text()


@pytest.mark.parametrize('effort,native', [('medium', 'medium'), ('high', 'high'), ('xhigh', 'max')])
@pytest.mark.parametrize('model', ['claude-opus-5-5', 'claude-sonnet-5-5'])
def test_anthropic_effort_capture_and_thinking_roundtrip(tmp_path, effort, native, model):
    from eval.anthropic_driver import run_anthropic
    from eval.capture import EpisodeCapture
    from eval.tests.test_contract import Session, api_response
    calls = []
    blocks = [{'type': 'thinking', 'thinking': 'summary', 'signature': 'signature'},
              {'type': 'redacted_thinking', 'data': 'opaque'},
              {'type': 'tool_use', 'id': 'c', 'name': 'get_scenario', 'input': {}}]
    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return (SimpleNamespace(content=blocks, usage=SimpleNamespace(input_tokens=1, output_tokens=1))
                if len(calls) == 1 else api_response())
    session = Session()
    session.capture = EpisodeCapture(tmp_path/'episode')
    result = run_anthropic(session, 'anthropic:'+model, effort=effort,
                          api=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert result['settings']['effort'] == native and result['settings']['launch_effort'] == effort
    assert calls[1]['messages'][1] == {'role': 'assistant', 'content': blocks}
    for path in (session.capture.root/'raw').glob('*request.json'):
        request = json.loads(path.read_text())
        assert request['output_config'] == {'effort': native}
        assert request['thinking'] == {'type': 'adaptive', 'display': 'summarized'}
        assert request['max_tokens'] == 32768


def test_board_rejects_mixed_efforts():
    from eval.report import build_report
    a, b = episode(), episode(repeat=2)
    a['settings'], b['settings'] = {'effort': 'medium'}, {'effort': 'high'}
    with pytest.raises(ValueError, match='different efforts'):
        build_report([a, b])


def test_board_rejects_different_launch_efforts_across_models():
    from eval.report import build_report
    a, b = episode(model='codex:gpt-6.1-sol'), episode(model='anthropic:claude-opus-5-5')
    a['settings'], b['settings'] = {'effort': 'medium'}, {'effort': 'high'}
    with pytest.raises(ValueError, match='different efforts'):
        build_report([a, b])
    a['settings'], b['settings'] = {'effort': 'xhigh'}, {'effort': 'max'}
    assert len(build_report([a, b])['models']) == 2


def test_shared_episode_fixture_uses_current_reward():
    from eval.tests.test_contract import episode as shared_episode
    for value in (shared_episode(), episode()):
        assert 'anchors' not in value['final']
        assert value['final']['reward'] == value['final']['prescore_usd_saved']/value['scenario']['draft_cost']


@pytest.mark.parametrize('effort', ['medium', 'high', 'xhigh'])
def test_launcher_passes_effort_to_every_driver(monkeypatch, effort):
    from eval import run_eval, transport
    from eval.settings import MODELS
    calls = []
    def provider(session, model, **kwargs):
        calls.append((model, kwargs['effort'], kwargs['max_tokens']))
    monkeypatch.setattr(run_eval.preflight, 'drivers', lambda _: {p: provider for p in ('codex', 'anthropic', 'claudecode')})
    monkeypatch.setattr(run_eval.preflight, 'server', lambda *a: None)
    monkeypatch.setattr(transport, 'Session', lambda *a, **kw: nullcontext(object()))
    monkeypatch.setattr(transport, 'recording_app', lambda *a, **kw: None)
    monkeypatch.setattr(transport, 'serve', lambda _: nullcontext('http://fake.invalid'))
    def panel(run, tasks, models, driver, **kwargs):
        for model in models:
            driver(model, tasks[0], None, None, None)
    monkeypatch.setattr(run_eval, 'run_panel', panel)
    monkeypatch.setattr(sys, 'argv', ['run_eval', '--run', 'fake', '--models', *MODELS,
        '--effort', effort, '--split', 'train', '--limit', '1', '--budget-usd', '190'])
    run_eval.main()
    assert calls == [(m, effort, 32768) for m in MODELS]


def test_corrupt_completed_import_is_refused_clearly(tmp_path):
    from eval.import_runs import import_completed
    source, target = tmp_path/'source', tmp_path/'target'
    tasks = [{'task_id': 'a', 'split': 'eval', 'station': 'X'}]
    run_panel(source, tasks, ['codex:test'], lambda m, t: episode(model=m))
    path = target/model_slug('codex:test')/'a/r1/episode.json'
    path.parent.mkdir(parents=True)
    path.write_text('partial JSON')
    with pytest.raises(ValueError, match='occupied.*repeat'):
        import_completed(target, [source], json.loads((source/'run.json').read_text()), 1)
    assert path.read_text() == 'partial JSON'
