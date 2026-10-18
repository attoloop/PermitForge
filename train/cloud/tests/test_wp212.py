import json
from pathlib import Path

import pytest

from train.cloud.config import CloudConfig, stage_plan

ROOT = Path(__file__).resolve().parents[3]


def test_h200_bf16_direct_grpo_and_smoke_envelope():
    cfg = CloudConfig(hourly_usd=5)
    assert cfg.flavor == 'h200' and cfg.precision == 'bf16'
    assert cfg.max_vram_bytes == 135_000_000_000
    assert cfg.max_completion_length >= 9728  # Local eight-call sizing floor, train/DESIGN.md.
    rows = stage_plan(cfg)
    assert [r['stage'] for r in rows] == ['smoke']
    assert rows[0]['usd_cap'] <= 15 and rows[0]['wall_seconds'] <= 10800
    full = stage_plan(CloudConfig(hourly_usd=5, recipe='grpo'))
    assert [r['stage'] for r in full] == ['fields', 'base-eval', 'grpo', 'eval', 'export']


def test_selection_is_repeatable_station_balanced_and_smoke_is_small():
    from permit_openenv.tasks import load_pack
    from train.cloud.selection import selection_manifest
    pack = load_pack(ROOT/'envs/permit/tasks')
    selection = selection_manifest(pack, ROOT/'results/subsets.json')
    assert selection == selection_manifest(pack, ROOT/'results/subsets.json')
    assert len(selection['train']) == 30
    stations = [pack.get(t).record['station'] for t in selection['train']]
    assert len(set(stations)) == 10
    assert all(stations.count(s) == 3 for s in stations)
    assert len(selection['smoke_train']) == 2 and len(selection['smoke_eval']) == 3
    assert set(selection['smoke_train']) <= set(selection['train'])
    assert set(selection['smoke_eval']) <= set(selection['eval'])
    assert len({pack.get(t).record['station'] for t in selection['smoke_eval']}) == 3
    assert not set(selection['train']) & set(selection['eval'])


def test_check_plan_keeps_full_scheduler_target_and_omits_final_eval(capsys):
    from train.cloud.run import main
    main(['--plan', '--recipe', 'grpo', '--grpo-steps', '100', '--stop-after-step', '20'])
    plan = json.loads(capsys.readouterr().out)
    assert plan['config']['hourly_usd'] == 5
    assert plan['config']['grpo_steps'] == 100
    assert plan['stop_after_step'] == 20
    assert [r['stage'] for r in plan['stages']] == ['fields', 'base-eval', 'grpo']
    assert next(r for r in plan['stages'] if r['stage'] == 'grpo')['optimizer_steps_this_invocation'] == 20
    assert plan['provider_timeout_seconds'] > sum(r['effective_seconds'] for r in plan['stages'])


def test_smoke_report_gates_on_model_behavior_and_gradients():
    from train.cloud.smoke import smoke_summary
    rows = [dict(task_id=str(i), ended='submitted', reward=.1, history=[
        {'tool':'check_design', 'result':{'valid':True}}]) for i in range(3)]
    training = dict(status='passed', weights_changed=True, adapter_reload_equal=True,
        optimizer_steps=2, seconds_per_step=100, groups=[{'reward_std':.1}], worst_case={'fits':True})
    result = smoke_summary(rows, training, {'peak_device_bytes':90}, elapsed_seconds=600, hourly_usd=5,
                           kernels={'status':'passed'})
    assert result['ready_for_grpo'] is True
    assert result['projections']['20']['grpo_compute_usd'] == pytest.approx(20*100/3600*5)
    assert set(result['projections']) == {'20','50','100'}
    assert result['base']['submitted'] == 3 and result['base']['positive_reward'] == 3
    assert smoke_summary([], training, {}, elapsed_seconds=1, hourly_usd=5)['ready_for_grpo'] is False
    training['weights_changed'] = False
    assert smoke_summary(rows, training, {}, elapsed_seconds=1, hourly_usd=5)['ready_for_grpo'] is False


def test_bf16_loader_has_no_quantization_and_nf4_is_explicit():
    import torch
    from train.cloud.training import model_load_kwargs
    bf16 = model_load_kwargs(CloudConfig(hourly_usd=5))
    assert bf16['dtype'] == torch.bfloat16
    assert bf16.get('quantization_config') is None
    nf4 = model_load_kwargs(CloudConfig(hourly_usd=5, precision='nf4'))
    assert nf4['quantization_config'].load_in_4bit is True


def test_trackio_is_explicit_and_preserves_live_space():
    from train.cloud.training import training_options
    assert training_options(CloudConfig(hourly_usd=5), 'grpo', Path('out'))['report_to'] == 'none'
    cfg = CloudConfig(hourly_usd=5, trackio_space='attoloop/chosen-space', trackio_bucket='attoloop/metrics')
    opts = training_options(cfg, 'grpo', Path('out'))
    assert opts['report_to'] == 'trackio'
    assert opts['trackio_space_id'] == 'attoloop/chosen-space'
    assert opts['trackio_static_space_id'] is False


def test_bundle_verifies_contents_without_sft_data(tmp_path):
    from train.cloud.bundle import create_bundle, verify_bundle
    destination = tmp_path/'input.tar.gz'
    metadata = create_bundle(ROOT, ROOT/'results/subsets.json', destination)
    assert not any('sft.jsonl' in p for p in metadata['files'])
    result = verify_bundle(destination, metadata['bundle_sha256'])
    assert result['verified_files'] == len(metadata['files'])
    assert result['selection']['train'] and result['selection']['smoke_eval']
    with pytest.raises(ValueError, match='SHA256'):
        verify_bundle(destination, '0'*64)


def test_continuation_requires_rng_and_scheduler_and_stops_without_changing_target(tmp_path):
    from types import SimpleNamespace
    from train.cloud.training import latest_checkpoint, stop_callback, training_options
    folder = tmp_path/'checkpoint-20'
    folder.mkdir()
    for filename in ['trainer_state.json','optimizer.pt','adapter_model.safetensors','.complete']:
        (folder/filename).write_text('x')
    assert latest_checkpoint(tmp_path) is None
    for filename in ['rng_state.pth','scheduler.pt']:
        (folder/filename).write_text('x')
    assert latest_checkpoint(tmp_path) == folder
    callback = stop_callback(20)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    callback.on_step_end(None,SimpleNamespace(global_step=19),control)
    assert control.should_training_stop is False
    callback.on_step_end(None,SimpleNamespace(global_step=20),control)
    assert control.should_training_stop and control.should_save
    assert training_options(CloudConfig(recipe='grpo',grpo_steps=100),'grpo',tmp_path)['max_steps'] == 100


def test_qualification_rejects_changed_model_or_failed_smoke(tmp_path):
    from train.cloud.smoke import smoke_gate
    current = dict(config=CloudConfig().record(),hashes={'code':'abc'},selection={'train':['a']})
    (tmp_path/'smoke').mkdir()
    (tmp_path/'run.json').write_text(json.dumps(current))
    path = tmp_path/'smoke/metrics.json'
    path.write_text(json.dumps(dict(status='passed',ready_for_grpo=True)))
    with pytest.raises(ValueError,match='passed smoke'):
        smoke_gate(tmp_path,current)
    path.write_text(json.dumps(dict(status='passed',ready_for_grpo=True,supervisor_status='complete')))
    for name in ('kernels','gpu'):
        (tmp_path/f'smoke/{name}.json').write_text(json.dumps({'status':'passed'}))
    smoke_gate(tmp_path,current)
    changed = json.loads(json.dumps(current))
    changed['config']['precision'] = 'nf4'
    with pytest.raises(ValueError,match='passed smoke'):
        smoke_gate(tmp_path,changed)
    path.write_text(json.dumps(dict(status='failed',ready_for_grpo=False)))
    with pytest.raises(ValueError,match='passed smoke'):
        smoke_gate(tmp_path,current)


def test_smoke_sequence_uses_only_five_tasks_untrained_base_then_direct_grpo(tmp_path, monkeypatch):
    from permit_openenv.tasks import load_pack
    from train.cloud.selection import selection_manifest
    from train.cloud.smoke import run_smoke
    from train.cloud.run import write_json
    import train.cloud.fields as fields
    import train.cloud.training as training
    import train.cloud.evaluation as evaluation
    pack = load_pack(ROOT/'envs/permit/tasks')
    chosen = selection_manifest(pack,ROOT/'results/subsets.json')
    record = dict(train_tasks=chosen['smoke_train'],eval_tasks=chosen['smoke_eval'],
                  fields_dir=str(tmp_path/'cache'),tasks_dir=str(ROOT/'envs/permit/tasks'),sft_data='unused')
    order = []
    def warm(tasks, *args):
        order.append('fields')
        assert {t.task_id for t in tasks} == set(chosen['smoke_train']+chosen['smoke_eval'])
    monkeypatch.setattr(fields,'warm',warm)
    monkeypatch.setattr(fields,'CloudFields',lambda *a,**kw:None)
    def load(cfg, adapter=None):
        order.append('base-load')
        assert adapter is None
        return object(),object()
    monkeypatch.setattr(training,'load_model',load)
    def evaluate(cfg, model, tokenizer, pack, ids, output, **kwargs):
        order.append('base-eval')
        assert cfg.eval_repeats == 1 and len(ids) == 3 and kwargs['adapter'] is None
        for tid in ids:
            write_json(output/tid/'episode.json',dict(task_id=tid,ended='submitted',reward=.1,
                steps=[dict(tool='check_design',result={'valid':True})]))
    monkeypatch.setattr(evaluation,'evaluate',evaluate)
    def train(cfg, kind, output, **kwargs):
        order.append('grpo')
        assert kind == 'grpo' and kwargs['smoke'] is True and kwargs.get('adapter') is None
        assert len(kwargs['tasks']) == 2
        return dict(status='passed',weights_changed=True,adapter_reload_equal=True,worst_case={'fits':True},
                    optimizer_steps=2,seconds_per_step=100,groups=[{'reward_std':.1}],
                    tasks_seen=chosen['smoke_train'])
    monkeypatch.setattr(training,'train',train)
    write_json(tmp_path/'kernels.json',{'status':'passed'})
    run_smoke(CloudConfig(),record,pack,tmp_path)
    assert order == ['fields','base-load','base-eval','grpo']
    assert json.loads((tmp_path/'metrics.json').read_text())['ready_for_grpo'] is True


def test_supervisor_continues_twenty_to_one_hundred_without_repeating_base(tmp_path, monkeypatch):
    import train.cloud.run as runner
    import train.cloud.smoke as smoke
    import train.cloud.storage as storage
    stages = []
    monkeypatch.setattr(storage,'sync',lambda *a:None)
    monkeypatch.setattr(smoke,'smoke_gate',lambda *a:None)
    def guarded(command, output, **kwargs):
        stage = command[command.index('--stage')+1]
        stages.append(stage)
        if stage == 'grpo':
            step = int(command[command.index('--stop-after-step')+1]) if '--stop-after-step' in command else 100
            runner.write_json(output/'metrics.json',{'optimizer_steps':step})
        return dict(status='complete',exit_code=0,elapsed_seconds=.1)
    monkeypatch.setattr(runner,'run_guarded',guarded)
    args = ['--execute','--recipe','grpo','--grpo-steps','100','--output',str(tmp_path/'run'),
            '--artifact-uri','hf://buckets/attoloop/test/check','--qualification-run',str(tmp_path/'smoke')]
    runner.main(args+['--stop-after-step','20'])
    assert stages == ['fields','base-eval','grpo']
    first = json.loads((tmp_path/'run/run.json').read_text())
    runner.main(args)
    assert stages == ['fields','base-eval','grpo','grpo','eval','export']
    assert json.loads((tmp_path/'run/run.json').read_text()) == first


def test_timed_out_smoke_still_has_a_failed_report(tmp_path):
    from train.cloud.smoke import finalize_report
    finalize_report(tmp_path,'cap_exceeded',120,5)
    report = json.loads((tmp_path/'metrics.json').read_text())
    assert report['ready_for_grpo'] is False
    assert report['supervisor_status'] == 'cap_exceeded'
    assert (tmp_path/'REPORT.md').is_file()


@pytest.mark.parametrize('trackio', [False, True])
@pytest.mark.parametrize('install_fails', [False, True])
def test_bootstrap_installs_one_frozen_stack_and_keeps_trackio_opt_in(tmp_path,monkeypatch,trackio,install_fails):
    import subprocess
    import sys
    import train.cloud.hf_job as launcher
    root = tmp_path/'permitforge'
    (root/'train/cloud').mkdir(parents=True)
    monkeypatch.setattr(launcher,'unpack',lambda *a:root)
    monkeypatch.chdir(tmp_path)
    for name in ('PERMIT_JOB_DEADLINE_UNIX','PERMIT_BOOTSTRAP_REPORT','USE_HUB_KERNELS'):
        monkeypatch.setenv(name,'')
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs['timeout'] <= 900
        if install_fails:
            raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(launcher.subprocess,'run',run)
    class Finished(Exception):
        pass
    def execute(*args):
        raise Finished()
    monkeypatch.setattr(launcher.os,'execvp',execute)
    monkeypatch.setattr(sys,'argv',['hf_job.py','--bundle','unused','--bundle-sha256','unused',
        '--','--execute','--recipe=smoke'] + (['--trackio-space=Attoloop/named'] if trackio else []))
    if install_fails:
        with pytest.raises(subprocess.CalledProcessError):
            launcher.main()
        assert len(calls) == 1
        assert not (root/'train/cloud/bootstrap.json').exists()
        return
    with pytest.raises(Finished):
        launcher.main()
    assert len(calls) == 2  # Frozen install, then bundle validation.
    assert calls[0] == ['uv','sync','--project','train/cloud','--frozen','--no-dev'] + (
        ['--extra','metrics'] if trackio else [])
    assert json.loads((root/'train/cloud/bootstrap.json').read_text())['stack_install'] == 'installed'
    assert float(launcher.os.environ['PERMIT_JOB_DEADLINE_UNIX']) > 0
