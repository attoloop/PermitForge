import copy
import json
from pathlib import Path

import pytest

from train.cloud.config import CloudConfig, stage_plan, remaining_seconds
from train.cloud.make_sft import build_dataset, episode_files, validate_catalog
from train.cloud.formatting import parse_calls, mask_spans
from train.cloud.run import run_guarded

ROOT = Path(__file__).resolve().parents[3]
SMOKE = ROOT / 'results/rollouts/wp2.0b-smoke'
from train.cloud.tests.checkout import requires_smoke_rollouts  # noqa: E402


def test_plan_is_pinned_bounded_and_covers_all_stages():
    config = CloudConfig(hourly_usd=5, recipe='grpo')
    plan = stage_plan(config, train_count=30)
    assert [s['stage'] for s in plan] == ['fields', 'base-eval', 'grpo', 'eval', 'export']
    assert config.model == 'Qwen/Qwen3.8-27B'
    assert config.revision == '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
    assert config.max_vram_bytes == 135_000_000_000
    assert config.field_jobs <= 4 and config.use_vllm is False
    assert plan[1]['episodes'] == plan[3]['episodes'] == 30
    assert all(s['estimated_usd'] == pytest.approx(s['estimated_seconds']/3600*5) for s in plan)
    assert all(s['effective_seconds'] <= s['wall_seconds'] for s in plan)


@pytest.mark.parametrize('kwargs', [dict(hourly_usd=0), dict(hourly_usd=float('nan')),
    dict(hourly_usd=2, field_jobs=5), dict(hourly_usd=2, use_vllm=True),
    dict(hourly_usd=2, max_vram_bytes=136_000_000_000)])
def test_config_rejects_unsafe_or_nonfinite_inputs(kwargs):
    with pytest.raises(ValueError):
        CloudConfig(**kwargs)


def test_caps_charge_resume_and_unknown_attempts():
    assert remaining_seconds(100, 1, 36, [{'charged_seconds': 20}]) == 80
    assert remaining_seconds(100, .5, 36, [{'charged_seconds': 20}]) == 30
    assert remaining_seconds(100, 1, 36, [{'reserved_seconds': 100}]) == 0


def test_supervisor_kills_entire_stage_on_time_cap(tmp_path):
    import sys
    result = run_guarded([sys.executable, '-c', 'import time; time.sleep(60)'],
                         tmp_path, seconds=.2, sync=None)
    assert result['status'] == 'cap_exceeded'
    assert result['elapsed_seconds'] < 5


@requires_smoke_rollouts
def test_smoke_handoff_is_deduplicated_and_train_only(tmp_path):
    rows, stats = build_dataset([SMOKE], ROOT/'envs/permit/tasks')
    assert len(rows) == 1
    assert stats['accepted'] == 1
    assert rows[0]['task_id'] == '074897d0f159-s47'
    assert rows[0]['split'] == 'train'
    assert rows[0]['station'] == 'KPIT'
    assert rows[0]['messages'][-2]['role'] == 'tool'
    assert len(list(episode_files([SMOKE]))) == 1


@requires_smoke_rollouts
def test_catalog_and_forged_record_leakage_fail_closed(tmp_path):
    with pytest.raises(ValueError, match='station'):
        validate_catalog([{'task_id':'a', 'split':'train', 'station':'X'},
                          {'task_id':'b', 'split':'eval', 'station':'X'}])
    record = json.loads(next(SMOKE.glob('*/**/episode.json')).read_text())
    for key, value in [('task_id', '003bd328a392-s47'), ('station', 'KJAX')]:
        forged = copy.deepcopy(record)
        forged[key] = value
        path = tmp_path/'episode.json'
        path.write_text(json.dumps(forged))
        with pytest.raises(ValueError, match='identity|station|task'):
            build_dataset([path], ROOT/'envs/permit/tasks')


@requires_smoke_rollouts
def test_data_filters_failures_and_no_submit(tmp_path):
    record = json.loads(next(SMOKE.glob('*/**/episode.json')).read_text())
    for change in [dict(ended='no_submit'), dict(grading_failed=True), dict(synthetic=True),
                   dict(final={'feasible':False, 'prescore_usd_saved':-1})]:
        path = tmp_path/'episode.json'
        path.write_text(json.dumps({**record, **change}))
        rows, stats = build_dataset([path], ROOT/'envs/permit/tasks')
        assert not rows and stats['rejected'] == 1


def test_xml_tool_roundtrip_rejects_wrong_tools_and_arguments():
    text = '<tool_call>\n<function=submit_design>\n<parameter=design>\n{"S1":{"control":"S1_none","height_m":20}}\n</parameter>\n</function>\n</tool_call>'
    assert parse_calls(text)[0]['function']['arguments']['design']['S1']['height_m'] == 20
    for bad in [text.replace('submit_design','unknown'), text.replace('parameter=design','parameter=path'),
                text.replace('</tool_call>',''), text + text.replace('height_m','height')]:
        with pytest.raises(ValueError):
            parse_calls(bad)


def test_mask_only_assistant_bodies_and_eos():
    text = '<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nUSER<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nCALL<|im_end|>\n<|im_start|>user\nTOOL<|im_end|>\n'
    spans = mask_spans(text)
    trained = ''.join(text[a:b] for a,b in spans)
    assert trained == 'CALL<|im_end|>'
    assert 'SYS' not in trained and 'TOOL' not in trained


def running(pid):
    import psutil
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def assert_killed(pid):
    import psutil
    try:
        assert not running(pid)
    finally:
        if running(pid):
            psutil.Process(pid).kill()


def test_supervisor_kills_detached_native_descendants(tmp_path):
    import sys
    child_file = tmp_path/'child.pid'
    program = ('import subprocess,sys,time; from pathlib import Path; '
               'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],start_new_session=True); '
               f'Path({str(child_file)!r}).write_text(str(p.pid)); time.sleep(60)')
    result = run_guarded([sys.executable,'-c',program],tmp_path,seconds=2,sync=None)
    assert_killed(int(child_file.read_text()))
    assert result['status'] == 'cap_exceeded'


def test_supervisor_kills_detached_descendants_orphaned_between_scans(tmp_path):
    import sys
    child_file = tmp_path/'child.pid'
    # The shell exits within milliseconds, so its detached child is orphaned before a scan can see it.
    program = ('import subprocess,time; '
               f'subprocess.run(["sh","-c","setsid sleep 60 & echo $! > {child_file}"]); time.sleep(60)')
    result = run_guarded([sys.executable,'-c',program],tmp_path,seconds=2,sync=None)
    assert_killed(int(child_file.read_text()))
    assert result['status'] == 'cap_exceeded'


def test_eval_summary_does_not_turn_infrastructure_into_zero():
    from train.cloud.evaluation import summarize_records
    rows = [dict(task_id='a',station='A',reward=.5,ended='submitted'),
            dict(task_id='a',station='A',reward=0,ended='no_submit'),
            dict(task_id='b',station='B',reward=0,ended='error',error_type='infrastructure')]
    result = summarize_records(rows,['a','b'],repeats=2)
    assert result['task_mean_saved_fraction'] == .25
    assert result['graded_tasks'] == 1 and result['infrastructure_episodes'] == 1
    assert result['complete'] is False


def test_checkpoint_resume_ignores_partial_writes(tmp_path):
    from train.cloud.training import latest_checkpoint
    for number in [1,2]:
        folder = tmp_path/f'checkpoint-{number}'
        folder.mkdir()
        for filename in ['trainer_state.json','optimizer.pt','adapter_model.safetensors','scheduler.pt','rng_state.pth']:
            (folder/filename).write_text('x')
    (tmp_path/'checkpoint-1/.complete').write_text('complete')
    assert latest_checkpoint(tmp_path) == tmp_path/'checkpoint-1'


def test_fields_preflight_needs_working_set_not_whole_cohort(tmp_path):
    from train.cloud.fields import check_capacity
    check_capacity([100,200,300], cache_bytes=350, free_bytes=400)
    with pytest.raises(ValueError):
        check_capacity([100,200,300], cache_bytes=299, free_bytes=400)
    with pytest.raises(ValueError):
        check_capacity([300], cache_bytes=350, free_bytes=200)
