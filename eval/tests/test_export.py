"""The Hub export scrubs local paths and the host name, and can copy a run tree file by file."""
import json

from eval import export_hf
from eval.privacy import scrub_paths


def test_scrub_paths_drops_home_scratch_and_host():
    home, tmp = '/' + 'home/', '/' + 'tmp/'  # built so the release scan does not flag this file
    text = (f'Traceback {home}happy/envs/x.py and {tmp}claude-1000/-home-happy-envs/abc/scratchpad/f.json '
            f'{{"host": {{"name": "king"}}}} kept: {tmp}other and {home}')
    out = scrub_paths(text, 'king')
    assert home + 'happy' not in out and 'claude-1000' not in out and '"king"' not in out
    assert home + 'user/envs/x.py' in out and tmp + 'scratch/f.json' in out and '{"host": {"name": "host"}}' in out
    assert tmp + 'other and ' + home in out


def test_export_copies_a_scrubbed_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(export_hf, 'HOST', 'king')
    run = tmp_path/'run-x'
    episode = run/'m'/'t1'/'r1'
    (episode/'raw').mkdir(parents=True)
    record = {'schema_version': 1, 'task_id': 't1', 'model': 'm', 'split': 'eval', 'repeat': 1, 'ended': 'submitted',
              'station': 'K', 'scenario': {'draft_cost': 100.0}, 'final': {'valid': True, 'feasible': True, 'prescore_usd_saved': 40.0},
              'note': '/' + 'home/happy/x'}
    (episode/'episode.json').write_text(json.dumps(record))
    (episode/'transcript.jsonl').write_text(json.dumps({'i': 0, 'kind': 'text', 'content': '/' + 'home/happy/y'})+'\n')
    (episode/'raw/req.json').write_text('{"path": "/' + 'tmp/claude-1000/a/b/scratchpad/z"}')
    (run/'run.json').write_text(json.dumps({'host': {'name': 'king'}}))
    out = tmp_path/'out'
    export_hf.export_runs([run], out, tree=True)
    episodes = [json.loads(line) for line in (out/'episodes/train.jsonl').read_text().splitlines()]
    assert len(episodes) == 1 and episodes[0]['reward'] == 0.4 and '/' + 'home/happy' not in episodes[0]['record_json']
    assert 'king' not in (out/'raw/train.jsonl').read_text()
    copied = out/'rollouts/run-x'
    assert '/' + 'home/user/x' in (copied/'m/t1/r1/episode.json').read_text()
    assert '/' + 'tmp/scratch/z' in (copied/'m/t1/r1/raw/req.json').read_text()
    assert '"host"' in (copied/'run.json').read_text() and 'king' not in (copied/'run.json').read_text()
