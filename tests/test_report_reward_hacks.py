"""Reward ledger requirements: uv run pytest tests/test_report_reward_hacks.py -q."""

import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

import pytest

from convert.report_reward_hacks import default_database, encode, extract, summarize


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def archive(tmp_path):
    database = tmp_path / 'archive.db'
    with sqlite3.connect(database) as conn:
        conn.executescript('''
            CREATE TABLE defects_v2 (
                defect_id TEXT, environment_id INTEGER, fingerprint TEXT,
                category TEXT, component TEXT, status TEXT, detail TEXT,
                root_cause TEXT, first_seen_attempt INTEGER, last_seen_attempt INTEGER,
                resolved_by_attempt INTEGER, resolved_at TEXT, regression_count INTEGER,
                waiver_reason TEXT, deferred_scope TEXT, resolution_paths_json TEXT,
                closure_checks_json TEXT, closure_evidence_digest TEXT,
                suggested_remedy_text TEXT, detail_history_json TEXT
            );
            CREATE TABLE defect_events_v2 (
                id INTEGER, defect_id TEXT, attempt_id INTEGER, old_status TEXT,
                new_status TEXT, evidence_digest TEXT, created_at TEXT
            );
            CREATE TABLE attempts_v2 (id INTEGER, environment_id INTEGER, node_id INTEGER);
            CREATE TABLE workflow_nodes (id INTEGER, environment_id INTEGER, node_key TEXT);
            INSERT INTO workflow_nodes VALUES (1,47,'ADVERSARIAL_CODEX'),(2,47,'TARGETED_REPAIR');
            INSERT INTO attempts_v2 VALUES (10,47,1),(11,47,2),(20,48,1);
            INSERT INTO defects_v2 VALUES (
                'DEF-abc',47,'fingerprint','reward_integrity','verifier','RESOLVED',
                'invalid file', 'Admission fixed',
                10,10,11,'date',2,NULL,NULL,'["tests/grade.py"]','["reject bad input"]',
                'digest','Suggested, not proof','[]'
            );
            INSERT INTO defects_v2 VALUES (
                'DEF-other',48,'other','other','other','OPEN','other',NULL,
                20,20,NULL,NULL,0,NULL,NULL,'[]','[]',NULL,'','[]'
            );
            INSERT INTO defect_events_v2 VALUES
                (2,'DEF-abc',11,'OPEN','RESOLVED','digest','date'),
                (1,'DEF-abc',10,NULL,'OPEN','digest','date'),
                (3,'DEF-other',20,NULL,'OPEN','digest','date');
        ''')
        conn.execute('UPDATE defects_v2 SET detail=? WHERE environment_id=47',
                     ('file /' + 'home/example/work/submission was invalid',))
    probes = tmp_path / 'probes.json'
    probes.write_text(json.dumps([
        {'node': 'adversarial_codex', 'attempt': '10', 'probe': 'file-check'},
        {'node': 'adversarial_codex', 'attempt': '10', 'probe': 'file-check'},
    ]))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'source_git': {'head': 'commit'}, 'generated': []}))
    return database, probes, manifest


def test_extract_all_fields_events_and_probe_occurrences(archive):
    before = [path.read_bytes() for path in archive]
    data = extract(*archive)
    assert [row['id'] for row in data['defects']] == ['DEF-abc']
    defect = data['defects'][0]
    assert defect['fingerprint'] == 'fingerprint'
    assert defect['first_seen'] == {'attempt': 10, 'node': 'ADVERSARIAL_CODEX'}
    assert defect['last_seen'] == defect['first_seen']
    assert defect['resolution']['attempt'] == 11
    assert defect['resolution']['node'] == 'TARGETED_REPAIR'
    assert defect['resolution']['summary'] == 'Admission fixed'
    assert defect['resolved_at'] == 'date'
    assert defect['regression_count'] == 2
    assert defect['waiver_reason'] is None
    assert [event['id'] for event in defect['events']] == [1, 2]
    assert len(data['probes']) == 2  # Occurrences are never silently deduplicated.
    assert data['summary']['probe_occurrences'] == 2
    assert data['summary']['unique_probe_names'] == 1
    assert data['summary']['probe_attempts'] == 1
    assert data['summary'] == summarize(data)
    assert encode(extract(*archive)) == encode(data)
    assert [path.read_bytes() for path in archive] == before
    assert ('/' + 'home/') not in encode(data).decode()


def test_unresolved_or_missing_attempt_is_preserved(archive):
    with sqlite3.connect(archive[0]) as conn:
        conn.execute("UPDATE defects_v2 SET first_seen_attempt=NULL, status='DEFERRED', "
                     "root_cause=NULL, resolved_by_attempt=NULL, resolved_at=NULL, "
                     "deferred_scope='DISCOVERY' WHERE environment_id=47")
    defect = extract(*archive)['defects'][0]
    assert defect['first_seen'] == {'attempt': None, 'node': None}
    assert defect['resolution']['summary'] is None
    assert defect['resolution']['attempt'] is None
    assert defect['status'] == 'DEFERRED' and defect['deferred_scope'] == 'DISCOVERY'


def test_refuses_journal_instead_of_ignoring_uncheckpointed_records(archive):
    Path(str(archive[0]) + '-wal').write_bytes(b'pending')
    with pytest.raises(ValueError, match='journal|WAL'):
        extract(*archive)


def test_check_detects_stale_output_without_rewriting(archive, tmp_path):
    out = tmp_path / 'evidence.json'
    out.write_bytes(b'stale\n')
    completed = subprocess.run([
        sys.executable, str(ROOT / 'convert/report_reward_hacks.py'), '--check',
        '--database', str(archive[0]), '--probes', str(archive[1]),
        '--manifest', str(archive[2]), '--out', str(out),
    ], capture_output=True, text=True)
    assert completed.returncode == 1
    assert 'stale' in completed.stderr
    assert out.read_bytes() == b'stale\n'


def test_committed_json_regenerates_byte_identically():
    if not default_database().exists() or not (ROOT / 'convert/v1-src').exists():
        pytest.skip('the historical database and v1 extraction are not in a public checkout')
    data = extract(default_database(), ROOT / 'convert/v1-src/evidence/adversarial_probes.json',
                   ROOT / 'convert/v1-src/EXTRACT_MANIFEST.json')
    assert encode(data) == (ROOT / 'results/evidence/reward-hacks.json').read_bytes()
    completed = subprocess.run([sys.executable, str(ROOT / 'convert/report_reward_hacks.py'),
                                '--check'], cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_public_ledger_references_and_discloses_unresolved_findings():
    text = (ROOT / 'results/reward-hacks.md').read_text()
    data = json.loads((ROOT / 'results/evidence/reward-hacks.json').read_text())
    ids = {row['id'] for row in data['defects']}
    cited = set(re.findall(r'DEF-[a-z0-9]+', text))
    assert cited and cited <= ids
    assert cited == ids
    assert {row['id'] for row in data['defects'] if row['status'] != 'RESOLVED'} <= cited
    assert not re.search(r'\bWP\d|\bCodex\b|\boperator\b', text, re.I)
    for value in (text, json.dumps(data)):
        assert not any(marker in value for marker in ('/' + 'home/', '/' + 'tmp/'))
    for target in re.findall(r'\]\(([^)]+)\)', text):
        target = target.split('#')[0]
        if target and '://' not in target:
            assert not target.startswith('/')
            assert (ROOT / 'results' / target).exists(), target
    for row in data['defects']:
        assert set(row) >= {'id', 'fingerprint', 'category', 'component', 'status', 'detail',
                            'root_cause', 'resolution', 'first_seen', 'last_seen', 'resolved_at',
                            'regression_count', 'waiver_reason', 'events'}
    probes = {row['probe'] for row in data['probes']}
    format_section = text.split('## Submission-format exploits')[1].split(
        '## Grader and verifier evasion')[0]
    for line in format_section.splitlines():
        if line.startswith('| `'):
            for probe in re.findall(r'`([^`]+)`', line.split('|')[1]):
                assert probe in probes, probe


def test_public_counts_match_evidence():
    text = (ROOT / 'results/reward-hacks.md').read_text()
    data = json.loads((ROOT / 'results/evidence/reward-hacks.json').read_text())
    summary = data['summary']
    assert sum(summary['defects_by_status'].values()) == summary['defects']
    assert f"**{summary['defects']} defects:" in text
    assert f"{summary['probe_occurrences']:,} probe occurrences" in text
    assert f"{summary['unique_probe_names']} distinct probe names" in text
    assert f"across {summary['probe_attempts']} attempts" in text
    for category in summary['defects_by_category']:
        counts = [sum(row['category'] == category and row['status'] == status
                      for row in data['defects'])
                  for status in ('RESOLVED', 'DEFERRED', 'WAIVED')]
        label = 'reported resource-accounting fault' if category == 'operator-reported' else category
        assert f"| {label} | {' | '.join(map(str, counts))} |" in text
