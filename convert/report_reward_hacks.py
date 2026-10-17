"""Extract the historical ledger: uv run python convert/report_reward_hacks.py [--check].

Uses only the standard library and an archived, checkpointed SQLite database.
--check compares bytes without writing. --summary recounts the saved public JSON
without requiring the private archive. --database overrides the archive location.
"""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/evidence/reward-hacks.json'


def default_database() -> Path:
    return Path.home() / 'envs/envfactory/state/envfactory.db'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def public_text(value: Any) -> Any:
    """Keep evidence text, replacing machine-specific path prefixes recursively."""
    if isinstance(value, dict):
        return {key: public_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_text(item) for item in value]
    if not isinstance(value, str):
        return value
    value = value.replace(str(ROOT) + '/', '')
    # Container paths remain recognizable but relative; host user names are removed.
    value = re.sub(r'/(?:home|Users)/[^/\s\"\'<>]+/', '<host>/', value)
    return re.sub(r'(?<![\w.:/])/(?!/)(?=[A-Za-z])', '<path>:', value)


def encode(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def summarize(data: dict[str, Any]) -> dict[str, Any]:
    defects, probes = data['defects'], data['probes']
    return {
        'defects': len(defects),
        'defects_by_category': dict(sorted(Counter(d['category'] for d in defects).items())),
        'defects_by_status': dict(sorted(Counter(d['status'] for d in defects).items())),
        'probe_occurrences': len(probes),
        'unique_probe_names': len({p['probe'] for p in probes}),
        'probe_attempts': len({(p['node'], p['attempt']) for p in probes}),
        'probes_by_node': dict(sorted(Counter(p['node'] for p in probes).items())),
        'attempts_by_node': dict(sorted(Counter(a['node'] for a in data['attempts']).items())),
    }


def extract(database: Path, probes_path: Path, manifest_path: Path) -> dict[str, Any]:
    database = database.resolve()
    for suffix in ('-wal', '-journal'):
        journal = Path(str(database) + suffix)
        if journal.exists() and journal.stat().st_size:
            raise ValueError('archive must be checkpointed: nonempty WAL/journal exists')
    before = digest(database)
    # Immutable mode avoids SQLite trying to create shared-memory files beside the
    # read-only archive. A journal check and before/after hash guard the snapshot.
    uri = database.as_uri() + '?mode=ro&immutable=1'
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        attempts = [dict(row) for row in conn.execute('''
            SELECT a.id AS attempt, n.node_key AS node
            FROM attempts_v2 a JOIN workflow_nodes n ON n.id=a.node_id
            WHERE a.environment_id=47 AND n.environment_id=47 ORDER BY a.id
        ''')]
        nodes = {row['attempt']: row['node'] for row in attempts}

        def seen(attempt: int | None) -> dict[str, Any]:
            return {'attempt': attempt, 'node': nodes.get(attempt)}

        defects = []
        for row in conn.execute('SELECT * FROM defects_v2 WHERE environment_id=47 ORDER BY defect_id'):
            item = dict(row)
            events = [dict(event) for event in conn.execute('''
                SELECT id, attempt_id, old_status, new_status, evidence_digest, created_at
                FROM defect_events_v2 WHERE defect_id=? ORDER BY id
            ''', (item['defect_id'],))]
            for event in events:
                event['node'] = nodes.get(event['attempt_id'])
            defects.append({
                'id': item['defect_id'],
                **{key: item[key] for key in (
                    'fingerprint', 'category', 'component', 'status', 'detail', 'root_cause',
                    'resolved_at', 'regression_count', 'waiver_reason', 'deferred_scope',
                    'suggested_remedy_text')},
                'first_seen': seen(item['first_seen_attempt']),
                'last_seen': seen(item['last_seen_attempt']),
                'resolution': {
                    **seen(item['resolved_by_attempt']),
                    'summary': item['root_cause'],
                    'paths': json.loads(item['resolution_paths_json']),
                    'closure_checks': json.loads(item['closure_checks_json']),
                    'evidence_digest': item['closure_evidence_digest'],
                },
                'detail_history': json.loads(item['detail_history_json']),
                'events': events,
            })
    if digest(database) != before:
        raise ValueError('archive changed during extraction; retry from a stable snapshot')
    probes = json.loads(probes_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    probe_hash = digest(probes_path)
    for generated in manifest['generated']:
        if generated['destination'] == 'evidence/adversarial_probes.json':
            if generated['sha256'] != probe_hash:
                raise ValueError('probe index differs from extraction manifest')
    data = public_text({
        'schema_version': 1,
        'environment_id': 47,
        'sources': {
            'database': {'name': 'envfactory/state/envfactory.db', 'sha256': before},
            'probe_index': {'name': 'convert/v1-src/evidence/adversarial_probes.json',
                            'sha256': probe_hash},
            'extraction_manifest': {'name': 'convert/v1-src/EXTRACT_MANIFEST.json',
                                    'sha256': digest(manifest_path)},
            'v1_git': manifest['source_git'],
        },
        'notes': [
            'RESOLVED is an archival status, not an independent certification of a fix.',
            'The database has no resolution-prose column. resolution.summary preserves root_cause; '
            'null means no closure narrative was recorded. Suggested remedies are not proof of fixes.',
            'Missing attempt IDs remain null. Events and detail history are retained separately.',
            'Probe occurrences are indexed directories, including repeated names across attempts; '
            'the index does not record outcomes or prove that each probe finished.',
            'Absolute filesystem paths are normalized to relative evidence labels; other text is retained.',
        ],
        'attempts': attempts,
        'defects': defects,
        'probes': sorted(probes, key=lambda p: (p['node'], int(p['attempt']), p['probe'])),
    })
    data['summary'] = summarize(data)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=default_database())
    parser.add_argument('--probes', type=Path,
                        default=ROOT / 'convert/v1-src/evidence/adversarial_probes.json')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'convert/v1-src/EXTRACT_MANIFEST.json')
    parser.add_argument('--out', type=Path, default=OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    try:
        if args.summary:
            data = json.loads(args.out.read_text())
        else:
            data = extract(args.database, args.probes, args.manifest)
            content = encode(data)
            if args.check:
                if args.out.read_bytes() != content:
                    raise ValueError('reward-hacks JSON is stale')
            else:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_bytes(content)
        print(json.dumps(summarize(data), sort_keys=True, indent=2))
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.exit(1, f'ledger extraction failed: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
