"""Build local HF dataset configs, without uploading: python eval/export_hf.py RUN --out DIR."""
from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'eval'

import argparse
import json
from pathlib import Path

from .capture import append_json
from .common import records, run_path, saved_fraction, write_text


def export_runs(runs: list[Path], output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError('export destination must be empty')
    for config in ('episodes', 'transcripts', 'raw'):
        write_text(output/config/'train.jsonl', '')
    for run in runs:
        for path, record in records([run]):
            root = path.parent if path.name == 'episode.json' else path.with_suffix('')
            identity = {'run': run.name, 'model': record['model'], 'task_id': record['task_id'],
                        'repeat': record.get('repeat', 1), 'task_split': record.get('split')}
            append_json(output/'episodes/train.jsonl', {**identity, 'ended': record.get('ended'),
                        'reward': saved_fraction(record),
                        'record_json': json.dumps(record, ensure_ascii=False)})
            transcript = root/'transcript.jsonl'
            if transcript.exists():
                for line in transcript.read_text().splitlines():
                    row = json.loads(line)
                    append_json(output/'transcripts/train.jsonl', {**identity,
                        **{k: row.get(k) for k in ('i', 'at', 'turn', 'role', 'kind', 'name')},
                        'content_json': json.dumps(row.get('content'), ensure_ascii=False),
                        'usage_json': json.dumps(row.get('usage'))})
            for folder in ('raw', 'attempts'):
                for artifact in sorted((root/folder).rglob('*')):
                    if artifact.is_file():
                        append_json(output/'raw/train.jsonl', {**identity,
                            'path': str(artifact.relative_to(root)), 'content': artifact.read_text()})
            if (root/'env.log').exists():
                append_json(output/'raw/train.jsonl', {**identity, 'path': 'env.log',
                            'content': (root/'env.log').read_text()})
        for name in ('run.json', 'harness.log', 'costs.jsonl', 'index.json'):
            if (run/name).exists():
                append_json(output/'raw/train.jsonl', {'run': run.name, 'model': None, 'task_id': None,
                            'task_split': None, 'path': name, 'content': (run/name).read_text()})
    write_text(output/'README.md', '''---
license: apache-2.0
configs:
- config_name: episodes
  data_files:
  - split: train
    path: episodes/train.jsonl
- config_name: transcripts
  data_files:
  - split: train
    path: transcripts/train.jsonl
- config_name: raw
  data_files:
  - split: train
    path: raw/train.jsonl
---
# PermitForge rollout export

Local export only; this command does not upload or publish.
Join configs by run, model, task_id and repeat. `task_split` preserves the environment
train/eval split; the dataset's train partition is a collection of recorded rows.
Episodes retain the complete private Episode in record_json; transcripts retain
full returned content (including provider-returned reasoning) and per-turn usage.
Raw rows contain requests, responses, CLI events/config/stderr, failed attempts,
environment logs and run provenance, with credential redaction applied.
No hidden reasoning is inferred or reconstructed when a provider omits it.
''')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    export_runs([run_path(run) for run in args.runs], args.out)


if __name__ == '__main__':
    main()
