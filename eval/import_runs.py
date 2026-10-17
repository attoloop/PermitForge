"""Read-only import of completed repeats; used by python -m eval.run_eval --import-run RUN."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import shutil

from .common import component, model_slug, records, saved_fraction, write_json


def occupied_repeat(target: Path) -> list[Path]:
    paths = [target, target.with_suffix('.json')]
    if target.name == 'r1':
        task = target.parent
        paths.append(task.with_suffix('.json'))
        if task.exists():
            paths.extend(p for p in task.iterdir() if not re.fullmatch(r'r[1-9][0-9]*(?:\.json)?', p.name))
    return [p for p in paths if p.exists() or p.is_symlink()]


def import_completed(destination: Path, sources: list[Path], manifest: dict, repeats: int) -> None:
    pending = {}
    allowed = {(m, t['task_id']): t for m in manifest['models'] for t in manifest['tasks']}
    for source in sources:
        if source.resolve() == destination.resolve():
            raise ValueError('cannot import a run into itself')
        previous = json.loads((source/'run.json').read_text())
        for key in ('protocol_sha256', 'prompt_sha256', 'prompt_version', 'sampling'):
            if previous.get(key) != manifest.get(key):
                raise ValueError(f'import protocol mismatch: {key}')
        for path, original in records([source]):
            identity = original['model'], original['task_id']
            if identity not in allowed:
                continue
            if any(original.get(k) != allowed[identity][k] for k in ('station', 'split')):
                raise ValueError('import task metadata mismatch')
            k = original.get('repeat', 1)
            if type(k) is not int or not 1 <= k <= repeats:
                raise ValueError('import repeat outside requested range')
            target = destination/model_slug(identity[0])/component(identity[1])/f'r{k}'
            receipt = {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            occupied = occupied_repeat(target)
            if occupied:
                if occupied == [target] and (target/'episode.json').is_file():
                    try:
                        existing = json.loads((target/'episode.json').read_text())
                    except (OSError, ValueError):
                        existing = None
                    if isinstance(existing, dict) and existing.get('imported_from') == receipt:
                        continue  # Identical completed import is read-only on resume.
                raise ValueError(f'occupied destination repeat {target}: {occupied}')
            if target in pending:
                raise ValueError('duplicate imported model/task/repeat')
            record = copy.deepcopy(original)
            record.update(repeat=k, run=destination.name, imported_from=receipt, import_provenance=previous)
            record.pop('anchors', None)
            record['reward'] = saved_fraction(record)
            if record.get('final') is not None:
                record['original_grade'] = copy.deepcopy(record['final'])
                score = saved_fraction(record)
                record['final'].pop('anchors', None)
                record['final'].update(reward=score, pct_saved=100*score)
                for step in record.get('steps', []):
                    if step['tool'] == 'submit_design':
                        step['result'] = copy.deepcopy(record['final'])
            pending[target] = path, record
    # Validate all manifests and identities before writing any import.
    for target, (path, record) in pending.items():
        target.mkdir(parents=True, exist_ok=False)
        origin = path.parent if path.name == 'episode.json' else path.with_suffix('')
        for name in ('raw', 'transcript.jsonl', 'env.log', 'attempts'):
            entry = origin/name
            if entry.is_dir():
                shutil.copytree(entry, target/name)
            elif entry.is_file():
                shutil.copyfile(entry, target/name)
        write_json(target/'episode.json', record)
