"""Shared records and file operations; used by `python -m eval.run_eval`."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .privacy import redact, redact_text

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ('get_scenario', 'check_design', 'evaluate_design', 'submit_design')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_slug(model: str) -> str:
    readable = re.sub(r'[^a-zA-Z0-9_.-]', '_', model).strip('.')[:100]
    return readable + '-' + hashlib.sha256(model.encode()).hexdigest()[:12]


def component(value: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'unsafe path component: {value!r}')
    return value


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(redact_text(text))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(redact(value), indent=2, allow_nan=False, ensure_ascii=False)+'\n')


def records(runs: list[Path], *, include_incomplete: bool = False) -> list[tuple[Path, dict]]:
    found = []
    seen = set()
    for run in runs:
        canonical = sorted(run.glob('*/*/episode.json')) + sorted(run.glob('*/*/r*/episode.json'))
        legacy = [p for p in sorted(run.glob('*/*.json')) if not (p.parent/p.stem/'episode.json').exists()]
        for path in canonical + legacy:
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            record = json.loads(path.read_text())
            if not include_incomplete and record.get('attempt_complete') is False:
                continue
            if record.get('schema_version') == 1 and 'task_id' in record and 'model' in record:
                record.setdefault('repeat', 1)
                found.append((path, record))
    return found


def report_records(runs: list[Path]) -> list[tuple[Path, dict]]:
    """Pooling completed runs requires the same frozen episode protocol."""
    if len({p.resolve() for p in runs}) > 1:
        manifests = [json.loads((p/'run.json').read_text()) for p in runs]
        for key in ('protocol_sha256', 'prompt_sha256', 'prompt_version', 'sampling'):
            if any(m.get(key) is None or m.get(key) != manifests[0].get(key) for m in manifests):
                raise ValueError(f'report protocol mismatch: {key}')
    return records(runs)


def run_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_dir() else ROOT/'results'/'rollouts'/component(value)


def finite(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def failure_category(record: dict) -> str:
    final = record.get('final')
    if final is not None:
        if not final.get('valid', True):
            return 'invalid'
        if not final.get('feasible') and final.get('errors') and not any(
                code in str(final['errors']) for code in ('E_NAAQS', 'E_GEP', 'E_MENU')):
            return 'error'
        return 'submitted' if final.get('feasible') else 'infeasible'
    if record.get('ended') == 'error':
        return 'timeout' if record.get('error_type') == 'timeout' else 'error'
    return record.get('ended', 'no_submit')


def saved_fraction(record: dict) -> float:
    """Recompute current grading from raw savings, including historical episodes."""
    from permit_core.reward import reward
    final = record.get('final') or {}
    if record.get('ended') != 'submitted' or not final.get('valid', True) or not final.get('feasible'):
        return 0.0
    if not finite(final.get('prescore_usd_saved')):
        raise ValueError('nonfinite prescore in episode')
    return reward(final['prescore_usd_saved'], record.get('scenario', {}).get('draft_cost'))
