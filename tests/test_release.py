"""Release layout checks: uv run pytest tests/test_release.py -q."""

from pathlib import Path
import re
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / 'envs/permit/openenv'


def release_files() -> list[Path]:
    """Inspect existing tracked files and additions, respecting pending removals."""
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT,
    ).decode().split('\0')
    candidates = sorted({name for name in names if name and (ROOT / name).is_file()})
    ignored = subprocess.run(
        ['git', 'check-ignore', '--no-index', '-z', '--stdin'], cwd=ROOT,
        input='\0'.join(candidates) + '\0', text=True, capture_output=True, check=False,
    )
    assert ignored.returncode in (0, 1), ignored.stderr
    excluded = set(ignored.stdout.split('\0'))
    return [ROOT / name for name in candidates if name not in excluded]


def test_release_layout():
    assert not (SPACE / 'web').exists()
    assert not (SPACE / 'permit_openenv/play_api.py').exists()
    assert not (SPACE / 'permit_openenv/api.py').exists()
    for name in ('PLAN.md', 'tasks', 'reports'):
        assert not (ROOT / 'docs' / name).exists()
    assert 'internal/' in (ROOT / '.gitignore').read_text().splitlines()
    assert 'AGENTS.md' in (ROOT / '.gitignore').read_text().splitlines()
    assert 'Copyright 2026 Attoloop' in (ROOT / 'LICENSE').read_text()
    assert 'Apache License' in (ROOT / 'LICENSE').read_text()
    assert 'title: PermitForge' in (ROOT / 'CITATION.cff').read_text()


def test_release_file_hygiene():
    violations = []
    for path in release_files():
        name = path.relative_to(ROOT).as_posix()
        if name.startswith('internal/') or path.suffix == '.log' or path.stat().st_size > 2_000_000:
            violations.append(name)
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        # Construct markers so the scanner does not flag its own source.
        if any(marker in text for marker in ('/' + 'home/', '/' + 'tmp/')):
            violations.append(name + ': absolute local path')
    assert not violations, '\n'.join(violations)


@pytest.mark.parametrize('name', [
    'README.md', 'DESIGN.md', 'CONTRACT.md', 'SOURCES.md', 'docs/SCHEMAS.md',
    'convert/README.md', 'envs/permit/core/README.md', 'envs/permit/openenv/README.md',
    'envs/permit/openenv/DESIGN.md', 'envs/permit/openenv/space_root/README.md',
])
def test_public_docs_are_self_contained(name):
    text = (ROOT / name).read_text()
    assert not re.search(r'\bWP\d|\bCodex\b|\breviewers?\b|\boperators?\b|\bPhase[ -]2\b|docs/(PLAN|tasks|reports)|internal/', text, re.I)
    assert not re.search(r'viewer|play_api|/play/|web/play|web/vendor', text, re.I)
    for target in re.findall(r'\]\(([^)]+)\)', text):
        target = target.split('#')[0]
        if not target or '://' in target or target.startswith('/'):
            continue
        assert (ROOT / name).parent.joinpath(target).exists(), (name, target)


def test_space_copy_sources_exist_without_custom_ui():
    dockerfile = (SPACE / 'Dockerfile').read_text()
    assert 'openenv/web' not in dockerfile
    assert 'openenv/web' not in (ROOT / '.dockerignore').read_text()
    for line in dockerfile.splitlines():
        if not line.startswith('COPY ') or '--from=' in line:
            continue
        for source in shlex.split(line)[1:-1]:
            assert (ROOT / source).exists(), source
    server = (SPACE / 'permit_openenv/server.py').read_text()
    assert 'mount_viewer' not in server and 'mount_play' not in server
    assert 'gradio_builder=gradio_builder' in server
