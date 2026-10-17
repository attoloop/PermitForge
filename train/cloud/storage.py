"""Explicit artifact destinations; no account/repository creation or implicit uploads."""
from pathlib import Path
import re
import subprocess


def sync_command(local: Path, uri: str, *, restore: bool = False) -> list[str]:
    match = re.fullmatch(r'hf://(buckets|datasets|models)/([\w.-]+/[\w.-]+)/([\w./-]+)', uri)
    if not match or '..' in match.group(3).split('/'):
        raise ValueError('Use hf://{buckets,datasets,models}/owner/name/run-prefix')
    kind, repo, prefix = match.groups()
    if kind == 'buckets':
        return ['hf', 'sync', uri if restore else str(local), str(local) if restore else uri]
    repo_type = 'dataset' if kind == 'datasets' else 'model'
    if restore:
        # snapshot_download preserves the prefix; restore() moves its contents.
        return ['hf', 'download', repo, '--repo-type', repo_type, '--include', prefix+'/**',
                '--local-dir', str(local)]
    return ['hf', 'upload', repo, str(local), prefix, '--repo-type', repo_type]


def sync(local: Path, uri: str, seconds: float) -> None:
    import time
    started = time.monotonic()
    command = sync_command(local, uri)
    excluded = ['*.lock', '.cache/**', '*.tmp', 'state.json', '**/.complete', '**/metrics.json']
    excluded.extend(str(p.relative_to(local))+'/**' for p in local.rglob('checkpoint-*')
                    if p.is_dir() and not (p/'.complete').exists())
    bulk = command + [arg for pattern in excluded for arg in ('--exclude',pattern)]
    subprocess.run(bulk, check=True, timeout=seconds,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    # Completion markers/state become visible only after all payload files arrived.
    markers = command + [arg for pattern in ('state.json','**/.complete','**/metrics.json')
                         for arg in ('--include',pattern)]
    markers += [arg for pattern in excluded[6:] for arg in ('--exclude',pattern)]
    subprocess.run(markers, check=True, timeout=max(.01,seconds-(time.monotonic()-started)),
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def restore(local: Path, uri: str, seconds: float) -> None:
    import shutil
    if local.exists() and any(local.iterdir()):
        raise ValueError('Restore requires an empty output directory')
    local.mkdir(parents=True, exist_ok=True)
    subprocess.run(sync_command(local, uri, restore=True), check=True, timeout=seconds)
    if not uri.startswith('hf://buckets/'):
        prefix = uri.split('/', 5)[5]
        nested = local/prefix
        if not nested.is_dir():
            raise ValueError('Remote run prefix is missing')
        temporary = local.parent/(local.name+'.restore')
        if temporary.exists():
            raise ValueError('Restore staging directory already exists')
        nested.rename(temporary)
        shutil.rmtree(local)
        temporary.rename(local)
