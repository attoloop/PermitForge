"""Assemble the files of the Hugging Face Space: python envs/permit/openenv/space_root/export_space.py OUT_DIR

The Space builds from a Dockerfile at its root with the Space repository as context. This copies exactly
what the Dockerfile's COPY lines name (code, native model source, task and met data), the Dockerfile itself,
the .dockerignore and the Space card as README.md. OUT_DIR must not exist. Upload it with
`hf upload-large-folder Attoloop/PermitForge OUT_DIR --repo-type space`.
"""
import shlex
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DOCKERFILE = ROOT / 'envs/permit/openenv/Dockerfile'


def sources() -> list[Path]:
    found = []
    for line in DOCKERFILE.read_text().splitlines():
        if line.startswith('COPY ') and '--from=' not in line:
            found.extend(Path(part) for part in shlex.split(line)[1:-1])
    return found


def main(out: Path) -> None:
    if out.exists():
        raise SystemExit(f'{out} exists; choose a new folder')
    for source in sources():
        if not (ROOT / source).exists():
            raise SystemExit(f'missing: {source}')
        target = out / source
        if (ROOT / source).is_dir():
            shutil.copytree(ROOT / source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'fields', '*.npy', '*.npz', '*.zst'))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / source, target)
    shutil.copy2(DOCKERFILE, out / 'Dockerfile')
    shutil.copy2(ROOT / '.dockerignore', out / '.dockerignore')
    shutil.copy2(ROOT / 'envs/permit/openenv/space_root/README.md', out / 'README.md')
    files = [p for p in out.rglob('*') if p.is_file()]
    print(f'{len(files)} files, {sum(p.stat().st_size for p in files)/1e6:.0f} MB, in {out}')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
