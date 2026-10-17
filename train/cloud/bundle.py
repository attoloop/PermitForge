"""Prepare a private cloud input archive locally: python -m train.cloud.bundle --help.

Copies code, converted tasks, native binaries and train-only SFT data. Never uploads.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from .run import ROOT


def bundle_files(root: Path) -> list[Path]:
    files = set()
    for pattern in ('train/*.py','train/cloud/*.py','train/cloud/*.toml','train/cloud/*.lock',
        'train/cloud/*.sh','train/cloud/*.md','envs/permit/core/pyproject.toml',
        'envs/permit/openenv/pyproject.toml','envs/permit/core/permit_core/*.py',
        'envs/permit/openenv/permit_openenv/*.py','eval/PROTOCOL.md',
        'envs/permit/tasks/permit-v2-*/*.json*','envs/permit/tasks/data/**/*',
        'envs/permit/tasks/met/**/*','envs/permit/core/native_build/bin/*'):
        files.update(p for p in root.glob(pattern) if p.is_file())
    return sorted(files)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sft-data',type=Path,required=True)
    p.add_argument('--cohort',type=Path,default=Path('results/subsets.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inventory-only',action='store_true')
    args = p.parse_args()
    files = {str(f.relative_to(ROOT)):f for f in bundle_files(ROOT)}
    files['train/cloud/data/sft.jsonl'] = args.sft_data
    files['results/subsets.json'] = args.cohort
    for name in ('aermod','bpipprm'):
        if not (ROOT/'envs/permit/core/native_build/bin'/name).is_file():
            raise ValueError('Build the pinned native binaries before bundling')
    metadata = {'files': {name: {'bytes':path.stat().st_size} for name,path in sorted(files.items())}}
    metadata['bytes'] = sum(row['bytes'] for row in metadata['files'].values())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not args.inventory_only:
        with tarfile.open(args.output,'w:gz',dereference=True) as archive:
            for name,path in sorted(files.items()):
                with path.open('rb') as handle:
                    metadata['files'][name]['sha256'] = hashlib.file_digest(handle,'sha256').hexdigest()
                archive.add(path,arcname='permitforge/'+name,recursive=False)
            payload = json.dumps(metadata,indent=2).encode()
            member = tarfile.TarInfo('permitforge/train/cloud/input-manifest.json')
            member.size = len(payload)
            archive.addfile(member,io.BytesIO(payload))
        with args.output.open('rb') as handle:
            metadata['bundle_sha256'] = hashlib.file_digest(handle,'sha256').hexdigest()
    args.output.with_suffix('.inventory.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k!='files'},indent=2))


if __name__ == '__main__':
    main()
