"""Prepare a private cloud input archive locally: python -m train.cloud.bundle --help.

Copies code, selected task inputs and native binaries. Never uploads or builds SFT data.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile

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


def create_bundle(root: Path, cohort: Path, output: Path, *, inventory_only: bool = False) -> dict:
    from permit_openenv.tasks import load_pack
    from .selection import selection_manifest
    pack = load_pack(root/'envs/permit/tasks')
    selection = selection_manifest(pack,cohort)
    ids = set(selection['train']+selection['eval'])
    stations = {pack.get(t).record['station'] for t in ids}
    files = {}
    for path in bundle_files(root):
        name = str(path.relative_to(root))
        if '/tasks/data/' in name and name.split('/')[4] not in ids:
            continue
        if '/tasks/met/' in name and name.split('/')[4] not in stations:
            continue
        files[name] = path
    files['results/subsets.json'] = cohort
    for name in ('aermod','bpipprm'):
        if not (root/'envs/permit/core/native_build/bin'/name).is_file():
            raise ValueError('Build the pinned native binaries before bundling')
    metadata = {'selection':selection,
                'files': {name: {'bytes':path.stat().st_size} for name,path in sorted(files.items())}}
    metadata['bytes'] = sum(row['bytes'] for row in metadata['files'].values())
    output.parent.mkdir(parents=True,exist_ok=True)
    if not inventory_only:
        with tarfile.open(output,'w:gz',dereference=True) as archive:
            for name,path in sorted(files.items()):
                with path.open('rb') as handle:
                    metadata['files'][name]['sha256'] = hashlib.file_digest(handle,'sha256').hexdigest()
                archive.add(path,arcname='permitforge/'+name,recursive=False)
            payload = json.dumps(metadata,indent=2).encode()
            member = tarfile.TarInfo('permitforge/train/cloud/input-manifest.json')
            member.size = len(payload)
            archive.addfile(member,io.BytesIO(payload))
        with output.open('rb') as handle:
            metadata['bundle_sha256'] = hashlib.file_digest(handle,'sha256').hexdigest()
    output.with_suffix('.inventory.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def verify_tree(root: Path, *, strict: bool = True) -> dict:
    manifest = json.loads((root/'train/cloud/input-manifest.json').read_text())
    if strict:
        actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
        expected = set(manifest['files']) | {'train/cloud/input-manifest.json'}
        if actual != expected:
            raise ValueError('Bundle member list differs from manifest')
    for name,row in manifest['files'].items():
        with (root/name).open('rb') as handle:
            digest = hashlib.file_digest(handle,'sha256').hexdigest()
        if digest != row['sha256'] or (root/name).stat().st_size != row['bytes']:
            raise ValueError('Bundle member SHA256 mismatch: '+name)
    from permit_openenv.tasks import load_pack
    pack = load_pack(root/'envs/permit/tasks')
    for tid in manifest['selection']['train']+manifest['selection']['eval']:
        task = pack.get(tid)
        task.scenario  # Validate native task and met availability without running models.
        for name,digest in task.record['sha256'].items():
            path = root/'envs/permit/tasks/data'/tid/name
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Converted task SHA256 mismatch: '+str(path))
        for name,digest in task.record['met_sha256'].items():
            path = root/'envs/permit/tasks/met'/task.record['station']/name
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Meteorology SHA256 mismatch: '+str(path))
    return dict(verified_files=len(manifest['files']),selection=manifest['selection'])


def verify_bundle(bundle: Path, sha256: str) -> dict:
    from .hf_job import unpack
    with tempfile.TemporaryDirectory(prefix='permit-bundle-') as folder:
        root = unpack(bundle,Path(folder),sha256)
        return verify_tree(root)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,default=Path('results/subsets.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inventory-only',action='store_true')
    p.add_argument('--verify',action='store_true')
    p.add_argument('--sha256')
    args = p.parse_args()
    if args.verify:
        print(json.dumps(verify_bundle(args.output,args.sha256),indent=2))
        return
    metadata = create_bundle(ROOT,args.cohort,args.output,inventory_only=args.inventory_only)
    print(json.dumps({k:v for k,v in metadata.items() if k!='files'},indent=2))


if __name__ == '__main__':
    main()
