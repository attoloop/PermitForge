# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["uv==0.11.32"]
# ///
"""HF Jobs launcher; mount a reviewed bundle and pass its SHA256. Never submits a Job."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tarfile


def unpack(bundle: Path, destination: Path, sha256: str) -> Path:
    with bundle.open('rb') as handle:
        actual = hashlib.file_digest(handle,'sha256').hexdigest()
    if actual != sha256:
        raise ValueError('Bundle SHA256 does not match')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Use an empty workspace for the pinned bundle')
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(bundle) as archive:
        archive.extractall(destination,filter='data')
    root = destination/'permitforge'
    if not (root/'train/cloud/uv.lock').exists():
        raise ValueError('Bundle is missing the locked cloud project')
    return root


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--bundle-sha256',required=True)
    p.add_argument('--workspace',type=Path,default=Path('/workspace/permit-cloud'))
    args, remaining = p.parse_known_args()
    root = unpack(args.bundle,args.workspace,args.bundle_sha256)
    if remaining[:1] == ['--']:
        remaining = remaining[1:]
    os.chdir(root)
    subprocess.run(['uv','sync','--project','train/cloud','--frozen','--no-dev'],check=True)
    os.execvp('uv',['uv','run','--project','train/cloud','--frozen','--no-sync',
                   'python','-m','train.cloud.run',*remaining])


if __name__ == '__main__':
    main()
