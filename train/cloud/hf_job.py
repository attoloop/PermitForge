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
import json
import time


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
    p.add_argument('--job-seconds',type=int,default=10800,help='Match provider timeout; includes setup')
    args, remaining = p.parse_known_args()
    if remaining[:1] == ['--']:
        remaining = remaining[1:]
    runtime_parser = argparse.ArgumentParser(add_help=False)
    runtime_parser.add_argument('--recipe',default='smoke')
    runtime_parser.add_argument('--trackio-space')
    runtime_parser.add_argument('--hourly-usd',type=float,default=5.)
    runtime, _ = runtime_parser.parse_known_args(remaining)
    if args.job_seconds <= 300:
        raise ValueError('Job must reserve time for setup and final upload')
    if runtime.recipe == 'smoke' and (args.job_seconds > 10800 or
                                      args.job_seconds*runtime.hourly_usd/3600 > 15):
        raise ValueError('Smoke provider and bootstrap timeout must not exceed three hours ($15 on H200)')
    started = time.time()
    root = unpack(args.bundle,args.workspace,args.bundle_sha256)
    os.chdir(root)
    os.environ['PERMIT_JOB_DEADLINE_UNIX'] = str(started+args.job_seconds-120)
    metrics = ['--extra','metrics'] if runtime.trackio_space else []
    command = ['uv','sync','--project','train/cloud','--frozen','--no-dev']
    subprocess.run(command+metrics,check=True,timeout=min(900,args.job_seconds-240))
    setup = dict(started_unix=started,stack_install='installed')
    setup_path = root/'train/cloud/bootstrap.json'
    subprocess.run(['train/cloud/.venv/bin/python','-c',
        'from pathlib import Path; from train.cloud.bundle import verify_tree; verify_tree(Path("."), strict=False)'],
        check=True,timeout=120)
    setup['elapsed_seconds'] = time.time()-started
    setup_path.write_text(json.dumps(setup,indent=2)+'\n')
    os.environ['PERMIT_BOOTSTRAP_REPORT'] = str(setup_path)
    os.execvp('uv',['uv','run','--project','train/cloud','--frozen','--no-sync',
                   'python','-m','train.cloud.run',*remaining])


if __name__ == '__main__':
    main()
