"""Run isolated suites: uv run --project envs/permit/openenv python envs/permit/openenv/space_root/check_local.py.

Uses temporary field storage and writes logs separately from public measurements.
Requires preserved v1 inputs, converted tasks and native binaries. No model APIs.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[4]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-only', action='store_true')
    parser.add_argument('--field-jobs', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--output', type=Path, default=ROOT / 'internal/reports/local-validation')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    suites = {'root': (ROOT, 'tests'),
              'core': (ROOT / 'envs/permit/core', 'tests'),
              'openenv': (ROOT / 'envs/permit/openenv', 'tests')}
    if args.core_only:
        suites = {'core': suites['core']}
    with tempfile.TemporaryDirectory(prefix='permitforge-validation-') as scratch:
        work = Path(scratch)
        (work / 'sitecustomize.py').write_text(
            'from pathlib import Path\n'
            'import permit_core.fields as fields\n'
            f'fields.cache_root = lambda: Path({str(work / "cache/fields")!r})\n'
        )
        env = dict(os.environ, PERMIT_V1_DIR=str(ROOT / 'convert/v1-src'),
                   PERMIT_TASKS_DIR=str(ROOT / 'envs/permit/tasks'),
                   PERMIT_FIELDS_JOBS=str(args.field_jobs), PERMIT_FIELDS_CACHE_GB='8',
                   PYTHONPATH=str(work), PYTHONDONTWRITEBYTECODE='1')
        for key in ('PERMIT_FIELDS_DIR', 'PERMIT_EVALUATOR'):
            env.pop(key, None)
        summary = {}
        for name, (cwd, tests) in suites.items():
            xml = args.output / f'{name}.xml'
            command = [sys.executable, '-m', 'pytest', tests, '-q', f'--junitxml={xml}']
            print(name, ' '.join(command), flush=True)
            with (args.output / f'{name}.log').open('w') as log:
                result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            cases = ET.parse(xml).findall('.//testcase') if xml.exists() else []
            summary[name] = {'command': command, 'exit_code': result.returncode,
                             'tests': len(cases), **{tag: sum(c.find(tag) is not None for c in cases)
                             for tag in ('failure', 'error', 'skipped')}}
            print(name, 'exit code:', result.returncode, flush=True)
        (args.output / 'suites.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
        raise SystemExit(int(any(row['exit_code'] for row in summary.values())))


if __name__ == '__main__':
    main()
