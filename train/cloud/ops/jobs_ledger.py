"""Dump every HF Job of a namespace with timings and cost: python train/cloud/ops/jobs_ledger.py Attoloop OUT.json

Needs HF_TOKEN. Cost is seconds times the flavor's hourly rate (h200 $5, cpu-basic $0); it is what the
provider bills per second, not an invoice. eval/learnability.py reads the result for its cost table.
"""
import json
import sys

from huggingface_hub import HfApi

RATES = {'h200': 5.0}


def main(namespace: str, output: str) -> None:
    rows = []
    for job in HfApi().list_jobs(namespace=namespace):
        started, finished = job.started_at, job.finished_at
        seconds = (finished - started).total_seconds() if started and finished else None
        rate = RATES.get(job.flavor, 0.0)
        command = ' '.join(job.command or [])
        rows.append({'id': job.id, 'stage': job.status.stage, 'flavor': job.flavor, 'created': str(job.created_at),
                     'started': str(started), 'finished': str(finished), 'seconds': seconds, 'usd_per_hour': rate,
                     'usd': None if seconds is None else round(seconds/3600*rate, 3),
                     'kind': 'probe' if command.startswith('bash -c') or 'print(' in command else 'job',
                     'docker_image': str(job.docker_image), 'command': command, 'args': ' '.join(job.arguments or [])})
    rows.sort(key=lambda r: r['created'])
    with open(output, 'w') as stream:
        json.dump(rows, stream, indent=1)
    print(len(rows), 'jobs,', round(sum(r['usd'] or 0 for r in rows), 2), 'USD')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
