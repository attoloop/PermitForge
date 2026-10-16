"""Recompute search measurement tables: python3 convert/report_search.py; from the saved measurement receipts; no models run."""

import json
from pathlib import Path


REPORTS = Path(__file__).resolve().parents[1]/'results/evidence'


def read(name: str) -> dict:
    return json.loads((REPORTS/name).read_text())


def search_table() -> None:
    print('| Sources / task | Run | Nodes | Evaluations | Seconds incl. verify | Closed | Best cost | Greedy | v1-CD |')
    print('|---|---|---:|---:|---:|---|---:|---:|---:|')
    for task in ('074897d0f159-s47', '05efc240e839-s47', '01addc2a86f1-s47'):
        for label in ('before', 'after'):
            data = read(f'search/{label}-{task}.json')
            if 'search' not in data:
                print(f'{task} {label}: measurement still running')
                continue
            result, policies = data['search'], data['baselines']
            print(f"| {data['sources']} / `{task}` | {label} | {result['nodes']} | "
                  f"{result['evaluations_used']} | {result['seconds']:.3f} | {str(result['closed']).lower()} | "
                  f"{result['cost']:.4f} | {policies['cheapest_first_greedy']['cost']:.4f} | "
                  f"{policies['v1_coordinate_descent']['cost']:.4f} |")


def budget() -> None:
    receipt = read('pipeline-budget.json')
    probe = receipt['cold_field_probe']
    per_field = probe['cpu_seconds']/probe['fields']
    cold, warm, no_search = {}, {}, {}
    for row in receipt['measurements']:
        sources = row['sources']
        cpu = row['run_timing']['cpu_seconds']
        cold[sources] = cpu+(row['field_count']-row['fields_built_in_run'])*per_field
        warm[sources] = cpu-row['fields_built_in_run']*per_field
        # Removing elapsed search/verification approximates zero-search CPU.
        no_search[sources] = cold[sources]-row['search_seconds']-row['final_verification_seconds']
    print('\n| Pipeline projection | CPU hours | Ideal elapsed hours / 12 cores |')
    print('|---|---:|---:|')
    for name, values in [('Cold fields + all policies + search', cold),
                         ('Warm fields + all policies + search', warm),
                         ('Cold fields + policies, removing search and its final check', no_search)]:
        values[3], values[5] = (values[2]+values[4])/2, (values[4]+values[6])/2
        hours = sum(values[int(n)]*count for n, count in receipt['sources_counts'].items())/3600
        print(f'| {name} | {hours:.3f} | {hours/12:.3f} |')
    print('Assumptions: '+' '.join(receipt['assumptions']))


if __name__ == '__main__':
    search_table()
    budget()
