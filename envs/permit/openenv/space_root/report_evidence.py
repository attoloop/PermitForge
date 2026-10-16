"""Print DESIGN's measured tables without native work: python3 envs/permit/openenv/space_root/report_evidence.py."""

from collections import Counter
from decimal import Decimal
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
REPORTS = ROOT / 'results/evidence'


def main() -> None:
    inventory = json.loads((REPORTS / 'conversion.json').read_text())['counts']
    print('Source inventory:', json.dumps(inventory, sort_keys=True))
    for split in ('train', 'eval'):
        rows = [json.loads(line) for line in (ROOT / f'envs/permit/tasks/permit-v2-{split}/tasks.jsonl').read_text().splitlines()]
        print(split, 'stations:', dict(sorted(Counter(r['station'] for r in rows).items())))
    fidelity = json.loads((REPORTS / 'fidelity.json').read_text())
    for name, summary in fidelity['summary']['standards'].items():
        comparisons = [row for case in fidelity['comparisons'] for row in case['standards']
                       if row['id'] == name]
        assert len(comparisons) == summary['comparisons']
        assert sum(Decimal(row['fast']) == Decimal(row['native']) for row in comparisons) == summary['exact']
    print('\n| Standard | Comparisons | Exact at five decimals | Max absolute error | Max relative error |')
    print('|---|---:|---:|---:|---:|')
    for name, row in fidelity['summary']['standards'].items():
        print(f"| {name} | {row['comparisons']} | {row['exact']} | {row['max_absolute_error']:g} | {row['max_relative_error']:g} |")
    print('Values:', sum(r['comparisons'] for r in fidelity['summary']['standards'].values()),
          'exact:', sum(r['exact'] for r in fidelity['summary']['standards'].values()),
          'feasibility:', fidelity['summary']['feasibility_agreements'], '/', len(fidelity['comparisons']))
    print('\n| Task / sources | Median ms | Maximum ms | Float64 compressed bytes | Float32 changed values |')
    print('|---|---:|---:|---:|---:|')
    for task in fidelity['tasks']:
        print(f"| `{task['id']}` / {task['sources']} | {task['evaluation_seconds']['median']*1000:.3f} | "
              f"{task['evaluation_seconds']['max']*1000:.3f} | {task['pack_bytes']['float64']} | "
              f"{task['float32_changed_standard_values']} |")
    print('Selected:', json.dumps(fidelity['selected']))
    print('Float32 changed total:', sum(t['float32_changed_standard_values'] for t in fidelity['tasks']))
    for split in ('train', 'eval'):
        rows = [json.loads(line) for line in (ROOT / f'envs/permit/tasks/permit-v2-{split}/tasks.jsonl').read_text().splitlines()]
        print('Admission:', split, dict(Counter(row.get('eligible') for row in rows)))
    for path in sorted((REPORTS / 'faithful-lift').glob('*-s47.json')):
        case = json.loads(path.read_text())
        print('Faithful lift:', case['instance'],
              sum(row['receptors_compared'] for row in case['standards']),
              'all match:', all(row['match'] for row in case['standards']))


if __name__ == '__main__':
    main()
