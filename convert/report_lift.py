"""Print faithful-lift evidence: python3 convert/report_lift.py > results/evidence/faithful-lift.md."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print('# Published verifier comparison\n')
    print('Binding values below summarize the preserved native replays. Each test also compared every receptor,')
    print('feasibility and exact submitted cost; receipts retain source submissions, hashes and timings.\n')
    print('Reproduce this table: `python3 convert/report_lift.py > results/evidence/faithful-lift.md`.')
    print('Re-run native comparisons: `uv run --directory envs/permit/core pytest tests/test_lift_faithful.py '
          '--evidence-dir ../../../results/evidence/faithful-lift -q`.\n')
    print('| Instance | Standard | Published value | Lifted value | Match | Receptors compared |')
    print('|---|---|---:|---:|---|---:|')
    for path in sorted((ROOT / 'results/evidence/faithful-lift').glob('*-s47.json')):
        case = json.loads(path.read_text())
        for row in case['standards']:
            print(f"| {case['instance']} | {row['standard']} | {row['v1']} | {row['ours']} | "
                  f"{'yes' if row['match'] else 'NO'} | {row['receptors_compared']} |")


if __name__ == '__main__':
    main()
