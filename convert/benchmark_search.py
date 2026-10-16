"""Measure real search and public baselines, retaining native verification evidence.

uv run --project envs/permit/core python convert/benchmark_search.py --task TASK_ID --label before
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import time
import subprocess
import types
import resource

from permit_core import search as search_module
from permit_core.baselines import cheapest_first_greedy, scenario_view, v1_coordinate_descent, verify_evaluation
from permit_core.evaluate import evaluate
from permit_core.fields import cached_pack, certified_prune
from permit_core.menu import height_menu
from permit_core.model import load_v1_instance


def search_implementation(*, reference_file: Path | None = None,
                          reference_revision: str | None = None) -> tuple[types.ModuleType, str]:
    """Load a trusted local benchmark source, or the current implementation."""
    if reference_file is not None:
        source = reference_file.read_text()
    elif reference_revision:
        source = subprocess.run(['git', 'show', f'{reference_revision}:envs/permit/core/permit_core/search.py'],
                                check=True, capture_output=True, text=True).stdout
    else:
        return search_module, Path(search_module.__file__).read_text()
    implementation = types.ModuleType('permit_core.search_reference')
    implementation.__package__ = 'permit_core'
    exec(compile(source, '<reference-search>', 'exec'), implementation.__dict__)
    return implementation, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--seconds', type=float, default=1200)
    parser.add_argument('--max-nodes', type=int)
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument('--reference-revision', help='read trusted search code from a local git revision')
    reference.add_argument('--reference-file', type=Path, help='read trusted preserved search source')
    parser.add_argument('--measure-fields', action='store_true', help='measure a fresh disposable on-demand pack, then stop')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    implementation, source = search_implementation(reference_file=args.reference_file,
                                                   reference_revision=args.reference_revision)
    task = load_v1_instance(args.task)
    task = replace(task, height_menu={s.id: height_menu(s.height_min_m, s.gep_height_m) for s in task.sources})
    destination = Path('results/evidence/search') / f'{args.label}-{task.id}.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.measure_fields:
        from permit_core import fields
        def cpu() -> float:
            return sum(r.ru_utime+r.ru_stime for r in (
                resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))
        original_root = fields.cache_root
        with tempfile.TemporaryDirectory(prefix='permitforge-probe-', dir=original_root()) as scratch:
            fields.cache_root = lambda: Path(scratch)
            start, cpu_start = time.monotonic(), cpu()
            try:
                with fields.ensure_pack(task, jobs=args.jobs) as pack:
                    result = {'task_id': task.id, 'sources': len(task.sources), 'fields': len(pack.fields),
                              'jobs': args.jobs, 'wall_seconds': time.monotonic()-start,
                              'cpu_seconds': cpu()-cpu_start,
                              'command': f'uv run --project envs/permit/core python convert/benchmark_search.py '
                                         f'--task {task.id} --label {args.label} --measure-fields --jobs {args.jobs}'}
            finally:
                fields.cache_root = original_root
        destination.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
        return
    started = time.monotonic()
    pack, pruning = certified_prune(task, cached_pack(task))
    report = {'task_id': task.id, 'sources': len(task.sources), 'label': args.label,
              'seconds_limit': args.seconds, 'max_nodes': args.max_nodes, 'pruning': pruning,
              'pack_seconds': time.monotonic()-started,
              'search_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'command': f'uv run --project envs/permit/core python convert/benchmark_search.py --task {task.id} '
                         f'--label {args.label} --seconds {args.seconds}' +
                         (f' --max-nodes {args.max_nodes}' if args.max_nodes is not None else '') +
                         (f' --reference-revision {args.reference_revision}' if args.reference_revision else '') +
                         (f' --reference-file {args.reference_file}' if args.reference_file else ''),
              'baselines': {}}
    def save() -> None:
        destination.write_text(json.dumps(report, indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='permitforge-search-') as work:
        for policy in (cheapest_first_greedy, v1_coordinate_descent):
            start = time.monotonic()
            result = policy(scenario_view(task), lambda d: evaluate(task, d, pack))
            result['evaluation_seconds'] = time.monotonic()-start
            if result['design'] is not None:
                result['verification'] = verify_evaluation(task, result['design'], Path(work))
                assert result['verification']['feasible'] == result['feasible']
            result['seconds'] = time.monotonic()-start
            report['baselines'][policy.__name__] = result
            save()
            print(task.id, policy.__name__, result['cost'], flush=True)
        report['search'] = implementation.search(task, pack, seconds=args.seconds,
                                               max_nodes=args.max_nodes, workdir=work)
    save()
    print(json.dumps({k: v for k, v in report['search'].items() if k not in {'design', 'verification'}}), flush=True)


if __name__ == '__main__':
    main()
