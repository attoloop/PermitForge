"""search certificates: uv run pytest -q tests/test_search.py -s."""

from decimal import Decimal as D
from fractions import Fraction as F
from itertools import product
import json
from math import comb
import time

import numpy as np
import pytest

from permit_core.baselines import scenario_view, source_options
from permit_core.evaluate import arm2, evaluate
from permit_core.search import LowerBound, search
from wp14_fixtures import synthetic


def completions(view, partial=None):
    partial = partial or {}
    missing = [s for s in view['sources'] if s['id'] not in partial]
    for choices in product(*(source_options(view, s['id']) for s in missing)):
        yield {**partial, **{s['id']: choice for s, choice in zip(missing, choices)}}


def test_arm2_monotonicity_exact_polynomial_certificate():
    # Exact rational Bernstein coefficients bound a polynomial by their convex hull.
    ratio = [F(x) for x in ('1.2441', '-2.7383e-3', '-5.6062e-6', '3.4555e-8',
                            '-5.8345e-11', '4.2795e-14', '-1.1723e-17')]
    derivative = [(i+1)*v for i, v in enumerate(ratio)]
    def shifted(poly, a, width=1):
        return [sum(poly[i]*comb(i, k)*a**(i-k)*width**k for i in range(k, len(poly)))
                for k in range(len(poly))]
    def bernstein(poly, a, b):
        power = shifted(poly, F(a), F(b-a))
        n = len(poly)-1
        return [sum(power[k]*F(comb(j, k), comb(n, k)) for k in range(j+1)) for j in range(n+1)]
    for a in range(0, 300, 25):
        assert min(bernstein(derivative, a, a+25)) > 0
    below = [ratio[0]-F('.5'), *ratio[1:]]
    for a in range(300, 1500, 25):
        assert max(bernstein(below, a, a+25)) < 0
    assert all(x < 0 for x in shifted(below, F(1500)))
    # Clipped branches have slopes .5/.9; the unclipped branch has positive derivative.
    x = np.unique(np.r_[np.linspace(0, 400, 40001), np.geomspace(400, 1e8, 1000)])
    assert np.all(np.diff(arm2(x)) >= 0)


@pytest.mark.parametrize('seed', [14, 41, 140])
@pytest.mark.parametrize('scale', [.001, 1, 30, 1000])
def test_lower_bound_never_exceeds_random_completions(seed, scale):
    task, pack = synthetic(3, seed=seed)
    for field in pack.fields.values():
        field *= scale
    view = scenario_view(task)
    bounds = LowerBound(task, pack)
    rng = np.random.default_rng(seed)
    for _ in range(30):
        design = {s['id']: source_options(view, s['id'])[rng.integers(6)] for s in view['sources']}
        partial = {sid: row for sid, row in design.items() if rng.random() < .5}
        bound = bounds.values(partial)
        for _ in range(5):
            completion = {s['id']: partial.get(s['id'], source_options(view, s['id'])[rng.integers(6)])
                          for s in view['sources']}
            actual = evaluate(task, completion, pack)
            assert all(bound[s['id']] <= D(str(s['value'])) for s in actual['standards'])


@pytest.mark.parametrize('seed', range(5))
def test_random_search_equals_brute_force(seed, tmp_path):
    from permit_core.verify import Verification
    task, pack = synthetic(3, seed=seed)
    evaluated = [evaluate(task, d, pack) for d in completions(scenario_view(task))]
    best = min(r['cost'] for r in evaluated if r['feasible'])
    def verify(task, ledger, workdir, *, no2_tier):
        result = evaluate(task, {s: {'control': r['control'], 'height_m': r['stack_height_m']}
                                 for s, r in ledger.items()}, pack)
        return Verification(feasible=result['feasible'], cost=D(str(result['cost'])))
    result = search(task, pack, verify_fn=verify, workdir=tmp_path)
    assert result['closed'] and result['cost'] == best


def test_no_feasible_design_is_closed_sentinel(tmp_path):
    task, pack = synthetic()
    for field in pack.fields.values():
        field *= 1000
    def forbidden(*args, **kwargs):
        pytest.fail('no design should be submitted to verify')
    result = search(task, pack, verify_fn=forbidden, workdir=tmp_path)
    assert result['closed'] and result['design'] is None and not result['verified']
    assert result['usd_saved'] == -1 and result['cost'] is None


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_branch_and_bound_equals_brute_force(n, evidence_dir):
    task, pack = synthetic(n)
    view = scenario_view(task)
    start = time.monotonic()
    evaluated = [evaluate(task, d, pack) for d in completions(view)]
    brute_seconds = time.monotonic()-start
    best = min(r['cost'] for r in evaluated if r['feasible'])
    checked = []
    def verifier(task_arg, ledger, workdir, *, no2_tier):
        from permit_core.verify import Verification
        assert task_arg is task and no2_tier == 'ARM2'
        design = {s: {'control': r['control'], 'height_m': r['stack_height_m']} for s, r in ledger.items()}
        checked.append(design)
        result = evaluate(task, design, pack)
        return Verification(feasible=result['feasible'], cost=D(str(result['cost'])))
    result = search(task, pack, verify_fn=verifier, workdir=evidence_dir)
    assert result['closed'] and result['cost'] == best
    assert checked == [result['design']]
    assert result['nodes'] < len(evaluated)
    row = {'sources': n, 'designs': len(evaluated), 'brute_force_cost': best,
           'search_cost': result['cost'], 'nodes': result['nodes'], 'closed': result['closed'],
           'brute_seconds': brute_seconds, 'search_seconds': result['seconds']}
    print(json.dumps(row))
    (evidence_dir / f'wp14-search-{n}.json').write_text(json.dumps(row, indent=2)+'\n')


def test_limits_and_failed_final_verification(tmp_path):
    from permit_core.verify import Verification
    task, pack = synthetic(3)
    calls = []
    def verifier(task, ledger, workdir, *, no2_tier):
        calls.append(ledger)
        return Verification(feasible=True, cost=D(24))
    result = search(task, pack, max_nodes=0, verify_fn=verifier, workdir=tmp_path)
    assert not result['closed'] and result['nodes'] == 0 and len(calls) == 1
    result = search(task, pack, seconds=0, verify_fn=verifier, workdir=tmp_path)
    assert not result['closed'] and result['design'] is None and len(calls) == 1
    with pytest.raises(ValueError, match='verification'):
        search(task, pack, verify_fn=lambda *a, **kw: Verification(), workdir=tmp_path)
