"""Best-known menu search, with conservative field bounds and final ARM2 verification.

For each unassigned source, min_h F[h] times its smallest quantized rate is a
pointwise lower bound even when the best height differs by hour/receptor. Assigned
contributions are fixed. Sums, daily maxima/means, order statistics, annual means
and maximum-over-years are monotone. Bounds allow for final five-place rounding.

ARM2 is also monotone for nonnegative NOx. test_search.py certifies the following
with exact rational Bernstein coefficients on 25-unit intervals: (x*p(x))' > 0 on
[0,300]; p(x) < .5 on [300,1500]; every coefficient of p(1500+t)-.5 is negative.
Thus above 300 the clipped function is .5*x. Below 300, each of its continuous
pieces has positive derivative: .5, .9, or (x*p(x))'. Coefficients are those in
v1 shared/environment/model-source/aermod/calc1.f:10813–10858.

Closing the tree is a statement about the supplied fast fields. The final native
check establishes feasibility of the selected design, not native global ordering.
"""

from copy import deepcopy
from collections import OrderedDict
from decimal import Decimal as D, localcontext, ROUND_HALF_EVEN
from pathlib import Path
import tempfile
import time
from typing import Callable

import numpy as np

from .baselines import (canonical, cheapest_first_greedy, design_cost, option_cost,
                        scenario_view, source_options, verify_evaluation)
from .evaluate import annual_statistics, arm2, evaluate
from .fields import FieldPack, height_key
from .model import Scenario


class LowerBound:
    def __init__(self, task: Scenario, pack: FieldPack):
        if pack.task_id != task.id:
            raise ValueError('field pack task mismatch')
        self.task, self.pack = task, pack
        self.view = scenario_view(task)
        self.options = {s.id: source_options(self.view, s.id) for s in task.sources}
        self.rates: dict[tuple[str, str, str], float] = {}
        self.minima, self.minimum_rates, self.impacts, maxima, means = {}, {}, {}, {}, {}
        controls = {c.id: c for c in task.controls}
        with localcontext() as context:
            context.prec, context.rounding = 28, ROUND_HALF_EVEN
            for source in task.sources:
                arrays = [pack.fields[source.id, height_key(h)] for h in task.height_menu[source.id]]
                expected = (len(pack.valid_hours), len(pack.receptor_indices))
                if any(a.shape != expected or not np.isfinite(a).all() or (a < 0).any() for a in arrays):
                    raise ValueError('bounds require finite nonnegative fields with matching dimensions')
                minimum = np.array(arrays[0], dtype=np.float64)
                for array in arrays[1:]:
                    np.minimum(minimum, array, out=minimum)
                self.minima[source.id] = minimum
                maxima[source.id] = np.maximum.reduce([a.max(axis=0) for a in arrays])
                self.impacts[source.id] = float(maxima[source.id].max()) * sum(float(v) for v in source.uncontrolled_rates_g_s.values())
                means.update({(source.id, height_key(h)): float(a.mean())
                              for h, a in zip(task.height_menu[source.id], arrays)})
                for pollutant in ('NO2', 'PM25', 'SO2'):
                    rates = []
                    for row in self.options[source.id]:
                        control = row['control']
                        efficiency = D(0) if control == 'none' else controls[control].efficiency_by_pollutant[pollutant]
                        rate = (source.uncontrolled_rates_g_s[pollutant]*(1-efficiency)).quantize(D('1e-9'))
                        if rate < 0:
                            raise ValueError('negative source rate')
                        self.rates[source.id, control, pollutant] = float(rate)
                        rates.append(float(rate))
                    self.minimum_rates[source.id, pollutant] = min(rates)
        self.order = sorted(self.options, key=lambda sid: (-self.impacts[sid], sid))
        self.columns, self.bound_receptors = {}, {}
        self.prefixes = {p: OrderedDict() for p in ('NO2', 'PM25', 'SO2')}
        self.suffixes = {p: OrderedDict() for p in ('NO2', 'PM25', 'SO2')}
        for pollutant in self.prefixes:
            upper = sum(float(s.uncontrolled_rates_g_s[pollutant])*maxima[s.id] for s in task.sources)
            upper = np.nextafter(upper, np.inf) + 1e-5 + np.abs(upper)*1e-12
            keep = np.zeros(len(pack.receptor_indices), dtype=bool)
            for standard in task.standards:
                if standard.pollutant == pollutant:
                    keep |= upper + float(standard.background) >= float(standard.limit)
            self.columns[pollutant] = slice(None) if keep.all() else keep
            self.bound_receptors[pollutant] = int(keep.sum())
        # Traversal only: cheap, low-concentration options first; no ordering is a bound.
        for sid, rows in self.options.items():
            burdens = {canonical(row): means[sid, height_key(row['height_m'])] *
                       sum(self.rates[sid, row['control'], p] for p in self.prefixes) for row in rows}
            max_burden = max(burdens.values()) or 1
            max_cost = max(option_cost(self.view, sid, row) for row in rows) or D(1)
            rows.sort(key=lambda row: (float(option_cost(self.view, sid, row)/max_cost) +
                                      burdens[canonical(row)]/max_burden,
                                      option_cost(self.view, sid, row), canonical(row)))

    @staticmethod
    def down(values: np.ndarray) -> np.ndarray:
        return np.maximum(0, np.nextafter(values, -np.inf))

    @staticmethod
    def remember(cache: OrderedDict, key: tuple, value: np.ndarray) -> np.ndarray:
        cache[key] = value
        cache.move_to_end(key)
        # Bound resident memory even on six-source trees and task-parallel batches.
        while len(cache) > 2:
            cache.popitem(last=False)
        return value

    def prefix(self, pollutant: str, keys: tuple) -> np.ndarray:
        cache = self.prefixes[pollutant]
        if keys in cache:
            cache.move_to_end(keys)
            return cache[keys]
        if not keys:
            return np.zeros((len(self.pack.valid_hours), self.bound_receptors[pollutant]))
        parent = self.prefix(pollutant, keys[:-1])
        sid, control, height = keys[-1]
        field = self.pack.fields[sid, height][:, self.columns[pollutant]]
        contribution = self.down(np.multiply(field, self.rates[sid, control, pollutant], dtype=np.float64))
        return self.remember(cache, keys, self.down(parent+contribution))

    def total(self, partial: dict, pollutant: str) -> np.ndarray:
        keys = tuple((sid, partial[sid]['control'], height_key(partial[sid]['height_m']))
                     for sid in self.order if sid in partial)
        missing = tuple(sid for sid in self.order if sid not in partial)
        cache = self.suffixes[pollutant]
        if missing not in cache:
            remaining = np.zeros((len(self.pack.valid_hours), self.bound_receptors[pollutant]))
            for sid in missing:
                minimum = self.minima[sid][:, self.columns[pollutant]]
                contribution = self.down(minimum*self.minimum_rates[sid, pollutant])
                remaining = self.down(remaining+contribution)
            self.remember(cache, missing, remaining)
        cache.move_to_end(missing)
        return self.down(self.prefix(pollutant, keys)+cache[missing])

    def pollutant_values(self, partial: dict, pollutant: str) -> dict[str, D]:
        standards = [s for s in self.task.standards if s.pollutant == pollutant]
        if not self.bound_receptors[pollutant]:
            return {s.id: D(0) for s in standards}
        total = self.total(partial, pollutant)
        if pollutant == 'NO2':
            total = np.where(total <= 300, arm2(np.minimum(total, 300)), .5*total)
        stats = annual_statistics(total, self.pack.valid_hours, self.pack.year_offsets, pollutant)
        result = {}
        for standard in standards:
            kind = 'annual' if standard.averaging_time == 'annual' else (
                'three' if standard.averaging_time == '3h' else 'daily')
            max_form = standard.form in {'max_receptor_max_annual_mean',
                                        'max_receptor_max_annual_second_highest_3h_block'}
            values = stats[kind].max(axis=0) if max_form else stats[kind].mean(axis=0)
            raw = float(values.max())
            lower = max(0., np.nextafter(raw, -np.inf) - 1e-5 - abs(raw)*1e-12)
            result[standard.id] = D(str(lower))
        return result

    def values(self, partial: dict) -> dict[str, D]:
        return {name: value for p in self.prefixes for name, value in self.pollutant_values(partial, p).items()}

    def infeasible(self, partial: dict) -> bool:
        for pollutant in self.prefixes:
            bounds = self.pollutant_values(partial, pollutant)
            if any(bounds[s.id] + s.background > s.limit for s in self.task.standards if s.pollutant == pollutant):
                return True
        return False

class _Limit(Exception):
    pass


def search(task: Scenario, pack: FieldPack, *, max_nodes: int | None = None,
           seconds: float | None = None, workdir: str | Path | None = None,
           verify_fn: Callable | None = None, evaluate_fn: Callable | None = None,
           verified_results: dict[str, dict] | None = None) -> dict:
    """Unlimited evaluations unless time/nodes are bounded; verification is always allowed to finish."""
    if max_nodes is not None and (isinstance(max_nodes, bool) or max_nodes < 0):
        raise ValueError('max_nodes must be nonnegative')
    if seconds is not None and (not np.isfinite(seconds) or seconds < 0):
        raise ValueError('seconds must be finite and nonnegative')
    started = time.monotonic()
    deadline = float('inf') if seconds is None else started+seconds
    view = scenario_view(task)
    incumbent: tuple[D, str, dict] | None = None
    nodes, evaluations = 0, 0
    closed = True

    def check_time() -> None:
        if time.monotonic() >= deadline:
            raise _Limit

    def run(design: dict) -> dict:
        nonlocal incumbent, evaluations
        check_time()
        result = evaluate_fn(design) if evaluate_fn else evaluate(task, design, pack)
        evaluations += 1
        if not result.get('valid', True):
            raise ValueError(f'invalid search candidate: {result}')
        if result['feasible']:
            candidate = (design_cost(view, design), canonical(design), deepcopy(design))
            if incumbent is None or candidate[:2] < incumbent[:2]:
                incumbent = candidate
        return result

    try:
        check_time()
        cheapest_first_greedy(view, run)
        check_time()
        bound = LowerBound(task, pack)
        order = bound.order
        cheapest = [min(option_cost(view, sid, row) for row in bound.options[sid]) for sid in order]

        def visit(partial: dict, depth: int, partial_cost: D) -> None:
            nonlocal nodes
            check_time()
            if max_nodes is not None and nodes >= max_nodes:
                raise _Limit
            nodes += 1
            floor = partial_cost + sum(cheapest[depth:], D(0))
            if incumbent is not None and floor >= incumbent[0]:
                return
            if bound.infeasible(partial):
                return
            if depth == len(order):
                run(partial)
                return
            sid = order[depth]
            for row in bound.options[sid]:
                partial[sid] = row
                visit(partial, depth+1, partial_cost+option_cost(view, sid, row))
            partial.pop(sid)

        visit({}, 0, D(0))
    except _Limit:
        closed = False
    result = {'design': None, 'feasible': False, 'verified': False, 'cost': None,
              'usd_saved': -1, 'nodes': nodes, 'closed': closed, 'evaluations_used': evaluations}
    result['search_seconds'] = time.monotonic()-started
    result['bound_receptors'] = bound.bound_receptors if 'bound' in locals() else {}
    result['verification_reused'] = False
    if incumbent is not None:
        if workdir is not None:
            Path(workdir).mkdir(parents=True, exist_ok=True)
        key = canonical(incumbent[2])
        if verified_results is not None and key in verified_results:
            checked = verified_results[key]
            result['verification_reused'] = True
        else:
            with tempfile.TemporaryDirectory(prefix='search-', dir=workdir) as scratch:
                checked = verify_evaluation(task, incumbent[2], Path(scratch), verify_fn=verify_fn)
        if not checked['feasible'] or checked['cost'] != float(incumbent[0]):
            raise ValueError(f'final search verification disagrees with fast evaluator: {checked}')
        result.update(design=incumbent[2], feasible=True, verified=True, cost=checked['cost'],
                      usd_saved=float(D(str(view['draft_cost']))-incumbent[0]), verification=checked)
    result['seconds'] = time.monotonic()-started
    result['verification_seconds'] = result['seconds']-result['search_seconds']
    return result
