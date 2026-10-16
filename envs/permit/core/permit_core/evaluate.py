"""Experimental ARM2 field evaluator; production use requires the fidelity gate."""

from decimal import Decimal as D, ROUND_HALF_EVEN, localcontext
from typing import Any

import numpy as np

from .cost import cost
from .menu import height_menu
from .model import Scenario
from .native import Reject
from .validate import validate_ledger


def arm2(nox: np.ndarray) -> np.ndarray:
    """calc1.f:10813–10858; default clipping in coset.f:657–663."""
    # Horner evaluation changes only double-precision operation ordering.
    ratio = np.polynomial.polynomial.polyval(
        nox, (1.2441, -2.7383e-3, -5.6062e-6, 3.4555e-8,
              -5.8345e-11, 4.2795e-14, -1.1723e-17))
    return nox * np.clip(ratio, .5, .9)


def rounded(value: float | D, kind: str) -> D:
    if kind == 'plot':
        return D(format(float(value), '.5f'))
    return D(str(value)).quantize(D('.00001'), rounding=ROUND_HALF_EVEN)


def annual_statistics(values: np.ndarray, valid: np.ndarray, offsets: tuple[int, ...],
                      pollutant: str) -> dict[str, np.ndarray]:
    """Return unrounded per-year statistics, with native short-term denominators."""
    if values.ndim != 2 or len(valid) != len(values) or offsets[0] != 0 or offsets[-1] != len(values):
        raise ValueError('field/calendar shape mismatch')
    annual, daily, three = [], [], []
    for start, end in zip(offsets[:-1], offsets[1:]):
        flags = valid[start:end]
        if not flags.any() or (end-start) % 24:
            raise ValueError('year must contain whole days and valid hours')
        hours = np.where(flags[:, None], values[start:end], 0.0)
        annual.append(hours.sum(axis=0)/flags.sum())
        days = hours.reshape(-1, 24, values.shape[1])
        if pollutant == 'PM25':
            day_values = days.sum(axis=1)/np.maximum(flags.reshape(-1, 24).sum(axis=1), 18)[:, None]
        else:
            day_values = days.max(axis=1)
        rank = 4 if pollutant == 'SO2' else 8
        if len(day_values) < rank:
            raise ValueError('not enough days for regulatory rank')
        daily.append(np.partition(day_values, -rank, axis=0)[-rank])
        if pollutant == 'SO2':
            blocks = hours.reshape(-1, 3, values.shape[1]).sum(axis=1)/3
            three.append(np.partition(blocks, -2, axis=0)[-2])
    result = {'annual': np.array(annual), 'daily': np.array(daily)}
    if three:
        result['three'] = np.array(three)
    return result


def normalize_design(task: Scenario, design: object) -> dict:
    if not isinstance(design, dict) or any(not isinstance(r, dict) or
            set(r) != {'control', 'height_m'} for r in design.values()):
        raise Reject('E_MENU', 'expected source rows with control,height_m only')
    ledger = {sid: {'control': row['control'], 'stack_height_m': row['height_m']}
              for sid, row in design.items()}
    menus = task.height_menu or {s.id: height_menu(s.height_min_m, s.gep_height_m) for s in task.sources}
    return validate_ledger(ledger, task.task, height_menu=menus)


def evaluate(task: Scenario, design: object, pack: Any, *, include_receptors: bool = False) -> dict:
    """Return the JSON Evaluation contract. Invalid designs never access the pack."""
    with localcontext() as context:
        context.prec = 28
        context.rounding = ROUND_HALF_EVEN
        try:
            ledger = normalize_design(task, design)
        except Reject as exc:
            return {'valid': False, 'errors': [f'{exc.code}: {exc.detail}']}
        from .fields import height_key, load_pack
        if not hasattr(pack, 'fields'):
            pack = load_pack(pack)
        if pack.task_id != task.id:
            raise ValueError('field pack task mismatch')
        if len(pack.receptor_indices) != len(task.receptors) and not pack.binding_certified:
            raise ValueError('omitted receptors require a binding-value certificate')
        if include_receptors and sorted(map(int, pack.receptor_indices)) != list(range(len(task.receptors))):
            raise ValueError('full receptor arrays require an unpruned pack')
        controls = {c.id: c for c in task.controls}
        results = []
        receptor_values, receptor_margins = {}, {}
        for pollutant in ('NO2', 'PM25', 'SO2'):
            total = np.zeros((len(pack.valid_hours), len(pack.receptor_indices)))
            for source in task.sources:
                choice = ledger[source.id]
                efficiency = (D(0) if choice['control'] == 'none' else
                              controls[choice['control']].efficiency_by_pollutant[pollutant])
                # Match the rate actually rendered into SRCPARAM, including small PM rates.
                rate = (D(str(source.uncontrolled_rates_g_s[pollutant]))*(1-efficiency)).quantize(D('1e-9'))
                total += np.multiply(pack.fields[source.id, height_key(choice['stack_height_m'])],
                                     float(rate), dtype=np.float64)
            if pollutant == 'NO2':
                total = arm2(total)
            stats = annual_statistics(total, pack.valid_hours, pack.year_offsets, pollutant)
            for standard in task.standards:
                if standard.pollutant != pollutant:
                    continue
                kind = 'annual' if standard.averaging_time == 'annual' else (
                    'three' if standard.averaging_time == '3h' else 'daily')
                max_form = standard.form in {'max_receptor_max_annual_mean',
                                            'max_receptor_max_annual_second_highest_3h_block'}
                if pollutant == 'SO2' and kind == 'daily':
                    values = [rounded(sum(D(format(float(v), '.17e')) for v in col)/len(col), 'decimal')
                              for col in stats[kind].T]
                else:
                    raw = stats[kind].max(axis=0) if max_form else stats[kind].mean(axis=0)
                    values = [rounded(v, 'decimal' if max_form else 'plot') for v in raw]
                binding = max(range(len(values)), key=values.__getitem__)
                index = int(pack.receptor_indices[binding])
                x, y = task.receptors[index]
                value = values[binding]
                margin = standard.limit - standard.background - value
                if include_receptors:
                    order = np.argsort(pack.receptor_indices)
                    receptor_values[standard.id] = [float(values[i]) for i in order]
                    receptor_margins[standard.id] = [float(standard.limit-standard.background-values[i]) for i in order]
                results.append({'id': standard.id, 'value': float(value),
                                'background': float(standard.background), 'limit': float(standard.limit),
                                'margin': float(margin),
                                'binding_receptor': {'index': index+1, 'x_m': float(x), 'y_m': float(y)}})
        return {'valid': True, 'errors': [], 'feasible': all(s['margin'] >= 0 for s in results),
                'cost': float(cost(ledger, task.task)), 'standards': results,
                **({'receptor_values': receptor_values, 'receptor_margins': receptor_margins}
                   if include_receptors else {})}
