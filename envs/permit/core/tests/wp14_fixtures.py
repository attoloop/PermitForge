"""Small synthetic field packs shared by tests; no native computation."""

from decimal import Decimal as D
from pathlib import Path

import numpy as np

from permit_core.fields import FieldPack
from permit_core.model import ControlRow, Scenario, Source, Standard


def synthetic(sources: int = 2, *, seed: int | None = None) -> tuple[Scenario, FieldPack]:
    units, controls, fields, draft = [], [], {}, {}
    rng = np.random.default_rng(seed)
    for i in range(sources):
        sid = f'S{i+1}'
        units.append(Source(sid, 'synthetic', {'x_m': D(i), 'y_m': D(0)}, {},
                            {p: D(1) for p in ('NO2', 'PM25', 'SO2')}, D('.1'), D(10), D(20), D(20)))
        for level, price, efficiency in [('none', 0, 0), ('moderate', 6, '.5'), ('maximum', 12, '.9')]:
            controls.append(ControlRow(sid+'_'+level, (sid,), D(price),
                                       {p: D(efficiency) for p in ('NO2', 'PM25', 'SO2')}))
        for height, concentrations in [(10, [10., 6.]), (20, [4., 8.])]:
            # Taller is better at receptor 0 and worse at receptor 1.
            field = np.tile(concentrations, (240, 1))
            if seed is not None:
                field *= rng.uniform(.5, 1.5, field.shape)
            fields[sid, str(height)] = field
        draft[sid] = {'control': sid+'_maximum', 'stack_height_m': D(20)}
    standards = tuple(Standard(name, p, averaging, form, D(limit), D(0), 'ug/m3')
                      for name, p, averaging, form, limit in [
                          ('NO2_1h', 'NO2', '1h', 'mean_yearly_rank', 12),
                          ('NO2_annual', 'NO2', 'annual', 'max_receptor_max_annual_mean', 12),
                          ('PM25_24h', 'PM25', '24h', 'mean_yearly_rank', 12),
                          ('PM25_annual_primary', 'PM25', 'annual', 'mean_annual', 12),
                          ('SO2_1h', 'SO2', '1h', 'mean_yearly_rank', 12),
                          ('SO2_3h', 'SO2', '3h', 'max_receptor_max_annual_second_highest_3h_block', 12),
                          ('SO2_annual', 'SO2', 'annual', 'mean_annual', 12)])
    task = {'scenario': {'units': [
        {'id': u.id, 'stack_bounds': {'height_min_m': u.height_min_m, 'height_max_m': u.height_max_m,
                                    'gep_height_m': u.gep_height_m},
         'uncontrolled_rates_g_s': u.uncontrolled_rates_g_s} for u in units],
        'locked_keywords': {'meteorology_years': [2020]}},
        'costs': {'stack_usd_per_m': D('.2'), 'controls': [
            {'id': c.id, 'applies_to': list(c.applies_to), 'annualized_usd': c.annualized_usd,
             'efficiency_by_pollutant': c.efficiency_by_pollutant} for c in controls]}}
    scenario = Scenario('synthetic', 'SYNTHETIC', Path('.'), Path('.'), Path('.'), task,
                        tuple(units), tuple(controls), standards, {}, ((D(0), D(0)), (D(1), D(0))),
                        draft, {u.id: (D(10), D(20)) for u in units})
    pack = FieldPack(scenario.id, fields, np.ones(240, dtype=bool), (0, 240), np.arange(2))
    return scenario, pack
