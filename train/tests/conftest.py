from decimal import Decimal as D
from pathlib import Path

import numpy as np
import pytest

from permit_core.fields import FieldPack
from permit_core.model import ControlRow, Scenario, Source, Standard
from permit_openenv.tasks import Task, TaskPack


@pytest.fixture
def task_pack(tmp_path):
    source = Source('S1', 'synthetic', {'x_m': D(0), 'y_m': D(0)}, {},
                    {p: D(1) for p in ('NO2', 'PM25', 'SO2')}, D('.1'), D(10), D(20), D(20))
    controls = tuple(ControlRow('S1_'+level, ('S1',), D(price),
                                {p: D(eff) for p in ('NO2', 'PM25', 'SO2')})
                     for level, price, eff in [('none', 0, 0), ('moderate', 6, '.5'), ('maximum', 12, '.9')])
    standard = Standard('PM25_annual_primary', 'PM25', 'annual', 'mean_annual', D(9), D(0), 'ug/m3')
    raw = {'scenario': {'units': [{'id': 'S1', 'stack_bounds': {
        'height_min_m': 10, 'height_max_m': 20, 'gep_height_m': 20}}],
        'locked_keywords': {'meteorology_years': [2020]}},
        'costs': {'stack_usd_per_m': 1, 'controls': [
            {'id': c.id, 'applies_to': list(c.applies_to), 'annualized_usd': c.annualized_usd,
             'efficiency_by_pollutant': c.efficiency_by_pollutant} for c in controls]}}
    scenario = Scenario('synthetic-train', 'SYNTHETIC', tmp_path, tmp_path, Path('.'), raw,
                        (source,), controls, (standard,), {}, ((D(0), D(0)),),
                        {'S1': {'control': 'S1_maximum', 'stack_height_m': D(20)}},
                        {'S1': (D(10), D(20))})
    task = Task(scenario.id, 'train', {'eligible': True})
    task.__dict__['scenario'] = scenario
    eval_task = Task('synthetic-eval', 'eval', task.record.copy())
    eval_task.__dict__['scenario'] = scenario
    fields = FieldPack(scenario.id, {('S1', '10'): np.full((240, 1), 10.),
                                   ('S1', '20'): np.full((240, 1), 8.)},
                       np.ones(240, dtype=bool), (0, 240), np.array([0]))
    return TaskPack([task, eval_task]), fields


@pytest.fixture
def env(task_pack):
    from train.permit_env import PermitTrainingEnv
    pack, fields = task_pack
    return PermitTrainingEnv(pack=pack, fields_loader=lambda task: fields)


@pytest.fixture
def design():
    return {'S1': {'control': 'S1_moderate', 'height_m': 10}}


@pytest.fixture
def episode(env, design):
    env.reset(index=0)
    env.get_scenario()
    env.evaluate_design(design)
    env.submit_design(design)
    env.get_reward()
    return env._record()
