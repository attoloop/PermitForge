import copy

from pathlib import Path

import pytest

from permit_core.cost import cost
from permit_openenv.environment import PermitForgeEnvironment
from permit_openenv.tasks import load_pack

TASK_DATA = Path(__file__).resolve().parents[2] / "tasks/data"


def pytest_collection_modifyitems(config, items):
    if not TASK_DATA.exists():
        for item in items:
            item.add_marker(pytest.mark.skip(reason="task data is not in a checkout; download tasks/data and tasks/met from the dataset (README, Run it locally)"))


class StubEvaluator:
    def __init__(self, feasible=True):
        self.calls = []
        self.feasible = feasible

    def evaluate(self, task, design):
        self.calls.append(copy.deepcopy(design))
        ledger = {s: {'control': r['control'], 'stack_height_m': r['height_m']} for s, r in design.items()}
        return {'valid': True, 'errors': [], 'feasible': self.feasible,
                'cost': float(cost(ledger, task.scenario.task)),
                'standards': [{'id': s.id, 'value': 1.0, 'background': float(s.background),
                               'limit': float(s.limit), 'margin': float(s.limit-s.background-1),
                               'binding_receptor': {'index': 1, 'x_m': -250.0, 'y_m': -250.0}}
                              for s in task.scenario.standards]}


@pytest.fixture(scope='session')
def pack():
    return load_pack()


@pytest.fixture
def env(pack):
    instance = PermitForgeEnvironment(pack, evaluator=StubEvaluator(), verifier=StubEvaluator())
    instance.reset(task_id='003bd328a392-s47')
    yield instance
    instance.close()
