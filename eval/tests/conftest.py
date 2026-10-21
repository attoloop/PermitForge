"""Skip the tests that drive the environment when a checkout has no task data."""
from pathlib import Path

import pytest

TASK_DATA = Path(__file__).resolve().parents[2] / 'envs/permit/tasks/data'
NEEDS_DATA = ('test_capture.py', 'test_extended_budget.py', 'test_transport.py', 'test_wp28b.py')


def pytest_collection_modifyitems(config, items):
    if TASK_DATA.exists():
        return
    for item in items:
        if item.fspath.basename in NEEDS_DATA:
            item.add_marker(pytest.mark.skip(reason="task data is not in a checkout; download tasks/data and tasks/met from the dataset (README, Run it locally)"))
