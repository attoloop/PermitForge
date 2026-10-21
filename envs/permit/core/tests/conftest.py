from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[4]
EXTRACT = REPO / "convert/v1-src"
requires_v1 = pytest.mark.skipif(not EXTRACT.exists(), reason="convert/v1-src (the v1 extraction) is not in a public checkout")
TASK_DATA = REPO / "envs/permit/tasks/data"
requires_task_data = pytest.mark.skipif(not TASK_DATA.exists(), reason="task data is not in a checkout; download tasks/data and tasks/met from the dataset (README, Run it locally)")
BINARIES = REPO / "envs/permit/core/native_build/bin"
requires_native_binaries = pytest.mark.skipif(not (BINARIES / "aermod").exists(), reason="native binaries are not built; run sh envs/permit/core/native_build/build.sh")


def pytest_addoption(parser):
    parser.addoption("--evidence-dir", type=Path, default=None)


@pytest.fixture
def scenario():
    if not EXTRACT.exists():
        pytest.skip("convert/v1-src (the v1 extraction) is not in a public checkout")
    from permit_core.model import load_v1_instance

    return load_v1_instance("003bd328a392-s47")


@pytest.fixture
def evidence_dir(request, tmp_path):
    result = request.config.getoption("--evidence-dir") or tmp_path
    result.mkdir(parents=True, exist_ok=True)
    return result
