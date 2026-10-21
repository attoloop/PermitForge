"""Skip, with a reason, when a test needs something a public checkout does not have (not a conftest: train/tests has one)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SMOKE_ROLLOUTS = ROOT / 'results/rollouts/wp2.0b-smoke'
TASK_DATA = ROOT / 'envs/permit/tasks/data'
requires_task_data = pytest.mark.skipif(not TASK_DATA.exists(), reason="task data is not in a checkout; download tasks/data and tasks/met from the dataset (README, Run it locally)")
SMOKE_BASE_EVAL = ROOT / 'train/cloud/data/smoke-result-grade/smoke/base-eval'
requires_smoke_rollouts = pytest.mark.skipif(not SMOKE_ROLLOUTS.exists(), reason='the local smoke rollouts are not in a public checkout')
requires_smoke_base_eval = pytest.mark.skipif(not SMOKE_BASE_EVAL.exists(), reason='the smoke base-eval episodes are in the dataset (training/runs/smoke-grade), not the checkout')


def cached_tokenizer():
    """The pinned Qwen tokenizer from the local cache; skip if it was never downloaded."""
    from train.cloud.formatting import load_tokenizer
    try:
        return load_tokenizer(local_files_only=True)
    except OSError:
        pytest.skip('the pinned Qwen3.8-27B tokenizer is not in the local Hugging Face cache')
