"""Portable historical benchmark evidence: uv run pytest tests/test_search_reference.py -q."""

import hashlib
import json
from pathlib import Path

from convert import benchmark_search


ROOT = Path(__file__).resolve().parents[1]


def test_preserved_search_reference_matches_measurements():
    reference = ROOT / 'results/evidence/search/reference.py.txt'
    implementation, source = benchmark_search.search_implementation(reference_file=reference)
    assert callable(implementation.search)
    for path in (ROOT / 'results/evidence/search').glob('before-*.json'):
        receipt = json.loads(path.read_text())
        assert hashlib.sha256(source.encode()).hexdigest() == receipt['search_sha256']
        assert '--reference-file results/evidence/search/reference.py.txt' in receipt['command']


def test_default_search_implementation_is_unchanged():
    implementation, source = benchmark_search.search_implementation()
    assert implementation is benchmark_search.search_module
    assert source == Path(implementation.__file__).read_text()
