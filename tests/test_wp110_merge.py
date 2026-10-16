"""Shared height menus and conversion's read-only identity check."""
import json

from convert import v1_to_v2
from permit_core.menu import height_menu


def test_converter_imports_shared_menu():
    assert v1_to_v2.height_menu is height_menu


def test_check_rejects_stale_record_menu_without_changing_files(tmp_path):
    task = {'scenario': {'units': [{'id': 'S1', 'stack_bounds': {'height_min_m': 10, 'gep_height_m': 65}}]},
            'height_menu': {'S1': [float(h) for h in height_menu(10, 65)]}}
    path = tmp_path/'data/fixture/task.json'
    v1_to_v2.write_json(path, task)
    row = {'id': 'fixture', 'height_menu': {'S1': [10, 65]}, 'station': 'TEST', 'met_sha256': {},
           'sha256': {'task.json': v1_to_v2.sha256(path)}}
    before = path.read_bytes()
    assert any('height menu' in e for e in v1_to_v2.check_row_integrity(tmp_path, row))
    row['height_menu'] = task['height_menu']
    assert v1_to_v2.check_row_integrity(tmp_path, row) == []
    assert json.loads(path.read_text()) == task and path.read_bytes() == before
