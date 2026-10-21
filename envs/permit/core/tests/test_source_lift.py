import ast
import copy
from pathlib import Path

from conftest import EXTRACT, requires_v1
from permit_core import cost, native, render, validate

pytestmark = requires_v1


def definitions(path):
    return {node.name: node for node in ast.parse(path.read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def without_size_scan(node):
    """Split off `size = ...`: run_native's disk scan is the one documented deviation (README)."""
    node = copy.deepcopy(node)
    scans = []
    for parent in ast.walk(node):
        for _, value in ast.iter_fields(parent):
            if isinstance(value, list):
                kept = []
                for child in value:
                    if (isinstance(child, ast.Assign) and len(child.targets) == 1
                            and isinstance(child.targets[0], ast.Name) and child.targets[0].id == "size"):
                        scans.append(ast.dump(child.value))
                    else:
                        kept.append(child)
                value[:] = kept
    return node, scans


def expression(source):
    return ast.dump(ast.parse(source).body[0].value)


def test_published_functions_unchanged():
    shared = EXTRACT / "shared"
    sources = [
        (native, shared / "environment/public_native.py", None),
        (native, shared / "environment/service.py", ["calendar", "parse_blocks"]),
        (validate, shared / "solution/validation.py",
         ["StrictLoader", "design", "plot_values", "statistics", "admit_air_quality"]),
        (cost, shared / "tests/grade.py", ["cost", "costs"]),
        (render, shared / "environment/render.py", ["bpip_template", "is_number"]),
    ]
    for module, path, names in sources:
        original = definitions(path)
        lifted = definitions(Path(module.__file__))
        for name in names or original:
            if name == "run_native":
                lifted_rest, lifted_scan = without_size_scan(lifted[name])
                original_rest, original_scan = without_size_scan(original[name])
                assert ast.dump(lifted_rest) == ast.dump(original_rest), name
                assert original_scan == [expression("sum(f.stat().st_size for f in cwd.iterdir() if f.is_file())")]
                assert lifted_scan == [expression("case_bytes(cwd)")]
                continue
            assert ast.dump(lifted[name]) == ast.dump(original[name]), (module.__name__, name)
