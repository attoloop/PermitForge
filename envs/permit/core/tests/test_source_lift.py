import ast
import inspect
from pathlib import Path

from conftest import EXTRACT
from permit_core import cost, native, render, reward, validate


def definitions(path):
    return {node.name: node for node in ast.parse(path.read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


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
            assert ast.dump(lifted[name]) == ast.dump(original[name]), (module.__name__, name)


def test_calibration_verbatim():
    source = (Path(__file__).parent / "fixtures/calibration.py.txt").read_text()
    function = inspect.getsource(reward.hyperbolic_calibration)
    assert function in source
