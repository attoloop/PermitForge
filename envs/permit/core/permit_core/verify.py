# v1 origin: tasks/permit-forge/instances/003bd328a392-s47/tests/grade.py (physical grade path)
"""Run the published BPIP/AERMOD physical checks on a design dictionary."""

from dataclasses import dataclass, field
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from .cost import cost
from .model import Scenario
from .native import Reject, bpip_results, cases, require, run_native
from .render import No2Tier, render, with_no2_tier
from .validate import admit_air_quality, statistics, validate_ledger


@dataclass(frozen=True)
class DesignValue:
    value: Decimal
    background: Decimal
    limit: Decimal
    margin: Decimal
    receptor_index: int
    x: Decimal
    y: Decimal


@dataclass
class Verification:
    feasible: bool = False
    design_values: dict[str, DesignValue] = field(default_factory=dict)
    gep: dict[str, Decimal] = field(default_factory=dict)
    cost: Decimal | None = None
    errors: list[dict[str, str]] = field(default_factory=list)
    native_seconds: dict[str, float] = field(default_factory=dict)
    receptor_values: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    binary_sha256: dict[str, str] = field(default_factory=dict)


def models_dir() -> Path:
    return Path(os.environ.get(
        'PERMIT_MODELS_DIR', Path(__file__).resolve().parents[1] / 'native_build/bin'
    )).resolve()


def run_bpip(input_path: Path, output: Path, guard: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, output / 'fort.10')
    run_native(models_dir() / 'bpipprm', output, guard, {'resources': {'native_launches': 0}})


def verify(scenario: Scenario, design: dict, workdir: str | Path, *,
           no2_tier: No2Tier = 'as_published') -> Verification:
    result = Verification()
    # The v1 grader starts in a fresh process with the default Decimal context.
    with localcontext() as context:
        context.prec = 28
        context.rounding = ROUND_HALF_EVEN
        try:
            ledger = validate_ledger(design, scenario.task, height_menu=scenario.height_menu)
            result.cost = cost(ledger, scenario.task)
            task = with_no2_tier(scenario.task, no2_tier)
            binaries = models_dir()
            result.binary_sha256 = {name: hashlib.sha256((binaries / name).read_bytes()).hexdigest()
                                    for name in ('bpipprm', 'aermod')}
            workdir = Path(workdir).resolve()
            workdir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='permit-verify-', dir=workdir) as tmp:
                scratch = Path(tmp)
                down = {}

                def bpip(input_path: Path, output: Path) -> None:
                    start = time.monotonic()
                    try:
                        run_bpip(input_path, output, task['guards']['native_seconds'])
                    finally:
                        result.native_seconds['bpipprm'] = time.monotonic() - start
                    directional, result.gep = bpip_results(output / 'fort.12', ledger)
                    down.update(directional)
                    for sid, row in ledger.items():
                        require(row['stack_height_m'] <= result.gep[sid], 'E_GEP',
                                f'{sid}: height {row["stack_height_m"]} > computed GEP {result.gep[sid]}')

                permit = scratch / 'permit'
                render(scenario.task_root, ledger, permit, bpip, task=task)
                suite = cases((permit / 'aermod.inp').read_text(), task, ledger, down)
                for pollutant, text in suite:
                    directory = scratch / pollutant
                    directory.mkdir()
                    # Short relative paths also work when the extraction path contains spaces.
                    (directory / 'met').symlink_to(scenario.met_path, target_is_directory=True)
                    text = text.replace('/task/met/', 'met/')
                    text = text.replace('/task/', str(scenario.task_root) + '/')
                    (directory / 'aermod.inp').write_text(text)
                    start = time.monotonic()
                    try:
                        run_native(binaries / 'aermod', directory, task['guards']['native_seconds'],
                                   {'resources': {'native_launches': 0}})
                    finally:
                        result.native_seconds[pollutant] = time.monotonic() - start
                    output = directory / 'aermod.out'
                    require(output.is_file(), 'E_OUTPUT', f'{pollutant}: aermod.out missing')
                    text = output.read_text(errors='replace')
                    require('AERMOD Finishes Successfully' in text, 'E_AERMOD_FATAL',
                            f'{pollutant}: missing successful native termination; ' + text[-700:])
                    result.receptor_values.update(statistics(directory / 'pf_stats.txt', pollutant, task))
                for name, rows in result.receptor_values.items():
                    binding = min(rows, key=lambda row: Decimal(row['margin']))
                    receptor = binding['receptor']
                    x, y = scenario.receptors[receptor - 1]
                    result.design_values[name] = DesignValue(
                        value=Decimal(binding['modeled']), background=Decimal(binding['background']),
                        limit=Decimal(binding['limit']), margin=Decimal(binding['margin']),
                        receptor_index=receptor, x=x, y=y,
                    )
                admit_air_quality(result.receptor_values)
                result.feasible = True
        except Reject as exc:
            result.errors.append({'code': exc.code, 'detail': exc.detail})
        except MemoryError as exc:
            result.errors.append({'code': 'E_RESOURCE', 'detail': str(exc)})
        except Exception as exc:
            result.errors.append({'code': 'E_OUTPUT',
                                  'detail': f'{type(exc).__name__}: {str(exc)[:400]}'})
    return result
