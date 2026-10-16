"""Load preserved v1 task data without modifying the extraction."""

from dataclasses import dataclass, field
from decimal import Decimal
import json
import os
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class Source:
    id: str
    type: str
    location: dict[str, Decimal]
    fixed_stack: dict[str, Decimal]
    uncontrolled_rates_g_s: dict[str, Decimal]
    no2_nox_in_stack_ratio: Decimal
    height_min_m: Decimal
    height_max_m: Decimal
    gep_height_m: Decimal


@dataclass(frozen=True)
class ControlRow:
    id: str
    applies_to: tuple[str, ...]
    annualized_usd: Decimal
    efficiency_by_pollutant: dict[str, Decimal]


@dataclass(frozen=True)
class Standard:
    id: str
    pollutant: str
    averaging_time: str
    form: str
    limit: Decimal
    background: Decimal
    units: str


@dataclass(frozen=True)
class Scenario:
    id: str
    station: str
    task_root: Path
    verifier_root: Path
    met_path: Path
    task: dict[str, Any]
    sources: tuple[Source, ...]
    controls: tuple[ControlRow, ...]
    standards: tuple[Standard, ...]
    buildings: dict[str, Any]
    receptors: tuple[tuple[Decimal, Decimal], ...]
    draft_design: dict[str, dict[str, Any]]
    height_menu: dict[str, tuple[Decimal, ...]] | None = field(default=None)


def extraction_root() -> Path:
    configured = os.environ.get("PERMIT_V1_DIR")
    if configured:
        return Path(configured).resolve()
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "convert/v1-src"
        if (candidate / "EXTRACT_MANIFEST.json").is_file():
            return candidate
    raise FileNotFoundError("Set PERMIT_V1_DIR to the extracted v1 source directory")


def load_v1_instance(id: str) -> Scenario:
    if not re.fullmatch(r"[0-9a-f]{12}-s47", id):
        raise ValueError(f"Invalid v1 instance id: {id!r}")
    root = extraction_root()
    manifest = json.loads((root / "EXTRACT_MANIFEST.json").read_text())
    station = manifest["instances"][id]["met_station"]
    instance = root / "instances" / id
    task_root = instance / "task"
    return _load_instance(id, station, task_root, instance / "verifier-task", root / "met" / station)


def load_v2_instance(root: str | Path, id: str) -> Scenario:
    """Read a converted task with exact Decimal values and shared station met."""
    if not re.fullmatch(r"[0-9a-f]{12}-s47", id):
        raise ValueError(f"Invalid v2 instance id: {id!r}")
    root = Path(root).resolve()
    task_root = root / "data" / id
    task = json.loads((task_root / "task.json").read_text(), parse_float=Decimal)
    menu = {sid: tuple(Decimal(str(h)) for h in heights)
            for sid, heights in task["height_menu"].items()}
    return _load_instance(id, task["station"], task_root, task_root,
                          root / "met" / task["station"], menu)


def _load_instance(id: str, station: str, task_root: Path, verifier_root: Path,
                   met_path: Path, height_menu: dict[str, tuple[Decimal, ...]] | None = None) -> Scenario:
    task = json.loads((task_root / "task.json").read_text(), parse_float=Decimal)
    sources = tuple(Source(
        id=u["id"], type=u["type"], location=u["location"], fixed_stack=u["fixed_stack"],
        uncontrolled_rates_g_s=u["uncontrolled_rates_g_s"],
        no2_nox_in_stack_ratio=Decimal(str(u["no2_nox_in_stack_ratio"])),
        **{k: Decimal(str(v)) for k, v in u["stack_bounds"].items()},
    ) for u in task["scenario"]["units"])
    controls = tuple(ControlRow(
        id=c["id"], applies_to=tuple(c["applies_to"]),
        annualized_usd=Decimal(str(c["annualized_usd"])),
        efficiency_by_pollutant={k: Decimal(str(v)) for k, v in c["efficiency_by_pollutant"].items()},
    ) for c in task["costs"]["controls"])
    standards = tuple(Standard(
        id=name, pollutant=s["pollutant"], averaging_time=s["averaging_time"],
        form=s["form"], units=s["units"], limit=Decimal(str(s["limit"])),
        background=Decimal(str(task["background"]["design_values"][name])),
    ) for name, s in task["scenario"]["naaqs_thresholds"].items())
    receptors = tuple((Decimal(t[2]), Decimal(t[3]))
                     for line in (task_root / "receptors.inp").read_text().splitlines()
                     if (t := line.split())[:2] == ["RE", "DISCCART"])
    for suffix in ("sfc", "pfl"):
        if not (met_path / f"{station}.{suffix}").is_file():
            raise FileNotFoundError(met_path / f"{station}.{suffix}")
    return Scenario(
        id=id, station=station, task_root=task_root, verifier_root=verifier_root,
        met_path=met_path, task=task, sources=sources, controls=controls, standards=standards,
        height_menu=height_menu,
        buildings=json.loads((task_root / "buildings.json").read_text(), parse_float=Decimal),
        receptors=receptors,
        draft_design=json.loads((task_root / "draft/design.yaml").read_text(), parse_float=Decimal),
    )
