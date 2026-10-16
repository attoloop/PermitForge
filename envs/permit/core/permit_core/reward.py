# Origin: club-mix/clubmix/calibration.py
"""Open-ended reward calibration, with exact draft savings."""
from dataclasses import dataclass
from decimal import Decimal
import math
from typing import Callable
from typing import TYPE_CHECKING

from .cost import cost
from .model import Scenario

if TYPE_CHECKING:
    from .verify import Verification

def hyperbolic_calibration(
    worst_observed: float,
    best_observed: float,
    prescore_near_saturation: float | None = None,
) -> Callable[[float], float]:
    if not (worst_observed < best_observed):
        raise ValueError(
            "Expected worst_observed < best_observed, "
            f"got {worst_observed}, {best_observed}."
        )

    linear_span = best_observed - worst_observed
    if prescore_near_saturation is None:
        prescore_near_saturation = best_observed + 3 * linear_span
    elif prescore_near_saturation <= best_observed:
        raise ValueError(
            "prescore_near_saturation must be greater than best_observed, "
            f"got {prescore_near_saturation} <= {best_observed}."
        )

    def f(x: float) -> float:
        if x <= worst_observed:
            return 0.0
        if x < best_observed:
            return 0.4 * (x - worst_observed) / linear_span
        return 0.4 + 1.2 * (x - best_observed) / (
            prescore_near_saturation - best_observed + 2 * (x - best_observed)
        )

    return f


@dataclass(frozen=True)
class Anchors:
    worst: Decimal
    best: Decimal
    saturation: Decimal | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        for name in ('worst', 'best', 'saturation'):
            value = getattr(self, name)
            if name == 'saturation' and value is None:
                value = self.best + 3 * (self.best - self.worst)
            if isinstance(value, bool):
                raise ValueError(f'{name} must be a finite number')
            value = Decimal(str(value))
            if not value.is_finite():
                raise ValueError(f'{name} must be finite')
            object.__setattr__(self, name, value)
        if not self.worst == 0 < self.best < self.saturation:
            raise ValueError('Expected worst = 0 < best < saturation')

    def to_dict(self) -> dict[str, str | None]:
        return {'worst': str(self.worst), 'best': str(self.best),
                'saturation': str(self.saturation), 'source': self.source}

    @classmethod
    def from_dict(cls, value: dict) -> 'Anchors':
        return cls(**{k: value[k] for k in ('worst', 'best', 'saturation', 'source') if k in value})


def prescore(verification: 'Verification | None', scenario: Scenario) -> Decimal:
    if verification is None or not verification.feasible or verification.cost is None:
        return Decimal(-1)
    return cost(scenario.draft_design, scenario.task) - verification.cost


def reward(prescore: Decimal | float | int | None, anchors: Anchors | dict) -> float:
    if isinstance(anchors, dict):
        anchors = Anchors.from_dict(anchors)
    f = hyperbolic_calibration(float(anchors.worst), float(anchors.best), float(anchors.saturation))
    if isinstance(prescore, bool) or not isinstance(prescore, (Decimal, float, int)):
        return 0.0
    if isinstance(prescore, Decimal) and not prescore.is_finite():
        return 0.0
    value = float(prescore)
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, f(value)))

