"""Feasible USD/year savings and the fraction of draft cost saved."""
from decimal import Decimal
from typing import TYPE_CHECKING

from .cost import cost
from .model import Scenario

if TYPE_CHECKING:
    from .verify import Verification


def prescore(verification: 'Verification | None', scenario: Scenario) -> Decimal:
    if verification is None or not verification.feasible or verification.cost is None:
        return Decimal(-1)
    return cost(scenario.draft_design, scenario.task) - verification.cost


def reward(prescore: Decimal | float | int | None, draft_cost: Decimal | float | int) -> float:
    """Gate failures use prescore -1; only a feasible graded submission earns savings."""
    if isinstance(draft_cost, bool) or not isinstance(draft_cost, (Decimal, float, int)):
        raise ValueError('draft cost must be finite and positive')
    draft = Decimal(str(draft_cost))
    if not draft.is_finite() or draft <= 0:
        raise ValueError('draft cost must be finite and positive')
    if isinstance(prescore, bool) or not isinstance(prescore, (Decimal, float, int)):
        return 0.0
    saved = Decimal(str(prescore))
    if not saved.is_finite() or saved <= 0:
        return 0.0
    return float(min(Decimal(1), saved / draft))
