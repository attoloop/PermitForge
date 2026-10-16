"""Discrete stack heights from PLAN A.6: six interior seventh-points."""

from decimal import Decimal, InvalidOperation, localcontext, ROUND_HALF_EVEN


def height_menu(h_min: object, gep: object) -> list[Decimal]:
    try:
        if isinstance(h_min, bool) or isinstance(gep, bool):
            raise ValueError('boolean height')
        low, high = Decimal(str(h_min)), Decimal(str(gep))
        if not low.is_finite() or not high.is_finite() or low > high:
            raise ValueError('height bounds must be finite and ordered')
    except InvalidOperation as exc:
        raise ValueError('invalid height bounds') from exc
    with localcontext() as context:
        context.prec = 28
        context.rounding = ROUND_HALF_EVEN
        interior = [((low + (high-low)*i/7)*2).quantize(Decimal(1))/2 for i in range(1, 7)]
    return sorted({low, high, *(h for h in interior if low < h < high)})
