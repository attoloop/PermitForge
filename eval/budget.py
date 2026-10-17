"""Durable, request-level worst-case reservations for the Anthropic hard cap."""
from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import threading

from .common import finite, now, write_json


class BudgetExhausted(RuntimeError):
    pass


def request_bound(model: str, max_tokens: int) -> float:
    from .anthropic_driver import prices
    if model not in ('claude-opus-5-5', 'claude-sonnet-5-5'):
        raise ValueError('no verified context ceiling for '+model)
    p = prices()[model]
    # Both models have a 1M context window. No cache/fast/region options are
    # requested. Reserve a full context plus all output, not a token estimate.
    return float(Decimal(2**20)*Decimal(str(p['input']))/1_000_000 +
                 Decimal(max_tokens)*Decimal(str(p['output']))/1_000_000)


class HardBudget:
    def __init__(self, path: Path, limit: float):
        if not finite(limit) or not 0 <= limit <= 190:
            raise ValueError('hard budget must be in [0, 190] USD')
        self.path, self.limit = path, Decimal(str(limit))
        self.lock = threading.Lock()
        self.entries = json.loads(path.read_text())['requests'] if path.exists() else {}
        if path.exists() and Decimal(str(json.loads(path.read_text())['limit_usd'])) != self.limit:
            raise ValueError('hard budget limit changed on resume')

    @property
    def committed(self) -> float:
        return float(sum((Decimal(str(r['charged_usd'])) for r in self.entries.values()), Decimal(0)))

    def save(self) -> None:
        write_json(self.path, {'limit_usd': float(self.limit), 'committed_usd': self.committed,
                               'requests': self.entries})

    def can_reserve(self, upper: float) -> bool:
        with self.lock:
            total = sum((Decimal(str(r['charged_usd'])) for r in self.entries.values()), Decimal(0))
            return total+Decimal(str(upper)) <= self.limit

    def reserve(self, key: str, upper: float) -> None:
        if not finite(upper) or upper < 0:
            raise ValueError('invalid request cost bound')
        with self.lock:
            if key in self.entries:
                raise ValueError('request key already reserved')
            total = sum((Decimal(str(r['charged_usd'])) for r in self.entries.values()), Decimal(0))
            if total+Decimal(str(upper)) > self.limit:
                raise BudgetExhausted('hard request budget exhausted')
            self.entries[key] = {'upper_usd': upper, 'charged_usd': upper,
                                 'status': 'reserved_or_unknown', 'at': now()}
            self.save()  # A crash/lost response keeps the full bound charged.

    def settle(self, key: str, actual: float) -> None:
        with self.lock:
            entry = self.entries[key]
            if not finite(actual) or not 0 <= actual <= entry['upper_usd']:
                raise ValueError('provider usage exceeds the verified request bound')
            entry.update(charged_usd=actual, status='settled')
            self.save()
