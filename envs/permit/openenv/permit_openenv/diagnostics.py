"""Route native diagnostics to the active private episode, including worker threads."""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator

_sink: ContextVar[Callable[[str], None]] = ContextVar('permit_diagnostics', default=lambda message: None)


def log(message: str) -> None:
    _sink.get()(message)


@contextmanager
def diagnostic_sink(sink: Callable[[str], None]) -> Iterator[None]:
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)
