"""Cross-process native slots shared by field preparation and interactive verification."""

from contextlib import contextmanager
from collections.abc import Iterator
import fcntl
import time


@contextmanager
def native_slot() -> Iterator[None]:
    from .fields import cache_root
    root = cache_root()/'.native-slots'
    root.mkdir(parents=True, exist_ok=True)
    while True:
        for index in range(12):
            handle = (root/f'{index}.lock').open('a+b')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            try:
                yield
            finally:
                handle.close()
            return
        time.sleep(.02)
