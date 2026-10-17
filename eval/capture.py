"""Append-only episode transcripts and raw provider artifacts for eval.run_eval."""
from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any

from .common import now, write_json, write_text
from .privacy import redact, redact_text


def append_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(redact(value), ensure_ascii=False, allow_nan=False)+'\n')


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(redact_text(text)+'\n')


class EpisodeCapture:
    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.RLock()
        self.last_diagnostic = -1
        self.turn = 0
        self.sequence = 0
        for name in ('raw', 'attempts'):
            (root/name).mkdir(parents=True, exist_ok=True)
        for name in ('transcript.jsonl', 'env.log'):
            (root/name).touch(exist_ok=True)

    def event(self, role: str, content: Any, *, kind: str = 'message', usage: dict | None = None,
              turn: int | None = None, **extra: Any) -> None:
        with self.lock:
            append_json(self.root/'transcript.jsonl', {'i': self.sequence, 'at': now(),
                        'turn': self.turn if turn is None else turn, 'role': role, 'kind': kind,
                        'content': content, 'usage': usage, **extra})
            self.sequence += 1

    def raw_json(self, name: str, value: Any) -> None:
        with self.lock:
            write_json(self.root/'raw'/name, value)

    def raw_text(self, name: str, text: str, *, append: bool = False) -> None:
        with self.lock:
            if append:
                append_log(self.root/'raw'/name, text.rstrip('\n'))
            else:
                write_text(self.root/'raw'/name, text)

    def failed(self, prefix: str, error: Any) -> None:
        """Keep each request attempt with its raw inputs/output, including exhausted failures."""
        with self.lock:
            destination = self.root/'attempts'/prefix
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination/'error.json', {'at': now(), 'error': error})
            for source in (self.root/'raw').glob(prefix+'*'):
                write_text(destination/'raw'/source.name, source.read_text())

    def environment(self, record: dict) -> None:
        with self.lock:
            for row in record.get('diagnostics', []):
                if row['i'] > self.last_diagnostic:
                    append_log(self.root/'env.log', row['at']+' '+row['message'])
                    self.last_diagnostic = row['i']

    def finish(self, record: dict) -> None:
        self.environment(record)
        write_json(self.root/'episode.json', record)
        if record.get('ended') != 'submitted' or record.get('error'):
            self.failed('terminal', record.get('error') or record.get('error_type') or record.get('ended'))
            write_json(self.root/'attempts/terminal/episode.json', record)
            for source in (self.root/'raw').iterdir():
                if source.is_file():
                    write_text(self.root/'attempts/terminal/raw'/source.name, source.read_text())
