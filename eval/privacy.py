"""Redact credentials at every rollout-file write; never log provider headers."""
from __future__ import annotations

import os
import re
from typing import Any

_SECRET_KEY = re.compile(r'(?i)(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|password|secret|cookie|credential)')
_ENV_KEY = re.compile(r'(?i)(token|secret|password|credential|api[_-]?key|authorization)')
_ASSIGNMENT = re.compile(r'''(?ix)(\b(?:authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|password|secret|cookie)\b["']?\s*[:=]\s*["']?)(?:Bearer\s+)?[^\s"'&,;}]+''')
_BEARER = re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+')
_TOKEN = re.compile(r'\b(?:sk-[A-Za-z0-9_-]{12,}|hf_[A-Za-z0-9]{12,}|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)')


def redact_text(text: str) -> str:
    for name, secret in os.environ.items():
        if _ENV_KEY.search(name) and len(secret) >= 4:
            text = text.replace(secret, '[REDACTED]')
    text = _ASSIGNMENT.sub(r'\1[REDACTED]', text)
    return _TOKEN.sub('[REDACTED]', _BEARER.sub('Bearer [REDACTED]', text))


_HOME = re.compile(r'/home/[^/\s"\'\\]+')
_SCRATCH = re.compile(r'/tmp/claude-[0-9]+/[^/\s"\'\\]+/[^/\s"\'\\]+/scratchpad')


def scrub_paths(text: str, hostname: str | None = None) -> str:
    """Local paths and the host name say nothing about an episode; drop them before publishing."""
    text = _SCRATCH.sub('/tmp/scratch', _HOME.sub('/home/user', text))
    if hostname:
        text = text.replace(f'"{hostname}"', '"host"')
    return text


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): '[REDACTED]' if _SECRET_KEY.search(str(k)) else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return redact_text(value) if isinstance(value, str) else value
