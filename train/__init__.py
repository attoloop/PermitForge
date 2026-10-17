"""Local training entry points; domain logic remains in permit_core."""

from pathlib import Path
import sys

# Reuse lightweight task/prompt modules without installing the HTTP server stack.
_root = Path(__file__).resolve().parents[1]
for _package in ('core', 'openenv'):
    sys.path.insert(0, str(_root / 'envs' / 'permit' / _package))
