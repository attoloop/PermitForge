"""Read-only launch checks used by python -m eval.run_eval before any episode."""
from __future__ import annotations

from importlib import import_module
import os
import shutil
import subprocess
from typing import Callable

import httpx


class PreflightError(RuntimeError):
    pass


def drivers(models: list[str]) -> dict[str, Callable]:
    loaded = {}
    for provider in sorted({model.partition(':')[0] for model in models}):
        try:
            module = import_module('eval.'+provider+'_driver')
            if provider == 'anthropic':
                import_module('anthropic')
        except Exception as exc:
            raise PreflightError(
                f'{provider} driver import failed ({type(exc).__name__}); run '
                '`uv sync --project envs/permit/openenv --extra agents --locked` and launch '
                'with `uv run --project envs/permit/openenv --extra agents --locked`') from exc
        loaded[provider] = getattr(module, 'run_'+provider, None)
        if provider == 'anthropic':
            if not os.environ.get('ANTHROPIC_API_KEY', '').strip():
                raise PreflightError('Anthropic requires ANTHROPIC_API_KEY')
            if hasattr(module, 'request_cost'):
                for model in models:
                    if model.startswith('anthropic:'):
                        module.request_cost(model.partition(':')[2], {})
        elif provider == 'codex':
            if not shutil.which('codex'):
                raise PreflightError('codex CLI is not on PATH')
            try:
                login = subprocess.run(['codex', 'login', 'status'],
                                       capture_output=True, text=True, timeout=10)
            except (OSError, subprocess.SubprocessError) as exc:
                raise PreflightError('codex CLI login status check failed') from exc
            if login.returncode or 'ChatGPT' not in login.stdout+login.stderr:
                raise PreflightError('Codex requires ChatGPT subscription login')
        else:
            raise PreflightError('supported drivers: anthropic:<model>, codex:<model>')
    return loaded


def server(url: str, token: str, tasks: list[dict]) -> None:
    """Check health, authenticated recording and task selection without resetting a session."""
    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            response = client.get(url.rstrip('/')+'/healthz')
            response.raise_for_status()
            if response.json().get('status') != 'ok':
                raise PreflightError('server is not healthy')
            response = client.get(url.rstrip('/')+'/eval/preflight',
                                  headers={'Authorization': 'Bearer '+token})
            response.raise_for_status()
            data = response.json()
            if data.get('status') != 'ok' or any(task not in data['tasks'] for task in tasks):
                raise PreflightError('requested tasks do not exist in the recording server cohort')
    except PreflightError:
        raise
    except Exception as exc:
        raise PreflightError('server health/recording check failed; check the URL, PERMIT_EVAL_TOKEN '
                             'and eval.transport server version') from exc
