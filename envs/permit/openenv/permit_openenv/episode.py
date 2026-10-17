"""Bounded live episodes and the shared replay record."""
from collections import OrderedDict
import copy
from datetime import datetime, timezone
import threading
import time
from typing import Any

from .geometry import geometry
from .prompts import PROMPT_VERSION
from .protocol import json_value
from .tasks import Task

RECORD_ARRAYS = ('receptor_values', 'receptor_margins')
MAX_STEPS = 256
MAX_DIAGNOSTICS = 256
MAX_DIAGNOSTIC_CHARS = 8192
PUBLIC_KEYS = {'valid', 'errors', 'error', 'feasible', 'cost', 'standards', 'backend', 'note',
               'evaluations_left', 'reward', 'prescore_usd_saved', 'pct_saved'}


def public_result(result: dict) -> dict:
    # Scenario is constructed by Task.view; backend result dictionaries are allowlisted.
    if 'draft_design' in result:
        return result
    return {k: v for k, v in result.items() if k in PUBLIC_KEYS}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Episode:
    def __init__(self, task: Task, episode_id: str):
        self.task, self.episode_id = task, episode_id
        self.started_at = utc_now()
        self.started = time.monotonic()
        self.finished_at: str | None = None
        self.steps: list[dict] = []
        self.steps_dropped = 0
        self.diagnostics: list[dict] = []
        self.diagnostics_dropped = 0
        self.evaluations_used = 0
        self.done = False
        self.ended = 'no_submit'
        self.final: dict | None = None

    def append(self, tool: str, args: dict, result: dict, assistant_text: str = '') -> None:
        from .protocol import design_size_ok
        if 'design' in args and not design_size_ok(args['design']):
            args = {'rejected': 'design exceeds input limits'}
        self.steps.append(copy.deepcopy(json_value({
            'i': self.steps_dropped + len(self.steps), 'tool': tool, 'args': args, 'result': public_result(result),
            **{key: result[key] for key in RECORD_ARRAYS if key in result},
            't_s': round(time.monotonic()-self.started, 6),
            **({'assistant_text': assistant_text[:2000]} if assistant_text else {}),
        })))
        if len(self.steps) > MAX_STEPS:
            # Preserve all evaluations/submission; evict free checks and repeated terminal calls first.
            index = next((i for i, s in enumerate(self.steps) if s['tool'] not in ('evaluate_design', 'submit_design')
                          or 'error' in s['result'] or not s['result'].get('valid', True)), 0)
            self.steps.pop(index)
            self.steps_dropped += 1

    def log(self, message: str) -> None:
        self.diagnostics.append({'i': self.diagnostics_dropped + len(self.diagnostics), 'at': utc_now(),
                                 'message': message[:MAX_DIAGNOSTIC_CHARS]})
        if len(self.diagnostics) > MAX_DIAGNOSTICS:
            self.diagnostics.pop(0)
            self.diagnostics_dropped += 1

    def finish(self, ended: str = 'no_submit') -> None:
        if self.finished_at is None:
            self.done, self.ended, self.finished_at = True, ended, utc_now()
            self.log('episode finished: ' + ended)

    def record(self, model: str = 'manual', usage: dict | None = None) -> dict[str, Any]:
        return copy.deepcopy(json_value({
            'schema_version': 1, 'task_id': self.task.task_id, 'split': self.task.split,
            'station': self.task.scenario.station, 'model': model, 'prompt_version': PROMPT_VERSION,
            'started_at': self.started_at, 'finished_at': self.finished_at,
            'scenario': self.task.view(), 'geometry': geometry(self.task), 'steps': self.steps,
            'evaluations_used': self.evaluations_used, 'ended': self.ended, 'final': self.final,
            'reward': self.final['reward'] if self.final else 0.0,
            'steps_dropped': self.steps_dropped, 'diagnostics': self.diagnostics,
            'diagnostics_dropped': self.diagnostics_dropped,
            'usage': usage or {'input_tokens': 0, 'output_tokens': 0, 'usd': None},
        }))

    def summary(self) -> dict:
        return {'episode_id': self.episode_id, 'task_id': self.task.task_id,
                'split': self.task.split, 'station': self.task.scenario.station,
                'started_at': self.started_at, 'done': self.done, 'ended': self.ended,
                'finished_at': self.finished_at,
                'evaluations_used': self.evaluations_used, 'final': copy.deepcopy(self.final)}


class EpisodeStore:
    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self._episodes: OrderedDict[str, Episode] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, episode: Episode) -> None:
        with self._lock:
            self._episodes[episode.episode_id] = episode
            while len(self._episodes) > self.capacity:
                self._episodes.popitem(last=False)

    def get(self, episode_id: str) -> Episode | None:
        with self._lock:
            return self._episodes.get(episode_id)

    def recent(self) -> list[Episode]:
        with self._lock:
            return list(reversed(self._episodes.values()))


STORE = EpisodeStore()
