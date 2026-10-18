"""PermitForge's four MCP tools and one terminal native grade per episode."""
from __future__ import annotations

import asyncio
import copy
import json
import random
import threading
import time
import traceback
import uuid
from typing import Annotated, Any

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.env_server.types import Action, EnvironmentMetadata, Observation, State
from pydantic import Field, WithJsonSchema
from permit_core.cost import cost
from permit_core.reward import prescore, reward
from permit_core.verify import Verification

from .episode import Episode, STORE, public_result
from .evaluator import Evaluator, VerifyEvaluator, select_evaluator
from .diagnostics import diagnostic_sink
from .prompts import PROMPT_VERSION, instructions
from .protocol import DESIGN_SCHEMA, json_value
from .rubric import PermitRubric
from .tasks import TaskPack, load_pack, to_ledger

EVALUATE_BUDGET = 30
TOOLS = ['get_scenario', 'check_design', 'evaluate_design', 'submit_design']
# Advertise the strict contract while retaining malformed values for free, structured errors.
DesignArg = Annotated[Any, WithJsonSchema(DESIGN_SCHEMA)]


class PermitState(State):
    task_id: str | None = None
    split: str | None = None
    evaluations_used: int = 0
    best_feasible_cost: float | None = None
    history: list[dict] = Field(default_factory=list)
    done: bool = False
    reward: float = 0.0
    grade: dict | None = None


class PermitForgeEnvironment(MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, pack: TaskPack | None = None, *, evaluator: Evaluator | None = None,
                 verifier: Evaluator | None = None, evaluate_budget: int = EVALUATE_BUDGET):
        self.instructions = instructions(evaluate_budget)
        self.evaluate_budget = evaluate_budget
        self.pack = pack if pack is not None else load_pack()
        self.evaluator = evaluator if evaluator is not None else select_evaluator()
        # Injection supports deterministic tests; production always uses VerifyEvaluator.
        self.verifier = verifier if verifier is not None else VerifyEvaluator()
        self._episode: Episode | None = None
        self._state = PermitState()
        self._lock = threading.RLock()
        mcp = FastMCP('permit_forge')

        @mcp.tool
        async def get_scenario() -> str:
            """Read sources, controls, height menus, standards, draft, and evaluations left."""
            return await asyncio.to_thread(self._dispatch, 'get_scenario', {})

        @mcp.tool
        async def check_design(design: DesignArg) -> str:
            """Check validity and exact annualized cost, for free; does not test air quality."""
            return await asyncio.to_thread(self._dispatch, 'check_design', {'design': design})

        @mcp.tool
        async def evaluate_design(design: DesignArg) -> str:
            """Test feasibility and binding standards; spends one evaluation only for a valid design."""
            return await asyncio.to_thread(self._dispatch, 'evaluate_design', {'design': design})

        @mcp.tool
        async def submit_design(design: DesignArg) -> str:
            """End the episode and independently verify the final design; grading may take minutes."""
            return await asyncio.to_thread(self._dispatch, 'submit_design', {'design': design})

        super().__init__(mcp)
        self.rubric = PermitRubric()

    @property
    def episode(self) -> Episode:
        if self._episode is None:
            raise ValueError('call reset before using tools')
        return self._episode

    @property
    def state(self) -> PermitState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(name='PermitForge v2', version='2.0.0', author='Attoloop',
                                   description='Minimize annualized emissions-control and stack costs while meeting every air-quality standard.')

    def list_splits(self) -> list[dict]:
        return [{'name': s, 'type': s, 'num_tasks': self.pack.count(s)} for s in self.pack.splits()]

    def num_tasks(self, split: str) -> int:
        return self.pack.count(split)

    def get_task(self, split: str, index: int) -> dict:
        return self.pack.public(self.pack.at(split, int(index)))

    def get_task_range(self, split: str, start: int | None = None, stop: int | None = None) -> list[dict]:
        n = self.pack.count(split)
        return [self.get_task(split, i) for i in range(max(0, start or 0), min(n, n if stop is None else stop))]

    def list_tasks(self, split: str) -> list[dict]:
        return self.get_task_range(split)

    def reset(self, seed: int | None = None, episode_id: str | None = None, split: str | None = None,
              index: int | None = None, task_id: str | None = None, **kwargs: Any) -> Observation:
        with self._lock:
            if task_id is not None:
                task = self.pack.get(task_id)
            else:
                split = split or ('train' if 'train' in self.pack.splits() else self.pack.splits()[0])
                index = random.Random(seed).randrange(self.pack.count(split)) if index is None else int(index)
                task = self.pack.at(split, index)
            if self._episode and not self._episode.done:
                self._episode.finish()
            self._episode = Episode(task, str(episode_id)[:64] if episode_id else uuid.uuid4().hex,
                                    evaluate_budget=self.evaluate_budget)
            STORE.add(self._episode)
            self._state = PermitState(episode_id=self.episode.episode_id, task_id=task.task_id, split=task.split)
            self._reset_rubric()
            self.episode.log('episode reset')
            return Observation(done=False, reward=0.0, metadata={
                'episode_id': self.episode.episode_id, 'task_id': task.task_id, 'split': task.split,
                'instructions': self.instructions, 'prompt_version': PROMPT_VERSION, 'tools': TOOLS,
                'scenario': task.view(self.evaluate_budget), 'evaluate_budget': self.evaluate_budget,
            })

    def _dispatch(self, name: str, args: dict) -> str:
        with self._lock:
            ep = self.episode
            ep.log('tool received: ' + name)
            left = self.evaluate_budget - self._state.evaluations_used
            if ep.done:
                result = {'error': 'episode is over; reset to start another', 'evaluations_left': left}
            elif name == 'get_scenario':
                result = ep.task.view(left)
            else:
                design = args.get('design')
                checked = ep.task.check(design)
                checked['evaluations_left'] = left
                if name == 'submit_design':
                    # Set terminal status before native work, so no retry can regrade.
                    ep.done = True
                    result = self._submit(design, checked)
                    ep.final = copy.deepcopy(public_result(result))
                    ep.finish('submitted')
                    self.rubric.pending_grade = ep.final
                elif not checked['valid'] or name == 'check_design':
                    result = checked
                elif left == 0:
                    result = {'error': 'no evaluations left; submit_design when ready', 'evaluations_left': 0}
                else:
                    self._state.evaluations_used += 1
                    ep.evaluations_used = self._state.evaluations_used
                    try:
                        result = self._evaluate(self.evaluator, design)
                    except Exception:
                        ep.log(traceback.format_exc())
                        result = {'valid': True, 'feasible': False, 'cost': checked['cost'], 'standards': [],
                                  'errors': ['E_INFRA: evaluation unavailable'], 'error': 'evaluation failed'}
                    result['evaluations_left'] = left-1
                    self._state.history.append(copy.deepcopy(json_value({'design': design, 'evaluation': public_result(result)})))
                    if result.get('feasible') and result.get('cost') is not None:
                        previous = self._state.best_feasible_cost
                        self._state.best_feasible_cost = min(previous, result['cost']) if previous is not None else result['cost']
            ep.append(name, args, result)
            self._state.done = ep.done
            self._state.grade = ep.final
            self._state.reward = ep.final['reward'] if ep.final else 0.0
            return json.dumps(json_value(public_result(result)), allow_nan=False)

    def _submit(self, design: Any, checked: dict) -> dict:
        task = self.episode.task
        result = {**checked, 'feasible': False, 'cost': checked.get('cost'), 'standards': []}
        if checked['valid']:
            try:
                result.update(self._evaluate(self.verifier, design))
            except Exception:
                self.episode.log(traceback.format_exc())
                result['errors'] = ['E_INFRA: verification unavailable']
        exact_cost = cost(to_ledger(design), task.scenario.task) if checked['valid'] else None
        verified = Verification(feasible=bool(result.get('feasible')), cost=exact_cost)
        saved = prescore(verified, task.scenario)
        score = reward(saved, task.draft_cost)
        result.update(reward=score, prescore_usd_saved=float(saved),
                      pct_saved=float(100 * saved / task.draft_cost) if verified.feasible else 0.0,
                      evaluations_left=self.evaluate_budget-self._state.evaluations_used)
        return result

    def _evaluate(self, evaluator: Evaluator, design: dict) -> dict:
        started = time.monotonic()
        with diagnostic_sink(self.episode.log):
            result = evaluator.evaluate(self.episode.task, design)
        self.episode.log(f'{type(evaluator).__name__} completed in {time.monotonic()-started:.6f}s; '
                         f'feasible={result.get("feasible")}; errors={result.get("errors", [])}')
        # Backend errors may contain native paths or diagnostics. Expose only known codes.
        messages = {'E_NAAQS': 'air-quality standard exceeded', 'E_GEP': 'stack exceeds GEP',
                    'E_MENU': 'invalid design', 'E_INFRA': 'model computation unavailable'}
        codes = list(dict.fromkeys(str(e).split(':', 1)[0] for e in result.get('errors', [])))[:4]
        result['errors'] = [f'{c}: {messages[c]}' if c in messages else 'E_INFRA: model computation unavailable'
                            for c in codes]
        return result

    def close(self) -> None:
        with self._lock:
            if self._episode is not None and self._episode.finished_at is None:
                self._episode.finish()
                self._state.done = True
                self._state.reward = 0.0
        super().close()

    def _after_step(self, action: Action, obs: Observation) -> Observation:
        if getattr(obs, 'error', None) is not None:
            if self._episode:
                self.episode.log(str(obs.error))
            obs.error = obs.error.model_copy(update={'message': 'tool request failed; check its name and arguments'})
            obs.result = None
        if isinstance(action, CallToolAction) and self._episode:
            self._state.step_count += 1
            obs.reward = self._apply_rubric(action, obs)
            obs.done = self.episode.done
            obs.metadata = {**(obs.metadata or {}), 'episode_id': self.episode.episode_id,
                            'grade': self.episode.final,
                            'rubric': {name: r.last_score for name, r in self.rubric.named_rubrics()}}
        else:
            obs.reward = 0.0
        return obs

    def step(self, action: Action, timeout_s: float | None = None, **kwargs: Any) -> Observation:
        return self._after_step(action, super().step(action, timeout_s=timeout_s or 1800, **kwargs))

    async def step_async(self, action: Action, timeout_s: float | None = None, **kwargs: Any) -> Observation:
        return self._after_step(action, await super().step_async(action, timeout_s=timeout_s or 1800, **kwargs))

    def _step_impl(self, action: Action, timeout_s: float | None = None, **kwargs: Any) -> Observation:
        return Observation(done=False, reward=0.0, metadata={'error': 'use the four MCP tools'})
