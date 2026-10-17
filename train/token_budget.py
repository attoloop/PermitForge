"""CPU tokenizer sizing: python -m train.token_budget --output train/evidence/token-budget.json.

Replays recorded ARM2 draft values as representative payloads, without physics,
model weights, or a training run. Episode lengths are constructed sizing cases,
not observed model trajectories. No environment/tool implementation is changed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from permit_openenv.tasks import Task, TaskPack, load_pack

from .permit_env import PermitTrainingEnv, TOOL_NAMES, scenario_text, training_instructions
from .runtime import MODEL_REVISIONS, load_tokenizer, prompt_ids


def select_measurement_tasks(pack: TaskPack) -> list[Task]:
    selected = []
    for sources in range(2, 7):
        matches = sorted((t for t in pack.tasks if t.split == 'train'
                          and t.record.get('eligible') is True
                          and t.record.get('n_sources') == sources), key=lambda t: t.task_id)
        if len(matches) < 2:
            raise ValueError(f'Need two eligible train tasks with {sources} sources')
        selected.extend(matches[:2])
    return selected


def episode_calls(count: int) -> list[str]:
    if count not in (8, 10, 12):
        raise ValueError('Sizing cases have 8, 10 or 12 tool calls')
    return ['get_scenario', 'check_design', *(['evaluate_design']*(count-4)),
            'check_design', 'submit_design']


def recommended_limit(lengths: list[int]) -> int:
    """Reserve 25% for assistant prose/variable numeric payloads; round to 512."""
    return math.ceil(max(lengths)*1.25/512)*512


def recorded_results(task: Task, path: Path) -> dict:
    saved = json.loads(path.read_text())
    if saved.get('feasible') is not True:
        raise ValueError(f'{path}: expected an admitted draft')
    standards = [dict(id=name, value=float(v['value']), background=float(v['background']),
                      limit=float(v['limit']), margin=float(v['margin']),
                      binding_receptor=dict(index=v['receptor_index'], x_m=float(v['x']), y_m=float(v['y'])))
                 for name, v in saved['design_values'].items()]
    view = task.view()
    checked = task.check(view['draft_design'])
    evaluated = dict(valid=True, feasible=True, cost=float(task.draft_cost), standards=standards)
    return {'get_scenario': view, 'check_design': checked, 'evaluate_design': evaluated,
            'submit_design': {**checked, **evaluated, 'reward': 0.0, 'prescore_usd_saved': 0.0,
                              'anchors': task.anchors, 'grading_failed': False, 'training_grade': 'fast'}}


def measure_task(tokenizer, task: Task, draft_path: Path, schemas: list[dict]) -> dict:
    payloads = recorded_results(task, draft_path)
    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)
    instruction = training_instructions(8)
    scenario = scenario_text(task)
    row = dict(task_id=task.task_id, sources=len(task.scenario.sources),
               instruction_tokens=len(encode(instruction)), scenario_tokens=len(encode(scenario)),
               draft_evidence=str(draft_path), draft_sha256=hashlib.sha256(draft_path.read_bytes()).hexdigest(),
               tool_result_tokens={}, episodes={})
    for name, payload in payloads.items():
        row['tool_result_tokens'][name] = len(encode(json.dumps(
            {**payload, 'evaluations_left': 29}, separators=(',', ':'))))
    for count in (8, 10, 12):
        messages = [{'role': 'system', 'content': training_instructions(count)},
                    {'role': 'user', 'content': 'Optimize the plant and submit your design.\n' + scenario}]
        prefix = prompt_ids(tokenizer, messages, schemas)
        total = len(prefix)
        used = 0
        for name in episode_calls(count):
            arguments = {} if name == 'get_scenario' else {'design': task.view()['draft_design']}
            assistant = {'role': 'assistant', 'content': '', 'tool_calls': [
                {'type': 'function', 'function': {'name': name, 'arguments': arguments}}]}
            before = len(prompt_ids(tokenizer, messages, schemas))
            messages.append(assistant)
            # Qwen's template preserves the prior generation prefix. Count the
            # assistant turn with EOS, then the tool suffix as TRL does.
            assistant_end = tokenizer.apply_chat_template(messages, tools=schemas,
                tokenize=True, add_generation_prompt=False, enable_thinking=False)
            total += len(assistant_end)-before
            used += name == 'evaluate_design'
            result = json.dumps({**payloads[name], 'evaluations_left': 30-used}, separators=(',', ':'))
            messages.append({'role': 'tool', 'name': name, 'content': result})
            after = len(prompt_ids(tokenizer, messages, schemas))
            total += after-len(assistant_end)
        # A final assistant acknowledgment after the submit result.
        total += len(encode('Submitted.')) + 1
        row['episodes'][str(count)] = dict(prompt_tokens=len(prefix),
                                           completion_tokens=total-len(prefix), total_tokens=total)
    row['prompt_tokens'] = row['episodes']['8']['prompt_tokens']
    return row


def markdown_tables(evidence: dict) -> str:
    rows = next(iter(evidence['models'].values()))['tasks']
    lines = ['| Task | Sources | Instructions | Scenario | Full prompt | get_scenario | check | evaluate | submit |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values = [row[k] for k in ('task_id', 'sources', 'instruction_tokens', 'scenario_tokens', 'prompt_tokens')]
        values.extend(row['tool_result_tokens'][name] for name in TOOL_NAMES)
        lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    lines.extend(['', 'Episode cells are **completion / total context** tokens.', '',
                  '| Task | 8 calls | 10 calls | 12 calls |', '|---|---:|---:|---:|'])
    for row in rows:
        cells = [row['task_id']]+[f"{v['completion_tokens']} / {v['total_tokens']}"
                                 for v in row['episodes'].values()]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)+'\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks-dir', type=Path, default=Path('envs/permit/tasks'))
    parser.add_argument('--drafts-dir', type=Path, default=Path('convert/cache/drafts-arm2'))
    parser.add_argument('--output', type=Path, default=Path('train/evidence/token-budget.json'))
    parser.add_argument('--markdown', type=Path, help='Optional reproducible Markdown tables')
    args = parser.parse_args()
    from transformers.utils import get_json_schema
    schemas = [get_json_schema(getattr(PermitTrainingEnv, name)) for name in sorted(TOOL_NAMES)]
    tasks = select_measurement_tasks(load_pack(args.tasks_dir))
    evidence = {'method': 'CPU tokenization of scripted draft-payload episodes; no observed policy rollout',
                'calls': 'get_scenario, two check_design, N-4 evaluate_design, submit_design',
                'headroom': 'ceil(max completion tokens * 1.25 / 512) * 512', 'models': {}}
    for model, revision in MODEL_REVISIONS.items():
        tokenizer = load_tokenizer(argparse.Namespace(model=model, revision=revision))
        rows = [measure_task(tokenizer, t, args.drafts_dir/f'{t.task_id}.json', schemas) for t in tasks]
        evidence['models'][model] = {'revision': revision, 'tasks': rows,
            'recommended_completion_limits': {str(n): recommended_limit([
                r['episodes'][str(n)]['completion_tokens'] for r in rows]) for n in (8, 10, 12)},
            'recommended_prompt_limit': math.ceil(max(r['prompt_tokens'] for r in rows)/512)*512}
    evidence['inputs_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        Path('train/permit_env.py'), Path('envs/permit/openenv/permit_openenv/prompts.py'),
        args.tasks_dir/'permit-v2-train/tasks.jsonl')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2)+'\n')
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_tables(evidence))
    for name, values in evidence['models'].items():
        print(name, 'prompt:', values['recommended_prompt_limit'],
              'completion:', values['recommended_completion_limits'])


if __name__ == '__main__':
    main()
