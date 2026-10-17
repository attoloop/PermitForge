"""In-process fast-evaluator GRPO: python -m train.grpo --help.

Consumes existing fields read-only. Does not launch native models or precompute.
"""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from permit_core.fields import FieldPack, height_key, load_pack as load_fields, met_calendar
from permit_openenv.tasks import Task, TaskPack, load_pack

from .permit_env import PermitTrainingEnv, resolve_anchors, scenario_text, tool_schemas, training_instructions
from .runtime import add_arguments, load_model, prompt_ids, run_training, validate_arguments


class FieldStore:
    """Read-only, shared memory maps; no native work or cache eviction during training."""

    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, FieldPack] = {}
        self._lock = threading.Lock()

    def __call__(self, task: Task) -> FieldPack:
        with self._lock:
            return self._load(task.task_id, task)

    def _load(self, task_id: str, task: Task) -> FieldPack:
        # Task is mutable/unhashable, so memoize by id with a small explicit cache.
        if task_id in self._cache:
            return self._cache[task_id]
        archive = self.root/f'{task_id}.npz'
        if archive.is_file():
            pack = load_fields(archive)
        else:
            scenario = task.scenario
            dates, valid = met_calendar(scenario)
            arrays = {}
            for sid, heights in scenario.height_menu.items():
                for height in heights:
                    path = self.root/task_id/sid/f'{height_key(height)}.npy'
                    metadata = json.loads(path.with_suffix('.json').read_text())
                    with path.open('rb') as handle:
                        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
                    if (metadata.get('task_id') != task_id or metadata.get('source_id') != sid
                            or str(metadata.get('height')) != height_key(height) or metadata['sha256'] != digest):
                        raise ValueError('Field identity/hash mismatch')
                    values = np.load(path, mmap_mode='r', allow_pickle=False)
                    if values.dtype != np.float64 or values.shape != (len(valid), len(scenario.receptors)):
                        raise ValueError('Training requires the validated float64 fields and complete receptor shape')
                    arrays[sid, height_key(height)] = values
            offsets = (0, *(np.flatnonzero(np.diff(dates//1000000))+1).tolist(), len(dates))
            pack = FieldPack(task_id, arrays, valid, offsets, np.arange(len(scenario.receptors)))
        if pack.task_id != task_id:
            raise ValueError('Field task identity mismatch')
        if any(a.dtype != np.float64 for a in pack.fields.values()):
            raise ValueError('Training requires float64 fields')
        if len(self._cache) >= 2:
            self._cache.pop(next(iter(self._cache)))
        self._cache[task_id] = pack
        return pack


def select_tasks(pack: TaskPack, task_ids: list[str] | None, limit: int | None) -> list[Task]:
    tasks = [pack.get(tid) for tid in task_ids] if task_ids else [t for t in pack.tasks if t.split == 'train']
    if any(t.split != 'train' or t.record.get('eligible') is not True for t in tasks):
        raise ValueError('Select only eligible train tasks')
    if len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError('Duplicate task selection')
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError('No training tasks selected')
    return tasks


def config_values(args: argparse.Namespace) -> dict:
    return dict(
        output_dir=str(args.output), max_steps=2 if args.smoke else args.steps,
        num_generations=args.num_generations, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum, max_tool_calling_iterations=args.max_turns,
        max_completion_length=args.max_completion_length, scale_rewards='group', beta=0,
        num_iterations=1, mask_truncated_completions=True,
        chat_template_kwargs={'enable_thinking': False}, use_vllm=False,
        learning_rate=args.learning_rate, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, optim='paged_adamw_8bit',
        logging_steps=1, save_strategy='no' if args.smoke else 'steps', save_steps=50,
        report_to='none', push_to_hub=False, seed=args.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, 'grpo')
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--fields-dir', type=Path, required=True)
    parser.add_argument('--diagnostics-dir', type=Path, help='Recorded per-task baseline/search JSON; smoke only')
    args = parser.parse_args()
    validate_arguments(args)
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    pack = load_pack(args.tasks_dir)
    tasks = select_tasks(pack, args.tasks, 2 if args.smoke else args.limit)
    if args.smoke and len(tasks) != 2:
        parser.error('Smoke requires two train tasks')
    anchors = {t.task_id: resolve_anchors(t, smoke=args.smoke, diagnostics_dir=args.diagnostics_dir) for t in tasks}
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [{'prompt': [{'role': 'system', 'content': training_instructions(args.max_turns)},
                        {'role': 'user', 'content': 'Optimize the plant and submit your design.\n'}],
             'split': 'train', 'task_id': t.task_id} for t in tasks]
    (args.output/'selection.json').write_text(json.dumps({'tasks': list(anchors), 'anchors': anchors,
                                                        'smoke': args.smoke}, indent=2)+'\n')
    (args.output/'config.json').write_text(json.dumps(config_values(args), indent=2)+'\n')
    if args.prepare_only:
        return
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    model, tokenizer = load_model(args)
    for task, row in zip(tasks, rows):
        prompt = [*row['prompt'][:-1], {'role': 'user', 'content': row['prompt'][-1]['content'] + scenario_text(task)}]
        if len(prompt_ids(tokenizer, prompt, tool_schemas())) > args.max_prompt_length:
            raise ValueError(f'{task.task_id}: prompt exceeds --max-prompt-length; increase explicitly')
    fields = FieldStore(args.fields_dir)
    # Warm only the smoke tasks; full training loads task-major groups lazily.
    if args.smoke:
        for task in tasks:
            fields(task)
    records = []
    with (args.output/'episodes.jsonl').open('w') as trace:
        def capture(record: dict) -> None:
            records.append(record)
            trace.write(json.dumps(record, allow_nan=False)+'\n')
            trace.flush()
        trainer = GRPOTrainer(model=model, processing_class=tokenizer,
                              args=GRPOConfig(**config_values(args)), train_dataset=Dataset.from_list(rows),
                              environment_factory=partial(PermitTrainingEnv, pack=pack, fields_loader=fields,
                                                          smoke=args.smoke, diagnostics_dir=args.diagnostics_dir,
                                                          trace_sink=capture))
        run_training(trainer, tokenizer, args, records_per_step=records)


if __name__ == '__main__':
    main()
