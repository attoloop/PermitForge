"""Train-only SFT from episode JSON/JSONL: python -m train.sft --help."""

import argparse
import copy
import json
import math
from pathlib import Path

from permit_core.cost import cost
from permit_openenv.tasks import TaskPack, load_pack, to_ledger

from .permit_env import TOOL_NAMES, tool_schemas, training_instructions


def episode_to_chat(record: dict, *, pack: TaskPack | None = None,
                    smoke: bool = False) -> dict | None:
    """Keep submitted, feasible improvements with a verified train task identity."""
    if record.get('split') != 'train' or record.get('ended') != 'submitted' or record.get('grading_failed'):
        return None
    synthetic = record.get('synthetic', False)
    if synthetic and not smoke:
        return None
    final = record.get('final') or {}
    saved = final.get('prescore_usd_saved')
    if (final.get('feasible') is not True or isinstance(saved, bool)
            or not isinstance(saved, (int, float)) or not math.isfinite(saved) or saved <= 0):
        return None
    steps = record.get('steps', [])
    if (not steps or steps[-1].get('tool') != 'submit_design'
            or sum(s.get('tool') == 'submit_design' for s in steps) != 1):
        return None
    if synthetic:
        scenario = record['scenario']
    else:
        if pack is None:
            raise ValueError('A task pack is required to enforce train-only SFT')
        try:
            task = pack.get(record['task_id'])
        except KeyError:
            return None
        if task.split != 'train' or task.record.get('eligible') is not True:
            return None
        design = steps[-1].get('args', {}).get('design')
        if not task.check(design)['valid']:
            return None
        exact_saved = task.draft_cost - cost(to_ledger(design), task.scenario.task)
        if exact_saved <= 0 or not math.isclose(float(exact_saved), saved, abs_tol=1e-6, rel_tol=1e-9):
            return None
        scenario = task.view()
    messages = [{'role': 'system', 'content': training_instructions()},
                {'role': 'user', 'content': 'Optimize this plant.\n' + json.dumps(scenario, separators=(',', ':'))}]
    for i, step in enumerate(steps):
        name, args = step.get('tool'), step.get('args', {})
        if name not in TOOL_NAMES or not isinstance(args, dict):
            return None
        call_id = f'call_{i}'
        messages.append({'role': 'assistant', 'content': step.get('assistant_text', ''), 'tool_calls': [
            {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': copy.deepcopy(args)}}]})
        result = ({'submitted': True, 'message': 'Episode is over. End your response.'}
                  if name == 'submit_design' else {k: v for k, v in step.get('result', {}).items()
                  if k not in ('anchors', 'best_known_search_cost', 'receptor_values', 'receptor_margins', 'note')})
        messages.append({'role': 'tool', 'name': name, 'tool_call_id': call_id,
                         'content': json.dumps(result, separators=(',', ':'), allow_nan=False)})
    messages.append({'role': 'assistant', 'content': 'Submitted.'})
    return {'messages': messages, 'tools': tool_schemas()}


def synthetic_records() -> list[dict]:
    """Four invented format exercises, explicitly excluded from regular SFT."""
    records = []
    for i, height in enumerate((10, 12, 14, 16)):
        design = {'S1': {'control': 'S1_moderate', 'height_m': height}}
        scenario = {'task_id': f'synthetic-format-{i}', 'synthetic': True,
                    'description': 'Invented tool-format exercise; no regulatory feasibility claim.',
                    'sources': [{'id': 'S1', 'height_menu_m': [10, 12, 14, 16, 20]}],
                    'controls': [{'id': 'S1_moderate', 'annualized_usd': 6}],
                    'draft_cost': 22, 'stack_usd_per_m': 1}
        result = {'valid': True, 'feasible': True, 'cost': height - 4, 'standards': []}
        records.append({'task_id': scenario['task_id'], 'split': 'train', 'synthetic': True,
                        'scenario': scenario, 'ended': 'submitted',
                        'final': {**result, 'prescore_usd_saved': 26-height},
                        'steps': [
                            {'tool': 'check_design', 'args': {'design': design}, 'result': {'valid': True, 'cost': height-4}},
                            {'tool': 'evaluate_design', 'args': {'design': design}, 'result': result},
                            {'tool': 'submit_design', 'args': {'design': design}, 'result': result}]})
    return records


def read_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        files = sorted(path.rglob('*.json')) if path.is_dir() else [path]
        for file in files:
            if file.suffix == '.jsonl':
                records.extend(json.loads(line) for line in file.read_text().splitlines() if line.strip())
            else:
                records.append(json.loads(file.read_text()))
    return records


def tokenize_rows(rows: list[dict], tokenizer, max_length: int) -> list[dict]:
    """Tokenize before Arrow can add null source keys to variable-sized designs."""
    tokenized = []
    for row in rows:
        encoded = tokenizer.apply_chat_template(
            row['messages'], tools=row['tools'], enable_thinking=False,
            return_dict=True, return_assistant_tokens_mask=True)
        if len(encoded['input_ids']) > max_length:
            raise ValueError('SFT episode exceeds --max-length; increase it or filter the episode')
        if not any(encoded['assistant_masks']):
            raise ValueError('SFT episode has no assistant tokens to train')
        tokenized.append({k: encoded[k] for k in ('input_ids', 'assistant_masks')})
    return tokenized


def main() -> None:
    from .runtime import add_arguments, load_model, run_training, validate_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, 'sft')
    parser.add_argument('--records', nargs='+', type=Path)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    validate_arguments(args)
    if args.synthetic and (not args.smoke or args.records):
        parser.error('--synthetic requires --smoke and cannot be mixed with episode records')
    if not args.synthetic and not args.records:
        parser.error('Supply --records or --smoke --synthetic')
    pack = None if args.synthetic else load_pack(args.tasks_dir)
    records = synthetic_records() if args.synthetic else read_records(args.records)
    rows = [row for record in records if (row := episode_to_chat(record, pack=pack, smoke=args.smoke))]
    if not rows:
        raise ValueError('No feasible, improving train episodes remain')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'conversion.json').write_text(json.dumps({
        'input_records': len(records), 'accepted': len(rows), 'synthetic': args.synthetic}, indent=2)+'\n')
    if args.prepare_only:
        (args.output / 'chat.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = load_model(args, sft=True)
    tokenized = tokenize_rows(rows, tokenizer, args.max_length)
    config = SFTConfig(
        output_dir=str(args.output), max_steps=2 if args.smoke else args.steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.accum,
        # Explicit assistant_masks already computed above; TRL always folds them into labels.
        max_length=args.max_length, packing=False, assistant_only_loss=False,
        learning_rate=args.learning_rate, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, optim='paged_adamw_8bit',
        logging_steps=1, save_strategy='no' if args.smoke else 'steps', save_steps=50,
        report_to='none', push_to_hub=False, seed=args.seed,
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, args=config,
                         train_dataset=Dataset.from_list(tokenized))
    run_training(trainer, tokenizer, args, records_per_step=None)


if __name__ == '__main__':
    main()
