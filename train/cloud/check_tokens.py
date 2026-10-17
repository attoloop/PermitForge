"""CPU-only recorded-episode audit: python -m train.cloud.check_tokens --output PATH."""
import argparse
import hashlib
import json
from pathlib import Path

from .formatting import encode_row, load_tokenizer, parse_calls, render
from .make_sft import build_dataset
from .config import CloudConfig


def audit(rows: list[dict], tokenizer) -> dict:
    result = []
    from trl.chat_template_utils import add_response_schema, parse_response, is_chat_template_prefix_preserving
    tokenizer = add_response_schema(tokenizer)
    for row in rows:
        encoded = encode_row(tokenizer, row)
        text = render(tokenizer, row['messages'], row['tools'])
        expected = [call for m in row['messages'] for call in m.get('tool_calls', [])]
        parsed = parse_calls(text.split('<|im_start|>assistant\n',1)[1])
        assert [c['function'] for c in parsed] == [c['function'] for c in expected]
        trl_roundtrips = 0
        for index, message in enumerate(row['messages']):
            if message['role'] != 'assistant':
                continue
            prefix = render(tokenizer, row['messages'][:index], row['tools'], prompt=True)
            full = render(tokenizer, row['messages'][:index+1], row['tools'])
            assert full.startswith(prefix), 'Template is not prefix-preserving'
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            full_ids = tokenizer.encode(full, add_special_tokens=False)
            assert full_ids[:len(prefix_ids)] == prefix_ids
            parsed_trl = parse_response(tokenizer, full_ids[len(prefix_ids):], prefix=prefix_ids)
            actual = [c['function'] for c in parsed_trl.get('tool_calls', [])]
            wanted = [c['function'] for c in message.get('tool_calls', [])]
            assert actual == wanted, (actual,wanted)
            trl_roundtrips += len(wanted)
        labels = tokenizer.decode([token for token,mask in zip(encoded['input_ids'], encoded['assistant_masks']) if mask])
        assert '<tool_response>' not in labels and '<|im_start|>user' not in labels
        assert '<think>' not in labels
        assert all(f'<function={c["function"]["name"]}>' in labels for c in expected)
        prompt = render(tokenizer, row['messages'][:2], row['tools'], prompt=True)
        on = tokenizer.apply_chat_template(row['messages'][:2], tools=row['tools'], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
        assert prompt.endswith('<think>\n\n</think>\n\n') and on.endswith('<think>\n')
        result.append(dict(task_id=row['task_id'], prompt_tokens=len(tokenizer.encode(prompt,add_special_tokens=False)),
            episode_tokens=len(encoded['input_ids']), assistant_tokens=sum(encoded['assistant_masks']),
            tool_calls=len(expected), trl_roundtrips=trl_roundtrips,
            fits_training_context=len(encoded['input_ids']) <= CloudConfig(hourly_usd=1).max_length))
    return dict(model=CloudConfig(hourly_usd=1).model, revision=CloudConfig(hourly_usd=1).revision,
        template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        thinking=False, assistant_mask='only assistant completion bodies and im_end; prompt, tool results and empty thinking prefix excluded',
        template_prefix_preserving=is_chat_template_prefix_preserving(tokenizer), episodes=result)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', nargs='+', type=Path, default=[Path('results/rollouts/wp2.0b-smoke')])
    p.add_argument('--tasks-dir', type=Path, default=Path('envs/permit/tasks'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows, stats = build_dataset(a.records, a.tasks_dir)
    if not rows:
        raise ValueError('No accepted recorded episodes')
    tokenizer = load_tokenizer()
    result = audit(rows, tokenizer)
    rendered = [render(tokenizer,r['messages'],r['tools']) for r in rows]
    args_output_parent = a.output.parent
    args_output_parent.mkdir(parents=True,exist_ok=True)
    a.output.with_suffix('.rendered.txt').write_text('\n'.join(rendered))
    result['data'] = stats
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
