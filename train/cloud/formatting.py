"""Pinned Qwen XML calls and assistant-only labels, without changing the template."""
import json
import math
import re

from train.permit_env import TOOL_NAMES
from .config import CloudConfig


CALL = re.compile(r'<tool_call>\s*<function=([a-z_]+)>\s*(.*?)\s*</function>\s*</tool_call>', re.S)
PARAMETER = re.compile(r'<parameter=([a-z_]+)>\s*(.*?)\s*</parameter>', re.S)


def parse_calls(text: str) -> list[dict]:
    calls = []
    for match in CALL.finditer(text):
        name, body = match.groups()
        if name not in TOOL_NAMES:
            raise ValueError('Unknown tool')
        arguments = {}
        for param in PARAMETER.finditer(body):
            key, value = param.groups()
            if key in arguments:
                raise ValueError('Duplicate parameter')
            arguments[key] = json.loads(value)
        if PARAMETER.sub('', body).strip() or set(arguments) != (set() if name == 'get_scenario' else {'design'}):
            raise ValueError('Unexpected tool arguments')
        if name != 'get_scenario':
            design = arguments['design']
            if not isinstance(design, dict) or not design:
                raise ValueError('design must be a mapping')
            for row in design.values():
                if (not isinstance(row, dict) or set(row) != {'control','height_m'} or
                    not isinstance(row['control'], str) or type(row['height_m']) not in (float,int) or
                    not math.isfinite(row['height_m'])):
                    raise ValueError('Invalid design shape')
        calls.append({'type':'function', 'function':{'name':name, 'arguments':arguments}})
    remainder = CALL.sub('', text)
    if '<tool_call' in remainder or '<function=' in remainder or '</tool_call>' in remainder:
        raise ValueError('Malformed or truncated tool call')
    return calls


def load_tokenizer(*, local_files_only: bool = False):
    from transformers import AutoTokenizer
    cfg = CloudConfig(hourly_usd=1)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision, local_files_only=local_files_only)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return tokenizer


def render(tokenizer, messages: list[dict], tools: list[dict], *, prompt: bool = False) -> str:
    return tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
        add_generation_prompt=prompt, enable_thinking=False)


def mask_spans(text: str) -> list[tuple[int,int]]:
    spans = []
    pattern = re.compile(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', re.S)
    empty_think = '<think>\n\n</think>\n\n'
    for match in pattern.finditer(text):
        start = match.start(1)
        if match.group(1).startswith(empty_think):
            start += len(empty_think)
        spans.append((start, match.end()))
    return spans


def encode_row(tokenizer, row: dict, max_length: int | None = None) -> dict:
    text = render(tokenizer, row['messages'], row['tools'])
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if max_length and len(encoded['input_ids']) > max_length:
        raise ValueError('Episode exceeds context limit; never truncate a final submission')
    spans = mask_spans(text)
    masks = [int(b>a and any(a>=start and b<=end for start,end in spans))
             for a,b in encoded['offset_mapping']]
    if not any(masks):
        raise ValueError('No assistant targets')
    return {'input_ids':encoded['input_ids'], 'assistant_masks':masks}
