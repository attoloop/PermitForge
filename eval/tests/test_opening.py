"""Our own agent loops open every episode with the one shared message (frontier Claude and cloud Qwen)."""
import copy
from types import SimpleNamespace

from eval.anthropic_driver import run_anthropic
from permit_openenv.prompts import INSTRUCTIONS, opening


def test_opening_text_is_the_recorded_frontier_message():
    assert opening(40) == 'Use the permitforge tools. You have 40 turns; submit before the last turn ends.'


def test_anthropic_driver_sends_the_shared_opening_and_instructions():
    from test_contract import Session as FakeSession
    sent = []
    def create(**kwargs):
        sent.append(copy.deepcopy(kwargs))
        return SimpleNamespace(content=[{'type': 'text', 'text': 'done'}],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=2))
    api = SimpleNamespace(messages=SimpleNamespace(create=create))
    run_anthropic(FakeSession(), 'anthropic:claude-sonnet-5-5', api=api, max_turns=40)
    assert sent[0]['messages'][0] == {'role': 'user', 'content': opening(40)}
    assert sent[0]['system'] == INSTRUCTIONS
