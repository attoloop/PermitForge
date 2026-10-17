import argparse
import inspect

from train.permit_env import PermitTrainingEnv, TOOL_NAMES
from train.runtime import add_arguments, load_tokenizer, validate_arguments


def test_transformers_extracts_exact_four_tools():
    from transformers.utils import get_json_schema
    from trl import GRPOTrainer

    assert 'environment_factory' in inspect.signature(GRPOTrainer.__init__).parameters
    schemas = [get_json_schema(getattr(PermitTrainingEnv, name)) for name in TOOL_NAMES]
    assert {s['function']['name'] for s in schemas} == set(TOOL_NAMES)
    for schema in schemas:
        function = schema['function']
        if function['name'] != 'get_scenario':
            assert function['parameters']['required'] == ['design']


def test_already_training_ready_template_is_kept(monkeypatch):
    import transformers
    import trl.chat_template_utils

    class Tokenizer:
        chat_template = 'already supports assistant masking'
        pad_token = 'pad'
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(trl.chat_template_utils, 'get_training_chat_template', lambda tokenizer: None)
    parser = argparse.ArgumentParser()
    add_arguments(parser, 'sft')
    args = parser.parse_args([])
    validate_arguments(args)
    assert 'already supports assistant masking' in load_tokenizer(args, sft=True).chat_template
