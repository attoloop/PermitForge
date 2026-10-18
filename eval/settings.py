"""Launch effort settings, used by requests, manifests and dry-run output."""
from __future__ import annotations

MODELS = ('anthropic:claude-opus-5-5', 'anthropic:claude-sonnet-5-5',
          'codex:gpt-6.1-sol', 'codex:gpt-5.6-sol',
          'anthropic:claude-fable-5-1', 'claudecode:claude-fable-5-1', 'codex:gpt-6-astra')
MAX_TOKENS = 32768
EFFORTS = ('medium', 'high', 'xhigh')
EPISODE_PRIORS = {'medium': 2.0, 'high': 3.5, 'xhigh': 5.0}
PRIOR_ASSUMPTION = ('Unmeasured planning assumption: paid USD/episode is 2.00 at medium, '
                    '3.50 at high, 5.00 at xhigh (40%, 70%, 100% of the former 5.00 prior). '
                    'The same prior applies to both Claude models; it is not a provider quote. '
                    '--prior-episode-usd overrides it; measured round-one costs replace it.')


def model_settings(model: str, effort: str = 'xhigh') -> dict:
    if effort not in EFFORTS:
        raise ValueError('effort must be medium, high or xhigh')
    if model.startswith('codex:'):
        return {'launch_effort': effort, 'effort': effort, 'model_reasoning_effort': effort}
    if model.startswith('claudecode:'):
        return {'launch_effort': effort, 'effort': effort, 'harness': 'claude-code-subscription'}
    if model.startswith('anthropic:') and model in MODELS:
        native = 'max' if effort == 'xhigh' else effort
        return {'launch_effort': effort, 'effort': native, 'output_config': {'effort': native},
                'thinking': {'type': 'adaptive', 'display': 'summarized'}}
    raise ValueError('no pinned effort for '+model)


def label(model: str, settings: dict) -> str:
    return model+'@'+settings['effort'] if settings.get('effort') else model
