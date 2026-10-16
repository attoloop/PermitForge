"""JSON boundary shared by tools and episode records."""
from decimal import Decimal
import json
import math
from typing import Any

DESIGN_SCHEMA = {
    'type': 'object',
    'additionalProperties': {
        'type': 'object', 'additionalProperties': False,
        'required': ['control', 'height_m'],
        'properties': {'control': {'type': 'string'}, 'height_m': {'type': 'number'}},
    },
}


def json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def tool_result(observation: Any) -> dict:
    if getattr(observation, 'error', None):
        error = observation.error
        return {'error': error.get('message') if isinstance(error, dict) else error.message}
    result = observation.result
    if hasattr(result, 'data') and result.data is not None:
        return json.loads(result.data) if isinstance(result.data, str) else result.data
    if hasattr(result, 'content'):
        result = {'content': [b.model_dump() for b in result.content]}
    if hasattr(result, 'model_dump'):
        result = result.model_dump()
    if isinstance(result, dict):
        for block in result.get('content', []):
            if block.get('type') == 'text':
                return json.loads(block['text'])
        if 'data' in result:
            return json.loads(result['data']) if isinstance(result['data'], str) else result['data']
        return result
    return json.loads(result)
