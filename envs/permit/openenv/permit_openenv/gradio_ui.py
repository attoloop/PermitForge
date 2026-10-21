"""The /web playground: OpenEnv's playground with our README, quick start and guide."""
import json
import os
import random
from pathlib import Path

from .prompts import INSTRUCTIONS
from .tasks import TaskPack

# The eval tasks the Space image prepares in advance (space_root/bake_showcases.py).
SHOWCASES = ('040cd27878ef-s47', '0558a6a6c25d-s47')
CARD = Path(__file__).resolve().parents[1] / 'space_root/README.md'

QUICK_START = '''**Connect an agent.** The MCP endpoint is `{url}/mcp`. From Python with `openenv==0.7.0`:

```python
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.mcp_client import MCPToolClient

url = "{url}"
with MCPToolClient(url, message_timeout_s=1800).sync() as env:
    env.reset(task_id="040cd27878ef-s47")
    action = CallToolAction(tool_name="get_scenario", arguments={{}})
    print(env.step(action).observation)
```'''

GUIDE = '''## Play one episode

1. **Reset** on the Playground tab loads one of two eval tasks that the Space prepares in advance, so evaluations take seconds.
2. Set Tool Name to `get_scenario` and Arguments to `{}`, then press **Step**. Leave Type empty. In the response, `structured_content.result` holds the sources, controls, height menus, standards and `draft_design`.
3. Copy `draft_design`, change one control or height, and run `check_design` with `{"design": ...}`. It returns validity and cost, for free.
4. Run `evaluate_design` with the same arguments to see each standard's margin. An episode has 30.
5. Run `submit_design` to end the episode. EPA's models grade it in 2 to 3 minutes, and the reward is the share of draft cost saved.

A design looks like `{"S1": {"control": "S1_moderate", "height_m": 45.0}, "S2": ...}`, with every source once.'''


def card_text() -> str:
    text = CARD.read_text() if CARD.exists() else ''
    return text.split('---', 2)[2].strip() if text.startswith('---') else text


def build_ui(pack: TaskPack, web_manager, action_fields, metadata, is_chat_env):
    import gradio as gr
    from openenv.core.env_server.gradio_ui import build_gradio_app

    reset, step = web_manager.reset_environment, web_manager.step_environment
    showcases = [t.task_id for t in pack.tasks if t.task_id in SHOWCASES]

    async def reset_showcase(reset_kwargs=None):
        kwargs = dict(reset_kwargs or {})
        if showcases and not kwargs.keys() & {'task_id', 'split', 'index'}:
            kwargs['task_id'] = random.choice(showcases)
        return await reset(kwargs)

    async def step_json(action_data):
        arguments = action_data.get('arguments')
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as error:
                raise ValueError(f'Arguments must be JSON, for example {{}}: {error}') from None
            action_data = {**action_data, 'arguments': arguments}
        data = await step(action_data)
        structured = ((data.get('observation') or {}).get('result') or {}).get('structured_content') or {}
        if isinstance(structured.get('result'), str):  # show the tool's JSON as JSON, not as an escaped string
            try:
                structured['result'] = json.loads(structured['result'])
            except json.JSONDecodeError:
                pass
        return data

    web_manager.reset_environment, web_manager.step_environment = reset_showcase, step_json
    host = os.environ.get('SPACE_HOST')
    url = f'https://{host}' if host else f"http://127.0.0.1:{os.environ.get('PORT', '8012')}"
    metadata = metadata.model_copy(update={'readme_content': card_text()})
    playground = build_gradio_app(web_manager, action_fields, metadata, is_chat_env, title='PermitForge',
                                  quick_start_md=QUICK_START.format(url=url))
    with gr.Blocks() as guide:
        gr.Markdown(GUIDE)
        gr.Markdown(f"{pack.count('eval')} eval and {pack.count('train')} train tasks are available.")
        with gr.Accordion('What an agent is told', open=False):
            gr.Markdown(INSTRUCTIONS)
    return gr.TabbedInterface([playground, guide], tab_names=['Playground', 'Guide'], title='PermitForge')
