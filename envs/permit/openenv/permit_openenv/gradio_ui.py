"""Small task guide beside OpenEnv's interactive MCP playground."""
from .prompts import INSTRUCTIONS
from .tasks import TaskPack


def build_ui(pack: TaskPack):
    import gradio as gr
    with gr.Blocks() as ui:
        gr.Markdown('# PermitForge v2')
        gr.Markdown(INSTRUCTIONS)
        gr.Markdown('Submission independently checks your design and may take several minutes.')
        gr.JSON(value={s: pack.count(s) for s in pack.splits()}, label='Available tasks')
    return ui
