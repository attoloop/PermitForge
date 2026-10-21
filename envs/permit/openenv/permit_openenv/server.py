"""Run locally: uv run uvicorn permit_openenv.server:app --port 8012."""
import os

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from .environment import PermitForgeEnvironment, PermitState
from .evaluator import Evaluator
from .gradio_ui import build_ui
from .limits import RequestLimits
from .tasks import TaskPack, load_pack


def create_server(*, pack: TaskPack | None = None, evaluator: Evaluator | None = None,
                  verifier: Evaluator | None = None, evaluate_budget: int = 30):
    pack = pack if pack is not None else load_pack()

    def factory():
        return PermitForgeEnvironment(pack, evaluator=evaluator, verifier=verifier, evaluate_budget=evaluate_budget)

    def gradio_builder(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
        return build_ui(pack, web_manager, action_fields, metadata, is_chat_env)

    os.environ.setdefault('ENABLE_WEB_INTERFACE', 'true')
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'false')
    app = create_app(factory, CallToolAction, CallToolObservation, env_name='permitforge',
                     max_concurrent_envs=min(16, int(os.environ.get('MAX_CONCURRENT_ENVS', '4'))),  # eval harness raises it; Space default 4
                     gradio_builder=gradio_builder, show_default_tab=False,
                     title_override='PermitForge', state_cls=PermitState)
    app.add_middleware(RequestLimits)

    @app.get('/healthz', include_in_schema=False)
    def healthz():
        return {'status': 'ok', 'tasks': {s: pack.count(s) for s in pack.splits()}}

    return app


_app = None


def __getattr__(name: str):
    global _app
    if name == 'app':
        if _app is None:
            _app = create_server()
        return _app
    raise AttributeError(name)


def main() -> None:
    import uvicorn
    uvicorn.run(create_server(), host='0.0.0.0', port=int(os.environ.get('PORT', '8000')),
                ws_ping_timeout=1800)


if __name__ == '__main__':
    main()
