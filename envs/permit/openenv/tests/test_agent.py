import sys
from types import SimpleNamespace

from permit_openenv.agent import run_episode
from permit_openenv.environment import PermitForgeEnvironment
from conftest import StubEvaluator


class Client:
    def __init__(self, env):
        self.env = env
    def reset(self, **kwargs):
        return SimpleNamespace(observation=self.env.reset(**kwargs))
    def list_tools(self):
        from openenv.core.env_server.mcp_types import ListToolsAction
        return self.env.step(ListToolsAction()).tools
    def step(self, action):
        obs = self.env.step(action)
        return SimpleNamespace(observation=obs, done=obs.done, reward=obs.reward)


def test_anthropic_loop_records_submission_without_network(pack, monkeypatch):
    env = PermitForgeEnvironment(pack, evaluator=StubEvaluator(), verifier=StubEvaluator())
    design = pack.get('003bd328a392-s47').view()['draft_design']
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type='text', text='x'*2200),
                                        SimpleNamespace(type='tool_use', name='submit_design',
                                                        input={'design': design}, id='call1')],
                               usage=SimpleNamespace(input_tokens=10, output_tokens=20))
    monkeypatch.setitem(sys.modules, 'anthropic', SimpleNamespace(
        Anthropic=lambda **kw: SimpleNamespace(messages=SimpleNamespace(create=create))))
    try:
        record = run_episode(Client(env), 'anthropic:test', task_id='003bd328a392-s47')
        assert record['ended'] == 'submitted' and record['final']['reward'] == 0
        assert record['usage'] == {'input_tokens': 10, 'output_tokens': 20, 'usd': None}
        assert len(record['steps'][0]['assistant_text']) == 2000
        assert record['geometry'] == {}
        assert env.episode.record()['geometry']['station'] == 'KMSP'
        assert len(calls) == 1
        assert {t['name'] for t in calls[0]['tools']} == {
            'get_scenario', 'check_design', 'evaluate_design', 'submit_design'}
    finally:
        env.close()
