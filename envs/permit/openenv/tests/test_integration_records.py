"""On-demand evaluation and private replay arrays through the real MCP dispatch."""
from contextlib import contextmanager

import numpy as np
from openenv.core.env_server.mcp_types import CallToolAction

from permit_core.fields import FieldPack, height_key
from permit_openenv.environment import PermitForgeEnvironment
from permit_openenv.evaluator import FastEvaluator
from permit_openenv.protocol import tool_result


def test_on_demand_note_once_and_record_only_arrays(pack, monkeypatch):
    import permit_core.fields as fields
    task = pack.get('003bd328a392-s47')
    n = len(task.scenario.receptors)
    data = FieldPack(task.task_id,
                     {(s.id, height_key(h)): np.full((192, n), 1e-6)
                      for s in task.scenario.sources for h in task.scenario.height_menu[s.id]},
                     np.ones(192, dtype=bool), (0, 192), np.arange(n))
    leases = []
    @contextmanager
    def ensure(scenario, *, jobs):
        leases.append('acquired')
        try:
            yield data
        finally:
            leases.append('released')
    monkeypatch.setattr(fields, 'ensure_pack', ensure)
    monkeypatch.delenv('PERMIT_FIELDS_DIR', raising=False)
    evaluator = FastEvaluator()
    env = PermitForgeEnvironment(pack, evaluator=evaluator, verifier=evaluator)
    try:
        env.reset(task_id=task.task_id)
        for i, tool in enumerate(['evaluate_design', 'evaluate_design', 'submit_design']):
            observation = env.step(CallToolAction(tool_name=tool, arguments={'design': task.view()['draft_design']}))
            result = tool_result(observation)
            assert result['feasible'], result
            assert ('note' in result) == (i == 0)
            assert not any(k.startswith('receptor_') for k in result)
            record = env.episode.record()
            step = record['steps'][-1]
            assert set(step['receptor_values']) == {s.id for s in task.scenario.standards}
            assert all(len(v) == n for v in step['receptor_values'].values())
            assert all(len(v) == n for v in step['receptor_margins'].values())
            assert 'receptor_values' not in step['result']
        assert leases == ['acquired', 'released']*3
        assert 'receptor_values' not in record['final']
    finally:
        env.close()
