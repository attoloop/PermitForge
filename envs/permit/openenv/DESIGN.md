# PermitForge OpenEnv integration

This package follows PortSim's FastMCP closures, MCPEnvironment subclass, task
API, terminal rubric, server factory and optional client package. It adapts the
public `height_m` field to the native core's `stack_height_m`. The native adapter
always selects ARM2; submission always uses the full native verifier.

Dependencies are editable `permit-core`, `openenv==0.7.0`, `fastmcp>=3,<4`,
`gradio>=5` and `uvicorn[standard]>=0.30`. Optional `anthropic>=0.40` implements
the model tool loop. The lockfile pins resolved versions. FastAPI, Pydantic and
httpx are OpenEnv dependencies; NumPy comes from the core evaluator.

OpenEnv's standard Gradio playground and task guide use its default `/web/`
mount and helper routes. Tests exercise the Gradio configuration and a real
MCPToolClient WebSocket session. Native work runs in a thread without blocking
the ASGI loop and retains terminal state even if submission fails.

A valid evaluation spends its budget, including on infrastructure failure;
malformed designs do not. Reward is the feasible saved fraction of draft cost;
failed submissions and no submission score zero. Search and baseline results
remain diagnostics, with no calibration prerequisites.

Fast evaluation acquires an on-demand float64 field-pack lease and releases it
after each call. Shared locks prevent duplicate builds and eviction of active
packs. Native work shares bounded process slots. Fast and native adapters retain
every receptor's value and margin in episode records; MCP responses omit arrays.
Geometry and wind counts remain available for later analysis.

The container installs the local Python packages and copies converted task/met
inputs. Showcase preparation uses the same `ensure_pack` API as runtime
preparation. See [the root design](../../../DESIGN.md) and
[Space card](space_root/README.md) for the build and data requirements.
