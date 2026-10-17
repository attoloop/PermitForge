---
title: PermitForge
sdk: docker
app_port: 7860
short_description: Optimize plant emissions controls and stack heights.
tags:
  - openenv
  - reinforcement-learning
  - simulation
  - air-quality
  - aermod
---

# PermitForge

Choose emissions controls and stack heights for a synthetic industrial plant,
using real NOAA weather and EPA AERMOD/BPIP-PRIME. Cut annualized cost while
meeting every task air-quality standard.

Open **[/web/](/web/)** for the standard OpenEnv Gradio playground and task guide,
and **[/healthz](/healthz)** for health. The MCP endpoint is **`/mcp`**; the
repository README provides a client example.
The shared playground is a demo, not a scored endpoint. Scored runs use the eval
harness, which forces `ENABLE_WEB_INTERFACE=false`.

The image prepares float64 fields for the eval tasks listed in
`space_root/bake_showcases.py`. Other tasks prepare fields on their first
evaluation, which may take minutes on a CPU Space. Later calls reuse an
evictable cache. Submission always runs full native BPIP/AERMOD verification
and may also take minutes. Use a local server for heavy evaluation runs.

All source tasks are admitted after native ARM2 draft verification. Reproduce
counts with `python3 envs/permit/openenv/space_root/report_evidence.py` from the
repository root. Model results remain **TODO**.
Reward is feasible USD/year saved divided by draft cost. The draft and failed or
absent submissions score zero. A score of 1.0 means zero annualized cost, which
is unreachable because the uncontrolled design fails an air-quality standard.
Raw USD/year and percentage savings are also returned. No calibration is needed.

Facilities are synthetic; weather comes from the selected NOAA station.
Concentrations are NAAQS design values across the task years. Code is Apache-2.0;
NOAA/EPA provenance and government-work terms are in the repository `SOURCES.md`.

Build from the repository root with converted task/met data present:

```sh
podman build -f envs/permit/openenv/Dockerfile -t permitforge-space .
podman run --rm --name permitforge-space -p 127.0.0.1:7860:7860 permitforge-space
```

Publication and a rebuild of the current container recipe remain **TODO**.
A Space export must put this card at its root, retain the repository-root build
context, and supply a root Dockerfile copied from `envs/permit/openenv/Dockerfile`.
