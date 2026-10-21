---
title: PermitForge
sdk: docker
app_port: 7860
license: apache-2.0
short_description: Optimize plant emissions controls and stack heights.
tags:
  - openenv
  - reinforcement-learning
  - simulation
  - air-quality
  - aermod
---

# PermitForge

Make a synthetic plant cheaper while it still meets every air-quality standard under five years of real weather.

The agent picks one control and one stack height for each source. It can price designs for free, evaluate
their air quality 30 times, and then submits once. EPA's BPIP-PRIME and AERMOD grade the submission.

| Tool | What it does |
|---|---|
| `get_scenario()` | Sources, controls, height menus, standards and the draft design |
| `check_design(design)` | Validity and exact annualized cost, free |
| `evaluate_design(design)` | Each standard's value, margin and binding receptor, 30 per episode |
| `submit_design(design)` | Ends the episode and grades once with the native models |

**Reward.** The fraction of the draft's annualized cost that a feasible design saves. The unchanged draft,
an infeasible design and no submission score zero.

**Data.** 360 tasks, 230 train and 130 eval, at 15 US weather stations. Weather is NOAA ISD and IGRA2,
2020 to 2024, processed by EPA AERMET. Plants, emissions and prices are synthetic.

**Results.** Share of draft cost saved on 10 held-out eval tasks, 3 repeats each: GPT-6.1 Sol 0.607,
GPT-5.6 Sol 0.586, Claude Opus 5.5 0.576, Claude Sonnet 5.5 0.507.

**Use it.** The Playground tab runs one episode by hand, and the Guide tab shows how. Agents connect over
MCP at `/mcp`. A submission runs the native models and takes 2 to 3 minutes on this free CPU, so use your
own server for many episodes:

```sh
podman build -f envs/permit/openenv/Dockerfile -t permitforge-space .   # from a checkout with the task data
podman run --rm -p 127.0.0.1:7860:7860 permitforge-space
```

[Code](https://github.com/attoloop/PermitForge) · [Dataset](https://huggingface.co/datasets/Attoloop/PermitForge) ·
[Trained model](https://huggingface.co/Attoloop/PermitForge-Qwen3.8-27B-GRPO). Code is Apache-2.0. NOAA and
EPA inputs are U.S. Government works.
