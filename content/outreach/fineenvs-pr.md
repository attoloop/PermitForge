# PR to adithya-s-k/FineEnvs

Title: `Add PermitForge, an air-permit design environment (link-only entry)`

Body:

PermitForge is an OpenEnv environment for a real engineering job: make an industrial plant cheaper while it
still meets every air-quality standard. The agent picks a control and a stack height for each source, can
test a design 30 times, and submits once. EPA's AERMOD grades the submission over five years of real weather.

Write-up: https://attoloop.com/research/permitforge/

This is a link-only entry. If you would rather have the full code here, we are happy to move it in.

## End-to-end recipe

- [x] Environment: [code](https://github.com/attoloop/PermitForge), [rules](https://github.com/attoloop/PermitForge/blob/main/CONTRACT.md)
- [x] Rollout with a readable trajectory: [GPT-6.1 Sol on a kiln plant](https://huggingface.co/datasets/Attoloop/PermitForge/blob/main/rollouts/final-medium3/codex_gpt-6.1-sol-cbf796da1b7a/d50e1051dcec-s47/r1/transcript.jsonl)
- [x] Training run: [setup and launch](https://github.com/attoloop/PermitForge/blob/main/train/cloud/README.md)
- [x] Results: [frontier eval](https://github.com/attoloop/PermitForge/blob/main/results/final-eval.md), [first training run](https://github.com/attoloop/PermitForge/blob/main/train/cloud/RESULTS.md)
- [x] Artifacts on the Hub: [Space](https://huggingface.co/spaces/Bhuminjay/PermitForge), [dataset](https://huggingface.co/datasets/Attoloop/PermitForge), [model](https://huggingface.co/Attoloop/PermitForge-Qwen3.8-27B-GRPO)

## Before opening the PR

- [x] `python3 tools/build_index.py --check` passes
- [x] The environment runs: `uv run --directory envs/permit/openenv python rollout.py --smoke`
- [x] Read a trajectory by hand: [GPT-6.1 Sol on a kiln plant](https://huggingface.co/datasets/Attoloop/PermitForge/blob/main/rollouts/final-medium3/codex_gpt-6.1-sol-cbf796da1b7a/d50e1051dcec-s47/r1/transcript.jsonl)

The training card sits next to the eval card, since the catalog has no training recipes yet. Happy to move it.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
