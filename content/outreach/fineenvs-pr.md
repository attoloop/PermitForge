# PR to adithya-s-k/FineEnvs

Title: `07-simulation-environments: PermitForge, an air-permit design environment (link-only entry)`

Body:

PermitForge is an OpenEnv environment where an agent chooses emissions controls and stack heights for
a synthetic industrial plant so that it meets every air-quality standard under five years of real NOAA
weather, at the lowest annualized cost. Every submission is graded by EPA's own dispersion models
(BPIP-PRIME and AERMOD), so there is no judge model and no answer key to leak. The agent explores with a
fast evaluator built from precomputed unit fields (checked against the native model on 1,920 values to
five decimals) and gets 30 evaluations per episode.

Two differences from the other 07 entries: there is no solver reference or proven optimum, since the
reward is the share of the draft's cost saved, and the plants are synthetic while the weather is real.

This is a link-only entry: `project.yaml`, a README card and two eval cards. Code, data and Hub repos
stay in our namespace. If you would rather have the full code in this repo, as the PortSimEnv entries do,
we are happy to move it in.

- Code: https://github.com/attoloop/PermitForge
- Space: https://huggingface.co/spaces/Bhuminjay/PermitForge
- Dataset (tasks, weather, every rollout with its transcript, the whole training run): https://huggingface.co/datasets/Attoloop/PermitForge
- Model (the GRPO adapter): https://huggingface.co/Attoloop/PermitForge-Qwen3.8-27B-GRPO

Results on 10 held-out eval tasks, 3 repeats each: GPT-6.1 Sol 0.607, GPT-5.6 Sol 0.586, Claude Opus 5.5
0.576, Claude Sonnet 5.5 0.507 of the draft cost saved, every episode feasible. The naive policies are the
unchanged draft (0.00) and cheapest-first greedy (0.33).

We also trained Qwen3.8-27B with GRPO (30 steps, one H200, $164 in cloud jobs) and evaluated it
before and after on the same tasks with the same harness. It is a null result, no measurable change at 0.439 both ways, and it is
reported as such in a second card (`eval-v2-qwen.json`) and in `train/cloud/RESULTS.md` in the repository,
with every failure we met on the way (hosts without CUDA libraries, a 27 GB logits OOM and the chunked
loss that fixed it, a watchdog death at step 15). The catalog has no training recipes for now, so the
training card sits beside the panel card rather than in a `train/` folder. Happy to move it if you prefer.

Changes: the new folder, a tile and two rows in the 07 README, one bullet in the root README's 07 section,
and the regenerated root table (`python3 tools/build_index.py --check` passes).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
