# PR to adithya-s-k/FineEnvs

Title: `Add PermitForge, an air-permit design environment (link-only entry)`

Body:

PermitForge is an OpenEnv environment for a real engineering job: making an industrial plant cheaper while it
still meets every air-quality standard.

The agent gets a plant that already passes, with every stack on full control at full height. It picks a
control and a height for each stack, can test a design's air quality 30 times, and then submits once. EPA's
dispersion model AERMOD grades the submission against five years of real weather. There is no judge model and
no answer to leak.

Full write-up: https://attoloop.com/research/permitforge/

This is a link-only entry: a manifest, a README card and two eval cards. The code, data and models stay in our
repos. If you would rather have the full code here, we are happy to move it in.

- Code: https://github.com/attoloop/PermitForge
- Space: https://huggingface.co/spaces/Bhuminjay/PermitForge
- Dataset: https://huggingface.co/datasets/Attoloop/PermitForge (tasks, weather, every rollout with its transcript, and the training run)
- Model: https://huggingface.co/Attoloop/PermitForge-Qwen3.8-27B-GRPO

## End-to-end recipe

- [x] **Environment.** Four tools: read the scenario, check a design's cost, test its air quality, and submit.
  Each test shows how close every standard is to its limit, and where.
  Rules: [CONTRACT.md](https://github.com/attoloop/PermitForge/blob/main/CONTRACT.md).
- [x] **A readable rollout.** Every frontier episode is in the dataset with its transcript, for example
  [GPT-6.1 Sol on a kiln plant in Albuquerque](https://huggingface.co/datasets/Attoloop/PermitForge/blob/main/rollouts/final-medium3/codex_gpt-6.1-sol-cbf796da1b7a/d50e1051dcec-s47/r1/transcript.jsonl).
- [x] **Training run.** Qwen3.8-27B with GRPO in TRL (LoRA, 30 steps, 6 rollouts each) on one H200 on HF Jobs.
  Setup and launch commands: [train/cloud/README.md](https://github.com/attoloop/PermitForge/blob/main/train/cloud/README.md).
- [x] **Results.** Share of draft cost saved on 10 held-out tasks, 3 runs each: GPT-6.1 Sol 0.606, GPT-5.6 Sol
  0.586, Claude Opus 5.5 0.576, Claude Sonnet 5.5 0.507. Every episode ended with a passing design. Simple
  policies score 0.00 (keep the draft) and 0.33 (greedy). Our first training run was small and did not change
  the held-out score (0.4394 before, 0.4390 after). A larger run on all 230 train tasks is next. Curves,
  per-task results and every problem we hit: [RESULTS.md](https://github.com/attoloop/PermitForge/blob/main/train/cloud/RESULTS.md).
  A check of every transcript found no calls outside the four tools, no draft submissions and no tests past
  the limit ([audit](https://github.com/attoloop/PermitForge/blob/main/results/reward-hacks.md)).
- [x] **Hub artifacts.** The Space, the dataset and the trained model, linked above.

## Before opening this PR

- [x] `python3 tools/build_index.py --check` passes.
- [x] The environment runs. `uv run --directory envs/permit/openenv python rollout.py --smoke --task-id 040cd27878ef-s47`
  finishes with a graded submission from a fresh clone, and the same check passes on the live Space.
- [x] We read one trajectory by hand, the GPT-6.1 Sol episode above. The model tests 28 designs. The early ones
  fail badly, and each result shows which standard fails and where. 11 pass, and it submits the cheapest:
  $1,376,183 a year instead of $3,569,382. The final check agrees, with a reward of 0.614.

The PR adds the new folder, its tile and eval rows in the section README, and one entry in the root README. The
catalog has no training recipes yet, so the training card sits next to the main eval card. Happy to move it.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
