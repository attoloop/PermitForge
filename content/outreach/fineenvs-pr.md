# PR to adithya-s-k/FineEnvs

Title: `Add PermitForge, an air-permit design environment (link-only entry)`

Body:

In PermitForge, an agent gets a plant that already meets its air-quality limits, with every stack on full
control at full height. Its job is to make the plant cheaper without breaking any limit. It picks a control
and a stack height for each source, can test a design's air quality 30 times, and then submits once. EPA's
own dispersion model, AERMOD, grades the submission over five years of real weather, so there is no judge
model and no answer to leak.

We wrote up the whole project, from the idea to the training run, here:
https://attoloop.com/research/permitforge/

This is a link-only entry: a manifest, a README card and two eval cards. The code, data and models stay in
our own repos. If you would rather have the full code here, like the PortSimEnv entries, we are happy to
move it in.

- Code: https://github.com/attoloop/PermitForge
- Space: https://huggingface.co/spaces/Bhuminjay/PermitForge
- Dataset: https://huggingface.co/datasets/Attoloop/PermitForge (tasks, weather, every rollout with its transcript, and the whole training run)
- Model: https://huggingface.co/Attoloop/PermitForge-Qwen3.8-27B-GRPO

## End-to-end recipe

- [x] **Environment.** Four tools: read the scenario, check a design's cost, test its air quality, and
  submit. Each test shows every standard's value, how close it is to the limit, and where. The rules are in
  [CONTRACT.md](https://github.com/attoloop/PermitForge/blob/main/CONTRACT.md).
- [x] **A working rollout you can read.** Every frontier episode is in the dataset with its full transcript,
  for example [GPT-6.1 Sol on a kiln plant in Albuquerque](https://huggingface.co/datasets/Attoloop/PermitForge/blob/main/rollouts/final-medium3/codex_gpt-6.1-sol-cbf796da1b7a/d50e1051dcec-s47/r1/transcript.jsonl).
- [x] **Training run.** As a small first experiment, we trained Qwen3.8-27B with GRPO in TRL (LoRA, 30 steps,
  6 rollouts each) on one H200 on HF Jobs. The setup and launch commands are in
  [train/cloud/README.md](https://github.com/attoloop/PermitForge/blob/main/train/cloud/README.md).
- [x] **Results, the honest version.** On 10 held-out tasks, 3 runs each, the models saved this share of the
  draft's cost: GPT-6.1 Sol 0.606, GPT-5.6 Sol 0.586, Claude Opus 5.5 0.576, Claude Sonnet 5.5 0.507. Every
  episode ended with a design that passes. Simple policies get 0.00 (keep the draft) and 0.33 (greedy).
  At this size, training did not move the score: 0.4394 before and 0.4390 after. A larger run on the full train
  split of 230 tasks is lined up next. The curve, the per-task table and every problem we hit are in [RESULTS.md](https://github.com/attoloop/PermitForge/blob/main/train/cloud/RESULTS.md).
  We also checked every transcript for calls outside the four tools, draft submissions and tests past the
  limit, and found none ([audit](https://github.com/attoloop/PermitForge/blob/main/results/reward-hacks.md)).
- [x] **Hub artifacts.** The Space, the dataset and the trained model, linked above.

## Before opening this PR

- [x] `python3 tools/build_index.py --check` passes.
- [x] The environment runs. `uv run --directory envs/permit/openenv python rollout.py --smoke --task-id 040cd27878ef-s47`
  finishes with a graded submission from a fresh clone. The same check passes on the live Space.
- [x] We read one trajectory by hand, the GPT-6.1 Sol episode above. The model reads the scenario and tests
  28 designs. The first ones fail badly, and each result shows which standard fails and where. 11 designs
  pass, and it submits the cheapest: $1,376,183 a year instead of $3,569,382. The final check agrees, and
  the reward is 0.614.

The PR adds the new folder, its tile and eval rows in the section README, and one entry in the root README. Since
the catalog has no training recipes yet, the training card sits next to the main eval card. Happy to move it.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
