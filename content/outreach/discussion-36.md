# Post for FineEnvs discussion #36

We built a simulation environment for a different kind of job and would like to add it to the catalog
(PR: <link>).

**PermitForge.** An engineer is handed a plant that already meets its air-quality limits with every stack
on maximum control at its tallest allowed height, and has to make it cheaper. The agent picks a control
and a stack height per source, can evaluate a design's air quality 30 times, and then submits once. The
grade comes from EPA's AERMOD and BPIP-PRIME over five years of real NOAA weather, so there is no judge
and no reference solution to leak. The plants are synthetic; the weather and the models are real.

- Play: https://huggingface.co/spaces/Attoloop/PermitForge
- Code: https://github.com/attoloop/PermitForge
- Dataset with every rollout and the training run: https://huggingface.co/datasets/Attoloop/PermitForge

Four frontier models on 10 held-out tasks, three repeats each: GPT-6.1 Sol 0.607, GPT-5.6 Sol 0.586,
Claude Opus 5.5 0.576, Claude Sonnet 5.5 0.507 of the draft cost saved. Every episode found a feasible
design, so the whole difference is how much cost they removed.

On post-training: we ran GRPO on Qwen3.8-27B (30 steps, six rollouts, one H200) and evaluated before and
after on the same held-out tasks. Nothing changed (0.439 both ways). The write-up in the repository keeps
every failure on the way, including TRL's full-completion logits not fitting next to a 248k-vocabulary
model, which we fixed with a one-method chunked loss that may be useful to others training Qwen3.8-class
models on one card.
