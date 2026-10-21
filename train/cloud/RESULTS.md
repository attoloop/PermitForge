# Qwen3.8-27B GRPO on PermitForge: results

Run `full-30`: 30 GRPO steps, 6 rollouts per step, LoRA rank 16 on one H200. Base and trained model evaluated on the same 10 held-out tasks, 3 repeats each, native verification. Everything here is computed from the run folder (`python -m eval.learnability --run training/runs/full-30 --jobs training/logs/jobs.json --write`, paths as in the dataset).

## Result

| Model | Mean saved fraction | 95% CI | Feasible | USD per episode |
|---|---:|---|---:|---:|
| Qwen3.8-27B | 0.4394 | 0.3742 to 0.4950 | 30/30 | 1.01 |
| Qwen3.8-27B + GRPO LoRA | 0.4390 | 0.3845 to 0.5005 | 30/30 | 1.05 |

Paired difference (trained minus base) over 10 tasks: mean -0.0004, sd 0.0379, se 0.0120, station-clustered 95% CI -0.0280 to +0.0264; 5 tasks up, 5 down, 0 unchanged.

No measurable change in the mean. The largest per-task moves are -6.9 and +4.6 points. Base and trained repeats overlap on 9 of 10 tasks, and nothing in the pattern says the model learned to find cheaper designs.

## Per task

| Task | Station | Base r1 / r2 / r3 | Trained r1 / r2 / r3 | Base mean | Trained mean | Diff |
|---|---|---|---|---:|---:|---:|
| `07b4e4050840-s47` | KOMA | 0.319 / 0.407 / 0.452 | 0.390 / 0.500 / 0.364 | 0.3926 | 0.4181 | +0.0255 |
| `4528c31805f3-s47` | KJAX | 0.620 / 0.582 / 0.582 | 0.629 / 0.612 / 0.661 | 0.5943 | 0.6339 | +0.0395 |
| `5a86424e2861-s47` | KJAX | 0.342 / 0.461 / 0.461 | 0.453 / 0.441 / 0.481 | 0.4212 | 0.4584 | +0.0372 |
| `660f38096246-s47` | KTOP | 0.460 / 0.509 / 0.591 | 0.308 / 0.628 / 0.628 | 0.5202 | 0.5211 | +0.0009 |
| `6d3f0a5005fd-s47` | KTOP | 0.431 / 0.513 / 0.436 | 0.432 / 0.434 / 0.432 | 0.4602 | 0.4326 | -0.0275 |
| `93b53359f60e-s47` | KDDC | 0.595 / 0.597 / 0.406 | 0.363 / 0.595 / 0.582 | 0.5325 | 0.5135 | -0.0190 |
| `9b1cffc36b20-s47` | KDDC | 0.317 / 0.499 / 0.441 | 0.307 / 0.422 / 0.322 | 0.4193 | 0.3505 | -0.0687 |
| `c90f3d90f654-s47` | KOMA | 0.220 / 0.250 / 0.325 | 0.344 / 0.313 / 0.275 | 0.2653 | 0.3108 | +0.0455 |
| `d50e1051dcec-s47` | KABQ | 0.455 / 0.465 / 0.465 | 0.472 / 0.455 / 0.455 | 0.4614 | 0.4607 | -0.0007 |
| `e1375aee7217-s47` | KABQ | 0.328 / 0.313 / 0.340 | 0.299 / 0.291 / 0.280 | 0.3270 | 0.2899 | -0.0372 |

## Training

204 rollouts were generated for 30 optimizer steps. Steps whose row shows 12 rollouts were generated twice: the first job died at step 15 and training resumed from checkpoint 10, so those steps were sampled again (only step 11's two groups come from the same checkpoint). Rewards are the training grade (fast evaluator); zero means an infeasible, invalid or missing submission.

| Steps | Mean reward | Rollouts | Zero reward | Turn limit | Submitted without evaluating | Budget cut |
|---|---:|---:|---:|---:|---:|---:|
| 1-10 | 0.448 | 60 | 4 | 2 | 3 | 0 |
| 11-20 | 0.523 | 84 | 6 | 3 | 3 | 0 |
| 21-30 | 0.556 | 60 | 2 | 0 | 2 | 0 |

| Step | Reward | Reward std | Grad norm | Entropy | Loss | LR | Mean completion tokens | Seconds | Rollouts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.359 | 0.185 | 0.037 | 0.109 | 0.1200 | 3.0e-05 | 4810 | 2213 | 6 |
| 2 | 0.461 | 0.097 | 0.049 | 0.122 | 0.0297 | 2.9e-05 | 4740 | 1986 | 6 |
| 3 | 0.246 | 0.122 | 0.034 | 0.145 | -0.3127 | 2.8e-05 | 4248 | 1844 | 6 |
| 4 | 0.512 | 0.112 | 0.058 | 0.138 | -0.1066 | 2.7e-05 | 4399 | 1826 | 6 |
| 5 | 0.396 | 0.203 | 0.023 | 0.170 | -0.3652 | 2.6e-05 | 3144 | 1211 | 6 |
| 6 | 0.458 | 0.021 | 0.042 | 0.154 | 0.0252 | 2.5e-05 | 5516 | 1919 | 6 |
| 7 | 0.515 | 0.078 | 0.058 | 0.134 | -0.0442 | 2.4e-05 | 4222 | 1713 | 6 |
| 8 | 0.474 | 0.028 | 0.037 | 0.152 | -0.1870 | 2.3e-05 | 5094 | 2178 | 6 |
| 9 | 0.380 | 0.196 | 0.030 | 0.095 | 0.2301 | 2.2e-05 | 6127 | 2148 | 6 |
| 10 | 0.683 | 0.113 | 0.071 | 0.152 | 0.2928 | 2.1e-05 | 2608 | 1209 | 6 |
| 11 | 0.461 | 0.037 | 0.038 | 0.097 | -0.0028 | 2.0e-05 | 5124 | 2378 | 12 |
| 12 | 0.627 | 0.057 | 0.134 | 0.246 | -0.1019 | 1.9e-05 | 2140 | 1101 | 12 |
| 13 | 0.537 | 0.269 | 0.036 | 0.170 | -0.3812 | 1.8e-05 | 3261 | 1632 | 12 |
| 14 | 0.705 | 0.096 | 0.058 | 0.231 | 0.1670 | 1.7e-05 | 2726 | 1218 | 12 |
| 15 | 0.290 | 0.028 | 0.066 | 0.136 | 0.1503 | 1.6e-05 | 3784 | 1575 | 6 |
| 16 | 0.487 | 0.088 | 0.042 | 0.180 | -0.0314 | 1.5e-05 | 4328 | 2041 | 6 |
| 17 | 0.646 | 0.233 | 0.078 | 0.199 | -0.2822 | 1.4e-05 | 2831 | 1368 | 6 |
| 18 | 0.826 | 0.058 | 0.050 | 0.116 | -0.0531 | 1.3e-05 | 4078 | 1694 | 6 |
| 19 | 0.251 | 0.047 | 0.057 | 0.117 | -0.0290 | 1.2e-05 | 6666 | 2575 | 6 |
| 20 | 0.395 | 0.242 | 0.055 | 0.228 | -0.3556 | 1.1e-05 | 3534 | 1496 | 6 |
| 21 | 0.410 | 0.203 | 0.045 | 0.239 | -0.3316 | 1.0e-05 | 3766 | 1771 | 6 |
| 22 | 0.723 | 0.037 | 0.053 | 0.141 | 0.0949 | 9.0e-06 | 3353 | 1586 | 6 |
| 23 | 0.686 | 0.055 | 0.051 | 0.137 | 0.0198 | 8.0e-06 | 5084 | 1936 | 6 |
| 24 | 0.668 | 0.246 | 0.080 | 0.184 | -0.2863 | 7.0e-06 | 2235 | 1063 | 6 |
| 25 | 0.493 | 0.028 | 0.045 | 0.143 | 0.0753 | 6.0e-06 | 4484 | 2070 | 6 |
| 26 | 0.563 | 0.036 | 0.037 | 0.096 | -0.0115 | 5.0e-06 | 3696 | 1663 | 6 |
| 27 | 0.548 | 0.048 | 0.100 | 0.187 | 0.0414 | 4.0e-06 | 2403 | 1222 | 6 |
| 28 | 0.553 | 0.092 | 0.053 | 0.122 | -0.0508 | 3.0e-06 | 5028 | 1808 | 6 |
| 29 | 0.487 | 0.135 | 0.062 | 0.124 | 0.1832 | 2.0e-06 | 4786 | 1641 | 6 |
| 30 | 0.424 | 0.217 | 0.075 | 0.243 | -0.3980 | 1.0e-06 | 1624 | 886 | 6 |

Peak device memory 105.0 GB reserved by torch; mean 1450 s per step in the last chunk; endings over all rollouts: {"submitted": 199, "turn_limit": 5}.

## Cost

Every HF job of this project, billed per second at $5 per H200 hour: 46 jobs, $163.69 in all. 1 job(s) have no recorded duration and count as zero; the frontier panel's API cost is separate.

| Phase | Jobs | Not completed | Hours | USD |
|---|---:|---:|---:|---:|
| full-30 grpo | 6 | 2 | 16.49 | 82.46 |
| full-30 eval | 1 | 0 | 6.29 | 31.46 |
| full-30 base-eval | 1 | 0 | 6.08 | 30.39 |
| smoke smoke-evalrules-c1 | 1 | 1 | 1.27 | 6.33 |
| smoke smoke-grade | 1 | 0 | 1.06 | 5.32 |
| smoke smoke-evalrules-d1 | 1 | 0 | 0.81 | 4.05 |
| smoke smoke-flatol | 1 | 1 | 0.40 | 2.02 |
| smoke smoke-evalrules-a1 | 1 | 1 | 0.09 | 0.47 |
| smoke smoke-cu129 | 1 | 1 | 0.04 | 0.21 |
| smoke smoke-wp215 | 1 | 1 | 0.04 | 0.20 |
| full-30 export | 1 | 0 | 0.03 | 0.14 |
| smoke smoke-v1 | 1 | 1 | 0.03 | 0.14 |
| node probes (1-second jobs) | 22 | 0 | 0.02 | 0.11 |
| smoke smoke-evalrules-b2 | 1 | 1 | 0.01 | 0.07 |
| smoke smoke-evalrules-b1 | 1 | 1 | 0.01 | 0.07 |
| smoke smoke-evalrules-b4 | 1 | 1 | 0.01 | 0.07 |
| smoke smoke-evalrules-b3 | 1 | 1 | 0.01 | 0.07 |
| smoke smoke-evalrules | 1 | 1 | 0.01 | 0.07 |
| smoke smoke-wp215b | 1 | 1 | 0.01 | 0.04 |
| smoke smoke-wp214 | 1 | 1 | 0.00 | 0.00 |

## What went wrong, and the fixes

- Some H200 hosts (driver 580.178.04) start the container without CUDA libraries. The GPU guard now reports it in a minute and every launch is gated on a one-second probe job landing on a good host (`train/cloud/ops/run-full.sh`).
- TRL computes the logits of the whole completion at once. With a 248,320-token vocabulary a 27k-token completion needs 27 GB of fp32 logits and the first full-length step ran out of memory. `train/cloud/chunked_logps.py` computes log-probabilities in 2,048-token chunks under activation checkpointing; a CPU test shows the values and gradients equal TRL's.
- The first training job died at step 15 on the device watchdog: 135 GB in use on the device while torch reported 118 GB reserved. The cause of the gap is not known. Training resumed from checkpoint 10 in 5-step jobs (`--restore --stop-after-step N`) and the gap never grew that large again.

## Reading

Three guesses, none established. The base model already submits a feasible design in every held-out episode, so the failure modes training could remove on the train tasks (turn-limit endings and submitting without evaluating) had nothing to fix on the eval tasks. The skill that would move the score, finding cheaper feasible designs in 30 evaluations, got a weak signal: little reward spread inside many groups, 180 rollouts in the final lineage, and a learning rate near zero by step 30. And the held-out stations have weather the model never trained on.

## Next experiment

If this is run again: start from a supervised fit on the frontier trajectories of the train split, keep the learning rate constant, use 8 to 16 rollouts per step for a usable spread, train for 100 or more steps on more than 30 tasks, and evaluate checkpoints along the way. At the measured pace that is roughly $400 to $600 of H200 time.
