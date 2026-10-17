# PermitForge baselines

Best-known search cost uses the supplied fast fields and a final ARM2 native check. Search and baselines are diagnostics, independent of admission.

Means cover completed tasks only; missing tasks are excluded and counted. Infeasible/no-submit savings use sentinel -1 in JSON and contribute zero USD and zero % to means. Mean % saved is the mean of per-task percentages, not a ratio of sums. Evaluation counts exclude final native verification. Search seconds include it.

Reward is the feasible fraction of draft cost saved; failed submissions score zero.

Reproduce: `uv run --project envs/permit/core python convert/run_baselines.py --tasks envs/permit/tasks --report`.

## train

Completed tasks: 0 / 230; missing or incomplete: 230.

| policy | mean USD saved | mean % saved | feasible rate | mean reward | mean cost (feasible) |
|---|---:|---:|---:|---:|---:|
| draft | pending | pending | pending | pending | pending |
| all_none_at_gep | pending | pending | pending | pending | pending |
| uniform_moderate_at_gep | pending | pending | pending | pending | pending |
| cheapest_first_greedy | pending | pending | pending | pending | pending |
| random_30 | pending | pending | pending | pending | pending |
| v1_coordinate_descent | pending | pending | pending | pending | pending |
| best-known search | pending | pending | pending | pending | pending |

Search trees closed: 0 / 0. Closed describes the fast-field search; final feasibility is checked by the native verifier.

## eval

Completed tasks: 0 / 130; missing or incomplete: 130.

| policy | mean USD saved | mean % saved | feasible rate | mean reward | mean cost (feasible) |
|---|---:|---:|---:|---:|---:|
| draft | pending | pending | pending | pending | pending |
| all_none_at_gep | pending | pending | pending | pending | pending |
| uniform_moderate_at_gep | pending | pending | pending | pending | pending |
| cheapest_first_greedy | pending | pending | pending | pending | pending |
| random_30 | pending | pending | pending | pending | pending |
| v1_coordinate_descent | pending | pending | pending | pending | pending |
| best-known search | pending | pending | pending | pending | pending |

Search trees closed: 0 / 0. Closed describes the fast-field search; final feasibility is checked by the native verifier.

