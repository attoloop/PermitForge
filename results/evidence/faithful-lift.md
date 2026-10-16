# Published verifier comparison

Binding values below summarize the preserved native replays. Each test also compared every receptor,
feasibility and exact submitted cost; receipts retain source submissions, hashes and timings.

Reproduce this table: `python3 convert/report_lift.py > results/evidence/faithful-lift.md`.
Re-run native comparisons: `uv run --directory envs/permit/core pytest tests/test_lift_faithful.py --evidence-dir ../../../results/evidence/faithful-lift -q`.

| Instance | Standard | Published value | Lifted value | Match | Receptors compared |
|---|---|---:|---:|---|---:|
| 00409ef904a8-s47 | NO2_1h | 114.82219 | 114.82219 | yes | 184 |
| 00409ef904a8-s47 | NO2_annual | 5.30306 | 5.30306 | yes | 184 |
| 00409ef904a8-s47 | PM25_24h | 3.76069 | 3.76069 | yes | 184 |
| 00409ef904a8-s47 | PM25_annual_primary | 0.40286 | 0.40286 | yes | 184 |
| 00409ef904a8-s47 | PM25_annual_secondary | 0.40286 | 0.40286 | yes | 184 |
| 00409ef904a8-s47 | SO2_1h | 1.50703 | 1.50703 | yes | 184 |
| 00409ef904a8-s47 | SO2_3h | 1.50478 | 1.50478 | yes | 184 |
| 00409ef904a8-s47 | SO2_annual | 0.03468 | 0.03468 | yes | 184 |
| 04ef47cb1279-s47 | NO2_1h | 80.48220 | 80.48220 | yes | 184 |
| 04ef47cb1279-s47 | NO2_annual | 3.00799 | 3.00799 | yes | 184 |
| 04ef47cb1279-s47 | PM25_24h | 8.26384 | 8.26384 | yes | 184 |
| 04ef47cb1279-s47 | PM25_annual_primary | 1.65398 | 1.65398 | yes | 184 |
| 04ef47cb1279-s47 | PM25_annual_secondary | 1.65398 | 1.65398 | yes | 184 |
| 04ef47cb1279-s47 | SO2_1h | 115.23679 | 115.23679 | yes | 184 |
| 04ef47cb1279-s47 | SO2_3h | 101.84091 | 101.84091 | yes | 184 |
| 04ef47cb1279-s47 | SO2_annual | 4.68627 | 4.68627 | yes | 184 |
| 068cd7da58fd-s47 | NO2_1h | 115.24870 | 115.24870 | yes | 184 |
| 068cd7da58fd-s47 | NO2_annual | 9.57462 | 9.57462 | yes | 184 |
| 068cd7da58fd-s47 | PM25_24h | 3.87943 | 3.87943 | yes | 184 |
| 068cd7da58fd-s47 | PM25_annual_primary | 0.78512 | 0.78512 | yes | 184 |
| 068cd7da58fd-s47 | PM25_annual_secondary | 0.78512 | 0.78512 | yes | 184 |
| 068cd7da58fd-s47 | SO2_1h | 10.47002 | 10.47002 | yes | 184 |
| 068cd7da58fd-s47 | SO2_3h | 8.39924 | 8.39924 | yes | 184 |
| 068cd7da58fd-s47 | SO2_annual | 0.47790 | 0.47790 | yes | 184 |
| 4ef687ff4418-s47 | NO2_1h | 148.35975 | 148.35975 | yes | 184 |
| 4ef687ff4418-s47 | NO2_annual | 11.15392 | 11.15392 | yes | 184 |
| 4ef687ff4418-s47 | PM25_24h | 1.15618 | 1.15618 | yes | 184 |
| 4ef687ff4418-s47 | PM25_annual_primary | 0.23275 | 0.23275 | yes | 184 |
| 4ef687ff4418-s47 | PM25_annual_secondary | 0.23275 | 0.23275 | yes | 184 |
| 4ef687ff4418-s47 | SO2_1h | 0.75538 | 0.75538 | yes | 184 |
| 4ef687ff4418-s47 | SO2_3h | 0.66100 | 0.66100 | yes | 184 |
| 4ef687ff4418-s47 | SO2_annual | 0.03154 | 0.03154 | yes | 184 |
| ffaa74509907-s47 | NO2_1h | 132.15578 | 132.15578 | yes | 184 |
| ffaa74509907-s47 | NO2_annual | 6.11420 | 6.11420 | yes | 184 |
| ffaa74509907-s47 | PM25_24h | 1.19521 | 1.19521 | yes | 184 |
| ffaa74509907-s47 | PM25_annual_primary | 0.26597 | 0.26597 | yes | 184 |
| ffaa74509907-s47 | PM25_annual_secondary | 0.26597 | 0.26597 | yes | 184 |
| ffaa74509907-s47 | SO2_1h | 83.73895 | 83.73895 | yes | 184 |
| ffaa74509907-s47 | SO2_3h | 53.28275 | 53.28275 | yes | 184 |
| ffaa74509907-s47 | SO2_annual | 3.09515 | 3.09515 | yes | 184 |
