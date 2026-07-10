# Step 04 A Final-Analog Audit

> Task A only. No new models, no B task, no submission files. Reuses Step 03 building blocks.

Run command: `python src/step04_a_final_analog_audit.py`

## 1. Why the Step 03 overall ranking is insufficient

The eight 7-day validation windows overlap heavily and every one of them covers the 2018-01-13..01-18 data-quality anomaly. The same anomalous validation dates are therefore counted multiple times, and the early folds are dominated by outage/severe days. Step 03's overall median SSE consequently reflects performance on anomalous targets and cannot by itself decide the final training policy.

## 2. Final-analog design

The final-analog fold trains on 2018-01-01~2018-01-18 (18d, which fully contains the china_coastal outage and the initial recovery) and validates on 2018-01-19~2018-01-24 (6d — the available data end on 01-24, so the window is not padded to 7). All six validation dates are `normal` in the Step 02 audit.

Leakage controls: prediction uses only the validation dates' `unique_vessel_count`; validation AIS record counts, china_coastal record counts, quality variables, and A labels are not used in prediction. The training quality class is recomputed from only 2018-01-01~2018-01-18. For the `all` policy the weighted prediction reproduces `optimized_baseline.predict_base` with max abs diff < 1e-10 (asserted). All predictions are non-negative and rounded predictions are integers.

## 3. Training-quality policy comparison

Median `sse_rounded` across the 8 methods, and best method, under each policy on three normal-target views:

| view | all (median / best sse) | exclude_outage_severe (median / best) | quality_weighted (median / best) |
| --- | --- | --- | --- |
| final_analog (sse) | 4906.0 / 4510.0 (hour_mean) | 4040.5 / 3928.0 (hour_mean) | 4007.0 / 3880.0 (hour_mean) |
| fold_08 normal (sse) | 4756.5 / 4498.0 (hour_mean) | 4040.5 / 3928.0 (hour_mean) | 4002.0 / 3866.0 (hour_mean) |
| all-fold normal (mse) | 10.1 / 9.7 (hour_scale_p025) | 9.1 / 8.8 (hour_scale_p025) | 8.9 / 8.8 (hour_mean) |

- Per-method best policy on final_analog: {'quality_weighted': 5, 'exclude_outage_severe': 3}.
- Per-method best policy on fold_08 normal dates: {'quality_weighted': 5, 'exclude_outage_severe': 3}.
- These normal-target views (not the Step 03 overall median) are the basis for judging whether excluding or down-weighting the anomalous training days helps when future targets are normal.

## 4. Daily vessel-count scaling comparison

`hour_mean` vs `hour_scale_p025` vs `hour_scale_p05` (lower is better). Scaling is worth keeping only if p025/p05 beat hour_mean on a majority of these normal-target views.

**final_analog (sse)**

| policy | hour_mean | hour_scale_p025 | hour_scale_p05 |
| --- | --- | --- | --- |
| all | 4510.0 | 4614.0 | 4601.0 |
| exclude_outage_severe | 3928.0 | 3957.0 | 3961.0 |
| quality_weighted | 3880.0 | 3954.0 | 4022.0 |

**fold_08 normal (sse)**

| policy | hour_mean | hour_scale_p025 | hour_scale_p05 |
| --- | --- | --- | --- |
| all | 4498.0 | 4528.0 | 4619.0 |
| exclude_outage_severe | 3928.0 | 3957.0 | 3961.0 |
| quality_weighted | 3866.0 | 3926.0 | 4000.0 |

**all-fold normal (mse)**

| policy | hour_mean | hour_scale_p025 | hour_scale_p05 |
| --- | --- | --- | --- |
| all | 9.8 | 9.7 | 9.8 |
| exclude_outage_severe | 8.9 | 8.8 | 8.8 |
| quality_weighted | 8.8 | 8.8 | 8.9 |

## 5. Error by region

Per-(policy, method) region SSE on the final-analog fold (region errors are reported per model, not averaged across models):

| policy | method | core | near | outer | total |
| --- | --- | --- | --- | --- | --- |
| all | dow_hour_mean | 3605 | 1497 | 540 | 5642 |
| all | global_mean | 3469 | 1986 | 387 | 5842 |
| all | hour_mean | 2691 | 1416 | 403 | 4510 |
| all | hour_scale_p025 | 2753 | 1444 | 417 | 4614 |
| all | hour_scale_p05 | 2742 | 1444 | 415 | 4601 |
| all | mixed_075_hour_dow | 2877 | 1410 | 405 | 4692 |
| all | recent_14d_hour | 3035 | 1602 | 483 | 5120 |
| all | recent_7d_hour | 4907 | 2378 | 613 | 7898 |
| exclude_outage_severe | dow_hour_mean | 3185 | 1262 | 466 | 4913 |
| exclude_outage_severe | global_mean | 2777 | 1650 | 387 | 4814 |
| exclude_outage_severe | hour_mean | 2315 | 1236 | 377 | 3928 |
| exclude_outage_severe | hour_scale_p025 | 2314 | 1270 | 373 | 3957 |
| exclude_outage_severe | hour_scale_p05 | 2292 | 1296 | 373 | 3961 |
| exclude_outage_severe | mixed_075_hour_dow | 2422 | 1180 | 379 | 3981 |
| exclude_outage_severe | recent_14d_hour | 2371 | 1332 | 397 | 4100 |
| exclude_outage_severe | recent_7d_hour | 5259 | 2268 | 665 | 8192 |
| quality_weighted | dow_hour_mean | 3172 | 1293 | 449 | 4914 |
| quality_weighted | global_mean | 2777 | 1650 | 387 | 4814 |
| quality_weighted | hour_mean | 2297 | 1212 | 371 | 3880 |
| quality_weighted | hour_scale_p025 | 2315 | 1272 | 367 | 3954 |
| quality_weighted | hour_scale_p05 | 2315 | 1342 | 365 | 4022 |
| quality_weighted | mixed_075_hour_dow | 2358 | 1204 | 379 | 3941 |
| quality_weighted | recent_14d_hour | 2329 | 1280 | 383 | 3992 |
| quality_weighted | recent_7d_hour | 4717 | 2288 | 645 | 7650 |

## 6. Decision for the next modeling stage

Top-3 (policy, method) on the final-analog fold:
- quality_weighted / hour_mean: sse=3880, bias=-0.27
- exclude_outage_severe / hour_mean: sse=3928, bias=-0.25
- quality_weighted / mixed_075_hour_dow: sse=3941, bias=-0.28

- Scaling vs hour_mean (policy=all, normal-target views where the scaler wins): hour_scale_p025=1/3, hour_scale_p05=0/3. Keep the scaler only if it wins a majority.
- Candidate training policies and methods should be carried forward based on the normal-target views in sections 3-4, not the Step 03 overall median. The decision table (decision_table.csv) lists the separate rankings without any composite score or single champion.
- This stage does not develop GLM / GAM / LightGBM / neural-net or any new model.
