# Step 03 A Rolling Backtest

> Diagnostic backtest for Task A only. No B task, no new models, no submission files.

Run command: `python src/step03_a_rolling_backtest.py`

## 1. Backtest design

Eight expanding-window folds. The training start is fixed at 2018-01-01 and the training end rolls forward by one day each fold; each validation window is the next 7 calendar days. A model in a given fold may use only data dated on or before that fold's training end.

| fold | train | valid |
| --- | --- | --- |
| fold_01 | 2018-01-01~2018-01-10 (10d) | 2018-01-11~2018-01-17 (7d) |
| fold_02 | 2018-01-01~2018-01-11 (11d) | 2018-01-12~2018-01-18 (7d) |
| fold_03 | 2018-01-01~2018-01-12 (12d) | 2018-01-13~2018-01-19 (7d) |
| fold_04 | 2018-01-01~2018-01-13 (13d) | 2018-01-14~2018-01-20 (7d) |
| fold_05 | 2018-01-01~2018-01-14 (14d) | 2018-01-15~2018-01-21 (7d) |
| fold_06 | 2018-01-01~2018-01-15 (15d) | 2018-01-16~2018-01-22 (7d) |
| fold_07 | 2018-01-01~2018-01-16 (16d) | 2018-01-17~2018-01-23 (7d) |
| fold_08 | 2018-01-01~2018-01-17 (17d) | 2018-01-18~2018-01-24 (7d) |

## 2. Leakage checks

- Validation A labels never enter prediction (they are only used to score after predicting).
- Validation AIS-quality variables (ais_record_count, china_coastal_record_count, audit_quality_regime) never enter prediction.
- The only exogenous variable used from the validation period is `unique_vessel_count` (it simulates the future daily vessel count that the competition provides).
- The training quality class (outage/severe/reduced/normal) is recomputed inside each fold from only that fold's training dates (`M_china_train` = median of non-zero china_coastal daily records within the fold's training span). The global Step 02 `quality_regime` is used only for post-hoc validation grouping, never for training or prediction.
- Consistency: for the `all` policy, the new weighted prediction function reproduces `optimized_baseline.predict_base` for all 8 folds and all 8 methods with max absolute difference < 1e-10. **PASS** (asserted in code).

## 3. Fold composition

Validation-day counts by global audit quality (post-hoc description only):

| fold | valid normal | reduced | severe | outage |
| --- | --- | --- | --- | --- |
| fold_01 | 1 | 1 | 1 | 4 |
| fold_02 | 0 | 1 | 2 | 4 |
| fold_03 | 1 | 0 | 2 | 4 |
| fold_04 | 2 | 0 | 2 | 3 |
| fold_05 | 3 | 0 | 2 | 2 |
| fold_06 | 4 | 0 | 2 | 1 |
| fold_07 | 5 | 0 | 2 | 0 |
| fold_08 | 6 | 0 | 1 | 0 |

## 4. Baseline ranking

All 24 (training_policy, method) combinations ranked by `median_sse_rounded` across the 8 folds. Prediction errors are integers (rounded), so `sse_rounded` is the ranking metric used for real submission.

| rank | policy | method | median | mean | worst | last fold | wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | all | hour_scale_p05 | 4857.0 | 4936.6 | 5741.0 | 4957.0 | 6 |
| 2 | exclude_outage_severe | hour_scale_p05 | 4901.0 | 4959.5 | 5741.0 | 4428.0 | 5 |
| 3 | all | hour_scale_p025 | 4920.0 | 5095.1 | 5965.0 | 4898.0 | 0 |
| 4 | quality_weighted | hour_scale_p05 | 5030.5 | 5030.4 | 5741.0 | 4482.0 | 2 |
| 5 | all | hour_mean | 5115.0 | 5312.6 | 6188.0 | 4905.0 | 0 |
| 6 | exclude_outage_severe | hour_scale_p025 | 5202.0 | 5174.5 | 5965.0 | 4494.0 | 0 |
| 7 | all | recent_14d_hour | 5290.5 | 5368.0 | 6188.0 | 5256.0 | 0 |
| 8 | exclude_outage_severe | recent_7d_hour | 5319.0 | 5507.5 | 6462.0 | 6462.0 | 0 |
| 9 | quality_weighted | hour_scale_p025 | 5366.0 | 5253.0 | 5965.0 | 4466.0 | 0 |
| 10 | all | recent_7d_hour | 5447.5 | 5572.8 | 7484.0 | 7484.0 | 0 |
| 11 | all | mixed_075_hour_dow | 5500.5 | 5562.5 | 6468.0 | 4985.0 | 0 |
| 12 | quality_weighted | recent_7d_hour | 5535.0 | 5555.2 | 6195.0 | 6195.0 | 0 |
| 13 | exclude_outage_severe | hour_mean | 5620.0 | 5479.6 | 6188.0 | 4559.0 | 0 |
| 13 | exclude_outage_severe | recent_14d_hour | 5620.0 | 5478.2 | 6188.0 | 4684.0 | 0 |
| 15 | quality_weighted | hour_mean | 5634.0 | 5464.4 | 6188.0 | 4494.0 | 0 |
| 15 | quality_weighted | recent_14d_hour | 5634.0 | 5472.1 | 6188.0 | 4587.0 | 0 |
| 17 | quality_weighted | mixed_075_hour_dow | 5684.0 | 5559.2 | 6468.0 | 4516.0 | 0 |
| 18 | exclude_outage_severe | mixed_075_hour_dow | 5737.0 | 5594.6 | 6468.0 | 4577.0 | 0 |
| 19 | all | global_mean | 5838.0 | 5877.0 | 6463.0 | 5653.0 | 0 |
| 20 | exclude_outage_severe | global_mean | 6035.5 | 6011.2 | 6463.0 | 5595.0 | 0 |
| 20 | quality_weighted | global_mean | 6035.5 | 6011.2 | 6463.0 | 5595.0 | 0 |
| 22 | quality_weighted | dow_hour_mean | 7210.0 | 7048.1 | 8578.0 | 5457.0 | 0 |
| 23 | exclude_outage_severe | dow_hour_mean | 7238.0 | 7065.5 | 8578.0 | 5456.0 | 0 |
| 24 | all | dow_hour_mean | 7757.5 | 7448.9 | 8578.0 | 6185.0 | 0 |

## 5. Effect of training-quality treatment

Median `sse_rounded` by method under each training policy:

| method | all | exclude_outage_severe | quality_weighted |
| --- | --- | --- | --- |
| global_mean | 5838.0 | 6035.5 | 6035.5 |
| hour_mean | 5115.0 | 5620.0 | 5634.0 |
| dow_hour_mean | 7757.5 | 7238.0 | 7210.0 |
| recent_7d_hour | 5447.5 | 5319.0 | 5535.0 |
| recent_14d_hour | 5290.5 | 5620.0 | 5634.0 |
| mixed_075_hour_dow | 5500.5 | 5737.0 | 5684.0 |
| hour_scale_p025 | 4920.0 | 5202.0 | 5366.0 |
| hour_scale_p05 | 4857.0 | 4901.0 | 5030.5 |

- Per-method best policy count (by median sse_rounded) — all: 6, exclude_outage_severe: 1, quality_weighted: 1.
- Whether excluding or down-weighting the outage/severe training days helps should be judged across methods and folds, not from a single fold.

## 6. Effect of daily vessel-count scaling

Median `sse_rounded` for the scaling family by policy:

| policy | hour_mean | hour_scale_p025 | hour_scale_p05 |
| --- | --- | --- | --- |
| all | 5115.0 | 4920.0 | 4857.0 |
| exclude_outage_severe | 5620.0 | 5202.0 | 4901.0 |
| quality_weighted | 5634.0 | 5366.0 | 5030.5 |

Mean squared rounded error on validation days by quality group (policy=all; folds overlap, so this is descriptive, not independent):

| method | normal | reduced | degraded |
| --- | --- | --- | --- |
| hour_mean | 9.754 | 9.257 | 11.162 |
| hour_scale_p025 | 9.735 | 9.250 | 10.420 |
| hour_scale_p05 | 9.785 | 9.368 | 9.828 |

- A single correlation is not causal evidence; these numbers only describe whether the daily vessel-count multiplier changes error across folds and quality groups.

## 7. Error by region and validation quality

Mean squared rounded error by region (policy=all, averaged across the 8 methods; folds overlap):

| region | MSE | mean bias |
| --- | --- | --- |
| core | 21.608 | 1.637 |
| near | 9.355 | 0.948 |
| outer | 2.648 | 0.268 |

Mean squared rounded error by validation-day audit quality (policy=all, averaged across methods; folds overlap):

| quality | MSE | mean bias | n_rows |
| --- | --- | --- | --- |
| normal | 11.059 | -0.842 | 12672 |
| reduced | 9.695 | 1.214 | 1152 |
| severe | 9.232 | 1.945 | 8064 |
| outage | 13.082 | 2.340 | 10368 |

- The 8 folds overlap heavily (each validation day appears in up to 7 folds), so they must not be treated as 8 independent samples. Aggregates here are descriptive only.

## 8. Conclusions for the next stage

- Best (policy, method) by median `sse_rounded`: **all / hour_scale_p05** (median 4857.0).
- The backtest framework is internally consistent (the `all` policy reproduces the existing baseline exactly), so cross-method comparisons on the same folds are meaningful.
- Training-quality treatment: the policy that is best for the most methods is **all**; confirm this is stable across folds before adopting it.
- Daily vessel-count scaling: judge from section 6 whether the multiplier improves error stably; do not infer causation from a single correlation.
- Hardest region to predict (highest MSE, policy=all): **core**.
- No new models are trained in this stage and no final prediction is produced.
