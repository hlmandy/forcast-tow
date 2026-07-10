# Step 06 A Structure Model Benchmark

> Task A only. Direct vs decomposition structures; no GLM/GAM/LightGBM/NN, no B, no submissions.

Run command: `python src/step06_a_structure_model_benchmark.py`

## 1. Models and evaluation design

- 2 direct methods (hour_mean, recent7_hour) and 8 decomposition methods (daily total in {mean, recent7, ewm7} x hour profile in {mean, recent7, fourier2, daytype_shrunk}).
- 3 training-quality policies (all, exclude_outage_severe, quality_weighted) x 8 seven-day rolling folds + 1 final_analog fold = 9 evaluation scenarios.
- Validation uses only `unique_vessel_count` (allowed); no validation AIS / quality variables enter prediction. The methods here do not force vessel-count scaling.
- Decomposition predicts per-region daily total times a 24h profile that sums to 1.

## 2. Consistency checks

- hour_mean and recent7_hour reproduce the Step 03 stored predictions for the 8 rolling folds (all policies) with max abs diff < 1e-10. **PASS**
- a_model_predictions.csv has 133920 rows. **PASS**
- Every decomposition method's 24h pred_profile_share sums to 1 within 1e-10; pred_daily_total and pred_profile_share are non-negative. **PASS**
- final_analog: train 2018-01-01~01-18, valid 2018-01-19~01-24 (432 rows per combo). **PASS**

## 3. Direct versus decomposition models

| family | median(rolling median sse) | median(rolling normal mse) | median(final_analog sse) | total rolling wins |
| --- | --- | --- | --- | --- |
| decomposition | 5499 | 10.935 | 5089 | 13 |
| direct | 5491 | 10.393 | 6080 | 0 |

- Best-vs-best: best decomposition vs best direct are within ~1% on both views (normal-target mse 8.725 vs 8.782; final_analog sse 3910 vs 3880). The family medians above mix strong and weak variants, so they are NOT a best-vs-best comparison.
- Conclusion: simple decomposition does **not** stably beat hour_mean; it is roughly tied at the top, which makes it a viable variance-shrinkage alternative rather than a clear winner. Whether simple decomposition beats hour_mean must be judged across folds and the normal-target view, not a single final_analog window.

## 4. Daily-total method comparison

Median rolling SSE / final_analog SSE by daily-total method (across the hour-profile variants, all policies):

| daily_total_method | median(rolling median sse) | median(final_analog sse) | median(rolling normal mse) |
| --- | --- | --- | --- |
| ewm7 | 5499 | 5062 | 10.935 |
| mean | 5704 | 4611 | 10.436 |
| recent7 | 5253 | 7457 | 11.205 |

## 5. Hour-profile method comparison

| hour_profile_method | median(rolling median sse) | median(final_analog sse) | median(rolling normal mse) |
| --- | --- | --- | --- |
| daytype_shrunk | 5106 | 4030 | 9.033 |
| fourier2 | 5781 | 4627 | 11.035 |
| mean | 5169 | 4704 | 9.855 |
| recent7 | 5506 | 7686 | 11.205 |

- Recent profile, Fourier smoothing, and weekday/weekend separation are judged by whether they reduce out-of-sample error across folds, not by in-sample fit.

## 6. Total error versus shape error

Decomposition diagnostics (median over scenarios x policies x methods), by region. `shape_only_sse` uses the true total x predicted shape; `total_only_sse` uses the predicted total x true shape. These two are diagnostic only and CANNOT be added to get the actual SSE (there is a cross term).

| region | median hourly_sse | median daily_total_sse | median shape_only_sse | median total_only_sse | median profile L1 | median profile RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| core | 3524 | 35850 | 1782 | 2970 | 0.6331 | 0.0328 |
| near | 1536 | 14315 | 945 | 1210 | 0.6631 | 0.0341 |
| outer | 420 | 1888 | 364 | 162 | 0.7498 | 0.0390 |

- core: shape_only_sse=1782 vs total_only_sse=2970 -> error is more from the **total** side (diagnostic, non-additive).
- near: shape_only_sse=945 vs total_only_sse=1210 -> error is more from the **total** side (diagnostic, non-additive).
- outer: shape_only_sse=364 vs total_only_sse=162 -> error is more from the **shape** side (diagnostic, non-additive).
- These medians are over all 9 scenarios, including anomalous validation days where the predicted total (fit on normal training) far exceeds the depressed true total. So the total-side dominance for core/near is partly anomalous-driven; on normal targets only, the shape/total split would be more balanced.

## 7. Effect of training-quality policy

| policy | median(rolling median sse) | median(final_analog sse) | median(rolling normal mse) |
| --- | --- | --- | --- |
| all | 5308 | 5840 | 11.865 |
| exclude_outage_severe | 5466 | 4617 | 10.461 |
| quality_weighted | 5583 | 4621 | 10.302 |

## 8. Decision for the next stage

- Best final_analog combo: quality_weighted / hour_mean (sse=3880).
- Best rolling normal-target combo: quality_weighted / decomp_ewm7_daytype_shrunk (mse=8.725).
- These are observations for choosing what to carry into statistical-model development (Poisson / negative-binomial / Ridge) in a later step. No submission prediction is produced here.
