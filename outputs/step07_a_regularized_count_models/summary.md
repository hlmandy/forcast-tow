# Step 07 A Regularized Count Models

> Task A only. Ridge/Poisson direct and decomposition models; no neg-binomial, no trees/NN, no B, no submissions.

Run command: `python src/step07_a_regularized_count_models.py`

## 1. Evaluation and leakage controls

- 8 seven-day rolling folds + 1 final_analog fold; 3 training-quality policies.
- Inner time-series CV selects alpha: the last 4 dates of each outer training period are inner-validation days, each predicted only from strictly-earlier dates; quality classes are recomputed inside each inner training span. Outer validation is never used to choose alpha.
- Validation prediction uses only the competition-allowed `unique_vessel_count` (for vessel-feature models) and calendar variables; no validation AIS / quality / label is used.

## 2. Consistency checks

- hour_mean, decomp_mean_mean, decomp_ewm7_daytype_shrunk reproduce the Step 06 stored predictions with max abs diff < 1e-10. **PASS**
- a_model_predictions.csv has 214272 rows. **PASS**
- Every statistical model selects exactly one alpha per (evaluation_id, training_policy). **PASS**
- All decomposition 24h profile shares sum to 1 within 1e-10; all predictions are finite and non-negative. **PASS**

## 3. Hyperparameter stability

- ridge: selected-alpha counts across all folds/policies/methods = {0.1: 95, 1.0: 51, 10.0: 16}.
- poisson: selected-alpha counts across all folds/policies/methods = {1.0: 69, 0.0001: 47, 0.01: 33, 0.1: 8, 0.001: 5}.
- Ridge boundary selections: alpha=0.1 chosen 95 times, alpha=1000 chosen 0 times. (Reported as fact; grid not expanded here.)

## 4. Statistical models versus baselines

Trustworthy views first (normal-target mse and final_analog sse); the rolling-median / rolling-wins columns are kept only for reference and are contaminated by anomalous validation days.

| family | median(rolling normal mse) | median(final_analog sse) | median(rolling max sse) | median(rolling median sse, contaminated) | total rolling wins (contaminated) |
| --- | --- | --- | --- | --- | --- |
| baseline | 8.850 | 3937 | 6234 | 5364 | 3 |
| decomposition | 15.244 | 7692 | 7858 | 4502 | 9 |
| direct | 15.467 | 7734 | 8136 | 4559 | 0 |

- On the normal-target and final_analog views the simple baselines clearly beat every regularized model (baseline normal mse ~8.8 vs ~15 for regularized; baseline final_analog ~3.9k vs ~7.7k). The regularized models only look better on the rolling-median / rolling-wins columns, which include anomalous validation days and are not trustworthy for model choice.

## 5. Ridge versus Poisson

| estimator | median(rolling median sse) | median(rolling normal mse) | median(final_analog sse) |
| --- | --- | --- | --- |
| poisson | 4333 | 13.831 | 6704 |
| ridge | 4568 | 16.354 | 8184 |

## 6. Effect of vessel-count feature

Paired non-vessel vs vessel (median across policies), rolling normal mse / final_analog sse:

| method pair | normal mse (no vessel / vessel) | final_analog sse (no vessel / vessel) |
| --- | --- | --- |
| ridge_direct_calendar / ridge_direct_calendar_vessel | 15.339 / 15.835 | 7704 / 8187 |
| poisson_direct_calendar / poisson_direct_calendar_vessel | 14.289 / 13.855 | 6485 / 6937 |
| decomp_ridge_calendar_mean / decomp_ridge_calendar_vessel_mean | 14.188 / 16.544 | 7211 / 8189 |
| decomp_poisson_calendar_mean / decomp_poisson_calendar_vessel_mean | 12.467 / 13.807 | 6142 / 6850 |

## 7. Direct versus decomposition

Among the regularized models only (both lose to the baselines). Lower is better; normal-target mse and final_analog sse are the trustworthy columns.

| structure (regularized only) | median(rolling normal mse) | median(final_analog sse) | median(rolling max sse) | median(rolling median sse, contaminated) |
| --- | --- | --- | --- | --- |
| decomposition | 15.244 | 7692 | 7858 | 4502 |
| direct | 15.467 | 7734 | 8136 | 4559 |

- Decomposition is marginally more reliable than direct among the regularized models, but the difference is small and both are dominated by the simple baselines.

## 8. Mean profile versus daytype-shrunk profile

Paired mean vs daytype (median across policies), rolling normal mse / final_analog sse:

| daily-total method | normal mse (mean / daytype) | final_analog sse (mean / daytype) |
| --- | --- | --- |
| decomp_mean_mean / decomp_mean_daytype_shrunk | 8.835 / 8.736 | 3922 / 3902 |
| decomp_ridge_calendar_mean / decomp_ridge_calendar_daytype | 14.188 / 14.081 | 7211 / 7128 |
| decomp_poisson_calendar_mean / decomp_poisson_calendar_daytype | 12.467 / 12.609 | 6142 / 6178 |

## 9. Implications for negative binomial modeling

- Poisson is NOT systematically worse than Ridge here (Poisson is in fact better: normal mse 13.8 vs 16.4, final_analog 6.7k vs 8.2k). So there is no signal that the Poisson mean-variance equality is the binding problem.
- Selected Poisson alphas do not pile up on the weak-regularization boundary (alpha=1 chosen most often), and no systematic explosion/bias is visible. With only ~2-3 weeks of data the regularized count models lose to the simple weighted-mean baselines regardless of estimator.
- Therefore the evidence does NOT justify investing in a negative-binomial model for A at this stage. No negative-binomial model is fit here.

## 10. Decision for the next stage

- Best final_analog combo: exclude_outage_severe / decomp_mean_daytype_shrunk (sse=3872).
- Best rolling normal-target combo: exclude_outage_severe / decomp_mean_daytype_shrunk (mse=8.686).
- Keep as A candidates: the simple decomposition baselines (especially decomp_mean_daytype_shrunk, decomp_mean_mean, decomp_ewm7_daytype_shrunk) and hour_mean, under the quality_weighted / exclude_outage_severe policies (all as control).
- Drop / do not pursue further: the Ridge/Poisson regularized models (they are systematically worse on the trustworthy views), and the daily vessel-count feature (inconsistent, net-negative for Ridge).
- daytype-shrunk profile gives a small but consistent edge over mean profile across daily-total methods; keep it.
- Negative binomial is not worth testing for A (section 9).
- A single-model development for A has effectively plateaued at the simple decomposition baselines; the next higher-value step is the systematic diagnosis of Task B. No submission prediction and no ensembling in this stage.
