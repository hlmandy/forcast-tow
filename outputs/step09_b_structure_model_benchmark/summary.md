# Step 09 B Structure Model Benchmark

> Task B leak-free structural benchmark. No Ridge/Poisson/trees/NN, no submission, no A.

Run command: `python src/step09_b_structure_model_benchmark.py`

## 1. Boundary-safe evaluation

- A B label at hour t depends on the same vessel's state at t+1.
- Each training period's last day 23:00 migration label was excluded from training (6 transition rows per scenario); that hour's source_present (a current-hour quantity) is still used for stock estimation.
- 2018-01-24 23:00 is right-censored (no 2018-01-25 00:00 state): fold_08 and final_analog each have 6 unscored validation rows; their y_true and error fields are blank, not 0.
- Earlier B backtest results may have been affected by this training-boundary leakage and by zero-filling the right-censored endpoint.

## 2. Models and leakage controls

- 5 direct models, 9 deployable state-structure models, 2 oracle diagnostics (deployable=False).
- Deployable models do NOT read validation source_present / source_linkable; only the vessel_p05 method uses the competition-provided validation daily vessel count. audit_quality_regime is not used in training (uniform training-date weight 1).

## 3. Direct baselines

| method | rolling median MSE | rolling max MSE | final_analog SSE |
| --- | --- | --- | --- |
| pair_global_mean | 0.983 | 1.057 | 800 |
| pair_hour_mean | 0.922 | 0.989 | 793 |
| pair_recent7_hour | 0.997 | 1.044 | 812 |
| pair_recent14_hour | 0.917 | 0.989 | 780 |
| pair_hour_mean_x085 | 0.908 | 1.006 | 760 |

## 4. Present versus linkable parameterization

| pair | present rolling med MSE / final SSE | linkable rolling med MSE / final SSE |
| --- | --- | --- |
| global_prob | 0.989 / 810 | 0.989 / 810 |
| hour_prob_raw | 0.922 / 793 | 0.924 / 793 |
| hour_prob_shrunk20 | 0.911 / 785 | 0.911 / 785 |
| recent7_stock_shrunk20 | 0.916 / 782 | 0.916 / 782 |

## 5. Effect of probability shrinkage

| pair | raw rolling med MSE / final SSE | shrunk20 rolling med MSE / final SSE |
| --- | --- | --- |
| present_hour_prob | 0.922 / 793 | 0.911 / 785 |
| linkable_hour_prob | 0.924 / 793 | 0.911 / 785 |
- Per-direction effect on the sparse directions (final_analog SSE):

## 6. Long-term versus recent stock

- present: long 785 vs recent7 782 (final_analog SSE).
- linkable: long 785 vs recent7 782 (final_analog SSE).

## 7. Effect of daily vessel count

- linkable_hour_prob_shrunk20: rolling med MSE 0.911, final SSE 785.
- +vessel_p05: rolling med MSE 0.921, final SSE 783.

## 8. Exposure forecast versus transition-probability error

- present_hour_prob_shrunk20: final SSE 785; oracle_present_hour_prob_shrunk20 (true exposure): final SSE 737.
- linkable_hour_prob_shrunk20: final SSE 785; oracle_linkable_hour_prob_shrunk20 (true exposure): final SSE 738.
- Oracle methods are excluded from ranking and win counts.

## 9. Performance during the A outage period

| method | normal | reduced | severe | outage |
| --- | --- | --- | --- | --- |
| pair_global_mean | 0.873 | 1.069 | 1.028 | 1.050 |
| pair_hour_mean | 0.880 | 1.028 | 0.912 | 0.968 |
| pair_recent7_hour | 0.918 | 1.104 | 1.016 | 1.026 |
| pair_recent14_hour | 0.868 | 1.028 | 0.907 | 0.968 |
| pair_hour_mean_x085 | 0.846 | 1.073 | 0.967 | 0.962 |
| present_global_prob | 0.881 | 1.080 | 1.036 | 1.054 |
| present_hour_prob_raw | 0.884 | 1.028 | 0.913 | 0.964 |
| present_hour_prob_shrunk20 | 0.863 | 1.031 | 0.914 | 0.961 |
| present_recent7_stock_shrunk20 | 0.857 | 1.017 | 0.938 | 0.961 |
| linkable_global_prob | 0.881 | 1.080 | 1.036 | 1.054 |
| linkable_hour_prob_raw | 0.880 | 1.031 | 0.913 | 0.968 |
| linkable_hour_prob_shrunk20 | 0.863 | 1.031 | 0.914 | 0.961 |
| linkable_recent7_stock_shrunk20 | 0.857 | 1.014 | 0.938 | 0.962 |
| linkable_hour_prob_shrunk20_vessel_p05 | 0.860 | 1.038 | 0.929 | 0.964 |
- audit_quality_regime is A's audit grouping, shown here only to observe B's behavior in the same periods; it is not a B training weight.

## 10. Decision for the next stage

- Best final_analog (deployable): pair_hour_mean_x085 (SSE=760).
- Best rolling median MSE (deployable): pair_hour_mean_x085 (0.908).
- These observations guide (not decide) Step 10. No submission is produced here.
