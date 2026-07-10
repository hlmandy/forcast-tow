# Step 10 B Shrinkage and Calibration

> Task B only. Nested inner-CV shrinkage/calibration/integerization. No Ridge/Poisson/trees/NN, no submission, no A.

Run command: `python src/step10_b_shrinkage_calibration.py`

## 1. Boundary-safe nested evaluation

- 9 outer evaluation scenarios; 63 inner time-order folds (3..11 per scenario). Inner-train boundary 23:00 labels excluded; inner-validation day 23:00 is scoreable; outer-train last day is never an inner-validation day. Outer validation state never read. 2018-01-24 23:00 stays right-censored. Parameter selection uses integer-prediction SSE.

## 2. Step 09 consistency checks

- pair_hour_mean, pair_hour_fixed085, pair_recent14_hour, present_shrunk20 (all independent round) reproduce Step 09 within 1e-10. 214272 prediction rows complete. **PASS**

## 3. Fixed 0.85 versus selected global scale

- pair_hour_mean: rolling med MSE 0.922, final SSE 793.
- pair_hour_fixed085: rolling med MSE 0.908, final SSE 760.
- pair_hour_hier_tau_scale_cv: rolling med MSE 0.900, final SSE 742.
- pair_hour_circular_scale_cv: rolling med MSE 0.899, final SSE 764.
- Selected global scale distribution (independent): {0.9: 15, 1.05: 7, 0.8: 2, 0.95: 2, 1.0: 1}.

## 4. Hour hierarchy

- pair_hour_mean: rolling med MSE 0.922, final SSE 793.
- pair_hour_hier_tau_cv: rolling med MSE 0.913, final SSE 750.
- Selected tau distribution: {50.0: 4, 20.0: 4, 10.0: 1}.

## 5. Circular hour smoothing

- Selected smoother distribution: {'triangular3': 9}.
- pair_hour_mean: rolling med MSE 0.922, final SSE 793.
- pair_hour_circular_cv: rolling med MSE 0.895, final SSE 766.

## 6. Global versus direction-specific scale

- pair_hour_hier_tau_scale_cv: rolling med MSE 0.900, rolling max MSE 1.040, final SSE 742.
- pair_hour_direction_scale_cv: rolling med MSE 0.908, rolling max MSE 1.031, final SSE 762.
- present_shrunk20_scale_cv: rolling med MSE 0.909, rolling max MSE 0.999, final SSE 760.
- present_shrunk20_direction_scale_cv: rolling med MSE 0.913, rolling max MSE 1.032, final SSE 758.

## 7. Independent versus paired rounding

| predictor | ind roll med MSE | paired roll med MSE | paired better folds | ind better folds | ties |
| --- | --- | --- | --- | --- | --- |
| pair_hour_mean | 0.922 | 0.962 | 0 | 8 | 0 |
| pair_hour_fixed085 | 0.908 | 0.911 | 5 | 3 | 0 |
| pair_recent14_hour | 0.917 | 0.962 | 0 | 8 | 0 |
| pair_recent14_fixed085 | 0.908 | 0.911 | 4 | 4 | 0 |
| pair_hour_hier_tau_cv | 0.913 | 0.915 | 6 | 2 | 0 |
| pair_hour_hier_tau_scale_cv | 0.900 | 0.893 | 4 | 4 | 0 |
| pair_hour_circular_cv | 0.895 | 0.922 | 1 | 6 | 1 |
| pair_hour_circular_scale_cv | 0.899 | 0.939 | 3 | 5 | 0 |
| present_shrunk20 | 0.911 | 0.937 | 0 | 8 | 0 |
| present_shrunk20_scale_cv | 0.909 | 0.917 | 3 | 5 | 0 |
| pair_hour_direction_scale_cv | 0.908 | 0.924 | 3 | 5 | 0 |
| present_shrunk20_direction_scale_cv | 0.913 | 0.938 | 3 | 5 | 0 |

## 8. Direct hierarchy versus present structure

- raw present and direction-hour-mean are algebraically close, so only shrinkage/calibration value is judged; linkable models are not discussed.

## 9. Hyperparameter stability

- tau selections (hier): {50.0: 4, 20.0: 4, 10.0: 1}.
- smoother selections: {'triangular3': 9}.

## 10. Decision for final B candidates

- Top-5 final_analog variants:
  - pair_hour_hier_tau_scale_cv / independent: SSE=742
  - pair_hour_hier_tau_cv / independent: SSE=750
  - pair_hour_hier_tau_scale_cv / paired: SSE=750
  - pair_hour_circular_scale_cv / paired: SSE=752
  - pair_hour_hier_tau_cv / paired: SSE=755
- At most 3 B candidates are retained, judged across rolling folds + final_analog (not a single window).
- When training on all 2018-01-01..01-24 for the real 01-25..01-31 forecast, 2018-01-24 23:00 stays right-censored (no 01-25 00:00 state), and no validation AIS state is used.
