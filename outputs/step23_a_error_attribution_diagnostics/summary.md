# Step 23 A Error Attribution Diagnostics

Run command: `python src/step23_a_error_attribution_diagnostics.py`

## Oracle counterfactual results (rolling mean SSE)

| method | rolling SSE | reduction % | final SSE | nonoverlap SSE |
| --- | --- | --- | --- | --- |
| baseline | 5462 | 0.0% | 3872 | 10843 |
| oracle_daily_total | 2732 | 48.1% | 3532 | 6168 |
| oracle_region_totals | 2756 | 47.5% | 3560 | 6112 |
| oracle_hour_profile | 5450 | 4.5% | 722 | 9647 |
| oracle_hour_totals | 1026 | 80.6% | 1258 | 2264 |

## Region-level error (baseline rolling total SSE)

| region | SSE |
| --- | --- |
| core | 28244 |
| near | 12136 |
| outer | 3314 |

## Signal stability
- Strong region-hour signals: 10
- Moderate: 1
- See signal_stability.csv for details.

## Daily total predictability
- R_d vs U_d correlation (normal): -0.377
- Step 22 already showed U_d vs H_d Pearson ~0.07.

## Modeling decision

| component | reduction ratio | recommended | priority |
| --- | --- | --- | --- |
| daily_total | 0.481 | robust_daily_level | high |
| region_allocation | 0.475 | no_action | low |
| hour_profile | 0.045 | no_action | low |
| region_hour_residual | 0.806 | cross_fitted_residual_calibration | medium |
| daily_total_auxiliary_features | n/a | weak_nonlinear_U_adjustment | medium |
