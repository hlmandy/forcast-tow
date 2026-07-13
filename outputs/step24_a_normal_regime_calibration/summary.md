# Step 24 A Normal-Regime Calibration

Run command: `python src/step24_a_normal_regime_calibration.py`

## Results (normal-only aggregate reduction)

| method | agg_red_norm | mean_fold_all | normal_improve | final_norm_red | post_outage_red |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.000 | 0.000 | 0/8 | 0.000 | 0.000 |
| nested_region_hour_unconstrained | 0.002 | -0.001 | 2/8 | 0.000 | 0.002 |
| nested_region_hour_total_preserving | 0.002 | -0.001 | 2/8 | 0.000 | 0.002 |
| normal_common_profile_cv | 0.000 | 0.015 | 4/8 | 0.006 | 0.002 |
| normal_profile_plus_nested_residual | 0.002 | 0.014 | 5/8 | 0.006 | 0.004 |
| step23_fixed10_diagnostic | 0.025 | 0.055 | 6/8 | 0.028 | 0.023 |
