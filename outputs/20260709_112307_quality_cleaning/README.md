# Data Quality Cleaning Experiment

This run quantifies sensor anomalies, missing signals, speed outliers, duplicate timestamps, and trajectory jumps in the tug AIS training set.

## Key Findings

- Raw data contains real quality issues:
  - `true_heading` invalid: 151,898 rows
  - `rot` missing: 11,542 rows
  - `SOG > 30`: 269 rows
  - duplicate `mmsi + time` rows to drop: 10,243
  - inferred step speed `> 50 kn`: 28,305 rows
  - inferred step speed `> 100 kn`: 12,543 rows

- However, simple cleaning barely changes official hourly labels:
  - `drop_step_gt_50`: A changes 9 counts total, B changes 22 counts total.
  - `dedup_drop_step_gt_50`: A changes 12 counts total, B changes 34 counts total.
  - Maximum single row label change is only 1.

- Backtest using cleaned labels for training and raw official labels for validation barely improves:
  - raw mean score: 8373.79
  - `drop_step_gt_50` mean score: 8372.31

## Interpretation

Point-level anomalies exist and have been quantified, but the official hourly aggregation is robust to simple point deletion. The current online gap is unlikely to be fixed mainly by deleting sensor anomalies. Future gains should focus on prediction methodology: validation-period scale, hourly shape, daily vessel-count calibration, and per-region/per-direction modeling.

## Files

- `quality_diagnostics.csv`: anomaly counts.
- `clean_version_summary.csv`: removed rows by cleaning version.
- `label_diff_summary.csv`: A/B label differences versus raw labels.
- `top_label_changes.csv`: rows most affected by cleaning.
- `clean_train_raw_valid_backtest_summary.csv`: backtest when training on cleaned labels and validating on raw labels.
- `a_labels_*.csv`, `b_labels_*.csv`: generated labels for each cleaning version.

## Recommendation

Use raw official labels as the main training target. Keep `drop_step_gt_50` only as a weak robustness/ensemble variant, not as the primary solution.
