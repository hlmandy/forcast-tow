# Step 27 A Label Caliber and Source Audit

Run command: `python src/step27_a_label_caliber_source_audit.py`

## Duplicate rates
- Exact message: 0 (0.00%)
- Same vessel timestamp: 10243 (0.50%)
- Cross-source threshold groups: 20/6593 (0.3%)

## Label variants vs V0 (current)

| variant | different cells | % | max diff | total diff |
| --- | --- | --- | --- | --- |
| V1 | 6 | 0.3% | 1 | 4 |
| V2 | 6 | 0.3% | 1 | 4 |
| V3 | 12 | 0.7% | 1 | 8 |
| V4 | 6 | 0.3% | 1 | 4 |
| V5 | 25 | 1.4% | 2 | -20 |

## Planar vs haversine
- 902 reclassifications (0.04% of points)

## Time alignment
- 262 matched pairs, median diff -16.0s

## Source cohort stability

| cohort | n_vessels | normal daily mean | normal CV |
| --- | --- | --- | --- |
| ('china_coastal', 'e_globe_daily', 'f_globe_dynamic') | 71 | 26.1 | 0.106 |
| ('e_globe_daily', 'f_globe_dynamic') | 25 | 3.2 | 0.483 |
| ('china_coastal', 'e_globe_daily') | 4 | 1.3 | 0.387 |

## Decision summary

- **planar_vs_haversine**: keep_current — 0.04% of points reclassified
- **exact_duplicate_effect**: keep_current — 0 exact duplicate rows
- **timestamp_duplicate_effect**: keep_current — 10243 same-timestamp duplicates
- **cross_source_threshold_effect**: keep_current — 0.3% of qualified groups rely on cross-source merging
- **source_time_alignment**: no_action — see timestamp_alignment.csv
- **source_cohort_stability**: build_source_cohort_model — see cohort analysis
