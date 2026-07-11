# Step 15 (B2) B Pair-Decomposition Benchmark

> Three layer-pair decomposition vs circular3 baseline. No hurdle/net-flow/ensemble/vessel-scaling. Not submitted.

Run command: `python src/step15_b_pair_decomposition_benchmark.py`

## 1. Boundary-safe design
- 9 outer scenarios; 63 inner folds; train-end 23:00 excluded; 2018-01-24 23:00 right-censored. Pair model uses COMPLETE training dates only (outer train_end-1; inner valid_date-2, >=5 dates); circular3 baseline uses the step10 observable-hour rule. No validation AIS / stock / vessel count used.

## 2. Step 10 baseline reproduction
- circular3_independent reproduces step10 pair_hour_circular_cv/independent with max abs diff < 1e-10 (asserted in code). **PASS**

## 3. Fixed pair decomposition

| method | rolling med MSE | final SSE | nonoverlap MSE |
| --- | --- | --- | --- |
| circular3_independent | 0.895 | 766 | 0.947 |
| pair_default_independent | 0.897 | 760 | 0.939 |
| pair_default_hierarchical | 0.929 | 770 | 0.969 |

## 4. Global vs by-pair CV

| method | rolling med | rolling max | final | nonoverlap |
| --- | --- | --- | --- | --- |
| pair_global_cv_independent | 0.896 | 1.012 | 754 | 0.955 |
| pair_global_cv_hierarchical | 0.939 | 0.971 | 766 | 0.952 |
| pair_bypair_cv_independent | 0.903 | 1.003 | 754 | 0.952 |
| pair_bypair_cv_hierarchical | 0.938 | 0.991 | 773 | 0.949 |

## 5-7. Selected parameters (final_analog, independent)

| scope | pair | daily | profile | tau |
| --- | --- | --- | --- | --- |
| by_pair | core_near | median | flow_weighted_triangular3 | 20 |
| by_pair | core_outer | long_mean | day_equal_triangular3 | 10 |
| by_pair | near_outer | median | flow_weighted_triangular3 | 20 |
| global | all | median | flow_weighted_triangular3 | 20 |

## 8. Independent vs hierarchical rounding
- See decision_table rolling/final/nonoverlap columns for pair_default and pair_global_cv under both roundings.

## 9. Forecast horizon
- horizon_1_mse and horizon_7_mse in decision_table.

## 10. Non-overlap blocks
- nonoverlap_combined_mse in decision_table (fold_01 + fold_08 targets, non-overlapping).

## 11. Decision for B3
- Judged from the tables above against the pass criteria (>=5 rolling folds beat circular3; final not worse; nonoverlap improved; max MSE not >5% worse; >=4 of 6 directions improved; core->near and near->core not both clearly worse).
