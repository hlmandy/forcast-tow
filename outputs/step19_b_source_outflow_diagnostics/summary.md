# Step 19 B Source-Outflow Diagnostics

Run command: `python src/step19_b_source_outflow_diagnostics.py`

## 1. Motivation
- Step 18 confirmed core->near local replacement is effective; near->core replacement is harmful. Current best online B SSE = 1741. Reciprocal-pair decomposition did not improve overall. This step tests the source-outflow mechanism.

## 2. Source-outflow sparsity

| grouping | series | zero_rate | daily_cv | normal profile Pearson |
| --- | --- | --- | --- | --- |
| direction | core->near | 0.000 | 0.222 | 0.182 |
| direction | core->outer | 0.000 | 0.435 | 0.125 |
| direction | near->core | 0.000 | 0.228 | 0.095 |
| direction | near->outer | 0.000 | 0.400 | 0.116 |
| direction | outer->core | 0.000 | 0.370 | 0.030 |
| direction | outer->near | 0.000 | 0.345 | 0.138 |
| reciprocal_pair | core_near | 0.000 | 0.212 | 0.200 |
| reciprocal_pair | near_outer | 0.000 | 0.327 | 0.117 |
| reciprocal_pair | core_outer | 0.000 | 0.343 | 0.093 |
| source_outflow | core_outflow | 0.000 | 0.198 | 0.236 |
| source_outflow | near_outflow | 0.000 | 0.180 | 0.153 |
| source_outflow | outer_outflow | 0.000 | 0.288 | 0.101 |

## 3-4. Daily-total and hour-profile stability
- See grouping_comparison.csv and source_profile_stability.csv for full details.

## 5. Destination-choice stability

| source | global share (dest1) | share std across hours |
| --- | --- | --- |
| core | 0.834 | 0.208 |
| near | 0.763 | 0.128 |
| outer | 0.322 | 0.235 |

## 6. Stock-times-rate feasibility
- See source_stock_rate_summary.csv.

## 7. Comparison with reciprocal-pair grouping
- See grouping_comparison.csv for side-by-side comparison.

## 8. Decision for Step 20

| source | recommended_structure | reason |
| --- | --- | --- |
| core | source_total_profile_share | cv=0.198, pearson=0.236, share_var=0.208 |
| near | source_total_profile_share | cv=0.180, pearson=0.153, share_var=0.128 |
| outer | no_clear_advantage | cv=0.288, pearson=0.101, share_var=0.235 |
