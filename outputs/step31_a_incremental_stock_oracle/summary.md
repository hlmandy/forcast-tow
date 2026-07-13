# Step 31 Incremental Stock Oracle

Run command: `python src/step31_a_incremental_stock_oracle.py`

## MMSI alignment (fixed)
- Row-level same_region: 0.5265
- Match-any: 0.9838

## Block scores (key methods)

| block | method | SSE | reduction |
| --- | --- | --- | --- |
| block_pre_normal | exact_step11_baseline | 2729 | 0.0% |
| block_pre_normal | oracle_abs_long | 2461 | 9.8% |
| block_pre_normal | oracle_inc_common | 2535 | 7.1% |
| block_pre_normal | oracle_inc_region | 2530 | 7.3% |
| block_pre_normal | oracle_true_stock_true_daily_factor | 2175 | 20.3% |
| block_target_like | exact_step11_baseline | 3399 | 0.0% |
| block_target_like | oracle_abs_long | 3453 | -1.6% |
| block_target_like | oracle_inc_common | 3391 | 0.2% |
| block_target_like | oracle_inc_region | 3374 | 0.7% |
| block_target_like | oracle_true_stock_true_daily_factor | 3374 | 0.7% |
| block_post_outage_stress | exact_step11_baseline | 3872 | 0.0% |
| block_post_outage_stress | oracle_abs_long | 3834 | 1.0% |
| block_post_outage_stress | oracle_inc_common | 3715 | 4.1% |
| block_post_outage_stress | oracle_inc_region | 3700 | 4.4% |
| block_post_outage_stress | oracle_true_stock_true_daily_factor | 3775 | 2.5% |

## Decision
- Incremental common on target_like: 0.2%
- Incremental common on pre_normal: 7.1%
- True daily factor on target_like: 0.7%
- One-day origins improved: 7/11
- Passes stock prediction criteria: NO
