# Step 29 A Dynamic State Benchmark

Run command: `python src/step29_a_dynamic_state_benchmark.py`

## A-stock alignment (normal dates)
- Aligned: 5386/5513 = 0.977
- Misaligned: 127

## Oracle stock results
- oracle_true_stock_fixed_emission: SSE=3496, reduction=9.7%
- oracle_true_stock_daily_factor: SSE=3275, reduction=15.4%

## Block scores

| block | method | SSE | reduction |
| --- | --- | --- | --- |
| block_target_like | exact_step11_baseline | 3347 | 0.0% |
| block_target_like | markov_openloop | 132875 | -3870.0% |
| block_target_like | markov_mean_reverting | 3310 | 1.1% |
| block_target_like | last_day_residual_decay | 3480 | -4.0% |
| block_target_like | recent3_residual_decay | 3262 | 2.5% |
| block_target_like | oracle_true_stock_fixed_emission | 3496 | -4.5% |
| block_post_outage_stress | exact_step11_baseline | 3872 | 0.0% |
| block_post_outage_stress | markov_openloop | 218747 | -5549.5% |
| block_post_outage_stress | markov_mean_reverting | 3923 | -1.3% |
| block_post_outage_stress | last_day_residual_decay | 4087 | -5.6% |
| block_post_outage_stress | recent3_residual_decay | 4121 | -6.4% |
| block_post_outage_stress | oracle_true_stock_fixed_emission | 3496 | 9.7% |

## Decision
- Deployable methods passing 5%: 0

- Oracle stock reduction: 9.7% — PASS