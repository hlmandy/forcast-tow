# Step 41 B Directional Hurdle Badcase

Run command: `python src/step41_b_directional_hurdle_badcase.py`

## Direction diagnostics

| direction | zero_rate | SSE | FP_share | FN_share | mag_share | top20_share |
| --- | --- | --- | --- | --- | --- | --- |
| core->near | 0.520 | 3037 | 26.6% | 1.7% | 71.7% | 15.0% |
| near->core | 0.440 | 2407 | 34.0% | 1.8% | 64.2% | 19.0% |
| near->outer | 0.781 | 925 | 12.2% | 75.0% | 12.8% | 31.2% |
| outer->near | 0.740 | 926 | 23.3% | 57.9% | 18.8% | 24.0% |
| core->outer | 0.839 | 434 | 5.3% | 83.6% | 11.1% | 31.1% |
| outer->core | 0.852 | 430 | 0.0% | 100.0% | 0.0% | 18.6% |

## Hurdle vs Direct (target_like)

- core->near direct_long_mean: -3.2%
- core->near hurdle_long_mean: -3.2%
- near->core direct_long_mean: -3.7%
- near->core hurdle_long_mean: -3.7%
- near->outer direct_long_mean: 7.8%
- near->outer hurdle_long_mean: 7.8%
- outer->near direct_long_mean: 7.6%
- outer->near hurdle_long_mean: 7.6%
- core->outer direct_long_mean: -1.5%
- core->outer hurdle_long_mean: -1.5%
- outer->core direct_long_mean: 16.9%
- outer->core hurdle_long_mean: 16.9%
