# Step 38 C Profile Closure Test

Run command: `python src/step38_a_C_profile_closure_test.py`

## Oracle scores

| block | Oracle | SSE | reduction |
| --- | --- | --- | --- |
| block_pre_normal | best_historical_profile_per_day | 2647 | 3.0% |
| block_pre_normal | convex_historical_profile | 2640 | 3.2% |
| block_target_like | best_historical_profile_per_day | 3153 | 7.2% |
| block_target_like | convex_historical_profile | 3134 | 7.8% |
| block_post_outage_stress | best_historical_profile_per_day | 3460 | 10.6% |
| block_post_outage_stress | convex_historical_profile | 3480 | 10.1% |

## Deployable safe mixing (target_like)

- reshape_long_a0.1: 0.0%
- reshape_long_a0.25: 0.0%
- reshape_long_a0.5: -1.2%
- reshape_long_a0.75: -1.9%
- reshape_long_a1.0: -1.9%
- reshape_recent6_a0.1: 0.0%
- reshape_recent6_a0.25: -0.3%
- reshape_recent6_a0.5: -2.3%
- reshape_recent6_a0.75: 0.1%
- reshape_recent6_a1.0: -1.8%
- reshape_recent3_a0.1: 0.0%
- reshape_recent3_a0.25: -1.4%
- reshape_recent3_a0.5: -2.5%
- reshape_recent3_a0.75: -6.3%
- reshape_recent3_a1.0: -11.4%
- exact_step11_baseline: 0.0%

## Closure decision
- best-historical Oracle >= 5%: YES
- convex hull >= 10%: NO
- deployable >= 2%: NO