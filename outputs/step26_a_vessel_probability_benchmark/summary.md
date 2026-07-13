# Step 26 A Vessel Probability Benchmark

Run command: `python src/step26_a_vessel_probability_benchmark.py`

## Results (post-outage block, Jan 19-24)

| method | SSE | reduction | deployable |
| --- | --- | --- | --- |
| exact_step11_baseline | 3872 | 0.0% | True |
| occupancy_long | 3959 | -2.2% | True |
| occupancy_recent7 | 4119 | -6.4% | True |
| occupancy_long_recent_blend | 3967 | -2.5% | True |
| occupancy_logU_slope | 4013 | -3.6% | True |
| vessel_prob_global | 4092 | -5.7% | True |
| vessel_prob_long_tau10 | 4186 | -8.1% | True |
| vessel_prob_weekly_tau10 | 4166 | -7.6% | True |
| vessel_prob_recent_tau10 | 4641 | -19.9% | True |
| vessel_prob_long_tau10_no_new_pool | 4124 | -6.5% | True |
| oracle_true_present_set | 4050 | -4.6% | False |
| oracle_true_presence_prob | 4092 | -5.7% | False |

## Key findings
- Deployable methods passing 5%: 0
- Oracle best reduction: -4.6%
- New vessel ratio: 0.038
- Baseline exact reproduction: PASS (from step07 final_analog)
