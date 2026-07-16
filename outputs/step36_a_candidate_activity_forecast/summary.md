# Step 36 Candidate Activity Forecast

Run command: `python src/step36_a_candidate_activity_forecast.py`

## C_daily vs U_d correlation: 0.273
## O_d (active hours per vessel) CV: 0.142

## C Oracle decomposition (target_like)

| Oracle | SSE | reduction |
| --- | --- | --- |
| oracle_true_C_total | 3207 | 5.6% |
| oracle_true_C_profile | 1941 | 42.9% |
| oracle_true_C_both | 1522 | 55.2% |
| oracle_true_O_per_vessel | 3207 | 5.6% |

## Deployable C methods (target_like)

- Ctotal_long_mean: -2.5%
- Ctotal_recent3: -5.4%
- Ctotal_recent6: -2.2%
- Ctotal_lag7_normal: -2.5%
- Ctotal_lag14_normal: -1.1%
- occupancy_long: -4.4%
- occupancy_recent6: -5.9%
- occupancy_lag7_normal: -4.4%
- exact_step11_baseline: 0.0%

## Lag7-normal information condition

- block_pre_normal: n=4, red=-4.8% (limited)
- block_target_like: n=0, red=0.0% (limited)
- block_post_outage_stress: n=0, red=0.0% (limited)

## Decision
- Oracle >= 10%: YES
- Deployable >= 5%: NO