# Step 25 A Weekly Vessel Model Benchmark

Run command: `python src/step25_a_weekly_vessel_model_benchmark.py`

## Normal-day rolling-origin (unique dates)

| method | total SSE | reduction | n_dates |
| --- | --- | --- | --- |
| baseline_step11 | 6513 | 0.0% | 10 |
| latest_normal_same_weekday | 9954 | -52.8% | 10 |
| mean_two_normal_same_weekdays | 9093 | -39.6% | 10 |
| weighted_two_normal_same_weekdays | 9336 | -43.3% | 10 |
| latest_normal_same_weekday_scaled_U_p025 | 9965 | -53.0% | 10 |
| latest_normal_same_weekday_scaled_U_p05 | 9867 | -51.5% | 10 |
| vessel_latest_weekly_pattern | 17254 | -164.9% | 10 |

## Post-outage block (01-19 to 01-24)

| method | total SSE | reduction |
| --- | --- | --- |
| baseline_step11 | 4026 | 0.0% |
| latest_normal_same_weekday | 5826 | -44.7% |
| mean_two_normal_same_weekdays | 4965 | -23.3% |
| weighted_two_normal_same_weekdays | 5208 | -29.4% |
| latest_normal_same_weekday_scaled_U_p025 | 5826 | -44.7% |
| latest_normal_same_weekday_scaled_U_p05 | 5661 | -40.6% |
| vessel_latest_weekly_pattern | 9216 | -128.9% |

## Weekly pair SSE by lag (normal-to-normal)

| lag | mean SSE | n_pairs |
| --- | --- | --- |
| 7 | 1032 | 4 |
| 14 | 971 | 6 |
| 21 | 992 | 3 |

## Target period analog map (01-25 to 01-31)

| target | lag7 | lag7_q | lag14_q | lag21_q | selected | reason |
| --- | --- | --- | --- | --- | --- | --- |
| 2018-01-25 | 2018-01-18 | severe | normal | normal | 2018-01-11 | normal lag-14 |
| 2018-01-26 | 2018-01-19 | normal | reduced | normal | 2018-01-19 | normal lag-7 |
| 2018-01-27 | 2018-01-20 | normal | outage | normal | 2018-01-20 | normal lag-7 |
| 2018-01-28 | 2018-01-21 | normal | outage | normal | 2018-01-21 | normal lag-7 |
| 2018-01-29 | 2018-01-22 | normal | outage | normal | 2018-01-22 | normal lag-7 |
| 2018-01-30 | 2018-01-23 | normal | outage | normal | 2018-01-23 | normal lag-7 |
| 2018-01-31 | 2018-01-24 | normal | severe | normal | 2018-01-24 | normal lag-7 |

## Decision
- Methods passing 10% threshold: 0
- Methods passing 15%: 0
