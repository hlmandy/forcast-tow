# Step 35 A Threshold Measurement Decomposition

Run command: `python src/step35_a_threshold_measurement_decomposition.py`

## Daily components (post_outage)

| date | level_resid | C_dev | M_dev | Q_dev | Q23_dev | stable_dev | bord_dev | nm_dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018-01-19 | -65 | -33 | -0.109 | -0.196 | -0.190 | -82 | 10 | 5 |
| 2018-01-20 | 91 | 50 | 0.030 | 0.016 | -0.005 | 88 | -7 | 1 |
| 2018-01-21 | 98 | 38 | 0.079 | 0.043 | 0.032 | 93 | -5 | -9 |
| 2018-01-22 | -51 | -36 | -0.074 | 0.013 | 0.019 | -49 | -9 | -7 |
| 2018-01-23 | 13 | -9 | 0.033 | 0.011 | 0.023 | 9 | -3 | -5 |
| 2018-01-24 | 54 | 31 | 0.003 | 0.000 | 0.009 | 42 | 5 | -2 |

## Component Oracles (target_like)

| Oracle | SSE | reduction |
| --- | --- | --- |
| oracle_true_C | 1537 | 54.8% |
| oracle_true_M | 2895 | 14.8% |
| oracle_true_Q | 3215 | 5.4% |
| oracle_true_Q23 | 3262 | 4.0% |
| oracle_true_stable_ge5 | 1196 | 64.8% |
| oracle_true_borderline | 3312 | 2.6% |
| oracle_true_CQ | 1364 | 59.9% |

## Decision
- Best Oracle: oracle_true_stable_ge5 (64.8%)
- Any Oracle >= 10%: YES