# Step 21 Joint Backtest Framework

Run command: `python src/step21_joint_backtest_framework.py`

## Baseline: candidate_10
- A = step07 decomp_mean_daytype_shrunk / exclude_outage_severe (A MSE ~10.84)
- B = step10 b_hier_scale (5 dirs) + step15 pair_default_hierarchical (near->core) (B MSE ~0.922)
- Joint objective: SSE_A + 3*SSE_B

## Rolling 8 folds (auxiliary)

| fold | SSE_A | SSE_B | Total |
| --- | --- | --- | --- |
| fold_01 | 6328 | 1062 | 9514 |
| fold_02 | 6381 | 1035 | 9486 |
| fold_03 | 5804 | 958 | 8678 |
| fold_04 | 5677 | 895 | 8362 |
| fold_05 | 5341 | 905 | 8056 |
| fold_06 | 4949 | 846 | 7487 |
| fold_07 | 4699 | 845 | 7234 |
| fold_08 | 4515 | 883 | 7164 |

- mean=8248 median=8209 max=9514

## Non-overlapping blocks (primary)

| block | SSE_A | SSE_B | Total |
| --- | --- | --- | --- |
| block_1 | 6328 | 1062 | 9514 |
| block_2 | 4515 | 883 | 7164 |
| combined | 10843 | 1945 | 16678 |

## Final analog
- SSE_A=3872 SSE_B=730 Total=6062

## Forecast horizon (rolling mean total SSE per day)

| horizon | mean total SSE |
| --- | --- |
| 1 | 1240.9 |
| 2 | 1221.4 |
| 3 | 1225.5 |
| 4 | 1235.9 |
| 5 | 1155.8 |
| 6 | 1069.4 |
| 7 | 1098.9 |

## A dominates
- A share of rolling mean total: 66%
- B weighted share: 34%
- A optimization (daily vessel count utilization) is the primary lever.

## Next steps
- Step 22: A daily vessel count diagnostic (U_d, H_d, c_d stability)
- Step 23: A vessel-constrained three-layer model
- Step 24: candidate_10 B direction-hour OOF residual diagnostics
- Step 25: A-prediction-driven B migration rate model
