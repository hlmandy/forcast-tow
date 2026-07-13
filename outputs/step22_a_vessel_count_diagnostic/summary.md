# Step 22 A Vessel-Count Diagnostic

Run command: `python src/step22_a_vessel_count_diagnostic.py`

## 1. Activity coefficient c_d = H_d / U_d

| subset | n | mean | std | cv | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| all | 24 | 5.13 | 1.673 | 0.326 | 2.09 | 7.62 |
| normal | 17 | 6.02 | 1.003 | 0.167 | 4.75 | 7.62 |
| pre_outage_normal | 11 | 5.75 | 0.755 | 0.131 | 4.87 | 7.17 |
| post_recovery_normal | 6 | 6.50 | 1.286 | 0.198 | 4.75 | 7.62 |
| degraded | 7 | 2.99 | 0.709 | 0.237 | 2.09 | 4.05 |
| weekday_normal | 13 | 5.88 | 0.953 | 0.162 | 4.75 | 7.62 |
| weekend_normal | 4 | 6.46 | 1.179 | 0.182 | 4.98 | 7.54 |

- Normal c_d cv = 0.167; if stable, U_d * c_hat is a viable daily-total predictor.
- U_d vs H_d Pearson (normal) = 0.070; U_d vs c_d Pearson = -0.364.
- If U_d vs c_d is near 0, c_d is vessel-count-independent -> U_d scaling is clean.

## 2. Region shares s_z

| region | subset | mean | cv |
| --- | --- | --- | --- |
| core | normal | 0.521 | 0.052 |
| near | normal | 0.333 | 0.053 |
| outer | normal | 0.146 | 0.188 |

## 3. Analog-day potential
- 136 normal-day pairs. corr(|dU|, |dH|) = -0.202.
- If positive, similar-U days have similar H -> analog weighting is viable.

## 4. Hourly profile by vessel-count quantile
- See hourly_profile_by_vessel_quantile.csv. If low-U and high-U profiles differ, U_d should modulate the shape, not just the total.

## 5. Decision for Step 23
- c_d stability (normal cv=0.167) determines whether U_d*c_hat is viable.
- Region share stability determines whether H_d -> H_{d,z} via s_z is viable.
- If both are sufficiently stable, proceed to Step 23 vessel-constrained three-layer model.
