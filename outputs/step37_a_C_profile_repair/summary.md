# Step 37 C Profile Repair and Reshape

Run command: `python src/step37_a_C_profile_repair.py`

## Block scores (key methods, target_like)

| method | SSE | reduction |
| --- | --- | --- |
| profile_long | 3437 | -1.1% |
| profile_recent3 | 3613 | -6.3% |
| profile_recent6 | 3613 | -6.3% |
| profile_lag7_direct | 3437 | -1.1% |
| profile_lag14_direct | 4301 | -26.5% |
| profile_lag_fallback | 4301 | -26.5% |
| oracle_true_profile | 1893 | 44.3% |
| exact_step11_baseline | 3399 | 0.0% |

## Profile gap diagnostics (by lag)

| lag | mean corr | median corr | n_pairs |
| --- | --- | --- | --- |
| 1 | 0.362 | 0.342 | 15 |
| 2 | 0.367 | 0.435 | 13 |
| 3 | 0.425 | 0.406 | 11 |
| 4 | 0.448 | 0.479 | 9 |
| 5 | 0.525 | 0.519 | 7 |
| 6 | 0.390 | 0.404 | 5 |
| 7 | 0.397 | 0.385 | 4 |
| 8 | 0.504 | 0.560 | 4 |
| 9 | 0.488 | 0.513 | 4 |
| 10 | 0.287 | 0.291 | 4 |
| 11 | 0.382 | 0.370 | 4 |
| 12 | 0.444 | 0.540 | 5 |
| 13 | 0.295 | 0.334 | 6 |
| 14 | 0.348 | 0.263 | 6 |

## Information conditions

- block_pre_normal all_target: n=4, red=-25.3% (limited)
- block_pre_normal lag7_normal: n=4, red=-25.3% (limited)
- block_pre_normal lag14_normal: n=0, red=0.0% (limited)
- block_target_like all_target: n=5, red=-26.5% 
- block_target_like lag7_normal: n=0, red=0.0% (limited)
- block_target_like lag14_normal: n=5, red=-26.5% 
- block_post_outage_stress all_target: n=6, red=-28.3% 
- block_post_outage_stress lag7_normal: n=0, red=0.0% (limited)
- block_post_outage_stress lag14_normal: n=6, red=-28.3% 

## Decision
- Oracle true profile reduction: 44.3%
- Best deployable: profile_daytype (0.3%)
