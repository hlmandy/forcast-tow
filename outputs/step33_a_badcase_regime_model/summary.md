# Step 33 A Bad-case Regime Model

Run command: `python src/step33_a_badcase_regime_model.py`

## Badcase daily catalog (target_like)

| date | actual | baseline | level_resid | SSE | top10_share |
| --- | --- | --- | --- | --- | --- |
| 2018-01-06 | 304 | 345 | -2 | 377 | 0.66 |
| 2018-01-07 | 363 | 337 | 1 | 734 | 0.58 |
| 2018-01-08 | 302 | 340 | -2 | 402 | 0.70 |
| 2018-01-09 | 296 | 334 | -2 | 1060 | 0.57 |
| 2018-01-10 | 263 | 332 | -3 | 523 | 0.55 |
| 2018-01-11 | 264 | 321 | -2 | 501 | 0.50 |
| 2018-01-19 | 247 | 312 | -3 | 525 | 0.69 |
| 2018-01-20 | 400 | 305 | 4 | 629 | 0.54 |
| 2018-01-21 | 407 | 314 | 4 | 995 | 0.55 |
| 2018-01-22 | 261 | 320 | -2 | 585 | 0.58 |
| 2018-01-23 | 325 | 314 | 0 | 481 | 0.65 |
| 2018-01-24 | 366 | 317 | 2 | 741 | 0.60 |

## Block scores (key methods)

| block | method | SSE | reduction |
| --- | --- | --- | --- |
| block_target_like | exact_step11_baseline | 3399 | 0.0% |
| block_target_like | level_zero_regime_zero | 3399 | 0.0% |
| block_target_like | regime_soft_recent3 | 3524 | -3.7% |
| block_target_like | regime_soft_lag_normal | 3469 | -2.1% |
| block_target_like | regime_soft_lag_blend | 3594 | -5.7% |
| block_target_like | oracle_true_level_regime | 3513 | -3.4% |
| block_target_like | oracle_true_regime_template | 3513 | -3.4% |
| block_post_outage_stress | exact_step11_baseline | 3872 | 0.0% |
| block_post_outage_stress | level_zero_regime_zero | 3872 | 0.0% |
| block_post_outage_stress | regime_soft_recent3 | 4132 | -6.7% |
| block_post_outage_stress | regime_soft_lag_normal | 4177 | -7.9% |
| block_post_outage_stress | regime_soft_lag_blend | 4202 | -8.5% |
| block_post_outage_stress | oracle_true_level_regime | 4102 | -5.9% |
| block_post_outage_stress | oracle_true_regime_template | 4102 | -5.9% |

## Decision
- Deployable passes 5%: NO
- Oracle passes 10%: NO