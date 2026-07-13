# Step 33 Fourier Coefficient Predictability

Run command: `python src/step33_a_fourier_coefficient_predictability.py`

## Cross-fitted dates: 12

## Oracle decomposition (target_like)

| K | level red | shape red | combined red |
| --- | --- | --- | --- |
| 1 | 10.7% | 7.8% | 19.4% |
| 2 | 10.7% | 15.1% | 26.1% |
| 3 | 10.7% | 16.7% | 26.5% |
| 4 | 10.7% | 23.3% | 35.5% |

## Deployable methods (target_like)

- fourier_K1_coef_long_mean_a0.5: 1.5%
- fourier_K1_coef_long_mean_a1.0: 1.1%
- fourier_K1_coef_recent3_mean_a0.5: 1.2%
- fourier_K2_coef_recent3_mean_a0.5: 0.9%

## Decision
- Shape Oracle passes 10%: YES
- Deployable passes 5%: NO