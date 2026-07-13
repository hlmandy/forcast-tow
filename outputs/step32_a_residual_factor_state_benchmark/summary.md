# Step 32 A Residual Factor State Benchmark

Run command: `python src/step32_a_residual_factor_state_benchmark.py`

## Cross-fitted residual dates: 11

## Oracle projections (key results)

| block | method | SSE | reduction |
| --- | --- | --- | --- |
| block_target_like | oracle_raw72_pca_rank1 | 4168 | -22.6% |
| block_target_like | oracle_raw72_pca_rank2 | 3889 | -14.4% |
| block_target_like | oracle_raw72_pca_rank3 | 3799 | -11.8% |
| block_target_like | oracle_hour24_pca_rank1 | 3925 | -15.5% |
| block_target_like | oracle_hour24_pca_rank2 | 3678 | -8.2% |
| block_target_like | oracle_hour24_pca_rank3 | 3438 | -1.1% |
| block_target_like | oracle_hour24_fourier_K1 | 2739 | 19.4% |
| block_target_like | oracle_hour24_fourier_K2 | 2511 | 26.1% |
| block_target_like | oracle_hour24_fourier_K3 | 2498 | 26.5% |
| block_target_like | oracle_hour24_fourier_K4 | 2192 | 35.5% |
| block_post_outage_stress | oracle_raw72_pca_rank1 | 4535 | -17.1% |
| block_post_outage_stress | oracle_raw72_pca_rank2 | 4341 | -12.1% |
| block_post_outage_stress | oracle_hour24_pca_rank1 | 4246 | -9.7% |
| block_post_outage_stress | oracle_hour24_pca_rank2 | 4037 | -4.3% |
| block_post_outage_stress | oracle_hour24_fourier_K1 | 2992 | 22.7% |
| block_post_outage_stress | oracle_hour24_fourier_K2 | 2682 | 30.7% |
| block_post_outage_stress | oracle_hour24_fourier_K3 | 2611 | 32.6% |
| block_post_outage_stress | oracle_hour24_fourier_K4 | 2376 | 38.6% |

## Deployable methods (key results)

- block_pre_normal: no deployable improvement > 0.1%
- block_target_like: no deployable improvement > 0.1%
- block_post_outage_stress: best deployable = residual_mean_alpha0.25 (1.2%)

## Decision
- Oracle passes 10%: 8 methods
- Deployable passes 5%: 0 methods
- Oracle passes but deployable fails → subspace exists but factor prediction fails
