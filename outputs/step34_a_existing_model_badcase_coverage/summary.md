# Step 34 Existing Model Bad-case Coverage

Run command: `python src/step34_a_existing_model_badcase_coverage.py`

## Models in pool: 48
## Baseline SSE (post_outage): 3872

## Oracle envelopes

| Oracle | SSE | reduction |
| --- | --- | --- |
| best_existing_per_day | 3673 | 5.1% |
| best_existing_per_region_day | 3624 | 6.4% |
| convex_hull_per_day | 3579 | 7.6% |
| correction_span_ols | 3174 | 18.0% |

## Per-model total SSE (top 5 by reduction)

- decomp_mean_daytype_shrunk/exclude_outage_severe: total=0.0%, badcase=0.0%
- decomp_mean_daytype_shrunk/quality_weighted: total=-0.8%, badcase=-0.9%
- decomp_mean_mean/quality_weighted: total=-1.0%, badcase=-3.2%
- decomp_mean_mean/exclude_outage_severe: total=-1.3%, badcase=-4.2%
- hour_mean/quality_weighted: total=-0.2%, badcase=-4.3%

## Top ensembles
- nnls_0.5_hour_mean/quality_we_decomp_mean_daytype__decomp_mean_mean/qua: 0.3%
- nnls_0.5_hour_mean/quality_we_decomp_mean_daytype__decomp_mean_mean/qua: 0.3%
- nnls_0.5_hour_mean/quality_we_decomp_mean_daytype__decomp_mean_mean/qua: 0.3%

## Decision
- Oracle best_per_day >= 15%: NO
- Oracle correction_span >= 20%: NO
- All existing models share similar bad-case errors → combination has low value