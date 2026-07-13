# Step 28 A Behavioral Cohort Mixture

Run command: `python src/step28_a_behavioral_cohort_mixture.py`

## Results (post-outage block, Jan 19-24)

| method | SSE | reduction | deployable |
| --- | --- | --- | --- |
| exact_step11_baseline | 3872 | 0.0% | True |
| cohort_aggregate_1_long | 5017 | -29.6% | True |
| cohort_aggregate_1_recent3 | 5017 | -29.6% | True |
| cohort_aggregate_1_recent7 | 5017 | -29.6% | True |
| cohort_aggregate_1_blend | 5017 | -29.6% | True |
| cohort_dominant_region_3_long | 5052 | -30.5% | True |
| cohort_dominant_region_3_recent3 | 5033 | -30.0% | True |
| cohort_dominant_region_3_recent7 | 5187 | -34.0% | True |
| cohort_dominant_region_3_blend | 5041 | -30.2% | True |
| cohort_region_intensity_6_long | 4992 | -28.9% | True |
| cohort_region_intensity_6_recent3 | 5016 | -29.5% | True |
| cohort_region_intensity_6_recent7 | 5148 | -33.0% | True |
| cohort_region_intensity_6_blend | 5046 | -30.3% | True |
| cohort_region_shift_6_long | 4952 | -27.9% | True |
| cohort_region_shift_6_recent3 | 4923 | -27.1% | True |
| cohort_region_shift_6_recent7 | 5003 | -29.2% | True |
| cohort_region_shift_6_blend | 4999 | -29.1% | True |
| oracle_true_cohort_counts_dominant3 | 4822 | -24.5% | False |
| oracle_true_counts_global_profile | 5142 | -32.8% | False |

## Key findings
- Deployable methods passing 5%: 0
- Oracle true cohort counts (dominant3) reduction: -24.5%
- Oracle true counts + global profile reduction: -32.8%
- New vessel ratio: 0.064
