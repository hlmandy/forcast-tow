# Step 11 Final Validation Candidates

> Frozen A/B components retrained on the full training period; predictions for 2018-01-25..01-31. Generated, NOT submitted.

Run command: `python src/step11_generate_validation_candidates.py`

## 1. Frozen components

- A: a_exclude_daytype (exclude_outage_severe / decomp_mean_daytype_shrunk), a_quality_hour (quality_weighted / hour_mean), a_quality_mean_profile (quality_weighted / decomp_mean_mean).
- B: b_hier_scale (pair_hour_hier_tau_scale_cv / independent), b_circular (pair_hour_circular_cv / independent), b_fixed085 (pair_hour_fixed085 / independent).
- No new models, no ensembling. No validation AIS, no validation labels. validation daily vessel count is retained only in the debug tables; frozen models do NOT scale by it.

## 2. Historical reproduction checks

| component | ref step | max abs diff | rounded exact | passed |
| --- | --- | --- | --- | --- |
| a_exclude_daytype | step07 | 1.78e-15 | 1.000 | True |
| a_quality_hour | step07 | 1.78e-15 | 1.000 | True |
| a_quality_mean_profile | step07 | 1.78e-15 | 1.000 | True |
| b_hier_scale | step10 | 4.44e-16 | 1.000 | True |
| b_circular | step10 | 2.22e-16 | 1.000 | True |
| b_fixed085 | step10 | 1.11e-16 | 1.000 | True |

## 3. Final A training

- Quality groups recomputed on the full 2018-01-01..01-24 training period; exclude_outage_severe keeps normal+reduced dates; quality_weighted uses weights normal=1/reduced=0.5/severe=0.1/outage=0.
- a_exclude_daytype: validation-period predicted total = 2241.
- a_quality_hour: validation-period predicted total = 2240.
- a_quality_mean_profile: validation-period predicted total = 2205.

## 4. Final B hyperparameter selection

- 17 inner validation dates (2018-01-07..01-23).
- b_hier_scale selected: tau=10, scale=0.95.
- b_circular selected: smoother=triangular3.
- b_hier_scale: best inner_sse=2239, second=2251, gap=12.0.
- b_circular: best inner_sse=2270, second=2350, gap=80.0.

## 5. Final B training boundary

- 2018-01-24 23:00 six migration labels are right-censored (no 2018-01-25 00:00 state) and did NOT enter training; last observable training migration hour is 2018-01-24 22:00. Final prediction still covers all 1008 rows 2018-01-25..01-31 (including 01-31 23:00) and uses no validation AIS state.

## 6. Candidate definitions

- candidate_01_main: A=a_exclude_daytype, B=b_hier_scale — final_analog-oriented main candidate
- candidate_02_a_sensitivity: A=a_quality_hour, B=b_hier_scale — only A changes (vs 01); isolates A training/structure effect
- candidate_03_b_robustness: A=a_exclude_daytype, B=b_circular — only B changes (vs 01); rolling-fold robust B
- candidate_02 changes only A; candidate_03 changes only B (vs candidate_01), so online sub-component results are easier to attribute.

## 7. Prediction totals and dispersion

| candidate | A total | B total | A daily range | B daily range |
| --- | --- | --- | --- | --- |
| candidate_01_main | 2241 | 427 | 318-321 | 61-61 |
| candidate_02_a_sensitivity | 2240 | 427 | 320-320 | 61-61 |
| candidate_03_b_robustness | 2241 | 497 | 318-321 | 71-71 |

| pair | A differing cells | A mean |diff| | B differing cells | B mean |diff| |
| --- | --- | --- | --- | --- |
| candidate_01_main vs candidate_02_a_sensitivity | 143 | 0.292 | 0 | 0.000 |
| candidate_01_main vs candidate_03_b_robustness | 0 | 0.000 | 98 | 0.097 |
| candidate_02_a_sensitivity vs candidate_03_b_robustness | 143 | 0.292 | 98 | 0.097 |

## 8. Official-file validation

- Each A file has 504 rows; each B file 1008 rows; columns match the template exactly; row order matches the template; keys unique; no missing/negative values; all integers; UTF-8 with BOM; no ZIP generated. See submission_checks.csv (all passed).

## 9. Files prepared for review

- submissions/candidate_01_main/提交结果1_区域活跃拖轮数量.csv
- submissions/candidate_01_main/提交结果2_圈层间拖轮迁移量.csv
- submissions/candidate_02_a_sensitivity/提交结果1_区域活跃拖轮数量.csv
- submissions/candidate_02_a_sensitivity/提交结果2_圈层间拖轮迁移量.csv
- submissions/candidate_03_b_robustness/提交结果1_区域活跃拖轮数量.csv
- submissions/candidate_03_b_robustness/提交结果2_圈层间拖轮迁移量.csv

These files have been generated but not submitted.
