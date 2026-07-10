# Step 05 A Temporal Structure Audit

> Diagnostic only. No models, no B task, no submissions. All A labels reused from optimized_baseline.

Run command: `python src/step05_a_temporal_structure_audit.py`

## 1. Data and consistency checks

- A-label rows: 1728 (24 days x 24 hours x 3 regions); each (date, region) has 24 hours. **PASS**
- Daily per-region totals match Step 01 a_core/near/outer_total exactly. **PASS**
- Normal dates: 17. profile_share sums to 1 within 1e-12 wherever daily_total > 0. **PASS**
- pre_outage (11 days): 2018-01-01, 2018-01-02, 2018-01-03, 2018-01-04, 2018-01-05, 2018-01-06, 2018-01-07, 2018-01-08, 2018-01-09, 2018-01-10, 2018-01-11
- degraded_period (7 days): 2018-01-12, 2018-01-13, 2018-01-14, 2018-01-15, 2018-01-16, 2018-01-17, 2018-01-18
- post_recovery (6 days): 2018-01-19, 2018-01-20, 2018-01-21, 2018-01-22, 2018-01-23, 2018-01-24

## 2. Daily regional totals

Normal-day daily totals per region (descriptive only; `cv` = std/mean):

| region | mean | std | cv | min | max | time slope (per day) | pearson vessels | spearman vessels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core | 169.0 | 27.6 | 0.163 | 126 | 209 | -0.123 | 0.099 | 0.229 |
| near | 107.8 | 16.6 | 0.154 | 76 | 136 | 0.147 | 0.070 | 0.101 |
| outer | 47.5 | 11.5 | 0.242 | 28 | 72 | 0.023 | -0.034 | 0.045 |

- These describe association only; no causal claim is made from 17 days.

## 3. Stability of normalized hourly profiles

Pairwise similarity of the 24-hour profile_share curves across the 17 normal days, per region:

| region | mean pearson | mean cosine | mean L1 | mean RMSE |
| --- | --- | --- | --- | --- |
| core | 0.3125 | 0.8249 | 0.5488 | 0.0284 |
| near | 0.3230 | 0.7866 | 0.6265 | 0.0327 |
| outer | 0.2053 | 0.7104 | 0.7491 | 0.0394 |

- Most stable shape (highest mean pairwise Pearson): **near**; least stable: **outer**.
- Five least-similar date pairs per region (lowest cosine):
  - core: 2018-01-06~2018-01-09 (cos=0.641); 2018-01-09~2018-01-22 (cos=0.677); 2018-01-01~2018-01-09 (cos=0.686); 2018-01-03~2018-01-09 (cos=0.695); 2018-01-09~2018-01-19 (cos=0.715)
  - near: 2018-01-07~2018-01-23 (cos=0.644); 2018-01-08~2018-01-22 (cos=0.649); 2018-01-02~2018-01-09 (cos=0.651); 2018-01-01~2018-01-07 (cos=0.663); 2018-01-06~2018-01-21 (cos=0.665)
  - outer: 2018-01-04~2018-01-19 (cos=0.416); 2018-01-06~2018-01-21 (cos=0.450); 2018-01-10~2018-01-21 (cos=0.457); 2018-01-01~2018-01-10 (cos=0.465); 2018-01-07~2018-01-10 (cos=0.475)

## 4. Pre-outage versus post-recovery profiles

Similarity between the mean profile_share curve of normal_pre_outage and normal_post_recovery:

| region | pearson | cosine | L1 | RMSE | pre peak hr | post peak hr |
| --- | --- | --- | --- | --- | --- | --- |
| core | 0.8249 | 0.9796 | 0.1635 | 0.0089 | 19 | 19 |
| near | 0.8501 | 0.9763 | 0.1754 | 0.0099 | 19 | 19 |
| outer | 0.6302 | 0.9459 | 0.2796 | 0.0148 | 12 | 16 |

- Sample is small (pre_outage normal days and post_recovery normal days); differences are not evidence of a structural break.

## 5. Weekday versus weekend profiles

- core: weekday 13 days vs weekend 4 days — pearson=0.6873, cosine=0.9626, L1=0.2119, RMSE=0.0121, weekday peak=19, weekend peak=3.
- near: weekday 13 days vs weekend 4 days — pearson=0.6593, cosine=0.9465, L1=0.2754, RMSE=0.0149, weekday peak=19, weekend peak=3.
- outer: weekday 13 days vs weekend 4 days — pearson=0.6594, cosine=0.9495, L1=0.2600, RMSE=0.0144, weekday peak=12, weekend peak=12.

- The mean-curve peak hour shifts between weekday and weekend (core 19->3; near 19->3; outer 12->12). This is a real descriptive signal, but with only 4 weekend days it is not enough on its own to justify a complex day-of-week model.

## 6. PCA dimensionality

Explained variance ratio of the 24-hour profile_share (normal days, column-centered):

| region | PC1 | cum PC2 | cum PC3 | cum PC5 |
| --- | --- | --- | --- | --- |
| core | 0.2308 | 0.3989 | 0.5267 | 0.7030 |
| near | 0.2051 | 0.3597 | 0.4945 | 0.7219 |
| outer | 0.2327 | 0.4066 | 0.5658 | 0.7488 |

- PCA is used only to judge whether the hourly curve is low-dimensional; it is not a prediction model here.

## 7. Implications for the next model

These are read directly from the audit numbers above, not from a fitted model:

- Daily region totals vary across normal days (cv core=0.16, near=0.15, outer=0.24; outer is the most variable) and are essentially unrelated to the daily vessel count (Pearson ~0.07 for a_total, see section 2). Daily total is therefore a genuine prediction target, not a rescaling of vessel count.
- Hourly shapes are NOT stable across individual normal days: the mean pairwise Pearson of the 24-hour profile is only 0.31/0.32/0.21 (core/near/outer). Cosine is higher (0.82-0.71) only because it is not mean-centered and is dominated by the shared mass distribution; PCA independently shows the shape is not low-dimensional (PC1 ~23%, 3 PCs only ~53-57% of the variation, 5 PCs ~70-75%). Most stable shape: near; least stable: outer.
- Because the per-day shape is noisy and not low-dimensional, a 'daily total x fixed hourly shape' decomposition would capture only the average pattern and leave substantial day-specific shape error. Whether that decomposition or direct hourly prediction is better is a modeling decision for a later step, not asserted here.
- On regional difficulty: core has the largest counts (hence the largest absolute errors) with moderate daily cv (0.16) and low shape Pearson (0.31); outer has the highest proportional daily variation (cv 0.24) and the least stable shape (0.21) but much smaller counts. So core's difficulty is driven mainly by scale, while outer's is driven mainly by proportional variability.
- Weekday vs weekend: see section 5; the mean-curve peak shifts but only 4 weekend days are available.
- No model is trained, scored, or selected in this stage.
