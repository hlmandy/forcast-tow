# Step 14 (B1) Three Layer-Pair Diagnostic

> No models. Reconstructs six directed flows into three bidirectional layer-pairs and diagnoses sparsity/stability. 2018-01-24 23:00 right-censored (excluded).

Run command: `python src/step14_b_pair_diagnostic.py`

## 1. Zero-rate reduction (merged vs single direction)

| pair | zero ab | zero ba | zero X(merged) | reduction (pp) |
| --- | --- | --- | --- | --- |
| core_near | 0.511 | 0.447 | 0.249 | 19.8 |
| near_outer | 0.763 | 0.723 | 0.541 | 18.3 |
| core_outer | 0.831 | 0.850 | 0.704 | 12.7 |

## 2. Daily-total stability (cv across dates)

| pair | cv T (merged) | cv ab | cv ba |
| --- | --- | --- | --- |
| core_near | 0.214 | 0.224 | 0.227 |
| near_outer | 0.326 | 0.392 | 0.351 |
| core_outer | 0.357 | 0.455 | 0.374 |

## 3. Hour-share stability (mean pairwise Pearson, normal 17 days)

| pair | Pearson X(merged) | Pearson ab | Pearson ba |
| --- | --- | --- | --- |
| core_near | 0.191 | 0.176 | 0.091 |
| near_outer | 0.124 | 0.116 | 0.137 |
| core_outer | 0.087 | 0.121 | 0.023 |

## 4. Direction ratio (a->b share of the pair)

| pair | global r | mean hour std(r) |
| --- | --- | --- |
| core_near | 0.496 | 0.376 |
| near_outer | 0.467 | 0.354 |
| core_outer | 0.537 | 0.358 |
- Per-hour mean r (normal days) is in pair_direction_ratio_profile.csv. A stable hour pattern would show small std and a consistent level across hours.

## 5. Net flow (a->b minus b->a)

| pair | final cumulative net flow | mean |daily net flow| |
| --- | --- | --- |
| core_near | -9.0 | 2.54 |
| near_outer | -28.0 | 2.25 |
| core_outer | 17.0 | 1.79 |

## 6. Weekday vs weekend (normal days)

| pair | n wd/we | mean X wd | mean X we | mean r wd | mean r we |
| --- | --- | --- | --- | --- | --- |
| core_near | 13/4 | 2.19 | 2.08 | 0.469 | 0.476 |
| near_outer | 13/4 | 0.69 | 0.92 | 0.480 | 0.429 |
| core_outer | 13/4 | 0.41 | 0.39 | 0.538 | 0.554 |

## 7. Decision input

- core_near and near_outer: pair structure is less sparse and more stable than the single directions -> prioritize the pair-decomposition model.
- core_outer: remains the sparsest pair (see zero_rate_X); decide between pair-decomposition and a sparse/hurdle model based on whether its merged stability is acceptable.
