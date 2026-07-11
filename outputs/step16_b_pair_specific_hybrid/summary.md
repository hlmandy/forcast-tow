# Step 16 B Pair-Specific Hybrid

Run command: `python src/step16_b_pair_specific_hybrid.py`

## 1. Motivation
- Overall pair decomposition did not stably beat circular3, but core_near had a local stable gain. near_outer and core_outer must stay circular3. This step only selects existing per-task predictions; no new models, no averaging.

## 2. Reproduction checks
- circular3_all == step15 circular3_independent (<1e-10). Hybrid non-core_near 4 directions == circular3 (<1e-10). core_near 2 directions == the specified pair method. **PASS**

## 3. Rolling comparison

| method | mean MSE | median MSE | max MSE |
| --- | --- | --- | --- |
| circular3_all | 0.9039 | 0.8950 | 0.9990 |
| hybrid_core_near_default | 0.8966 | 0.8925 | 0.9554 |
| hybrid_core_near_globalcv | 0.8928 | 0.8935 | 0.9296 |
| hybrid_core_near_bypaircv | 0.8980 | 0.8935 | 0.9544 |

## 4. Final analog and non-overlap

| method | final SSE | nonoverlap1 MSE | nonoverlap2 MSE | combined MSE |
| --- | --- | --- | --- | --- |
| circular3_all | 766 | 0.9990 | 0.8952 | 0.9473 |
| hybrid_core_near_default | 746 | 0.9554 | 0.8912 | 0.9234 |
| hybrid_core_near_globalcv | 746 | 0.9296 | 0.8962 | 0.9129 |
| hybrid_core_near_bypaircv | 746 | 0.9355 | 0.8922 | 0.9139 |

## 5. Direction-level effect

| method | core->near | near->core | near->outer | outer->near | core->outer | outer->core |
| --- | --- | --- | --- | --- | --- | --- |
| circular3_all | 2661 | 2176 | 816 | 872 | 366 | 393 |
| hybrid_core_near_default | 2618 | 2160 | 816 | 872 | 366 | 393 |
| hybrid_core_near_globalcv | 2596 | 2151 | 816 | 872 | 366 | 393 |
| hybrid_core_near_bypaircv | 2641 | 2148 | 816 | 872 | 366 | 393 |

## 6. Forecast horizon
- See b_horizon_scores.csv.

## 7. Decision
- Pass criteria: >=5 rolling folds beat circular3; final < circular3; nonoverlap combined < circular3; rolling max not worse; core->near and near->core both improve; other 4 directions identical to circular3.
