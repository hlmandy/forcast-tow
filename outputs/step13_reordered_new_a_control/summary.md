# Step 13 Reordered New-A Control

## 1. Experimental purpose

- Step 12 reproduced A SSE=6063 and B SSE=1762.
- Step 11's new A used a different zone row order; online A SSE=14671.
- This candidate maps Step 11 new-A prediction VALUES onto the old high-score A row order; B stays as the confirmed 1762 file.
- Purpose: distinguish a row-order problem from a prediction-value problem.

## 2. A value preservation

- New-A source: `outputs\step11_final_validation_candidates\submissions\candidate_01_main\提交结果1_区域活跃拖轮数量.csv`
- New-A total: 2241
- All 504 prediction values preserved by key; no value recalculated or calibrated.

No A prediction value was recalculated or calibrated.

## 3. A order replacement

- Old-A order template: `outputs\step12_recombined_candidate\提交结果1_区域活跃拖轮数量.csv`
- Keys whose row position changed (old vs new): 504/504
- Output rows are in old-A order; only positions changed, not values.

## 4. B preservation

- B source: `outputs\step12_recombined_candidate\提交结果2_圈层间拖轮迁移量.csv`
- B SHA256: c1b9ddd5b71a431b61a67a7ec3ae63fab5988e095457892ad17ac641dd34afb5
- Output B is byte-for-byte identical to the source.

## 5. Submission interpretation

- If this candidate's A SSE drops far below 14671, row order was a major cause of Step 11's failure.
- If it remains well above 6063, Step 11's new-A values themselves are inaccurate.
- If it is near or below 6063, Step 11's model may be effective and the main issue was row order.
- No online score is predicted here.

## 6. Files prepared for review

- `outputs\step13_reordered_new_a_control\candidate_05_newA_oldOrder_newB\提交结果1_区域活跃拖轮数量.csv`
- `outputs\step13_reordered_new_a_control\candidate_05_newA_oldOrder_newB\提交结果2_圈层间拖轮迁移量.csv`

These files have been generated but not submitted.
