# Step 17 Controlled B Core-Near Hybrid

Run command: `python src/step17_generate_controlled_b_hybrid.py`

## 1. Purpose
- A fixed to the online-confirmed A SSE=4423 file (step13). Baseline B = online-confirmed B SSE=1762 file. New candidate replaces ONLY core->near and near->core; the other four B directions are byte-identical. Strict controlled experiment.

## 2. Historical offline comparison

| method | rolling mean | median | max | final | nonoverlap |
| --- | --- | --- | --- | --- | --- |
| baseline_confirmed_structure | 0.9246 | 0.8998 | 1.0397 | 742 | (see csv) |
| hybrid_confirmed_core_near | 0.8995 | 0.8866 | 0.9861 | 738 | (see csv) |

- Hybrid beats baseline in 7/8 rolling folds. core->near 2799->2618; near->core 2182->2160. Other 4 directions identical.

## 3. Final core-near training
- Complete training dates 2018-01-01..01-23; long_mean daily total; triangular3 profile; tau=20 direction ratio; pair-hierarchical rounding.

- core->near total over 7 days = 175; near->core total = 189 (per-day forward 25 / reverse 27).

## 4. Controlled value replacement
- B cells changed (core_near): 56/336. Mean |diff| and max diff in b_value_comparison.csv. Other 4 directions 672/672 unchanged.

## 5. File-order preservation
- A keeps the verified old zone order; B keeps the online 1762 file row order. No template re-sort, no Excel re-save.

## 6. Interpretation
- The candidate's online total-score difference vs the current confirmed submission comes ONLY from B's two core_near directions. A should remain SSE=4423; the other four B directions are unchanged. Whether online B SSE improves still requires submission. No online improvement is claimed from offline results.

## 7. Files prepared for review
- candidate_07_A4423_B_core_near_hybrid/提交结果1_区域活跃拖轮数量.csv
- candidate_07_A4423_B_core_near_hybrid/提交结果2_圈层间拖轮迁移量.csv

These files have been generated but not submitted.
