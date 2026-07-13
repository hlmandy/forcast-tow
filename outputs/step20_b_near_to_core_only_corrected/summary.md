# Step 20 near->core Only (Score-Corrected)

Run command: `python src/step20_b_near_to_core_only_corrected.py`

## Corrected baseline
- Step 11 candidate_01_main: A=4423, B=952, total=7279 (best confirmed).
- Step 17 B=940 (both core->near + near->core replaced).
- Step 18 B=967 (only core->near replaced).
- Derived: near->core only B = 952 + (940-967) = **925**.
- Expected total = 4423 + 3×925 = **7198**.

## Candidate design
- A: step 11 candidate_01_main (SHA256-identical).
- B: step 11 baseline, only near->core replaced with step 17 pair_default_hierarchical.
- near->core: 168 -> 189 (changed 35 cells, inc 28, dec 7).
- B total: 427 -> 448.
- Other 5 directions: 840/840 unchanged.

## Interpretation after submission
- S_new = online B SSE. near->core delta = S_new - 952.
- Expected S_new ≈ 925. If confirmed, total ≈ 7198.

## Files prepared for review
- candidate_10.../提交结果1...
- candidate_10.../提交结果2...

These files have been generated but not submitted.
