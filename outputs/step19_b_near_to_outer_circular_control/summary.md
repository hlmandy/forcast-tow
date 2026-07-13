# Step 19 Near-to-Outer Circular Control

Run command: `python src/step19_b_near_to_outer_circular_control.py`

## 1. Purpose
- Current best online B SSE = 1741. core->near already confirmed effective. This candidate adds near->outer circular3 only.

## 2. Historical evidence
- current: mean=0.8995 med=0.8866 max=0.9861 fa=738
- new:     mean=0.8944 med=0.8885 max=0.9683 fa=736
- folds: better=4 tie=1 worse=3
- Signal weaker than core->near; rolling median slightly worse. Cannot pre-assert online improvement.

## 3. Final model
- Full training 01-01..01-24 (01-24 23:00 excluded). Long-term hour mean + triangular3 + independent round.
- Changed hours: [7, 9, 10]

## 4. Controlled replacement
- near->outer: 21 cells changed, all 0->1. Total 21->42. B total +21 (420->441).
- Other 5 directions 840/840 unchanged.

## 5. Interpretation after submission
- S_new = online B SSE. near->outer delta = S_new - 1741.
- S_new < 1741: keep near->outer circular3. S_new = 1741: neutral. S_new > 1741: revert.

## 6. Files prepared for review
- candidate_09.../提交结果1...
- candidate_09.../提交结果2...

These files have been generated but not submitted.
