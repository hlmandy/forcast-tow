# Implementation Audit

## Step 34
- correction_span_OLS: high-dimensional same-sample Oracle on 432 cells, not transferable
- NNLS: target-block grid search, not nested OOF
- No need to rerun Step 34; current conclusions stand

## This step
- Decomposes A into C × M × Q components
- Identifies whether bad-case comes from activity (C), sampling (Q), or multiplicity (M)
