# Score Correction — 2026-07-13

## Background

Steps 12–19 were conducted under incorrect online score feedback. The actual
penalty/scoring function used by the competition differs from what was assumed.
All prior online scores cited in steps 12–19 (A SSE 6063, B SSE 1762/1741/1790,
etc.) were wrong. This file records the corrected scores to prevent future
reference to the incorrect baselines.

## Corrected online scores

| Candidate | Corrected A | Corrected B | Total (A + 3B) | Notes |
|-----------|------------|------------|-----------------|-------|
| Step 11 candidate_01_main | **4423** | **952** | **7279** | Best confirmed |
| Step 12 (old A + step11 B) | 14279 | 952 | 17135 | Old A file is terrible under correct scoring |
| Step 13 (step11 A values, old order) | 15073 | 952 | 17929 | Reorder experiment; A values wrong for scorer |
| Step 17 (step13 A + B core_near hybrid) | 15073 | 940 | 17893 | B improved −12 vs 952 |
| Step 18 (step13 A + B core→near only) | 15073 | 967 | 17974 | B worse +15 vs 952 |

## Key corrected derivations

From step 11 B = 952 (baseline), step 17 B = 940 (both core→near + near→core
replaced), step 18 B = 967 (only core→near replaced):

- core→near delta = 967 − 952 = **+15** (core→near replacement HURTS)
- near→core delta = 940 − 967 = **−27** (near→core replacement HELPS)
- near→core only B = 952 + (−27) = **925**

Therefore: replacing only near→core (keeping step 11 baseline for everything
else) should yield B = 925, total = 4423 + 3 × 925 = **7198**.

## What was wrong

The incorrect scores led to wrong conclusions in steps 12–19:
- Steps 12–13 assumed old A (A SSE 6063) was good; actually A = 14279/15073.
- Steps 17–18 assumed B baselines of 1762/1741; actually B = 940/967.
- The direction of improvement was sometimes reversed (core→near was thought
  helpful but actually hurts +15; near→core was thought harmful but actually
  helps −27).

## What remains valid

The controlled experiment STRUCTURE is still valid:
- The per-direction isolation logic is correct.
- The additive decomposition (both = core_near_delta + near_core_delta) holds.
- The step 11 candidate_01_main files are the correct best baseline.

## Tags

- `online-step11-A4423-B952` → commit a2c7fc6 (best confirmed: A=4423, B=952)
- `online-step17-A15073-B940` → commit 2acf049 (A=15073, B=940)
- `online-step18-A15073-B967` → commit 256ad01 (A=15073, B=967)

## Going forward

All future candidates must use step 11 candidate_01_main as the A+B baseline
(A=4423, B=952). The step 12–13 A files and their derivatives are invalid under
correct scoring. Only step 11's row order and values are confirmed correct.
