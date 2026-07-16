# Step 43 Candidate 12 Joint Fix

Run command: `python src/step43_candidate_12_joint_fix.py`

## Candidate_11 B assembly bug

candidate_11's B was built from Step17's B file (which contains core→near
changes from pair_default_hierarchical). This leaked 21 core→near changes
that were known harmful (+15 B SSE). The 'B unchanged' claim was wrong.

## Candidate_12 A

Reverses candidate_11's A -1 corrections to **+1**:
- core @ 03:00, 06:00 → +1 (all 7 days)
- near @ 07:00 → +1 (all 7 days)
- Total: 21 cells changed

From candidate_11's online result (A=4570, ΔA=+147 from -1 corrections):
  Σ(y-p) = 63, average underestimate = 3 per cell.
Reversing to +1: ΔSSE = 21 - 2×63 = **-105**

**Expected A SSE = 4423 - 105 = 4318**

## Candidate_12 B

outer→core hour h*=10: 7 cells changed 0→1 (worst case ΔB ≤ +7)
All other directions (including core→near and near→core) = candidate_10 exact.

**Expected B SSE ≤ 925 + 7 = 932**

## Total bound

4318 + 3 × 932 = **7114** (worst case)

If B h* is correct (gain > 0): total could be ≤ 4318 + 3×925 = **7093**
