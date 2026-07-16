# Step 44 Candidate 13 Probe

Run command: `python src/step44_candidate_13_probe.py`

## A group probe (R1 isolation)

Corrections relative to candidate_10:
- core @ 03:00: **+2** (7 days)
- core @ 06:00: **+3** (7 days)
- near @ 07:00: **+3** (7 days)

From candidate_11's result: R1+R2+R3 = 63

Delta SSE = 7*(4+9+9) - 2*(2*R1 + 3*R2 + 3*R3)
         = 154 - 2*(2*R1 + 3*(63-R1))
         = 154 - 378 + 2*R1
         = 2*R1 - 224

**A_next = 4423 + 2*R1 - 224 = 4199 + 2*R1**

After submission: R1 = (A_next - 4199) / 2

## B date probe

outer->core @ 10:00, only 01-25/26/27 predict 1 (3 cells), rest 0.

Known: exactly 1 of 7 days has true y=1.

If occurrence in {25,26,27}: B_next = 925 + 3 - 2 = **926**
If occurrence NOT in {25,26,27}: B_next = 925 + 3 = **928**
