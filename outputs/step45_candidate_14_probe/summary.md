# Step 45 Candidate 14 Probe (R2 + B binary)

Run command: `python src/step45_candidate_14_probe.py`

## A R2 probe

core@03h: **0** (candidate_10, R1=-3 proven, optimal k1=0)
core@06h: **+2** (7 days)
near@07h: **+3** (7 days)

Delta SSE = 7*(4+9) - 2*(2*R2 + 3*R3)
         = 91 - 2*(2*R2 + 3*(66-R2))
         = 91 - 2*(198 - R2)
         = 2*R2 - 305

**A_next = 4423 + 2*R2 - 305 = 4118 + 2*R2**

After submission: R2 = (A_next - 4118) / 2
Then: R3 = 66 - R2
Optimal: k1=0, k2=round(R2/7), k3=round(R3/7)

## B binary search

outer->core @ 10:00, only 01-28/29 = 1.
Known: occurrence in {28,29,30,31}.

If B_next = 925: occurrence in {28,29}
If B_next = 927: occurrence in {30,31}
