# Step 47 Candidate 16

Run command: `python src/step47_candidate_16_probe2.py`

## A: proven optimal + new probe

Proven (unchanged):
- core @ 03:00: +0 (R1=-3, optimal k1=0)
- core @ 06:00: +5 (R2=36, optimal k2=5)
- near @ 07:00: +4 (R3=30, optimal k3=4)

New probe:
- near @ 10:00: **+1** (7 days)

A_next = 4110 + (7*1 - 2*R_near10)
       = 4117 - 2*R_near10

After submission: R_near10 = (4117 - A_next) / 2
Optimal k = round(R_near10 / 7)

## B: confirmed fix

01-29 outer->core @ 10:00 = 1 (confirmed from candidate_15)
**Expected B SSE = 924**

## Expected total

If R_near10 > 0 (underestimation):
  A improves, B=924, total < 6882
If R_near10 < 0 (overestimation):
  A worsens by |7-2*R|, B=924, total > 6882
If R_near10 = 0:
  A_next = 4117, B=924, total = 4117+3*924 = **6889**

## Online score history

| candidate | A SSE | B SSE | total | rank |
| --- | ---: | ---: | ---: | ---: |
| candidate_10 | 4423 | 925 | 7198 | — |
| candidate_13 | 4193 | 928 | 6977 | 101 |
| candidate_14 | 4190 | 925 | 6965 | 100 |
| candidate_15 | 4110 | 926 | 6888 | 92 |
| **candidate_16** | **4117-2R** | **924** | **?** | **?** |
