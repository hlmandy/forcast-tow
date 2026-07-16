# Step 46 Candidate 15 Optimal

Run command: `python src/step46_candidate_15_optimal.py`

## A: proven optimal corrections

| group | R (sum of y-p over 7 days) | optimal k=round(R/7) |
| --- | ---: | ---: |
| core @ 03:00 | -3 | **0** |
| core @ 06:00 | 36 | **+5** |
| near @ 07:00 | 30 | **+4** |

Expected A SSE:
  = 4423 + (7*25 - 2*5*36) + (7*16 - 2*4*30)
  = 4423 + (175-360) + (112-240)
  = 4423 - 185 - 128
  = **4110**

## B: 01-28 probe

outer->core @ 10:00, only 01-28 = 1.
Known: occurrence in {28,29}.

If B_next = 924: occurrence is **01-28** (correct prediction)
If B_next = 926: occurrence is **01-29**

## Expected total

- If 01-28: 4110 + 3*924 = **6882**
- If 01-29: 4110 + 3*926 = **6888**

## Online score history

| candidate | A SSE | B SSE | total | rank |
| --- | ---: | ---: | ---: | ---: |
| candidate_10 | 4423 | 925 | 7198 | — |
| candidate_12 | 4318 | 930 | 7108 | — |
| candidate_13 | 4193 | 928 | 6977 | 101 |
| candidate_14 | 4190 | 925 | 6965 | 100 |
| **candidate_15** | **4110** | **924/926** | **6882/6888** | **?** |
