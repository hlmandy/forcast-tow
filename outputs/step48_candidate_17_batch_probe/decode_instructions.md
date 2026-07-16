# Step 48 Decode Instructions

Run command: `python src/step48_candidate_17_batch_probe.py`

## A decode

Base: A0_SSE = 4094 (candidate_10 + proven optimal: core@06h+5, near@07h+4, near@10h+2)
Probe G1: 近港区@12:00 +1 (7 days)
Probe G2: 外围区@14:00 +256 (7 days)

A_next = A0_SSE + (7*1 - 2*R_G1) + (7*256^2 - 2*256*R_G2)
       = 4094 + (7 - 2*R_G1) + (458752 - 512*R_G2)

Let D1 = A_next - 4094 - 458752
Then: D1 = 7 - 2*R_G1
=> **R_G1 = (7 - D1) / 2**

Optimal k_G1 = round(R_G1 / 7)
Then: R_G2 = (458752 - 2*R_G1 - (A_next - 4094)) ... wait

Actually: A_next = 4094 + (7 - 2*R_G1) + (458752 - 512*R_G2)
Let total_delta = A_next - 4094
Then: total_delta = 7 - 2*R_G1 + 458752 - 512*R_G2
=> 512*R_G2 = 7 + 458752 - 2*R_G1 - total_delta
=> **R_G2 = (7 + 458752 - 2*R_G1 - total_delta) / 512**

After deriving R_G1, R_G2:
Optimal corrections: k_G1 = round(R_G1/7), k_G2 = round(R_G2/7)
Optimal A = 4094 + min_k(7k^2-2kR_G1) + min_k(7k^2-2kR_G2)

## B decode

Base: B0_SSE = 924 (candidate_10 + 01-29 outer->core@10=1)
Probe cells (4 cells with base_B=16):

  Digit 1: 01-25 outer->near @ 17:00 = +1
  Digit 2: 01-26 near->outer @ 19:00 = +16
  Digit 3: 01-27 near->outer @ 11:00 = +256
  Digit 4: 01-28 outer->near @ 07:00 = +4096

B_next = B0_SSE + sum of digit_contributions
Each cell with true y=0 contributes +1 to B (from (increment-0)^2 - 0^2 = increment^2)
Each cell with true y>=1 contributes increment^2 - (increment-y)^2 = 2*increment*y - y^2

For y in {0,1} cells:
  y=0: contributes +increment^2 to B (worsening)
  y=1: contributes 2*increment - 1 (improvement if increment large)

To decode: compute B_next - 924. Then identify which digits are set.
Since increments are powers of 16, the binary representation of (B_next - 924) in base 16
reveals which cells have y>=1.
