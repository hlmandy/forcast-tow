# Step 12 — Recombined candidate_04

Combines two already-submitted files with no retraining, reordering, or modification.

## Sources

- **A** (A SSE=6063 online): `outputs\20260710_144658_daily_count_constrained_csv\submit_02_A_daily_total_constraint25_B085\提交结果1_区域活跃拖轮数量.csv` — copied byte-for-byte.
- **B** (B SSE=1762 online, from step11 candidate_01_main): `outputs\step11_final_validation_candidates\submissions\candidate_01_main\提交结果2_圈层间拖轮迁移量.csv` — copied byte-for-byte.

## Verification

- A rows: 504 (expected 504). **PASS**
- A total vessel_count: 1817 (expected 1817). **PASS**
- A daily totals match the 7 expected values exactly. **PASS**
- B rows: 1008 (expected 1008). **PASS**
- A SHA256 source == output: True. **PASS**
- B SHA256 source == output: True. **PASS**
- A normalized content/order identical: True. **PASS**
- B normalized content/order identical: True. **PASS**

## Daily totals (B)

B total = 427.

## Note

This candidate is a pure recombination of two confirmed submission files. No model was retrained, no value recomputed, no row reordered.
