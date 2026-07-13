# Step 30 A Dynamic State Repair

Run command: `python src/step30_a_dynamic_state_repair.py`

## MMSI-level A-stock alignment
- Total A-qualified vessel-hour-region: 6593
- Same region: 0 (0.0000)
- Different region: 0
- Multi A region: 3027
- No rep state: 3566

## Transition conservation
- Checked: 1725, Passed: 1725, Failed: 0

## Per-block Oracle (no leakage)

| block | Oracle SSE | baseline | reduction |
| --- | --- | --- | --- |
| block_pre_normal | 2461 | 2729 | 9.8% |
| block_target_like | 3453 | 3399 | -1.6% |
| block_post_outage_stress | 3834 | 3872 | 1.0% |

## Markov results

| block | Markov SSE | MR SSE | MR rho | baseline |
| --- | --- | --- | --- | --- |
| block_pre_normal | 19003411400786956039532960013682783289344 | 16010853354314425102551120770806740680704 | 0.5 | 2729 |
| block_target_like | 26409255953355737724945333562248467054592 | 24686121430174201984170183710906169425920 | 0.5 | 3399 |
| block_post_outage_stress | 32339998652746021735489344326333307551744 | 30509339262372395940376821511797799387136 | 0.5 | 3872 |

## Decision
- Oracle passes 5% on both normal blocks: NO
- Alignment rate > 90%: NO