# Step 39 Tide Phase Profile

Run command: `python src/step39_a_tide_phase_profile.py`

## Clock vs Tide alignment
- Clock mean corr: 0.382
- Tide mean corr: 0.618
- Improvement: +0.236
- Threshold (>= +0.10): PASS

## Circular shift Oracle

| block | SSE | reduction |
| --- | --- | --- |
| block_pre_normal | 2277 | 16.6% |
| block_target_like | 2937 | 13.6% |
| block_post_outage_stress | 3406 | 12.0% |

## Decision
- Tide alignment improvement >= 0.10: YES
- Circular shift Oracle >= 10%: YES