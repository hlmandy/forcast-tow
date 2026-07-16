# Step 37 Implementation Audit

1. reshape_region_totals didn't preserve integer region totals (used np.rint per cell)
2. No alpha mixing tested (all methods at full replacement)
3. No oracle_best_historical_profile or oracle_convex_hull
4. profile_lag7_direct fell back to long for target_like (lag7 not normal)
5. Only 7 files generated, missing Oracles, inner CV, badcase repair

## This step
- Strict largest-remainder per region
- Two coverage Oracles
- Alpha mixing 0.10-1.00 with inner CV
- Per-day badcase analysis
