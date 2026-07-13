# Step 25 Implementation Audit

## Missing components
- `vessel_hierarchical_probability` was NOT implemented. Only `vessel_latest_weekly_pattern` (raw pattern copy) was coded.
- `baseline_step11` was approximated as 'mean of normal dates', not the exact decomp_mean_daytype_shrunk model.
- `vessel_latest_weekly_pattern` summed ALL historical vessel patterns without predicting vessel presence.

## Weekly pair count discrepancy
- Console reported 10/10/5 but summary.md had 4/6/3. The console values include pairs where both dates are normal; the summary values may have used a different filter. This discrepancy confirms Step 25 had consistency issues.

## What Step 25 can and cannot rule out
- CAN rule out: direct same-weekday 72-vector copying (worse by 40-165%).
- CAN rule out: summing all historical vessel patterns without presence prediction (worse by 165%).
- CANNOT rule out: vessel presence probability models with U_d calibration.
- CANNOT rule out: conditional activity profiles with hierarchical shrinkage.
