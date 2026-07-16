# Step 36 Implementation Audit

1. Step 36 only implemented C-total methods; NO C-profile methods tested
2. lag7-normal subset for target_like = 0 days
3. lag7_or_lag14 didn't actually use lag14 profile
4. Step 36 replaced Step 11 entirely; didn't test profile-only reshape
5. Therefore 'C profile unpredictable' was NOT tested

## Correlation reports
- corr(C_total, U_d) all dates: 0.273
- corr(C_total, U_d) normal dates: 0.086
