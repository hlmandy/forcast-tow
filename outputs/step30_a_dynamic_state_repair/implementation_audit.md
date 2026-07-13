# Step 29 Implementation Audit

## Confirmed issues
1. A-stock alignment used min(A_count, stock_count), not MMSI-level matching
2. Transition conservation was never constructed or asserted
3. Exit was rarely counted (only non-consecutive states from simplified rep construction)
4. Entry was approximated as max(0, stock_next - stock_now), not actual gross entry
5. 23→0 transition used wrong hour index
6. Global parameters (transition, entry, mean stock, Oracle emission) included target dates
7. Same Oracle SSE (3496) was reused for different blocks with different date ranges
8. ρ and β were selected by looking at target block SSE, not inner CV
9. target_like baseline used Step 07 final_analog (train_end=01-18), not train_end=01-19
10. pre-normal block, activity factor, lag-24 ridge, fusion, one-day origins not implemented

## Conclusion
Step 29 results are invalid for decision-making. All Oracle, Markov, and alignment numbers
must be recomputed with correct implementation.
