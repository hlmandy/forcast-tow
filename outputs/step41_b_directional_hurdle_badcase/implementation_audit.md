# Implementation Audit

## Candidate_10 B composition
- 5 directions (core→near, near→outer, outer→near, core→outer, outer→core): Step 10 pair_hour_hier_tau_scale_cv / independent
- near→core: Step 15 pair_default_hierarchical
- Online B SSE: 925 (corrected scoring)
- near→core replacement contribution: -27 (online confirmed)
- core→near replacement contribution: +15 (online confirmed, kept Step 11)

## This step
- A completely frozen (candidate_10)
- B baseline = candidate_10
- Tests Hurdle (occurrence × conditional mean) per direction
- Does NOT continue online per-direction probing
