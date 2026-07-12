# Step 18 B Single-Direction Control

## Known online scores

- Baseline B SSE: 1762
- Both-direction hybrid B SSE: 1790
- Combined SSE change: +28

## Candidate design

- Only core->near is replaced (from step17 hybrid)
- near->core is restored to baseline
- Other four directions remain baseline
- A remains the confirmed A SSE=4423 file

## Exact interpretation after submission

Let this candidate's online B SSE be S_core.

Then:

- core_to_near_delta = S_core - 1762
- near_to_core_delta = 28 - core_to_near_delta
- near_to_core_only_sse = 1762 + near_to_core_delta = **3552 - S_core**

Decision rules:

- S_core < 1762: keep only core->near
- 1762 < S_core < 1790: neither direction should be kept
- S_core > 1790: core->near is harmful; near->core alone may help
- S_core = 1762: core->near is neutral; all degradation comes from near->core
