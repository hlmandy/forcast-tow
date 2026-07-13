# Implementation Audit

## Step 31
- MMSI alignment fixed: row-level 52.7%, match-any 98.4%
- Incremental stock Oracle target_like: +0.2% (common), +0.7% (true daily factor)
- Stock-emission route CLOSED
- oracle_abs/inc should be deployable=False (uses true target stock)

## Step 05 PCA
- Only analyzed normalized true profiles, NOT Step 11 OOF residuals
- PC1 ~23%, 3 PCs ~49-57% — insufficient for 'low-dim' conclusion on profiles
- This step analyzes RESIDUALS, not profiles
