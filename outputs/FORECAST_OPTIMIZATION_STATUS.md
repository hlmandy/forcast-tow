# Forecast Optimization Status

## Current Method

Use the validation daily vessel counts directly:

1. Learn the relationship between training daily vessel count and daily A total.
2. Predict validation daily A totals from the provided validation daily vessel counts.
3. Keep the stable A100 hourly shape, then rescale each day's 72 hourly-region cells to the daily target.
4. Keep B at the best known lower-migration component, because B has weak relationship with daily vessel count.

New outputs are CSV-only. No zip files should be generated.

## Current Candidate Directory

`outputs/20260710_144658_daily_count_constrained_csv/`

Recommended order:

1. `submit_02_A_daily_total_constraint25_B085`
   - A_sum=1817, B_sum=395
   - Move each daily A total 25% toward the validation-vessel-count regression target.
   - Online score: total 11421, A SSE 6063, B SSE 1786.
   - This is the current best confirmed candidate.

2. `submit_03_A_daily_total_constraint50_B085`
   - A_sum=1793, B_sum=395
   - Stronger daily-total constraint; higher risk because A085 was worse online.
   - Online score: total 11583, A SSE 6225, B SSE 1786.
   - Worse than 25%; do not strengthen the daily-total constraint further without a new reason.

3. `submit_01_A100_B085_anchor`
   - A_sum=1839, B_sum=395
   - Anchor: old stable A hourly shape plus B085.

All candidates were verified:

- A: 504 rows, columns `time_window, zone, vessel_count`, non-negative integers.
- B: 1008 rows, columns `time_window, source_zone, target_zone, vessel_count`, non-negative integers.
- No zip files in the candidate directory.

## Daily Targets

Daily A target totals from validation daily vessel count are in:

`outputs/20260710_144658_daily_count_constrained_csv/daily_total_targets.csv`

The 25% candidate daily A totals are:

- 2018-01-25: 238
- 2018-01-26: 262
- 2018-01-27: 288
- 2018-01-28: 266
- 2018-01-29: 258
- 2018-01-30: 258
- 2018-01-31: 247

## Online Results For This Method

- 25% daily-total constraint:
  - Submitted 2026-07-10 14:48
  - Total 11421
  - A SSE 6063
  - B SSE 1786

- 50% daily-total constraint:
  - Submitted 2026-07-10 14:51
  - Total 11583
  - A SSE 6225
  - B SSE 1786

Conclusion: the daily-total idea is correct, but 50% is already slightly too strong. Keep 25% as the current confirmed best.

## Remaining Output Directories

- `_model_sources/`: minimal source CSVs and online score evidence.
- `_official_examples_backup/`: official template CSVs.
- `20260709_112307_quality_cleaning/`: data quality diagnostics and labels.
- `20260710_144658_daily_count_constrained_csv/`: current submission candidates.

## Reproducible Command

```powershell
python src/package_daily_count_constrained_candidates.py
```
