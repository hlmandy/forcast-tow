# 2026 CTS AIS Notebook Plan

## Goal

Create a runnable Jupyter Notebook that explains the CTS tugboat AIS task while showing code and outputs step by step. The notebook should help us understand the data, reproduce the official A/B label definitions, define cleaning principles, and prepare a baseline modeling framework.

## Current Scope

1. Explain the competition tasks.
   - Task A: hourly active tug count by ring region.
   - Task B: hourly inter-region migration count.
   - Clarify the exact official counting rules.

2. Inspect raw data.
   - File size, columns, dtypes, first rows.
   - Time range and daily coverage.
   - Unique vessels and ship type distribution.
   - Missing values and abnormal values.
   - SOG, ROT, COG, true heading, longitude, latitude summaries.

3. Understand temporal behavior.
   - Daily AIS record counts.
   - Daily unique vessel counts.
   - Hourly record counts.
   - Per-vessel AIS sampling interval distribution.

4. Understand spatial behavior.
   - Compute distance from official center point: 117.79E, 38.97N.
   - Assign each AIS point to:
     - core: 0-3 km
     - near: 3-10 km
     - outer: 10-30 km
     - outside: 30 km+
   - Visualize sampled AIS points by region.

5. Define cleaning strategy.
   - Keep official label construction conservative.
   - Do not over-clean data before generating labels.
   - For Task A, use only official SOG and minimum-record filters.
   - For Task B, do not filter by SOG; use representative region rules.
   - Treat extra cleaning as modeling features, not label changes.

6. Build Task A training labels.
   - Filter official three regions.
   - Filter active AIS points with `2 <= SOG <= 10`.
   - Count AIS points by `hour + region + mmsi`.
   - Keep vessel-region-hour groups with at least 3 records.
   - Count unique vessels by `hour + region`.
   - Fill missing hour-region combinations with 0.

7. Build Task B training labels.
   - Filter official three regions.
   - Count points by `mmsi + hour + region`.
   - Use most frequent region as representative region.
   - Break ties by latest timestamp.
   - Compare adjacent hours per vessel.
   - Count directed migrations for all 6 source-target pairs.
   - Fill missing hour-pair combinations with 0.

8. Use validation daily vessel counts.
   - Load validation daily unique vessel counts.
   - Compare with training daily vessel counts.
   - Use as future scale hint or exogenous feature.

9. Prepare baseline framework.
   - Hour-of-day mean.
   - Day-of-week + hour mean.
   - Recent 7-day mean.
   - Mixed mean baseline.
   - Daily vessel-count calibration.

10. Prepare backtesting framework.
    - Simulate validation inside training data.
    - Example split: train 2018-01-01 to 2018-01-17, validate 2018-01-18 to 2018-01-24.
    - Compute `Score = SSE_A + 3 * SSE_B`.

## Immediate Next Steps

1. Rename the notebook to a stable ASCII path:
   - `notebooks/01_task_understanding_eda_cleaning_framework.ipynb`

2. Execute the notebook.
   - Confirm all key cells run.
   - Keep outputs in the notebook if execution succeeds.

3. Fix any runtime or dtype issues.

4. Add concrete observations from the actual run.

5. Summarize notebook location, structure, and next modeling direction.

## Notes

- The current priority is correctness of task understanding and label construction.
- Complex models should come after the official A/B aggregation logic is validated.
- The validation file only gives daily vessel counts, not validation AIS details, so final prediction must rely on training-period temporal patterns and daily count calibration.
