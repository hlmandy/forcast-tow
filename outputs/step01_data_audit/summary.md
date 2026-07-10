# Step 01 Data Audit

> Objective statistics only. No model decisions, no anomaly deletion, no submissions.

Run command: `python src/step01_data_audit.py`

## 1. Input information

- Training date range: 2018-01-01 ~ 2018-01-24
- Number of AIS records: 2,039,846
- Number of unique vessels (whole training period): 109
- Number of data sources: 3 (china_coastal, e_globe_daily, f_globe_dynamic)
- Validation date range: 2018-01-25 ~ 2018-01-31

## 2. Completeness checks

- Missing training dates: none
- Dates with fewer than 24 observed hours: 2 — 2018-01-14(18h), 2018-01-15(23h)
- Hours with zero AIS records (on the 24x24 grid): 7
  - 2018-01-14 14:00, 2018-01-14 15:00, 2018-01-14 17:00, 2018-01-14 19:00, 2018-01-14 20:00, 2018-01-14 22:00, 2018-01-15 7:00
- Five dates with the lowest AIS record counts:
  - 2018-01-15: 6,632 records
  - 2018-01-14: 6,849 records
  - 2018-01-17: 7,786 records
  - 2018-01-16: 9,971 records
  - 2018-01-18: 10,271 records
- Five dates with the lowest median records per vessel-hour:
  - 2018-01-18: 7.00
  - 2018-01-15: 8.00
  - 2018-01-16: 8.00
  - 2018-01-17: 8.00
  - 2018-01-13: 9.00
- Source-level zero-record (date, source) cells: 4
  - (2018-01-13, china_coastal), (2018-01-14, china_coastal), (2018-01-15, china_coastal), (2018-01-16, china_coastal)

## 3. Daily vessel-count relationships

Pearson and Spearman correlations between daily `unique_vessel_count` and A/B totals.

| feature | pearson | spearman |
| --- | --- | --- |
| a_total | 0.3806 | 0.3300 |
| a_core_total | 0.3833 | 0.3982 |
| a_near_total | 0.3733 | 0.3511 |
| a_outer_total | 0.2830 | 0.3324 |
| b_total | -0.1215 | -0.0206 |
| b_core_to_near | -0.3648 | -0.2305 |
| b_core_to_outer | 0.2385 | 0.3058 |
| b_near_to_core | -0.2511 | -0.1467 |
| b_near_to_outer | 0.2475 | 0.1021 |
| b_outer_to_core | -0.1701 | -0.1814 |
| b_outer_to_near | 0.3299 | 0.3727 |

These correlations are descriptive only and do not fix any model choice.

## 4. Validation comparison

| set | min | max | mean |
| --- | --- | --- | --- |
| training | 46 | 61 | 53.29 |
| validation | 47 | 55 | 50.43 |

- Validation values exceed the training range: no (training 46-61, validation 47-55).

## 5. Detected anomalies

Dates and hours with obvious missing or abnormally sparse data. Nothing is deleted here.

- Dates with gaps (fewer than 24 observed hours):
  - 2018-01-14: only 18 hours observed
  - 2018-01-15: only 23 hours observed
- (date, hour) cells with zero AIS records: 7 (listed in section 2)
- Low-sampling-density days (lowest quartile of median records per vessel-hour):
  - 2018-01-13: 9.00
  - 2018-01-14: 9.00
  - 2018-01-15: 8.00
  - 2018-01-16: 8.00
  - 2018-01-17: 8.00
  - 2018-01-18: 7.00
