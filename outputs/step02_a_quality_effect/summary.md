# Step 02 A Quality Effect

> Diagnostic only. No deletions, no imputation, no model selection.

Run command: `python src/step02_a_quality_effect.py`

## 1. Input and consistency checks

- Fixed training date range: 2018-01-01 ~ 2018-01-24 (24 days)
- Number of AIS records: 2,039,846
- Number of unique vessels (whole period): 109
- Data sources: china_coastal, e_globe_daily, f_globe_dynamic
- Missing training dates (from raw AIS): none
- Consistency assertion `a_label == active_vessels_ge3` over all 1728 (day,hour,region) cells: **PASS**
- Consistency assertion `qualified_ge3_vessel_hours == a_total` over all 24 days: **PASS**

## 2. Source-level completeness

Total records and per-day record-count statistics (over the fixed 24-day range, zero days included).

| source | total records | daily min | daily median | daily max |
| --- | --- | --- | --- | --- |
| china_coastal | 1,817,446 | 0 | 89630.5 | 139,379 |
| e_globe_daily | 149,844 | 3,809 | 6027.0 | 10,393 |
| f_globe_dynamic | 72,556 | 10 | 3130.0 | 6,161 |

- Median china_coastal record count over non-zero days (used as `M_china`): 96,480.5
- Consecutive zero-record intervals per source:
  - china_coastal: 2018-01-13~2018-01-16 (4d)
  - e_globe_daily: none
  - f_globe_dynamic: none

## 3. A-filter funnel

Aggregate vessel-hours per region over all 24 days x 24 hours. `retention_active_to_ge3` = sum(active_vessels_ge3) / sum(active_region_vessel_count).

| region | raw_vessel_hours | active_vessel_hours | ge2 | ge3 | ge4 | retention active->ge3 |
| --- | --- | --- | --- | --- | --- | --- |
| core | 8063 | 4050 | 3710 | 3409 | 3220 | 0.8417 |
| near | 15359 | 2762 | 2405 | 2171 | 1973 | 0.7860 |
| outer | 3439 | 1303 | 1113 | 1013 | 936 | 0.7774 |

- Overall retention active->ge3: 6593/8115 = 0.8124

## 4. Quality regimes

`M_china` = 96,480.5 (median of china_coastal daily records over non-zero days).

- **outage** (4 days): 2018-01-13, 2018-01-14, 2018-01-15, 2018-01-16
- **severe** (2 days): 2018-01-17, 2018-01-18
- **reduced** (1 day): 2018-01-12
- **normal** (17 days): 2018-01-01, 2018-01-02, 2018-01-03, 2018-01-04, 2018-01-05, 2018-01-06, 2018-01-07, 2018-01-08, 2018-01-09, 2018-01-10, 2018-01-11, 2018-01-19, 2018-01-20, 2018-01-21, 2018-01-22, 2018-01-23, 2018-01-24

- Mean A total on `normal` days: 324.29
- Mean A total on `degraded` (outage+severe) days: 141.50

## 5. Relationships under different quality subsets

Pearson / Spearman correlations between each predictor and each A target, within each day subset. Empty cell = too few days or a constant variable.

| subset | target | n_days | pearson daily_vessels | spearman daily_vessels | pearson ais_records | pearson active_records | pearson qualification_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | a_total | 24 | 0.3806 | 0.3300 | 0.9018 | 0.9321 | 0.9343 |
| all | a_core_total | 24 | 0.3833 | 0.3982 | 0.8928 | 0.9159 | 0.9293 |
| all | a_near_total | 24 | 0.3733 | 0.3511 | 0.9039 | 0.9265 | 0.9328 |
| all | a_outer_total | 24 | 0.2830 | 0.3324 | 0.6826 | 0.7503 | 0.6992 |
| normal | a_total | 17 | 0.0703 | 0.0692 | 0.5367 | 0.7627 | 0.7194 |
| normal | a_core_total | 17 | 0.0986 | 0.2285 | 0.5111 | 0.7161 | 0.7079 |
| normal | a_near_total | 17 | 0.0699 | 0.1014 | 0.5210 | 0.7047 | 0.6940 |
| normal | a_outer_total | 17 | -0.0335 | 0.0446 | 0.3445 | 0.5660 | 0.4135 |
| usable | a_total | 18 | -0.0137 | -0.0249 | 0.6383 | 0.8067 | 0.7555 |
| usable | a_core_total | 18 | 0.0332 | 0.1487 | 0.5788 | 0.7393 | 0.6985 |
| usable | a_near_total | 18 | -0.0417 | 0.0073 | 0.6706 | 0.7946 | 0.7943 |
| usable | a_outer_total | 18 | -0.0756 | -0.0203 | 0.4093 | 0.5866 | 0.4504 |
| degraded | a_total | 6 | 0.2230 | 0.1739 | 0.9860 | 0.8089 | 0.3920 |
| degraded | a_core_total | 6 | -0.1130 | -0.2609 | 0.8801 | 0.8029 | 0.1624 |
| degraded | a_near_total | 6 | 0.3887 | 0.3769 | 0.9781 | 0.6595 | 0.3961 |
| degraded | a_outer_total | 6 | 0.4781 | 0.4058 | 0.7343 | 0.6447 | 0.5946 |

Key comparisons (predictor vs `a_total`):
- all days: daily_vessels pearson=0.3806 spearman=0.3300; ais_records pearson=0.9018; active_records pearson=0.9321; qualification_rate pearson=0.9343
- normal days: daily_vessels pearson=0.0703 spearman=0.0692; ais_records pearson=0.5367; active_records pearson=0.7627; qualification_rate pearson=0.7194

## 6. Conclusions supported by this audit

- The A label and the `>=3 active records` funnel are identical by construction (both assertions PASS), so all A variation below is real label variation, not a reconstruction artifact.
- Mean A total drops from 324.3 on normal days to 141.5 on degraded days (0.44x).
- Source-level completeness shows the china_coastal outage that drives the degraded regime (see section 2).
- These results diagnose the data-quality effect only. They do NOT decide which days to delete, down-weight, or repair, and they do NOT choose a prediction model. Those decisions belong to later steps.
