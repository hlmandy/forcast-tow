# Step 08 B State Transition Audit

> Task B diagnostics only. No models, no backtest, no submission, no Task A.

Run command: `python src/step08_b_state_transition_audit.py`

## 1. Official-label consistency

- Vessel-hour state rows: 23953 (unique mmsi-hour key).
- 3456 hour-direction rows complete; reconstructed six-direction labels match `make_b_labels` exactly. **PASS**
- Daily six-direction totals match Step 01 `train_daily_overview` exactly. **PASS**
- All flow-conservation assertions (present=linkable+no_consecutive; linkable=3 destinations; linkable=stay+outflow; 2-direction sum=outflow; outflow=b_total) PASS.

## 2. State availability and linkability

| source | mean present | mean linkable | link_rate | mean stay | outflow_rate(linkable) | mean no_consecutive |
| --- | --- | --- | --- | --- | --- | --- |
| core | 12.14 | 12.01 | 0.953 | 10.71 | 0.105 | 0.13 |
| near | 24.02 | 23.27 | 0.937 | 21.82 | 0.059 | 0.75 |
| outer | 5.43 | 5.11 | 0.924 | 4.53 | 0.097 | 0.32 |

## 3. Direction-level sparsity

| direction | total flow | mean hourly | variance | var/mean | zero rate | max hourly |
| --- | --- | --- | --- | --- | --- | --- |
| core->near | 624 | 1.08 | 2.16 | 2.00 | 0.512 | 9 |
| core->outer | 124 | 0.22 | 0.29 | 1.34 | 0.832 | 3 |
| near->core | 633 | 1.10 | 1.87 | 1.70 | 0.448 | 9 |
| near->outer | 197 | 0.34 | 0.56 | 1.63 | 0.764 | 6 |
| outer->core | 107 | 0.19 | 0.23 | 1.23 | 0.851 | 3 |
| outer->near | 225 | 0.39 | 0.56 | 1.44 | 0.724 | 5 |

## 4. Effect of the china_coastal outage

| period | mean b_total | mean rep vessel-hours | mean linkable vessel-hours | overall link_rate |
| --- | --- | --- | --- | --- |
| pre_outage | 80.1 | 1077.0 | 1059.1 | 0.983 |
| degraded_period | 80.1 | 884.7 | 837.7 | 0.943 |
| post_recovery | 78.0 | 985.5 | 957.7 | 0.970 |

- **B did NOT collapse like A.** Mean daily `b_total` is essentially flat across periods (~80/80/78 for pre_outage/degraded/post_recovery), versus A which fell to ~0.44x. During the outage the source stock (`representative_vessel_hours`) and `overall_link_rate` dipped, but `overall_outflow_rate_among_linkable` rose enough to keep total migration roughly constant.
- Therefore A's `quality_regime` (defined from china_coastal record quality) is NOT appropriate as a B training weight: B's labels stayed healthy through the A outage. B would need its own, source-availability-based quality metric if any — but the ablation in section 5 shows B is already robust to losing a single source.

## 5. Source-ablation results

| configuration | overall SSE vs full | retained total ratio | exact match rate |
| --- | --- | --- | --- |
| full | 0 | 1.000 | 1.000 |
| without_china_coastal | 616 | 0.987 | 0.899 |
| without_e_globe_daily | 325 | 0.972 | 0.931 |
| without_f_globe_dynamic | 218 | 0.984 | 0.950 |
| china_coastal_only | 1807 | 0.623 | 0.876 |
| e_globe_daily_only | 872 | 0.970 | 0.842 |
| f_globe_dynamic_only | 2190 | 0.708 | 0.775 |

- Largest-deviation subset (by overall SSE vs full) per non-full config:
  - without_china_coastal: all (SSE=616)
  - without_e_globe_daily: all (SSE=325)
  - without_f_globe_dynamic: all (SSE=218)
  - china_coastal_only: all (SSE=1807)
  - e_globe_daily_only: all (SSE=872)
  - f_globe_dynamic_only: all (SSE=2190)

## 6. Relationship between stock and flow

| direction | pearson y~present(hourly) | pearson y~linkable(hourly) | pearson daily~vessels | pearson daily~source_present |
| --- | --- | --- | --- | --- |
| core->near | 0.167 | 0.175 | -0.216 | 0.121 |
| core->outer | 0.027 | 0.029 | -0.104 | -0.427 |
| near->core | 0.341 | 0.334 | -0.255 | -0.063 |
| near->outer | 0.240 | 0.228 | 0.212 | 0.206 |
| outer->core | 0.201 | 0.211 | -0.263 | 0.216 |
| outer->near | 0.429 | 0.441 | 0.259 | 0.255 |

- Descriptive only. The hourly flow correlates more with source stock (present/linkable, Pearson up to ~0.44) than with the daily total vessel count (mostly near zero or negative). So a 'source stock x transition probability' structure is more consistent with the data than scaling by daily vessel count, although the stock signal itself is only weak-to-moderate.

## 7. Hourly transition structure

| direction | mean pairwise Pearson | mean cosine | mean L1 | mean RMSE | peak hour (normal mean curve) | zero rate (normal) |
| --- | --- | --- | --- | --- | --- | --- |
| core->near | 0.176 | 0.477 | 1.1701 | 0.0714 | 6 | 0.502 |
| core->outer | 0.121 | 0.254 | 1.5949 | 0.1351 | 6 | 0.824 |
| near->core | 0.091 | 0.457 | 1.1563 | 0.0694 | 19 | 0.441 |
| near->outer | 0.116 | 0.296 | 1.5035 | 0.1110 | 19 | 0.752 |
| outer->core | 0.023 | 0.161 | 1.7491 | 0.1514 | 10 | 0.846 |
| outer->near | 0.137 | 0.328 | 1.4583 | 0.1061 | 7 | 0.725 |

## 8. Period and day-type comparison

- **normal_pre_outage vs normal_post_recovery** (mean hourly curves):

  | direction | pearson | cosine | L1 | RMSE | peak A | peak B |
  | --- | --- | --- | --- | --- | --- | --- |
  | core->near | 0.512 | 0.838 | 12.2576 | 0.7372 | 6 | 19 |
  | core->outer | 0.737 | 0.852 | 3.6515 | 0.1961 | 6 | 6 |
  | near->core | 0.369 | 0.854 | 13.4242 | 0.6759 | 19 | 20 |
  | near->outer | 0.591 | 0.819 | 5.2424 | 0.2723 | 19 | 19 |
  | outer->core | 0.231 | 0.655 | 4.1061 | 0.2136 | 10 | 6 |
  | outer->near | 0.476 | 0.759 | 6.2273 | 0.3763 | 13 | 7 |

- **normal_weekday vs normal_weekend** (mean hourly curves):

  | direction | pearson | cosine | L1 | RMSE | peak A | peak B |
  | --- | --- | --- | --- | --- | --- | --- |
  | core->near | 0.307 | 0.775 | 14.6154 | 0.8591 | 18 | 2 |
  | core->outer | 0.422 | 0.630 | 5.2885 | 0.3210 | 6 | 6 |
  | near->core | 0.336 | 0.853 | 13.0000 | 0.6680 | 19 | 3 |
  | near->outer | 0.564 | 0.784 | 6.5962 | 0.3851 | 19 | 11 |
  | outer->core | 0.112 | 0.524 | 4.1923 | 0.2560 | 10 | 2 |
  | outer->near | 0.374 | 0.693 | 8.7885 | 0.5294 | 13 | 1 |

- Small samples; differences are descriptive, not evidence of a structural break or day-of-week effect.

## 9. Implications for B modeling

- Keep direction-hourly-mean as the simple baseline, but note B's hourly shape is very unstable across normal days (mean pairwise Pearson 0.02-0.18, section 7) and flows are sparse (zero rate 0.44-0.85), so this baseline is expected to be weak.
- A 'source stock x transition probability' structure is worth building: flow correlates more with source present/linkable than with daily vessel count, and it naturally enforces the conservation identities that hold exactly in the data.
- Prefer `source_linkable` over `source_present` as the exposure denominator: only linkable vessels can migrate, so transition probability = flow / linkable is the well-defined quantity (present includes no_consecutive vessels that cannot migrate).
- Stay and no_consecutive matter for the mechanism (stay dominates the linkable mass; no_consecutive drives the link_rate dip in the outage) but the six official directions exclude stay, so a stock x transition model over the six directions captures what is scored.
- Do NOT use A's quality weights for B: B did not share A's outage collapse and is source-robust (section 4-5).
- Step 09 priority: compare (a) direction-hourly-mean baseline, (b) source_linkable x transition-probability (per direction/hour, with smoothing/shrinkage for the sparse directions), and possibly (c) a small regularized per-direction count model — judged on normal-target and final-analog views, not the contaminated rolling folds.
- No model is trained or scored in this stage.
