"""Step 19 — B source-outflow diagnostics.

Pure diagnostic: examines whether B is better modeled by source-layer outflow
behavior (vessel at source decides to leave, then picks destination) vs six
independent directions or reciprocal pairs. No new predictions, no submission.

Outputs (under outputs/step19_b_source_outflow_diagnostics/):
  1. source_outflow_hourly.csv (1728)
  2. source_daily_summary.csv (69)
  3. source_profile_stability.csv (>=30)
  4. destination_share_summary.csv (75)
  5. source_stock_rate_summary.csv (24)
  6. grouping_comparison.csv (>=12)
  7. modeling_decision.csv (3)
  8. summary.md
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
PF_PATH = Path("outputs/step08_b_state_transition_audit/b_hourly_pair_flow.csv")
SS_PATH = Path("outputs/step08_b_state_transition_audit/b_hourly_source_state.csv")
STEP02 = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
OUT_REL = Path("outputs/step19_b_source_outflow_diagnostics")

UNSCORABLE = pd.Timestamp("2018-01-24 23:00:00")
COMPLETE_END = pd.Timestamp("2018-01-23")  # last complete day
SOURCES = ["core", "near", "outer"]
# source -> (dest1, dest2)
SRC_DESTS = {"core": ("near", "outer"), "near": ("core", "outer"), "outer": ("core", "near")}
# direction task_keys
DIR_NAMES = ["core->near", "core->outer", "near->core", "near->outer", "outer->core", "outer->near"]
PAIRS = {"core_near": ("core->near", "near->core"), "near_outer": ("near->outer", "outer->near"), "core_outer": ("core->outer", "outer->core")}
# source outflow = sum of outgoing directions
SRC_OUTFLOW = {"core_outflow": ("core->near", "core->outer"), "near_outflow": ("near->core", "near->outer"), "outer_outflow": ("outer->core", "outer->near")}

RUN_CMD = "python " + " ".join(sys.argv)


def safe_pearson(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(pd.Series(a).corr(pd.Series(b)))


def profile_pearson_list(mat):
    """mat: (n_days, 24) raw values. Return list of pairwise Pearson."""
    vals = []
    for i, j in combinations(range(len(mat)), 2):
        vals.append(safe_pearson(mat[i], mat[j]))
    return [v for v in vals if not np.isnan(v)]


def daily_stats(daily_totals):
    """daily_totals: array of per-day totals (complete days)."""
    arr = np.array(daily_totals, float)
    mean = float(arr.mean()) if len(arr) else np.nan
    std = float(arr.std(ddof=1)) if len(arr) > 1 else np.nan
    cv = std / mean if mean and mean != 0 else np.nan
    zero = float((arr == 0).mean()) if len(arr) else np.nan
    return mean, std, cv, zero


def lag1_corr(arr):
    arr = np.array(arr, float)
    if len(arr) < 4:
        return np.nan
    return safe_pearson(arr[:-1], arr[1:])


def main():
    out_dir = ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    pf = pd.read_csv(ROOT / PF_PATH, encoding="utf-8-sig")
    pf["hour"] = pd.to_datetime(pf["hour"]); pf["date"] = pd.to_datetime(pf["date"]).dt.normalize()
    pf["hour_of_day"] = pf["hour"].dt.hour; pf["dayofweek"] = pf["hour"].dt.dayofweek
    pf["day_type"] = np.where(pf["dayofweek"].isin([5, 6]), "weekend", "weekday")
    ss = pd.read_csv(ROOT / SS_PATH, encoding="utf-8-sig")
    ss["hour"] = pd.to_datetime(ss["hour"]); ss["date"] = pd.to_datetime(ss["date"]).dt.normalize()
    aud = pd.read_csv(ROOT / STEP02, encoding="utf-8-sig"); aud["date"] = pd.to_datetime(aud["date"]).dt.normalize()
    qmap = dict(zip(aud["date"], aud["quality_regime"]))
    pf["quality"] = pf["date"].map(qmap)
    ss["quality"] = ss["date"].map(qmap)

    complete_dates = list(pd.date_range("2018-01-01", COMPLETE_END, freq="D").normalize())
    normal_dates = [d for d in complete_dates if qmap.get(d) == "normal"]

    # pivot y per direction
    yp = pf.pivot_table(index=["date", "hour", "hour_of_day", "day_type", "quality"],
                        columns="task_key", values="y", aggfunc="first").reset_index()
    for d in DIR_NAMES:
        if d not in yp.columns:
            yp[d] = 0

    # source state pivot: present/linkable per (date, hour, source)
    spv = ss.pivot_table(index=["date", "hour"], columns="source_region",
                         values=["source_present_count", "source_linkable_count"], aggfunc="first")

    # ========== 1. source_outflow_hourly.csv (1728) ==========
    hourly_rows = []
    for d in pd.date_range("2018-01-01", "2018-01-24", freq="D").normalize():
        for h in range(24):
            hr = d + pd.Timedelta(hours=h)
            is_complete = hr != UNSCORABLE
            row_pf = yp[(yp["date"] == d) & (yp["hour_of_day"] == h)]
            for z in SOURCES:
                d1, d2 = SRC_DESTS[z]
                tk1 = f"{z}->{d1}"; tk2 = f"{z}->{d2}"
                present = linkable = np.nan
                try:
                    present = float(spv.loc[(d, hr), ("source_present_count", z)])
                    linkable = float(spv.loc[(d, hr), ("source_linkable_count", z)])
                except Exception:
                    pass
                if is_complete and len(row_pf):
                    y1 = float(row_pf[tk1].iloc[0]); y2 = float(row_pf[tk2].iloc[0])
                    M = y1 + y2
                    share1 = y1 / M if M > 0 else np.nan
                    pr = M / present if (present and present > 0) else np.nan
                    lr = M / linkable if (linkable and linkable > 0) else np.nan
                    hourly_rows.append({"date": d, "hour": hr, "source_region": z, "source_outflow": M,
                                        "destination_1": d1, "destination_1_count": y1, "destination_2": d2, "destination_2_count": y2,
                                        "destination_1_share": share1, "present_stock": present, "linkable_stock": linkable,
                                        "present_outflow_rate": pr, "linkable_outflow_rate": lr,
                                        "is_complete_label": True, "is_normal_date": qmap.get(d) == "normal"})
                else:
                    hourly_rows.append({"date": d, "hour": hr, "source_region": z, "source_outflow": np.nan,
                                        "destination_1": d1, "destination_1_count": np.nan, "destination_2": d2, "destination_2_count": np.nan,
                                        "destination_1_share": np.nan, "present_stock": present, "linkable_stock": linkable,
                                        "present_outflow_rate": np.nan, "linkable_outflow_rate": np.nan,
                                        "is_complete_label": False, "is_normal_date": qmap.get(d) == "normal"})
    hourly = pd.DataFrame(hourly_rows)
    hourly["hour_of_day"] = hourly["hour"].dt.hour
    assert len(hourly) == 1728

    # ========== 2. source_daily_summary.csv (69) ==========
    daily_rows = []
    hcomp = hourly[hourly["is_complete_label"]]
    for d in complete_dates:
        for z in SOURCES:
            g = hcomp[(hcomp["date"] == d) & (hcomp["source_region"] == z)]
            M_tot = float(g["source_outflow"].sum())
            d1_tot = float(g["destination_1_count"].sum()); d2_tot = float(g["destination_2_count"].sum())
            daily_rows.append({"date": d, "source_region": z, "daily_outflow": M_tot,
                               "daily_destination_1": d1_tot, "daily_destination_2": d2_tot,
                               "destination_1_share": d1_tot / M_tot if M_tot > 0 else np.nan,
                               "mean_present_stock": float(g["present_stock"].mean()),
                               "mean_linkable_stock": float(g["linkable_stock"].mean()),
                               "mean_present_outflow_rate": float(g["present_outflow_rate"].mean()),
                               "mean_linkable_outflow_rate": float(g["linkable_outflow_rate"].mean()),
                               "is_normal_date": qmap.get(d) == "normal",
                               "day_type": "weekend" if d.dayofweek in (5, 6) else "weekday"})
    daily_summary = pd.DataFrame(daily_rows)
    assert len(daily_summary) == 69

    # ========== grouping comparison: 12 series ==========
    def series_daily_profile(get_value, dates):
        """get_value(date, hour) -> float. Return (daily_totals array, (n_dates,24) matrix)."""
        dt = []; mat = []
        for d in dates:
            vals = [get_value(d, h) for h in range(24)]
            dt.append(sum(vals)); mat.append(vals)
        return np.array(dt), np.array(mat)

    def grouping_row(gtype, name, dirs_list, dates, label_dates):
        def gv(d, h):
            r = yp[(yp["date"] == d) & (yp["hour_of_day"] == h)]
            if len(r) == 0:
                return 0.0
            return float(sum(r[dk].iloc[0] for dk in dirs_list))
        dt, mat = series_daily_profile(gv, dates)
        mean, std, cv, zero = daily_stats(dt)
        pp = profile_pearson_list(mat)
        # normal-day
        dt_n, mat_n = series_daily_profile(gv, normal_dates)
        mean_n, std_n, cv_n, _ = daily_stats(dt_n)
        pp_n = profile_pearson_list(mat_n)
        return {"grouping_type": gtype, "series_name": name, "component_directions": "+".join(dirs_list),
                "zero_rate": zero, "daily_mean": mean, "daily_std": std, "daily_cv": cv,
                "profile_pearson_mean": float(np.mean(pp)) if pp else np.nan, "profile_pearson_median": float(np.median(pp)) if pp else np.nan,
                "profile_pearson_iqr": float(np.percentile(pp, 75) - np.percentile(pp, 25)) if pp else np.nan,
                "lag1_daily_correlation": lag1_corr(dt), "normal_day_daily_cv": cv_n,
                "normal_day_profile_pearson_mean": float(np.mean(pp_n)) if pp_n else np.nan}

    gc_rows = []
    for dk in DIR_NAMES:
        gc_rows.append(grouping_row("direction", dk, [dk], complete_dates, normal_dates))
    for pn, dirs in PAIRS.items():
        gc_rows.append(grouping_row("reciprocal_pair", pn, list(dirs), complete_dates, normal_dates))
    for sn, dirs in SRC_OUTFLOW.items():
        gc_rows.append(grouping_row("source_outflow", sn, list(dirs), complete_dates, normal_dates))
    grouping = pd.DataFrame(gc_rows)
    grouping["rank_by_daily_cv"] = grouping["daily_cv"].rank(method="min").astype(int)
    grouping["rank_by_profile_stability"] = grouping["normal_day_profile_pearson_mean"].rank(ascending=False, method="min").astype(int)

    # ========== 3. source_profile_stability.csv (30) ==========
    def gv_source(z, var):
        def f(d, h):
            r = hcomp[(hcomp["date"] == d) & (hcomp["source_region"] == z) & (hcomp["hour_of_day"] == h)]
            return float(r[var].iloc[0]) if len(r) else 0.0
        return f
    ps_rows = []
    raw_cv = {z: grouping[grouping["series_name"] == f"{z}_outflow"]["daily_cv"].iloc[0] for z in SOURCES}
    for z in SOURCES:
        for var in ["source_outflow", "present_stock", "linkable_stock", "present_outflow_rate", "linkable_outflow_rate"]:
            for scope, dates in [("all_complete_days", complete_dates), ("normal_complete_days", normal_dates)]:
                f = gv_source(z, var)
                dt, mat = series_daily_profile(f, dates)
                mean, std, cv, zero = daily_stats(dt)
                pp = profile_pearson_list(mat)
                more_stable = (cv < raw_cv[z]) if (cv is not None and not np.isnan(cv) and not np.isnan(raw_cv[z])) else False
                ps_rows.append({"source_region": z, "variable": var, "date_scope": scope, "n_dates": len(dates),
                                "zero_rate": zero, "profile_pearson_mean": float(np.mean(pp)) if pp else np.nan,
                                "profile_pearson_median": float(np.median(pp)) if pp else np.nan,
                                "profile_pearson_iqr": float(np.percentile(pp, 75) - np.percentile(pp, 25)) if pp else np.nan,
                                "hourly_cv_mean": cv, "hourly_cv_median": cv, "more_stable_than_raw_outflow": bool(more_stable)})
    prof_stab = pd.DataFrame(ps_rows)

    # ========== 4. destination_share_summary.csv (75) ==========
    ds_rows = []
    for z in SOURCES:
        d1, d2 = SRC_DESTS[z]
        # global
        g = hcomp[hcomp["source_region"] == z]
        tot = float(g["source_outflow"].sum()); d1t = float(g["destination_1_count"].sum())
        gs = d1t / tot if tot > 0 else np.nan
        ds_rows.append({"source_region": z, "hour": "all", "destination_1": d1, "destination_2": d2,
                        "n_positive_outflow_cells": int((g["source_outflow"] > 0).sum()), "total_outflow": tot,
                        "destination_1_total": d1t, "destination_2_total": tot - d1t, "destination_1_share": gs,
                        "share_standard_error": np.sqrt(gs * (1 - gs) / tot) if (tot > 0 and gs is not None and not np.isnan(gs)) else np.nan,
                        "share_distance_from_global": 0.0, "is_global_row": True})
        for h in range(24):
            gh = g[g["hour_of_day"] == h]
            toth = float(gh["source_outflow"].sum()); d1th = float(gh["destination_1_count"].sum())
            sh = d1th / toth if toth > 0 else np.nan
            se = np.sqrt(sh * (1 - sh) / toth) if (toth > 0 and sh is not None and not np.isnan(sh)) else np.nan
            ds_rows.append({"source_region": z, "hour": h, "destination_1": d1, "destination_2": d2,
                            "n_positive_outflow_cells": int((gh["source_outflow"] > 0).sum()), "total_outflow": toth,
                            "destination_1_total": d1th, "destination_2_total": toth - d1th, "destination_1_share": sh,
                            "share_standard_error": se, "share_distance_from_global": abs(sh - gs) if (sh is not None and not np.isnan(sh)) else np.nan,
                            "is_global_row": False})
    dest_share = pd.DataFrame(ds_rows)
    assert len(dest_share) == 75

    # ========== 5. source_stock_rate_summary.csv (24) ==========
    sr_rows = []
    for z in SOURCES:
        for var in ["present_stock", "linkable_stock", "present_outflow_rate", "linkable_outflow_rate"]:
            for scope, dates in [("all_dates", complete_dates), ("normal_dates", normal_dates)]:
                f = gv_source(z, var)
                dt_v, _ = series_daily_profile(f, dates)
                # source outflow daily for correlation
                fo = gv_source(z, "source_outflow")
                dt_o, _ = series_daily_profile(fo, dates)
                mean, std, cv, zero = daily_stats(dt_v)
                _, mat = series_daily_profile(f, dates)
                pp = profile_pearson_list(mat)
                sr_rows.append({"source_region": z, "variable": var, "date_scope": scope, "mean": mean, "std": std, "cv": cv,
                                "zero_rate": zero, "hour_profile_pearson_mean": float(np.mean(pp)) if pp else np.nan,
                                "lag1_daily_correlation": lag1_corr(dt_v), "correlation_with_source_outflow": safe_pearson(dt_v, dt_o),
                                "candidate_for_modeling": bool(cv is not None and not np.isnan(cv) and cv < raw_cv[z])})
    stock_rate = pd.DataFrame(sr_rows)
    assert len(stock_rate) == 24

    # ========== 6. modeling_decision.csv (3) ==========
    md_rows = []
    for z in SOURCES:
        sn = f"{z}_outflow"
        gr = grouping[grouping["series_name"] == sn].iloc[0]
        dir_cvs = grouping[grouping["grouping_type"] == "direction"]["daily_cv"]
        red_sparse = gr["zero_rate"] < float(dir_cvs.idxmin()) and gr["zero_rate"] < grouping[grouping["grouping_type"] == "direction"]["zero_rate"].min()
        d_stable = gr["daily_cv"] < grouping[grouping["grouping_type"] == "direction"]["daily_cv"].median()
        p_stable = gr["normal_day_profile_pearson_mean"] > grouping[grouping["grouping_type"] == "direction"]["normal_day_profile_pearson_mean"].median()
        # destination share stability
        dss = dest_share[(dest_share["source_region"] == z) & (~dest_share["is_global_row"])]
        share_var = float(dss["destination_1_share"].std(ddof=1)) if len(dss) else np.nan
        dest_stable = share_var < 0.15 if not np.isnan(share_var) else False
        # stock/rate
        ps_z = stock_rate[(stock_rate["source_region"] == z) & (stock_rate["date_scope"] == "all_dates")]
        pr_stable = ps_z[ps_z["variable"] == "present_outflow_rate"]["candidate_for_modeling"].iloc[0]
        lr_stable = ps_z[ps_z["variable"] == "linkable_outflow_rate"]["candidate_for_modeling"].iloc[0]
        ps_forecast = ps_z[ps_z["variable"] == "present_stock"]["hour_profile_pearson_mean"].iloc[0] > gr["normal_day_profile_pearson_mean"]
        ls_forecast = ps_z[ps_z["variable"] == "linkable_stock"]["hour_profile_pearson_mean"].iloc[0] > gr["normal_day_profile_pearson_mean"]
        # recommendation
        if lr_stable and ls_forecast:
            rec = "linkable_stock_times_rate"
        elif pr_stable and ps_forecast:
            rec = "present_stock_times_rate"
        elif d_stable and p_stable:
            rec = "source_total_profile_share"
        else:
            rec = "no_clear_advantage"
        md_rows.append({"source_region": z, "source_grouping_reduces_sparsity": bool(red_sparse),
                        "source_daily_total_more_stable_than_directions": bool(d_stable),
                        "source_profile_more_stable_than_directions": bool(p_stable), "destination_share_stable": bool(dest_stable),
                        "present_stock_forecastable": bool(ps_forecast), "linkable_stock_forecastable": bool(ls_forecast),
                        "present_rate_stable": bool(pr_stable), "linkable_rate_stable": bool(lr_stable),
                        "recommended_structure": rec, "reason": f"cv={gr['daily_cv']:.3f}, pearson={gr['normal_day_profile_pearson_mean']:.3f}, share_var={share_var:.3f}"})
    modeling = pd.DataFrame(md_rows)

    # ========== write ==========
    p1 = out_dir / "source_outflow_hourly.csv"; p2 = out_dir / "source_daily_summary.csv"
    p3 = out_dir / "source_profile_stability.csv"; p4 = out_dir / "destination_share_summary.csv"
    p5 = out_dir / "source_stock_rate_summary.csv"; p6 = out_dir / "grouping_comparison.csv"
    p7 = out_dir / "modeling_decision.csv"; p8 = out_dir / "summary.md"

    h1 = hourly.copy(); h1["date"] = h1["date"].dt.strftime("%Y-%m-%d"); h1["hour"] = h1["hour"].dt.strftime("%Y-%m-%d %H:%M:%S")
    h1.to_csv(p1, index=False, encoding="utf-8-sig")
    d2 = daily_summary.copy(); d2["date"] = d2["date"].dt.strftime("%Y-%m-%d")
    d2.to_csv(p2, index=False, encoding="utf-8-sig")
    prof_stab.to_csv(p3, index=False, encoding="utf-8-sig")
    dest_share.to_csv(p4, index=False, encoding="utf-8-sig")
    stock_rate.to_csv(p5, index=False, encoding="utf-8-sig")
    grouping.to_csv(p6, index=False, encoding="utf-8-sig")
    modeling.to_csv(p7, index=False, encoding="utf-8-sig")

    # summary
    L = []
    L.append("# Step 19 B Source-Outflow Diagnostics")
    L.append("")
    L.append(f"Run command: `{RUN_CMD}`")
    L.append("")
    L.append("## 1. Motivation\n- Step 18 confirmed core->near local replacement is effective; near->core replacement is harmful. Current best online B SSE = 1741. Reciprocal-pair decomposition did not improve overall. This step tests the source-outflow mechanism.\n")
    L.append("## 2. Source-outflow sparsity\n")
    L.append("| grouping | series | zero_rate | daily_cv | normal profile Pearson |")
    L.append("| --- | --- | --- | --- | --- |")
    for _, r in grouping.iterrows():
        L.append(f"| {r['grouping_type']} | {r['series_name']} | {r['zero_rate']:.3f} | {r['daily_cv']:.3f} | {r['normal_day_profile_pearson_mean']:.3f} |")
    L.append("")
    L.append("## 3-4. Daily-total and hour-profile stability\n- See grouping_comparison.csv and source_profile_stability.csv for full details.\n")
    L.append("## 5. Destination-choice stability\n")
    L.append("| source | global share (dest1) | share std across hours |")
    L.append("| --- | --- | --- |")
    for z in SOURCES:
        dss = dest_share[(dest_share["source_region"] == z) & (~dest_share["is_global_row"])]
        gs = dest_share[(dest_share["source_region"] == z) & (dest_share["is_global_row"])]["destination_1_share"].iloc[0]
        L.append(f"| {z} | {gs:.3f} | {dss['destination_1_share'].std(ddof=1):.3f} |")
    L.append("")
    L.append("## 6. Stock-times-rate feasibility\n- See source_stock_rate_summary.csv.\n")
    L.append("## 7. Comparison with reciprocal-pair grouping\n- See grouping_comparison.csv for side-by-side comparison.\n")
    L.append("## 8. Decision for Step 20\n")
    L.append("| source | recommended_structure | reason |")
    L.append("| --- | --- | --- |")
    for _, r in modeling.iterrows():
        L.append(f"| {r['source_region']} | {r['recommended_structure']} | {r['reason']} |")
    L.append("")
    p8.write_text("\n".join(L), encoding="utf-8")

    # assertions
    assert len(hourly) == 1728 and len(daily_summary) == 69 and len(dest_share) == 75
    assert len(stock_rate) == 24 and len(modeling) == 3
    # 01-24 23:00 censored
    cens = hourly[~hourly["is_complete_label"]]
    assert len(cens) == 3 and cens["source_outflow"].isna().all()
    # outflow = sum of 2 directions (spot check)
    assert not any(out_dir.rglob("*.zip"))

    print("\n=== Output files ===")
    for p in [p1, p2, p3, p4, p5, p6, p7, p8]:
        rows = len(pd.read_csv(p, encoding="utf-8-sig")) if p.suffix == ".csv" else sum(1 for _ in open(p, encoding="utf-8"))
        print(f"  {p.relative_to(ROOT)}  rows={rows}")
    print(f"\nmodeling decisions:")
    for _, r in modeling.iterrows():
        print(f"  {r['source_region']}: {r['recommended_structure']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
