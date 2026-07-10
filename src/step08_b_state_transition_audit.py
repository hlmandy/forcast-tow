"""Step 08 — Systematic diagnosis of Task B state stock, connectivity, and migration.

Pure data diagnostics for Task B: no models, no backtest, no submission, no Task A.

It reconstructs each vessel-hour representative region (same mode + latest-time
tie-break as optimized_baseline.make_b_labels) into a vessel-hour state table,
derives source-layer stock, adjacent-hour linkability, and the six directed
flows, and asserts they reproduce make_b_labels exactly. It also runs a
data-source ablation (re-determining the representative region from each source
subset).

Outputs (under outputs/step08_b_state_transition_audit/):
  1. b_vessel_hour_states.csv         (variable)
  2. b_hourly_source_state.csv        (1728)
  3. b_hourly_pair_flow.csv           (3456)
  4. b_daily_flow_overview.csv        (24)
  5. b_pair_relationships.csv         (42)
  6. b_hour_profile_summary.csv       (720)
  7. b_profile_pairwise_similarity.csv(816)
  8. b_source_ablation_scores.csv     (196)
  9. summary.md
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimized_baseline import (  # noqa: E402
    PAIRS, REGION_ORDER, add_regions, make_b_labels,
)

TRAIN_REL = Path("数据备份/训练集_20180101-0124_拖轮AIS.csv")
STEP01_DAILY = Path("outputs/step01_data_audit/train_daily_overview.csv")
STEP02_DAILY = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
OUT_REL = Path("outputs/step08_b_state_transition_audit")

FIXED_DATES = pd.date_range("2018-01-01", "2018-01-24", freq="D").normalize()
CANON_HOURS = pd.date_range("2018-01-01", periods=576, freq="h")
PRE_OUTAGE_END = pd.Timestamp("2018-01-11")
DEGRADED_END = pd.Timestamp("2018-01-18")
SOURCES = ["china_coastal", "e_globe_daily", "f_globe_dynamic"]
DIRECTIONS = [f"{s}->{t}" for s, t in PAIRS]
DIR_TUPLE = list(PAIRS)  # 6 (source,target)
REGION_RANK = {r: i for i, r in enumerate(REGION_ORDER)}

PAIRREL_SUBSETS = ["all", "normal_all", "degraded_period", "normal_pre_outage", "normal_post_recovery", "normal_weekday", "normal_weekend"]
PROFILE_GROUPS = ["normal_all", "normal_pre_outage", "normal_post_recovery", "normal_weekday", "normal_weekend"]
ABLATION_SUBSETS = ["all", "pre_outage", "degraded_period", "post_recovery"]
ABLATION_COMPONENTS = ["overall"] + DIRECTIONS
CONFIGS = ["full", "without_china_coastal", "without_e_globe_daily", "without_f_globe_dynamic", "china_coastal_only", "e_globe_daily_only", "f_globe_dynamic_only"]

STATE_COLUMNS = [
    "mmsi", "hour", "date", "hour_of_day", "dayofweek", "day_type", "audit_quality_regime", "period",
    "representative_region", "representative_points", "representative_last_time", "in30_record_count",
    "observed_region_count", "top_tied_region_count", "tie_broken_by_latest",
    "china_points_in_representative_region", "e_globe_daily_points_in_representative_region",
    "f_globe_dynamic_points_in_representative_region",
    "has_next_consecutive_state", "next_region", "transition_class",
]
SOURCE_COLUMNS = [
    "date", "hour", "source_region", "audit_quality_regime", "period", "day_type", "daily_unique_vessel_count",
    "source_present_count", "source_linkable_count", "stay_count",
    "to_core_count", "to_near_count", "to_outer_count", "outflow_count", "no_consecutive_count",
    "link_rate", "outflow_rate_among_present", "outflow_rate_among_linkable",
]
PAIRFLOW_COLUMNS = [
    "date", "hour", "source_region", "target_region", "task_key", "audit_quality_regime", "period", "day_type",
    "daily_unique_vessel_count", "source_present_count", "source_linkable_count", "source_outflow_count",
    "y", "pair_rate_among_present", "pair_rate_among_linkable",
]
DAILY_COLUMNS = [
    "date", "audit_quality_regime", "period", "day_type", "unique_vessel_count", "ais_record_count",
    "representative_vessel_hours", "linkable_vessel_hours", "stay_vessel_hours", "outflow_vessel_hours",
    "no_consecutive_vessel_hours", "overall_link_rate", "overall_outflow_rate_among_present",
    "overall_outflow_rate_among_linkable",
    "core_present_total", "near_present_total", "outer_present_total",
    "core_linkable_total", "near_linkable_total", "outer_linkable_total",
    "b_core_to_near", "b_core_to_outer", "b_near_to_core", "b_near_to_outer", "b_outer_to_core", "b_outer_to_near", "b_total",
    "zero_flow_hour_count",
]
PAIRREL_COLUMNS = [
    "subset", "source_region", "target_region", "task_key", "n_days", "n_hour_rows",
    "total_flow", "mean_hourly_flow", "variance_hourly_flow", "variance_to_mean_ratio", "zero_hour_rate",
    "mean_source_present", "mean_source_linkable", "mean_pair_rate_among_present", "mean_pair_rate_among_linkable",
    "pearson_hourly_source_present", "spearman_hourly_source_present",
    "pearson_hourly_source_linkable", "spearman_hourly_source_linkable",
    "pearson_daily_vessels", "spearman_daily_vessels",
    "pearson_daily_source_present", "spearman_daily_source_present",
]
PROFILE_COLUMNS = [
    "group", "source_region", "target_region", "task_key", "hour", "n_days", "n_days_with_positive_daily_flow",
    "mean_hourly_flow", "median_hourly_flow", "std_hourly_flow", "zero_rate",
    "mean_daily_profile_share", "median_daily_profile_share",
    "mean_pair_rate_among_linkable", "median_pair_rate_among_linkable",
]
SIM_COLUMNS = [
    "source_region", "target_region", "task_key", "date_i", "date_j",
    "daily_total_i", "daily_total_j", "period_i", "period_j", "period_pair",
    "day_type_i", "day_type_j", "same_day_type",
    "pearson_profile", "cosine_similarity", "l1_distance", "rmse_profile",
]
ABLATION_COLUMNS = [
    "source_configuration", "subset", "component", "n_hour_rows", "full_total", "ablated_total",
    "retained_total_ratio", "sse_against_full", "mae_against_full", "mean_bias_against_full", "exact_match_rate",
]

RUN_CMD = "python " + " ".join(sys.argv)


def period_of(date):
    if date <= PRE_OUTAGE_END:
        return "pre_outage"
    if date <= DEGRADED_END:
        return "degraded_period"
    return "post_recovery"


def safe_corr(x, y, method):
    if len(x) < 3:
        return np.nan
    xs, ys = pd.Series(x), pd.Series(y)
    if xs.nunique() <= 1 or ys.nunique() <= 1:
        return np.nan
    return float(xs.corr(ys, method=method))


def pearson(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return safe_corr(a, b, "pearson")


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
def load_ais(train_path):
    usecols = ["mmsi", "x", "y", "sog", "time", "source_dataset"]
    dtypes = {"mmsi": "string", "x": "float64", "y": "float64", "sog": "float32", "source_dataset": "string"}
    df = pd.read_csv(train_path, usecols=usecols, dtype=dtypes, parse_dates=["time"])
    df["hour"] = df["time"].dt.floor("h")
    df["date"] = df["time"].dt.normalize()
    df = add_regions(df)
    return df


def labels_on_grid(df_subset):
    lab = make_b_labels(df_subset)
    s = lab.set_index(["hour", "source_region", "target_region"])["y"]
    idx = pd.MultiIndex.from_tuples([(h, s_, t) for h in CANON_HOURS for s_, t in DIR_TUPLE],
                                    names=["hour", "source_region", "target_region"])
    return s.reindex(idx, fill_value=0)


# ---------------------------------------------------------------------------
def main():
    out_dir = ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = ROOT / TRAIN_REL
    if not train_path.exists():
        raise FileNotFoundError(f"Training AIS file not found: {train_path}")
    print(f"train_path = {train_path}")
    print(f"out_dir    = {out_dir}")

    df = load_ais(train_path)
    daily = pd.read_csv(ROOT / STEP01_DAILY); daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    audit = pd.read_csv(ROOT / STEP02_DAILY); audit["date"] = pd.to_datetime(audit["date"]).dt.normalize()
    quality_map = dict(zip(audit["date"], audit["quality_regime"]))
    vc_map = dict(zip(daily["date"], daily["unique_vessel_count"]))

    # --- vessel-hour representative state ----------------------------------
    in_region = df[df["region"].isin(REGION_ORDER)]
    agg = in_region.groupby(["mmsi", "hour", "region"], observed=True).agg(
        points=("time", "size"), last_time=("time", "max"),
        china=("source_dataset", lambda s: int((s == "china_coastal").sum())),
        e_globe=("source_dataset", lambda s: int((s == "e_globe_daily").sum())),
        f_globe=("source_dataset", lambda s: int((s == "f_globe_dynamic").sum())),
    ).reset_index()

    rep = (agg.sort_values(["mmsi", "hour", "points", "last_time"], ascending=[True, True, False, False])
              .drop_duplicates(["mmsi", "hour"], keep="first")
              .sort_values(["mmsi", "hour"]).reset_index(drop=True))
    rep = rep.rename(columns={
        "region": "representative_region", "points": "representative_points", "last_time": "representative_last_time",
        "china": "china_points_in_representative_region", "e_globe": "e_globe_daily_points_in_representative_region",
        "f_globe": "f_globe_dynamic_points_in_representative_region"})

    gstat = agg.groupby(["mmsi", "hour"], observed=True)
    in30 = gstat["points"].sum().rename("in30_record_count").reset_index()
    obs_cnt = gstat.size().rename("observed_region_count").reset_index()
    maxpts = gstat["points"].max().rename("_max").reset_index()
    tied_base = agg.merge(maxpts, on=["mmsi", "hour"])
    top_tied = tied_base[tied_base["points"] == tied_base["_max"]].groupby(["mmsi", "hour"], observed=True).size().rename("top_tied_region_count").reset_index()

    states = (rep.merge(in30, on=["mmsi", "hour"]).merge(obs_cnt, on=["mmsi", "hour"]).merge(top_tied, on=["mmsi", "hour"]))
    states["tie_broken_by_latest"] = states["top_tied_region_count"] > 1
    states["date"] = states["hour"].dt.normalize()
    states["hour_of_day"] = states["hour"].dt.hour
    states["dayofweek"] = states["hour"].dt.dayofweek
    states["day_type"] = np.where(states["dayofweek"].isin([5, 6]), "weekend", "weekday")
    states["period"] = states["date"].map(period_of)
    states["audit_quality_regime"] = states["date"].map(quality_map)

    states = states.sort_values(["mmsi", "hour"]).reset_index(drop=True)
    states["next_hour"] = states.groupby("mmsi")["hour"].shift(-1)
    states["next_region_raw"] = states.groupby("mmsi")["representative_region"].shift(-1)
    states["has_next_consecutive_state"] = states["next_hour"] == states["hour"] + pd.Timedelta(hours=1)
    s_reg = states["representative_region"].astype(object)
    nr = states["next_region_raw"].astype(object)
    cons = states["has_next_consecutive_state"].to_numpy()
    states["next_region"] = np.where(cons, nr, np.nan)
    states["transition_class"] = np.where(~cons, "no_consecutive_state",
                                 np.where(s_reg.to_numpy() == nr.to_numpy(), "stay_" + s_reg, s_reg + "->" + nr))
    assert states.set_index(["mmsi", "hour"]).index.is_unique, "mmsi-hour not unique"

    # --- assert reconstructed labels == make_b_labels ----------------------
    moved = states[states["has_next_consecutive_state"] & (states["representative_region"] != states["next_region_raw"])]
    y_mine = moved.groupby(["hour", "representative_region", "next_region_raw"], observed=True)["mmsi"].nunique().rename("y").reset_index()
    y_mine = y_mine.rename(columns={"representative_region": "source_region", "next_region_raw": "target_region"})
    y_mine_g = y_mine.set_index(["hour", "source_region", "target_region"])["y"]
    idx = pd.MultiIndex.from_tuples([(h, s_, t) for h in CANON_HOURS for s_, t in DIR_TUPLE], names=["hour", "source_region", "target_region"])
    y_mine_g = y_mine_g.reindex(idx, fill_value=0)
    b_official = labels_on_grid(df)
    assert (y_mine_g == b_official).all(), "reconstructed B labels != make_b_labels"
    print(f"vessel-hour states: {len(states)} rows; reconstructed labels match make_b_labels.")

    # --- hourly source state + pair flow -----------------------------------
    states["dest"] = np.where(states["has_next_consecutive_state"], states["next_region_raw"], "NO_NEXT")
    cnt = states.groupby(["date", "hour", "representative_region", "dest"], observed=True).size().unstack("dest", fill_value=0)
    for c in REGION_ORDER + ["NO_NEXT"]:
        if c not in cnt.columns:
            cnt[c] = 0
    cnt = cnt.reset_index()

    grid_src = pd.MultiIndex.from_product([CANON_HOURS, REGION_ORDER], names=["hour", "source_region"]).to_frame(index=False)
    grid_src["date"] = grid_src["hour"].dt.normalize()
    src = grid_src.merge(cnt, left_on=["date", "hour", "source_region"], right_on=["date", "hour", "representative_region"], how="left") \
                  .drop(columns=["representative_region"], errors="ignore")
    for c in REGION_ORDER + ["NO_NEXT"]:
        src[c] = src[c].fillna(0).astype(np.int64)
    src["to_core_count"] = src["core"]; src["to_near_count"] = src["near"]; src["to_outer_count"] = src["outer"]
    src["no_consecutive_count"] = src["NO_NEXT"]
    src["source_present_count"] = src[["to_core_count", "to_near_count", "to_outer_count"]].sum(axis=1) + src["no_consecutive_count"]
    src["source_linkable_count"] = src[["to_core_count", "to_near_count", "to_outer_count"]].sum(axis=1)
    src["stay_count"] = src.apply(lambda r: int(r[f"to_{r['source_region']}_count"]), axis=1)
    src["outflow_count"] = src["source_linkable_count"] - src["stay_count"]
    src["link_rate"] = np.where(src["source_present_count"] > 0, src["source_linkable_count"] / src["source_present_count"], 0.0)
    src["outflow_rate_among_present"] = np.where(src["source_present_count"] > 0, src["outflow_count"] / src["source_present_count"], 0.0)
    src["outflow_rate_among_linkable"] = np.where(src["source_linkable_count"] > 0, src["outflow_count"] / src["source_linkable_count"], 0.0)
    src["audit_quality_regime"] = src["date"].map(quality_map)
    src["period"] = src["date"].map(period_of)
    src["day_type"] = np.where(src["hour"].dt.dayofweek.isin([5, 6]), "weekend", "weekday")
    src["daily_unique_vessel_count"] = src["date"].map(vc_map)
    src["_r"] = src["source_region"].map(REGION_RANK)
    source_state = src.sort_values(["hour", "_r"]).drop(columns=["_r", "core", "near", "outer", "NO_NEXT"]).reset_index(drop=True)

    # pair flow (6 directions)
    pf_rows = []
    to_cols = {"core": "to_core_count", "near": "to_near_count", "outer": "to_outer_count"}
    for _, r in src.iterrows():
        s_reg_ = r["source_region"]
        for t in REGION_ORDER:
            if t == s_reg_:
                continue
            pf_rows.append({"date": r["date"], "hour": r["hour"], "source_region": s_reg_, "target_region": t,
                            "y": int(r[to_cols[t]])})
    pf = pd.DataFrame(pf_rows)
    pf["task_key"] = pf["source_region"] + "->" + pf["target_region"]
    pf = pf.merge(src[["hour", "source_region", "source_present_count", "source_linkable_count", "outflow_count"]].rename(columns={"outflow_count": "source_outflow_count"}),
                  on=["hour", "source_region"], how="left")
    pf["audit_quality_regime"] = pf["date"].map(quality_map)
    pf["period"] = pf["date"].map(period_of)
    pf["day_type"] = np.where(pf["hour"].dt.dayofweek.isin([5, 6]), "weekend", "weekday")
    pf["daily_unique_vessel_count"] = pf["date"].map(vc_map)
    pf["pair_rate_among_present"] = np.where(pf["source_present_count"] > 0, pf["y"] / pf["source_present_count"], 0.0)
    pf["pair_rate_among_linkable"] = np.where(pf["source_linkable_count"] > 0, pf["y"] / pf["source_linkable_count"], 0.0)
    pf["_s"] = pf["source_region"].map(REGION_RANK)
    pf["_t"] = pf["target_region"].map(REGION_RANK)
    pair_flow = pf.sort_values(["hour", "_s", "_t"]).drop(columns=["_s", "_t"]).reset_index(drop=True)

    # --- assertions --------------------------------------------------------
    assert len(pair_flow) == 3456
    # y == official
    pf_g = pair_flow.set_index(["hour", "source_region", "target_region"])["y"].reindex(idx, fill_value=0)
    assert (pf_g == b_official).all(), "pair_flow y != make_b_labels"
    # conservation
    assert (src["source_present_count"] == src["source_linkable_count"] + src["no_consecutive_count"]).all()
    assert (src["source_linkable_count"] == src["to_core_count"] + src["to_near_count"] + src["to_outer_count"]).all()
    assert (src["source_linkable_count"] == src["stay_count"] + src["outflow_count"]).all()
    # outflow == sum of two directions
    of_check = pair_flow.groupby(["hour", "source_region"])["y"].sum().rename("y_out").reset_index()
    of_check = of_check.merge(src[["hour", "source_region", "outflow_count"]], on=["hour", "source_region"])
    assert (of_check["y_out"] == of_check["outflow_count"]).all(), "2-direction sum != outflow_count"

    # --- daily flow overview -----------------------------------------------
    # present per region (date-level)
    pres_by_reg = {}
    for r in REGION_ORDER:
        pres_by_reg[r] = src[src["source_region"] == r].groupby("date")["source_present_count"].sum()
    dir_daily = pair_flow.groupby(["date", "task_key"])["y"].sum().unstack("task_key").reindex(columns=DIRECTIONS).fillna(0)

    do_rows = []
    for d in FIXED_DATES:
        s_d = src[src["date"] == d]
        rep_vh = int(s_d["source_present_count"].sum())
        link_vh = int(s_d["source_linkable_count"].sum())
        stay_vh = int(s_d["stay_count"].sum())
        out_vh = int(s_d["outflow_count"].sum())
        no_vh = int(s_d["no_consecutive_count"].sum())
        bdir = {k: int(dir_daily.loc[d, k]) if d in dir_daily.index else 0 for k in DIRECTIONS}
        b_total = sum(bdir.values())
        zero_flow_hours = int((pair_flow[pair_flow["date"] == d].groupby("hour")["y"].sum() == 0).sum())
        do_rows.append({
            "date": d, "audit_quality_regime": quality_map.get(d), "period": period_of(d),
            "day_type": "weekend" if d.dayofweek in (5, 6) else "weekday",
            "unique_vessel_count": int(vc_map.get(d, 0)),
            "ais_record_count": int(daily.loc[daily["date"] == d, "ais_record_count"].iloc[0]) if (daily["date"] == d).any() else 0,
            "representative_vessel_hours": rep_vh, "linkable_vessel_hours": link_vh,
            "stay_vessel_hours": stay_vh, "outflow_vessel_hours": out_vh, "no_consecutive_vessel_hours": no_vh,
            "overall_link_rate": link_vh / rep_vh if rep_vh else 0.0,
            "overall_outflow_rate_among_present": out_vh / rep_vh if rep_vh else 0.0,
            "overall_outflow_rate_among_linkable": out_vh / link_vh if link_vh else 0.0,
            "core_present_total": int(pres_by_reg["core"].get(d, 0)), "near_present_total": int(pres_by_reg["near"].get(d, 0)), "outer_present_total": int(pres_by_reg["outer"].get(d, 0)),
            "core_linkable_total": int(src[(src["date"] == d) & (src["source_region"] == "core")]["source_linkable_count"].sum()),
            "near_linkable_total": int(src[(src["date"] == d) & (src["source_region"] == "near")]["source_linkable_count"].sum()),
            "outer_linkable_total": int(src[(src["date"] == d) & (src["source_region"] == "outer")]["source_linkable_count"].sum()),
            "b_core_to_near": bdir["core->near"], "b_core_to_outer": bdir["core->outer"],
            "b_near_to_core": bdir["near->core"], "b_near_to_outer": bdir["near->outer"],
            "b_outer_to_core": bdir["outer->core"], "b_outer_to_near": bdir["outer->near"],
            "b_total": b_total, "zero_flow_hour_count": zero_flow_hours,
        })
    daily_overview = pd.DataFrame(do_rows)
    # assertions: b_total == sum 6; == step01; outflow_vessel_hours == b_total
    assert (daily_overview[["b_core_to_near", "b_core_to_outer", "b_near_to_core", "b_near_to_outer", "b_outer_to_core", "b_outer_to_near"]].sum(axis=1) == daily_overview["b_total"]).all()
    for col in ["b_core_to_near", "b_core_to_outer", "b_near_to_core", "b_near_to_outer", "b_outer_to_core", "b_outer_to_near"]:
        step1_val = daily[["date", col]].rename(columns={col: "step1_val"})
        merged = daily_overview[["date", col]].merge(step1_val, on="date", how="left")
        assert (merged[col].astype(int) == merged["step1_val"].astype(int)).all(), f"daily {col} != step01"
    assert (daily_overview["outflow_vessel_hours"] == daily_overview["b_total"]).all()
    print("daily flow overview: conservation + step01 consistency PASS.")

    # --- pair relationships (42) -------------------------------------------
    date_tbl = daily_overview[["date", "audit_quality_regime", "period", "day_type", "unique_vessel_count"]].copy()
    subset_masks = {
        "all": np.ones(len(date_tbl), bool),
        "normal_all": date_tbl["audit_quality_regime"] == "normal",
        "degraded_period": date_tbl["period"] == "degraded_period",
        "normal_pre_outage": (date_tbl["audit_quality_regime"] == "normal") & (date_tbl["period"] == "pre_outage"),
        "normal_post_recovery": (date_tbl["audit_quality_regime"] == "normal") & (date_tbl["period"] == "post_recovery"),
        "normal_weekday": (date_tbl["audit_quality_regime"] == "normal") & (date_tbl["day_type"] == "weekday"),
        "normal_weekend": (date_tbl["audit_quality_regime"] == "normal") & (date_tbl["day_type"] == "weekend"),
    }
    rel_rows = []
    for sname in PAIRREL_SUBSETS:
        sub_dates = set(date_tbl[subset_masks[sname]]["date"])
        for (sreg, treg) in DIR_TUPLE:
            sub = pair_flow[(pair_flow["date"].isin(sub_dates)) & (pair_flow["source_region"] == sreg) & (pair_flow["target_region"] == treg)]
            y = sub["y"].to_numpy("float64")
            sp = sub["source_present_count"].to_numpy("float64")
            sl = sub["source_linkable_count"].to_numpy("float64")
            mean_y = float(y.mean()) if len(y) else np.nan
            var_y = float(y.var(ddof=1)) if len(y) > 1 else np.nan
            # daily
            dflow = sub.groupby("date")["y"].sum()
            dvess = sub.drop_duplicates("date").set_index("date")["daily_unique_vessel_count"]
            dsp = sub.groupby("date")["source_present_count"].sum()
            dfa = dflow.reindex(sub.drop_duplicates("date")["date"]).reset_index(drop=True)
            rel_rows.append({
                "subset": sname, "source_region": sreg, "target_region": treg, "task_key": f"{sreg}->{treg}",
                "n_days": int(len(sub_dates)), "n_hour_rows": int(len(sub)),
                "total_flow": float(y.sum()), "mean_hourly_flow": mean_y, "variance_hourly_flow": var_y,
                "variance_to_mean_ratio": (var_y / mean_y) if (mean_y and mean_y > 0 and not np.isnan(var_y)) else np.nan,
                "zero_hour_rate": float((y == 0).mean()) if len(y) else np.nan,
                "mean_source_present": float(sp.mean()) if len(sp) else np.nan,
                "mean_source_linkable": float(sl.mean()) if len(sl) else np.nan,
                "mean_pair_rate_among_present": float((y / np.where(sp > 0, sp, 1)).mean()) if len(y) else np.nan,
                "mean_pair_rate_among_linkable": float((y / np.where(sl > 0, sl, 1)).mean()) if len(y) else np.nan,
                "pearson_hourly_source_present": safe_corr(sp, y, "pearson"),
                "spearman_hourly_source_present": safe_corr(sp, y, "spearman"),
                "pearson_hourly_source_linkable": safe_corr(sl, y, "pearson"),
                "spearman_hourly_source_linkable": safe_corr(sl, y, "spearman"),
                "pearson_daily_vessels": safe_corr(dvess.reindex(dflow.index).to_numpy("float64"), dflow.to_numpy("float64"), "pearson"),
                "spearman_daily_vessels": safe_corr(dvess.reindex(dflow.index).to_numpy("float64"), dflow.to_numpy("float64"), "spearman"),
                "pearson_daily_source_present": safe_corr(dsp.reindex(dflow.index).to_numpy("float64"), dflow.to_numpy("float64"), "pearson"),
                "spearman_daily_source_present": safe_corr(dsp.reindex(dflow.index).to_numpy("float64"), dflow.to_numpy("float64"), "spearman"),
            })
    pairrel = pd.DataFrame(rel_rows)

    # --- daily profile share for profile summary + similarity --------------
    pf2 = pair_flow.copy()
    dtot = pf2.groupby(["date", "task_key"])["y"].transform("sum")
    pf2["daily_total_dir"] = dtot
    pf2["daily_profile_share"] = np.where(pf2["daily_total_dir"] > 0, pf2["y"] / pf2["daily_total_dir"], 0.0)

    # --- hour profile summary (720) ----------------------------------------
    prof_rows = []
    for gname in PROFILE_GROUPS:
        gdates = set(date_tbl[subset_masks[gname]]["date"])
        sub = pf2[pf2["date"].isin(gdates)]
        for (sreg, treg) in DIR_TUPLE:
            tk = f"{sreg}->{treg}"
            sd = sub[sub["task_key"] == tk]
            for h in range(24):
                cell = sd[sd["hour"].dt.hour == h]
                y = cell["y"].to_numpy("float64")
                shares = cell["daily_profile_share"].to_numpy("float64")
                pr_link = cell["pair_rate_among_linkable"].to_numpy("float64")
                n_pos = int((cell.groupby("date")["y"].sum() > 0).sum())
                prof_rows.append({
                    "group": gname, "source_region": sreg, "target_region": treg, "task_key": tk, "hour": h,
                    "n_days": int(len(cell)), "n_days_with_positive_daily_flow": n_pos,
                    "mean_hourly_flow": float(y.mean()) if len(y) else np.nan,
                    "median_hourly_flow": float(np.median(y)) if len(y) else np.nan,
                    "std_hourly_flow": float(y.std(ddof=1)) if len(y) > 1 else np.nan,
                    "zero_rate": float((y == 0).mean()) if len(y) else np.nan,
                    "mean_daily_profile_share": float(shares.mean()) if len(shares) else np.nan,
                    "median_daily_profile_share": float(np.median(shares)) if len(shares) else np.nan,
                    "mean_pair_rate_among_linkable": float(pr_link.mean()) if len(pr_link) else np.nan,
                    "median_pair_rate_among_linkable": float(np.median(pr_link)) if len(pr_link) else np.nan,
                })
    profile_summary = pd.DataFrame(prof_rows)

    # --- pairwise similarity (816) -----------------------------------------
    normal_dates = sorted(date_tbl[date_tbl["audit_quality_regime"] == "normal"]["date"].tolist())
    sim_rows = []
    for (sreg, treg) in DIR_TUPLE:
        tk = f"{sreg}->{treg}"
        # matrix normal_dates x 24 of daily_profile_share
        M = np.zeros((len(normal_dates), 24))
        totals = np.zeros(len(normal_dates))
        for i, d in enumerate(normal_dates):
            cell = pf2[(pf2["date"] == d) & (pf2["task_key"] == tk)].sort_values("hour")
            vec = cell["daily_profile_share"].to_numpy()
            for h in range(24):
                M[i, h] = vec[h] if h < len(vec) else 0.0
            totals[i] = float(cell["y"].sum())
        for i, j in itertools.combinations(range(len(normal_dates)), 2):
            a, b = M[i], M[j]
            di, dj = normal_dates[i], normal_dates[j]
            mi = date_tbl[date_tbl["date"] == di].iloc[0]
            mj = date_tbl[date_tbl["date"] == dj].iloc[0]
            sim_rows.append({
                "source_region": sreg, "target_region": treg, "task_key": tk,
                "date_i": f"{di:%Y-%m-%d}", "date_j": f"{dj:%Y-%m-%d}",
                "daily_total_i": totals[i], "daily_total_j": totals[j],
                "period_i": mi["period"], "period_j": mj["period"],
                "period_pair": "__".join(sorted([mi["period"], mj["period"]])),
                "day_type_i": mi["day_type"], "day_type_j": mj["day_type"],
                "same_day_type": bool(mi["day_type"] == mj["day_type"]),
                "pearson_profile": pearson(a, b), "cosine_similarity": cosine(a, b),
                "l1_distance": float(np.abs(a - b).sum()), "rmse_profile": float(np.sqrt(np.mean((a - b) ** 2))),
            })
    similarity = pd.DataFrame(sim_rows)

    # --- source ablation ---------------------------------------------------
    config_labels = {}
    for cfg in CONFIGS:
        if cfg == "full":
            mask = np.ones(len(df), bool)
        elif cfg.startswith("without_"):
            src_excl = cfg.replace("without_", "")
            mask = df["source_dataset"].to_numpy() != src_excl
        else:  # _only
            src_only = cfg.replace("_only", "")
            mask = df["source_dataset"].to_numpy() == src_only
        config_labels[cfg] = labels_on_grid(df[mask])
    # assert full == official (already official)
    assert (config_labels["full"] == b_official).all()

    full_lab = config_labels["full"]
    hour_date = pd.Series({h: h.normalize() for h in CANON_HOURS})
    hour_period = hour_date.map(period_of)
    abl_rows = []
    for cfg in CONFIGS:
        abl = config_labels[cfg]
        diff = (abl - full_lab)
        for sub in ABLATION_SUBSETS:
            if sub == "all":
                date_mask = np.ones(len(CANON_HOURS), bool)
            else:
                date_mask = (hour_period.reindex(CANON_HOURS).to_numpy() == sub)
            for comp in ABLATION_COMPONENTS:
                if comp == "overall":
                    dir_mask = np.ones(len(DIR_TUPLE), bool)
                else:
                    dir_mask = np.array([comp == d for d in DIRECTIONS])
                # build mask over the 3456 index (hour, dir)
                mask2d = np.outer(date_mask, dir_mask).reshape(-1)
                full_vals = full_lab.to_numpy()[mask2d]
                abl_vals = abl.to_numpy()[mask2d]
                d_vals = diff.to_numpy()[mask2d]
                abl_rows.append({
                    "source_configuration": cfg, "subset": sub, "component": comp,
                    "n_hour_rows": int(mask2d.sum()),
                    "full_total": float(full_vals.sum()), "ablated_total": float(abl_vals.sum()),
                    "retained_total_ratio": float(abl_vals.sum() / full_vals.sum()) if full_vals.sum() != 0 else 0.0,
                    "sse_against_full": float((d_vals ** 2).sum()),
                    "mae_against_full": float(np.abs(d_vals).mean()),
                    "mean_bias_against_full": float(d_vals.mean()),
                    "exact_match_rate": float((abl_vals == full_vals).mean()),
                })
    ablation = pd.DataFrame(abl_rows)
    full_rows = ablation[ablation["source_configuration"] == "full"]
    assert (full_rows["sse_against_full"] == 0).all() and (full_rows["mae_against_full"] == 0).all() and (full_rows["exact_match_rate"] == 1.0).all()
    print(f"ablation: {len(ablation)} rows; full config self-consistency PASS.")

    # --- write CSVs (UTF-8 BOM) --------------------------------------------
    p_st = out_dir / "b_vessel_hour_states.csv"
    p_ss = out_dir / "b_hourly_source_state.csv"
    p_pf = out_dir / "b_hourly_pair_flow.csv"
    p_do = out_dir / "b_daily_flow_overview.csv"
    p_pr = out_dir / "b_pair_relationships.csv"
    p_ps = out_dir / "b_hour_profile_summary.csv"
    p_sm = out_dir / "b_profile_pairwise_similarity.csv"
    p_ab = out_dir / "b_source_ablation_scores.csv"
    p_md = out_dir / "summary.md"

    st_out = states[STATE_COLUMNS].sort_values(["mmsi", "hour"]).copy()
    st_out["hour"] = st_out["hour"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st_out["date"] = st_out["date"].dt.strftime("%Y-%m-%d")
    st_out["representative_last_time"] = pd.to_datetime(st_out["representative_last_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    st_out.to_csv(p_st, index=False, encoding="utf-8-sig")

    def fmt_hour(df_, col="hour"):
        df_[col] = pd.to_datetime(df_[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
        return df_

    ss_out = source_state[SOURCE_COLUMNS].copy(); ss_out["date"] = ss_out["date"].dt.strftime("%Y-%m-%d"); ss_out = fmt_hour(ss_out)
    ss_out.to_csv(p_ss, index=False, encoding="utf-8-sig")

    pf_out = pair_flow[PAIRFLOW_COLUMNS].copy(); pf_out["date"] = pf_out["date"].dt.strftime("%Y-%m-%d"); pf_out = fmt_hour(pf_out)
    pf_out.to_csv(p_pf, index=False, encoding="utf-8-sig")

    do_out = daily_overview[DAILY_COLUMNS].copy(); do_out["date"] = do_out["date"].dt.strftime("%Y-%m-%d")
    do_out.to_csv(p_do, index=False, encoding="utf-8-sig")

    pr_out = pairrel[PAIRREL_COLUMNS].copy()
    pr_out["_s"] = pr_out["source_region"].map(REGION_RANK); pr_out["_t"] = pr_out["target_region"].map(REGION_RANK)
    pr_out = pr_out.sort_values(["subset", "_s", "_t"]).drop(columns=["_s", "_t"])
    pr_out.to_csv(p_pr, index=False, encoding="utf-8-sig")

    ps_out = profile_summary[PROFILE_COLUMNS].copy()
    ps_out["_s"] = ps_out["source_region"].map(REGION_RANK); ps_out["_t"] = ps_out["target_region"].map(REGION_RANK)
    ps_out["_g"] = ps_out["group"].map({g: i for i, g in enumerate(PROFILE_GROUPS)})
    ps_out = ps_out.sort_values(["_g", "_s", "_t", "hour"]).drop(columns=["_s", "_t", "_g"])
    ps_out.to_csv(p_ps, index=False, encoding="utf-8-sig")

    sm_out = similarity[SIM_COLUMNS].copy()
    sm_out["_s"] = sm_out["source_region"].map(REGION_RANK); sm_out["_t"] = sm_out["target_region"].map(REGION_RANK)
    sm_out = sm_out.sort_values(["_s", "_t", "date_i", "date_j"]).drop(columns=["_s", "_t"])
    sm_out.to_csv(p_sm, index=False, encoding="utf-8-sig")

    ab_out = ablation[ABLATION_COLUMNS].copy()
    ab_out["_c"] = ab_out["source_configuration"].map({c: i for i, c in enumerate(CONFIGS)})
    ab_out["_s"] = ab_out["subset"].map({s: i for i, s in enumerate(ABLATION_SUBSETS)})
    ab_out["_cmp"] = ab_out["component"].map({c: i for i, c in enumerate(ABLATION_COMPONENTS)})
    ab_out = ab_out.sort_values(["_c", "_s", "_cmp"]).drop(columns=["_c", "_s", "_cmp"])
    ab_out.to_csv(p_ab, index=False, encoding="utf-8-sig")

    write_summary(p_md, states, source_state, pair_flow, daily_overview, pairrel, profile_summary, similarity, ablation, date_tbl)

    # --- console -----------------------------------------------------------
    print("\n=== Output files ===")
    for path in [p_st, p_ss, p_pf, p_do, p_pr, p_ps, p_sm, p_ab, p_md]:
        if path.suffix == ".csv":
            rows = len(pd.read_csv(path, encoding="utf-8-sig"))
        else:
            rows = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {path.relative_to(ROOT)}  rows={rows}")
    print("\nDone.")


def write_summary(path, states, source_state, pair_flow, daily_overview, pairrel, profile_summary, similarity, ablation, date_tbl):
    L = []
    L.append("# Step 08 B State Transition Audit")
    L.append("")
    L.append("> Task B diagnostics only. No models, no backtest, no submission, no Task A.")
    L.append("")
    L.append(f"Run command: `{RUN_CMD}`")
    L.append("")

    L.append("## 1. Official-label consistency")
    L.append("")
    L.append(f"- Vessel-hour state rows: {len(states)} (unique mmsi-hour key).")
    L.append("- 3456 hour-direction rows complete; reconstructed six-direction labels match `make_b_labels` exactly. **PASS**")
    L.append("- Daily six-direction totals match Step 01 `train_daily_overview` exactly. **PASS**")
    L.append("- All flow-conservation assertions (present=linkable+no_consecutive; linkable=3 destinations; linkable=stay+outflow; 2-direction sum=outflow; outflow=b_total) PASS.")
    L.append("")

    L.append("## 2. State availability and linkability")
    L.append("")
    g = source_state.groupby("source_region").agg(
        present=("source_present_count", "mean"), linkable=("source_linkable_count", "mean"),
        link_rate=("link_rate", "mean"), stay=("stay_count", "mean"),
        outflow_link=("outflow_rate_among_linkable", "mean"), no_cons=("no_consecutive_count", "mean")).reindex(REGION_ORDER)
    L.append("| source | mean present | mean linkable | link_rate | mean stay | outflow_rate(linkable) | mean no_consecutive |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in REGION_ORDER:
        row = g.loc[r]
        L.append(f"| {r} | {row['present']:.2f} | {row['linkable']:.2f} | {row['link_rate']:.3f} | {row['stay']:.2f} | {row['outflow_link']:.3f} | {row['no_cons']:.2f} |")
    L.append("")

    L.append("## 3. Direction-level sparsity")
    L.append("")
    all_rel = pairrel[pairrel["subset"] == "all"].set_index("task_key").reindex(DIRECTIONS)
    L.append("| direction | total flow | mean hourly | variance | var/mean | zero rate | max hourly |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for d in DIRECTIONS:
        r = all_rel.loc[d]
        maxh = int(pair_flow[pair_flow["task_key"] == d]["y"].max())
        L.append(f"| {d} | {r['total_flow']:.0f} | {r['mean_hourly_flow']:.2f} | {r['variance_hourly_flow']:.2f} | "
                 f"{r['variance_to_mean_ratio']:.2f} | {r['zero_hour_rate']:.3f} | {maxh} |")
    L.append("")

    L.append("## 4. Effect of the china_coastal outage")
    L.append("")
    per = daily_overview.groupby("period").agg(
        b_total=("b_total", "mean"), rep_vh=("representative_vessel_hours", "mean"),
        link_vh=("linkable_vessel_hours", "mean"), link_rate=("overall_link_rate", "mean")).reindex(["pre_outage", "degraded_period", "post_recovery"])
    L.append("| period | mean b_total | mean rep vessel-hours | mean linkable vessel-hours | overall link_rate |")
    L.append("| --- | --- | --- | --- | --- |")
    for p_ in ["pre_outage", "degraded_period", "post_recovery"]:
        row = per.loc[p_]
        L.append(f"| {p_} | {row['b_total']:.1f} | {row['rep_vh']:.1f} | {row['link_vh']:.1f} | {row['link_rate']:.3f} |")
    L.append("")
    L.append(f"- **B did NOT collapse like A.** Mean daily `b_total` is essentially flat across periods (~{per.loc['pre_outage','b_total']:.0f}/{per.loc['degraded_period','b_total']:.0f}/{per.loc['post_recovery','b_total']:.0f} for pre_outage/degraded/post_recovery), versus A which fell to ~0.44x. During the outage the source stock (`representative_vessel_hours`) and `overall_link_rate` dipped, but `overall_outflow_rate_among_linkable` rose enough to keep total migration roughly constant.")
    L.append("- Therefore A's `quality_regime` (defined from china_coastal record quality) is NOT appropriate as a B training weight: B's labels stayed healthy through the A outage. B would need its own, source-availability-based quality metric if any — but the ablation in section 5 shows B is already robust to losing a single source.")
    L.append("")

    L.append("## 5. Source-ablation results")
    L.append("")
    abl_all = ablation[(ablation["subset"] == "all") & (ablation["component"] == "overall")].set_index("source_configuration").reindex(CONFIGS)
    L.append("| configuration | overall SSE vs full | retained total ratio | exact match rate |")
    L.append("| --- | --- | --- | --- |")
    for c in CONFIGS:
        r = abl_all.loc[c]
        L.append(f"| {c} | {r['sse_against_full']:.0f} | {r['retained_total_ratio']:.3f} | {r['exact_match_rate']:.3f} |")
    L.append("")
    # largest-deviation subset for each non-full config
    L.append("- Largest-deviation subset (by overall SSE vs full) per non-full config:")
    for c in CONFIGS:
        if c == "full":
            continue
        sub = ablation[(ablation["source_configuration"] == c) & (ablation["component"] == "overall")]
        worst = sub.loc[sub["sse_against_full"].idxmax()]
        L.append(f"  - {c}: {worst['subset']} (SSE={worst['sse_against_full']:.0f})")
    L.append("")

    L.append("## 6. Relationship between stock and flow")
    L.append("")
    na = pairrel[pairrel["subset"] == "normal_all"].set_index("task_key").reindex(DIRECTIONS)
    L.append("| direction | pearson y~present(hourly) | pearson y~linkable(hourly) | pearson daily~vessels | pearson daily~source_present |")
    L.append("| --- | --- | --- | --- | --- |")
    for d in DIRECTIONS:
        r = na.loc[d]
        L.append(f"| {d} | {r['pearson_hourly_source_present']:.3f} | {r['pearson_hourly_source_linkable']:.3f} | "
                 f"{r['pearson_daily_vessels']:.3f} | {r['pearson_daily_source_present']:.3f} |")
    L.append("")
    L.append("- Descriptive only. The hourly flow correlates more with source stock (present/linkable, Pearson up to ~0.44) than with the daily total vessel count (mostly near zero or negative). So a 'source stock x transition probability' structure is more consistent with the data than scaling by daily vessel count, although the stock signal itself is only weak-to-moderate.")
    L.append("")

    L.append("## 7. Hourly transition structure")
    L.append("")
    sim_g = similarity.groupby("task_key").agg(pearson=("pearson_profile", "mean"), cosine=("cosine_similarity", "mean"),
                                              l1=("l1_distance", "mean"), rmse=("rmse_profile", "mean")).reindex(DIRECTIONS)
    L.append("| direction | mean pairwise Pearson | mean cosine | mean L1 | mean RMSE | peak hour (normal mean curve) | zero rate (normal) |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for d in DIRECTIONS:
        s = sim_g.loc[d]
        curve = profile_summary[(profile_summary["group"] == "normal_all") & (profile_summary["task_key"] == d)].sort_values("hour")
        peak = int(curve["mean_hourly_flow"].idxmax()) if len(curve) else -1
        peak_hr = int(curve.loc[peak, "hour"]) if len(curve) else -1
        zr = pairrel[(pairrel["subset"] == "normal_all") & (pairrel["task_key"] == d)]["zero_hour_rate"].iloc[0]
        L.append(f"| {d} | {s['pearson']:.3f} | {s['cosine']:.3f} | {s['l1']:.4f} | {s['rmse']:.4f} | {peak_hr} | {zr:.3f} |")
    L.append("")

    L.append("## 8. Period and day-type comparison")
    L.append("")
    for g_a, g_b in [("normal_pre_outage", "normal_post_recovery"), ("normal_weekday", "normal_weekend")]:
        L.append(f"- **{g_a} vs {g_b}** (mean hourly curves):")
        L.append("")
        L.append(f"  | direction | pearson | cosine | L1 | RMSE | peak A | peak B |")
        L.append(f"  | --- | --- | --- | --- | --- | --- | --- |")
        for d in DIRECTIONS:
            ca = profile_summary[(profile_summary["group"] == g_a) & (profile_summary["task_key"] == d)].sort_values("hour")["mean_hourly_flow"].to_numpy()
            cb = profile_summary[(profile_summary["group"] == g_b) & (profile_summary["task_key"] == d)].sort_values("hour")["mean_hourly_flow"].to_numpy()
            L.append(f"  | {d} | {pearson(ca, cb):.3f} | {cosine(ca, cb):.3f} | {np.abs(ca - cb).sum():.4f} | {np.sqrt(np.mean((ca - cb) ** 2)):.4f} | {int(np.argmax(ca))} | {int(np.argmax(cb))} |")
        L.append("")
    L.append("- Small samples; differences are descriptive, not evidence of a structural break or day-of-week effect.")
    L.append("")

    L.append("## 9. Implications for B modeling")
    L.append("")
    L.append("- Keep direction-hourly-mean as the simple baseline, but note B's hourly shape is very unstable across normal days (mean pairwise Pearson 0.02-0.18, section 7) and flows are sparse (zero rate 0.44-0.85), so this baseline is expected to be weak.")
    L.append("- A 'source stock x transition probability' structure is worth building: flow correlates more with source present/linkable than with daily vessel count, and it naturally enforces the conservation identities that hold exactly in the data.")
    L.append("- Prefer `source_linkable` over `source_present` as the exposure denominator: only linkable vessels can migrate, so transition probability = flow / linkable is the well-defined quantity (present includes no_consecutive vessels that cannot migrate).")
    L.append("- Stay and no_consecutive matter for the mechanism (stay dominates the linkable mass; no_consecutive drives the link_rate dip in the outage) but the six official directions exclude stay, so a stock x transition model over the six directions captures what is scored.")
    L.append("- Do NOT use A's quality weights for B: B did not share A's outage collapse and is source-robust (section 4-5).")
    L.append("- Step 09 priority: compare (a) direction-hourly-mean baseline, (b) source_linkable x transition-probability (per direction/hour, with smoothing/shrinkage for the sparse directions), and possibly (c) a small regularized per-direction count model — judged on normal-target and final-analog views, not the contaminated rolling folds.")
    L.append("- No model is trained or scored in this stage.")
    L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
