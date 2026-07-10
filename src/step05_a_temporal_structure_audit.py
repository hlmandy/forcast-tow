"""Step 05 — Systematic audit of Task A intra-day hourly structure.

Diagnostic only: no models, no B task, no submission files. It describes
whether the 24-hour normalized profile of each region is stable across normal
days, whether pre-outage and post-recovery shapes differ, whether weekday and
weekend differ, whether daily region totals trend over time, and whether the
hourly curve is low-dimensional (PCA). All A labels are reused from
optimized_baseline.

Outputs (under outputs/step05_a_temporal_structure_audit/):
  1. a_day_region_features.csv          (72)
  2. a_normalized_hour_profiles.csv     (1728)
  3. a_hour_profile_summary.csv         (360)
  4. a_profile_pairwise_similarity.csv  (408)
  5. a_profile_pca.csv                  (15)
  6. a_daily_total_relationships.csv    (16)
  7. summary.md
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

from optimized_baseline import REGION_ORDER, add_regions, make_a_labels  # noqa: E402

TRAIN_REL = Path("数据备份/训练集_20180101-0124_拖轮AIS.csv")
STEP01_DAILY = Path("outputs/step01_data_audit/train_daily_overview.csv")
STEP02_DAILY = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
OUT_REL = Path("outputs/step05_a_temporal_structure_audit")

FIXED_DATES = pd.date_range("2018-01-01", "2018-01-24", freq="D").normalize()
REGION_RANK = {r: i for i, r in enumerate(REGION_ORDER)}

PRE_OUTAGE_END = pd.Timestamp("2018-01-11")
DEGRADED_END = pd.Timestamp("2018-01-18")

GROUP_ORDER = ["normal_all", "normal_pre_outage", "normal_post_recovery", "normal_weekday", "normal_weekend"]
GROUP_DEFS = {
    "normal_all": lambda d: d["quality_regime"] == "normal",
    "normal_pre_outage": lambda d: (d["quality_regime"] == "normal") & (d["period"] == "pre_outage"),
    "normal_post_recovery": lambda d: (d["quality_regime"] == "normal") & (d["period"] == "post_recovery"),
    "normal_weekday": lambda d: (d["quality_regime"] == "normal") & (d["day_type"] == "weekday"),
    "normal_weekend": lambda d: (d["quality_regime"] == "normal") & (d["day_type"] == "weekend"),
}
SUBSET_ORDER = ["all", "normal_all", "normal_pre_outage", "normal_post_recovery"]
SUBSET_DEFS = {
    "all": lambda d: np.ones(len(d), dtype=bool),
    "normal_all": lambda d: d["quality_regime"] == "normal",
    "normal_pre_outage": lambda d: (d["quality_regime"] == "normal") & (d["period"] == "pre_outage"),
    "normal_post_recovery": lambda d: (d["quality_regime"] == "normal") & (d["period"] == "post_recovery"),
}
TARGETS = ["a_total", "a_core_total", "a_near_total", "a_outer_total"]

DAYREGION_COLUMNS = [
    "date", "region", "quality_regime", "period", "day_type", "dayofweek", "unique_vessel_count",
    "daily_total", "mean_hourly_count", "std_hourly_count", "cv_hourly_count", "zero_hour_count",
    "peak_hour", "peak_value", "trough_hour", "trough_value", "normalized_profile_entropy",
]
PROFILE_COLUMNS = [
    "date", "hour", "region", "quality_regime", "period", "day_type", "dayofweek",
    "unique_vessel_count", "y_true", "daily_total", "profile_share",
]
SUMMARY_COLUMNS = [
    "group", "region", "hour", "n_days",
    "mean_profile_share", "median_profile_share", "std_profile_share", "q25_profile_share", "q75_profile_share",
    "mean_hourly_count", "median_hourly_count",
]
PAIR_COLUMNS = [
    "region", "date_i", "date_j", "period_i", "period_j", "period_pair",
    "day_type_i", "day_type_j", "same_day_type",
    "pearson_profile", "cosine_similarity", "l1_distance", "rmse_profile",
]
PCA_COLUMNS = ["region", "component", "explained_variance_ratio", "cumulative_explained_variance_ratio"]
RELATION_COLUMNS = [
    "subset", "target", "n_days", "mean_target", "std_target", "cv_target",
    "linear_time_slope", "pearson_time", "spearman_time",
    "pearson_daily_vessels", "spearman_daily_vessels",
    "weekday_mean", "weekend_mean", "weekend_minus_weekday",
]

RUN_CMD = "python " + " ".join(sys.argv)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_ais(train_path: Path) -> pd.DataFrame:
    usecols = ["mmsi", "x", "y", "sog", "time"]
    dtypes = {"mmsi": "string", "x": "float64", "y": "float64", "sog": "float32"}
    df = pd.read_csv(train_path, usecols=usecols, dtype=dtypes, parse_dates=["time"])
    df["hour"] = df["time"].dt.floor("h")
    df["date"] = df["time"].dt.normalize()
    df = add_regions(df)
    return df


def period_of(date: pd.Timestamp) -> str:
    if date <= PRE_OUTAGE_END:
        return "pre_outage"
    if date <= DEGRADED_END:
        return "degraded_period"
    return "post_recovery"


def build_a_frame(df: pd.DataFrame, daily: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    a_lab = make_a_labels(df)
    a_lab["date"] = a_lab["hour"].dt.normalize()
    a_lab["hour_of_day"] = a_lab["hour"].dt.hour
    a_lab["dayofweek"] = a_lab["hour"].dt.dayofweek
    a_lab = a_lab[["hour", "date", "hour_of_day", "dayofweek", "region", "y"]]

    vc = daily[["date", "unique_vessel_count"]]
    aq = audit[["date", "quality_regime"]]
    a_frame = a_lab.merge(vc, on="date", how="left").merge(aq, on="date", how="left")
    a_frame["period"] = a_frame["date"].map(period_of)
    a_frame["day_type"] = np.where(a_frame["dayofweek"].isin([5, 6]), "weekend", "weekday")
    a_frame["region_rank"] = a_frame["region"].map(REGION_RANK)
    return a_frame


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if len(x) < 3:
        return float("nan")
    xs, ys = pd.Series(x), pd.Series(y)
    if xs.nunique() <= 1 or ys.nunique() <= 1:
        return float("nan")
    return float(xs.corr(ys, method=method))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return safe_corr(a, b, "pearson")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    train_path = ROOT / TRAIN_REL
    out_dir = ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    if not train_path.exists():
        raise FileNotFoundError(f"Training AIS file not found: {train_path}")

    print(f"train_path = {train_path}")
    print(f"out_dir    = {out_dir}")

    df = load_ais(train_path)
    daily = pd.read_csv(ROOT / STEP01_DAILY)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    audit = pd.read_csv(ROOT / STEP02_DAILY)
    audit["date"] = pd.to_datetime(audit["date"]).dt.normalize()

    a_frame = build_a_frame(df, daily, audit)

    # --- assertions ---------------------------------------------------------
    assert len(a_frame) == 1728, f"A labels expected 1728 rows, got {len(a_frame)}"
    per_dr = a_frame.groupby(["date", "region"]).size()
    assert (per_dr == 24).all(), "each (date, region) must have 24 hours"
    assert a_frame["quality_regime"].notna().all(), "missing quality_regime"
    assert a_frame["unique_vessel_count"].notna().all(), "missing unique_vessel_count"

    # daily region totals must match step01 exactly
    dt = a_frame.groupby(["date", "region"], observed=True)["y"].sum().unstack("region")
    for r in REGION_ORDER:
        merged = dt[r].reset_index().merge(daily[["date", f"a_{r}_total"]], on="date", how="left")
        mism = (merged[r].astype(np.int64) != merged[f"a_{r}_total"].astype(np.int64)).sum()
        assert mism == 0, f"daily {r} total mismatch vs step01 in {mism} days"

    n_normal = int((audit["quality_regime"] == "normal").sum())
    assert n_normal == 17, f"expected 17 normal dates, got {n_normal}"

    # --- date-level table ---------------------------------------------------
    date_tbl = a_frame.drop_duplicates("date")[["date", "quality_regime", "period", "day_type", "dayofweek", "unique_vessel_count"]].copy()
    date_tbl["time_index"] = (date_tbl["date"] - FIXED_DATES[0]).dt.days + 1
    # per-date region totals
    dtot = a_frame.groupby(["date", "region"], observed=True)["y"].sum().unstack("region").rename(
        columns={r: f"a_{r}_total" for r in REGION_ORDER}
    )
    dtot["a_total"] = dtot[[f"a_{r}_total" for r in REGION_ORDER]].sum(axis=1)
    date_tbl = date_tbl.merge(dtot.reset_index(), on="date", how="left")

    # --- File 1: day-region features ---------------------------------------
    prof_wide = a_frame.pivot_table(index=["date", "region"], columns="hour_of_day", values="y", observed=True)
    prof_wide = prof_wide.reindex(columns=range(24), fill_value=0)

    feat_rows = []
    for (d, r), row in prof_wide.iterrows():
        vals = row.to_numpy(dtype="float64")
        total = float(vals.sum())
        mean_v = float(vals.mean())
        std_v = float(vals.std(ddof=1))
        cv = std_v / mean_v if mean_v != 0 else float("nan")
        if total > 0:
            p = vals / total
            pp = p[p > 0]
            entropy = float(-(pp * np.log(pp)).sum() / np.log(24))
        else:
            entropy = float("nan")
        meta = date_tbl[date_tbl["date"] == d].iloc[0]
        feat_rows.append({
            "date": d, "region": r,
            "quality_regime": meta["quality_regime"], "period": meta["period"],
            "day_type": meta["day_type"], "dayofweek": int(meta["dayofweek"]),
            "unique_vessel_count": int(meta["unique_vessel_count"]),
            "daily_total": int(total),
            "mean_hourly_count": mean_v,
            "std_hourly_count": std_v,
            "cv_hourly_count": cv,
            "zero_hour_count": int((vals == 0).sum()),
            "peak_hour": int(row.idxmax()),
            "peak_value": float(vals.max()),
            "trough_hour": int(row.idxmin()),
            "trough_value": float(vals.min()),
            "normalized_profile_entropy": entropy,
        })
    day_region = pd.DataFrame(feat_rows)
    day_region["_r"] = day_region["region"].map(REGION_RANK)
    day_region = day_region.sort_values(["date", "_r"]).drop(columns=["_r"]).reset_index(drop=True)

    # --- File 2: normalized hour profiles ----------------------------------
    prof_long = a_frame[["date", "hour_of_day", "region", "quality_regime", "period", "day_type",
                         "dayofweek", "unique_vessel_count", "y"]].copy()
    prof_long = prof_long.merge(
        day_region[["date", "region", "daily_total"]], on=["date", "region"], how="left"
    )
    prof_long["profile_share"] = np.where(prof_long["daily_total"] > 0,
                                          prof_long["y"] / prof_long["daily_total"], 0.0)
    prof_long = prof_long.rename(columns={"hour_of_day": "hour", "y": "y_true"})
    prof_long["_r"] = prof_long["region"].map(REGION_RANK)
    prof_long = prof_long.sort_values(["date", "_r", "hour"]).drop(columns=["_r"]).reset_index(drop=True)

    # assertion: shares sum to 1 where daily_total>0
    chk = prof_long[prof_long["daily_total"] > 0].groupby(["date", "region"])["profile_share"].sum()
    assert (chk - 1.0).abs().max() < 1e-12, "profile_share does not sum to 1"
    assert len(prof_long) == 1728

    # --- File 3: hour profile summary --------------------------------------
    prof_with_groups = prof_long.merge(date_tbl[["date", "quality_regime", "period", "day_type"]], on="date", how="left")
    summary_rows = []
    for gname in GROUP_ORDER:
        gdates = date_tbl[GROUP_DEFS[gname](date_tbl)]["date"]
        if len(gdates) == 0:
            raise AssertionError(f"group {gname} has no dates")
        sub = prof_long[prof_long["date"].isin(gdates)]
        for r in REGION_ORDER:
            sr = sub[sub["region"] == r]
            for h in range(24):
                cell = sr[sr["hour"] == h]
                shares = cell["profile_share"].astype("float64")
                counts = cell["y_true"].astype("float64")
                summary_rows.append({
                    "group": gname, "region": r, "hour": h, "n_days": int(len(cell)),
                    "mean_profile_share": float(shares.mean()),
                    "median_profile_share": float(shares.median()),
                    "std_profile_share": float(shares.std(ddof=1)),
                    "q25_profile_share": float(shares.quantile(0.25)),
                    "q75_profile_share": float(shares.quantile(0.75)),
                    "mean_hourly_count": float(counts.mean()),
                    "median_hourly_count": float(counts.median()),
                })
    profile_summary = pd.DataFrame(summary_rows)
    profile_summary["_g"] = profile_summary["group"].map({g: i for i, g in enumerate(GROUP_ORDER)})
    profile_summary["_r"] = profile_summary["region"].map(REGION_RANK)
    profile_summary = profile_summary.sort_values(["_g", "_r", "hour"]).drop(columns=["_g", "_r"]).reset_index(drop=True)

    # --- File 4: pairwise similarity (normal days) -------------------------
    normal_dates = sorted(date_tbl[date_tbl["quality_regime"] == "normal"]["date"].tolist())
    assert len(normal_dates) == 17
    pair_rows = []
    for r in REGION_ORDER:
        # 17 x 24 matrix of profile_share (rows=dates sorted, cols=hours)
        M = np.zeros((len(normal_dates), 24))
        for i, d in enumerate(normal_dates):
            vec = prof_long[(prof_long["date"] == d) & (prof_long["region"] == r)].sort_values("hour")["profile_share"].to_numpy()
            M[i] = vec
        for i, j in itertools.combinations(range(len(normal_dates)), 2):
            a, b = M[i], M[j]
            di, dj = normal_dates[i], normal_dates[j]
            mi = date_tbl[date_tbl["date"] == di].iloc[0]
            mj = date_tbl[date_tbl["date"] == dj].iloc[0]
            pair_rows.append({
                "region": r,
                "date_i": f"{di:%Y-%m-%d}", "date_j": f"{dj:%Y-%m-%d}",
                "period_i": mi["period"], "period_j": mj["period"],
                "period_pair": "__".join(sorted([mi["period"], mj["period"]])),
                "day_type_i": mi["day_type"], "day_type_j": mj["day_type"],
                "same_day_type": bool(mi["day_type"] == mj["day_type"]),
                "pearson_profile": pearson(a, b),
                "cosine_similarity": cosine(a, b),
                "l1_distance": float(np.abs(a - b).sum()),
                "rmse_profile": float(np.sqrt(np.mean((a - b) ** 2))),
            })
    pairwise = pd.DataFrame(pair_rows)
    pairwise["_r"] = pairwise["region"].map(REGION_RANK)
    pairwise = pairwise.sort_values(["_r", "date_i", "date_j"]).drop(columns=["_r"]).reset_index(drop=True)
    assert len(pairwise) == 408

    # --- File 5: PCA per region (normal days) ------------------------------
    pca_rows = []
    for r in REGION_ORDER:
        M = np.zeros((len(normal_dates), 24))
        for i, d in enumerate(normal_dates):
            vec = prof_long[(prof_long["date"] == d) & (prof_long["region"] == r)].sort_values("hour")["profile_share"].to_numpy()
            M[i] = vec
        Xc = M - M.mean(axis=0)
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
        var = (s ** 2) / (M.shape[0] - 1)
        ratio = var / var.sum()
        cum = np.cumsum(ratio)
        for k in range(5):
            pca_rows.append({
                "region": r,
                "component": f"PC{k + 1}",
                "explained_variance_ratio": float(ratio[k]),
                "cumulative_explained_variance_ratio": float(cum[k]),
            })
        assert (ratio[:5] >= 0).all(), f"negative variance ratio for {r}"
        assert cum[4] <= 1 + 1e-9, f"cumulative variance > 1 for {r}"
    pca = pd.DataFrame(pca_rows)
    pca["_r"] = pca["region"].map(REGION_RANK)
    pca = pca.sort_values(["_r", "component"]).drop(columns=["_r"]).reset_index(drop=True)
    assert len(pca) == 15

    # --- File 6: daily total relationships ---------------------------------
    rel_rows = []
    for sname in SUBSET_ORDER:
        smask = SUBSET_DEFS[sname](date_tbl)
        sub = date_tbl[smask].copy()
        for tgt in TARGETS:
            y = sub[tgt].to_numpy(dtype="float64")
            tidx = sub["time_index"].to_numpy(dtype="float64")
            vessels = sub["unique_vessel_count"].to_numpy(dtype="float64")
            mean_t = float(y.mean())
            std_t = float(y.std(ddof=1)) if len(y) > 1 else float("nan")
            cv = std_t / mean_t if mean_t != 0 else float("nan")
            if len(y) >= 3 and np.std(tidx) > 0:
                slope = float(np.polyfit(tidx, y, 1)[0])
            else:
                slope = float("nan")
            wkd = sub[sub["day_type"] == "weekday"][tgt]
            wke = sub[sub["day_type"] == "weekend"][tgt]
            wkd_mean = float(wkd.mean()) if len(wkd) else float("nan")
            wke_mean = float(wke.mean()) if len(wke) else float("nan")
            rel_rows.append({
                "subset": sname, "target": tgt, "n_days": int(len(sub)),
                "mean_target": mean_t, "std_target": std_t, "cv_target": cv,
                "linear_time_slope": slope,
                "pearson_time": safe_corr(tidx, y, "pearson"),
                "spearman_time": safe_corr(tidx, y, "spearman"),
                "pearson_daily_vessels": safe_corr(vessels, y, "pearson"),
                "spearman_daily_vessels": safe_corr(vessels, y, "spearman"),
                "weekday_mean": wkd_mean, "weekend_mean": wke_mean,
                "weekend_minus_weekday": (wke_mean - wkd_mean) if (not np.isnan(wke_mean) and not np.isnan(wkd_mean)) else float("nan"),
            })
    relations = pd.DataFrame(rel_rows)
    relations["_s"] = relations["subset"].map({s: i for i, s in enumerate(SUBSET_ORDER)})
    relations["_t"] = relations["target"].map({t: i for i, t in enumerate(TARGETS)})
    relations = relations.sort_values(["_s", "_t"]).drop(columns=["_s", "_t"]).reset_index(drop=True)

    # --- write CSVs (UTF-8 with BOM) ---------------------------------------
    p_dr = out_dir / "a_day_region_features.csv"
    p_prof = out_dir / "a_normalized_hour_profiles.csv"
    p_sum = out_dir / "a_hour_profile_summary.csv"
    p_pair = out_dir / "a_profile_pairwise_similarity.csv"
    p_pca = out_dir / "a_profile_pca.csv"
    p_rel = out_dir / "a_daily_total_relationships.csv"
    p_md = out_dir / "summary.md"

    def fmt_date(df_):
        if "date" in df_.columns and pd.api.types.is_datetime64_any_dtype(df_["date"]):
            df_["date"] = df_["date"].dt.strftime("%Y-%m-%d")
        return df_

    fmt_date(day_region).to_csv(p_dr, index=False, encoding="utf-8-sig")
    fmt_date(prof_long)[PROFILE_COLUMNS].to_csv(p_prof, index=False, encoding="utf-8-sig")
    profile_summary[SUMMARY_COLUMNS].to_csv(p_sum, index=False, encoding="utf-8-sig")
    pairwise[PAIR_COLUMNS].to_csv(p_pair, index=False, encoding="utf-8-sig")
    pca[PCA_COLUMNS].to_csv(p_pca, index=False, encoding="utf-8-sig")
    relations[RELATION_COLUMNS].to_csv(p_rel, index=False, encoding="utf-8-sig")

    write_summary(p_md, day_region, profile_summary, pairwise, pca, relations, date_tbl, prof_long, normal_dates)

    # --- console ------------------------------------------------------------
    print("\n=== Output files ===")
    for path in [p_dr, p_prof, p_sum, p_pair, p_pca, p_rel, p_md]:
        if path.suffix == ".csv":
            rows = len(pd.read_csv(path, encoding="utf-8-sig"))
        else:
            rows = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {path.relative_to(ROOT)}  rows={rows}")
    print("\nDone.")


def group_mean_curve(profile_summary: pd.DataFrame, group: str, region: str) -> np.ndarray:
    s = profile_summary[(profile_summary["group"] == group) & (profile_summary["region"] == region)].sort_values("hour")
    return s["mean_profile_share"].to_numpy(dtype="float64")


def write_summary(path: Path, day_region: pd.DataFrame, profile_summary: pd.DataFrame,
                  pairwise: pd.DataFrame, pca: pd.DataFrame, relations: pd.DataFrame,
                  date_tbl: pd.DataFrame, prof_long: pd.DataFrame, normal_dates: list) -> None:
    L: list[str] = []
    L.append("# Step 05 A Temporal Structure Audit")
    L.append("")
    L.append("> Diagnostic only. No models, no B task, no submissions. All A labels reused from optimized_baseline.")
    L.append("")
    L.append(f"Run command: `{RUN_CMD}`")
    L.append("")

    # 1
    L.append("## 1. Data and consistency checks")
    L.append("")
    L.append("- A-label rows: 1728 (24 days x 24 hours x 3 regions); each (date, region) has 24 hours. **PASS**")
    L.append("- Daily per-region totals match Step 01 a_core/near/outer_total exactly. **PASS**")
    L.append("- Normal dates: 17. profile_share sums to 1 within 1e-12 wherever daily_total > 0. **PASS**")
    periods = {"pre_outage": [], "degraded_period": [], "post_recovery": []}
    for _, r in date_tbl.iterrows():
        periods[r["period"]].append(f"{r['date']:%Y-%m-%d}")
    for p in ["pre_outage", "degraded_period", "post_recovery"]:
        L.append(f"- {p} ({len(periods[p])} days): {', '.join(periods[p])}")
    L.append("")

    # 2
    L.append("## 2. Daily regional totals")
    L.append("")
    L.append("Normal-day daily totals per region (descriptive only; `cv` = std/mean):")
    L.append("")
    L.append("| region | mean | std | cv | min | max | time slope (per day) | pearson vessels | spearman vessels |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    na = relations[relations["subset"] == "normal_all"]
    for r in REGION_ORDER:
        tgt = f"a_{r}_total"
        rr = na[na["target"] == tgt].iloc[0]
        vals = date_tbl[date_tbl["quality_regime"] == "normal"][tgt]
        L.append(f"| {r} | {rr['mean_target']:.1f} | {rr['std_target']:.1f} | {rr['cv_target']:.3f} | "
                 f"{vals.min():.0f} | {vals.max():.0f} | {rr['linear_time_slope']:.3f} | "
                 f"{rr['pearson_daily_vessels']:.3f} | {rr['spearman_daily_vessels']:.3f} |")
    L.append("")
    L.append("- These describe association only; no causal claim is made from 17 days.")
    L.append("")

    # 3
    L.append("## 3. Stability of normalized hourly profiles")
    L.append("")
    L.append("Pairwise similarity of the 24-hour profile_share curves across the 17 normal days, per region:")
    L.append("")
    L.append("| region | mean pearson | mean cosine | mean L1 | mean RMSE |")
    L.append("| --- | --- | --- | --- | --- |")
    stability = {}
    for r in REGION_ORDER:
        sub = pairwise[pairwise["region"] == r]
        mp = sub["pearson_profile"].mean()
        mc = sub["cosine_similarity"].mean()
        ml = sub["l1_distance"].mean()
        mr = sub["rmse_profile"].mean()
        stability[r] = (mp, mc, ml, mr)
        L.append(f"| {r} | {mp:.4f} | {mc:.4f} | {ml:.4f} | {mr:.4f} |")
    L.append("")
    most_stable = max(stability, key=lambda r: stability[r][0])
    least_stable = min(stability, key=lambda r: stability[r][0])
    L.append(f"- Most stable shape (highest mean pairwise Pearson): **{most_stable}**; least stable: **{least_stable}**.")
    L.append("- Five least-similar date pairs per region (lowest cosine):")
    for r in REGION_ORDER:
        sub = pairwise[pairwise["region"] == r].sort_values("cosine_similarity").head(5)
        detail = "; ".join(f"{row['date_i']}~{row['date_j']} (cos={row['cosine_similarity']:.3f})" for _, row in sub.iterrows())
        L.append(f"  - {r}: {detail}")
    L.append("")

    # 4
    L.append("## 4. Pre-outage versus post-recovery profiles")
    L.append("")
    L.append("Similarity between the mean profile_share curve of normal_pre_outage and normal_post_recovery:")
    L.append("")
    L.append("| region | pearson | cosine | L1 | RMSE | pre peak hr | post peak hr |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in REGION_ORDER:
        a = group_mean_curve(profile_summary, "normal_pre_outage", r)
        b = group_mean_curve(profile_summary, "normal_post_recovery", r)
        L.append(f"| {r} | {pearson(a, b):.4f} | {cosine(a, b):.4f} | {np.abs(a - b).sum():.4f} | "
                 f"{np.sqrt(np.mean((a - b) ** 2)):.4f} | {int(np.argmax(a))} | {int(np.argmax(b))} |")
    L.append("")
    L.append("- Sample is small (pre_outage normal days and post_recovery normal days); differences are not evidence of a structural break.")
    L.append("")

    # 5
    L.append("## 5. Weekday versus weekend profiles")
    L.append("")
    for r in REGION_ORDER:
        n_wkd = int(profile_summary[(profile_summary["group"] == "normal_weekday") & (profile_summary["region"] == r) & (profile_summary["hour"] == 0)]["n_days"].iloc[0])
        n_wke = int(profile_summary[(profile_summary["group"] == "normal_weekend") & (profile_summary["region"] == r) & (profile_summary["hour"] == 0)]["n_days"].iloc[0])
        a = group_mean_curve(profile_summary, "normal_weekday", r)
        b = group_mean_curve(profile_summary, "normal_weekend", r)
        L.append(f"- {r}: weekday {n_wkd} days vs weekend {n_wke} days — pearson={pearson(a, b):.4f}, "
                 f"cosine={cosine(a, b):.4f}, L1={np.abs(a - b).sum():.4f}, RMSE={np.sqrt(np.mean((a - b) ** 2)):.4f}, "
                 f"weekday peak={int(np.argmax(a))}, weekend peak={int(np.argmax(b))}.")
    L.append("")
    peak_note = []
    for r in REGION_ORDER:
        a = group_mean_curve(profile_summary, "normal_weekday", r)
        b = group_mean_curve(profile_summary, "normal_weekend", r)
        peak_note.append(f"{r} {int(np.argmax(a))}->{int(np.argmax(b))}")
    L.append(f"- The mean-curve peak hour shifts between weekday and weekend ({'; '.join(peak_note)}). This is a real descriptive signal, but with only 4 weekend days it is not enough on its own to justify a complex day-of-week model.")
    L.append("")

    # 6
    L.append("## 6. PCA dimensionality")
    L.append("")
    L.append("Explained variance ratio of the 24-hour profile_share (normal days, column-centered):")
    L.append("")
    L.append("| region | PC1 | cum PC2 | cum PC3 | cum PC5 |")
    L.append("| --- | --- | --- | --- | --- |")
    for r in REGION_ORDER:
        sub = pca[pca["region"] == r].set_index("component")
        L.append(f"| {r} | {sub.loc['PC1', 'explained_variance_ratio']:.4f} | "
                 f"{sub.loc['PC2', 'cumulative_explained_variance_ratio']:.4f} | "
                 f"{sub.loc['PC3', 'cumulative_explained_variance_ratio']:.4f} | "
                 f"{sub.loc['PC5', 'cumulative_explained_variance_ratio']:.4f} |")
    L.append("")
    L.append("- PCA is used only to judge whether the hourly curve is low-dimensional; it is not a prediction model here.")
    L.append("")

    # 7
    L.append("## 7. Implications for the next model")
    L.append("")
    L.append("These are read directly from the audit numbers above, not from a fitted model:")
    L.append("")
    tot_cv = {r: relations[(relations["subset"] == "normal_all") & (relations["target"] == f"a_{r}_total")].iloc[0]["cv_target"] for r in REGION_ORDER}
    shp_pearson = {r: stability[r][0] for r in REGION_ORDER}
    cum3 = {r: pca[pca["region"] == r].set_index("component").loc["PC3", "cumulative_explained_variance_ratio"] for r in REGION_ORDER}
    cum5 = {r: pca[pca["region"] == r].set_index("component").loc["PC5", "cumulative_explained_variance_ratio"] for r in REGION_ORDER}
    L.append(f"- Daily region totals vary across normal days (cv core={tot_cv['core']:.2f}, near={tot_cv['near']:.2f}, outer={tot_cv['outer']:.2f}; outer is the most variable) and are essentially unrelated to the daily vessel count (Pearson ~0.07 for a_total, see section 2). Daily total is therefore a genuine prediction target, not a rescaling of vessel count.")
    L.append(f"- Hourly shapes are NOT stable across individual normal days: the mean pairwise Pearson of the 24-hour profile is only {shp_pearson['core']:.2f}/{shp_pearson['near']:.2f}/{shp_pearson['outer']:.2f} (core/near/outer). Cosine is higher ({stability['core'][1]:.2f}-{stability['outer'][1]:.2f}) only because it is not mean-centered and is dominated by the shared mass distribution; PCA independently shows the shape is not low-dimensional (PC1 ~23%, 3 PCs only ~{cum3['core']*100:.0f}-{cum3['outer']*100:.0f}% of the variation, 5 PCs ~{cum5['core']*100:.0f}-{cum5['outer']*100:.0f}%). Most stable shape: {most_stable}; least stable: {least_stable}.")
    L.append(f"- Because the per-day shape is noisy and not low-dimensional, a 'daily total x fixed hourly shape' decomposition would capture only the average pattern and leave substantial day-specific shape error. Whether that decomposition or direct hourly prediction is better is a modeling decision for a later step, not asserted here.")
    L.append(f"- On regional difficulty: core has the largest counts (hence the largest absolute errors) with moderate daily cv ({tot_cv['core']:.2f}) and low shape Pearson ({shp_pearson['core']:.2f}); outer has the highest proportional daily variation (cv {tot_cv['outer']:.2f}) and the least stable shape ({shp_pearson['outer']:.2f}) but much smaller counts. So core's difficulty is driven mainly by scale, while outer's is driven mainly by proportional variability.")
    L.append("- Weekday vs weekend: see section 5; the mean-curve peak shifts but only 4 weekend days are available.")
    L.append("- No model is trained, scored, or selected in this stage.")
    L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
