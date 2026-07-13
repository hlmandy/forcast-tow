"""Step 22 — A daily vessel count diagnostic.

Analyzes whether U_d (daily unique vessel count) can effectively constrain
H_d (daily A total) and H_{d,z} (daily region totals). Computes the activity
coefficient c_d = H_d / U_d, region shares s_z, and their stability across
normal vs anomaly dates. No models, no predictions, no submission.

Outputs (under outputs/step22_a_vessel_count_diagnostic/):
  1. daily_activity_features.csv  (24)
  2. coefficient_stability.csv    (>=8)
  3. region_share_stability.csv   (>=6)
  4. vessel_count_relationships.csv
  5. analog_day_analysis.csv      (C(17,2)=136)
  6. hourly_profile_by_vessel_quantile.csv
  7. summary.md
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
from itertools import combinations
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from optimized_baseline import REGION_ORDER, add_regions, make_a_labels  # noqa: E402

TRAIN_REL = Path("数据备份/训练集_20180101-0124_拖轮AIS.csv")
STEP01_DAILY = Path("outputs/step01_data_audit/train_daily_overview.csv")
STEP02_DAILY = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
OUT_REL = Path("outputs/step22_a_vessel_count_diagnostic")
PRE_OUTAGE_END = pd.Timestamp("2018-01-11")
DEGRADED_END = pd.Timestamp("2018-01-18")
RUN_CMD = "python " + " ".join(sys.argv)


def period_of(d):
    if d <= PRE_OUTAGE_END: return "pre_outage"
    if d <= DEGRADED_END: return "degraded_period"
    return "post_recovery"


def safe_corr(x, y):
    if len(x) < 3: return np.nan
    xs, ys = pd.Series(x), pd.Series(y)
    if xs.nunique() <= 1 or ys.nunique() <= 1: return np.nan
    return float(xs.corr(ys))


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    # load A labels
    df = pd.read_csv(ROOT / TRAIN_REL, usecols=["mmsi","x","y","sog","time"], dtype={"mmsi":"string","x":"float64","y":"float64","sog":"float32"}, parse_dates=["time"])
    df["hour"] = df["time"].dt.floor("h"); df["date"] = df["time"].dt.normalize()
    df = add_regions(df)
    a_lab = make_a_labels(df)
    a_lab["date"] = a_lab["hour"].dt.normalize(); a_lab["hour_of_day"] = a_lab["hour"].dt.hour

    daily = pd.read_csv(ROOT / STEP01_DAILY); daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    aud = pd.read_csv(ROOT / STEP02_DAILY); aud["date"] = pd.to_datetime(aud["date"]).dt.normalize()
    qmap = dict(zip(aud["date"], aud["quality_regime"]))
    vc_map = dict(zip(daily["date"], daily["unique_vessel_count"]))

    # daily A totals
    dtot = a_lab.groupby(["date","region"])["y"].sum().unstack("region")
    for r in REGION_ORDER:
        if r not in dtot.columns: dtot[r] = 0
    dtot = dtot[REGION_ORDER].fillna(0)
    dtot["H_d"] = dtot.sum(axis=1)
    dtot = dtot.reset_index()
    dtot["U_d"] = dtot["date"].map(vc_map)
    dtot["c_d"] = dtot["H_d"] / dtot["U_d"]
    for r in REGION_ORDER: dtot[f"s_{r}"] = dtot[r] / dtot["H_d"]
    dtot["quality"] = dtot["date"].map(qmap)
    dtot["period"] = dtot["date"].map(period_of)
    dtot["day_type"] = dtot["date"].dt.dayofweek.map(lambda d: "weekend" if d in (5,6) else "weekday")

    # ========== 1. daily_activity_features.csv ==========
    feat = dtot[["date","H_d","core","near","outer","U_d","c_d","s_core","s_near","s_outer","quality","period","day_type"]].copy()
    feat = feat.sort_values("date").reset_index(drop=True)
    feat.to_csv(out_dir / "daily_activity_features.csv", index=False, encoding="utf-8-sig")

    # ========== 2. coefficient_stability.csv ==========
    def stats(series, label):
        s = series.dropna()
        return {"subset": label, "n": int(len(s)), "mean": float(s.mean()) if len(s) else np.nan,
                "std": float(s.std(ddof=1)) if len(s)>1 else np.nan, "cv": float(s.std(ddof=1)/s.mean()) if len(s)>1 and s.mean() else np.nan,
                "min": float(s.min()) if len(s) else np.nan, "max": float(s.max()) if len(s) else np.nan,
                "median": float(s.median()) if len(s) else np.nan}

    cs_rows = []
    for var in ["c_d", "H_d", "U_d"]:
        for subset, mask in [("all", np.ones(len(dtot), bool)),
                             ("normal", dtot["quality"]=="normal"),
                             ("pre_outage_normal", (dtot["quality"]=="normal")&(dtot["period"]=="pre_outage")),
                             ("post_recovery_normal", (dtot["quality"]=="normal")&(dtot["period"]=="post_recovery")),
                             ("degraded", dtot["period"]=="degraded_period"),
                             ("weekday_normal", (dtot["quality"]=="normal")&(dtot["day_type"]=="weekday")),
                             ("weekend_normal", (dtot["quality"]=="normal")&(dtot["day_type"]=="weekend"))]:
            r = stats(dtot.loc[mask, var], subset); r["variable"] = var; cs_rows.append(r)
    cs = pd.DataFrame(cs_rows)
    cs.to_csv(out_dir / "coefficient_stability.csv", index=False, encoding="utf-8-sig")

    # ========== 3. region_share_stability.csv ==========
    rs_rows = []
    for r in REGION_ORDER:
        var = f"s_{r}"
        for subset, mask in [("all", np.ones(len(dtot),bool)), ("normal", dtot["quality"]=="normal"),
                             ("pre_outage_normal", (dtot["quality"]=="normal")&(dtot["period"]=="pre_outage")),
                             ("post_recovery_normal", (dtot["quality"]=="normal")&(dtot["period"]=="post_recovery"))]:
            s = stats(dtot.loc[mask, var], subset); s["region"] = r; rs_rows.append(s)
    rs = pd.DataFrame(rs_rows)
    rs.to_csv(out_dir / "region_share_stability.csv", index=False, encoding="utf-8-sig")

    # ========== 4. vessel_count_relationships.csv ==========
    vr_rows = []
    normal = dtot[dtot["quality"]=="normal"]
    all_d = dtot
    for subset_label, subset in [("normal", normal), ("all", all_d)]:
        vr_rows.append({"subset": subset_label, "pair": "U_d vs H_d", "pearson": safe_corr(subset["U_d"].to_numpy(float), subset["H_d"].to_numpy(float)),
                        "spearman": safe_corr(subset["U_d"].rank().to_numpy(float), subset["H_d"].rank().to_numpy(float))})
        for r in REGION_ORDER:
            vr_rows.append({"subset": subset_label, "pair": f"U_d vs H_{r}", "pearson": safe_corr(subset["U_d"].to_numpy(float), subset[r].to_numpy(float)),
                            "spearman": safe_corr(subset["U_d"].rank().to_numpy(float), subset[r].rank().to_numpy(float))})
        vr_rows.append({"subset": subset_label, "pair": "U_d vs c_d", "pearson": safe_corr(subset["U_d"].to_numpy(float), subset["c_d"].to_numpy(float)),
                        "spearman": safe_corr(subset["U_d"].rank().to_numpy(float), subset["c_d"].rank().to_numpy(float))})
    vr = pd.DataFrame(vr_rows)
    vr.to_csv(out_dir / "vessel_count_relationships.csv", index=False, encoding="utf-8-sig")

    # ========== 5. analog_day_analysis.csv ==========
    normal_dates = sorted(normal["date"].unique())
    analog_rows = []
    for i, j in combinations(range(len(normal_dates)), 2):
        di, dj = normal_dates[i], normal_dates[j]
        ri = normal[normal["date"]==di].iloc[0]; rj = normal[normal["date"]==dj].iloc[0]
        analog_rows.append({"date_i": f"{di:%Y-%m-%d}", "date_j": f"{dj:%Y-%m-%d}",
                            "U_i": int(ri["U_d"]), "U_j": int(rj["U_d"]), "dU": int(ri["U_d"]-rj["U_d"]),
                            "H_i": float(ri["H_d"]), "H_j": float(rj["H_d"]), "dH": float(ri["H_d"]-rj["H_d"]),
                            "c_i": float(ri["c_d"]), "c_j": float(rj["c_d"]), "dc": float(ri["c_d"]-rj["c_d"])})
    analog = pd.DataFrame(analog_rows)
    analog.to_csv(out_dir / "analog_day_analysis.csv", index=False, encoding="utf-8-sig")

    # ========== 6. hourly_profile_by_vessel_quantile.csv ==========
    # split normal dates into low/high U_d, compare profiles
    normal_sorted = normal.sort_values("U_d")
    n_norm = len(normal_sorted)
    low = normal_sorted.head(n_norm//2); high = normal_sorted.tail(n_norm//2)
    hp_rows = []
    for quantile, dates_df in [("low_U", low), ("high_U", high)]:
        for r in REGION_ORDER:
            for h in range(24):
                vals = []
                for _, drow in dates_df.iterrows():
                    d = drow["date"]
                    v = a_lab[(a_lab["date"]==d)&(a_lab["region"]==r)&(a_lab["hour_of_day"]==h)]["y"]
                    vals.append(float(v.iloc[0]) if len(v) else 0.0)
                H_r = float(dates_df[r].mean())
                share = np.mean(vals)/H_r if H_r > 0 else 0.0
                hp_rows.append({"vessel_quantile": quantile, "region": r, "hour": h, "mean_hourly": float(np.mean(vals)), "mean_share": share})
    hp = pd.DataFrame(hp_rows)
    hp.to_csv(out_dir / "hourly_profile_by_vessel_quantile.csv", index=False, encoding="utf-8-sig")

    # ========== summary ==========
    cn = cs[cs["variable"]=="c_d"]
    c_norm = cn[cn["subset"]=="normal"].iloc[0]
    c_pre = cn[cn["subset"]=="pre_outage_normal"].iloc[0]
    c_post = cn[cn["subset"]=="post_recovery_normal"].iloc[0]
    u_h_pearson = vr[(vr["subset"]=="normal")&(vr["pair"]=="U_d vs H_d")]["pearson"].iloc[0]
    u_c_pearson = vr[(vr["subset"]=="normal")&(vr["pair"]=="U_d vs c_d")]["pearson"].iloc[0]

    L = []
    L.append("# Step 22 A Vessel-Count Diagnostic")
    L.append("")
    L.append(f"Run command: `{RUN_CMD}`")
    L.append("")
    L.append("## 1. Activity coefficient c_d = H_d / U_d")
    L.append("")
    L.append("| subset | n | mean | std | cv | min | max |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for _, r in cn.iterrows():
        L.append(f"| {r['subset']} | {int(r['n'])} | {r['mean']:.2f} | {r['std']:.3f} | {r['cv']:.3f} | {r['min']:.2f} | {r['max']:.2f} |")
    L.append("")
    L.append(f"- Normal c_d cv = {c_norm['cv']:.3f}; if stable, U_d * c_hat is a viable daily-total predictor.")
    L.append(f"- U_d vs H_d Pearson (normal) = {u_h_pearson:.3f}; U_d vs c_d Pearson = {u_c_pearson:.3f}.")
    L.append(f"- If U_d vs c_d is near 0, c_d is vessel-count-independent -> U_d scaling is clean.")
    L.append("")
    L.append("## 2. Region shares s_z")
    L.append("")
    L.append("| region | subset | mean | cv |")
    L.append("| --- | --- | --- | --- |")
    for _, r in rs[rs["subset"]=="normal"].iterrows():
        L.append(f"| {r['region']} | normal | {r['mean']:.3f} | {r['cv']:.3f} |")
    L.append("")
    L.append("## 3. Analog-day potential")
    L.append(f"- {len(analog)} normal-day pairs. corr(|dU|, |dH|) = {safe_corr(analog['dU'].abs().to_numpy(float), analog['dH'].abs().to_numpy(float)):.3f}.")
    L.append("- If positive, similar-U days have similar H -> analog weighting is viable.")
    L.append("")
    L.append("## 4. Hourly profile by vessel-count quantile")
    L.append("- See hourly_profile_by_vessel_quantile.csv. If low-U and high-U profiles differ, U_d should modulate the shape, not just the total.")
    L.append("")
    L.append("## 5. Decision for Step 23")
    L.append(f"- c_d stability (normal cv={c_norm['cv']:.3f}) determines whether U_d*c_hat is viable.")
    L.append(f"- Region share stability determines whether H_d -> H_{{d,z}} via s_z is viable.")
    L.append("- If both are sufficiently stable, proceed to Step 23 vessel-constrained three-layer model.")
    L.append("")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\nc_d normal: mean={c_norm['mean']:.2f} cv={c_norm['cv']:.3f}")
    print(f"U_d vs H_d pearson (normal): {u_h_pearson:.3f}")
    print(f"U_d vs c_d pearson (normal): {u_c_pearson:.3f}")
    print("\n=== Output files ===")
    for p in sorted(out_dir.glob("*")):
        if p.suffix == ".csv":
            rows = len(pd.read_csv(p, encoding="utf-8-sig"))
        else:
            rows = sum(1 for _ in open(p, encoding="utf-8"))
        print(f"  {p.relative_to(ROOT)}  rows={rows}")
    print("\nDone.")


if __name__ == "__main__":
    main()
