"""Step 14 (B1) — Three layer-pair data reconstruction diagnostic for Task B.

No models. Reconstructs the six directed flows into three bidirectional
layer-pairs (core<->near, near<->outer, core<->outer) and diagnoses whether
the pair structure is more stable / less sparse than the six independent
directions. Boundary rule reused: 2018-01-24 23:00 is right-censored and
excluded from all statistics.

For pair (a,b):  X = y(a->b) + y(b->a)   (bidirectional total)
                 r = y(a->b) / X          (direction ratio; NaN if X=0)
                 F = y(a->b) - y(b->a)    (net flow)
                 p_h = X_h / T_d          (hour share; 0 if T_d=0), T_d = sum_h X

Outputs (under outputs/step14_b_pair_diagnostic/):
  1. pair_hourly_decomp.csv
  2. pair_daily_features.csv
  3. pair_zero_rate_comparison.csv
  4. pair_stability_summary.csv
  5. pair_direction_ratio_profile.csv
  6. pair_net_flow.csv
  7. pair_weekday_weekend.csv
  8. summary.md
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PF_PATH = Path("outputs/step08_b_state_transition_audit/b_hourly_pair_flow.csv")
STEP02 = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
OUT_REL = Path("outputs/step14_b_pair_diagnostic")

UNSCORABLE = pd.Timestamp("2018-01-24 23:00:00")
PRE_OUTAGE_END = pd.Timestamp("2018-01-11")
DEGRADED_END = pd.Timestamp("2018-01-18")
# pair: name, a, b, dir_ab, dir_ba
PAIRS = [
    ("core_near", "core", "near", "core->near", "near->core"),
    ("near_outer", "near", "outer", "near->outer", "outer->near"),
    ("core_outer", "core", "outer", "core->outer", "outer->core"),
]
PAIR_ORDER = {p[0]: i for i, p in enumerate(PAIRS)}

HOURLY_COLS = ["date", "hour", "hour_of_day", "dayofweek", "day_type", "audit_quality_regime", "period",
               "pair", "y_ab", "y_ba", "X", "r", "F", "T_daily", "p"]
DAILY_COLS = ["date", "pair", "audit_quality_regime", "period", "day_type", "T", "mean_X", "std_X", "cv_X",
              "zero_hour_count", "peak_hour", "net_flow_daily", "abs_net_flow", "r_daily"]
ZERO_COLS = ["pair", "dir_ab", "dir_ba", "zero_rate_ab", "zero_rate_ba", "zero_rate_X", "zero_rate_reduction_pp"]
STAB_COLS = ["pair", "daily_cv_T", "daily_cv_ab", "daily_cv_ba",
             "hourshare_pearson_X_normal", "hourshare_pearson_ab_normal", "hourshare_pearson_ba_normal",
             "dirratio_hour_std_normal", "dirratio_global"]
DIRRATIO_COLS = ["pair", "hour", "n_days", "mean_r", "std_r", "q25_r", "q75_r", "n_X_positive"]
NETFLOW_COLS = ["date", "pair", "net_flow_daily", "cum_net_flow"]
WKE_COLS = ["pair", "n_weekday", "n_weekend", "weekday_mean_X", "weekend_mean_X",
            "weekday_mean_r", "weekend_mean_r", "weekday_mean_T", "weekend_mean_T"]

RUN_CMD = "python " + " ".join(sys.argv)


def period_of(d):
    if d <= PRE_OUTAGE_END:
        return "pre_outage"
    if d <= DEGRADED_END:
        return "degraded_period"
    return "post_recovery"


def safe_corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(pd.Series(a).corr(pd.Series(b), method="pearson"))


def main():
    out_dir = ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    pf = pd.read_csv(ROOT / PF_PATH, encoding="utf-8-sig")
    pf["hour"] = pd.to_datetime(pf["hour"])
    pf["date"] = pd.to_datetime(pf["date"]).dt.normalize()
    pf["hour_of_day"] = pf["hour"].dt.hour
    pf["dayofweek"] = pf["hour"].dt.dayofweek
    pf["day_type"] = np.where(pf["dayofweek"].isin([5, 6]), "weekend", "weekday")
    pf["period"] = pf["date"].map(period_of)
    aud = pd.read_csv(ROOT / STEP02, encoding="utf-8-sig")
    aud["date"] = pd.to_datetime(aud["date"]).dt.normalize()
    qmap = dict(zip(aud["date"], aud["quality_regime"]))
    pf["audit_quality_regime"] = pf["date"].map(qmap)

    # exclude right-censored 2018-01-24 23:00
    obs = pf[pf["hour"] != UNSCORABLE].copy()
    # pivot y by direction
    yp = obs.pivot_table(index=["date", "hour", "hour_of_day", "dayofweek", "day_type", "audit_quality_regime", "period"],
                         columns="task_key", values="y", aggfunc="first").reset_index()

    hourly_rows = []
    daily_rows = []
    for name, a, b, dab, dba in PAIRS:
        sub = yp[["date", "hour", "hour_of_day", "dayofweek", "day_type", "audit_quality_regime", "period", dab, dba]].copy()
        sub = sub.rename(columns={dab: "y_ab", dba: "y_ba"})
        sub["X"] = sub["y_ab"] + sub["y_ba"]
        sub["r"] = np.where(sub["X"] > 0, sub["y_ab"] / sub["X"], np.nan)
        sub["F"] = sub["y_ab"] - sub["y_ba"]
        # daily T per date
        T = sub.groupby("date")["X"].sum().rename("T_daily").reset_index()
        sub = sub.merge(T, on="date", how="left")
        sub["p"] = np.where(sub["T_daily"] > 0, sub["X"] / sub["T_daily"], 0.0)
        sub["pair"] = name
        hourly_rows.append(sub[HOURLY_COLS])
        # daily features
        for d, g in sub.groupby("date"):
            X = g["X"].to_numpy(float)
            Td = float(X.sum())
            mean_X = float(X.mean()); std_X = float(X.std(ddof=1)) if len(X) > 1 else np.nan
            cv = std_X / mean_X if mean_X != 0 else np.nan
            daily_rows.append({
                "date": d, "pair": name, "audit_quality_regime": g["audit_quality_regime"].iloc[0],
                "period": g["period"].iloc[0], "day_type": g["day_type"].iloc[0], "T": Td,
                "mean_X": mean_X, "std_X": std_X, "cv_X": cv, "zero_hour_count": int((X == 0).sum()),
                "peak_hour": int(g.loc[g["X"].idxmax(), "hour_of_day"]),
                "net_flow_daily": float(g["F"].sum()), "abs_net_flow": float(abs(g["F"].sum())),
                "r_daily": float(g["y_ab"].sum() / Td) if Td > 0 else np.nan,
            })
    hourly = pd.concat(hourly_rows, ignore_index=True)
    daily = pd.DataFrame(daily_rows)
    hourly["_p"] = hourly["pair"].map(PAIR_ORDER)
    hourly = hourly.sort_values(["_p", "date", "hour_of_day"]).drop(columns=["_p"]).reset_index(drop=True)
    daily["_p"] = daily["pair"].map(PAIR_ORDER)
    daily = daily.sort_values(["_p", "date"]).drop(columns=["_p"]).reset_index(drop=True)

    # --- zero-rate comparison ---
    zero_rows = []
    for name, a, b, dab, dba in PAIRS:
        s = hourly[hourly["pair"] == name]
        zr_ab = float((s["y_ab"] == 0).mean()); zr_ba = float((s["y_ba"] == 0).mean()); zr_X = float((s["X"] == 0).mean())
        zero_rows.append({"pair": name, "dir_ab": dab, "dir_ba": dba, "zero_rate_ab": zr_ab,
                          "zero_rate_ba": zr_ba, "zero_rate_X": zr_X,
                          "zero_rate_reduction_pp": (min(zr_ab, zr_ba) - zr_X) * 100})
    zero_cmp = pd.DataFrame(zero_rows)

    # --- stability summary (normal days) ---
    normal_dates = sorted(daily[daily["audit_quality_regime"] == "normal"]["date"].unique())
    stab_rows = []
    for name, a, b, dab, dba in PAIRS:
        d_d = daily[(daily["pair"] == name)]
        cvT = d_d["T"].std(ddof=1) / d_d["T"].mean() if d_d["T"].mean() != 0 else np.nan
        cvab = d_d["r_daily"].std(ddof=1)  # placeholder
        # daily cv of single-direction totals
        s_ab = hourly[hourly["pair"] == name].groupby("date")["y_ab"].sum()
        s_ba = hourly[hourly["pair"] == name].groupby("date")["y_ba"].sum()
        cv_ab = s_ab.std(ddof=1) / s_ab.mean() if s_ab.mean() != 0 else np.nan
        cv_ba = s_ba.std(ddof=1) / s_ba.mean() if s_ba.mean() != 0 else np.nan
        # hourshare pairwise Pearson on normal days
        M = np.zeros((len(normal_dates), 24))
        Mab = np.zeros((len(normal_dates), 24)); Mba = np.zeros((len(normal_dates), 24))
        for i, d in enumerate(normal_dates):
            g = hourly[(hourly["pair"] == name) & (hourly["date"] == d)].sort_values("hour_of_day")
            g = g.set_index("hour_of_day").reindex(range(24), fill_value=0.0)
            M[i] = g["p"].to_numpy()
            Td = g["T_daily"].iloc[0]
            Mab[i] = np.where(Td > 0, g["y_ab"].to_numpy() / Td, 0.0)
            Mba[i] = np.where(Td > 0, g["y_ba"].to_numpy() / Td, 0.0)
        pX = [safe_corr(M[i], M[j]) for i, j in itertools.combinations(range(len(normal_dates)), 2)]
        pab = [safe_corr(Mab[i], Mab[j]) for i, j in itertools.combinations(range(len(normal_dates)), 2)]
        pba = [safe_corr(Mba[i], Mba[j]) for i, j in itertools.combinations(range(len(normal_dates)), 2)]
        # direction ratio hour std (normal days, X>0 cells)
        rh = hourly[(hourly["pair"] == name) & (hourly["audit_quality_regime"] == "normal")].groupby("hour_of_day")["r"]
        dr_std = float(rh.std(ddof=1).mean())
        # global r
        tot = hourly[hourly["pair"] == name]
        r_glob = float(tot["y_ab"].sum() / tot["X"].sum()) if tot["X"].sum() > 0 else np.nan
        stab_rows.append({"pair": name, "daily_cv_T": cvT, "daily_cv_ab": cv_ab, "daily_cv_ba": cv_ba,
                          "hourshare_pearson_X_normal": float(np.nanmean(pX)),
                          "hourshare_pearson_ab_normal": float(np.nanmean(pab)),
                          "hourshare_pearson_ba_normal": float(np.nanmean(pba)),
                          "dirratio_hour_std_normal": dr_std, "dirratio_global": r_glob})
    stab = pd.DataFrame(stab_rows)

    # --- direction ratio profile (normal days, per hour) ---
    dr_rows = []
    for name, a, b, dab, dba in PAIRS:
        sn = hourly[(hourly["pair"] == name) & (hourly["audit_quality_regime"] == "normal")]
        for h in range(24):
            g = sn[sn["hour_of_day"] == h]
            r = g["r"].dropna()
            dr_rows.append({"pair": name, "hour": h, "n_days": int(len(g)),
                            "mean_r": float(r.mean()) if len(r) else np.nan, "std_r": float(r.std(ddof=1)) if len(r) > 1 else np.nan,
                            "q25_r": float(r.quantile(0.25)) if len(r) else np.nan, "q75_r": float(r.quantile(0.75)) if len(r) else np.nan,
                            "n_X_positive": int(len(r))})
    drratio = pd.DataFrame(dr_rows)

    # --- net flow (daily + cumulative) ---
    nf_rows = []
    for name, *_ in PAIRS:
        d_d = daily[daily["pair"] == name].sort_values("date")
        cum = 0.0
        for _, r in d_d.iterrows():
            cum += r["net_flow_daily"]
            nf_rows.append({"date": r["date"], "pair": name, "net_flow_daily": r["net_flow_daily"], "cum_net_flow": cum})
    netflow = pd.DataFrame(nf_rows)

    # --- weekday vs weekend (normal days) ---
    wke_rows = []
    for name, a, b, dab, dba in PAIRS:
        sn = hourly[(hourly["pair"] == name) & (hourly["audit_quality_regime"] == "normal")]
        wd = sn[sn["day_type"] == "weekday"]; we = sn[sn["day_type"] == "weekend"]
        wke_rows.append({
            "pair": name, "n_weekday": int(wd["date"].nunique()), "n_weekend": int(we["date"].nunique()),
            "weekday_mean_X": float(wd["X"].mean()), "weekend_mean_X": float(we["X"].mean()),
            "weekday_mean_r": float(wd["r"].mean()), "weekend_mean_r": float(we["r"].mean()),
            "weekday_mean_T": float(daily[(daily["pair"] == name) & (daily["audit_quality_regime"] == "normal") & (daily["day_type"] == "weekday")]["T"].mean()),
            "weekend_mean_T": float(daily[(daily["pair"] == name) & (daily["audit_quality_regime"] == "normal") & (daily["day_type"] == "weekend")]["T"].mean()),
        })
    wke = pd.DataFrame(wke_rows)

    # --- write ---
    p1 = out_dir / "pair_hourly_decomp.csv"; p2 = out_dir / "pair_daily_features.csv"
    p3 = out_dir / "pair_zero_rate_comparison.csv"; p4 = out_dir / "pair_stability_summary.csv"
    p5 = out_dir / "pair_direction_ratio_profile.csv"; p6 = out_dir / "pair_net_flow.csv"
    p7 = out_dir / "pair_weekday_weekend.csv"; p8 = out_dir / "summary.md"
    ho = hourly.copy(); ho["date"] = ho["date"].dt.strftime("%Y-%m-%d"); ho["hour"] = ho["hour"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ho.to_csv(p1, index=False, encoding="utf-8-sig")
    do = daily.copy(); do["date"] = do["date"].dt.strftime("%Y-%m-%d")
    do.to_csv(p2, index=False, encoding="utf-8-sig")
    zero_cmp.to_csv(p3, index=False, encoding="utf-8-sig")
    stab.to_csv(p4, index=False, encoding="utf-8-sig")
    drratio.to_csv(p5, index=False, encoding="utf-8-sig")
    no = netflow.copy(); no["date"] = no["date"].dt.strftime("%Y-%m-%d")
    no.to_csv(p6, index=False, encoding="utf-8-sig")
    wke.to_csv(p7, index=False, encoding="utf-8-sig")

    # assertions
    assert len(hourly) == 24 * 24 * 3 - 3, f"hourly rows {len(hourly)}"  # 1728 - 3 censored
    assert len(daily) == 24 * 3
    assert len(drratio) == 3 * 24
    assert len(netflow) == 24 * 3
    assert len(zero_cmp) == 3 and len(stab) == 3 and len(wke) == 3

    write_summary(p8, zero_cmp, stab, drratio, netflow, wke, daily)
    print("\n=== Output files ===")
    for p in [p1, p2, p3, p4, p5, p6, p7, p8]:
        rows = len(pd.read_csv(p, encoding="utf-8-sig")) if p.suffix == ".csv" else sum(1 for _ in open(p, encoding="utf-8"))
        print(f"  {p.relative_to(ROOT)}  rows={rows}")
    print("\nDone.")


def write_summary(path, zero_cmp, stab, drratio, netflow, wke, daily):
    L = []
    L.append("# Step 14 (B1) Three Layer-Pair Diagnostic")
    L.append("")
    L.append("> No models. Reconstructs six directed flows into three bidirectional layer-pairs and diagnoses sparsity/stability. 2018-01-24 23:00 right-censored (excluded).")
    L.append("")
    L.append(f"Run command: `{RUN_CMD}`")
    L.append("")

    L.append("## 1. Zero-rate reduction (merged vs single direction)")
    L.append("")
    L.append("| pair | zero ab | zero ba | zero X(merged) | reduction (pp) |")
    L.append("| --- | --- | --- | --- | --- |")
    for _, r in zero_cmp.iterrows():
        L.append(f"| {r['pair']} | {r['zero_rate_ab']:.3f} | {r['zero_rate_ba']:.3f} | {r['zero_rate_X']:.3f} | {r['zero_rate_reduction_pp']:.1f} |")
    L.append("")

    L.append("## 2. Daily-total stability (cv across dates)")
    L.append("")
    L.append("| pair | cv T (merged) | cv ab | cv ba |")
    L.append("| --- | --- | --- | --- |")
    for _, r in stab.iterrows():
        L.append(f"| {r['pair']} | {r['daily_cv_T']:.3f} | {r['daily_cv_ab']:.3f} | {r['daily_cv_ba']:.3f} |")
    L.append("")

    L.append("## 3. Hour-share stability (mean pairwise Pearson, normal 17 days)")
    L.append("")
    L.append("| pair | Pearson X(merged) | Pearson ab | Pearson ba |")
    L.append("| --- | --- | --- | --- |")
    for _, r in stab.iterrows():
        L.append(f"| {r['pair']} | {r['hourshare_pearson_X_normal']:.3f} | {r['hourshare_pearson_ab_normal']:.3f} | {r['hourshare_pearson_ba_normal']:.3f} |")
    L.append("")

    L.append("## 4. Direction ratio (a->b share of the pair)")
    L.append("")
    L.append("| pair | global r | mean hour std(r) |")
    L.append("| --- | --- | --- |")
    for _, r in stab.iterrows():
        L.append(f"| {r['pair']} | {r['dirratio_global']:.3f} | {r['dirratio_hour_std_normal']:.3f} |")
    L.append("- Per-hour mean r (normal days) is in pair_direction_ratio_profile.csv. A stable hour pattern would show small std and a consistent level across hours.")
    L.append("")

    L.append("## 5. Net flow (a->b minus b->a)")
    L.append("")
    L.append("| pair | final cumulative net flow | mean |daily net flow| |")
    L.append("| --- | --- | --- |")
    for name, *_ in PAIRS:
        nf = netflow[netflow["pair"] == name]
        L.append(f"| {name} | {nf['cum_net_flow'].iloc[-1]:.1f} | {nf['net_flow_daily'].abs().mean():.2f} |")
    L.append("")

    L.append("## 6. Weekday vs weekend (normal days)")
    L.append("")
    L.append("| pair | n wd/we | mean X wd | mean X we | mean r wd | mean r we |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for _, r in wke.iterrows():
        L.append(f"| {r['pair']} | {int(r['n_weekday'])}/{int(r['n_weekend'])} | {r['weekday_mean_X']:.2f} | {r['weekend_mean_X']:.2f} | {r['weekday_mean_r']:.3f} | {r['weekend_mean_r']:.3f} |")
    L.append("")

    L.append("## 7. Decision input")
    L.append("")
    L.append("- core_near and near_outer: pair structure is less sparse and more stable than the single directions -> prioritize the pair-decomposition model.")
    L.append("- core_outer: remains the sparsest pair (see zero_rate_X); decide between pair-decomposition and a sparse/hurdle model based on whether its merged stability is acceptable.")
    L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
