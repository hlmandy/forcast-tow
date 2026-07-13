"""Step 31 — A incremental stock oracle and emission stability audit.

Tests whether true future stock contains incremental information BEYOND the
Step 11 baseline, using: baseline + β(S_true - μ_S) instead of replacing
the baseline entirely. Also tests emission stability (long/recent/ewma) and
daily activity factor upper bounds.

Outputs (under outputs/step31_a_incremental_stock_oracle/):
  1-13 as specified.
"""

from __future__ import annotations
import sys, warnings, math
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from optimized_baseline import REGION_ORDER, add_regions, make_a_labels  # noqa: E402
from step03_a_rolling_backtest import classify_fold_quality, POLICY_WEIGHTS  # noqa: E402
from step06_a_structure_model_benchmark import compute_components  # noqa: E402

TRAIN_REL = Path("数据备份/训练集_20180101-0124_拖轮AIS.csv")
STEP01 = Path("outputs/step01_data_audit/train_daily_overview.csv")
STEP02 = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
VHS_PATH = Path("outputs/step08_b_state_transition_audit/b_vessel_hour_states.csv")
SS_PATH = Path("outputs/step08_b_state_transition_audit/b_hourly_source_state.csv")
OUT_REL = Path("outputs/step31_a_incremental_stock_oracle")
REGIONS = REGION_ORDER
RUN_CMD = "python " + " ".join(sys.argv)
UNSCORABLE = pd.Timestamp("2018-01-24 23:00:00")


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    # ---- implementation audit ----
    (out_dir / "implementation_audit.md").write_text(
        "# Step 30 Implementation Audit\n\n"
        "## Confirmed\n"
        "- Transition conservation: 1725/1725 PASS\n"
        "- Block-specific baselines: no leakage, post_outage=3872 exact\n"
        "- Per-block Oracle: no target date leakage\n\n"
        "## Not resolved\n"
        "- MMSI alignment: 0% due to hour_str format mismatch (needs pd.to_datetime unified)\n"
        "- Markov: transition normalization produces exploding values\n\n"
        "## What Step 30 did NOT test\n"
        "- Incremental stock: baseline + β(S_true - μ_S) was never tested\n"
        "- Emission stability: only long-term emission was used\n"
        "- Daily activity factor: not tested\n"
        "- Therefore Step 30 CANNOT close the stock route\n",
        encoding="utf-8")

    # ---- load ----
    df = pd.read_csv(ROOT / TRAIN_REL, usecols=["mmsi","x","y","sog","time"],
                     dtype={"mmsi":"string","x":"float64","y":"float64","sog":"float32"}, parse_dates=["time"])
    df["hour"] = df["time"].dt.floor("h"); df["date"] = df["time"].dt.normalize()
    df["hour_of_day"] = df["time"].dt.hour
    df = add_regions(df)
    aud = pd.read_csv(ROOT / STEP02); aud["date"] = pd.to_datetime(aud["date"]).dt.normalize()
    qmap = dict(zip(aud["date"], aud["quality_regime"]))
    vc_df = pd.read_csv(ROOT / STEP01); vc_df["date"] = pd.to_datetime(vc_df["date"]).dt.normalize()
    vc_map = dict(zip(vc_df["date"], vc_df["unique_vessel_count"]))
    china_map = dict(zip(aud["date"], aud["china_coastal_record_count"]))
    all_dates = sorted(pd.date_range("2018-01-01","2018-01-24",freq="D").normalize())

    # ---- A labels ----
    a_lab = make_a_labels(df)
    a_lab["date"] = a_lab["hour"].dt.normalize(); a_lab["hour_of_day"] = a_lab["hour"].dt.hour
    a_pivot = a_lab.pivot_table(index=["date","hour_of_day"], columns="region", values="y", aggfunc="first").reset_index()
    for r in REGIONS:
        if r not in a_pivot.columns: a_pivot[r] = 0

    def get_a_vec(d):
        sub = a_pivot[a_pivot["date"]==d].sort_values("hour_of_day")
        m = np.zeros((24, 3))
        for _, row in sub.iterrows():
            h = int(row["hour_of_day"])
            for ri, r in enumerate(REGIONS): m[h, ri] = float(row[r])
        return m

    # ---- stock from Step 08 source_state ----
    ss = pd.read_csv(ROOT / SS_PATH, encoding="utf-8-sig")
    ss["hour"] = pd.to_datetime(ss["hour"]); ss["date"] = pd.to_datetime(ss["date"]).dt.normalize()
    ss["hour_of_day"] = ss["hour"].dt.hour
    ss_pivot = ss.pivot_table(index=["date","hour_of_day"], columns="source_region", values="source_present_count", aggfunc="first").reset_index()
    for r in REGIONS:
        if r not in ss_pivot.columns: ss_pivot[r] = 0

    def get_stock_vec(d):
        sub = ss_pivot[ss_pivot["date"]==d].sort_values("hour_of_day")
        m = np.zeros((24, 3))
        for _, row in sub.iterrows():
            h = int(row["hour_of_day"])
            for ri, r in enumerate(REGIONS): m[h, ri] = float(row[r])
        return m

    # ========== 1. FIX MMSI ALIGNMENT ==========
    print("\n=== MMSI-level alignment (fixed) ===")
    # A-qualified vessel-hour-region
    active = df[df["region"].isin(REGIONS) & df["sog"].between(2, 10, inclusive="both")]
    src = active.groupby(["mmsi","hour","date","hour_of_day","region"]).size().rename("n").reset_index()
    a_qualified = src[src["n"] >= 3][["mmsi","hour","date","hour_of_day","region"]].copy()
    a_qualified["a_region"] = a_qualified["region"]
    a_qualified["mmsi_s"] = a_qualified["mmsi"].astype(str)
    a_qualified["date_s"] = a_qualified["date"].dt.strftime("%Y-%m-%d")
    a_qualified["hod"] = a_qualified["hour_of_day"].astype(int)

    # Step 08 vessel-hour states
    vhs = pd.read_csv(ROOT / VHS_PATH, encoding="utf-8-sig", dtype={"mmsi":"string"})
    vhs["mmsi_s"] = vhs["mmsi"].astype(str)
    vhs["date_s"] = pd.to_datetime(vhs["date"]).dt.strftime("%Y-%m-%d")
    vhs["hod"] = pd.to_datetime(vhs["hour"]).dt.hour.astype(int)
    vhs_rep = vhs[["mmsi_s","date_s","hod","representative_region"]].copy()

    # merge on string keys
    merged = a_qualified.merge(vhs_rep, on=["mmsi_s","date_s","hod"], how="left")
    # classify
    merged["alignment"] = "no_representative_state"
    mask_rep = merged["representative_region"].notna()
    merged.loc[mask_rep & (merged["a_region"] == merged["representative_region"]), "alignment"] = "same_region"
    merged.loc[mask_rep & (merged["a_region"] != merged["representative_region"]), "alignment"] = "different_region"
    # multi-A-region
    multi = merged.groupby(["mmsi_s","date_s","hod"]).size().rename("n_a_regions").reset_index()
    merged = merged.merge(multi, on=["mmsi_s","date_s","hod"], how="left")
    rep_in_a = merged.groupby(["mmsi_s","date_s","hod"]).apply(
        lambda g: g["representative_region"].iloc[0] in set(g["a_region"]) if g["representative_region"].notna().all() and len(g["representative_region"].dropna()) > 0 else False
    ).rename("rep_in_a_set").reset_index()
    merged = merged.merge(rep_in_a, on=["mmsi_s","date_s","hod"], how="left")
    merged.loc[(merged["n_a_regions"] > 1) & merged["rep_in_a_set"], "alignment"] = "multi_a_region_same_rep"
    merged.loc[(merged["n_a_regions"] > 1) & ~merged["rep_in_a_set"] & mask_rep, "alignment"] = "multi_a_region_different_rep"

    total_a = len(merged)
    align_counts = merged["alignment"].value_counts()
    same = int(align_counts.get("same_region", 0))
    multi_same = int(align_counts.get("multi_a_region_same_rep", 0))
    align_rate = same / total_a if total_a else 0
    match_any_rate = (same + multi_same) / total_a if total_a else 0
    print(f"Total A-qualified vhr: {total_a}")
    print(f"Same region (row-level): {same} ({align_rate:.4f})")
    print(f"Match-any (incl multi_same): {same + multi_same} ({match_any_rate:.4f})")
    print(f"Multi A region total: {int(merged['n_a_regions'].gt(1).sum())}")
    for ac, cnt in align_counts.items():
        print(f"  {ac}: {cnt}")

    merged["quality"] = merged["date"].map(qmap)
    for q in ["normal", "outage", "severe", "reduced"]:
        sub = merged[merged["quality"] == q]
        if len(sub):
            sr = sub["alignment"].eq("same_region").mean()
            ma = sub["alignment"].isin(["same_region","multi_a_region_same_rep"]).mean()
            print(f"  {q}: row-level={sr:.4f}, match-any={ma:.4f} ({len(sub)})")

    merged.to_csv(out_dir / "exact_alignment_rows.csv", index=False, encoding="utf-8-sig")
    align_summary = align_counts.reset_index(); align_summary.columns = ["alignment","count"]
    align_summary["pct"] = align_summary["count"] / total_a * 100
    align_summary.to_csv(out_dir / "alignment_summary.csv", index=False, encoding="utf-8-sig")

    if align_rate < 0.01 and match_any_rate < 0.01:
        print("ERROR: Alignment still 0%. Aborting Oracle.")
        return

    # ========== 2. BLOCK-SPECIFIC BASELINE ==========
    print("\n=== Block-specific baselines ===")
    blocks = {
        "block_pre_normal": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-07"),
                             pd.Timestamp("2018-01-08"), pd.Timestamp("2018-01-11")),
        "block_target_like": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-19"),
                              pd.Timestamp("2018-01-20"), pd.Timestamp("2018-01-24")),
        "block_post_outage_stress": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-18"),
                                     pd.Timestamp("2018-01-19"), pd.Timestamp("2018-01-24")),
    }

    def compute_baseline(train_dates_blk, te, valid_dates_blk):
        classes_blk, _ = classify_fold_quality(train_dates_blk, china_map)
        weights_blk = {d: POLICY_WEIGHTS["exclude_outage_severe"][classes_blk[d]] for d in train_dates_blk}
        train_df = a_lab[a_lab["date"].isin(train_dates_blk)].copy()
        train_df["day_type"] = np.where(train_df["hour"].dt.dayofweek.isin([5, 6]), "weekend", "weekday")
        T_hat, P_hat = compute_components(train_df, train_dates_blk, te, weights_blk)
        # predict valid
        preds = {}
        for d in valid_dates_blk:
            vd = a_lab[a_lab["date"] == d].sort_values(["hour","region"]).reset_index(drop=True)
            vr = vd["region"].to_numpy(); vh = vd["hour_of_day"].to_numpy().astype(int)
            vdt = vd["hour"].dt.dayofweek.to_numpy()
            vdt_type = np.array(["weekend" if d_ in (5,6) else "weekday" for d_ in vdt])
            Tvec = np.array([T_hat["mean"][r] for r in vr], dtype=float)
            prof = np.array([P_hat["daytype_shrunk"][r][vdt_type[i]][vh[i]] for i, r in enumerate(vr)])
            pf = Tvec * prof
            pr = np.clip(np.rint(pf), 0, None).astype(int)
            # store as (24,3)
            mat = np.zeros((24, 3))
            for i, r in enumerate(vr):
                mat[vh[i], REGIONS.index(r)] = float(pr[i])
            preds[d] = mat
        return preds

    block_baselines = {}
    bl_sse = {}
    for bname, (ts, te, vs_, ve) in blocks.items():
        td = sorted(pd.date_range(ts, te, freq="D").normalize())
        vd = list(pd.date_range(vs_, ve, freq="D").normalize())
        preds = compute_baseline(td, te, vd)
        block_baselines[bname] = preds
        sse = sum(float(((preds[d] - get_a_vec(d))**2).sum()) for d in vd)
        bl_sse[bname] = sse
        print(f"  {bname}: baseline SSE={sse:.0f}")

    # ========== 3. CROSS-FITTED TRAINING RESIDUALS ==========
    print("\n=== Cross-fitted training residuals ===")
    cf_rows = []
    for bname, (ts, te, vs_, ve) in blocks.items():
        train_dates_blk = sorted(pd.date_range(ts, te, freq="D").normalize())
        train_normal = [d for d in train_dates_blk if qmap.get(d) == "normal"]
        # rolling-origin: for each normal date d (from index 6+), predict using data before d
        for d in train_normal[6:]:
            prev_dates = [dd for dd in train_dates_blk if dd < d]
            prev_normal = [dd for dd in prev_dates if qmap.get(dd) == "normal"]
            if len(prev_normal) < 3: continue
            # baseline for d
            bl_pred = compute_baseline(prev_dates, d - pd.Timedelta(days=1), [d])[d]  # (24,3)
            # stock mean from prev_normal
            s_mean = np.zeros((24, 3))
            for dd in prev_normal:
                s_mean += get_stock_vec(dd)
            s_mean /= len(prev_normal)
            # actual
            a_vec = get_a_vec(d); s_vec = get_stock_vec(d)
            resid = a_vec - bl_pred  # (24,3)
            stock_dev = s_vec - s_mean  # (24,3)
            for h in range(24):
                for ri in range(3):
                    cf_rows.append({"block": bname, "date": f"{d:%Y-%m-%d}", "hour": h, "region": REGIONS[ri],
                                    "residual": float(resid[h, ri]), "stock_dev": float(stock_dev[h, ri]),
                                    "a_true": float(a_vec[h, ri]), "bl_pred": float(bl_pred[h, ri]),
                                    "stock_true": float(s_vec[h, ri]), "stock_mean": float(s_mean[h, ri])})
    cf = pd.DataFrame(cf_rows)
    cf.to_csv(out_dir / "crossfitted_training_residuals.csv", index=False, encoding="utf-8-sig")
    print(f"Cross-fitted samples: {len(cf)}")

    # ========== 4. EMISSION ESTIMATES ==========
    def compute_emission(train_normal_blk):
        a_sum = np.zeros((24,3)); s_sum = np.zeros((24,3))
        for d in train_normal_blk:
            a_sum += get_a_vec(d); s_sum += get_stock_vec(d)
        return np.where(s_sum > 0, a_sum / s_sum, 0)

    # ========== 5. ORACLE PREDICTIONS ==========
    print("\n=== Oracle predictions ===")
    methods = ["exact_step11_baseline", "oracle_abs_long", "oracle_abs_recent3", "oracle_abs_recent7",
               "oracle_inc_common", "oracle_inc_region", "oracle_inc_full_ridge",
               "oracle_true_stock_true_daily_factor"]

    # Fit incremental models on cross-fitted residuals
    # common: single beta
    all_resid = cf["residual"].to_numpy(float)
    all_dev = cf["stock_dev"].to_numpy(float)
    beta_common = np.cov(all_dev, all_resid)[0,1] / np.var(all_dev) if np.var(all_dev) > 0 else 0
    beta_common = np.clip(beta_common, -2, 2)

    # region-specific
    beta_region = {}
    for ri, r in enumerate(REGIONS):
        sub = cf[cf["region"] == r]
        x = sub["stock_dev"].to_numpy(float); y = sub["residual"].to_numpy(float)
        if np.var(x) > 0:
            b = np.cov(x, y)[0,1] / np.var(x)
        else:
            b = 0
        beta_region[r] = np.clip(b, -2, 2)

    # full ridge (3x3) - aggregate duplicates first
    cf_agg = cf.groupby(["date","hour","region"])[["residual","stock_dev"]].mean().reset_index()
    cf_pivot = cf_agg.pivot_table(index=["date","hour"], columns="region", values=["stock_dev","residual"])
    X_cf = cf_pivot["stock_dev"][REGIONS].to_numpy()
    Y_cf = cf_pivot["residual"][REGIONS].to_numpy()
    if len(X_cf) > 0:
        lam = 10  # fixed ridge
        XtX = X_cf.T @ X_cf + lam * np.eye(3)
        XtY = X_cf.T @ Y_cf
        B_full = np.linalg.solve(XtX, XtY)
    else:
        B_full = np.zeros((3, 3))

    print(f"  beta_common={beta_common:.4f}")
    print(f"  beta_region={{{', '.join(f'{r}:{beta_region[r]:.4f}' for r in REGIONS)}}}")
    print(f"  B_full diag={np.diag(B_full)}")

    # daily factors
    df_rows = []
    for bname, (ts, te, vs_, ve) in blocks.items():
        train_normal_blk = [d for d in sorted(pd.date_range(ts, te, freq="D").normalize()) if qmap.get(d) == "normal"]
        em_long = compute_emission(train_normal_blk)
        for d in train_normal_blk:
            av = get_a_vec(d); sv = get_stock_vec(d)
            pred_em = sv * em_long
            total_pred = pred_em.sum()
            total_true = av.sum()
            f = total_true / total_pred if total_pred > 0 else 1.0
            df_rows.append({"block": bname, "date": f"{d:%Y-%m-%d}", "daily_factor": f, "quality": "normal"})
    df_df = pd.DataFrame(df_rows)
    df_df.to_csv(out_dir / "emission_daily_factors.csv", index=False, encoding="utf-8-sig")

    # ========== 6. SCORE ALL METHODS ==========
    print("\n=== Scoring ===")
    bs_rows = []
    for bname, (ts, te, vs_, ve) in blocks.items():
        train_normal_blk = [d for d in sorted(pd.date_range(ts, te, freq="D").normalize()) if qmap.get(d) == "normal"]
        valid_dates_blk = list(pd.date_range(vs_, ve, freq="D").normalize())
        bl = block_baselines[bname]; bl_s = bl_sse[bname]
        em_long = compute_emission(train_normal_blk)
        em_rec3 = compute_emission(train_normal_blk[-3:]) if len(train_normal_blk) >= 3 else em_long
        em_rec7 = compute_emission(train_normal_blk[-7:]) if len(train_normal_blk) >= 7 else em_long
        # stock mean
        s_mean_blk = np.zeros((24, 3))
        for d in train_normal_blk: s_mean_blk += get_stock_vec(d)
        s_mean_blk /= max(1, len(train_normal_blk))

        for method in methods:
            total_sse = 0
            for d in valid_dates_blk:
                tv = get_a_vec(d); sv = get_stock_vec(d)
                if method == "exact_step11_baseline":
                    pred = bl[d]
                elif method == "oracle_abs_long":
                    pred = np.clip(np.round(sv * em_long), 0, None)
                elif method == "oracle_abs_recent3":
                    pred = np.clip(np.round(sv * em_rec3), 0, None)
                elif method == "oracle_abs_recent7":
                    pred = np.clip(np.round(sv * em_rec7), 0, None)
                elif method == "oracle_inc_common":
                    pred = bl[d] + beta_common * (sv - s_mean_blk)
                    pred = np.clip(np.round(pred), 0, None)
                elif method == "oracle_inc_region":
                    pred = bl[d].copy()
                    for ri, r in enumerate(REGIONS):
                        pred[:, ri] += beta_region[r] * (sv[:, ri] - s_mean_blk[:, ri])
                    pred = np.clip(np.round(pred), 0, None)
                elif method == "oracle_inc_full_ridge":
                    dev = sv - s_mean_blk  # (24,3)
                    pred = bl[d] + (dev @ B_full.T)
                    pred = np.clip(np.round(pred), 0, None)
                elif method == "oracle_true_stock_true_daily_factor":
                    # true daily factor
                    pred_em = sv * em_long
                    total_pred = pred_em.sum(); total_true = tv.sum()
                    f = total_true / total_pred if total_pred > 0 else 1
                    pred = np.clip(np.round(sv * em_long * f), 0, None)
                else:
                    pred = bl[d]
                total_sse += float(((pred - tv)**2).sum())
            red = 1 - total_sse / bl_s if bl_s else 0
            bs_rows.append({"block": bname, "method": method, "deployable": not method.startswith("oracle_true"),
                           "sse": total_sse, "baseline_sse": bl_s, "reduction_ratio": red})
            print(f"  {bname:30s} {method:42s} SSE={total_sse:.0f} red={red*100:.1f}%")

    block_scores = pd.DataFrame(bs_rows)
    block_scores.to_csv(out_dir / "block_scores.csv", index=False, encoding="utf-8-sig")

    # ========== 7. NORMAL ONE-DAY ORIGINS ==========
    print("\n=== Normal one-day origins ===")
    normal_dates = [d for d in all_dates if qmap.get(d) == "normal"]
    one_day_rows = []
    for d in normal_dates:
        prev_dates = [dd for dd in all_dates if dd < d]
        prev_normal = [dd for dd in prev_dates if qmap.get(dd) == "normal"]
        if len(prev_normal) < 6: continue
        bl_pred = compute_baseline(prev_dates, d - pd.Timedelta(days=1), [d])[d]
        tv = get_a_vec(d); sv = get_stock_vec(d)
        s_mean = np.zeros((24,3))
        for dd in prev_normal: s_mean += get_stock_vec(dd)
        s_mean /= len(prev_normal)
        em = compute_emission(prev_normal)
        bl_sse = float(((bl_pred - tv)**2).sum())
        inc_sse = float(((np.clip(np.round(bl_pred + beta_common * (sv - s_mean)), 0, None) - tv)**2).sum())
        abs_sse = float(((np.clip(np.round(sv * em), 0, None) - tv)**2).sum())
        one_day_rows.append({"date": f"{d:%Y-%m-%d}", "baseline_sse": bl_sse,
                             "incremental_sse": inc_sse, "absolute_sse": abs_sse,
                             "inc_improved": inc_sse < bl_sse, "abs_improved": abs_sse < bl_sse})
    od = pd.DataFrame(one_day_rows)
    od.to_csv(out_dir / "normal_one_day_scores.csv", index=False, encoding="utf-8-sig")
    n_inc_improve = int(od["inc_improved"].sum()) if len(od) else 0
    n_total = len(od)
    print(f"One-day: {n_inc_improve}/{n_total} improved by incremental")

    # ========== 8. ORACLE GAP ==========
    og_rows = []
    for bname in blocks:
        sub = block_scores[block_scores["block"] == bname]
        bl_r = sub[sub["method"]=="exact_step11_baseline"].iloc[0]
        for _, r in sub.iterrows():
            og_rows.append({"block": bname, "method": r["method"], "sse": r["sse"],
                           "baseline_sse": bl_r["sse"], "reduction_pct": r["reduction_ratio"]*100})
    pd.DataFrame(og_rows).to_csv(out_dir / "oracle_gap.csv", index=False, encoding="utf-8-sig")

    # ========== 9. SUMMARY ==========
    tl_inc = block_scores[(block_scores["block"]=="block_target_like") & (block_scores["method"]=="oracle_inc_common")]
    tl_red = float(tl_inc["reduction_ratio"].iloc[0]) if len(tl_inc) else 0
    pn_inc = block_scores[(block_scores["block"]=="block_pre_normal") & (block_scores["method"]=="oracle_inc_common")]
    pn_red = float(pn_inc["reduction_ratio"].iloc[0]) if len(pn_inc) else 0
    tl_df = block_scores[(block_scores["block"]=="block_target_like") & (block_scores["method"]=="oracle_true_stock_true_daily_factor")]
    tl_df_red = float(tl_df["reduction_ratio"].iloc[0]) if len(tl_df) else 0

    L = ["# Step 31 Incremental Stock Oracle", "", f"Run command: `{RUN_CMD}`", ""]
    L.append("## MMSI alignment (fixed)")
    L.append(f"- Row-level same_region: {align_rate:.4f}")
    L.append(f"- Match-any: {match_any_rate:.4f}")
    L.append("")
    L.append("## Block scores (key methods)")
    L.append("")
    L.append("| block | method | SSE | reduction |")
    L.append("| --- | --- | --- | --- |")
    for _, r in block_scores[block_scores["method"].isin(["exact_step11_baseline","oracle_abs_long","oracle_inc_common","oracle_inc_region","oracle_true_stock_true_daily_factor"])].iterrows():
        L.append(f"| {r['block']} | {r['method']} | {r['sse']:.0f} | {r['reduction_ratio']*100:.1f}% |")
    L.append("")
    L.append(f"## Decision")
    L.append(f"- Incremental common on target_like: {tl_red*100:.1f}%")
    L.append(f"- Incremental common on pre_normal: {pn_red*100:.1f}%")
    L.append(f"- True daily factor on target_like: {tl_df_red*100:.1f}%")
    L.append(f"- One-day origins improved: {n_inc_improve}/{n_total}")
    passes = tl_red >= 0.05 and pn_red >= 0 and n_inc_improve >= 0.6 * n_total if n_total else False
    L.append(f"- Passes stock prediction criteria: {'YES' if passes else 'NO'}")
    L.append("")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    assert not any(out_dir.rglob("*.zip"))
    print(f"\n=== Summary ===")
    print(f"Alignment: row={align_rate:.4f}, match-any={match_any_rate:.4f}")
    print(f"Incremental common: target_like={tl_red*100:.1f}%, pre_normal={pn_red*100:.1f}%")
    print(f"True daily factor: target_like={tl_df_red*100:.1f}%")
    print(f"One-day improved: {n_inc_improve}/{n_total}")
    print(f"Passes criteria: {passes}")
    print("\nDone.")


if __name__ == "__main__":
    main()
