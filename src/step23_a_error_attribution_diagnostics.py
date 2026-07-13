"""Step 23 — A error attribution diagnostics.

Decomposes A SSE=4423 into sources (daily total, region allocation, hour shape,
region-hour local bias) using oracle counterfactuals. Pure diagnostic: no
models, no submissions, no validation labels used for prediction.

Outputs (under outputs/step23_a_error_attribution_diagnostics/):
  1. a_baseline_predictions.csv
  2. a_daily_error_decomposition.csv
  3. a_oracle_scores.csv
  4. a_region_scores.csv
  5. a_region_hour_residuals.csv
  6. a_horizon_scores.csv
  7. a_nonoverlap_scores.csv
  8. signal_stability.csv
  9. modeling_decision.csv
  10. summary.md
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
STEP07 = Path("outputs/step07_a_regularized_count_models/a_model_predictions.csv")
STEP01 = Path("outputs/step01_data_audit/train_daily_overview.csv")
STEP02 = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
STEP21 = Path("outputs/step21_joint_backtest_framework/joint_fold_scores.csv")
OUT_REL = Path("outputs/step23_a_error_attribution_diagnostics")

REGIONS = ["core", "near", "outer"]
SCENARIOS = [f"fold_{i:02d}" for i in range(1, 9)] + ["final_analog"]
ROLLING = [f"fold_{i:02d}" for i in range(1, 9)]
RUN_CMD = "python " + " ".join(sys.argv)


def lr_alloc(target, props):
    """Largest remainder: distribute integer `target` across cells by `props`."""
    if target <= 0: return np.zeros(len(props), dtype=int)
    fp = np.array(props, dtype=float)
    s = fp.sum()
    if s <= 0: return np.zeros(len(props), dtype=int)
    fp = fp / s * target
    base = np.floor(fp).astype(int)
    rem = int(target) - int(base.sum())
    if rem > 0:
        frac = fp - base
        order = np.argsort(-frac)
        for j in range(min(rem, len(order))): base[order[j]] += 1
    elif rem < 0:
        frac = fp - base
        order = np.argsort(frac)
        for j in range(min(-rem, len(order))):
            if base[order[j]] > 0: base[order[j]] -= 1
    return np.clip(base, 0, None)


def sse_int(pred, true): return float(((pred - true) ** 2).sum())


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    # ---- load baseline A predictions ----
    a7 = pd.read_csv(ROOT / STEP07, encoding="utf-8-sig")
    a7["date"] = pd.to_datetime(a7["date"]).dt.normalize()
    bp = a7[(a7["method"] == "decomp_mean_daytype_shrunk") & (a7["training_policy"] == "exclude_outage_severe")].copy()
    bp["forecast_horizon"] = bp.groupby("evaluation_id")["date"].rank(method="dense").astype(int)

    daily = pd.read_csv(ROOT / STEP01); daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    aud = pd.read_csv(ROOT / STEP02); aud["date"] = pd.to_datetime(aud["date"]).dt.normalize()
    qmap = dict(zip(aud["date"], aud["quality_regime"]))
    vc_map = dict(zip(daily["date"], daily["unique_vessel_count"]))

    # ---- verify step21 reproduction ----
    s21 = pd.read_csv(ROOT / STEP21, encoding="utf-8-sig")
    for _, r in s21.iterrows():
        eid = r["evaluation_id"]
        sc = bp[bp["evaluation_id"] == eid]
        er = (sc["pred_rounded"] - sc["y_true"]).astype(float).to_numpy()
        my_sse = float((er ** 2).sum())
        assert abs(my_sse - r["sse_a"]) < 1e-6, f"SSE mismatch {eid}: {my_sse} vs {r['sse_a']}"
    print("Step 21 A baseline reproduction PASS.")

    # ---- pivot per (eval, date): 24h x 3 region matrices ----
    # Build per-date oracle predictions
    oracle_methods = ["baseline", "oracle_daily_total", "oracle_region_totals", "oracle_hour_profile", "oracle_hour_totals"]
    all_oracle_preds = {}  # (eval, method) -> DataFrame with pred_rounded per row

    for eid in SCENARIOS:
        bp_e = bp[bp["evaluation_id"] == eid].copy()
        # pivot to (date, hour, region) matrix
        for method in oracle_methods:
            df_m = bp_e.copy()
            if method == "baseline":
                pass  # keep pred_rounded
            else:
                for d in bp_e["date"].unique():
                    mask = df_m["date"] == d
                    sub = df_m[mask].sort_values(["hour_of_day", "region"])
                    hrs = sub["hour_of_day"].to_numpy().astype(int)
                    regs = sub["region"].to_numpy()
                    n = len(sub)  # 72
                    y = sub["y_true"].to_numpy().astype(float)
                    p = sub["pred_rounded"].to_numpy().astype(float)

                    if method == "oracle_daily_total":
                        H_d = int(y.sum())
                        props = p / p.sum() if p.sum() > 0 else np.ones(n) / n
                        new_pred = lr_alloc(H_d, props)

                    elif method == "oracle_region_totals":
                        new_pred = np.zeros(n, dtype=int)
                        for z_idx, z in enumerate(REGIONS):
                            zmask = regs == z
                            Hz = int(y[zmask].sum())
                            pz = p[zmask]
                            props = pz / pz.sum() if pz.sum() > 0 else np.ones(zmask.sum()) / zmask.sum()
                            new_pred[zmask] = lr_alloc(Hz, props)

                    elif method == "oracle_hour_profile":
                        new_pred = np.zeros(n, dtype=int)
                        for z_idx, z in enumerate(REGIONS):
                            zmask = regs == z
                            Hz_hat = int(p[zmask].sum())  # baseline region total
                            yz = y[zmask]
                            yz_sum = yz.sum()
                            if yz_sum > 0:
                                props = yz / yz_sum
                            else:
                                pz = p[zmask]; props = pz / pz.sum() if pz.sum() > 0 else np.ones(zmask.sum()) / zmask.sum()
                            new_pred[zmask] = lr_alloc(Hz_hat, props)

                    elif method == "oracle_hour_totals":
                        new_pred = np.zeros(n, dtype=int)
                        for h in range(24):
                            hmask = hrs == h
                            Gh = int(y[hmask].sum())
                            ph = p[hmask]
                            props = ph / ph.sum() if ph.sum() > 0 else np.ones(hmask.sum()) / hmask.sum()
                            new_pred[hmask] = lr_alloc(Gh, props)

                    df_m.loc[mask, "pred_rounded"] = new_pred
            df_m["oracle_method"] = method
            all_oracle_preds[(eid, method)] = df_m

    # ---- oracle scores ----
    oracle_score_rows = []
    region_score_rows = []
    horizon_rows = []

    for eid in SCENARIOS:
        etype = "rolling_7d" if eid in ROLLING else "final_analog"
        # per-method per-view scores
        for method in oracle_methods:
            df_m = all_oracle_preds[(eid, method)]
            y = df_m["y_true"].to_numpy(float); p = df_m["pred_rounded"].to_numpy(float)
            pf = df_m["pred_float"].to_numpy(float)
            sse_i = float(((p - y) ** 2).sum()); sse_f = float(((pf - y) ** 2).sum())
            bl = all_oracle_preds[(eid, "baseline")]
            bl_sse = float(((bl["pred_rounded"] - bl["y_true"]) ** 2).sum())
            oracle_score_rows.append({"evaluation_id": eid, "evaluation_type": etype, "method": method,
                                      "scope": "overall", "region": "all", "n_rows": int(len(df_m)),
                                      "sse_float": sse_f, "sse_integer": sse_i, "baseline_sse": bl_sse,
                                      "sse_reduction": bl_sse - sse_i, "reduction_ratio": (bl_sse - sse_i) / bl_sse if bl_sse else 0.0})
            # per-region
            for r in REGIONS:
                rm = df_m[df_m["region"] == r]
                rm_bl = bl[bl["region"] == r]
                sse_r = float(((rm["pred_rounded"] - rm["y_true"]) ** 2).sum())
                bl_r = float(((rm_bl["pred_rounded"] - rm_bl["y_true"]) ** 2).sum())
                region_score_rows.append({"evaluation_id": eid, "evaluation_type": etype, "method": method,
                                          "region": r, "sse_integer": sse_r, "baseline_sse": bl_r,
                                          "reduction": bl_r - sse_r, "reduction_ratio": (bl_r - sse_r) / bl_r if bl_r else 0.0})
        # horizon
        for hz in sorted(bp[bp["evaluation_id"] == eid]["forecast_horizon"].unique()):
            for method in oracle_methods:
                df_m = all_oracle_preds[(eid, method)]
                sh = df_m[df_m["forecast_horizon"] == hz]
                sse_i = float(((sh["pred_rounded"] - sh["y_true"]) ** 2).sum())
                horizon_rows.append({"evaluation_id": eid, "evaluation_type": etype, "method": method,
                                     "forecast_horizon": int(hz), "sse_integer": sse_i, "n_rows": int(len(sh))})

    oracle_scores = pd.DataFrame(oracle_score_rows)
    region_scores = pd.DataFrame(region_score_rows)
    horizons = pd.DataFrame(horizon_rows)

    # ---- non-overlap scores ----
    no_rows = []
    for block, eids in [("block_1", ["fold_01"]), ("block_2", ["fold_08"]), ("combined", ["fold_01", "fold_08"])]:
        for method in oracle_methods:
            frames = [all_oracle_preds[(e, method)] for e in eids]
            sub = pd.concat(frames, ignore_index=True)
            sse_i = float(((sub["pred_rounded"] - sub["y_true"]) ** 2).sum())
            bl_sub = pd.concat([all_oracle_preds[(e, "baseline")] for e in eids], ignore_index=True)
            bl_sse = float(((bl_sub["pred_rounded"] - bl_sub["y_true"]) ** 2).sum())
            no_rows.append({"block": block, "method": method, "sse_integer": sse_i, "baseline_sse": bl_sse,
                            "reduction": bl_sse - sse_i, "reduction_ratio": (bl_sse - sse_i) / bl_sse if bl_sse else 0.0})
    nonoverlap = pd.DataFrame(no_rows)

    # ---- daily error decomposition ----
    daily_rows = []
    bl_all = all_oracle_preds[("fold_01", "baseline")]  # placeholder, will iterate per eid
    for eid in SCENARIOS:
        bl = all_oracle_preds[(eid, "baseline")]
        for d in bl["date"].unique():
            sub = bl[bl["date"] == d]
            y = sub["y_true"].to_numpy(float); p = sub["pred_rounded"].to_numpy(float)
            H_d = float(y.sum()); Hp = float(p.sum())
            sse_d = float(((p - y) ** 2).sum())
            day_q = qmap.get(d, "")
            vc = vc_map.get(d, np.nan)
            daily_rows.append({"evaluation_id": eid, "date": f"{d:%Y-%m-%d}", "H_d_true": H_d, "H_d_pred": Hp,
                               "daily_total_error": Hp - H_d, "daily_sse": sse_d, "U_d": vc,
                               "is_normal": day_q == "normal", "quality": day_q,
                               "forecast_horizon": int(sub["forecast_horizon"].iloc[0]),
                               **{f"H_{r}_true": float(sub[sub["region"]==r]["y_true"].sum()) for r in REGIONS},
                               **{f"H_{r}_pred": float(sub[sub["region"]==r]["pred_rounded"].sum()) for r in REGIONS}})
    daily_decomp = pd.DataFrame(daily_rows)

    # ---- region-hour residuals + ±1 deltas ----
    rh_rows = []
    for eid in SCENARIOS:
        bl = all_oracle_preds[(eid, "baseline")]
        for r in REGIONS:
            for h in range(24):
                sub = bl[(bl["region"] == r) & (bl["hour_of_day"] == h)]
                if len(sub) == 0: continue
                y = sub["y_true"].to_numpy(float); p = sub["pred_rounded"].to_numpy(float)
                resid = y - p
                bl_sse = float((resid ** 2).sum())
                p1_sse = float(((y - (p + 1)) ** 2).sum())
                m1_sse = float(((y - (p - 1)) ** 2).sum())
                rh_rows.append({"evaluation_id": eid, "region": r, "hour_of_day": h, "n_days": int(len(sub)),
                                "mean_residual": float(resid.mean()), "median_residual": float(np.median(resid)),
                                "residual_std": float(resid.std(ddof=1)) if len(resid) > 1 else np.nan,
                                "positive_count": int((resid > 0).sum()), "negative_count": int((resid < 0).sum()),
                                "zero_count": int((resid == 0).sum()), "baseline_sse": bl_sse,
                                "plus_one_sse": p1_sse, "minus_one_sse": m1_sse,
                                "plus_one_delta_sse": p1_sse - bl_sse, "minus_one_delta_sse": m1_sse - bl_sse})
    rh_resid = pd.DataFrame(rh_rows)

    # ---- signal stability ----
    sig_rows = []
    for r in REGIONS:
        for h in range(24):
            for action, col in [("plus_one", "plus_one_delta_sse"), ("minus_one", "minus_one_delta_sse")]:
                fold_deltas = {}; no1 = no2 = fa_d = np.nan; norm_d = anom_d = np.nan
                improve = worsen = tie = 0; total_delta = 0; deltas_list = []
                for eid in ROLLING:
                    row = rh_resid[(rh_resid["evaluation_id"] == eid) & (rh_resid["region"] == r) & (rh_resid["hour_of_day"] == h)]
                    if len(row) == 0: continue
                    d = float(row[col].iloc[0]); fold_deltas[eid] = d; deltas_list.append(d); total_delta += d
                    if d < 0: improve += 1
                    elif d > 0: worsen += 1
                    else: tie += 1
                # nonoverlap
                if "fold_01" in fold_deltas: no1 = fold_deltas["fold_01"]
                if "fold_08" in fold_deltas: no2 = fold_deltas["fold_08"]
                fa_row = rh_resid[(rh_resid["evaluation_id"] == "final_analog") & (rh_resid["region"] == r) & (rh_resid["hour_of_day"] == h)]
                if len(fa_row): fa_d = float(fa_row[col].iloc[0])
                # signal strength
                med = float(np.median(deltas_list)) if deltas_list else np.nan
                no_c = (no1 or 0) + (no2 or 0)
                strong = improve >= 6 and (no1 is not np.nan and no1 <= 0) and (no2 is not np.nan and no2 <= 0) and (fa_d is not np.nan and fa_d <= 0) and total_delta < 0 and med < 0
                moderate = (not strong) and improve >= 5 and no_c <= 0 and (fa_d is not np.nan and fa_d <= 0) and total_delta < 0
                if total_delta >= 0 and worsen >= 5: strength = "harmful"
                elif strong: strength = "strong"
                elif moderate: strength = "moderate"
                else: strength = "weak"
                sig_rows.append({"region": r, "hour_of_day": h, "action": action,
                                 "rolling_improve_count": improve, "rolling_worsen_count": worsen, "rolling_tie_count": tie,
                                 "rolling_total_delta_sse": total_delta, "rolling_median_delta_sse": med,
                                 "nonoverlap_block_1_delta": no1, "nonoverlap_block_2_delta": no2, "final_analog_delta": fa_d,
                                 "signal_strength": strength})
    signal = pd.DataFrame(sig_rows)

    # ---- daily total predictability (simple) ----
    # residuals R_d = H_d - H_hat_d
    # correlate with U_d, log(U_d), dayofweek, trend, recent H
    normal_dr = daily_decomp[daily_decomp["is_normal"]]
    if len(normal_dr) > 3:
        Rd = normal_dr["daily_total_error"].to_numpy(float)
        Ud = normal_dr["U_d"].to_numpy(float)
        dt_pred_corr = float(pd.Series(Rd).corr(pd.Series(Ud)))
    else:
        dt_pred_corr = np.nan

    # ---- modeling decision ----
    def oracle_reduction(method, view="rolling"):
        if view == "rolling":
            sub = oracle_scores[(oracle_scores["method"] == method) & (oracle_scores["evaluation_type"] == "rolling_7d") & (oracle_scores["scope"] == "overall")]
            return float(sub["reduction_ratio"].mean()) if len(sub) else np.nan
        elif view == "final":
            sub = oracle_scores[(oracle_scores["method"] == method) & (oracle_scores["evaluation_id"] == "final_analog")]
            return float(sub["reduction_ratio"].iloc[0]) if len(sub) else np.nan
        elif view == "nonoverlap":
            sub = nonoverlap[(nonoverlap["method"] == method) & (nonoverlap["block"] == "combined")]
            return float(sub["reduction_ratio"].iloc[0]) if len(sub) else np.nan

    md_rows = [
        {"component": "daily_total", "oracle_reduction_ratio": oracle_reduction("oracle_daily_total"),
         "rolling_stability": "see oracle_scores", "nonoverlap_result": oracle_reduction("oracle_daily_total", "nonoverlap"),
         "final_analog_result": oracle_reduction("oracle_daily_total", "final"),
         "recommended_next_model": "robust_daily_level" if oracle_reduction("oracle_daily_total") > 0.1 else "no_action",
         "priority": "high" if oracle_reduction("oracle_daily_total") > 0.1 else "low",
         "reason": f"oracle reduction {oracle_reduction('oracle_daily_total'):.3f} rolling; U_d corr {dt_pred_corr:.3f}"},
        {"component": "region_allocation", "oracle_reduction_ratio": oracle_reduction("oracle_region_totals"),
         "rolling_stability": "see oracle_scores", "nonoverlap_result": oracle_reduction("oracle_region_totals", "nonoverlap"),
         "final_analog_result": oracle_reduction("oracle_region_totals", "final"),
         "recommended_next_model": "region_share_model" if oracle_reduction("oracle_region_totals") - oracle_reduction("oracle_daily_total") > 0.03 else "no_action",
         "priority": "medium" if oracle_reduction("oracle_region_totals") - oracle_reduction("oracle_daily_total") > 0.03 else "low",
         "reason": f"incremental over daily_total: {oracle_reduction('oracle_region_totals') - oracle_reduction('oracle_daily_total'):.3f}"},
        {"component": "hour_profile", "oracle_reduction_ratio": oracle_reduction("oracle_hour_profile"),
         "rolling_stability": "see oracle_scores", "nonoverlap_result": oracle_reduction("oracle_hour_profile", "nonoverlap"),
         "final_analog_result": oracle_reduction("oracle_hour_profile", "final"),
         "recommended_next_model": "hour_profile_model" if oracle_reduction("oracle_hour_profile") > 0.05 else "no_action",
         "priority": "medium" if oracle_reduction("oracle_hour_profile") > 0.05 else "low",
         "reason": f"oracle reduction {oracle_reduction('oracle_hour_profile'):.3f}"},
        {"component": "region_hour_residual", "oracle_reduction_ratio": oracle_reduction("oracle_hour_totals"),
         "rolling_stability": "see signal_stability.csv", "nonoverlap_result": oracle_reduction("oracle_hour_totals", "nonoverlap"),
         "final_analog_result": oracle_reduction("oracle_hour_totals", "final"),
         "recommended_next_model": "cross_fitted_residual_calibration" if len(signal[signal["signal_strength"].isin(["strong", "moderate"])]) > 0 else "no_action",
         "priority": "medium" if len(signal[signal["signal_strength"].isin(["strong", "moderate"])]) > 0 else "low",
         "reason": f"strong/moderate signals: {len(signal[signal['signal_strength']=='strong'])}/{len(signal[signal['signal_strength']=='moderate'])}"},
        {"component": "daily_total_auxiliary_features", "oracle_reduction_ratio": np.nan,
         "rolling_stability": f"R_d vs U_d corr={dt_pred_corr:.3f}", "nonoverlap_result": np.nan,
         "final_analog_result": np.nan, "recommended_next_model": "weak_nonlinear_U_adjustment" if abs(dt_pred_corr) > 0.3 else "no_action",
         "priority": "low" if abs(dt_pred_corr) < 0.3 else "medium",
         "reason": f"U_d vs R_d correlation {dt_pred_corr:.3f}"},
    ]
    modeling = pd.DataFrame(md_rows)

    # ---- write ----
    # baseline predictions
    bl_pred = all_oracle_preds[("fold_01", "baseline")][["evaluation_id", "date", "hour", "hour_of_day", "region", "y_true", "pred_float", "pred_rounded", "forecast_horizon"]].head(0)
    bl_frames = []
    for eid in SCENARIOS:
        bl_frames.append(all_oracle_preds[(eid, "baseline")][["evaluation_id", "date", "hour", "hour_of_day", "region", "y_true", "pred_float", "pred_rounded", "forecast_horizon"]])
    bl_pred = pd.concat(bl_frames, ignore_index=True)
    bl_pred["date"] = pd.to_datetime(bl_pred["date"]).dt.strftime("%Y-%m-%d")
    bl_pred.to_csv(out_dir / "a_baseline_predictions.csv", index=False, encoding="utf-8-sig")
    daily_decomp.to_csv(out_dir / "a_daily_error_decomposition.csv", index=False, encoding="utf-8-sig")
    oracle_scores.to_csv(out_dir / "a_oracle_scores.csv", index=False, encoding="utf-8-sig")
    region_scores.to_csv(out_dir / "a_region_scores.csv", index=False, encoding="utf-8-sig")
    rh_resid.to_csv(out_dir / "a_region_hour_residuals.csv", index=False, encoding="utf-8-sig")
    horizons.to_csv(out_dir / "a_horizon_scores.csv", index=False, encoding="utf-8-sig")
    nonoverlap.to_csv(out_dir / "a_nonoverlap_scores.csv", index=False, encoding="utf-8-sig")
    signal.to_csv(out_dir / "signal_stability.csv", index=False, encoding="utf-8-sig")
    modeling.to_csv(out_dir / "modeling_decision.csv", index=False, encoding="utf-8-sig")

    # ---- summary ----
    def fmt_oracle(method):
        r = oracle_reduction(method); f = oracle_reduction(method, "final"); n = oracle_reduction(method, "nonoverlap")
        roll_sse = oracle_scores[(oracle_scores["method"]==method)&(oracle_scores["evaluation_type"]=="rolling_7d")]["sse_integer"].mean()
        fa_sse = oracle_scores[(oracle_scores["method"]==method)&(oracle_scores["evaluation_id"]=="final_analog")]["sse_integer"]
        fa_sse = float(fa_sse.iloc[0]) if len(fa_sse) else np.nan
        no_sse = nonoverlap[(nonoverlap["method"]==method)&(nonoverlap["block"]=="combined")]["sse_integer"]
        no_sse = float(no_sse.iloc[0]) if len(no_sse) else np.nan
        return r, f, n, roll_sse, fa_sse, no_sse

    L = ["# Step 23 A Error Attribution Diagnostics", "", f"Run command: `{RUN_CMD}`", ""]
    L.append("## Oracle counterfactual results (rolling mean SSE)")
    L.append("")
    L.append("| method | rolling SSE | reduction % | final SSE | nonoverlap SSE |")
    L.append("| --- | --- | --- | --- | --- |")
    for m in oracle_methods:
        r, f, n, rs, fs, ns = fmt_oracle(m)
        L.append(f"| {m} | {rs:.0f} | {r*100:.1f}% | {fs:.0f} | {ns:.0f} |")
    L.append("")
    L.append("## Region-level error (baseline rolling total SSE)")
    L.append("")
    L.append("| region | SSE |")
    L.append("| --- | --- |")
    for r in REGIONS:
        rs = region_scores[(region_scores["method"]=="baseline")&(region_scores["region"]==r)&(region_scores["evaluation_type"]=="rolling_7d")]["sse_integer"].sum()
        L.append(f"| {r} | {float(rs):.0f} |")
    L.append("")
    n_strong = len(signal[signal["signal_strength"]=="strong"])
    n_mod = len(signal[signal["signal_strength"]=="moderate"])
    L.append(f"## Signal stability\n- Strong region-hour signals: {n_strong}\n- Moderate: {n_mod}\n- See signal_stability.csv for details.\n")
    L.append(f"## Daily total predictability\n- R_d vs U_d correlation (normal): {dt_pred_corr:.3f}\n- Step 22 already showed U_d vs H_d Pearson ~0.07.\n")
    L.append("## Modeling decision\n")
    L.append("| component | reduction ratio | recommended | priority |")
    L.append("| --- | --- | --- | --- |")
    for _, r in modeling.iterrows():
        rr = r['oracle_reduction_ratio']
        rr_str = f"{rr:.3f}" if pd.notna(rr) else "n/a"
        L.append(f"| {r['component']} | {rr_str} | {r['recommended_next_model']} | {r['priority']} |")
    L.append("")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    assert not any(out_dir.rglob("*.zip"))
    print("\n=== Oracle summary ===")
    for m in oracle_methods:
        r,f,n,rs,fs,ns = fmt_oracle(m)
        print(f"  {m:30s} roll={rs:.0f} ({r*100:.1f}%) fa={fs:.0f} no={ns:.0f}")
    print(f"\nStrong signals: {n_strong}, Moderate: {n_mod}")
    print(f"R_d vs U_d: {dt_pred_corr:.3f}")
    print("\n=== Output files ===")
    for p in sorted(out_dir.glob("*")):
        rows = len(pd.read_csv(p, encoding="utf-8-sig")) if p.suffix==".csv" else sum(1 for _ in open(p,encoding="utf-8"))
        print(f"  {p.relative_to(ROOT)}  rows={rows}")
    print("\nDone.")


if __name__ == "__main__":
    main()
