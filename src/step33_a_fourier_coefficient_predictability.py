"""Step 33 — Fourier Oracle decomposition and coefficient predictability repair.

Splits Step 32's Fourier Oracle into level-only (true daily mean residual) and
shape-only (true zero-mean hourly Fourier coefficients). Tests whether the 35%
improvement came from the daily level (already known from Step 23) or from a
genuine transferable harmonic shape. Implements deployable coefficient forecasting.

Outputs (under outputs/step33_a_fourier_coefficient_predictability/):
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
OUT_REL = Path("outputs/step33_a_fourier_coefficient_predictability")
REGIONS = REGION_ORDER
RUN_CMD = "python " + " ".join(sys.argv)

# Cache for loaded data
_data_cache = {}


def get_data():
    if _data_cache: return _data_cache
    df = pd.read_csv(ROOT / TRAIN_REL, usecols=["mmsi","x","y","sog","time"],
                     dtype={"mmsi":"string","x":"float64","y":"float64","sog":"float32"}, parse_dates=["time"])
    df["hour"] = df["time"].dt.floor("h"); df["date"] = df["time"].dt.normalize()
    df["hour_of_day"] = df["time"].dt.hour
    df = add_regions(df)
    a_lab = make_a_labels(df)
    a_lab["date"] = a_lab["hour"].dt.normalize(); a_lab["hour_of_day"] = a_lab["hour"].dt.hour
    a_pivot = a_lab.pivot_table(index=["date","hour_of_day"], columns="region", values="y", aggfunc="first").reset_index()
    for r in REGIONS:
        if r not in a_pivot.columns: a_pivot[r] = 0
    aud = pd.read_csv(ROOT / STEP02); aud["date"] = pd.to_datetime(aud["date"]).dt.normalize()
    qmap = dict(zip(aud["date"], aud["quality_regime"]))
    china_map = dict(zip(aud["date"], aud["china_coastal_record_count"]))
    vc_df = pd.read_csv(ROOT / STEP01); vc_df["date"] = pd.to_datetime(vc_df["date"]).dt.normalize()
    vc_map = dict(zip(vc_df["date"], vc_df["unique_vessel_count"]))
    all_dates = sorted(pd.date_range("2018-01-01","2018-01-24",freq="D").normalize())
    _data_cache.update({"df": df, "a_lab": a_lab, "a_pivot": a_pivot, "qmap": qmap, "china_map": china_map,
                         "vc_map": vc_map, "all_dates": all_dates})
    return _data_cache


def compute_baseline(train_dates, te, valid_dates):
    d = get_data()
    classes, _ = classify_fold_quality(train_dates, d["china_map"])
    weights = {dd: POLICY_WEIGHTS["exclude_outage_severe"][classes[dd]] for dd in train_dates}
    train_df = d["a_lab"][d["a_lab"]["date"].isin(train_dates)].copy()
    train_df["day_type"] = np.where(train_df["hour"].dt.dayofweek.isin([5,6]), "weekend", "weekday")
    T_hat, P_hat = compute_components(train_df, train_dates, te, weights)
    preds = {}
    for dd in valid_dates:
        vd = d["a_lab"][d["a_lab"]["date"] == dd].sort_values(["hour","region"]).reset_index(drop=True)
        vr = vd["region"].to_numpy(); vh = vd["hour_of_day"].to_numpy().astype(int)
        vdt = vd["hour"].dt.dayofweek.to_numpy()
        vdt_type = np.array(["weekend" if x in (5,6) else "weekday" for x in vdt])
        Tvec = np.array([T_hat["mean"][r] for r in vr], dtype=float)
        prof = np.array([P_hat["daytype_shrunk"][r][vdt_type[i]][vh[i]] for i, r in enumerate(vr)])
        pr = np.clip(np.rint(Tvec * prof), 0, None).astype(int)
        mat = np.zeros((24, 3))
        for i, r in enumerate(vr): mat[vh[i], REGIONS.index(r)] = float(pr[i])
        preds[dd] = mat
    return preds


def get_a_vec(d):
    data = get_data()
    sub = data["a_pivot"][data["a_pivot"]["date"]==d].sort_values("hour_of_day")
    m = np.zeros((24, 3))
    for _, row in sub.iterrows():
        h = int(row["hour_of_day"])
        for ri, r in enumerate(REGIONS): m[h, ri] = float(row[r])
    return m


def fourier_basis_nocst(K):
    """Sine/cosine basis WITHOUT constant term. Returns (24, 2K)."""
    h = np.arange(24)
    cols = []
    for k in range(1, K+1):
        cols.append(np.sin(2*np.pi*k*h/24))
        cols.append(np.cos(2*np.pi*k*h/24))
    return np.column_stack(cols)


def fit_fourier_nocst(s_24, K):
    """Fit zero-mean shape to Fourier basis. Returns coefficients (2K,)."""
    basis = fourier_basis_nocst(K)
    coef, *_ = np.linalg.lstsq(basis, s_24, rcond=None)
    return coef


def reconstruct_shape(coef, K):
    return fourier_basis_nocst(K) @ coef


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")
    data = get_data()
    qmap = data["qmap"]; all_dates = data["all_dates"]; vc_map = data["vc_map"]
    normal_dates = [d for d in all_dates if qmap.get(d) == "normal"]

    # ---- audit ----
    (out_dir / "implementation_audit.md").write_text(
        "# Step 32 Implementation Audit\n\n"
        "1. Fourier basis is a fixed mathematical basis, NOT learned from training residuals\n"
        "2. Fourier Oracle fits target-day TRUE residual coefficients — not a historical transfer test\n"
        "3. `mean_coef` was computed but NOT used in Oracle reconstruction\n"
        "4. K=4 with constant = 9 free params fitting 24 hours — this is a smoothing upper bound\n"
        "5. No deployable Fourier coefficient forecasting was implemented\n"
        "6. Inner CV, run-length, region scores not generated\n"
        "7. Step 32 only proves: residual contains smooth low-frequency components\n"
        "8. Cannot conclude Fourier coefficients are unpredictable\n",
        encoding="utf-8")

    # ---- cross-fitted residuals (expanded: min 5 earlier calendar dates) ----
    print("=== Cross-fitted residuals ===")
    cf = {}  # date -> dict
    for d in normal_dates:
        prev = [dd for dd in all_dates if dd < d]
        if len(prev) < 5: continue
        prev_normal = [dd for dd in prev if qmap.get(dd) == "normal"]
        if len(prev_normal) < 4: continue
        te = d - pd.Timedelta(days=1)
        bl = compute_baseline(prev, te, [d])[d]
        a = get_a_vec(d)
        g24 = (a - bl).sum(axis=1)  # hourly total residual
        level = float(g24.mean())
        shape = g24 - level  # zero-mean
        assert abs(shape.sum()) < 1e-10
        cf[d] = {"bl": bl, "a": a, "g24": g24, "level": level, "shape": shape,
                 "te": te, "day_type": "weekend" if d.dayofweek in (5,6) else "weekday",
                 "U_d": vc_map.get(d, 0)}
    cf_dates = sorted(cf.keys())
    print(f"Cross-fitted dates: {len(cf_dates)}")

    # ---- fit Fourier coefficients for each date ----
    for K in [1, 2, 3, 4]:
        for d in cf_dates:
            coef = fit_fourier_nocst(cf[d]["shape"], K)
            cf[d][f"coef_K{K}"] = coef

    # ---- coefficient history ----
    ch_rows = []
    for d in cf_dates:
        r = cf[d]
        row = {"date": f"{d:%Y-%m-%d}", "level": r["level"], "day_type": r["day_type"], "U_d": r["U_d"]}
        for K in [1, 2, 3, 4]:
            coef = r[f"coef_K{K}"]
            for k in range(K):
                a_k = coef[2*k]; b_k = coef[2*k+1]
                amp = math.sqrt(a_k**2 + b_k**2)
                phase = math.atan2(a_k, b_k) if amp > 1e-10 else 0
                row[f"K{K}_a{k+1}"] = a_k; row[f"K{K}_b{k+1}"] = b_k
                row[f"K{K}_amp{k+1}"] = amp; row[f"K{K}_phase{k+1}"] = phase
        ch_rows.append(row)
    ch_df = pd.DataFrame(ch_rows)
    ch_df.to_csv(out_dir / "fourier_coefficient_history.csv", index=False, encoding="utf-8-sig")

    # ---- coefficient diagnostics ----
    print("\n=== Coefficient diagnostics ===")
    diag_rows = []
    for K in [1, 2]:
        for k in range(K):
            for var_name, col_a, col_b in [
                (f"K{K}_harmonic{k+1}", f"K{K}_a{k+1}", f"K{K}_b{k+1}")]:
                a_vals = ch_df[col_a].to_numpy(float)
                b_vals = ch_df[col_b].to_numpy(float)
                amps = np.sqrt(a_vals**2 + b_vals**2)
                phases = np.arctan2(a_vals, b_vals)
                # lag-1
                lag1_a = float(pd.Series(a_vals).autocorr(1)) if len(a_vals) > 2 else np.nan
                lag1_amp = float(pd.Series(amps).autocorr(1)) if len(amps) > 2 else np.nan
                # weekday/weekend
                wd = ch_df[ch_df["day_type"]=="weekday"]
                we = ch_df[ch_df["day_type"]=="weekend"]
                diag_rows.append({"variable": var_name, "K": K, "harmonic": k+1,
                                 "a_mean": float(a_vals.mean()), "a_std": float(a_vals.std(ddof=1)) if len(a_vals)>1 else np.nan,
                                 "b_mean": float(b_vals.mean()), "b_std": float(b_vals.std(ddof=1)) if len(b_vals)>1 else np.nan,
                                 "amp_mean": float(amps.mean()), "amp_std": float(amps.std(ddof=1)) if len(amps)>1 else np.nan,
                                 "phase_mean": float(phases.mean()), "lag1_a": lag1_a, "lag1_amp": lag1_amp,
                                 "n_weekday": len(wd), "n_weekend": len(we),
                                 "weekday_a_mean": float(wd[col_a].mean()) if len(wd) else np.nan,
                                 "weekend_a_mean": float(we[col_a].mean()) if len(we) else np.nan})
                print(f"  {var_name}: amp={float(amps.mean()):.3f}±{float(amps.std(ddof=1)) if len(amps)>1 else 0:.3f}, lag1_amp={lag1_a:.3f}")
    pd.DataFrame(diag_rows).to_csv(out_dir / "coefficient_diagnostics.csv", index=False, encoding="utf-8-sig")

    # ---- blocks ----
    blocks = {
        "block_pre_normal": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-07"),
                             pd.Timestamp("2018-01-08"), pd.Timestamp("2018-01-11")),
        "block_target_like": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-19"),
                              pd.Timestamp("2018-01-20"), pd.Timestamp("2018-01-24")),
        "block_post_outage_stress": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-18"),
                                     pd.Timestamp("2018-01-19"), pd.Timestamp("2018-01-24")),
    }

    # ---- block baselines ----
    print("\n=== Block baselines ===")
    block_bl = {}; block_bl_sse = {}
    for bname, (ts, te, vs_, ve) in blocks.items():
        td = sorted(pd.date_range(ts, te, freq="D").normalize())
        vd = list(pd.date_range(vs_, ve, freq="D").normalize())
        preds = compute_baseline(td, te, vd)
        block_bl[bname] = preds
        sse = sum(float(((preds[d] - get_a_vec(d))**2).sum()) for d in vd)
        block_bl_sse[bname] = sse
        print(f"  {bname}: SSE={sse:.0f}")

    # ---- Oracle decomposition ----
    print("\n=== Oracle decomposition ===")
    oracle_rows = []
    for bname, (ts, te, vs_, ve) in blocks.items():
        vd = list(pd.date_range(vs_, ve, freq="D").normalize())
        bl = block_bl[bname]; bl_s = block_bl_sse[bname]

        for d in vd:
            a = get_a_vec(d); bl_d = bl[d]
            g24 = (a - bl_d).sum(axis=1)
            level = float(g24.mean())
            shape = g24 - level
            # distribute level and shape to regions by baseline proportion
            def apply_correction(level_val, shape_val):
                pred = bl_d.copy()
                for h in range(24):
                    bl_h = bl_d[h]; s = bl_h.sum()
                    g_h = level_val + shape_val[h]
                    if s > 0: pred[h] = bl_h + g_h * bl_h / s
                    else: pred[h] = bl_h
                return np.clip(np.rint(pred), 0, None).astype(int)

            # level only
            pred_level = apply_correction(level, np.zeros(24))
            sse_level = float(((pred_level - a)**2).sum())

            # shape only (for each K)
            for K in [1, 2, 3, 4]:
                coef = fit_fourier_nocst(shape, K)
                shape_recon = reconstruct_shape(coef, K)
                pred_shape = apply_correction(0.0, shape_recon)
                sse_shape = float(((pred_shape - a)**2).sum())

                # combined
                pred_comb = apply_correction(level, shape_recon)
                sse_comb = float(((pred_comb - a)**2).sum())

                oracle_rows.append({"block": bname, "date": f"{d:%Y-%m-%d}", "K": K,
                                   "sse_baseline": float(((bl_d - a)**2).sum()),
                                   "sse_level_only": sse_level, "sse_shape_only": sse_shape,
                                   "sse_combined": sse_comb,
                                   "red_level": 1 - sse_level/float(((bl_d-a)**2).sum()) if float(((bl_d-a)**2).sum()) else 0,
                                   "red_shape": 1 - sse_shape/float(((bl_d-a)**2).sum()) if float(((bl_d-a)**2).sum()) else 0,
                                   "red_combined": 1 - sse_comb/float(((bl_d-a)**2).sum()) if float(((bl_d-a)**2).sum()) else 0})

    oracle_df = pd.DataFrame(oracle_rows)
    # aggregate per block
    oracle_agg = oracle_df.groupby(["block", "K"]).agg(
        sse_baseline=("sse_baseline", "sum"), sse_level=("sse_level_only", "sum"),
        sse_shape=("sse_shape_only", "sum"), sse_combined=("sse_combined", "sum")).reset_index()
    oracle_agg["red_level"] = 1 - oracle_agg["sse_level"] / oracle_agg["sse_baseline"]
    oracle_agg["red_shape"] = 1 - oracle_agg["sse_shape"] / oracle_agg["sse_baseline"]
    oracle_agg["red_combined"] = 1 - oracle_agg["sse_combined"] / oracle_agg["sse_baseline"]
    oracle_agg.to_csv(out_dir / "oracle_decomposition_scores.csv", index=False, encoding="utf-8-sig")

    print("\nOracle decomposition (aggregated per block):")
    for _, r in oracle_agg.iterrows():
        print(f"  {r['block']:30s} K={int(r['K'])} level={r['red_level']*100:.1f}% shape={r['red_shape']*100:.1f}% combined={r['red_combined']*100:.1f}%")

    # ---- deployable: level_zero + coef predictions ----
    print("\n=== Deployable Fourier ===")
    bs_rows = []
    for bname, (ts, te, vs_, ve) in blocks.items():
        vd = list(pd.date_range(vs_, ve, freq="D").normalize())
        bl = block_bl[bname]; bl_s = block_bl_sse[bname]
        train_cf = [d for d in cf_dates if d <= te]
        if len(train_cf) < 5: continue

        for K in [1, 2]:
            # training coefficients
            train_coefs = np.array([cf[d][f"coef_K{K}"] for d in train_cf])
            train_mean_coef = train_coefs.mean(axis=0)
            # recent3
            rec3_coefs = train_coefs[-3:].mean(axis=0) if len(train_coefs) >= 3 else train_mean_coef
            # recent1
            rec1_coef = train_coefs[-1] if len(train_coefs) > 0 else train_mean_coef

            for coef_name, coef_val in [("coef_zero", np.zeros_like(train_mean_coef)),
                                         ("coef_long_mean", train_mean_coef),
                                         ("coef_recent1", rec1_coef),
                                         ("coef_recent3_mean", rec3_coefs)]:
                for alpha in [0.5, 1.0]:
                    total_sse = 0
                    for di, d in enumerate(vd):
                        bl_d = bl[d]; a_d = get_a_vec(d)
                        shape_pred = reconstruct_shape(coef_val, K)
                        pred = bl_d.copy()
                        for h in range(24):
                            bl_h = bl_d[h]; s = bl_h.sum()
                            if s > 0: pred[h] = bl_h + alpha * shape_pred[h] * bl_h / s
                            else: pred[h] = bl_h
                        pred_int = np.clip(np.rint(pred), 0, None).astype(int)
                        total_sse += float(((pred_int - a_d)**2).sum())
                    red = 1 - total_sse / bl_s if bl_s else 0
                    method = f"fourier_K{K}_{coef_name}_a{alpha}"
                    bs_rows.append({"block": bname, "method": method, "deployable": True,
                                  "K": K, "coef_method": coef_name, "alpha": alpha,
                                  "sse": total_sse, "baseline_sse": bl_s, "reduction_ratio": red})
                    if red > 0.005:
                        print(f"  {bname} {method}: SSE={total_sse:.0f} red={red*100:.1f}%")

        # baseline
        bs_rows.append({"block": bname, "method": "exact_step11_baseline", "deployable": True,
                       "K": 0, "coef_method": "none", "alpha": 0,
                       "sse": bl_s, "baseline_sse": bl_s, "reduction_ratio": 0})

    bs_df = pd.DataFrame(bs_rows)
    bs_df.to_csv(out_dir / "block_scores.csv", index=False, encoding="utf-8-sig")

    # ---- normal one-day ----
    print("\n=== Normal one-day origins ===")
    od_rows = []
    for d in cf_dates:
        train_cf_before = [dd for dd in cf_dates if dd < d]
        if len(train_cf_before) < 3: continue
        bl_d = cf[d]["bl"]; a_d = cf[d]["a"]
        bl_sse = float(((bl_d - a_d)**2).sum())
        for K in [1, 2]:
            mean_coef = np.mean([cf[dd][f"coef_K{K}"] for dd in train_cf_before], axis=0)
            shape_pred = reconstruct_shape(mean_coef, K)
            for alpha in [0.5, 1.0]:
                pred = bl_d.copy()
                for h in range(24):
                    bl_h = bl_d[h]; s = bl_h.sum()
                    if s > 0: pred[h] = bl_h + alpha * shape_pred[h] * bl_h / s
                pred_int = np.clip(np.rint(pred), 0, None).astype(int)
                sse = float(((pred_int - a_d)**2).sum())
                od_rows.append({"date": f"{d:%Y-%m-%d}", "K": K, "alpha": alpha,
                               "baseline_sse": bl_sse, "method_sse": sse, "improved": sse < bl_sse})
    od_df = pd.DataFrame(od_rows)
    od_df.to_csv(out_dir / "normal_one_day_scores.csv", index=False, encoding="utf-8-sig")
    if len(od_df):
        for K in [1, 2]:
            sub = od_df[(od_df["K"]==K) & (od_df["alpha"]==1.0)]
            n_imp = int(sub["improved"].sum()); n_tot = len(sub)
            print(f"  K={K} a=1.0: {n_imp}/{n_tot} improved")

    # ---- summary ----
    L = ["# Step 33 Fourier Coefficient Predictability", "", f"Run command: `{RUN_CMD}`", ""]
    L.append(f"## Cross-fitted dates: {len(cf_dates)}")
    L.append("")
    L.append("## Oracle decomposition (target_like)")
    L.append("")
    L.append("| K | level red | shape red | combined red |")
    L.append("| --- | --- | --- | --- |")
    tl = oracle_agg[oracle_agg["block"]=="block_target_like"]
    for _, r in tl.iterrows():
        L.append(f"| {int(r['K'])} | {r['red_level']*100:.1f}% | {r['red_shape']*100:.1f}% | {r['red_combined']*100:.1f}% |")
    L.append("")
    L.append("## Deployable methods (target_like)")
    L.append("")
    tl_dep = bs_df[(bs_df["block"]=="block_target_like") & (bs_df["deployable"]) & (bs_df["reduction_ratio"] > 0.001)]
    if len(tl_dep):
        for _, r in tl_dep.iterrows():
            L.append(f"- {r['method']}: {r['reduction_ratio']*100:.1f}%")
    else:
        L.append("- No deployable method improves > 0.1%")
    L.append("")
    shape_pass = any(oracle_agg[(oracle_agg["block"]=="block_target_like")]["red_shape"] >= 0.10)
    dep_pass = any(bs_df[(bs_df["block"]=="block_target_like") & (bs_df["deployable"])]["reduction_ratio"] >= 0.05)
    L.append(f"## Decision")
    L.append(f"- Shape Oracle passes 10%: {'YES' if shape_pass else 'NO'}")
    L.append(f"- Deployable passes 5%: {'YES' if dep_pass else 'NO'}")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    assert not any(out_dir.rglob("*.zip"))
    print(f"\n=== Summary ===")
    print("Oracle decomposition (target_like):")
    for _, r in oracle_agg[oracle_agg["block"]=="block_target_like"].iterrows():
        print(f"  K={int(r['K'])}: level={r['red_level']*100:.1f}% shape={r['red_shape']*100:.1f}% combined={r['red_combined']*100:.1f}%")
    print(f"\nShape Oracle passes 10%: {shape_pass}")
    print(f"Deployable passes 5%: {dep_pass}")
    print("\nDone.")


if __name__ == "__main__":
    main()
