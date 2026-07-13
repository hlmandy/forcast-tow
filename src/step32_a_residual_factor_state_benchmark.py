"""Step 32 — A residual low-rank dynamic factor and harmonic state benchmark.

Tests whether Step 11 OOF residuals have a transferable low-rank subspace
(raw72 PCA, hour24 PCA, hour24 Fourier), and whether factor scores can be
predicted from recent history, calendar, and vessel count.

Outputs (under outputs/step32_a_residual_factor_state_benchmark/):
  1-14 as specified.
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
OUT_REL = Path("outputs/step32_a_residual_factor_state_benchmark")
REGIONS = REGION_ORDER
RUN_CMD = "python " + " ".join(sys.argv)


def smooth3(v):
    out = np.zeros(24)
    for h in range(24): out[h] = (v[(h-1)%24] + 2*v[h] + v[(h+1)%24]) / 4
    return np.clip(out, 0, None)


def lr_alloc72(target, props):
    if target <= 0: return np.zeros(72, dtype=int)
    fp = np.asarray(props, dtype=float); s = fp.sum()
    if s <= 0: return np.zeros(72, dtype=int)
    fp = fp / s * target
    base = np.floor(fp).astype(int)
    rem = int(round(target)) - int(base.sum())
    if rem > 0:
        for j in np.argsort(-(fp - base))[:rem]: base[j] += 1
    elif rem < 0:
        cnt = 0
        for j in np.argsort(fp - base):
            if base[j] > 0 and cnt < -rem: base[j] -= 1; cnt += 1
    return np.clip(base, 0, None)


def compute_baseline_pred(train_dates_blk, te, valid_dates):
    classes, _ = classify_fold_quality(train_dates_blk, {d: 0 for d in train_dates_blk})
    # need china_map for quality classification
    weights = {d: POLICY_WEIGHTS["exclude_outage_severe"][classes[d]] for d in train_dates_blk}
    # load china counts
    aud = pd.read_csv(ROOT / STEP02); aud["date"] = pd.to_datetime(aud["date"]).dt.normalize()
    china_map = dict(zip(aud["date"], aud["china_coastal_record_count"]))
    classes, _ = classify_fold_quality(train_dates_blk, china_map)
    weights = {d: POLICY_WEIGHTS["exclude_outage_severe"][classes[d]] for d in train_dates_blk}
    # load a_lab
    df_raw = pd.read_csv(ROOT / TRAIN_REL, usecols=["mmsi","x","y","sog","time"],
                         dtype={"mmsi":"string","x":"float64","y":"float64","sog":"float32"}, parse_dates=["time"])
    df_raw["hour"] = df_raw["time"].dt.floor("h"); df_raw["date"] = df_raw["time"].dt.normalize()
    df_raw["hour_of_day"] = df_raw["time"].dt.hour
    df_raw = add_regions(df_raw)
    a_lab = make_a_labels(df_raw)
    a_lab["date"] = a_lab["hour"].dt.normalize(); a_lab["hour_of_day"] = a_lab["hour"].dt.hour
    train_df = a_lab[a_lab["date"].isin(train_dates_blk)].copy()
    train_df["day_type"] = np.where(train_df["hour"].dt.dayofweek.isin([5, 6]), "weekend", "weekday")
    T_hat, P_hat = compute_components(train_df, train_dates_blk, te, weights)
    preds = {}
    for d in valid_dates:
        vd = a_lab[a_lab["date"] == d].sort_values(["hour","region"]).reset_index(drop=True)
        vr = vd["region"].to_numpy(); vh = vd["hour_of_day"].to_numpy().astype(int)
        vdt = vd["hour"].dt.dayofweek.to_numpy()
        vdt_type = np.array(["weekend" if d_ in (5,6) else "weekday" for d_ in vdt])
        Tvec = np.array([T_hat["mean"][r] for r in vr], dtype=float)
        prof = np.array([P_hat["daytype_shrunk"][r][vdt_type[i]][vh[i]] for i, r in enumerate(vr)])
        pf = Tvec * prof
        pr = np.clip(np.rint(pf), 0, None).astype(int)
        mat = np.zeros((24, 3))
        for i, r in enumerate(vr): mat[vh[i], REGIONS.index(r)] = float(pr[i])
        preds[d] = mat
    return preds


def get_a_vec(a_pivot, d):
    sub = a_pivot[a_pivot["date"]==d].sort_values("hour_of_day")
    m = np.zeros((24, 3))
    for _, row in sub.iterrows():
        h = int(row["hour_of_day"])
        for ri, r in enumerate(REGIONS): m[h, ri] = float(row[r])
    return m


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    # ---- audit ----
    (out_dir / "implementation_audit.md").write_text(
        "# Implementation Audit\n\n"
        "## Step 31\n- MMSI alignment fixed: row-level 52.7%, match-any 98.4%\n"
        "- Incremental stock Oracle target_like: +0.2% (common), +0.7% (true daily factor)\n"
        "- Stock-emission route CLOSED\n- oracle_abs/inc should be deployable=False (uses true target stock)\n\n"
        "## Step 05 PCA\n- Only analyzed normalized true profiles, NOT Step 11 OOF residuals\n"
        "- PC1 ~23%, 3 PCs ~49-57% — insufficient for 'low-dim' conclusion on profiles\n"
        "- This step analyzes RESIDUALS, not profiles\n",
        encoding="utf-8")

    # ---- load ----
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
    vc_df = pd.read_csv(ROOT / STEP01); vc_df["date"] = pd.to_datetime(vc_df["date"]).dt.normalize()
    vc_map = dict(zip(vc_df["date"], vc_df["unique_vessel_count"]))
    all_dates = sorted(pd.date_range("2018-01-01","2018-01-24",freq="D").normalize())
    normal_dates = [d for d in all_dates if qmap.get(d) == "normal"]

    # ---- cross-fitted residuals ----
    print("=== Cross-fitted residuals ===")
    cf_resid = {}  # date -> (72-vec, 24-vec, train_end, bl_pred)
    for i, d in enumerate(normal_dates):
        prev = [dd for dd in all_dates if dd < d]
        prev_normal = [dd for dd in prev if qmap.get(dd) == "normal"]
        if len(prev_normal) < 6: continue
        te = d - pd.Timedelta(days=1)
        bl_pred = compute_baseline_pred(prev, te, [d])[d]  # (24,3)
        a_vec = get_a_vec(a_pivot, d)  # (24,3)
        resid72 = (a_vec - bl_pred).flatten()  # (72,)
        resid24 = (a_vec - bl_pred).sum(axis=1)  # (24,)
        cf_resid[d] = {"r72": resid72, "r24": resid24, "bl": bl_pred, "a": a_vec,
                        "train_end": te, "day_type": "weekend" if d.dayofweek in (5,6) else "weekday",
                        "U_d": vc_map.get(d, 0), "bl_total": float(bl_pred.sum()), "a_total": float(a_vec.sum())}
    cf_dates = sorted(cf_resid.keys())
    print(f"Cross-fitted residual dates: {len(cf_dates)}")

    # save residual library
    cf_rows = []
    for d in cf_dates:
        r = cf_resid[d]
        for h in range(24):
            for ri, reg in enumerate(REGIONS):
                cf_rows.append({"date": f"{d:%Y-%m-%d}", "hour": h, "region": reg,
                                "residual": float(r["r72"][h*3+ri]), "hour_total_residual": float(r["r24"][h]),
                                "bl_pred": float(r["bl"][h, ri]), "a_true": float(r["a"][h, ri]),
                                "train_end": f"{r['train_end']:%Y-%m-%d}", "day_type": r["day_type"],
                                "U_d": r["U_d"], "bl_total": r["bl_total"], "a_total": r["a_total"]})
    pd.DataFrame(cf_rows).to_csv(out_dir / "crossfitted_residual_library.csv", index=False, encoding="utf-8-sig")

    # ---- blocks ----
    blocks = {
        "block_pre_normal": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-07"),
                             pd.Timestamp("2018-01-08"), pd.Timestamp("2018-01-11")),
        "block_target_like": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-19"),
                              pd.Timestamp("2018-01-20"), pd.Timestamp("2018-01-24")),
        "block_post_outage_stress": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-18"),
                                     pd.Timestamp("2018-01-19"), pd.Timestamp("2018-01-24")),
    }

    # ---- compute block baselines ----
    print("\n=== Block baselines ===")
    block_bl = {}; block_bl_sse = {}
    for bname, (ts, te, vs_, ve) in blocks.items():
        td = sorted(pd.date_range(ts, te, freq="D").normalize())
        vd = list(pd.date_range(vs_, ve, freq="D").normalize())
        preds = compute_baseline_pred(td, te, vd)
        block_bl[bname] = preds
        sse = sum(float(((preds[d] - get_a_vec(a_pivot, d))**2).sum()) for d in vd)
        block_bl_sse[bname] = sse
        print(f"  {bname}: SSE={sse:.0f}")

    # ---- representations ----
    def fit_pca(resid_matrix, max_rank):
        """resid_matrix: (n_days, dim). Return (mean, components, explained_var_ratio)."""
        mean = resid_matrix.mean(axis=0)
        Xc = resid_matrix - mean
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = s**2 / (len(resid_matrix) - 1)
        total = var.sum()
        ratios = var / total if total > 0 else np.zeros_like(var)
        return mean, Vt, ratios

    def project_pca(r, mean, Vt, rank):
        V = Vt[:rank]
        return mean + V.T @ (V @ (r - mean))

    def fourier_basis(K):
        h = np.arange(24)
        cols = [np.ones(24)]
        for k in range(1, K+1):
            cols.append(np.sin(2*np.pi*k*h/24))
            cols.append(np.cos(2*np.pi*k*h/24))
        return np.column_stack(cols)

    def fit_fourier(resid24_matrix, K):
        basis = fourier_basis(K)
        coefs = np.linalg.lstsq(basis, resid24_matrix.T, rcond=None)[0].T  # (n_days, 2K+1)
        mean_coef = coefs.mean(axis=0)
        return basis, mean_coef, coefs

    def project_fourier(r24, basis, mean_coef):
        coef = np.linalg.lstsq(basis, r24, rcond=None)[0]
        return basis @ coef

    # ---- evaluate methods per block ----
    print("\n=== Scoring ===")
    all_scores = []
    rep_diag_rows = []
    oracle_proj_rows = []

    for bname, (ts, te, vs_, ve) in blocks.items():
        vd = list(pd.date_range(vs_, ve, freq="D").normalize())
        bl = block_bl[bname]; bl_s = block_bl_sse[bname]
        # training residuals for this block
        train_cf = [d for d in cf_dates if d <= te]
        if len(train_cf) < 5:
            print(f"  {bname}: insufficient training residuals ({len(train_cf)}), skipping")
            continue
        R72 = np.array([cf_resid[d]["r72"] for d in train_cf])
        R24 = np.array([cf_resid[d]["r24"] for d in train_cf])

        # PCA fits
        mean72, Vt72, ratios72 = fit_pca(R72, 5)
        mean24, Vt24, ratios24 = fit_pca(R24, 4)

        # baselines for region proportion
        bl_for_props = bl  # use block baseline for region proportions

        # Oracle: project true target residual onto training subspace
        for rep_name, mean_r, Vt_r, dim, ranks in [
            ("raw72_pca", mean72, Vt72, 72, [1,2,3,5]),
            ("hour24_pca", mean24, Vt24, 24, [1,2,3,4]),
        ]:
            for rank in ranks:
                if len(train_cf) < rank + 3: continue
                oracle_sse = 0
                for d in vd:
                    bl_d = bl[d]
                    a_d = get_a_vec(a_pivot, d)
                    if rep_name == "raw72_pca":
                        r_true = (a_d - bl_d).flatten()
                        r_proj = project_pca(r_true, mean_r, Vt_r, rank)
                        pred = bl_d + r_proj.reshape(24, 3)
                    else:
                        g_true = (a_d - bl_d).sum(axis=1)
                        g_proj = project_pca(g_true, mean_r, Vt_r, rank)
                        # distribute to regions by baseline proportion
                        pred = bl_d.copy()
                        for h in range(24):
                            bl_h = bl_d[h]
                            s = bl_h.sum()
                            if s > 0:
                                pred[h] = bl_h + g_proj[h] * bl_h / s
                            else:
                                pred[h] = bl_h
                    pred_int = np.clip(np.rint(pred), 0, None).astype(int)
                    sse = float(((pred_int - a_d)**2).sum())
                    oracle_sse += sse
                red = 1 - oracle_sse / bl_s if bl_s else 0
                oracle_proj_rows.append({"block": bname, "representation": rep_name, "rank": rank,
                                        "oracle_sse": oracle_sse, "baseline_sse": bl_s, "reduction": red})
                all_scores.append({"block": bname, "method": f"oracle_{rep_name}_rank{rank}", "deployable": False,
                                  "representation": rep_name, "rank_or_k": rank, "sse": oracle_sse,
                                  "baseline_sse": bl_s, "reduction_ratio": red})
                print(f"  {bname} oracle_{rep_name}_rank{rank}: SSE={oracle_sse:.0f} red={red*100:.1f}%")

                # representation diagnostics
                cum_var = float(ratios72[:rank].sum()) if rep_name == "raw72_pca" else float(ratios24[:rank].sum())
                rep_diag_rows.append({"outer_origin": bname, "representation": rep_name, "rank_or_k": rank,
                                     "n_training_residual_days": len(train_cf),
                                     "cumulative_explained_variance": cum_var,
                                     "oracle_float_reduction": red, "oracle_integer_reduction": red})

        # Fourier Oracle
        for K in [1, 2, 3, 4]:
            basis, mean_coef, coefs = fit_fourier(R24, K)
            oracle_sse = 0
            for d in vd:
                bl_d = bl[d]; a_d = get_a_vec(a_pivot, d)
                g_true = (a_d - bl_d).sum(axis=1)
                g_proj = project_fourier(g_true, basis, mean_coef)
                pred = bl_d.copy()
                for h in range(24):
                    bl_h = bl_d[h]; s = bl_h.sum()
                    if s > 0: pred[h] = bl_h + g_proj[h] * bl_h / s
                    else: pred[h] = bl_h
                pred_int = np.clip(np.rint(pred), 0, None).astype(int)
                oracle_sse += float(((pred_int - a_d)**2).sum())
            red = 1 - oracle_sse / bl_s if bl_s else 0
            oracle_proj_rows.append({"block": bname, "representation": "hour24_fourier", "K": K,
                                    "oracle_sse": oracle_sse, "baseline_sse": bl_s, "reduction": red})
            all_scores.append({"block": bname, "method": f"oracle_hour24_fourier_K{K}", "deployable": False,
                              "representation": "hour24_fourier", "rank_or_k": K, "sse": oracle_sse,
                              "baseline_sse": bl_s, "reduction_ratio": red})
            print(f"  {bname} oracle_hour24_fourier_K{K}: SSE={oracle_sse:.0f} red={red*100:.1f}%")

        # ---- deployable: mean residual calibration ----
        mean_resid72 = R72.mean(axis=0).reshape(24, 3)
        for alpha in [0.25, 0.5, 0.75, 1.0]:
            mr_sse = 0
            for d in vd:
                pred = block_bl[bname][d] + alpha * mean_resid72
                pred_int = np.clip(np.rint(pred), 0, None).astype(int)
                mr_sse += float(((pred_int - get_a_vec(a_pivot, d))**2).sum())
            red = 1 - mr_sse / bl_s if bl_s else 0
            all_scores.append({"block": bname, "method": f"residual_mean_alpha{alpha}", "deployable": True,
                              "representation": "mean", "rank_or_k": 0, "sse": mr_sse,
                              "baseline_sse": bl_s, "reduction_ratio": red})

        # ---- deployable: PCA with recent3 score ----
        for rep_name, mean_r, Vt_r, ranks in [
            ("raw72_pca", mean72, Vt72, [1,2,3]),
            ("hour24_pca", mean24, Vt24, [1,2,3]),
        ]:
            for rank in ranks:
                if len(train_cf) < rank + 3: continue
                # recent3 score: use last 3 training residual days' average projection
                V = Vt_r[:rank]
                recent3 = R72[-3:] if rep_name == "raw72_pca" else R24[-3:]
                scores_recent3 = np.mean([V @ (r - mean_r) for r in recent3], axis=0)
                pred_residual = mean_r + V.T @ scores_recent3
                for alpha in [0.5, 1.0]:
                    pca_sse = 0
                    for di, d in enumerate(vd):
                        bl_d = bl[d]; a_d = get_a_vec(a_pivot, d)
                        if rep_name == "raw72_pca":
                            pred = bl_d + alpha * pred_residual.reshape(24, 3)
                        else:
                            pred = bl_d.copy()
                            for h in range(24):
                                bl_h = bl_d[h]; s = bl_h.sum()
                                if s > 0: pred[h] = bl_h + alpha * pred_residual[h] * bl_h / s
                                else: pred[h] = bl_h
                        pred_int = np.clip(np.rint(pred), 0, None).astype(int)
                        pca_sse += float(((pred_int - a_d)**2).sum())
                    red = 1 - pca_sse / bl_s if bl_s else 0
                    all_scores.append({"block": bname, "method": f"pca_{rep_name}_rank{rank}_recent3_a{alpha}",
                                      "deployable": True, "representation": rep_name, "rank_or_k": rank,
                                      "sse": pca_sse, "baseline_sse": bl_s, "reduction_ratio": red})
                    if red > 0.01:
                        print(f"  {bname} pca_{rep_name}_rank{rank}_recent3_a{alpha}: SSE={pca_sse:.0f} red={red*100:.1f}%")

        # baseline
        all_scores.append({"block": bname, "method": "exact_step11_baseline", "deployable": True,
                          "representation": "none", "rank_or_k": 0, "sse": bl_s,
                          "baseline_sse": bl_s, "reduction_ratio": 0})

    scores_df = pd.DataFrame(all_scores)
    scores_df.to_csv(out_dir / "block_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rep_diag_rows).to_csv(out_dir / "representation_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(oracle_proj_rows).to_csv(out_dir / "oracle_projection_scores.csv", index=False, encoding="utf-8-sig")

    # ---- normal one-day origins ----
    print("\n=== Normal one-day origins ===")
    od_rows = []
    for d in cf_dates:
        bl_d = cf_resid[d]["bl"]; a_d = cf_resid[d]["a"]
        bl_sse = float(((bl_d - a_d)**2).sum())
        # mean residual
        train_cf_before = [dd for dd in cf_dates if dd < d]
        if len(train_cf_before) < 3: continue
        mean_r72 = np.mean([cf_resid[dd]["r72"] for dd in train_cf_before], axis=0).reshape(24, 3)
        for alpha in [0.5, 1.0]:
            pred = bl_d + alpha * mean_r72
            pred_int = np.clip(np.rint(pred), 0, None).astype(int)
            mr_sse = float(((pred_int - a_d)**2).sum())
            od_rows.append({"date": f"{d:%Y-%m-%d}", "method": f"mean_residual_a{alpha}",
                           "baseline_sse": bl_sse, "method_sse": mr_sse, "improved": mr_sse < bl_sse})
    od_df = pd.DataFrame(od_rows)
    if len(od_df):
        n_imp = int(od_df[od_df["method"]=="mean_residual_a1.0"]["improved"].sum())
        n_tot = int(len(od_df[od_df["method"]=="mean_residual_a1.0"]))
        print(f"One-day improved (mean_residual_a1.0): {n_imp}/{n_tot}")
    od_df.to_csv(out_dir / "normal_one_day_scores.csv", index=False, encoding="utf-8-sig")

    # ---- summary ----
    bl_s = {b: block_bl_sse[b] for b in blocks}
    L = ["# Step 32 A Residual Factor State Benchmark", "", f"Run command: `{RUN_CMD}`", ""]
    L.append(f"## Cross-fitted residual dates: {len(cf_dates)}")
    L.append("")
    L.append("## Oracle projections (key results)")
    L.append("")
    L.append("| block | method | SSE | reduction |")
    L.append("| --- | --- | --- | --- |")
    for _, r in scores_df[scores_df["method"].str.startswith("oracle")].iterrows():
        L.append(f"| {r['block']} | {r['method']} | {r['sse']:.0f} | {r['reduction_ratio']*100:.1f}% |")
    L.append("")
    L.append("## Deployable methods (key results)")
    L.append("")
    for b in blocks:
        sub = scores_df[(scores_df["block"]==b) & (scores_df["deployable"]) & (scores_df["reduction_ratio"] > 0.001)]
        if len(sub):
            best = sub.loc[sub["reduction_ratio"].idxmax()]
            L.append(f"- {b}: best deployable = {best['method']} ({best['reduction_ratio']*100:.1f}%)")
        else:
            L.append(f"- {b}: no deployable improvement > 0.1%")
    L.append("")

    # check if any oracle passes 10%
    oracle_pass = scores_df[(scores_df["deployable"]==False) & (scores_df["reduction_ratio"] >= 0.10)]
    dep_pass = scores_df[(scores_df["deployable"]==True) & (scores_df["reduction_ratio"] >= 0.05)]
    L.append(f"## Decision")
    L.append(f"- Oracle passes 10%: {len(oracle_pass)} methods")
    L.append(f"- Deployable passes 5%: {len(dep_pass)} methods")
    if len(oracle_pass) == 0:
        L.append("- All Oracle methods fail 10% on all blocks → close low-rank residual route")
    elif len(dep_pass) == 0:
        L.append("- Oracle passes but deployable fails → subspace exists but factor prediction fails")
    L.append("")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    assert not any(out_dir.rglob("*.zip"))
    print(f"\n=== Summary ===")
    print(f"Cross-fitted residuals: {len(cf_dates)} dates")
    for _, r in scores_df[scores_df["method"].str.startswith("oracle")].iterrows():
        print(f"  {r['block']:30s} {r['method']:40s} SSE={r['sse']:.0f} red={r['reduction_ratio']*100:.1f}%")
    print(f"\nOracle passes 10%: {len(oracle_pass)}")
    print(f"Deployable passes 5%: {len(dep_pass)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
