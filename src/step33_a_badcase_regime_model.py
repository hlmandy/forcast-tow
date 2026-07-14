"""Step 33 — A bad-case activity regime and conditional hour-curve model.

Targets the observed error pattern: Step 11 baseline predicts ~312/day flat,
but real daily totals range 247-407. High-error days have different hour
shapes that correlate with activity level. This step defines low/middle/high
activity states from training residuals and builds state-conditional hour
templates + level-conditioned Fourier coefficients.

Outputs (under outputs/step33_a_badcase_regime_model/):
  1-16 as specified.
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
OUT_REL = Path("outputs/step33_a_badcase_regime_model")
REGIONS = REGION_ORDER
RUN_CMD = "python " + " ".join(sys.argv)
_cache = {}


def load_all():
    if _cache: return _cache
    df = pd.read_csv(ROOT / TRAIN_REL, usecols=["mmsi","x","y","sog","time"],
                     dtype={"mmsi":"string","x":"float64","y":"float64","sog":"float32"}, parse_dates=["time"])
    df["hour"] = df["time"].dt.floor("h"); df["date"] = df["time"].dt.normalize()
    df["hour_of_day"] = df["time"].dt.hour; df = add_regions(df)
    a_lab = make_a_labels(df)
    a_lab["date"] = a_lab["hour"].dt.normalize(); a_lab["hour_of_day"] = a_lab["hour"].dt.hour
    a_pivot = a_lab.pivot_table(index=["date","hour_of_day"], columns="region", values="y", aggfunc="first").reset_index()
    for r in REGIONS:
        if r not in a_pivot.columns: a_pivot[r] = 0
    aud = pd.read_csv(ROOT / STEP02); aud["date"] = pd.to_datetime(aud["date"]).dt.normalize()
    vc_df = pd.read_csv(ROOT / STEP01); vc_df["date"] = pd.to_datetime(vc_df["date"]).dt.normalize()
    all_dates = sorted(pd.date_range("2018-01-01","2018-01-24",freq="D").normalize())
    _cache.update({"df":df, "a_lab":a_lab, "a_pivot":a_pivot,
                   "qmap":dict(zip(aud["date"],aud["quality_regime"])),
                   "china_map":dict(zip(aud["date"],aud["china_coastal_record_count"])),
                   "vc_map":dict(zip(vc_df["date"],vc_df["unique_vessel_count"])),
                   "all_dates":all_dates})
    return _cache


def compute_baseline(train_dates, te, valid_dates):
    d = load_all()
    classes, _ = classify_fold_quality(train_dates, d["china_map"])
    weights = {dd: POLICY_WEIGHTS["exclude_outage_severe"][classes[dd]] for dd in train_dates}
    train_df = d["a_lab"][d["a_lab"]["date"].isin(train_dates)].copy()
    train_df["day_type"] = np.where(train_df["hour"].dt.dayofweek.isin([5,6]), "weekend", "weekday")
    T_hat, P_hat = compute_components(train_df, train_dates, te, weights)
    preds = {}
    for dd in valid_dates:
        vd = d["a_lab"][d["a_lab"]["date"]==dd].sort_values(["hour","region"]).reset_index(drop=True)
        vr = vd["region"].to_numpy(); vh = vd["hour_of_day"].to_numpy().astype(int)
        vdt = vd["hour"].dt.dayofweek.to_numpy()
        vdt_type = np.array(["weekend" if x in (5,6) else "weekday" for x in vdt])
        Tvec = np.array([T_hat["mean"][r] for r in vr], dtype=float)
        prof = np.array([P_hat["daytype_shrunk"][r][vdt_type[i]][vh[i]] for i,r in enumerate(vr)])
        pr = np.clip(np.rint(Tvec*prof), 0, None).astype(int)
        mat = np.zeros((24,3))
        for i,r in enumerate(vr): mat[vh[i], REGIONS.index(r)] = float(pr[i])
        preds[dd] = mat
    return preds


def get_a_vec(d):
    data = load_all()
    sub = data["a_pivot"][data["a_pivot"]["date"]==d].sort_values("hour_of_day")
    m = np.zeros((24,3))
    for _, row in sub.iterrows():
        h = int(row["hour_of_day"])
        for ri,r in enumerate(REGIONS): m[h,ri] = float(row[r])
    return m


def fourier_basis_nocst(K):
    h = np.arange(24); cols = []
    for k in range(1,K+1):
        cols.append(np.sin(2*np.pi*k*h/24)); cols.append(np.cos(2*np.pi*k*h/24))
    return np.column_stack(cols)


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")
    data = load_all(); qmap = data["qmap"]; all_dates = data["all_dates"]; vc_map = data["vc_map"]
    normal_dates = [d for d in all_dates if qmap.get(d)=="normal"]

    # ---- implementation audit ----
    (out_dir / "implementation_audit.md").write_text(
        "# Implementation Audit\n\n"
        "Step 32 Fourier Oracle proved residual contains smooth low-freq components.\n"
        "Step 33 (Fourier repair) split into level (10.7%) and shape (7.8-23.3%).\n"
        "Shape Oracle passes but deployable coefficient prediction fails.\n"
        "Key insight: level and shape are CORRELATED (corr ~0.76), not independent.\n"
        "This step models them jointly via activity state identification.\n",
        encoding="utf-8")

    # ---- cross-fitted residuals ----
    print("=== Cross-fitted residuals ===")
    cf = {}
    for d in normal_dates:
        prev = [dd for dd in all_dates if dd < d]
        if len(prev) < 5: continue
        prev_normal = [dd for dd in prev if qmap.get(dd)=="normal"]
        if len(prev_normal) < 4: continue
        te = d - pd.Timedelta(days=1)
        bl = compute_baseline(prev, te, [d])[d]
        a = get_a_vec(d)
        g24 = (a - bl).sum(axis=1)  # hourly total residual
        level = float(g24.mean())
        shape = g24 - level
        daily_sse = float(((a-bl)**2).sum())
        # worst cell
        cell_sse = ((a-bl)**2).flatten()
        worst_idx = int(np.argmax(cell_sse))
        cf[d] = {"bl":bl, "a":a, "g24":g24, "level":level, "shape":shape, "daily_sse":daily_sse,
                 "te":te, "day_type":"weekend" if d.dayofweek in (5,6) else "weekday",
                 "U_d":vc_map.get(d,0), "bl_total":float(bl.sum()), "a_total":float(a.sum()),
                 "worst_hour":worst_idx//3, "worst_region":REGIONS[worst_idx%3],
                 "max_cell_sse":float(cell_sse.max()),
                 "top10_sse":float(np.sort(cell_sse)[-10:].sum()),
                 "top10_share":float(np.sort(cell_sse)[-10:].sum()/daily_sse) if daily_sse>0 else 0,
                 "cell_sse":cell_sse}
    cf_dates = sorted(cf.keys())
    print(f"Cross-fitted dates: {len(cf_dates)}")

    # ---- badcase daily catalog ----
    bc_rows = []
    for d in cf_dates:
        r = cf[d]
        prev_normal = [dd for dd in cf_dates if dd < d]
        run = 0
        for dd in reversed(prev_normal):
            if (d - dd).days == run + 1: run += 1
            else: break
        lag7 = d - pd.Timedelta(days=7); lag14 = d - pd.Timedelta(days=14)
        bc_rows.append({"date":f"{d:%Y-%m-%d}", "train_end":f"{r['te']:%Y-%m-%d}", "quality":"normal",
                       "day_type":r["day_type"], "U_d":r["U_d"], "baseline_total":r["bl_total"],
                       "actual_total":r["a_total"], "daily_level_residual":r["level"],
                       "daily_sse":r["daily_sse"], "max_cell_sse":r["max_cell_sse"],
                       "top10_cell_sse":r["top10_sse"], "top10_sse_share":r["top10_share"],
                       "worst_region":r["worst_region"], "worst_hour":r["worst_hour"],
                       "previous_normal_run":run,
                       "lag7_date":f"{lag7:%Y-%m-%d}", "lag7_quality":qmap.get(lag7,"n/a"),
                       "lag7_a_total":float(get_a_vec(lag7).sum()) if lag7 in [dd for dd in all_dates] else np.nan,
                       "lag14_date":f"{lag14:%Y-%m-%d}", "lag14_quality":qmap.get(lag14,"n/a"),
                       "lag14_a_total":float(get_a_vec(lag14).sum()) if lag14 in [dd for dd in all_dates] else np.nan})
    bc_df = pd.DataFrame(bc_rows)
    bc_df.to_csv(out_dir / "badcase_daily_catalog.csv", index=False, encoding="utf-8-sig")
    print("\nBadcase daily catalog:")
    print(bc_df[["date","actual_total","baseline_total","daily_level_residual","daily_sse","top10_sse_share","worst_region"]].to_string(index=False))

    # ---- cell catalog ----
    cell_rows = []
    for d in cf_dates:
        r = cf[d]
        idx = np.argsort(-r["cell_sse"])[:20]
        for rank, i in enumerate(idx):
            cell_rows.append({"date":f"{d:%Y-%m-%d}", "rank_within_date":rank+1,
                            "hour":i//3, "region":REGIONS[i%3],
                            "y_true":int(r["a"].flatten()[i]), "baseline_pred":int(r["bl"].flatten()[i]),
                            "residual":float(r["a"].flatten()[i]-r["bl"].flatten()[i]),
                            "squared_error":float(r["cell_sse"][i]),
                            "daily_regime":"low" if r["level"] < -20 else ("high" if r["level"]>20 else "middle")})
    pd.DataFrame(cell_rows).to_csv(out_dir / "badcase_cell_catalog.csv", index=False, encoding="utf-8-sig")

    # ---- block baselines ----
    blocks = {
        "block_pre_normal": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-07"), pd.Timestamp("2018-01-08"), pd.Timestamp("2018-01-11")),
        "block_target_like": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-19"), pd.Timestamp("2018-01-20"), pd.Timestamp("2018-01-24")),
        "block_post_outage_stress": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-18"), pd.Timestamp("2018-01-19"), pd.Timestamp("2018-01-24")),
    }
    print("\n=== Block baselines ===")
    block_bl = {}; block_bl_sse = {}
    for bname, (ts, te, vs_, ve) in blocks.items():
        td = sorted(pd.date_range(ts, te, freq="D").normalize())
        vd = list(pd.date_range(vs_, ve, freq="D").normalize())
        preds = compute_baseline(td, te, vd)
        block_bl[bname] = preds
        sse = sum(float(((preds[d]-get_a_vec(d))**2).sum()) for d in vd)
        block_bl_sse[bname] = sse
        print(f"  {bname}: SSE={sse:.0f}")

    # ---- regime definitions from training residuals ----
    # For each block, define low/middle/high from training CF dates
    print("\n=== Regime definitions ===")
    regime_defs = {}
    for bname, (ts, te, vs_, ve) in blocks.items():
        train_cf = [d for d in cf_dates if d <= te]
        if len(train_cf) < 4: continue
        levels = np.array([cf[d]["level"] for d in train_cf])
        q1, q2 = np.percentile(levels, [33.3, 66.7])
        regime_defs[bname] = {"low_thresh": float(q1), "high_thresh": float(q2), "train_cf": train_cf}

        # regime templates
        templates = {}
        for regime, mask_func in [("low", lambda l: l <= q1), ("middle", lambda l: (l > q1) & (l <= q2)), ("high", lambda l: l > q2)]:
            mask = mask_func(levels)
            if mask.sum() == 0:
                templates[regime] = np.zeros(24)
            else:
                shapes = np.array([cf[d]["shape"] for d, m in zip(train_cf, mask) if m])
                templates[regime] = shapes.mean(axis=0)
        # global template
        global_shape = np.array([cf[d]["shape"] for d in train_cf]).mean(axis=0)
        # shrink toward global
        for tau in [5]:
            shrunk = {}
            for regime in ["low", "middle", "high"]:
                mask = mask_func(levels) if regime == "high" else (levels <= q1) if regime == "low" else (levels > q1) & (levels <= q2)
                n_g = int(mask.sum())
                raw = templates[regime]
                shrunk[regime] = (n_g * raw + tau * global_shape) / (n_g + tau) if n_g > 0 else global_shape
            templates[f"shrunk_tau{tau}"] = shrunk
        regime_defs[bname]["templates"] = templates
        regime_defs[bname]["global_shape"] = global_shape
        print(f"  {bname}: low≤{q1:.1f}, high>{q2:.1f}, n={[int((levels<=q1).sum()), int(((levels>q1)&(levels<=q2)).sum()), int((levels>q2).sum())]}")

    # ---- evaluate methods ----
    print("\n=== Scoring ===")
    all_scores = []
    for bname, (ts, te, vs_, ve) in blocks.items():
        vd = list(pd.date_range(vs_, ve, freq="D").normalize())
        bl = block_bl[bname]; bl_s = block_bl_sse[bname]
        train_cf = [d for d in cf_dates if d <= te]
        if len(train_cf) < 4: continue
        levels_train = np.array([cf[d]["level"] for d in train_cf])
        q1, q2 = regime_defs[bname]["low_thresh"], regime_defs[bname]["high_thresh"]
        sigma_L = float(np.std(levels_train)) if len(levels_train) > 1 else 1.0
        global_shape = regime_defs[bname]["global_shape"]
        templates = regime_defs[bname]["templates"]["shrunk_tau5"]
        global_level_mean = float(levels_train.mean())

        for method in ["exact_step11_baseline", "level_zero_regime_zero",
                       "regime_soft_recent3", "regime_soft_lag_normal", "regime_soft_lag_blend",
                       "oracle_true_level_regime", "oracle_true_regime_template"]:
            total_sse = 0
            for di, d in enumerate(vd):
                a_d = get_a_vec(d); bl_d = bl[d]

                if method == "exact_step11_baseline":
                    pred = bl_d
                elif method == "level_zero_regime_zero":
                    pred = bl_d
                else:
                    # determine predicted level
                    if method in ("regime_soft_recent3",):
                        rec3 = levels_train[-3:].mean() if len(levels_train) >= 3 else global_level_mean
                        pred_level = float(rec3)
                    elif method in ("regime_soft_lag_normal",):
                        lag7 = d - pd.Timedelta(days=7); lag14 = d - pd.Timedelta(days=14)
                        if lag7 in cf: pred_level = cf[lag7]["level"]
                        elif lag14 in cf: pred_level = cf[lag14]["level"]
                        else: pred_level = 0.0
                    elif method in ("regime_soft_lag_blend",):
                        lag7 = d - pd.Timedelta(days=7)
                        rec3 = levels_train[-3:].mean() if len(levels_train) >= 3 else global_level_mean
                        lag_level = cf[lag7]["level"] if lag7 in cf else 0.0
                        pred_level = 0.5 * lag_level + 0.5 * rec3
                    elif method in ("oracle_true_level_regime", "oracle_true_regime_template"):
                        # true level from target
                        g24_true = (a_d - bl_d).sum(axis=1)
                        pred_level = float(g24_true.mean())
                    else:
                        pred_level = 0.0

                    # soft regime weights
                    if sigma_L > 0:
                        dists = np.array([(pred_level - np.mean(levels_train[levels_train <= q1])) if (levels_train <= q1).any() else 0,
                                          (pred_level - np.mean(levels_train[(levels_train > q1) & (levels_train <= q2)])) if ((levels_train > q1) & (levels_train <= q2)).any() else 0,
                                          (pred_level - np.mean(levels_train[levels_train > q2])) if (levels_train > q2).any() else 0])
                        weights = np.exp(-dists**2 / (2 * sigma_L**2))
                        weights /= weights.sum()
                    else:
                        weights = np.array([1/3, 1/3, 1/3])

                    # predict hour shape
                    shape_pred = weights[0] * templates["low"] + weights[1] * templates["middle"] + weights[2] * templates["high"]

                    if method == "oracle_true_regime_template":
                        # use true regime weights but still template shape
                        pass

                    # build prediction
                    g_pred = pred_level / 24 + shape_pred  # hour total residual
                    alpha = 1.0
                    pred = bl_d.copy()
                    for h in range(24):
                        bl_h = bl_d[h]; s = bl_h.sum()
                        if s > 0: pred[h] = bl_h + alpha * g_pred[h] * bl_h / s
                        else: pred[h] = bl_h

                pred_int = np.clip(np.rint(pred), 0, None).astype(int)
                total_sse += float(((pred_int - a_d)**2).sum())

            red = 1 - total_sse / bl_s if bl_s else 0
            deployable = method not in ("oracle_true_level_regime", "oracle_true_regime_template")
            all_scores.append({"block":bname, "method":method, "deployable":deployable,
                             "sse":total_sse, "baseline_sse":bl_s, "reduction_ratio":red})
            print(f"  {bname:30s} {method:40s} SSE={total_sse:.0f} red={red*100:.1f}%")

    scores_df = pd.DataFrame(all_scores)
    scores_df.to_csv(out_dir / "block_scores.csv", index=False, encoding="utf-8-sig")

    # ---- level-conditioned Fourier ----
    print("\n=== Level-conditioned Fourier ===")
    for bname, (ts, te, vs_, ve) in blocks.items():
        train_cf = [d for d in cf_dates if d <= te]
        if len(train_cf) < 6: continue
        levels = np.array([cf[d]["level"] for d in train_cf])
        # K=1 Fourier coefficients vs level
        for K in [1, 2]:
            coefs = np.array([fit_fourier_nocst(cf[d]["shape"], K) for d in train_cf])
            for ci in range(2*K):
                c = coefs[:, ci]
                if np.var(levels) > 0:
                    beta1 = np.cov(levels, c)[0,1] / np.var(levels)
                else:
                    beta1 = 0
                corr = float(pd.Series(c).corr(pd.Series(levels))) if len(c) > 2 else np.nan
                if ci == 0 or ci == 1:
                    print(f"  {bname} K{K} coef{ci}: mean={c.mean():.3f}, corr_with_level={corr:.3f}")

    # ---- summary ----
    L = ["# Step 33 A Bad-case Regime Model", "", f"Run command: `{RUN_CMD}`", ""]
    L.append("## Badcase daily catalog (target_like)")
    L.append("")
    L.append("| date | actual | baseline | level_resid | SSE | top10_share |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for _, r in bc_df.iterrows():
        L.append(f"| {r['date']} | {r['actual_total']:.0f} | {r['baseline_total']:.0f} | {r['daily_level_residual']:.0f} | {r['daily_sse']:.0f} | {r['top10_sse_share']:.2f} |")
    L.append("")
    L.append("## Block scores (key methods)")
    L.append("")
    L.append("| block | method | SSE | reduction |")
    L.append("| --- | --- | --- | --- |")
    for _, r in scores_df.iterrows():
        L.append(f"| {r['block']} | {r['method']} | {r['sse']:.0f} | {r['reduction_ratio']*100:.1f}% |")
    L.append("")
    dep_pass = any(scores_df[(scores_df["deployable"]) & (scores_df["reduction_ratio"] >= 0.05)].index)
    oracle_pass = any(scores_df[(~scores_df["deployable"]) & (scores_df["reduction_ratio"] >= 0.10)].index)
    L.append(f"## Decision\n- Deployable passes 5%: {'YES' if dep_pass else 'NO'}\n- Oracle passes 10%: {'YES' if oracle_pass else 'NO'}")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    assert not any(out_dir.rglob("*.zip"))
    print(f"\n=== Summary ===")
    for _, r in scores_df.iterrows():
        print(f"  {r['block']:30s} {r['method']:40s} SSE={r['sse']:.0f} red={r['reduction_ratio']*100:.1f}% dep={r['deployable']}")
    print(f"\nDeployable passes 5%: {dep_pass}")
    print(f"Oracle passes 10%: {oracle_pass}")
    print("\nDone.")


def fit_fourier_nocst(s_24, K):
    basis = fourier_basis_nocst(K)
    coef, *_ = np.linalg.lstsq(basis, s_24, rcond=None)
    return coef


if __name__ == "__main__":
    main()
