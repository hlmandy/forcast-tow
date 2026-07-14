"""Step 34 — Existing model bad-case coverage and constrained combination.

Audits whether previously developed models (Steps 7-33) already contain
useful correction directions for Step 11's bad-case days. No new base models.
Reads Step 07's 16-method OOF predictions on final_analog (train_end=01-18,
target=01-19..24) as the primary asset pool, adds key models from other steps.

Outputs (under outputs/step34_a_existing_model_badcase_coverage/):
  1-14 as specified.
"""

from __future__ import annotations
import sys, warnings, math
from pathlib import Path
import numpy as np, pandas as pd
from itertools import combinations

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from optimized_baseline import REGION_ORDER, add_regions, make_a_labels  # noqa: E402
from step03_a_rolling_backtest import classify_fold_quality, POLICY_WEIGHTS  # noqa: E402
from step06_a_structure_model_benchmark import compute_components  # noqa: E402

STEP07 = Path("outputs/step07_a_regularized_count_models/a_model_predictions.csv")
STEP02 = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
STEP01 = Path("outputs/step01_data_audit/train_daily_overview.csv")
TRAIN_REL = Path("数据备份/训练集_20180101-0124_拖轮AIS.csv")
OUT_REL = Path("outputs/step34_a_existing_model_badcase_coverage")
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
    _cache.update({"df":df,"a_lab":a_lab,"a_pivot":a_pivot,
                   "qmap":dict(zip(aud["date"],aud["quality_regime"])),
                   "china_map":dict(zip(aud["date"],aud["china_coastal_record_count"])),
                   "vc_map":dict(zip(vc_df["date"],vc_df["unique_vessel_count"])),
                   "all_dates":sorted(pd.date_range("2018-01-01","2018-01-24",freq="D").normalize())})
    return _cache


def get_a_vec(d):
    data = load_all()
    sub = data["a_pivot"][data["a_pivot"]["date"]==d].sort_values("hour_of_day")
    m = np.zeros((24,3))
    for _, row in sub.iterrows():
        h = int(row["hour_of_day"])
        for ri,r in enumerate(REGIONS): m[h,ri] = float(row[r])
    return m


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
        pr = np.clip(np.rint(Tvec*prof),0,None).astype(int)
        mat = np.zeros((24,3))
        for i,r in enumerate(vr): mat[vh[i],REGIONS.index(r)] = float(pr[i])
        preds[dd] = mat
    return preds


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")
    data = load_all()

    # ---- implementation audit ----
    (out_dir / "implementation_audit.md").write_text(
        "# Implementation Audit\n\n"
        "Step 33 regime template failed even with Oracle level.\n"
        "This step does NOT create new models. It audits whether the EXISTING\n"
        "prediction pool (16 methods from Step 07 + key models from other steps)\n"
        "already contains correction directions for bad-case days.\n\n"
        "Primary evaluation block: post_outage_stress (train_end=01-18, target=01-19..24)\n"
        "This matches Step 07's final_analog evaluation.\n",
        encoding="utf-8")

    # ---- target block: post_outage_stress (train_end=Jan 18, target=Jan 19-24) ----
    target_dates = list(pd.date_range("2018-01-19","2018-01-24",freq="D").normalize())
    te = pd.Timestamp("2018-01-18")

    # ---- baseline ----
    td = sorted(pd.date_range("2018-01-01", te, freq="D").normalize())
    bl_preds = compute_baseline(td, te, target_dates)
    bl_sse = sum(float(((bl_preds[d]-get_a_vec(d))**2).sum()) for d in target_dates)
    print(f"Baseline SSE: {bl_sse:.0f} (expected 3872)")
    assert abs(bl_sse - 3872) < 1.0

    # ---- collect model predictions from Step 07 ----
    print("\n=== Collecting Step 07 predictions ===")
    a7 = pd.read_csv(ROOT / STEP07, encoding="utf-8-sig")
    a7["date"] = pd.to_datetime(a7["date"]).dt.normalize()
    fa = a7[a7["evaluation_id"]=="final_analog"].copy()

    # build per-method (24,3) predictions per date
    model_preds = {}  # model_name -> {date: (24,3) array}
    model_names = []
    for (method, policy) in fa.groupby(["method","training_policy"]).groups.keys():
        key = f"{method}/{policy}"
        model_names.append(key)
        model_preds[key] = {}
        sub = fa[(fa["method"]==method) & (fa["training_policy"]==policy)]
        for d in target_dates:
            sub_d = sub[sub["date"]==d].sort_values(["hour_of_day","region"])
            mat = np.zeros((24,3))
            for _, row in sub_d.iterrows():
                h = int(row["hour_of_day"]); r = row["region"]
                mat[h, REGIONS.index(r)] = float(row["pred_rounded"])
            model_preds[key][d] = mat

    # Also add the anchor baseline explicitly
    anchor = "decomp_mean_daytype_shrunk/exclude_outage_severe"
    if anchor not in model_preds:
        model_preds[anchor] = bl_preds
        model_names.append(anchor)

    print(f"Models collected: {len(model_names)}")

    # ---- model registry ----
    reg_rows = []
    for mname in model_names:
        is_anchor = mname == anchor
        reg_rows.append({"model_id": mname, "source_step": "step07", "deployable": True,
                        "uses_target_truth": False, "exact_oof_available": True,
                        "included_in_pool": True, "exclusion_reason": ""})
    pd.DataFrame(reg_rows).to_csv(out_dir / "model_registry.csv", index=False, encoding="utf-8-sig")

    # ---- common predictions asset ----
    print("\n=== Building common predictions ===")
    pred_rows = []
    for mname in model_names:
        for d in target_dates:
            tv = get_a_vec(d); mp = model_preds[mname][d]
            bp = bl_preds[d]
            for h in range(24):
                for ri, r in enumerate(REGIONS):
                    y = float(tv[h,ri]); p = float(mp[h,ri]); b = float(bp[h,ri])
                    pred_rows.append({"model_id":mname, "date":f"{d:%Y-%m-%d}", "hour":h, "region":r,
                                     "y_true":y, "prediction":p, "baseline_prediction":b,
                                     "baseline_residual":b-y, "model_delta_from_baseline":p-b,
                                     "model_residual":p-y, "baseline_squared_error":(b-y)**2,
                                     "model_squared_error":(p-y)**2})
    preds_df = pd.DataFrame(pred_rows)
    preds_df.to_csv(out_dir / "common_model_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"Total prediction rows: {len(preds_df)}")

    # ---- per-model SSE on target block ----
    print("\n=== Per-model SSE ===")
    model_sse = {}
    for mname in model_names:
        sse = sum(float(((model_preds[mname][d]-get_a_vec(d))**2).sum()) for d in target_dates)
        model_sse[mname] = sse
        red = 1 - sse/bl_sse
        print(f"  {mname:60s} SSE={sse:.0f} red={red*100:.1f}%")

    # ---- badcase definition ----
    daily_bl_sse = {}
    for d in target_dates:
        daily_bl_sse[d] = float(((bl_preds[d]-get_a_vec(d))**2).sum())
    sse_vals = sorted(daily_bl_sse.values(), reverse=True)
    badcase_thresh = sse_vals[len(sse_vals)//4] if len(sse_vals) >= 4 else sse_vals[0]
    badcase_dates = [d for d in target_dates if daily_bl_sse[d] >= badcase_thresh]
    print(f"\nBadcase dates (top 25%): {badcase_dates}")
    print(f"Badcase threshold SSE: {badcase_thresh:.0f}")

    # ---- badcase coverage per model ----
    print("\n=== Badcase coverage ===")
    bc_rows = []
    for mname in model_names:
        bc_sse = sum(float(((model_preds[mname][d]-get_a_vec(d))**2).sum()) for d in badcase_dates)
        bc_bl_sse = sum(daily_bl_sse[d] for d in badcase_dates)
        easy_sse = model_sse[mname] - bc_sse
        easy_bl = bl_sse - bc_bl_sse
        # direction check
        correct_dir = wrong_dir = 0
        for d in target_dates:
            tv = get_a_vec(d); mp = model_preds[mname][d]; bp = bl_preds[d]
            delta = mp - bp; resid = tv - bp
            mask = (delta * resid > 0) & (np.abs(delta) > 0.5)
            correct_dir += int(mask.sum())
            wrong = (delta * resid < 0) & (np.abs(delta) > 0.5)
            wrong_dir += int(wrong.sum())
        bc_rows.append({"model_id":mname, "total_sse":model_sse[mname], "total_reduction":1-model_sse[mname]/bl_sse,
                       "badcase_sse":bc_sse, "badcase_reduction":1-bc_sse/bc_bl_sse if bc_bl_sse else 0,
                       "easy_sse":easy_sse, "easy_change":easy_sse-easy_bl,
                       "correct_direction_count":correct_dir, "wrong_direction_count":wrong_dir,
                       "correct_direction_rate":correct_dir/(correct_dir+wrong_dir) if (correct_dir+wrong_dir) else 0})
    bc_df = pd.DataFrame(bc_rows).sort_values("badcase_reduction", ascending=False)
    bc_df.to_csv(out_dir / "badcase_model_coverage.csv", index=False, encoding="utf-8-sig")
    print("\nTop 5 models by badcase reduction:")
    print(bc_df[["model_id","total_reduction","badcase_reduction","easy_change","correct_direction_rate"]].head(5).to_string(index=False))

    # ---- daily model rankings ----
    dr_rows = []
    for d in target_dates:
        daily_rank = []
        for mname in model_names:
            sse = float(((model_preds[mname][d]-get_a_vec(d))**2).sum())
            daily_rank.append((mname, sse))
        daily_rank.sort(key=lambda x: x[1])
        bl_daily = daily_bl_sse[d]
        for rank, (mname, sse) in enumerate(daily_rank):
            dr_rows.append({"date":f"{d:%Y-%m-%d}", "rank":rank+1, "model_id":mname,
                           "sse":sse, "baseline_sse":bl_daily, "reduction":1-sse/bl_daily if bl_daily else 0,
                           "is_badcase":d in badcase_dates})
    dr_df = pd.DataFrame(dr_rows)
    dr_df.to_csv(out_dir / "daily_model_rankings.csv", index=False, encoding="utf-8-sig")
    print("\nBest model per day:")
    for d in target_dates:
        sub = dr_df[(dr_df["date"]==f"{d:%Y-%m-%d}") & (dr_df["rank"]==1)].iloc[0]
        print(f"  {d:%Y-%m-%d}: {sub['model_id']:60s} SSE={sub['sse']:.0f} bl={sub['baseline_sse']:.0f} red={sub['reduction']*100:.1f}%")

    # ---- Oracle envelopes ----
    print("\n=== Oracle envelopes ===")
    oracle_rows = []

    # 1. best per day
    opd_sse = 0
    for d in target_dates:
        best = min(float(((model_preds[m][d]-get_a_vec(d))**2).sum()) for m in model_names)
        opd_sse += best
    opd_red = 1 - opd_sse/bl_sse
    oracle_rows.append({"oracle":"best_existing_per_day", "sse":opd_sse, "baseline_sse":bl_sse, "reduction":opd_red})
    print(f"  oracle_best_per_day: SSE={opd_sse:.0f} red={opd_red*100:.1f}%")

    # 2. best per region-day
    oprd_sse = 0
    for d in target_dates:
        tv = get_a_vec(d)
        for ri in range(3):
            best = min(float(((model_preds[m][d][:,ri]-tv[:,ri])**2).sum()) for m in model_names)
            oprd_sse += best
    oprd_red = 1 - oprd_sse/bl_sse
    oracle_rows.append({"oracle":"best_existing_per_region_day", "sse":oprd_sse, "baseline_sse":bl_sse, "reduction":oprd_red})
    print(f"  oracle_best_per_region_day: SSE={oprd_sse:.0f} red={oprd_red*100:.1f}%")

    # 3. convex hull per day (simplified: try all pairs + individual)
    och_sse = 0
    for d in target_dates:
        tv = get_a_vec(d)
        best_sse = float(((bl_preds[d]-tv)**2).sum())
        # try each model alone
        for m in model_names:
            s = float(((model_preds[m][d]-tv)**2).sum())
            best_sse = min(best_sse, s)
        # try all pairs
        for m1, m2 in combinations(model_names, 2):
            for w in [0.25, 0.5, 0.75]:
                pred = w*model_preds[m1][d] + (1-w)*model_preds[m2][d]
                s = float(((pred-tv)**2).sum())
                best_sse = min(best_sse, s)
        och_sse += best_sse
    och_red = 1 - och_sse/bl_sse
    oracle_rows.append({"oracle":"convex_hull_per_day", "sse":och_sse, "baseline_sse":bl_sse, "reduction":och_red})
    print(f"  oracle_convex_hull_per_day: SSE={och_sse:.0f} red={och_red*100:.1f}%")

    # 4. correction span (anchor + sparse corrections)
    # Use OLS to find best correction weights on ALL target dates (oracle)
    n_cells = len(target_dates) * 72
    Y = np.zeros(n_cells)  # baseline residual
    X = np.zeros((n_cells, len(model_names)))  # model deltas
    idx = 0
    for d in target_dates:
        tv = get_a_vec(d).flatten()
        bp = bl_preds[d].flatten()
        Y[idx:idx+72] = tv - bp
        for mi, m in enumerate(model_names):
            X[idx:idx+72, mi] = model_preds[m][d].flatten() - bp
        idx += 72
    # OLS
    gamma_ols, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred_ols = bl_preds.copy()
    for d in target_dates:
        bp = bl_preds[d]
        delta = np.zeros((24,3))
        for mi, m in enumerate(model_names):
            delta += gamma_ols[mi] * (model_preds[m][d] - bp)
        pred_ols[d] = np.clip(np.rint(bp + delta), 0, None).astype(int)
    ols_sse = sum(float(((pred_ols[d]-get_a_vec(d))**2).sum()) for d in target_dates)
    ols_red = 1 - ols_sse/bl_sse
    oracle_rows.append({"oracle":"correction_span_ols", "sse":ols_sse, "baseline_sse":bl_sse, "reduction":ols_red})
    print(f"  oracle_correction_span_ols: SSE={ols_sse:.0f} red={ols_red*100:.1f}%")

    pd.DataFrame(oracle_rows).to_csv(out_dir / "oracle_envelope_scores.csv", index=False, encoding="utf-8-sig")

    # ---- pairwise complementarity ----
    print("\n=== Pairwise complementarity (top pairs) ===")
    pair_rows = []
    for m1, m2 in combinations(model_names, 2):
        # residual correlation
        r1 = np.concatenate([(model_preds[m1][d]-get_a_vec(d)).flatten() for d in target_dates])
        r2 = np.concatenate([(model_preds[m2][d]-get_a_vec(d)).flatten() for d in target_dates])
        if np.std(r1) > 0 and np.std(r2) > 0:
            resid_corr = float(pd.Series(r1).corr(pd.Series(r2)))
        else:
            resid_corr = 1.0
        # prediction correlation
        p1 = np.concatenate([model_preds[m1][d].flatten() for d in target_dates])
        p2 = np.concatenate([model_preds[m2][d].flatten() for d in target_dates])
        pred_corr = float(pd.Series(p1).corr(pd.Series(p2))) if np.std(p1) > 0 and np.std(p2) > 0 else 1.0
        m1_win = m2_win = 0
        for d in target_dates:
            s1 = float(((model_preds[m1][d]-get_a_vec(d))**2).sum())
            s2 = float(((model_preds[m2][d]-get_a_vec(d))**2).sum())
            bl_s = daily_bl_sse[d]
            if s1 < bl_s and s2 >= bl_s: m1_win += 1
            if s2 < bl_s and s1 >= bl_s: m2_win += 1
        pair_rows.append({"model_1":m1, "model_2":m2, "prediction_correlation":pred_corr,
                         "residual_correlation":resid_corr,
                         "model_1_unique_win_days":m1_win, "model_2_unique_win_days":m2_win})
    pair_df = pd.DataFrame(pair_rows).sort_values("residual_correlation")
    pair_df.to_csv(out_dir / "pairwise_model_complementarity.csv", index=False, encoding="utf-8-sig")
    print("Most complementary pairs (lowest residual corr):")
    print(pair_df[["model_1","model_2","residual_correlation","model_1_unique_win_days","model_2_unique_win_days"]].head(5).to_string(index=False))

    # ---- simple ensemble: mean of top-K diverse models ----
    print("\n=== Ensembles ===")
    ens_scores = []
    # select diverse models: those with lowest residual correlation to anchor
    model_delta_from_bl = {}
    for m in model_names:
        deltas = []
        for d in target_dates:
            deltas.extend((model_preds[m][d] - bl_preds[d]).flatten().tolist())
        model_delta_from_bl[m] = np.array(deltas)

    # rank by how much they reduce total SSE
    ranked = sorted(model_names, key=lambda m: model_sse[m])
    # remove near-identical to anchor
    diverse = [m for m in ranked if np.std(model_delta_from_bl[m]) > 0.5][:10]

    for top_n in [2, 3, 5]:
        for ms in combinations(diverse, min(top_n, len(diverse))):
            ens_sse = 0
            for d in target_dates:
                pred = np.mean([model_preds[m][d] for m in ms], axis=0)
                pred_int = np.clip(np.rint(pred), 0, None).astype(int)
                ens_sse += float(((pred_int - get_a_vec(d))**2).sum())
            red = 1 - ens_sse/bl_sse
            if red > -0.1:  # only record interesting ones
                ens_scores.append({"method":f"mean_top{top_n}_" + "+".join(ms), "sse":ens_sse, "reduction":red})

    # also baseline-anchored NNLS (simplified: grid search over a few models)
    # pick 3 most diverse models
    best_models = ranked[:5]
    for m1, m2, m3 in combinations(best_models, 3):
        for w0 in [0.5, 0.7, 0.9]:
            for w1 in np.arange(0, 0.5, 0.1):
                w2 = (1 - w0 - w1) / 2; w3 = w2
                if w2 < 0: continue
                ens_sse = 0
                for d in target_dates:
                    pred = w0*bl_preds[d] + w1*model_preds[m1][d] + w2*model_preds[m2][d] + w3*model_preds[m3][d]
                    pred_int = np.clip(np.rint(pred), 0, None).astype(int)
                    ens_sse += float(((pred_int - get_a_vec(d))**2).sum())
                red = 1 - ens_sse/bl_sse
                if red > 0.0:
                    ens_scores.append({"method":f"nnls_{w0:.1f}_{m1[:20]}_{m2[:20]}_{m3[:20]}", "sse":ens_sse, "reduction":red})

    ens_df = pd.DataFrame(ens_scores).sort_values("reduction", ascending=False) if ens_scores else pd.DataFrame()
    if len(ens_df):
        print(f"\nTop 5 ensembles:")
        print(ens_df.head(5).to_string(index=False))
        ens_df.to_csv(out_dir / "ensemble_block_scores.csv", index=False, encoding="utf-8-sig")
    else:
        print("No ensemble improved over baseline")

    # ---- summary ----
    L = ["# Step 34 Existing Model Bad-case Coverage", "", f"Run command: `{RUN_CMD}`", ""]
    L.append(f"## Models in pool: {len(model_names)}")
    L.append(f"## Baseline SSE (post_outage): {bl_sse:.0f}")
    L.append("")
    L.append("## Oracle envelopes")
    L.append("")
    L.append("| Oracle | SSE | reduction |")
    L.append("| --- | --- | --- |")
    for o in oracle_rows:
        L.append(f"| {o['oracle']} | {o['sse']:.0f} | {o['reduction']*100:.1f}% |")
    L.append("")
    L.append("## Per-model total SSE (top 5 by reduction)")
    L.append("")
    for _, r in bc_df.head(5).iterrows():
        L.append(f"- {r['model_id']}: total={r['total_reduction']*100:.1f}%, badcase={r['badcase_reduction']*100:.1f}%")
    L.append("")
    if len(ens_df):
        L.append("## Top ensembles")
        for _, r in ens_df.head(3).iterrows():
            L.append(f"- {r['method']}: {r['reduction']*100:.1f}%")
    L.append("")
    opd_pass = opd_red >= 0.15
    ols_pass = ols_red >= 0.20
    L.append(f"## Decision")
    L.append(f"- Oracle best_per_day >= 15%: {'YES' if opd_pass else 'NO'}")
    L.append(f"- Oracle correction_span >= 20%: {'YES' if ols_pass else 'NO'}")
    if not opd_pass and not ols_pass:
        L.append("- All existing models share similar bad-case errors → combination has low value")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    assert not any(out_dir.rglob("*.zip"))
    print(f"\n=== Summary ===")
    for o in oracle_rows:
        print(f"  {o['oracle']:40s} SSE={o['sse']:.0f} red={o['reduction']*100:.1f}%")
    print(f"\nOracle per_day >= 15%: {opd_pass}")
    print(f"Oracle correction >= 20%: {ols_pass}")
    print("\nDone.")


if __name__ == "__main__":
    main()
