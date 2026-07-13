"""Step 24 — A normal-regime hour-shape and nested residual calibration.

Focus on NORMAL-date evaluation (not anomaly-contaminated). Tests 5 methods +
1 diagnostic: baseline, nested region-hour unconstrained, total-preserving,
normal common-profile CV, profile+residual combo, step23 fixed-10 (diagnostic).
Dual metrics: aggregate reduction vs mean-fold reduction. No submissions.

Outputs (under outputs/step24_a_normal_regime_calibration/):
  1. method_definitions.csv (6)
  2. inner_selection.csv
  3. calibrated_predictions.csv
  4. fold_scores.csv
  5. regime_scores.csv
  6. horizon_scores.csv
  7. selected_region_hour_actions.csv
  8. profile_selection.csv
  9. decision_table.csv
  10. summary.md
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from optimized_baseline import REGION_ORDER, add_regions, make_a_labels  # noqa: E402

TRAIN_REL = Path("数据备份/训练集_20180101-0124_拖轮AIS.csv")
STEP07 = Path("outputs/step07_a_regularized_count_models/a_model_predictions.csv")
STEP01 = Path("outputs/step01_data_audit/train_daily_overview.csv")
STEP02 = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
STEP21 = Path("outputs/step21_joint_backtest_framework/joint_fold_scores.csv")
OUT_REL = Path("outputs/step24_a_normal_regime_calibration")

REGIONS = REGION_ORDER
SCENARIOS = [f"fold_{i:02d}" for i in range(1, 9)] + ["final_analog"]
ROLLING = [f"fold_{i:02d}" for i in range(1, 9)]
SCN_MAP = {}
for k in range(8):
    SCN_MAP[f"fold_{k+1:02d}"] = (pd.Timestamp("2018-01-01"), pd.Timestamp(f"2018-01-{10+k:02d}"),
                                   pd.Timestamp(f"2018-01-{11+k:02d}"), pd.Timestamp(f"2018-01-{17+k:02d}"))
SCN_MAP["final_analog"] = (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-18"),
                           pd.Timestamp("2018-01-19"), pd.Timestamp("2018-01-24"))
RUN_CMD = "python " + " ".join(sys.argv)


def smooth3(v):
    out = np.zeros(24)
    for h in range(24): out[h] = (v[(h-1)%24] + 2*v[h] + v[(h+1)%24]) / 4
    return np.clip(out, 0, None)


def lr_alloc(target, props):
    if target <= 0: return np.zeros(len(props), dtype=int)
    fp = np.asarray(props, dtype=float); s = fp.sum()
    if s <= 0: return np.zeros(len(props), dtype=int)
    fp = fp / s * target
    base = np.floor(fp).astype(int)
    rem = int(round(target)) - int(base.sum())
    if rem > 0:
        for j in np.argsort(-(fp - base))[:rem]: base[j] += 1
    elif rem < 0:
        for j in np.argsort(fp - base)[:min(-rem, (base > 0).sum())]:
            if base[j] > 0: base[j] -= 1
    return np.clip(base, 0, None)


def sse_int(p, y): return float(((p - y) ** 2).sum())


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    # ---- load baseline A predictions ----
    a7 = pd.read_csv(ROOT / STEP07, encoding="utf-8-sig")
    a7["date"] = pd.to_datetime(a7["date"]).dt.normalize()
    bp = a7[(a7["method"] == "decomp_mean_daytype_shrunk") & (a7["training_policy"] == "exclude_outage_severe")].copy()

    # ---- load A labels for training ----
    df = pd.read_csv(ROOT / TRAIN_REL, usecols=["mmsi","x","y","sog","time"], dtype={"mmsi":"string","x":"float64","y":"float64","sog":"float32"}, parse_dates=["time"])
    df["hour"] = df["time"].dt.floor("h"); df["date"] = df["time"].dt.normalize()
    df = add_regions(df)
    a_lab = make_a_labels(df)
    a_lab["date"] = a_lab["hour"].dt.normalize(); a_lab["hour_of_day"] = a_lab["hour"].dt.hour
    # pivot A labels to (date, hour, region) matrix
    a_pivot = a_lab.pivot_table(index=["date","hour_of_day"], columns="region", values="y", aggfunc="first").reset_index()
    for r in REGIONS:
        if r not in a_pivot.columns: a_pivot[r] = 0

    aud = pd.read_csv(ROOT / STEP02); aud["date"] = pd.to_datetime(aud["date"]).dt.normalize()
    qmap = dict(zip(aud["date"], aud["quality_regime"]))

    # ---- verify step 21 ----
    s21 = pd.read_csv(ROOT / STEP21, encoding="utf-8-sig")
    for _, r in s21.iterrows():
        eid = r["evaluation_id"]
        sc = bp[bp["evaluation_id"] == eid]
        my_sse = float(((sc["pred_rounded"] - sc["y_true"]).astype(float) ** 2).sum())
        assert abs(my_sse - r["sse_a"]) < 1e-6, f"SSE mismatch {eid}"
    print("Step 21 baseline reproduction PASS.")

    # ---- per-fold baseline matrices ----
    # For each fold: baseline pred per (date, hour, region) and y_true
    fold_data = {}
    for eid in SCENARIOS:
        tstart, tend, vstart, vend = SCN_MAP[eid]
        bp_e = bp[bp["evaluation_id"] == eid].copy()
        # baseline pred per (date, hour, region)
        bp_pivot = bp_e.pivot_table(index=["date","hour_of_day"], columns="region", values="pred_rounded", aggfunc="first").reset_index()
        yt_pivot = bp_e.pivot_table(index=["date","hour_of_day"], columns="region", values="y_true", aggfunc="first").reset_index()
        for r in REGIONS:
            if r not in bp_pivot.columns: bp_pivot[r] = 0
            if r not in yt_pivot.columns: yt_pivot[r] = 0
        # training normal dates
        train_dates = pd.date_range(tstart, tend, freq="D").normalize()
        train_normal = [d for d in train_dates if qmap.get(d) == "normal"]
        fold_data[eid] = {"bp": bp_pivot, "yt": yt_pivot, "train_normal": train_normal, "vstart": vstart, "vend": vend}

    # ---- step23 fixed 10 signals ----
    # load from step23 signal_stability
    try:
        sig23 = pd.read_csv(ROOT / "outputs/step23_a_error_attribution_diagnostics/signal_stability.csv", encoding="utf-8-sig")
        fixed10 = sig23[sig23["signal_strength"] == "strong"][["region","hour_of_day","action"]].to_dict("records")
    except Exception:
        fixed10 = []
    print(f"Step 23 fixed signals: {len(fixed10)}")

    # ========== METHOD DEFINITIONS ==========
    methods = ["baseline", "nested_region_hour_unconstrained", "nested_region_hour_total_preserving",
               "normal_common_profile_cv", "normal_profile_plus_nested_residual", "step23_fixed10_diagnostic"]

    # ========== COMPUTE PREDICTIONS PER METHOD ==========
    all_preds = {}  # (eid, method) -> DataFrame
    inner_sel_rows = []
    action_rows = []
    profile_sel_rows = []

    for eid in SCENARIOS:
        fd = fold_data[eid]
        bp_pivot = fd["bp"]; yt_pivot = fd["yt"]; train_normal = fd["train_normal"]
        valid_dates = sorted(bp_pivot["date"].unique())

        for method in methods:
            if method == "baseline":
                pred_pivot = bp_pivot.copy()
            elif method == "step23_fixed10_diagnostic":
                pred_pivot = bp_pivot.copy()
                for sig in fixed10:
                    r = sig["region"]; h = int(sig["hour_of_day"]); act = sig["action"]
                    delta = 1 if act == "plus_one" else -1
                    mask = (pred_pivot["hour_of_day"] == h)
                    pred_pivot.loc[mask, r] = np.clip(pred_pivot.loc[mask, r] + delta, 0, None)
            elif method in ("nested_region_hour_unconstrained", "nested_region_hour_total_preserving"):
                # nested selection
                pred_pivot = bp_pivot.copy()
                if len(train_normal) >= 7:
                    # compute per-cell deltas on training normal dates
                    # for each (region, hour, action), compute per-date delta SSE
                    candidates = []
                    for r in REGIONS:
                        for h in range(24):
                            base_val = float(bp_pivot[bp_pivot["hour_of_day"]==h][r].iloc[0])  # constant baseline
                            for act_name, delta in [("plus_one", 1), ("minus_one", -1)]:
                                if base_val + delta < 0: continue
                                per_date_deltas = []
                                for d in train_normal:
                                    y_val = float(a_pivot[(a_pivot["date"]==d)&(a_pivot["hour_of_day"]==h)][r].iloc[0])
                                    d_sse = (y_val - (base_val + delta))**2 - (y_val - base_val)**2
                                    per_date_deltas.append(d_sse)
                                arr = np.array(per_date_deltas)
                                n_improve = int((arr < 0).sum()); n_total = len(arr)
                                improve_rate = n_improve / n_total if n_total else 0
                                cum_delta = float(arr.sum())
                                mean_resid = float(np.mean([float(a_pivot[(a_pivot["date"]==d)&(a_pivot["hour_of_day"]==h)][r].iloc[0]) - base_val for d in train_normal]))
                                # split halves
                                half = len(arr) // 2
                                first_half_ok = float(arr[:half].sum()) <= 0 if half > 0 else True
                                second_half_ok = float(arr[half:].sum()) <= 0 if half > 0 else True
                                if improve_rate >= 0.70 and cum_delta < 0 and first_half_ok and second_half_ok and abs(mean_resid) > 0.5:
                                    candidates.append({"region": r, "hour": h, "action": act_name, "delta": delta, "cum_delta": cum_delta, "base_val": base_val})
                    # sort by cum_delta (most negative first), take top 12
                    candidates.sort(key=lambda c: c["cum_delta"])
                    selected = candidates[:12]
                    # for total_preserving: pair +1 and -1
                    if method == "nested_region_hour_total_preserving":
                        plus_ones = [c for c in selected if c["delta"] > 0]
                        minus_ones = [c for c in selected if c["delta"] < 0]
                        n_pairs = min(len(plus_ones), len(minus_ones))
                        selected = plus_ones[:n_pairs] + minus_ones[:n_pairs]
                    # apply
                    for s in selected:
                        mask = (pred_pivot["hour_of_day"] == s["hour"])
                        pred_pivot.loc[mask, s["region"]] = np.clip(pred_pivot.loc[mask, s["region"]] + s["delta"], 0, None)
                        action_rows.append({"evaluation_id": eid, "method": method, "region": s["region"],
                                            "hour_of_day": s["hour"], "action": "plus_one" if s["delta"]>0 else "minus_one",
                                            "cum_delta_sse": s["cum_delta"]})
                    inner_sel_rows.append({"evaluation_id": eid, "method": method, "n_selected": len(selected),
                                           "n_candidates": len(candidates), "n_train_normal": len(train_normal)})
                else:
                    inner_sel_rows.append({"evaluation_id": eid, "method": method, "n_selected": 0, "n_candidates": 0, "n_train_normal": len(train_normal)})
            elif method == "normal_common_profile_cv":
                pred_pivot = bp_pivot.copy()
                if len(train_normal) >= 7:
                    # precompute baseline hour totals per day_type from valid dates
                    bl_hr_by_dt = {}; bl_mat_by_dt = {}
                    for d_v in bp_pivot["date"].unique():
                        dt = "weekend" if pd.Timestamp(d_v).dayofweek in (5,6) else "weekday"
                        sub = bp_pivot[bp_pivot["date"]==d_v]
                        hr = np.zeros(24); mat = np.zeros((24,3))
                        for h in range(24):
                            sh = sub[sub["hour_of_day"]==h]
                            for ri, r in enumerate(REGIONS):
                                mat[h, ri] = float(sh[r].iloc[0]) if len(sh) else 0.0
                            hr[h] = float(mat[h].sum())
                        bl_hr_by_dt[dt] = hr; bl_mat_by_dt[dt] = mat
                    def bl_hr_for(d):
                        dt = "weekend" if d.dayofweek in (5,6) else "weekday"
                        return bl_hr_by_dt.get(dt, np.zeros(24))
                    def bl_mat_for(d):
                        dt = "weekend" if d.dayofweek in (5,6) else "weekday"
                        return bl_mat_by_dt.get(dt, np.zeros((24,3)))
                    # inner CV to select (smoother, lambda)
                    candidates_pl = [("raw", lam) for lam in [0.25, 0.5, 0.75, 1.0]] + [("tri3", lam) for lam in [0.25, 0.5, 0.75, 1.0]]
                    best_sse = np.inf; best_sel = ("raw", 0.25)
                    # estimate q_h from ALL training normal (for outer application)
                    q_raw = np.zeros(24)
                    for h in range(24):
                        vals = []
                        for d in train_normal:
                            dr = a_pivot[a_pivot["date"]==d]
                            H_d = float(dr[REGIONS].sum(axis=1).sum())
                            if H_d > 0:
                                vals.append(float(dr[dr["hour_of_day"]==h][REGIONS].sum(axis=1).iloc[0]) / H_d)
                        q_raw[h] = np.mean(vals) if vals else 0.0
                    q_raw = q_raw / q_raw.sum() if q_raw.sum() > 0 else np.ones(24)/24
                    q_tri3 = smooth3(q_raw); q_tri3 = q_tri3 / q_tri3.sum()

                    # inner CV
                    for smoother, lam in candidates_pl:
                        q = q_tri3 if smoother == "tri3" else q_raw
                        inner_sse = 0.0
                        for d in train_normal[6:]:  # inner valid from index 6
                            inner_train = [td for td in train_normal if td < d]
                            if len(inner_train) < 5: continue
                            qi = np.zeros(24)
                            for h in range(24):
                                vs = []
                                for td in inner_train:
                                    dr = a_pivot[a_pivot["date"]==td]
                                    H = float(dr[REGIONS].sum(axis=1).sum())
                                    if H > 0: vs.append(float(dr[dr["hour_of_day"]==h][REGIONS].sum(axis=1).iloc[0]) / H)
                                qi[h] = np.mean(vs) if vs else 0.0
                            qi = qi / qi.sum() if qi.sum() > 0 else np.ones(24)/24
                            if smoother == "tri3": qi = smooth3(qi); qi = qi / qi.sum()
                            # baseline daily total for inner valid date d
                            bl_hr = bl_hr_for(d)
                            H_hat = float(bl_hr.sum())
                            bl_share = bl_hr / bl_hr.sum() if bl_hr.sum() > 0 else np.ones(24)/24
                            # blended share
                            new_share = lam * qi + (1 - lam) * bl_share
                            new_share = new_share / new_share.sum()
                            new_hr_total = H_hat * new_share
                            # allocate to regions by baseline proportions
                            dr_true = a_pivot[a_pivot["date"]==d]
                            bl_mat = bl_mat_for(d)
                            day_sse = 0.0
                            for h in range(24):
                                reg_props = bl_mat[h].astype(float)
                                rs = reg_props.sum()
                                reg_props = reg_props / rs if rs > 0 else np.ones(3)/3
                                alloc = lr_alloc(new_hr_total[h], reg_props)
                                dh = dr_true[dr_true["hour_of_day"]==h]
                                yt_h = np.array([float(dh[r].iloc[0]) if len(dh) else 0.0 for r in REGIONS])
                                day_sse += float(((alloc - yt_h)**2).sum())
                            inner_sse += day_sse
                        if inner_sse < best_sse:
                            best_sse = inner_sse; best_sel = (smoother, lam)
                    # apply best to all valid dates
                    q = q_tri3 if best_sel[0] == "tri3" else q_raw; lam = best_sel[1]
                    for d in valid_dates:
                        bp_d = pred_pivot[pred_pivot["date"]==d]
                        H_hat = float(bp_d[REGIONS].sum(axis=1).sum())
                        bl_hr = np.zeros(24)
                        for h in range(24):
                            bl_hr[h] = float(bp_d[bp_d["hour_of_day"]==h][REGIONS].sum(axis=1).iloc[0])
                        bl_share = bl_hr / bl_hr.sum() if bl_hr.sum() > 0 else np.ones(24)/24
                        new_share = lam * q + (1 - lam) * bl_share; new_share /= new_share.sum()
                        new_hr_total = H_hat * new_share
                        for h in range(24):
                            mask = pred_pivot["hour_of_day"] == h
                            bp_h = bp_pivot[bp_pivot["date"]==d]
                            if len(bp_h[bp_h["hour_of_day"]==h]) == 0: continue
                            reg_props = np.array([float(bp_h[bp_h["hour_of_day"]==h][r].iloc[0]) for r in REGIONS])
                            rs = reg_props.sum(); reg_props = reg_props / rs if rs > 0 else np.ones(3)/3
                            alloc = lr_alloc(new_hr_total[h], reg_props)
                            for ri, r in enumerate(REGIONS):
                                pred_pivot.loc[(pred_pivot["date"]==d)&(pred_pivot["hour_of_day"]==h), r] = alloc[ri]
                    profile_sel_rows.append({"evaluation_id": eid, "smoother": best_sel[0], "lambda": best_sel[1], "inner_sse": best_sse})
                else:
                    profile_sel_rows.append({"evaluation_id": eid, "smoother": "raw", "lambda": 0.25, "inner_sse": np.nan})
            elif method == "normal_profile_plus_nested_residual":
                # apply profile first (reuse normal_common_profile_cv result), then total-preserving residual
                # get profile prediction
                prof_key = (eid, "normal_common_profile_cv")
                if prof_key in all_preds:
                    pred_pivot = all_preds[prof_key].copy()
                else:
                    pred_pivot = bp_pivot.copy()  # fallback
                # then apply total-preserving residual on top
                if len(train_normal) >= 7:
                    candidates = []
                    for r in REGIONS:
                        for h in range(24):
                            base_val = float(pred_pivot[pred_pivot["hour_of_day"]==h][r].iloc[0])
                            for act_name, delta in [("plus_one",1),("minus_one",-1)]:
                                if base_val + delta < 0: continue
                                per_date = []
                                for d in train_normal:
                                    y_val = float(a_pivot[(a_pivot["date"]==d)&(a_pivot["hour_of_day"]==h)][r].iloc[0])
                                    per_date.append((y_val-(base_val+delta))**2 - (y_val-base_val)**2)
                                arr = np.array(per_date); n_imp = int((arr<0).sum()); cum = float(arr.sum())
                                mr = float(np.mean([float(a_pivot[(a_pivot["date"]==d)&(a_pivot["hour_of_day"]==h)][r].iloc[0])-base_val for d in train_normal]))
                                half = len(arr)//2
                                if n_imp/len(arr)>=0.70 and cum<0 and (half==0 or (float(arr[:half].sum())<=0 and float(arr[half:].sum())<=0)) and abs(mr)>0.5:
                                    candidates.append({"region":r,"hour":h,"delta":delta,"cum_delta":cum})
                    candidates.sort(key=lambda c:c["cum_delta"])
                    plus = [c for c in candidates if c["delta"]>0]; minus = [c for c in candidates if c["delta"]<0]
                    n_pairs = min(len(plus), len(minus)); selected = plus[:n_pairs]+minus[:n_pairs]
                    for s in selected:
                        mask = pred_pivot["hour_of_day"]==s["hour"]
                        pred_pivot.loc[mask, s["region"]] = np.clip(pred_pivot.loc[mask, s["region"]]+s["delta"], 0, None)
                        action_rows.append({"evaluation_id":eid,"method":method,"region":s["region"],"hour_of_day":s["hour"],
                                            "action":"plus_one" if s["delta"]>0 else "minus_one","cum_delta_sse":s["cum_delta"]})
                    inner_sel_rows.append({"evaluation_id":eid,"method":method,"n_selected":len(selected),"n_candidates":len(candidates),"n_train_normal":len(train_normal)})
                else:
                    inner_sel_rows.append({"evaluation_id":eid,"method":method,"n_selected":0,"n_candidates":0,"n_train_normal":len(train_normal)})

            all_preds[(eid, method)] = pred_pivot

    # ========== SCORING ==========
    fold_score_rows = []
    for eid in SCENARIOS:
        yt = fold_data[eid]["yt"]
        etype = "rolling_7d" if eid in ROLLING else "final_analog"
        for method in methods:
            pred = all_preds[(eid, method)]
            # merge pred and yt on (date, hour_of_day)
            merged = pred.merge(yt, on=["date","hour_of_day"], suffixes=("_p","_y"))
            # compute SSE
            for r in REGIONS:
                merged[f"e_{r}"] = merged[f"{r}_p"] - merged[f"{r}_y"]
            merged["sse"] = sum(merged[f"e_{r}"]**2 for r in REGIONS)
            total_sse = float(merged["sse"].sum())
            # per-date
            dates = merged["date"].unique()
            normal_dates_in_valid = [d for d in dates if qmap.get(pd.Timestamp(d)) == "normal"]
            abnormal_dates = [d for d in dates if qmap.get(pd.Timestamp(d)) != "normal"]
            normal_sse = float(merged[merged["date"].isin(normal_dates_in_valid)]["sse"].sum())
            abn_sse = float(merged[merged["date"].isin(abnormal_dates)]["sse"].sum())
            # post-outage normal (01-19..01-24)
            po_dates = [d for d in normal_dates_in_valid if pd.Timestamp(d) >= pd.Timestamp("2018-01-19")]
            po_sse = float(merged[merged["date"].isin(po_dates)]["sse"].sum())
            fold_score_rows.append({"evaluation_id": eid, "evaluation_type": etype, "method": method,
                                    "n_dates": int(len(dates)), "total_sse": total_sse, "normal_sse": normal_sse,
                                    "abnormal_sse": abn_sse, "post_outage_normal_sse": po_sse,
                                    "n_normal_dates": int(len(normal_dates_in_valid)),
                                    "n_abnormal_dates": int(len(abnormal_dates))})
    fold_scores = pd.DataFrame(fold_score_rows)

    # ---- baseline for reduction computation ----
    bl = fold_scores[fold_scores["method"]=="baseline"].set_index("evaluation_id")

    # ---- decision table ----
    dec_rows = []
    for method in methods:
        if method == "step23_fixed10_diagnostic": dep = False
        else: dep = True
        ms = fold_scores[fold_scores["method"]==method]
        roll = ms[ms["evaluation_type"]=="rolling_7d"]
        fa = ms[ms["evaluation_id"]=="final_analog"]
        # aggregate reduction (total SSE ratio)
        bl_roll_total = float(bl.loc[roll["evaluation_id"],"total_sse"].sum())
        m_roll_total = float(roll["total_sse"].sum())
        agg_red = 1 - m_roll_total/bl_roll_total if bl_roll_total else 0
        # normal-only aggregate
        bl_roll_norm = float(bl.loc[roll["evaluation_id"],"normal_sse"].sum())
        m_roll_norm = float(roll["normal_sse"].sum())
        agg_red_norm = 1 - m_roll_norm/bl_roll_norm if bl_roll_norm else 0
        # mean-fold reduction
        fold_reds = []
        for _, r in roll.iterrows():
            bl_v = float(bl.loc[r["evaluation_id"], "total_sse"])
            fold_reds.append(1 - r["total_sse"]/bl_v if bl_v else 0)
        mean_fold_red = float(np.mean(fold_reds)) if fold_reds else 0
        # normal-only fold wins
        n_improve_norm = 0
        for _, r in roll.iterrows():
            bl_v = float(bl.loc[r["evaluation_id"], "normal_sse"])
            if bl_v > 0 and r["normal_sse"] < bl_v: n_improve_norm += 1
        # final analog
        fa_bl = float(bl.loc["final_analog","total_sse"]); fa_m = float(fa["total_sse"].iloc[0]) if len(fa) else np.nan
        fa_norm_bl = float(bl.loc["final_analog","normal_sse"]); fa_norm_m = float(fa["normal_sse"].iloc[0]) if len(fa) else np.nan
        # post-outage normal
        po_bl = float(bl.loc[roll["evaluation_id"],"post_outage_normal_sse"].sum())
        po_m = float(roll["post_outage_normal_sse"].sum())
        dec_rows.append({"method": method, "deployable": dep,
                         "agg_reduction_all": agg_red, "agg_reduction_normal": agg_red_norm,
                         "mean_fold_reduction_all": mean_fold_red,
                         "normal_fold_improve_count": n_improve_norm,
                         "final_analog_sse": fa_m, "final_analog_reduction": 1-fa_m/fa_bl if fa_bl else 0,
                         "final_analog_normal_sse": fa_norm_m, "final_analog_normal_reduction": 1-fa_norm_m/fa_norm_bl if fa_norm_bl else 0,
                         "post_outage_normal_sse": po_m, "post_outage_normal_reduction": 1-po_m/po_bl if po_bl else 0,
                         "rolling_max_sse": float(roll["total_sse"].max()), "bl_rolling_max": float(bl.loc[roll["evaluation_id"],"total_sse"].max())})
    decision = pd.DataFrame(dec_rows)

    # ---- write outputs ----
    # method_definitions
    md = pd.DataFrame([
        {"method": "baseline", "deployable": True},
        {"method": "nested_region_hour_unconstrained", "deployable": True},
        {"method": "nested_region_hour_total_preserving", "deployable": True},
        {"method": "normal_common_profile_cv", "deployable": True},
        {"method": "normal_profile_plus_nested_residual", "deployable": True},
        {"method": "step23_fixed10_diagnostic", "deployable": False},
    ])
    md.to_csv(out_dir / "method_definitions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inner_sel_rows).to_csv(out_dir / "inner_selection.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(action_rows).to_csv(out_dir / "selected_region_hour_actions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(profile_sel_rows).to_csv(out_dir / "profile_selection.csv", index=False, encoding="utf-8-sig")
    fold_scores.to_csv(out_dir / "fold_scores.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(out_dir / "decision_table.csv", index=False, encoding="utf-8-sig")

    # summary
    L = ["# Step 24 A Normal-Regime Calibration", "", f"Run command: `{RUN_CMD}`", ""]
    L.append("## Results (normal-only aggregate reduction)")
    L.append("")
    L.append("| method | agg_red_norm | mean_fold_all | normal_improve | final_norm_red | post_outage_red |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for _, r in decision.iterrows():
        L.append(f"| {r['method']} | {r['agg_reduction_normal']:.3f} | {r['mean_fold_reduction_all']:.3f} | {int(r['normal_fold_improve_count'])}/8 | {r['final_analog_normal_reduction']:.3f} | {r['post_outage_normal_reduction']:.3f} |")
    L.append("")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    assert not any(out_dir.rglob("*.zip"))
    print("\n=== Decision table ===")
    for _, r in decision.iterrows():
        print(f"  {r['method']:42s} agg_norm={r['agg_reduction_normal']:.3f} improve={int(r['normal_fold_improve_count'])}/8 fa_norm={r['final_analog_normal_reduction']:.3f}")
    print("\nDone.")


if __name__ == "__main__": main()
