"""Step 25 — A weekly cycle and vessel-level modeling benchmark.

Tests fundamentally different A prediction structures: same-weekday analog
copying, vessel-level weekly pattern replication, and vessel presence
probability models. Uses three evaluation frameworks designed for weekly
signals: normal-day rolling-origin (unique dates), normal-to-normal weekly
pairs (lag 7/14/21), and post-outage block simulation.

Key insight: the target period Jan 25-31 has excellent lag-7 normal source
days (Jan 19-24, all normal), which the existing overlapping folds could
not test because their lag-7 fell in the anomaly period.

Outputs (under outputs/step25_a_weekly_vessel_model_benchmark/):
  1. vessel_hour_region_labels.csv
  2. vessel_daily_features.csv
  3. normal_day_predictions.csv
  4. normal_day_scores.csv
  5. weekly_pair_scores.csv
  6. post_outage_block_scores.csv
  7. model_component_scores.csv
  8. vessel_model_diagnostics.csv
  9. final_target_analog_map.csv
  10. decision_table.csv
  11. summary.md
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from optimized_baseline import REGION_ORDER, add_regions, make_a_labels  # noqa: E402

TRAIN_REL = Path("数据备份/训练集_20180101-0124_拖轮AIS.csv")
STEP01 = Path("outputs/step01_data_audit/train_daily_overview.csv")
STEP02 = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
STEP01_VAL = Path("outputs/step01_data_audit/validation_daily_vessels.csv")
OUT_REL = Path("outputs/step25_a_weekly_vessel_model_benchmark")
REGIONS = REGION_ORDER
RUN_CMD = "python " + " ".join(sys.argv)
TAU = 20


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
        cnt = 0
        for j in np.argsort(fp - base):
            if base[j] > 0 and cnt < -rem: base[j] -= 1; cnt += 1
    return np.clip(base, 0, None)


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    # ---- load AIS ----
    df = pd.read_csv(ROOT / TRAIN_REL, usecols=["mmsi","x","y","sog","time","source_dataset"],
                     dtype={"mmsi":"string","x":"float64","y":"float64","sog":"float32","source_dataset":"string"}, parse_dates=["time"])
    df["hour"] = df["time"].dt.floor("h"); df["date"] = df["time"].dt.normalize()
    df["hour_of_day"] = df["time"].dt.hour; df["dayofweek"] = df["time"].dt.dayofweek
    df = add_regions(df)

    daily = pd.read_csv(ROOT / STEP01); daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    aud = pd.read_csv(ROOT / STEP02); aud["date"] = pd.to_datetime(aud["date"]).dt.normalize()
    qmap = dict(zip(aud["date"], aud["quality_regime"]))
    vc_map = dict(zip(daily["date"], daily["unique_vessel_count"]))
    all_dates = sorted(pd.date_range("2018-01-01", "2018-01-24", freq="D").normalize())

    # ---- 1. vessel-hour-region qualified_active labels ----
    active = df[df["region"].isin(REGIONS) & df["sog"].between(2, 10, inclusive="both")].copy()
    vrc = active.groupby(["mmsi","hour","date","hour_of_day","region"]).size().rename("n_records").reset_index()
    vrc["qualified_active"] = (vrc["n_records"] >= 3).astype(int)
    # verify: sum qualified_active per (hour, region) == make_a_labels
    check = vrc[vrc["qualified_active"]==1].groupby(["hour","region"])["mmsi"].nunique().rename("y_check").reset_index()
    a_lab = make_a_labels(df)
    a_lab["date"] = a_lab["hour"].dt.normalize()
    a_lab["hour_of_day"] = a_lab["hour"].dt.hour
    merged = a_lab.merge(check, on=["hour","region"], how="left")
    merged["y_check"] = merged["y_check"].fillna(0).astype(int)
    mism = (merged["y"] != merged["y_check"]).sum()
    assert mism == 0, f"vessel labels mismatch make_a_labels: {mism} cells"
    print(f"Vessel-hour-region labels: {len(vrc)} rows; verified against make_a_labels.")

    # ---- 2. vessel daily features ----
    # vessel present (any AIS record that day, any region)
    present = df.groupby(["mmsi","date"]).size().reset_index().rename(columns={0:"n"})
    present_dates = present.groupby("mmsi")["date"].apply(set).to_dict()
    # vessel active (contributes to A that day)
    active_vessels = vrc[vrc["qualified_active"]==1].groupby(["mmsi","date"]).size().reset_index().rename(columns={0:"n"})
    active_dates = active_vessels.groupby("mmsi")["date"].apply(set).to_dict()
    # build per-vessel per-date features
    all_mmsi = sorted(df["mmsi"].unique())
    vfeat_rows = []
    for d in all_dates:
        dt = int(d.dayofweek)
        for m in all_mmsi:
            pres = d in present_dates.get(m, set())
            act = d in active_dates.get(m, set())
            if not pres: continue  # only track present vessels
            # daily active hours
            dah = int(active_vessels[(active_vessels["mmsi"]==m)&(active_vessels["date"]==d)]["n"].iloc[0]) if act else 0
            # dominant region
            if act:
                dr = vrc[(vrc["mmsi"]==m)&(vrc["date"]==d)&(vrc["qualified_active"]==1)].groupby("region").size()
                dom = dr.idxmax() if len(dr) else "none"
            else:
                dom = "none"
            # historical presence frequency (up to d)
            hist_pres = present[present["mmsi"]==m]
            hist_freq = len(hist_pres[hist_pres["date"]<d]) / max(1, len([dd for dd in all_dates if dd < d]))
            # recent 7d
            rec7_start = d - pd.Timedelta(days=7)
            rec7 = len(hist_pres[(hist_pres["date"]>=rec7_start)&(hist_pres["date"]<d)])
            rec7_freq = rec7 / 7.0
            # same weekday
            sw_dates = [dd for dd in all_dates if dd < d and int(dd.dayofweek)==dt]
            sw_pres = len(hist_pres[hist_pres["date"].isin(sw_dates)])
            sw_freq = sw_pres / max(1, len(sw_dates))
            vfeat_rows.append({"mmsi":m,"date":d,"vessel_present_on_day":pres,"vessel_active_on_day":act,
                               "daily_active_hours":dah,"dominant_region":dom,
                               "historical_presence_frequency":hist_freq,"recent_7d_presence_frequency":rec7_freq,
                               "same_weekday_presence_frequency":sw_freq,"dayofweek":dt})
    vfeat = pd.DataFrame(vfeat_rows)

    # ---- helper: get A 72-vector for a date ----
    a_pivot = a_lab.pivot_table(index=["date","hour_of_day"], columns="region", values="y", aggfunc="first").reset_index()
    for r in REGIONS:
        if r not in a_pivot.columns: a_pivot[r] = 0

    def get_a_vector(d):
        """Return 24x3 numpy array of A labels for date d."""
        sub = a_pivot[a_pivot["date"]==d].sort_values("hour_of_day")
        mat = np.zeros((24, 3))
        for _, row in sub.iterrows():
            h = int(row["hour_of_day"])
            for ri, r in enumerate(REGIONS):
                mat[h, ri] = float(row[r])
        return mat

    def get_pred_vector_from_date(d):
        """Return 24x3 numpy array of A labels for date d (same as get_a_vector but for predictions)."""
        return get_a_vector(d)

    # ---- quality-aware same-weekday source finder ----
    def find_normal_same_weekday_sources(d, max_lag=21):
        """Find normal same-weekday source dates before d, up to max_lag."""
        dt = int(d.dayofweek)
        sources = []
        for lag in [7, 14, 21]:
            sd = d - pd.Timedelta(days=lag)
            if sd < all_dates[0]: continue
            if sd not in qmap: continue
            sources.append({"date": sd, "lag": lag, "quality": qmap.get(sd, "unknown"), "is_normal": qmap.get(sd)=="normal"})
        normal_sources = [s for s in sources if s["is_normal"]]
        return sources, normal_sources

    # ---- methods ----
    method_names = ["baseline_step11", "latest_normal_same_weekday", "mean_two_normal_same_weekdays",
                    "weighted_two_normal_same_weekdays", "latest_normal_same_weekday_scaled_U_p025",
                    "latest_normal_same_weekday_scaled_U_p05"]

    def predict_analog(d, method, train_end=None):
        """Predict A 72-vector for date d using analog method."""
        sources, normal_sources = find_normal_same_weekday_sources(d)
        if train_end and train_end < d:
            sources = [s for s in sources if s["date"] <= train_end]
            normal_sources = [s for s in normal_sources if s["date"] <= train_end]
        if not normal_sources:
            # fallback: use mean of all normal dates before d
            nd = [dd for dd in all_dates if dd < d and qmap.get(dd)=="normal"]
            if not nd: return None
            vecs = [get_a_vector(dd) for dd in nd]
            return np.round(np.mean(vecs, axis=0)).astype(int)

        if method == "baseline_step11":
            # current model: decomp_mean_daytype constant prediction
            # approximate by mean of normal dates before d
            nd = [dd for dd in all_dates if dd < d and qmap.get(dd)=="normal"]
            if not nd: return None
            vecs = [get_a_vector(dd) for dd in nd]
            return np.round(np.mean(vecs, axis=0)).astype(int)

        src = normal_sources
        if method in ("latest_normal_same_weekday", "latest_normal_same_weekday_scaled_U_p025", "latest_normal_same_weekday_scaled_U_p05"):
            s_date = src[0]["date"]  # most recent (lag-7 first)
            vec = get_pred_vector_from_date(s_date)
            if "scaled_U" in method:
                p = 0.25 if "p025" in method else 0.5
                U_d = vc_map.get(d, 1); U_s = vc_map.get(s_date, 1)
                scale = (U_d / U_s) ** p if U_s > 0 else 1.0
                vec = vec * scale
            return np.clip(np.round(vec), 0, None).astype(int)

        if method == "mean_two_normal_same_weekdays":
            if len(src) >= 2:
                v1 = get_pred_vector_from_date(src[0]["date"]); v2 = get_pred_vector_from_date(src[1]["date"])
                return np.round((v1 + v2) / 2).astype(int)
            else:
                return get_pred_vector_from_date(src[0]["date"]).astype(int)

        if method == "weighted_two_normal_same_weekdays":
            if len(src) >= 2:
                v1 = get_pred_vector_from_date(src[0]["date"]); v2 = get_pred_vector_from_date(src[1]["date"])
                return np.round(0.75 * v1 + 0.25 * v2).astype(int)
            else:
                return get_pred_vector_from_date(src[0]["date"]).astype(int)

        return None

    # ---- 3. normal-day rolling-origin evaluation ----
    normal_dates = [d for d in all_dates if qmap.get(d) == "normal"]
    eval_dates = []
    for d in normal_dates:
        _, ns = find_normal_same_weekday_sources(d)
        if ns: eval_dates.append(d)
    print(f"Normal dates with same-weekday source: {len(eval_dates)}")

    nd_pred_rows = []; nd_score_rows = []
    for d in eval_dates:
        true_vec = get_a_vector(d)  # 24x3
        for method in method_names:
            pred = predict_analog(d, method)
            if pred is None:
                pred = np.zeros((24, 3), dtype=int)
            sse = float(((pred - true_vec) ** 2).sum())
            nd_score_rows.append({"target_date": f"{d:%Y-%m-%d}", "method": method, "sse": sse, "U_d": vc_map.get(d, np.nan)})
            # store predictions
            for h in range(24):
                for ri, r in enumerate(REGIONS):
                    nd_pred_rows.append({"target_date": f"{d:%Y-%m-%d}", "method": method, "hour_of_day": h, "region": r,
                                         "y_true": int(true_vec[h, ri]), "pred": int(pred[h, ri])})

    nd_scores = pd.DataFrame(nd_score_rows)
    nd_preds = pd.DataFrame(nd_pred_rows)

    # ---- 4. normal-to-normal weekly pairs (lag 7/14/21) ----
    wp_rows = []
    for d_target in normal_dates:
        dt = int(d_target.dayofweek)
        for lag in [7, 14, 21]:
            d_source = d_target - pd.Timedelta(days=lag)
            if d_source < all_dates[0]: continue
            if qmap.get(d_source) != "normal": continue
            true_vec = get_a_vector(d_target)
            src_vec = get_a_vector(d_source)
            sse = float(((src_vec - true_vec) ** 2).sum())
            wp_rows.append({"target_date": f"{d_target:%Y-%m-%d}", "source_date": f"{d_source:%Y-%m-%d}",
                            "lag": lag, "sse": sse, "U_target": vc_map.get(d_target, np.nan), "U_source": vc_map.get(d_source, np.nan)})
    wp_scores = pd.DataFrame(wp_rows)

    # ---- 5. post-outage block (predict 01-19..01-24 using train <= 01-18) ----
    po_dates = pd.date_range("2018-01-19", "2018-01-24", freq="D").normalize()
    po_train_end = pd.Timestamp("2018-01-18")
    po_rows = []
    for method in method_names:
        total_sse = 0.0
        for d in po_dates:
            pred = predict_analog(d, method, train_end=po_train_end)
            if pred is None: pred = np.zeros((24,3), dtype=int)
            true_vec = get_a_vector(d)
            sse = float(((pred - true_vec) ** 2).sum())
            total_sse += sse
            po_rows.append({"method": method, "target_date": f"{d:%Y-%m-%d}", "sse": sse,
                            "U_d": vc_map.get(d, np.nan), "source_quality_info": str(find_normal_same_weekday_sources(d)[1])})
        # block total
        bl_sse = sum(r["sse"] for r in po_rows if r["method"]=="baseline_step11" and r["target_date"] in [f"{d:%Y-%m-%d}" for d in po_dates])
    po_scores = pd.DataFrame(po_rows)

    # ---- 6. vessel-level model: vessel_latest_weekly_pattern ----
    def predict_vessel_weekly(d, train_end=None):
        """Predict A by summing vessel-level weekly patterns."""
        dt = int(d.dayofweek)
        # for each vessel present in training, find latest normal same-weekday pattern
        pred_mat = np.zeros((24, 3))
        te = train_end or (d - pd.Timedelta(days=1))
        for m in all_mmsi:
            # find latest normal same-weekday appearance before d (and <= train_end)
            best_date = None
            for lag in [7, 14, 21]:
                sd = d - pd.Timedelta(days=lag)
                if sd < all_dates[0] or sd > te: continue
                if qmap.get(sd) != "normal": continue
                if sd in active_dates.get(m, set()):
                    best_date = sd; break
            if best_date is None:
                # fallback: latest normal appearance any day
                ad = sorted([dd for dd in active_dates.get(m, set()) if dd <= te and qmap.get(dd)=="normal"], reverse=True)
                if ad: best_date = ad[0]
            if best_date:
                mv = vrc[(vrc["mmsi"]==m)&(vrc["date"]==best_date)&(vrc["qualified_active"]==1)]
                for _, row in mv.iterrows():
                    h = int(row["hour_of_day"]); ri = REGIONS.index(row["region"])
                    pred_mat[h, ri] += 1
        return pred_mat.astype(int)

    # evaluate vessel model on post-outage block
    vm_po_sse = 0.0
    vm_po_rows = []
    for d in po_dates:
        vp = predict_vessel_weekly(d, train_end=po_train_end)
        tv = get_a_vector(d)
        sse = float(((vp - tv) ** 2).sum())
        vm_po_sse += sse
        vm_po_rows.append({"model": "vessel_latest_weekly_pattern", "target_date": f"{d:%Y-%m-%d}", "sse": sse})
    # also evaluate on normal-day rolling-origin (subset)
    vm_nd_sse = 0.0; vm_nd_n = 0
    for d in eval_dates:
        vp = predict_vessel_weekly(d)
        tv = get_a_vector(d)
        sse = float(((vp - tv) ** 2).sum())
        vm_nd_sse += sse; vm_nd_n += 1

    # ---- 7. final target analog map (01-25..01-31) ----
    try:
        val_v = pd.read_csv(ROOT / STEP01_VAL); val_dates = pd.to_datetime(val_v["date"]).dt.normalize().tolist()
    except Exception:
        val_dates = pd.date_range("2018-01-25", "2018-01-31", freq="D").normalize().tolist()
    val_vc = dict(zip(pd.to_datetime(val_v["date"]).dt.normalize(), val_v["unique_vessel_count"])) if len(val_v) else {}
    ta_rows = []
    for d in val_dates:
        sources, normal_sources = find_normal_same_weekday_sources(d)
        sel = normal_sources[0]["date"] if normal_sources else (sources[0]["date"] if sources else None)
        sel_reason = "normal lag-%d" % normal_sources[0]["lag"] if normal_sources else ("fallback lag-%d" % sources[0]["lag"] if sources else "none")
        ta_rows.append({"target_date": f"{d:%Y-%m-%d}", "target_weekday": int(d.dayofweek),
                        "lag7_date": f"{d-pd.Timedelta(days=7):%Y-%m-%d}", "lag7_quality": qmap.get(d-pd.Timedelta(days=7),"n/a"),
                        "lag14_date": f"{d-pd.Timedelta(days=14):%Y-%m-%d}", "lag14_quality": qmap.get(d-pd.Timedelta(days=14),"n/a"),
                        "lag21_date": f"{d-pd.Timedelta(days=21):%Y-%m-%d}", "lag21_quality": qmap.get(d-pd.Timedelta(days=21),"n/a"),
                        "selected_source_date": f"{sel:%Y-%m-%d}" if sel else "none", "selection_reason": sel_reason,
                        "target_U": val_vc.get(d, ""), "source_U": vc_map.get(sel, "") if sel else ""})
    target_map = pd.DataFrame(ta_rows)

    # ---- 8. model component scores ----
    bl_nd_sse = float(nd_scores[nd_scores["method"]=="baseline_step11"]["sse"].sum()) if len(nd_scores) else 0
    mc_rows = []
    for method in method_names:
        ms = nd_scores[nd_scores["method"]==method]["sse"]
        mc_rows.append({"method": method, "framework": "normal_day_rolling", "total_sse": float(ms.sum()),
                        "n_dates": int(len(ms)), "baseline_sse": bl_nd_sse,
                        "reduction_ratio": 1 - float(ms.sum())/bl_nd_sse if bl_nd_sse else 0})
    # post-outage block
    bl_po = float(po_scores[po_scores["method"]=="baseline_step11"]["sse"].sum())
    for method in method_names:
        ms = po_scores[po_scores["method"]==method]["sse"]
        mc_rows.append({"method": method, "framework": "post_outage_block", "total_sse": float(ms.sum()),
                        "n_dates": int(len(ms)), "baseline_sse": bl_po,
                        "reduction_ratio": 1 - float(ms.sum())/bl_po if bl_po else 0})
    # vessel model
    mc_rows.append({"method": "vessel_latest_weekly_pattern", "framework": "post_outage_block", "total_sse": vm_po_sse,
                    "n_dates": 6, "baseline_sse": bl_po, "reduction_ratio": 1 - vm_po_sse/bl_po if bl_po else 0})
    mc_rows.append({"method": "vessel_latest_weekly_pattern", "framework": "normal_day_rolling", "total_sse": vm_nd_sse,
                    "n_dates": vm_nd_n, "baseline_sse": bl_nd_sse, "reduction_ratio": 1 - vm_nd_sse/bl_nd_sse if bl_nd_sse else 0})
    mc = pd.DataFrame(mc_rows)

    # ---- 9. write files ----
    vrc_out = vrc[["mmsi","date","hour","hour_of_day","region","n_records","qualified_active"]].copy()
    vrc_out["date"] = pd.to_datetime(vrc_out["date"]).dt.strftime("%Y-%m-%d")
    vrc_out.to_csv(out_dir / "vessel_hour_region_labels.csv", index=False, encoding="utf-8-sig")
    vf = vfeat.copy(); vf["date"] = pd.to_datetime(vf["date"]).dt.strftime("%Y-%m-%d")
    vf.to_csv(out_dir / "vessel_daily_features.csv", index=False, encoding="utf-8-sig")
    nd_preds.to_csv(out_dir / "normal_day_predictions.csv", index=False, encoding="utf-8-sig")
    nd_scores.to_csv(out_dir / "normal_day_scores.csv", index=False, encoding="utf-8-sig")
    wp_scores.to_csv(out_dir / "weekly_pair_scores.csv", index=False, encoding="utf-8-sig")
    po_scores.to_csv(out_dir / "post_outage_block_scores.csv", index=False, encoding="utf-8-sig")
    mc.to_csv(out_dir / "model_component_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(vm_po_rows).to_csv(out_dir / "vessel_model_diagnostics.csv", index=False, encoding="utf-8-sig")
    target_map.to_csv(out_dir / "final_target_analog_map.csv", index=False, encoding="utf-8-sig")

    # decision table
    dec_rows = []
    for method in method_names + ["vessel_latest_weekly_pattern"]:
        for fw in ["normal_day_rolling", "post_outage_block"]:
            r = mc[(mc["method"]==method)&(mc["framework"]==fw)]
            if len(r):
                r = r.iloc[0]
                dec_rows.append({"method": method, "framework": fw, "total_sse": r["total_sse"],
                                 "baseline_sse": r["baseline_sse"], "reduction_ratio": r["reduction_ratio"],
                                 "passes_10pct": r["reduction_ratio"] >= 0.10, "passes_15pct": r["reduction_ratio"] >= 0.15})
    decision = pd.DataFrame(dec_rows)
    decision.to_csv(out_dir / "decision_table.csv", index=False, encoding="utf-8-sig")

    # ---- summary ----
    L = ["# Step 25 A Weekly Vessel Model Benchmark", "", f"Run command: `{RUN_CMD}`", ""]
    L.append("## Normal-day rolling-origin (unique dates)")
    L.append("")
    L.append("| method | total SSE | reduction | n_dates |")
    L.append("| --- | --- | --- | --- |")
    for _, r in mc[mc["framework"]=="normal_day_rolling"].iterrows():
        L.append(f"| {r['method']} | {r['total_sse']:.0f} | {r['reduction_ratio']*100:.1f}% | {int(r['n_dates'])} |")
    L.append("")
    L.append("## Post-outage block (01-19 to 01-24)")
    L.append("")
    L.append("| method | total SSE | reduction |")
    L.append("| --- | --- | --- |")
    for _, r in mc[mc["framework"]=="post_outage_block"].iterrows():
        L.append(f"| {r['method']} | {r['total_sse']:.0f} | {r['reduction_ratio']*100:.1f}% |")
    L.append("")
    L.append("## Weekly pair SSE by lag (normal-to-normal)")
    L.append("")
    L.append("| lag | mean SSE | n_pairs |")
    L.append("| --- | --- | --- |")
    for lag in [7, 14, 21]:
        sub = wp_scores[wp_scores["lag"]==lag]
        L.append(f"| {lag} | {float(sub['sse'].mean()):.0f} | {len(sub)} |")
    L.append("")
    L.append("## Target period analog map (01-25 to 01-31)")
    L.append("")
    L.append("| target | lag7 | lag7_q | lag14_q | lag21_q | selected | reason |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for _, r in target_map.iterrows():
        L.append(f"| {r['target_date']} | {r['lag7_date']} | {r['lag7_quality']} | {r['lag14_quality']} | {r['lag21_quality']} | {r['selected_source_date']} | {r['selection_reason']} |")
    L.append("")
    n_pass = len(decision[decision["passes_10pct"]])
    L.append(f"## Decision\n- Methods passing 10% threshold: {n_pass}\n- Methods passing 15%: {len(decision[decision['passes_15pct']])}\n")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    assert not any(out_dir.rglob("*.zip"))
    print("\n=== Normal-day rolling-origin ===")
    for _, r in mc[mc["framework"]=="normal_day_rolling"].iterrows():
        print(f"  {r['method']:42s} SSE={r['total_sse']:.0f} reduction={r['reduction_ratio']*100:.1f}%")
    print("\n=== Post-outage block ===")
    for _, r in mc[mc["framework"]=="post_outage_block"].iterrows():
        print(f"  {r['method']:42s} SSE={r['total_sse']:.0f} reduction={r['reduction_ratio']*100:.1f}%")
    print(f"\nWeekly pairs: lag-7 mean SSE={float(wp_scores[wp_scores['lag']==7]['sse'].mean()):.0f}, lag-14={float(wp_scores[wp_scores['lag']==14]['sse'].mean()):.0f}, lag-21={float(wp_scores[wp_scores['lag']==21]['sse'].mean()):.0f}")
    print(f"\nTarget map: {len(target_map)} dates")
    for _, r in target_map.iterrows():
        print(f"  {r['target_date']}: selected {r['selected_source_date']} ({r['selection_reason']})")
    print("\nDone.")


if __name__ == "__main__":
    main()
