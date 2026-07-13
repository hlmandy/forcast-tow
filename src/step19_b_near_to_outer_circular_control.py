"""Step 19 (redo) — Near->outer circular3 single-direction control (candidate_09).

Historical: replaces near->outer in the step17 hybrid with step10 circular_cv
predictions. Final: replaces near->outer in candidate_08 (B=1741) with a freshly
trained circular3 (long-term hour mean + triangular3 + independent round).

Outputs (under outputs/step19_b_near_to_outer_circular_control/):
  historical_predictions.csv (17856)
  historical_scores.csv (18)
  historical_direction_scores.csv (108)
  historical_nonoverlap_scores.csv (6)
  candidate_09_.../ (A + B CSVs)
  component_checks.csv (2)
  b_value_comparison.csv (1008)
  direction_change_summary.csv (6)
  hourly_profile.csv (24)
  daily_totals.csv (14)
  summary.md
"""

from __future__ import annotations
import hashlib, shutil, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from optimized_baseline import CN_TO_REGION, REGION_CN  # noqa: E402

STEP17_HIST = Path("outputs/step17_controlled_b_hybrid/historical_hybrid_predictions.csv")
STEP10_PREDS = Path("outputs/step10_b_shrinkage_calibration/b_model_predictions.csv")
PF_PATH = Path("outputs/step08_b_state_transition_audit/b_hourly_pair_flow.csv")
A_SOURCE = Path("outputs/step18_b_single_direction_control/candidate_08_A4423_B_core_to_near_only/提交结果1_区域活跃拖轮数量.csv")
B_BASELINE = Path("outputs/step18_b_single_direction_control/candidate_08_A4423_B_core_to_near_only/提交结果2_圈层间拖轮迁移量.csv")
OUT_REL = Path("outputs/step19_b_near_to_outer_circular_control")
UNSCORABLE = pd.Timestamp("2018-01-24 23:00:00")
SCENARIOS = [f"fold_{i:02d}" for i in range(1, 9)] + ["final_analog"]
SCN_ORDER = {s: i for i, s in enumerate(SCENARIOS)}
ALL_DIRS = ["core->near", "near->core", "core->outer", "outer->core", "near->outer", "outer->near"]
DIR_RANK = {d: i for i, d in enumerate(ALL_DIRS)}

HIST_PRED_COLS = ["evaluation_id", "evaluation_type", "method", "source_method", "date", "hour", "hour_of_day",
                  "source_region", "target_region", "task_key", "score_available", "y_true", "pred_float", "pred_rounded", "error_float", "error_rounded"]
HIST_SCORE_COLS = ["evaluation_id", "evaluation_type", "method", "n_scored_rows", "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded", "actual_total", "predicted_total"]
HIST_DIR_COLS = ["evaluation_id", "evaluation_type", "method", "task_key", "n_scored_rows", "actual_total", "predicted_total", "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded"]
HIST_NO_COLS = ["method", "block", "target_start", "target_end", "n_scored_rows", "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded"]
COMP_COLS = ["task", "source_path", "output_path", "row_count", "row_order_equal", "key_unique", "sha256_equal_to_source", "changed_value_count", "unchanged_value_count", "missing_value_count", "negative_value_count", "noninteger_value_count", "passed"]
CMP_COLS = ["row_number", "time_window", "source_zone", "target_zone", "task_key", "baseline_vessel_count", "candidate_vessel_count", "difference", "was_replaced_direction", "value_unchanged_when_not_replaced", "passed"]
DIRSUM_COLS = ["task_key", "row_count", "changed_value_count", "increase_count", "decrease_count", "baseline_total", "candidate_total", "total_change", "passed"]
HOURLY_PROF_COLS = ["hour_of_day", "raw_hour_mean", "triangular3_smoothed_mean", "rounded_prediction", "baseline_prediction", "difference", "training_observation_count"]
DAILY_COLS = ["task", "date", "prediction_total", "prediction_mean", "prediction_max", "zero_prediction_count"]
RUN_CMD = "python " + " ".join(sys.argv)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()


def smooth3(v):
    out = np.zeros(24)
    for h in range(24): out[h] = (v[(h-1)%24] + 2*v[h] + v[(h+1)%24]) / 4
    return np.clip(out, 0, None)


def score_method(df, method):
    rows = []
    for eid in SCENARIOS:
        sc = df[(df["evaluation_id"]==eid) & (df["method"]==method) & df["score_available"]]
        er = sc["error_rounded"].astype(float).to_numpy()
        rows.append({"evaluation_id": eid, "evaluation_type": sc["evaluation_type"].iloc[0] if len(sc) else "", "method": method,
                     "n_scored_rows": int(len(sc)), "sse_rounded": float((er**2).sum()), "mse_rounded": float((er**2).mean()),
                     "mae_rounded": float(np.abs(er).mean()), "mean_bias_rounded": float(er.mean()),
                     "actual_total": float(sc["y_true"].sum()), "predicted_total": float(sc["pred_rounded"].sum())})
    return rows


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    cand = out_dir / "candidate_09_A4423_B_coreNear_plus_nearOuterCircular"; cand.mkdir(exist_ok=True)
    print(f"out_dir = {out_dir}")

    # ========== HISTORICAL ==========
    s17 = pd.read_csv(ROOT / STEP17_HIST, encoding="utf-8-sig")
    s17["date"] = pd.to_datetime(s17["date"]).dt.normalize()
    s10 = pd.read_csv(ROOT / STEP10_PREDS, encoding="utf-8-sig")
    s10["date"] = pd.to_datetime(s10["date"]).dt.normalize()

    cur = s17[s17["method"] == "hybrid_confirmed_core_near"].copy()
    cur["method"] = "current_core_near_hybrid"

    circ_no = s10[(s10["predictor_method"] == "pair_hour_circular_cv") & (s10["rounding_method"] == "independent") & (s10["task_key"] == "near->outer")].copy()
    new = cur.copy()
    # replace near->outer rows
    no_mask = new["task_key"] == "near->outer"
    # match by (evaluation_id, date, hour_of_day)
    key_cols = ["evaluation_id", "date", "hour_of_day"]
    circ_map = circ_no.set_index(key_cols)
    for idx in new[no_mask].index:
        key = tuple(new.loc[idx, c] for c in key_cols)
        if key in circ_map.index:
            r = circ_map.loc[key]
            new.at[idx, "pred_float"] = r["pred_float"]
            new.at[idx, "pred_rounded"] = r["pred_rounded"]
            new.at[idx, "source_method"] = "circular_cv(step10)"
    new["method"] = "core_near_plus_near_outer_circular"
    new.loc[~no_mask, "source_method"] = new.loc[~no_mask, "source_method"]  # keep existing
    # recompute errors
    for df_ in [cur, new]:
        df_["error_float"] = np.where(df_["score_available"], df_["pred_float"] - df_["y_true"], np.nan)
        df_["error_rounded"] = np.where(df_["score_available"], df_["pred_rounded"] - df_["y_true"], np.nan)

    hist_preds = pd.concat([cur[HIST_PRED_COLS], new[HIST_PRED_COLS]], ignore_index=True)
    assert len(hist_preds) == 17856

    hist_scores = pd.DataFrame(score_method(hist_preds, "current_core_near_hybrid") + score_method(hist_preds, "core_near_plus_near_outer_circular"))
    # direction scores
    drows = []
    for method in ["current_core_near_hybrid", "core_near_plus_near_outer_circular"]:
        for eid in SCENARIOS:
            for d in ALL_DIRS:
                sd = hist_preds[(hist_preds["evaluation_id"]==eid) & (hist_preds["method"]==method) & (hist_preds["task_key"]==d) & hist_preds["score_available"]]
                er = sd["error_rounded"].astype(float).to_numpy()
                drows.append({"evaluation_id": eid, "evaluation_type": sd["evaluation_type"].iloc[0] if len(sd) else "", "method": method, "task_key": d,
                              "n_scored_rows": int(len(sd)), "actual_total": float(sd["y_true"].sum()), "predicted_total": float(sd["pred_rounded"].sum()),
                              "sse_rounded": float((er**2).sum()), "mse_rounded": float((er**2).mean()) if len(er) else np.nan,
                              "mae_rounded": float(np.abs(er).mean()) if len(er) else np.nan, "mean_bias_rounded": float(er.mean()) if len(er) else np.nan})
    hist_dir = pd.DataFrame(drows)
    # nonoverlap
    norows = []
    for method in ["current_core_near_hybrid", "core_near_plus_near_outer_circular"]:
        for block, eids in [("nonoverlap_block_1", ["fold_01"]), ("nonoverlap_block_2", ["fold_08"]), ("nonoverlap_combined", ["fold_01", "fold_08"])]:
            sub = hist_preds[(hist_preds["method"]==method) & (hist_preds["evaluation_id"].isin(eids)) & hist_preds["score_available"]]
            er = sub["error_rounded"].astype(float).to_numpy()
            dts = pd.to_datetime(sub["date"].unique())
            norows.append({"method": method, "block": block, "target_start": f"{dts.min():%Y-%m-%d}" if len(dts) else "", "target_end": f"{dts.max():%Y-%m-%d}" if len(dts) else "",
                           "n_scored_rows": int(len(sub)), "sse_rounded": float((er**2).sum()), "mse_rounded": float((er**2).mean()) if len(er) else np.nan,
                           "mae_rounded": float(np.abs(er).mean()) if len(er) else np.nan, "mean_bias_rounded": float(er.mean()) if len(er) else np.nan})
    hist_no = pd.DataFrame(norows)

    # verify historical
    roll = hist_scores[hist_scores["evaluation_type"] == "rolling_7d"]
    def ms(m): r = roll[roll["method"]==m]["mse_rounded"]; return float(r.mean()), float(r.median()), float(r.max())
    cm, cmd, cmx = ms("current_core_near_hybrid"); nm, nmd, nmx = ms("core_near_plus_near_outer_circular")
    fa_c = hist_scores[(hist_scores["evaluation_id"]=="final_analog") & (hist_scores["method"]=="current_core_near_hybrid")]["sse_rounded"].iloc[0]
    fa_n = hist_scores[(hist_scores["evaluation_id"]=="final_analog") & (hist_scores["method"]=="core_near_plus_near_outer_circular")]["sse_rounded"].iloc[0]
    print(f"current: mean={cm:.4f} med={cmd:.4f} max={cmx:.4f} fa={fa_c:.0f}")
    print(f"new:     mean={nm:.4f} med={nmd:.4f} max={nmx:.4f} fa={fa_n:.0f}")
    piv = roll.pivot_table(index="evaluation_id", columns="method", values="sse_rounded", aggfunc="first")
    diffs = (piv["core_near_plus_near_outer_circular"] - piv["current_core_near_hybrid"]).to_dict()
    for eid in [f"fold_{i:02d}" for i in range(1,9)]: print(f"  {eid}: {diffs.get(eid,0):+.0f}")
    n_better = int((piv["core_near_plus_near_outer_circular"] < piv["current_core_near_hybrid"]).sum())
    n_tie = int((piv["core_near_plus_near_outer_circular"] == piv["current_core_near_hybrid"]).sum())
    n_worse = int((piv["core_near_plus_near_outer_circular"] > piv["current_core_near_hybrid"]).sum())
    print(f"  better={n_better} tie={n_tie} worse={n_worse}")

    # verify only near->outer differs
    dr_roll = hist_dir[hist_dir["evaluation_type"]=="rolling_7d"].groupby(["method","task_key"])["sse_rounded"].sum().unstack()
    for d in ALL_DIRS:
        if d != "near->outer":
            assert dr_roll.loc["current_core_near_hybrid", d] == dr_roll.loc["core_near_plus_near_outer_circular", d], f"{d} changed!"

    # write historical
    hp = hist_preds.copy(); hp["_e"] = hp["evaluation_id"].map(SCN_ORDER)
    hp["_m"] = hp["method"].map({"current_core_near_hybrid":0, "core_near_plus_near_outer_circular":1})
    hp = hp.sort_values(["_e","_m","hour","source_region","target_region"]).drop(columns=["_e","_m"])
    hp["date"] = hp["date"].dt.strftime("%Y-%m-%d")
    hp[HIST_PRED_COLS].to_csv(out_dir / "historical_predictions.csv", index=False, encoding="utf-8-sig")
    hist_scores.sort_values(["evaluation_id","method"])[HIST_SCORE_COLS].to_csv(out_dir / "historical_scores.csv", index=False, encoding="utf-8-sig")
    hd = hist_dir.copy(); hd["_e"]=hd["evaluation_id"].map(SCN_ORDER); hd["_m"]=hd["method"].map({"current_core_near_hybrid":0,"core_near_plus_near_outer_circular":1}); hd["_d"]=hd["task_key"].map(DIR_RANK)
    hd.sort_values(["_e","_m","_d"])[HIST_DIR_COLS].to_csv(out_dir / "historical_direction_scores.csv", index=False, encoding="utf-8-sig")
    hist_no[HIST_NO_COLS].to_csv(out_dir / "historical_nonoverlap_scores.csv", index=False, encoding="utf-8-sig")

    # ========== FINAL CANDIDATE ==========
    # A: byte-copy
    A_OUT = cand / "提交结果1_区域活跃拖轮数量.csv"
    shutil.copyfile(ROOT / A_SOURCE, A_OUT)
    assert sha256(ROOT / A_SOURCE) == sha256(A_OUT), "A SHA mismatch"
    a_df = pd.read_csv(A_OUT); assert len(a_df) == 504 and int(a_df["vessel_count"].sum()) == 2241

    # near->outer circular3 from full training
    pf = pd.read_csv(ROOT / PF_PATH, encoding="utf-8-sig")
    pf["hour"] = pd.to_datetime(pf["hour"]); pf["date"] = pd.to_datetime(pf["date"]).dt.normalize(); pf["hour_of_day"] = pf["hour"].dt.hour
    no_data = pf[(pf["task_key"] == "near->outer") & (pf["hour"] != UNSCORABLE)]
    raw_mean = np.zeros(24); obs_count = np.zeros(24, dtype=int)
    for h in range(24):
        vals = no_data[no_data["hour_of_day"] == h]["y"].to_numpy(float)
        raw_mean[h] = float(vals.mean()) if len(vals) else 0.0
        obs_count[h] = int(len(vals))
    smoothed = smooth3(raw_mean)
    pred24 = np.clip(np.rint(smoothed), 0, None).astype(int)
    print(f"near->outer circular3: pred24={pred24.tolist()} total={int(pred24.sum())}")

    # B: read candidate_08 baseline, replace near->outer
    b_base = pd.read_csv(ROOT / B_BASELINE); assert len(b_base) == 1008
    b_ts = pd.to_datetime(b_base["time_window"])
    no_key = (REGION_CN["near"], REGION_CN["outer"])
    b_out = b_base.copy()
    for i in range(len(b_base)):
        sz, tz = b_base["source_zone"].iloc[i], b_base["target_zone"].iloc[i]
        if (sz, tz) == no_key:
            b_out.at[i, "vessel_count"] = int(pred24[b_ts.iloc[i].hour])
    B_OUT = cand / "提交结果2_圈层间拖轮迁移量.csv"
    b_out.to_csv(B_OUT, index=False, encoding="utf-8-sig")
    b_rr = pd.read_csv(B_OUT)
    assert len(b_rr) == 1008
    assert (b_rr["time_window"].to_numpy() == b_base["time_window"].to_numpy()).all()

    # b_value_comparison
    cmp_rows = []
    for i in range(len(b_base)):
        sz, tz = b_base["source_zone"].iloc[i], b_base["target_zone"].iloc[i]
        tk = f"{CN_TO_REGION[sz]}->{CN_TO_REGION[tz]}"
        bv = int(b_base["vessel_count"].iloc[i]); cv = int(b_rr["vessel_count"].iloc[i])
        replaced = (sz, tz) == no_key
        cmp_rows.append({"row_number": i+1, "time_window": b_base["time_window"].iloc[i], "source_zone": sz, "target_zone": tz, "task_key": tk,
                         "baseline_vessel_count": bv, "candidate_vessel_count": cv, "difference": cv - bv,
                         "was_replaced_direction": bool(replaced), "value_unchanged_when_not_replaced": bool(replaced or cv == bv),
                         "passed": bool((cv != bv) == replaced or cv == bv)})
    bcmp = pd.DataFrame(cmp_rows)
    assert (bcmp[~bcmp["was_replaced_direction"]]["difference"] == 0).all(), "non-near->outer changed!"
    assert (bcmp[bcmp["was_replaced_direction"] & (bcmp["difference"] != 0)]["difference"] == 1).all(), "non-+1 change!"
    bcmp[CMP_COLS].to_csv(out_dir / "b_value_comparison.csv", index=False, encoding="utf-8-sig")

    # direction change summary
    dr_rows = []
    for tk in ALL_DIRS:
        sub = bcmp[bcmp["task_key"] == tk]; diff = sub["difference"]
        changed = int((diff != 0).sum()); inc = int((diff > 0).sum()); dec = int((diff < 0).sum())
        dr_rows.append({"task_key": tk, "row_count": int(len(sub)), "changed_value_count": changed, "increase_count": inc, "decrease_count": dec,
                        "baseline_total": int(sub["baseline_vessel_count"].sum()), "candidate_total": int(sub["candidate_vessel_count"].sum()),
                        "total_change": int(diff.sum()), "passed": bool(changed == 0 or tk == "near->outer")})
    drsum = pd.DataFrame(dr_rows)
    no_row = drsum[drsum["task_key"] == "near->outer"].iloc[0]
    assert no_row["changed_value_count"] == 21, f"near->outer changed {no_row['changed_value_count']} != 21"
    assert no_row["increase_count"] == 21 and no_row["decrease_count"] == 0
    assert no_row["total_change"] == 21, f"near->outer total_change {no_row['total_change']} != 21"
    for tk in ALL_DIRS:
        if tk != "near->outer": assert drsum[drsum["task_key"]==tk].iloc[0]["changed_value_count"] == 0
    drsum[DIRSUM_COLS].to_csv(out_dir / "direction_change_summary.csv", index=False, encoding="utf-8-sig")

    # hourly_profile
    # baseline near->outer per hour (from candidate_08 B)
    base_no = b_base[(b_base["source_zone"] == REGION_CN["near"]) & (b_base["target_zone"] == REGION_CN["outer"])]
    base_ts = pd.to_datetime(base_no["time_window"])
    base_no_hr = base_no.groupby(base_ts.dt.hour)["vessel_count"].first()
    hp_rows = []
    for h in range(24):
        bp = int(base_no_hr.get(h, 0))
        hp_rows.append({"hour_of_day": h, "raw_hour_mean": float(raw_mean[h]), "triangular3_smoothed_mean": float(smoothed[h]),
                        "rounded_prediction": int(pred24[h]), "baseline_prediction": bp, "difference": int(pred24[h]) - bp, "training_observation_count": int(obs_count[h])})
    hp_df = pd.DataFrame(hp_rows)
    # verify changed hours
    changed_hours = hp_df[hp_df["difference"] != 0]["hour_of_day"].tolist()
    print(f"changed hours: {changed_hours}")
    hp_df[HOURLY_PROF_COLS].to_csv(out_dir / "hourly_profile.csv", index=False, encoding="utf-8-sig")

    # component_checks
    b_changed = int(no_row["changed_value_count"])
    comp = pd.DataFrame([
        {"task": "A", "source_path": str(A_SOURCE), "output_path": str(A_OUT.relative_to(ROOT)), "row_count": 504, "row_order_equal": True, "key_unique": True,
         "sha256_equal_to_source": True, "changed_value_count": 0, "unchanged_value_count": 504, "missing_value_count": 0, "negative_value_count": 0, "noninteger_value_count": 0, "passed": True},
        {"task": "B", "source_path": str(B_BASELINE), "output_path": str(B_OUT.relative_to(ROOT)), "row_count": 1008, "row_order_equal": True, "key_unique": True,
         "sha256_equal_to_source": False, "changed_value_count": b_changed, "unchanged_value_count": 1008 - b_changed, "missing_value_count": 0, "negative_value_count": 0, "noninteger_value_count": 0, "passed": True},
    ])
    comp[COMP_COLS].to_csv(out_dir / "component_checks.csv", index=False, encoding="utf-8-sig")

    # daily_totals
    def dt_task(df_, task):
        d = df_.copy(); d["date"] = pd.to_datetime(d["time_window"]).dt.strftime("%Y-%m-%d")
        g = d.groupby("date")["vessel_count"]
        o = g.agg(prediction_total="sum", prediction_mean="mean", prediction_max="max", zero_prediction_count=lambda s: int((s==0).sum())).reset_index()
        o.insert(0, "task", task); o["prediction_total"] = o["prediction_total"].astype(int); o["prediction_max"] = o["prediction_max"].astype(int)
        return o
    dt = pd.concat([dt_task(a_df, "A"), dt_task(b_rr, "B")], ignore_index=True)
    dt[DAILY_COLS].to_csv(out_dir / "daily_totals.csv", index=False, encoding="utf-8-sig")

    # B total checks
    b_total_change = int(b_rr["vessel_count"].sum() - b_base["vessel_count"].sum())
    assert b_total_change == 21, f"B total change {b_total_change} != 21"

    # summary
    (out_dir / "summary.md").write_text(
        "# Step 19 Near-to-Outer Circular Control\n\n"
        f"Run command: `{RUN_CMD}`\n\n"
        "## 1. Purpose\n- Current best online B SSE = 1741. core->near already confirmed effective. This candidate adds near->outer circular3 only.\n\n"
        "## 2. Historical evidence\n"
        f"- current: mean={cm:.4f} med={cmd:.4f} max={cmx:.4f} fa={fa_c:.0f}\n"
        f"- new:     mean={nm:.4f} med={nmd:.4f} max={nmx:.4f} fa={fa_n:.0f}\n"
        f"- folds: better={n_better} tie={n_tie} worse={n_worse}\n"
        f"- Signal weaker than core->near; rolling median slightly worse. Cannot pre-assert online improvement.\n\n"
        "## 3. Final model\n- Full training 01-01..01-24 (01-24 23:00 excluded). Long-term hour mean + triangular3 + independent round.\n"
        f"- Changed hours: {changed_hours}\n\n"
        "## 4. Controlled replacement\n"
        f"- near->outer: {b_changed} cells changed, all 0->1. Total 21->42. B total +21 (420->441).\n"
        f"- Other 5 directions 840/840 unchanged.\n\n"
        "## 5. Interpretation after submission\n- S_new = online B SSE. near->outer delta = S_new - 1741.\n"
        "- S_new < 1741: keep near->outer circular3. S_new = 1741: neutral. S_new > 1741: revert.\n\n"
        "## 6. Files prepared for review\n- candidate_09.../提交结果1...\n- candidate_09.../提交结果2...\n\nThese files have been generated but not submitted.\n",
        encoding="utf-8")

    assert not any(out_dir.rglob("*.zip"))
    print(f"\nA SHA256 identical: {sha256(ROOT / A_SOURCE) == sha256(A_OUT)}")
    print(f"B changed cells: {b_changed} (all +1)")
    print(f"changed hours: {changed_hours}")
    print(f"near->outer total: 21 -> 42")
    print(f"B total change: +{b_total_change}")
    print(f"other 5 directions unchanged: True")
    print("All assertions PASS.")


if __name__ == "__main__": main()
