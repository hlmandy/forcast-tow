"""Step 17 — Controlled B core-near hybrid (candidate_07).

Historical offline check: baseline = step10 b_hier_scale (all 6 dirs); hybrid =
core_near from step15 pair_default_hierarchical, other 4 from b_hier_scale.
Final candidate: A byte-copied from step13 (A SSE=4423); B = step12 confirmed
B (SSE=1762) with ONLY core->near and near->core replaced by a freshly-trained
pair_default_hierarchical (full training 2018-01-01..01-23 complete dates).

Outputs (under outputs/step17_controlled_b_hybrid/):
  historical_hybrid_predictions.csv (17856)
  historical_hybrid_scores.csv (18)
  historical_direction_scores.csv (108)
  historical_nonoverlap_scores.csv (6)
  candidate_07_A4423_B_core_near_hybrid/ (A + B CSVs)
  component_checks.csv (2)
  b_value_comparison.csv (1008)
  daily_totals.csv (14)
  summary.md
"""

from __future__ import annotations
import hashlib
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimized_baseline import CN_TO_REGION, REGION_CN  # noqa: E402

STEP10_PREDS = Path("outputs/step10_b_shrinkage_calibration/b_model_predictions.csv")
STEP15_PREDS = Path("outputs/step15_b_pair_decomposition_benchmark/b_model_predictions.csv")
PF_PATH = Path("outputs/step08_b_state_transition_audit/b_hourly_pair_flow.csv")
A_SOURCE = Path("outputs/step13_reordered_new_a_control/candidate_05_newA_oldOrder_newB/提交结果1_区域活跃拖轮数量.csv")
B_SOURCE = Path("outputs/step12_recombined_candidate/提交结果2_圈层间拖轮迁移量.csv")
OUT_REL = Path("outputs/step17_controlled_b_hybrid")

CORE_NEAR_DIRS = ["core->near", "near->core"]
OTHER_DIRS = ["near->outer", "outer->near", "core->outer", "outer->core"]
ALL_DIRS = CORE_NEAR_DIRS + OTHER_DIRS
DIR_RANK = {d: i for i, d in enumerate(ALL_DIRS)}
PAIR_OF = {"core->near": "core_near", "near->core": "core_near", "near->outer": "near_outer",
           "outer->near": "near_outer", "core->outer": "core_outer", "outer->core": "core_outer"}
ROLE = {"core->near": "forward", "near->core": "reverse", "near->outer": "forward",
        "outer->near": "reverse", "core->outer": "forward", "outer->core": "reverse"}
SCENARIOS = [f"fold_{i:02d}" for i in range(1, 9)] + ["final_analog"]
SCN_ORDER = {s: i for i, s in enumerate(SCENARIOS)}

HIST_PRED_COLS = ["evaluation_id", "evaluation_type", "method", "source_method", "date", "hour", "hour_of_day",
                  "source_region", "target_region", "task_key", "score_available", "y_true", "pred_float", "pred_rounded", "error_float", "error_rounded"]
HIST_SCORE_COLS = ["evaluation_id", "evaluation_type", "method", "n_scored_rows", "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded", "actual_total", "predicted_total"]
HIST_DIR_COLS = ["evaluation_id", "evaluation_type", "method", "task_key", "n_scored_rows", "actual_total", "predicted_total", "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded"]
HIST_NO_COLS = ["method", "block", "target_start", "target_end", "n_scored_rows", "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded"]
COMP_COLS = ["task", "source_path", "output_path", "row_count", "row_order_equal", "key_unique", "unchanged_direction_values_equal",
             "replaced_direction_values_verified", "sha256_equal_to_source", "missing_value_count", "negative_value_count", "noninteger_value_count", "passed"]
CMP_COLS = ["row_number", "time_window", "source_zone", "target_zone", "task_key", "baseline_vessel_count", "hybrid_vessel_count", "difference", "was_replaced", "value_unchanged_when_not_replaced"]
DAILY_COLS = ["task", "date", "prediction_total", "prediction_mean", "prediction_max", "zero_prediction_count"]

RUN_CMD = "python " + " ".join(sys.argv)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def smooth3(v):
    out = np.zeros(24)
    for h in range(24):
        out[h] = (v[(h - 1) % 24] + 2 * v[h] + v[(h + 1) % 24]) / 4
    return np.clip(out, 0, None)


def alloc_hours(K, p):
    fa = K * p
    base = np.floor(fa).astype(int)
    rem = K - int(base.sum())
    if rem > 0:
        frac = fa - base
        order = np.lexsort((np.arange(24), -fa, -frac))
        for j in range(rem):
            base[order[j]] += 1
    return base


def train_core_near(pf, complete_dates):
    dab, dba = "core->near", "near->core"
    sub = pf[pf["date"].isin(complete_dates) & pf["task_key"].isin([dab, dba])]
    fwd_df = sub[sub["task_key"] == dab].pivot_table(index="date", columns="hour_of_day", values="y").reindex(index=complete_dates, columns=range(24)).fillna(0)
    rev_df = sub[sub["task_key"] == dba].pivot_table(index="date", columns="hour_of_day", values="y").reindex(index=complete_dates, columns=range(24)).fillna(0)
    fwd = fwd_df.to_numpy(float); rev = rev_df.to_numpy(float)
    X = fwd + rev
    T = X.sum(axis=1)
    X_h = X.sum(axis=0); Yfwd_h = fwd.sum(axis=0); sum_X = X_h.sum()
    r_global = float(Yfwd_h.sum() / sum_X) if sum_X > 0 else 0.5
    T_hat = float(T.mean())
    p_day = np.where(T[:, None] > 0, X / np.where(T[:, None] > 0, T[:, None], 1.0), 0.0)
    mask = T > 0
    de_raw = p_day[mask].mean(axis=0) if mask.any() else np.full(24, 1 / 24)
    de_raw = de_raw / de_raw.sum() if de_raw.sum() > 0 else np.full(24, 1 / 24)
    de_tri = smooth3(de_raw); de_tri = de_tri / de_tri.sum()
    r_h = (Yfwd_h + 20 * r_global) / (X_h + 20)
    r_h = np.clip(r_h, 0, 1)
    return T_hat, de_tri, r_h


def predict_core_near(T_hat, p_hat, r_hat):
    K = int(round(T_hat))
    alloc = alloc_hours(K, p_hat)
    assert alloc.sum() == K
    fwd = np.clip(np.rint(alloc * r_hat), 0, alloc).astype(int)
    rev = alloc - fwd
    assert (fwd + rev == alloc).all()
    return fwd, rev  # 24-vectors


def main():
    out_dir = ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    cand = out_dir / "candidate_07_A4423_B_core_near_hybrid"
    cand.mkdir(exist_ok=True)
    print(f"out_dir = {out_dir}")

    # ============ HISTORICAL OFFLINE ============
    s10 = pd.read_csv(ROOT / STEP10_PREDS, encoding="utf-8-sig")
    s10["date"] = pd.to_datetime(s10["date"]).dt.normalize()
    s15 = pd.read_csv(ROOT / STEP15_PREDS, encoding="utf-8-sig")
    s15["date"] = pd.to_datetime(s15["date"]).dt.normalize()

    base_src = s10[(s10["predictor_method"] == "pair_hour_hier_tau_scale_cv") & (s10["rounding_method"] == "independent")].copy()
    pdh = s15[s15["method"] == "pair_default_hierarchical"].copy()

    def mk(method_name, core_near_df, other_df):
        cn = core_near_df[core_near_df["task_key"].isin(CORE_NEAR_DIRS)].copy()
        ot = other_df[other_df["task_key"].isin(OTHER_DIRS)].copy()
        cn["source_method"] = "pair_default_hierarchical(step15)"; ot["source_method"] = "b_hier_scale(step10)"
        df = pd.concat([cn, ot], ignore_index=True)
        df["method"] = method_name
        df["error_float"] = np.where(df["score_available"], df["pred_float"] - df["y_true"], np.nan)
        df["error_rounded"] = np.where(df["score_available"], df["pred_rounded"] - df["y_true"], np.nan)
        return df

    baseline = mk("baseline_confirmed_structure", base_src, base_src)
    baseline["source_method"] = "b_hier_scale(step10)"
    hybrid = mk("hybrid_confirmed_core_near", pdh, base_src)
    hist_preds = pd.concat([baseline[HIST_PRED_COLS], hybrid[HIST_PRED_COLS]], ignore_index=True)
    assert len(hist_preds) == 17856, f"hist preds {len(hist_preds)}"

    # scores
    def score_group(df, method):
        rows = []
        for eid in SCENARIOS:
            sc = df[(df["evaluation_id"] == eid) & (df["method"] == method) & df["score_available"]]
            er = sc["error_rounded"].astype(float).to_numpy()
            rows.append({"evaluation_id": eid, "evaluation_type": sc["evaluation_type"].iloc[0] if len(sc) else "", "method": method,
                         "n_scored_rows": int(len(sc)), "sse_rounded": float((er ** 2).sum()), "mse_rounded": float((er ** 2).mean()),
                         "mae_rounded": float(np.abs(er).mean()), "mean_bias_rounded": float(er.mean()),
                         "actual_total": float(sc["y_true"].sum()), "predicted_total": float(sc["pred_rounded"].sum())})
        return rows
    hist_scores = pd.DataFrame(score_group(hist_preds, "baseline_confirmed_structure") + score_group(hist_preds, "hybrid_confirmed_core_near"))

    # direction scores
    drows = []
    for method in ["baseline_confirmed_structure", "hybrid_confirmed_core_near"]:
        for eid in SCENARIOS:
            for d in ALL_DIRS:
                sd = hist_preds[(hist_preds["evaluation_id"] == eid) & (hist_preds["method"] == method) & (hist_preds["task_key"] == d) & hist_preds["score_available"]]
                er = sd["error_rounded"].astype(float).to_numpy()
                drows.append({"evaluation_id": eid, "evaluation_type": sd["evaluation_type"].iloc[0] if len(sd) else "", "method": method, "task_key": d,
                              "n_scored_rows": int(len(sd)), "actual_total": float(sd["y_true"].sum()), "predicted_total": float(sd["pred_rounded"].sum()),
                              "sse_rounded": float((er ** 2).sum()), "mse_rounded": float((er ** 2).mean()) if len(er) else np.nan,
                              "mae_rounded": float(np.abs(er).mean()) if len(er) else np.nan, "mean_bias_rounded": float(er.mean()) if len(er) else np.nan})
    hist_dir = pd.DataFrame(drows)

    # nonoverlap
    norows = []
    for method in ["baseline_confirmed_structure", "hybrid_confirmed_core_near"]:
        for block, eids in [("nonoverlap_block_1", ["fold_01"]), ("nonoverlap_block_2", ["fold_08"]), ("nonoverlap_combined", ["fold_01", "fold_08"])]:
            sub = hist_preds[(hist_preds["method"] == method) & (hist_preds["evaluation_id"].isin(eids)) & hist_preds["score_available"]]
            er = sub["error_rounded"].astype(float).to_numpy()
            dts = pd.to_datetime(sub["date"].unique())
            norows.append({"method": method, "block": block, "target_start": f"{dts.min():%Y-%m-%d}" if len(dts) else "", "target_end": f"{dts.max():%Y-%m-%d}" if len(dts) else "",
                           "n_scored_rows": int(len(sub)), "sse_rounded": float((er ** 2).sum()), "mse_rounded": float((er ** 2).mean()) if len(er) else np.nan,
                           "mae_rounded": float(np.abs(er).mean()) if len(er) else np.nan, "mean_bias_rounded": float(er.mean()) if len(er) else np.nan})
    hist_no = pd.DataFrame(norows)

    # ---- verify historical expectations ----
    roll = hist_scores[hist_scores["evaluation_type"] == "rolling_7d"]
    def mstats(m):
        r = roll[roll["method"] == m]["mse_rounded"]
        return float(r.mean()), float(r.median()), float(r.max())
    bm, bmd, bmx = mstats("baseline_confirmed_structure")
    hm, hmd, hmx = mstats("hybrid_confirmed_core_near")
    fa_b = hist_scores[(hist_scores["evaluation_id"] == "final_analog") & (hist_scores["method"] == "baseline_confirmed_structure")]["sse_rounded"].iloc[0]
    fa_h = hist_scores[(hist_scores["evaluation_id"] == "final_analog") & (hist_scores["method"] == "hybrid_confirmed_core_near")]["sse_rounded"].iloc[0]
    print(f"baseline: mean={bm:.4f} med={bmd:.4f} max={bmx:.4f} fa={fa_b:.0f}")
    print(f"hybrid:   mean={hm:.4f} med={hmd:.4f} max={hmx:.4f} fa={fa_h:.0f}")
    # fold wins
    piv = roll.pivot_table(index="evaluation_id", columns="method", values="sse_rounded", aggfunc="first")
    wins = int((piv["hybrid_confirmed_core_near"] < piv["baseline_confirmed_structure"]).sum())
    print(f"hybrid beats baseline in {wins}/8 rolling folds")
    assert wins == 7, f"expected 7/8 folds, got {wins}"
    assert fa_h <= 738, f"final analog {fa_h} > 738"
    # direction check: other 4 identical
    dr_roll = hist_dir[hist_dir["evaluation_type"] == "rolling_7d"].groupby(["method", "task_key"])["sse_rounded"].sum().unstack()
    for d in OTHER_DIRS:
        assert dr_roll.loc["baseline_confirmed_structure", d] == dr_roll.loc["hybrid_confirmed_core_near", d], f"{d} changed"
    print(f"core->near: baseline {dr_roll.loc['baseline_confirmed_structure','core->near']:.0f} -> hybrid {dr_roll.loc['hybrid_confirmed_core_near','core->near']:.0f}")
    print(f"near->core: baseline {dr_roll.loc['baseline_confirmed_structure','near->core']:.0f} -> hybrid {dr_roll.loc['hybrid_confirmed_core_near','near->core']:.0f}")

    # write historical
    hp = hist_preds.copy()
    hp["_e"] = hp["evaluation_id"].map(SCN_ORDER); hp["_m"] = hp["method"].map({"baseline_confirmed_structure": 0, "hybrid_confirmed_core_near": 1})
    hp = hp.sort_values(["_e", "_m", "hour", "source_region", "target_region"]).drop(columns=["_e", "_m"])
    hp["date"] = hp["date"].dt.strftime("%Y-%m-%d")
    hp[HIST_PRED_COLS].to_csv(out_dir / "historical_hybrid_predictions.csv", index=False, encoding="utf-8-sig")
    hist_scores.sort_values(["evaluation_id", "method"])[HIST_SCORE_COLS].to_csv(out_dir / "historical_hybrid_scores.csv", index=False, encoding="utf-8-sig")
    hd = hist_dir.copy(); hd["_e"] = hd["evaluation_id"].map(SCN_ORDER); hd["_m"] = hd["method"].map({"baseline_confirmed_structure": 0, "hybrid_confirmed_core_near": 1}); hd["_d"] = hd["task_key"].map(DIR_RANK)
    hd.sort_values(["_e", "_m", "_d"])[HIST_DIR_COLS].to_csv(out_dir / "historical_direction_scores.csv", index=False, encoding="utf-8-sig")
    hist_no[HIST_NO_COLS].to_csv(out_dir / "historical_nonoverlap_scores.csv", index=False, encoding="utf-8-sig")

    # ============ FINAL CANDIDATE ============
    # A: byte-copy from step13
    A_OUT = cand / "提交结果1_区域活跃拖轮数量.csv"
    shutil.copyfile(ROOT / A_SOURCE, A_OUT)
    a_sha_src, a_sha_out = sha256(ROOT / A_SOURCE), sha256(A_OUT)
    assert a_sha_src == a_sha_out, "A SHA mismatch"
    a_df = pd.read_csv(A_OUT)
    assert len(a_df) == 504 and int(a_df["vessel_count"].sum()) == 2241

    # B: read step12 confirmed, replace core_near
    pf = pd.read_csv(ROOT / PF_PATH, encoding="utf-8-sig")
    pf["hour"] = pd.to_datetime(pf["hour"]); pf["date"] = pd.to_datetime(pf["date"]).dt.normalize(); pf["hour_of_day"] = pf["hour"].dt.hour
    complete_dates = list(pd.date_range("2018-01-01", "2018-01-23", freq="D").normalize())
    T_hat, p_hat, r_hat = train_core_near(pf, complete_dates)
    fwd24, rev24 = predict_core_near(T_hat, p_hat, r_hat)
    print(f"core_near: T_hat={T_hat:.3f} K={int(round(T_hat))} fwd_total={fwd24.sum()} rev_total={rev24.sum()}")

    b_src = pd.read_csv(ROOT / B_SOURCE)
    assert len(b_src) == 1008
    assert not b_src.duplicated(["time_window", "source_zone", "target_zone"]).any()
    b_ts = pd.to_datetime(b_src["time_window"])
    # build replacement dict: (ts, source_zone, target_zone) -> value, for core_near
    replace = {}
    for i in range(len(b_src)):
        sz, tz = b_src["source_zone"].iloc[i], b_src["target_zone"].iloc[i]
        if (sz, tz) in [(REGION_CN["core"], REGION_CN["near"]), (REGION_CN["near"], REGION_CN["core"])]:
            h = b_ts.iloc[i].hour
            val = int(fwd24[h]) if sz == REGION_CN["core"] else int(rev24[h])
            replace[(b_ts.iloc[i], sz, tz)] = val
    # apply
    b_out = b_src.copy()
    new_vals = []
    n_changed = 0
    for i in range(len(b_src)):
        key = (b_ts.iloc[i], b_src["source_zone"].iloc[i], b_src["target_zone"].iloc[i])
        if key in replace:
            new_vals.append(replace[key]); n_changed += int(replace[key] != b_src["vessel_count"].iloc[i])
        else:
            new_vals.append(int(b_src["vessel_count"].iloc[i]))
    b_out["vessel_count"] = new_vals
    B_OUT = cand / "提交结果2_圈层间拖轮迁移量.csv"
    b_out.to_csv(B_OUT, index=False, encoding="utf-8-sig")

    # ---- component checks ----
    # B: row order preserved, 4 unchanged directions identical, core_near replaced
    b_reread = pd.read_csv(B_OUT)
    assert len(b_reread) == 1008
    assert (b_reread["time_window"].to_numpy() == b_src["time_window"].to_numpy()).all()
    assert (b_reread["source_zone"].to_numpy() == b_src["source_zone"].to_numpy()).all()
    assert (b_reread["target_zone"].to_numpy() == b_src["target_zone"].to_numpy()).all()
    # 4 unchanged
    for i in range(len(b_src)):
        sz, tz = b_src["source_zone"].iloc[i], b_src["target_zone"].iloc[i]
        if (sz, tz) not in [(REGION_CN["core"], REGION_CN["near"]), (REGION_CN["near"], REGION_CN["core"])]:
            assert b_reread["vessel_count"].iloc[i] == b_src["vessel_count"].iloc[i], f"row {i} changed"
    # core_near replaced == new prediction
    for i in range(len(b_src)):
        sz, tz = b_src["source_zone"].iloc[i], b_src["target_zone"].iloc[i]
        if (sz, tz) in [(REGION_CN["core"], REGION_CN["near"]), (REGION_CN["near"], REGION_CN["core"])]:
            h = b_ts.iloc[i].hour
            expected = int(fwd24[h]) if sz == REGION_CN["core"] else int(rev24[h])
            assert b_reread["vessel_count"].iloc[i] == expected, f"row {i} core_near mismatch"
    assert b_reread["vessel_count"].notna().all() and (b_reread["vessel_count"] >= 0).all() and (b_reread["vessel_count"] % 1 == 0).all()

    comp = pd.DataFrame([
        {"task": "A", "source_path": str(A_SOURCE), "output_path": str(A_OUT.relative_to(ROOT)), "row_count": 504, "row_order_equal": True,
         "key_unique": True, "unchanged_direction_values_equal": True, "replaced_direction_values_verified": True,
         "sha256_equal_to_source": True, "missing_value_count": 0, "negative_value_count": 0, "noninteger_value_count": 0, "passed": True},
        {"task": "B", "source_path": str(B_SOURCE), "output_path": str(B_OUT.relative_to(ROOT)), "row_count": 1008, "row_order_equal": True,
         "key_unique": True, "unchanged_direction_values_equal": True, "replaced_direction_values_verified": True,
         "sha256_equal_to_source": False, "missing_value_count": 0, "negative_value_count": 0, "noninteger_value_count": 0, "passed": True},
    ])
    comp[COMP_COLS].to_csv(out_dir / "component_checks.csv", index=False, encoding="utf-8-sig")

    # ---- b_value_comparison ----
    cmp_rows = []
    for i in range(len(b_src)):
        sz, tz = b_src["source_zone"].iloc[i], b_src["target_zone"].iloc[i]
        is_cn = (sz, tz) in [(REGION_CN["core"], REGION_CN["near"]), (REGION_CN["near"], REGION_CN["core"])]
        task_key = f"{CN_TO_REGION[sz]}->{CN_TO_REGION[tz]}"
        bv = int(b_src["vessel_count"].iloc[i]); hv = int(b_reread["vessel_count"].iloc[i])
        cmp_rows.append({"row_number": i + 1, "time_window": b_src["time_window"].iloc[i], "source_zone": sz, "target_zone": tz,
                         "task_key": task_key, "baseline_vessel_count": bv, "hybrid_vessel_count": hv, "difference": hv - bv,
                         "was_replaced": bool(is_cn), "value_unchanged_when_not_replaced": bool(is_cn or hv == bv)})
    bcmp = pd.DataFrame(cmp_rows)
    assert (bcmp[~bcmp["was_replaced"]]["difference"] == 0).all(), "non-core_near direction changed!"
    bcmp[CMP_COLS].to_csv(out_dir / "b_value_comparison.csv", index=False, encoding="utf-8-sig")

    # ---- daily totals ----
    def daily_task(df, task):
        d = df.copy(); d["date"] = pd.to_datetime(d["time_window"]).dt.strftime("%Y-%m-%d")
        g = d.groupby("date")["vessel_count"]
        out = g.agg(prediction_total="sum", prediction_mean="mean", prediction_max="max", zero_prediction_count=lambda s: int((s == 0).sum())).reset_index()
        out.insert(0, "task", task)
        out["prediction_total"] = out["prediction_total"].astype(int); out["prediction_max"] = out["prediction_max"].astype(int)
        return out
    dt = pd.concat([daily_task(a_df, "A"), daily_task(b_reread, "B")], ignore_index=True)
    dt[DAILY_COLS].to_csv(out_dir / "daily_totals.csv", index=False, encoding="utf-8-sig")

    # ---- summary ----
    cn_changed = int(bcmp[bcmp["was_replaced"]]["difference"].abs() > 0).sum() if False else int((bcmp[bcmp["was_replaced"]]["difference"] != 0).sum())
    write_summary(out_dir / "summary.md", bm, bmd, bmx, fa_b, hm, hmd, hmx, fa_h, wins, dr_roll, b_src, b_reread, fwd24, rev24, n_changed, cn_changed)

    # no zip
    assert not any(out_dir.rglob("*.zip"))
    print("\nAll assertions PASS.")


def write_summary(path, bm, bmd, bmx, fa_b, hm, hmd, hmx, fa_h, wins, dr_roll, b_src, b_reread, fwd24, rev24, n_changed, cn_changed):
    L = []
    L.append("# Step 17 Controlled B Core-Near Hybrid")
    L.append("")
    L.append(f"Run command: `{RUN_CMD}`")
    L.append("")
    L.append("## 1. Purpose\n- A fixed to the online-confirmed A SSE=4423 file (step13). Baseline B = online-confirmed B SSE=1762 file. New candidate replaces ONLY core->near and near->core; the other four B directions are byte-identical. Strict controlled experiment.\n")
    L.append("## 2. Historical offline comparison\n")
    L.append("| method | rolling mean | median | max | final | nonoverlap |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    L.append(f"| baseline_confirmed_structure | {bm:.4f} | {bmd:.4f} | {bmx:.4f} | {fa_b:.0f} | (see csv) |")
    L.append(f"| hybrid_confirmed_core_near | {hm:.4f} | {hmd:.4f} | {hmx:.4f} | {fa_h:.0f} | (see csv) |")
    L.append(f"\n- Hybrid beats baseline in {wins}/8 rolling folds. core->near {dr_roll.loc['baseline_confirmed_structure','core->near']:.0f}->{dr_roll.loc['hybrid_confirmed_core_near','core->near']:.0f}; near->core {dr_roll.loc['baseline_confirmed_structure','near->core']:.0f}->{dr_roll.loc['hybrid_confirmed_core_near','near->core']:.0f}. Other 4 directions identical.\n")
    L.append("## 3. Final core-near training\n- Complete training dates 2018-01-01..01-23; long_mean daily total; triangular3 profile; tau=20 direction ratio; pair-hierarchical rounding.\n")
    L.append(f"- core->near total over 7 days = {int(fwd24.sum()*7)}; near->core total = {int(rev24.sum()*7)} (per-day forward {int(fwd24.sum())} / reverse {int(rev24.sum())}).\n")
    L.append(f"## 4. Controlled value replacement\n- B cells changed (core_near): {cn_changed}/336. Mean |diff| and max diff in b_value_comparison.csv. Other 4 directions 672/672 unchanged.\n")
    L.append("## 5. File-order preservation\n- A keeps the verified old zone order; B keeps the online 1762 file row order. No template re-sort, no Excel re-save.\n")
    L.append("## 6. Interpretation\n- The candidate's online total-score difference vs the current confirmed submission comes ONLY from B's two core_near directions. A should remain SSE=4423; the other four B directions are unchanged. Whether online B SSE improves still requires submission. No online improvement is claimed from offline results.\n")
    L.append("## 7. Files prepared for review\n- candidate_07_A4423_B_core_near_hybrid/提交结果1_区域活跃拖轮数量.csv\n- candidate_07_A4423_B_core_near_hybrid/提交结果2_圈层间拖轮迁移量.csv\n\nThese files have been generated but not submitted.\n")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
