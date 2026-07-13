"""Step 21 — Joint pseudo-online backtest framework.

Establishes the competition-simulating pipeline: for each fold, A and B are
predicted independently from training data only, scored jointly as
SSE_A + 3·SSE_B. Candidate_10 components are the baseline:
  A = step07 decomp_mean_daytype_shrunk / exclude_outage_severe
  B = step10 b_hier_scale (5 dirs) + step15 pair_default_hierarchical (near->core)

Four evaluation views: non-overlapping blocks, final_analog, rolling folds
(8, auxiliary), forecast horizon 1-7. Normal vs anomaly split.

Outputs (under outputs/step21_joint_backtest_framework/):
  1. joint_fold_scores.csv       (9)
  2. joint_nonoverlap_scores.csv (3)
  3. joint_horizon_scores.csv    (56)
  4. joint_direction_a_sse.csv   (54)
  5. joint_direction_b_sse.csv   (54)
  6. summary.md
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
STEP07 = Path("outputs/step07_a_regularized_count_models/a_model_predictions.csv")
STEP10 = Path("outputs/step10_b_shrinkage_calibration/b_model_predictions.csv")
STEP15 = Path("outputs/step15_b_pair_decomposition_benchmark/b_model_predictions.csv")
STEP02 = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
OUT_REL = Path("outputs/step21_joint_backtest_framework")

SCENARIOS = [f"fold_{i:02d}" for i in range(1, 9)] + ["final_analog"]
SCN_ORDER = {s: i for i, s in enumerate(SCENARIOS)}
A_REGIONS = ["core", "near", "outer"]
B_DIRS = ["core->near", "near->core", "core->outer", "outer->core", "near->outer", "outer->near"]
RUN_CMD = "python " + " ".join(sys.argv)


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    # ---- A predictions (step07: decomp_mean_daytype_shrunk / exclude_outage_severe) ----
    a7 = pd.read_csv(ROOT / STEP07, encoding="utf-8-sig")
    a7["date"] = pd.to_datetime(a7["date"]).dt.normalize()
    a_pred = a7[(a7["method"] == "decomp_mean_daytype_shrunk") & (a7["training_policy"] == "exclude_outage_severe")].copy()
    a_pred["forecast_horizon"] = a_pred.groupby("evaluation_id")["date"].rank(method="dense").astype(int)

    # ---- B predictions (candidate_10: 5 dirs from step10 b_hier_scale, near->core from step15) ----
    b10 = pd.read_csv(ROOT / STEP10, encoding="utf-8-sig")
    b10["date"] = pd.to_datetime(b10["date"]).dt.normalize()
    b15 = pd.read_csv(ROOT / STEP15, encoding="utf-8-sig")
    b15["date"] = pd.to_datetime(b15["date"]).dt.normalize()

    b_base = b10[(b10["predictor_method"] == "pair_hour_hier_tau_scale_cv") & (b10["rounding_method"] == "independent")].copy()
    b_nc = b15[b15["method"] == "pair_default_hierarchical"].copy()
    # assemble: near->core from step15, other 5 from step10
    b_other = b_base[b_base["task_key"] != "near->core"].copy()
    b_nc_only = b_nc[b_nc["task_key"] == "near->core"].copy()
    key_cols_b = ["evaluation_id", "date", "hour_of_day", "source_region", "target_region"]
    b_pred = pd.concat([b_other, b_nc_only[key_cols_b + ["task_key", "score_available", "y_true", "pred_float", "pred_rounded"]]], ignore_index=True)
    b_pred["forecast_horizon"] = b_pred.groupby("evaluation_id")["date"].rank(method="dense").astype(int)

    # ---- quality map ----
    aud = pd.read_csv(ROOT / STEP02, encoding="utf-8-sig"); aud["date"] = pd.to_datetime(aud["date"]).dt.normalize()
    qmap = dict(zip(aud["date"], aud["quality_regime"]))

    # ---- joint fold scores ----
    fold_rows = []
    for eid in SCENARIOS:
        a_sc = a_pred[a_pred["evaluation_id"] == eid]  # A has no score_available; all scored
        b_sc = b_pred[(b_pred["evaluation_id"] == eid) & b_pred["score_available"]]
        if len(a_sc) == 0 or len(b_sc) == 0:
            continue
        er_a = (a_sc["pred_rounded"] - a_sc["y_true"]).astype(float).to_numpy()
        er_b = (b_sc["pred_rounded"] - b_sc["y_true"]).astype(float).to_numpy()
        sse_a = float((er_a ** 2).sum()); sse_b = float((er_b ** 2).sum())
        total = sse_a + 3 * sse_b
        fold_rows.append({
            "evaluation_id": eid, "evaluation_type": a_sc["evaluation_type"].iloc[0],
            "n_scored_a": int(len(a_sc)), "n_scored_b": int(len(b_sc)),
            "sse_a": sse_a, "sse_b": sse_b, "total_sse": total,
            "mse_a": sse_a / len(a_sc) if len(a_sc) else np.nan,
            "mse_b": sse_b / len(b_sc) if len(b_sc) else np.nan,
            "actual_a_total": float(a_sc["y_true"].sum()), "predicted_a_total": float(a_sc["pred_rounded"].sum()),
            "actual_b_total": float(b_sc["y_true"].sum()), "predicted_b_total": float(b_sc["pred_rounded"].sum()),
        })
    fold_scores = pd.DataFrame(fold_rows)

    # ---- non-overlap ----
    no_rows = []
    for block, eids in [("block_1", ["fold_01"]), ("block_2", ["fold_08"]), ("combined", ["fold_01", "fold_08"])]:
        sub_a = a_pred[(a_pred["evaluation_id"].isin(eids)) & True]
        sub_b = b_pred[(b_pred["evaluation_id"].isin(eids)) & b_pred["score_available"]]
        era = (sub_a["pred_rounded"] - sub_a["y_true"]).astype(float).to_numpy()
        erb = (sub_b["pred_rounded"] - sub_b["y_true"]).astype(float).to_numpy()
        sa = float((era ** 2).sum()); sb = float((erb ** 2).sum())
        no_rows.append({"block": block, "n_a": int(len(sub_a)), "n_b": int(len(sub_b)),
                        "sse_a": sa, "sse_b": sb, "total_sse": sa + 3 * sb,
                        "mse_a": sa / len(sub_a) if len(sub_a) else np.nan, "mse_b": sb / len(sub_b) if len(sub_b) else np.nan})
    nonoverlap = pd.DataFrame(no_rows)

    # ---- horizon (rolling folds only, 1-7) ----
    hz_rows = []
    for eid in [f"fold_{i:02d}" for i in range(1, 9)]:
        for hz in range(1, 8):
            sa_sc = a_pred[(a_pred["evaluation_id"] == eid) & (a_pred["forecast_horizon"] == hz) & True]
            sb_sc = b_pred[(b_pred["evaluation_id"] == eid) & (b_pred["forecast_horizon"] == hz) & b_pred["score_available"]]
            era = (sa_sc["pred_rounded"] - sa_sc["y_true"]).astype(float).to_numpy()
            erb = (sb_sc["pred_rounded"] - sb_sc["y_true"]).astype(float).to_numpy()
            hz_rows.append({"evaluation_id": eid, "forecast_horizon": hz,
                            "sse_a": float((era ** 2).sum()), "sse_b": float((erb ** 2).sum()),
                            "total_sse": float((era ** 2).sum()) + 3 * float((erb ** 2).sum())})
    # final_analog horizon 1-6
    for hz in range(1, 7):
        sa_sc = a_pred[(a_pred["evaluation_id"] == "final_analog") & (a_pred["forecast_horizon"] == hz) & True]
        sb_sc = b_pred[(b_pred["evaluation_id"] == "final_analog") & (b_pred["forecast_horizon"] == hz) & b_pred["score_available"]]
        era = (sa_sc["pred_rounded"] - sa_sc["y_true"]).astype(float).to_numpy()
        erb = (sb_sc["pred_rounded"] - sb_sc["y_true"]).astype(float).to_numpy()
        hz_rows.append({"evaluation_id": "final_analog", "forecast_horizon": hz,
                        "sse_a": float((era ** 2).sum()), "sse_b": float((erb ** 2).sum()),
                        "total_sse": float((era ** 2).sum()) + 3 * float((erb ** 2).sum())})
    horizons = pd.DataFrame(hz_rows)

    # ---- per-direction A SSE (per fold) ----
    da_rows = []
    for eid in SCENARIOS:
        for r in A_REGIONS:
            sc = a_pred[(a_pred["evaluation_id"] == eid) & (a_pred["region"] == r) & True]
            er = (sc["pred_rounded"] - sc["y_true"]).astype(float).to_numpy()
            da_rows.append({"evaluation_id": eid, "region": r, "sse_a": float((er ** 2).sum()), "n": int(len(sc))})
    dir_a = pd.DataFrame(da_rows)

    # ---- per-direction B SSE (per fold) ----
    db_rows = []
    for eid in SCENARIOS:
        for d in B_DIRS:
            sc = b_pred[(b_pred["evaluation_id"] == eid) & (b_pred["task_key"] == d) & b_pred["score_available"]]
            er = (sc["pred_rounded"] - sc["y_true"]).astype(float).to_numpy()
            db_rows.append({"evaluation_id": eid, "task_key": d, "sse_b": float((er ** 2).sum()), "n": int(len(sc))})
    dir_b = pd.DataFrame(db_rows)

    # ---- write ----
    fold_scores.to_csv(out_dir / "joint_fold_scores.csv", index=False, encoding="utf-8-sig")
    nonoverlap.to_csv(out_dir / "joint_nonoverlap_scores.csv", index=False, encoding="utf-8-sig")
    horizons.to_csv(out_dir / "joint_horizon_scores.csv", index=False, encoding="utf-8-sig")
    dir_a.to_csv(out_dir / "joint_direction_a_sse.csv", index=False, encoding="utf-8-sig")
    dir_b.to_csv(out_dir / "joint_direction_b_sse.csv", index=False, encoding="utf-8-sig")

    # ---- summary ----
    rolling = fold_scores[fold_scores["evaluation_type"] == "rolling_7d"]
    fa = fold_scores[fold_scores["evaluation_id"] == "final_analog"]
    rmean = float(rolling["total_sse"].mean()); rmed = float(rolling["total_sse"].median()); rmax = float(rolling["total_sse"].max())
    fa_tot = float(fa["total_sse"].iloc[0]) if len(fa) else np.nan
    no_c = nonoverlap[nonoverlap["block"] == "combined"]
    no_tot = float(no_c["total_sse"].iloc[0]) if len(no_c) else np.nan

    # horizon trend
    hz_roll = horizons[horizons["evaluation_id"].str.startswith("fold")].groupby("forecast_horizon")["total_sse"].mean()

    L = []
    L.append("# Step 21 Joint Backtest Framework")
    L.append("")
    L.append(f"Run command: `{RUN_CMD}`")
    L.append("")
    L.append("## Baseline: candidate_10")
    L.append(f"- A = step07 decomp_mean_daytype_shrunk / exclude_outage_severe (A MSE ~{float(rolling['mse_a'].mean()):.2f})")
    L.append(f"- B = step10 b_hier_scale (5 dirs) + step15 pair_default_hierarchical (near->core) (B MSE ~{float(rolling['mse_b'].mean()):.3f})")
    L.append(f"- Joint objective: SSE_A + 3*SSE_B")
    L.append("")
    L.append("## Rolling 8 folds (auxiliary)")
    L.append("")
    L.append("| fold | SSE_A | SSE_B | Total |")
    L.append("| --- | --- | --- | --- |")
    for _, r in rolling.iterrows():
        L.append(f"| {r['evaluation_id']} | {r['sse_a']:.0f} | {r['sse_b']:.0f} | {r['total_sse']:.0f} |")
    L.append(f"\n- mean={rmean:.0f} median={rmed:.0f} max={rmax:.0f}")
    L.append("")
    L.append("## Non-overlapping blocks (primary)")
    L.append("")
    L.append("| block | SSE_A | SSE_B | Total |")
    L.append("| --- | --- | --- | --- |")
    for _, r in nonoverlap.iterrows():
        L.append(f"| {r['block']} | {r['sse_a']:.0f} | {r['sse_b']:.0f} | {r['total_sse']:.0f} |")
    L.append("")
    L.append("## Final analog")
    L.append(f"- SSE_A={float(fa['sse_a'].iloc[0]):.0f} SSE_B={float(fa['sse_b'].iloc[0]):.0f} Total={fa_tot:.0f}")
    L.append("")
    L.append("## Forecast horizon (rolling mean total SSE per day)")
    L.append("")
    L.append("| horizon | mean total SSE |")
    L.append("| --- | --- |")
    for hz in range(1, 8):
        L.append(f"| {hz} | {hz_roll.get(hz, np.nan):.1f} |")
    L.append("")
    L.append("## A dominates")
    L.append(f"- A share of rolling mean total: {float(rolling['sse_a'].mean()) / rmean * 100:.0f}%")
    L.append(f"- B weighted share: {3 * float(rolling['sse_b'].mean()) / rmean * 100:.0f}%")
    L.append("- A optimization (daily vessel count utilization) is the primary lever.")
    L.append("")
    L.append("## Next steps")
    L.append("- Step 22: A daily vessel count diagnostic (U_d, H_d, c_d stability)")
    L.append("- Step 23: A vessel-constrained three-layer model")
    L.append("- Step 24: candidate_10 B direction-hour OOF residual diagnostics")
    L.append("- Step 25: A-prediction-driven B migration rate model")
    L.append("")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    print("\n=== Output files ===")
    for p in sorted(out_dir.glob("*.csv")) + [out_dir / "summary.md"]:
        rows = len(pd.read_csv(p, encoding="utf-8-sig")) if p.suffix == ".csv" else sum(1 for _ in open(p, encoding="utf-8"))
        print(f"  {p.relative_to(ROOT)}  rows={rows}")
    print(f"\nRolling: mean={rmean:.0f} median={rmed:.0f} max={rmax:.0f}")
    print(f"Final analog total: {fa_tot:.0f}")
    print(f"Non-overlap combined total: {no_tot:.0f}")
    print(f"A share: {float(rolling['sse_a'].mean())/rmean*100:.0f}%")
    print("\nDone.")


if __name__ == "__main__":
    main()
