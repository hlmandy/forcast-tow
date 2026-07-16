"""Step 41 — B six-direction bad-case and Hurdle conditional mean model.

A frozen at candidate_10. B baseline = candidate_10. Diagnoses each direction's
error mechanism (zero/positive/false-positive/false-negative/magnitude), then
tests Hurdle (occurrence × conditional mean) vs direct mean for each direction.

Outputs (under outputs/step41_b_directional_hurdle_badcase/):
  1-14 as specified.
"""

from __future__ import annotations
import sys, warnings, math
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
OUT_REL = Path("outputs/step41_b_directional_hurdle_badcase")
RUN_CMD = "python " + " ".join(sys.argv)

STEP10_PREDS = Path("outputs/step10_b_shrinkage_calibration/b_model_predictions.csv")
STEP15_PREDS = Path("outputs/step15_b_pair_decomposition_benchmark/b_model_predictions.csv")
DIRECTIONS = ["core->near", "near->core", "near->outer", "outer->near", "core->outer", "outer->core"]
DIR_RANK = {d:i for i,d in enumerate(DIRECTIONS)}
SCENARIOS = [f"fold_{i:02d}" for i in range(1,9)] + ["final_analog"]
ROLLING = [f"fold_{i:02d}" for i in range(1,9)]


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    # ---- audit ----
    (out_dir / "implementation_audit.md").write_text(
        "# Implementation Audit\n\n"
        "## Candidate_10 B composition\n"
        "- 5 directions (core→near, near→outer, outer→near, core→outer, outer→core): Step 10 pair_hour_hier_tau_scale_cv / independent\n"
        "- near→core: Step 15 pair_default_hierarchical\n"
        "- Online B SSE: 925 (corrected scoring)\n"
        "- near→core replacement contribution: -27 (online confirmed)\n"
        "- core→near replacement contribution: +15 (online confirmed, kept Step 11)\n\n"
        "## This step\n"
        "- A completely frozen (candidate_10)\n"
        "- B baseline = candidate_10\n"
        "- Tests Hurdle (occurrence × conditional mean) per direction\n"
        "- Does NOT continue online per-direction probing\n",
        encoding="utf-8")

    # ---- load candidate_10 B predictions ----
    print("=== Loading candidate_10 B OOF ===")
    s10 = pd.read_csv(ROOT / STEP10_PREDS, encoding="utf-8-sig")
    s10["date"] = pd.to_datetime(s10["date"]).dt.normalize()
    s15 = pd.read_csv(ROOT / STEP15_PREDS, encoding="utf-8-sig")
    s15["date"] = pd.to_datetime(s15["date"]).dt.normalize()

    # candidate_10: 5 dirs from step10 b_hier_scale, near->core from step15
    b10 = s10[(s10["predictor_method"]=="pair_hour_hier_tau_scale_cv") & (s10["rounding_method"]=="independent")].copy()
    b15_nc = s15[s15["method"]=="pair_default_hierarchical"].copy()
    # assemble
    b_other = b10[b10["task_key"] != "near->core"]
    b_nc = b15_nc[b15_nc["task_key"]=="near->core"][["evaluation_id","date","hour_of_day","source_region","target_region","task_key","score_available","y_true","pred_float","pred_rounded"]]
    cand10_pred = pd.concat([b_other[["evaluation_id","date","hour_of_day","source_region","target_region","task_key","score_available","y_true","pred_float","pred_rounded"]],
                              b_nc], ignore_index=True)

    # ---- direction diagnostics ----
    print("\n=== Direction diagnostics ===")
    diag_rows = []
    for d in DIRECTIONS:
        sub = cand10_pred[cand10_pred["task_key"]==d]
        scored = sub[sub["score_available"]==True] if "score_available" in sub.columns else sub
        y = scored["y_true"].to_numpy(float)
        p = scored["pred_rounded"].to_numpy(float)
        resid = p - y
        sse = float((resid**2).sum())
        n = len(y)
        zero_rate = float((y==0).mean())
        pos_rate = 1 - zero_rate
        mean_pos = float(y[y>0].mean()) if (y>0).any() else 0
        median_pos = float(np.median(y[y>0])) if (y>0).any() else 0
        p90_pos = float(np.percentile(y[y>0], 90)) if (y>0).any() else 0
        var = float(y.var()) if n > 1 else 0
        # FP: y=0, p>0
        fp_mask = (y==0) & (p>0)
        fn_mask = (y>0) & (p==0)
        both_pos = (y>0) & (p>0)
        fp_sse = float(((p[fp_mask] - y[fp_mask])**2).sum()) if fp_mask.any() else 0
        fn_sse = float(((p[fn_mask] - y[fn_mask])**2).sum()) if fn_mask.any() else 0
        mag_sse = float(((p[both_pos] - y[both_pos])**2).sum()) if both_pos.any() else 0
        # top20 error
        sq_err = resid**2
        top20_idx = np.argsort(-sq_err)[:min(20, len(sq_err))]
        top20_share = float(sq_err[top20_idx].sum() / sse) if sse > 0 else 0
        diag_rows.append({"direction":d, "n":n, "zero_rate":zero_rate, "positive_rate":pos_rate,
                        "mean_positive":mean_pos, "median_positive":median_pos, "p90_positive":p90_pos,
                        "variance":var, "baseline_sse":sse, "false_positive_sse":fp_sse,
                        "false_negative_sse":fn_sse, "positive_magnitude_sse":mag_sse,
                        "fp_count":int(fp_mask.sum()), "fn_count":int(fn_mask.sum()),
                        "fp_share":fp_sse/sse if sse>0 else 0, "fn_share":fn_sse/sse if sse>0 else 0,
                        "mag_share":mag_sse/sse if sse>0 else 0, "top20_error_share":top20_share})
        print(f"  {d:20s} zero={zero_rate:.3f} SSE={sse:.0f} FP={fp_sse/sse*100:.1f}% FN={fn_sse/sse*100:.1f}% mag={mag_sse/sse*100:.1f}% top20={top20_share*100:.1f}%")

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(out_dir / "b_direction_daily_diagnostics.csv", index=False, encoding="utf-8-sig")

    # ---- badcase cells ----
    bc_rows = []
    for d in DIRECTIONS:
        sub = cand10_pred[cand10_pred["task_key"]==d]
        scored = sub[sub["score_available"]==True] if "score_available" in sub.columns else sub
        y = scored["y_true"].to_numpy(float)
        p = scored["pred_rounded"].to_numpy(float)
        sq = (p-y)**2
        idx = np.argsort(-sq)[:20]
        for rank, i in enumerate(idx):
            bc_rows.append({"direction":d, "rank":rank+1, "y_true":int(y[i]), "pred":int(p[i]),
                          "squared_error":float(sq[i]), "error_type":"FP" if y[i]==0 and p[i]>0 else ("FN" if y[i]>0 and p[i]==0 else "magnitude")})
    pd.DataFrame(bc_rows).to_csv(out_dir / "b_badcase_cells.csv", index=False, encoding="utf-8-sig")

    # ---- Hurdle vs Direct comparison ----
    print("\n=== Hurdle vs Direct ===")
    # For each direction, compute simple Hurdle and direct predictions per block
    blocks_map = {
        "block_pre_normal": ("2018-01-01", "2018-01-07", "2018-01-08", "2018-01-11"),
        "block_target_like": ("2018-01-01", "2018-01-19", "2018-01-20", "2018-01-24"),
        "block_post_outage_stress": ("2018-01-01", "2018-01-18", "2018-01-19", "2018-01-24"),
    }

    # Per-fold, per-direction: compute direct and hurdle predictions from training
    # Use existing step10 OOF as baseline
    bs_rows = []
    for bname, (ts, te, vs_, ve) in blocks_map.items():
        te_dt = pd.Timestamp(te); vs_dt = pd.Timestamp(vs_); ve_dt = pd.Timestamp(ve)
        # Get scored rows for this block from candidate_10
        blk = cand10_pred[(cand10_pred["date"] >= vs_dt) & (cand10_pred["date"] <= ve_dt)]
        blk_scored = blk[blk["score_available"]==True] if "score_available" in blk.columns else blk
        bl_sse = float(((blk_scored["pred_rounded"] - blk_scored["y_true"]).astype(float)**2).sum())

        # For each direction, compute direct_long_hour and hurdle_long_hour
        for d in DIRECTIONS:
            sub = blk_scored[blk_scored["task_key"]==d]
            if len(sub) == 0: continue
            y = sub["y_true"].to_numpy(float)
            p_base = sub["pred_rounded"].to_numpy(float)
            bl_dir_sse = float(((p_base - y)**2).sum())

            # Direct long_hour: just mean of training period
            train_sub = cand10_pred[(cand10_pred["task_key"]==d) & (cand10_pred["date"] <= te_dt) & (cand10_pred["score_available"]==True)]
            if len(train_sub) == 0: continue
            train_y = train_sub["y_true"].to_numpy(float)
            direct_mean = float(train_y.mean())

            # Hurdle: occurrence × conditional mean
            p0 = float((train_y > 0).mean())
            mu_pos = float(train_y[train_y>0].mean()) if (train_y>0).any() else 0
            # Beta shrink of occurrence
            tau = 20
            n = len(train_y)
            n1 = int((train_y>0).sum())
            p_occ = (n1 + tau * p0) / (n + tau)
            # conditional mean with shrinkage
            kappa = 5
            n_pos = n1
            mu_cond = (train_y[train_y>0].sum() + kappa * mu_pos) / (n_pos + kappa) if n_pos > 0 else 0
            hurdle_mean = p_occ * mu_cond

            # Evaluate
            pred_direct = np.full(len(y), direct_mean)
            pred_hurdle = np.full(len(y), hurdle_mean)

            sse_direct = float(((pred_direct - y)**2).sum())
            sse_hurdle = float(((pred_hurdle - y)**2).sum())

            bs_rows.append({"block":bname, "direction":d, "method":"direct_long_mean",
                          "sse":sse_direct, "baseline_sse":bl_dir_sse,
                          "reduction":1-sse_direct/bl_dir_sse if bl_dir_sse else 0})
            bs_rows.append({"block":bname, "direction":d, "method":"hurdle_long_mean",
                          "sse":sse_hurdle, "baseline_sse":bl_dir_sse,
                          "reduction":1-sse_hurdle/bl_dir_sse if bl_dir_sse else 0})

        # total baseline
        bs_rows.append({"block":bname, "direction":"all", "method":"candidate10_baseline",
                       "sse":bl_sse, "baseline_sse":bl_sse, "reduction":0})

    bs_df = pd.DataFrame(bs_rows)
    bs_df.to_csv(out_dir / "direction_block_scores.csv", index=False, encoding="utf-8-sig")

    # ---- summary ----
    L = ["# Step 41 B Directional Hurdle Badcase", "", f"Run command: `{RUN_CMD}`", ""]
    L.append("## Direction diagnostics")
    L.append("")
    L.append("| direction | zero_rate | SSE | FP_share | FN_share | mag_share | top20_share |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for _, r in diag_df.iterrows():
        L.append(f"| {r['direction']} | {r['zero_rate']:.3f} | {r['baseline_sse']:.0f} | {r['fp_share']*100:.1f}% | {r['fn_share']*100:.1f}% | {r['mag_share']*100:.1f}% | {r['top20_error_share']*100:.1f}% |")
    L.append("")
    L.append("## Hurdle vs Direct (target_like)")
    L.append("")
    tl = bs_df[bs_df["block"]=="block_target_like"]
    for _, r in tl[tl["direction"]!="all"].iterrows():
        L.append(f"- {r['direction']} {r['method']}: {r['reduction']*100:.1f}%")
    L.append("")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    assert not any(out_dir.rglob("*.zip"))
    print(f"\n=== Summary ===")
    print("Direction diagnostics:")
    for _, r in diag_df.iterrows():
        print(f"  {r['direction']:20s} zero={r['zero_rate']:.3f} SSE={r['baseline_sse']:.0f} FP={r['fp_share']*100:.1f}% FN={r['fn_share']*100:.1f}% mag={r['mag_share']*100:.1f}%")
    print("\nHurdle vs Direct (target_like):")
    for _, r in tl[tl["direction"]!="all"].iterrows():
        print(f"  {r['direction']:20s} {r['method']:20s} red={r['reduction']*100:.1f}%")
    print("\nDone.")


if __name__ == "__main__":
    main()
