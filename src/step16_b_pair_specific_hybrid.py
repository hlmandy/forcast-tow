"""Step 16 — B pair-specific hybrid (core_near from pair-hierarchical, rest circular3).

No new training, no value modification. Selects per task_key from the Step 15
predictions: the two core_near directions (core->near, near->core) take a pair
hierarchical variant; the other four directions stay as circular3_independent.

Outputs (under outputs/step16_b_pair_specific_hybrid/):
  1. method_definitions.csv (4)
  2. b_model_predictions.csv (35712)
  3. b_fold_scores.csv (36)
  4. b_pair_scores.csv (108)
  5. b_direction_scores.csv (216)
  6. b_horizon_scores.csv (248)
  7. b_nonoverlap_scores.csv (12)
  8. summary.md
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
STEP15 = Path("outputs/step15_b_pair_decomposition_benchmark/b_model_predictions.csv")
OUT_REL = Path("outputs/step16_b_pair_specific_hybrid")

CORE_NEAR_DIRS = ["core->near", "near->core"]
OTHER_DIRS = ["near->outer", "outer->near", "core->outer", "outer->core"]
ALL_DIRS = CORE_NEAR_DIRS + OTHER_DIRS
DIR_RANK = {d: i for i, d in enumerate(ALL_DIRS)}
PAIRS = ["core_near", "near_outer", "core_outer"]
PAIR_ORDER = {p: i for i, p in enumerate(PAIRS)}
PAIR_OF = {"core->near": "core_near", "near->core": "core_near", "near->outer": "near_outer",
           "outer->near": "near_outer", "core->outer": "core_outer", "outer->core": "core_outer"}
ROLE = {"core->near": "forward", "near->core": "reverse", "near->outer": "forward",
        "outer->near": "reverse", "core->outer": "forward", "outer->core": "reverse"}

# method: (core_near_source, label)
METHODS = [
    ("circular3_all", "circular3_independent", None),
    ("hybrid_core_near_default", "circular3_independent", "pair_default_hierarchical"),
    ("hybrid_core_near_globalcv", "circular3_independent", "pair_global_cv_hierarchical"),
    ("hybrid_core_near_bypaircv", "circular3_independent", "pair_bypair_cv_hierarchical"),
]
METHOD_ORDER = {m[0]: i for i, m in enumerate(METHODS)}

PRED_COLS = ["evaluation_id", "evaluation_type", "method", "source_method", "date", "hour", "hour_of_day",
             "forecast_horizon", "source_region", "target_region", "task_key", "pair", "direction_role",
             "score_available", "y_true", "pred_float", "pred_rounded", "error_float", "error_rounded"]
FOLD_COLS = ["evaluation_id", "evaluation_type", "method", "n_prediction_rows", "n_scored_rows", "sse_float",
             "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded", "actual_total", "predicted_total", "rolling_rank"]
PAIR_COLS = ["evaluation_id", "evaluation_type", "method", "pair", "n_scored_rows", "actual_total", "predicted_total",
             "sse_float", "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded", "zero_true_rate", "zero_prediction_rate"]
DIR_COLS = ["evaluation_id", "evaluation_type", "method", "task_key", "pair", "direction_role", "n_scored_rows",
            "actual_total", "predicted_total", "sse_float", "sse_rounded", "mse_rounded", "mae_rounded",
            "mean_bias_rounded", "zero_true_rate", "zero_prediction_rate"]
HORIZON_COLS = ["evaluation_id", "evaluation_type", "method", "forecast_horizon", "n_scored_rows", "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded"]
NONOVERLAP_COLS = ["method", "block", "target_start", "target_end", "n_scored_rows", "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded"]
DEF_COLS = ["method", "core_near_source", "near_outer_source", "core_outer_source", "uses_model_average", "uses_new_training", "deployable"]

RUN_CMD = "python " + " ".join(sys.argv)


def main():
    out_dir = ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    p15 = pd.read_csv(ROOT / STEP15, encoding="utf-8-sig")
    p15["date"] = pd.to_datetime(p15["date"]).dt.normalize()
    p15["hour"] = pd.to_datetime(p15["hour"])
    scenarios = sorted(p15["evaluation_id"].unique(),
                       key=lambda e: ([f"fold_{i:02d}" for i in range(1, 9)] + ["final_analog"]).index(e))

    circ = p15[p15["method"] == "circular3_independent"].copy()
    pair_src = {m: p15[p15["method"] == m].copy() for _, _, m in METHODS if m}

    pred_frames = []
    fold_rows = []
    pair_rows = []
    dir_rows = []
    horizon_rows = []

    for mname, circ_label, cn_src in METHODS:
        if cn_src is None:
            sel = circ.copy()
        else:
            cn = pair_src[cn_src][pair_src[cn_src]["task_key"].isin(CORE_NEAR_DIRS)].copy()
            ot = circ[circ["task_key"].isin(OTHER_DIRS)].copy()
            sel = pd.concat([cn, ot], ignore_index=True)
        sel = sel.copy()
        sel["method"] = mname
        sel["source_method"] = np.where(sel["task_key"].isin(CORE_NEAR_DIRS),
                                        cn_src if cn_src else "circular3_independent",
                                        "circular3_independent")
        # recompute errors from pred - y_true (scored rows)
        sel["error_float"] = np.where(sel["score_available"], sel["pred_float"] - sel["y_true"], np.nan)
        sel["error_rounded"] = np.where(sel["score_available"], sel["pred_rounded"] - sel["y_true"], np.nan)
        pred_frames.append(sel[PRED_COLS])

        for eid in scenarios:
            sc = sel[(sel["evaluation_id"] == eid) & sel["score_available"]]
            er = sc["error_rounded"].astype(float).to_numpy()
            ef = sc["error_float"].astype(float).to_numpy()
            etype = sc["evaluation_type"].iloc[0]
            fold_rows.append({"evaluation_id": eid, "evaluation_type": etype, "method": mname,
                              "n_prediction_rows": int(len(sel[sel["evaluation_id"] == eid])), "n_scored_rows": int(len(sc)),
                              "sse_float": float((ef ** 2).sum()), "sse_rounded": float((er ** 2).sum()),
                              "mse_rounded": float((er ** 2).mean()), "mae_rounded": float(np.abs(er).mean()),
                              "mean_bias_rounded": float(er.mean()), "actual_total": float(sc["y_true"].sum()),
                              "predicted_total": float(sc["pred_rounded"].sum()), "rolling_rank": np.nan})
            for pair in PAIRS:
                sp = sc[sc["pair"] == pair]
                pair_rows.append(_pair_row(eid, etype, mname, pair, sp))
            for d in ALL_DIRS:
                sd = sc[sc["task_key"] == d]
                dir_rows.append(_dir_row(eid, etype, mname, d, sd))
            for hz in sorted(sc["forecast_horizon"].unique()):
                sh = sc[sc["forecast_horizon"] == hz]
                erH = sh["error_rounded"].astype(float).to_numpy()
                horizon_rows.append({"evaluation_id": eid, "evaluation_type": etype, "method": mname, "forecast_horizon": int(hz),
                                     "n_scored_rows": int(len(sh)), "sse_rounded": float((erH ** 2).sum()),
                                     "mse_rounded": float((erH ** 2).mean()) if len(erH) else np.nan,
                                     "mae_rounded": float(np.abs(erH).mean()) if len(erH) else np.nan,
                                     "mean_bias_rounded": float(erH.mean()) if len(erH) else np.nan})

    preds = pd.concat(pred_frames, ignore_index=True)
    preds["_e"] = preds["evaluation_id"].map({s: i for i, s in enumerate(scenarios)})
    preds["_m"] = preds["method"].map(METHOD_ORDER)
    preds = preds.sort_values(["_e", "_m", "hour", "source_region", "target_region"]).reset_index(drop=True)
    assert len(preds) == 35712, f"preds {len(preds)} != 35712"

    fold_scores = pd.DataFrame(fold_rows)
    rolling = fold_scores[fold_scores["evaluation_type"] == "rolling_7d"].copy()
    rolling["rolling_rank"] = rolling.groupby("evaluation_id")["sse_rounded"].rank(method="min").astype(int)
    fold_scores = fold_scores.drop(columns=["rolling_rank"]).merge(
        rolling[["evaluation_id", "method", "rolling_rank"]], on=["evaluation_id", "method"], how="left")
    fold_scores["rolling_rank"] = fold_scores["rolling_rank"].where(fold_scores["evaluation_type"] == "rolling_7d")

    pairscores = pd.DataFrame(pair_rows)
    dirscores = pd.DataFrame(dir_rows)
    horizons = pd.DataFrame(horizon_rows)

    # non-overlap
    no_rows = []
    for mname, *_ in METHODS:
        for block, eids in [("nonoverlap_block_1", ["fold_01"]), ("nonoverlap_block_2", ["fold_08"]), ("nonoverlap_combined", ["fold_01", "fold_08"])]:
            sub = preds[(preds["method"] == mname) & (preds["evaluation_id"].isin(eids)) & preds["score_available"]]
            er = sub["error_rounded"].astype(float).to_numpy()
            dts = pd.to_datetime(sub["date"].unique())
            no_rows.append({"method": mname, "block": block, "target_start": f"{dts.min():%Y-%m-%d}" if len(dts) else "",
                            "target_end": f"{dts.max():%Y-%m-%d}" if len(dts) else "", "n_scored_rows": int(len(sub)),
                            "sse_rounded": float((er ** 2).sum()), "mse_rounded": float((er ** 2).mean()) if len(er) else np.nan,
                            "mae_rounded": float(np.abs(er).mean()) if len(er) else np.nan, "mean_bias_rounded": float(er.mean()) if len(er) else np.nan})
    nonoverlap = pd.DataFrame(no_rows)

    # --- reproduction assertions ---
    # circular3_all == step15 circular3
    c_all = preds[preds["method"] == "circular3_all"]
    c15 = p15[p15["method"] == "circular3_independent"]
    merged = c_all[["evaluation_id", "date", "hour", "task_key", "pred_float", "pred_rounded"]].merge(
        c15[["evaluation_id", "date", "hour", "task_key", "pred_float", "pred_rounded"]].rename(columns={"pred_float": "f15", "pred_rounded": "r15"}),
        on=["evaluation_id", "date", "hour", "task_key"])
    md = float((merged["pred_float"] - merged["f15"]).abs().max())
    assert md < 1e-10 and (merged["pred_rounded"] == merged["r15"]).all(), f"circular3_all repro {md}"
    # other 4 directions == circular3 in hybrids
    for mname, _, cn_src in METHODS:
        if cn_src is None:
            continue
        h = preds[preds["method"] == mname]
        ho = h[h["task_key"].isin(OTHER_DIRS)]
        m2 = ho.merge(c15[["evaluation_id", "date", "hour", "task_key", "pred_float", "pred_rounded"]].rename(columns={"pred_float": "f15", "pred_rounded": "r15"}),
                      on=["evaluation_id", "date", "hour", "task_key"])
        assert float((m2["pred_float"] - m2["f15"]).abs().max()) < 1e-10
        assert (m2["pred_rounded"] == m2["r15"]).all()
    print("reproduction assertions PASS")

    # --- write ---
    p_def = out_dir / "method_definitions.csv"
    p_pred = out_dir / "b_model_predictions.csv"
    p_fs = out_dir / "b_fold_scores.csv"
    p_ps = out_dir / "b_pair_scores.csv"
    p_ds = out_dir / "b_direction_scores.csv"
    p_hz = out_dir / "b_horizon_scores.csv"
    p_no = out_dir / "b_nonoverlap_scores.csv"
    p_md = out_dir / "summary.md"

    md_rows = []
    for mname, _, cn_src in METHODS:
        md_rows.append({"method": mname, "core_near_source": cn_src if cn_src else "circular3_independent",
                        "near_outer_source": "circular3_independent", "core_outer_source": "circular3_independent",
                        "uses_model_average": False, "uses_new_training": False, "deployable": True})
    pd.DataFrame(md_rows)[DEF_COLS].to_csv(p_def, index=False, encoding="utf-8-sig")

    pred_out = preds[PRED_COLS].copy()
    pred_out["hour"] = pd.to_datetime(pred_out["hour"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    pred_out["date"] = pd.to_datetime(pred_out["date"]).dt.strftime("%Y-%m-%d")
    pred_out.to_csv(p_pred, index=False, encoding="utf-8-sig")

    fold_scores.sort_values(["evaluation_id", "method"])[FOLD_COLS].to_csv(p_fs, index=False, encoding="utf-8-sig")
    ps_out = pairscores.copy(); ps_out["_e"] = ps_out["evaluation_id"].map({s: i for i, s in enumerate(scenarios)})
    ps_out["_m"] = ps_out["method"].map(METHOD_ORDER); ps_out["_p"] = ps_out["pair"].map(PAIR_ORDER)
    ps_out.sort_values(["_e", "_m", "_p"])[PAIR_COLS].to_csv(p_ps, index=False, encoding="utf-8-sig")
    ds_out = dirscores.copy(); ds_out["_e"] = ds_out["evaluation_id"].map({s: i for i, s in enumerate(scenarios)})
    ds_out["_m"] = ds_out["method"].map(METHOD_ORDER); ds_out["_d"] = ds_out["task_key"].map(DIR_RANK)
    ds_out.sort_values(["_e", "_m", "_d"])[DIR_COLS].to_csv(p_ds, index=False, encoding="utf-8-sig")
    hz_out = horizons.copy(); hz_out["_e"] = hz_out["evaluation_id"].map({s: i for i, s in enumerate(scenarios)})
    hz_out["_m"] = hz_out["method"].map(METHOD_ORDER)
    hz_out.sort_values(["_e", "_m", "forecast_horizon"])[HORIZON_COLS].to_csv(p_hz, index=False, encoding="utf-8-sig")
    nonoverlap[NONOVERLAP_COLS].to_csv(p_no, index=False, encoding="utf-8-sig")

    write_summary(p_md, fold_scores, nonoverlap, dirscores)

    print("\n=== Output files ===")
    for path in [p_def, p_pred, p_fs, p_ps, p_ds, p_hz, p_no, p_md]:
        rows = len(pd.read_csv(path, encoding="utf-8-sig")) if path.suffix == ".csv" else sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {path.relative_to(ROOT)}  rows={rows}")
    print("\n=== rolling MSE by method ===")
    rmed = fold_scores[fold_scores.evaluation_type == "rolling_7d"]
    print(rmed.groupby("method").agg(mean_mse=("mse_rounded", "mean"), median_mse=("mse_rounded", "median"), max_mse=("mse_rounded", "max")).to_string())
    print("\nDone.")


def _pair_row(eid, etype, mname, pair, sp):
    er = sp["error_rounded"].astype(float).to_numpy()
    ef = sp["error_float"].astype(float).to_numpy()
    return {"evaluation_id": eid, "evaluation_type": etype, "method": mname, "pair": pair, "n_scored_rows": int(len(sp)),
            "actual_total": float(sp["y_true"].sum()), "predicted_total": float(sp["pred_rounded"].sum()),
            "sse_float": float((ef ** 2).sum()), "sse_rounded": float((er ** 2).sum()),
            "mse_rounded": float((er ** 2).mean()) if len(er) else np.nan, "mae_rounded": float(np.abs(er).mean()) if len(er) else np.nan,
            "mean_bias_rounded": float(er.mean()) if len(er) else np.nan,
            "zero_true_rate": float((sp["y_true"] == 0).mean()) if len(sp) else np.nan,
            "zero_prediction_rate": float((sp["pred_rounded"] == 0).mean()) if len(sp) else np.nan}


def _dir_row(eid, etype, mname, d, sd):
    er = sd["error_rounded"].astype(float).to_numpy()
    ef = sd["error_float"].astype(float).to_numpy()
    return {"evaluation_id": eid, "evaluation_type": etype, "method": mname, "task_key": d, "pair": PAIR_OF[d],
            "direction_role": ROLE[d], "n_scored_rows": int(len(sd)), "actual_total": float(sd["y_true"].sum()),
            "predicted_total": float(sd["pred_rounded"].sum()), "sse_float": float((ef ** 2).sum()),
            "sse_rounded": float((er ** 2).sum()), "mse_rounded": float((er ** 2).mean()) if len(er) else np.nan,
            "mae_rounded": float(np.abs(er).mean()) if len(er) else np.nan, "mean_bias_rounded": float(er.mean()) if len(er) else np.nan,
            "zero_true_rate": float((sd["y_true"] == 0).mean()) if len(sd) else np.nan,
            "zero_prediction_rate": float((sd["pred_rounded"] == 0).mean()) if len(sd) else np.nan}


def write_summary(path, fold_scores, nonoverlap, dirscores):
    rolling = fold_scores[fold_scores["evaluation_type"] == "rolling_7d"]
    fa = fold_scores[fold_scores["evaluation_id"] == "final_analog"].set_index("method")
    no = nonoverlap.set_index(["method", "block"])
    L = []
    L.append("# Step 16 B Pair-Specific Hybrid")
    L.append("")
    L.append(f"Run command: `{RUN_CMD}`")
    L.append("")
    L.append("## 1. Motivation\n- Overall pair decomposition did not stably beat circular3, but core_near had a local stable gain. near_outer and core_outer must stay circular3. This step only selects existing per-task predictions; no new models, no averaging.\n")
    L.append("## 2. Reproduction checks\n- circular3_all == step15 circular3_independent (<1e-10). Hybrid non-core_near 4 directions == circular3 (<1e-10). core_near 2 directions == the specified pair method. **PASS**\n")
    L.append("## 3. Rolling comparison\n")
    L.append("| method | mean MSE | median MSE | max MSE |")
    L.append("| --- | --- | --- | --- |")
    g = rolling.groupby("method")["mse_rounded"].agg(["mean", "median", "max"])
    for m in [x[0] for x in METHODS]:
        L.append(f"| {m} | {g.loc[m,'mean']:.4f} | {g.loc[m,'median']:.4f} | {g.loc[m,'max']:.4f} |")
    L.append("")
    # folds beating circular3
    piv = rolling.pivot_table(index="evaluation_id", columns="method", values="sse_rounded", aggfunc="first")
    c3 = piv["circular3_all"]
    L.append("## 4. Final analog and non-overlap\n")
    L.append("| method | final SSE | nonoverlap1 MSE | nonoverlap2 MSE | combined MSE |")
    L.append("| --- | --- | --- | --- | --- |")
    for m in [x[0] for x in METHODS]:
        L.append(f"| {m} | {fa.loc[m,'sse_rounded']:.0f} | {no.loc[(m,'nonoverlap_block_1'),'mse_rounded']:.4f} | {no.loc[(m,'nonoverlap_block_2'),'mse_rounded']:.4f} | {no.loc[(m,'nonoverlap_combined'),'mse_rounded']:.4f} |")
    L.append("")
    L.append("## 5. Direction-level effect\n")
    L.append("| method | core->near | near->core | near->outer | outer->near | core->outer | outer->core |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    dr = dirscores[dirscores["evaluation_type"] == "rolling_7d"].groupby(["method", "task_key"])["sse_rounded"].sum().unstack()
    for m in [x[0] for x in METHODS]:
        L.append("| " + m + " | " + " | ".join(f"{dr.loc[m,d]:.0f}" for d in ALL_DIRS) + " |")
    L.append("")
    L.append("## 6. Forecast horizon\n- See b_horizon_scores.csv.\n")
    L.append("## 7. Decision\n- Pass criteria: >=5 rolling folds beat circular3; final < circular3; nonoverlap combined < circular3; rolling max not worse; core->near and near->core both improve; other 4 directions identical to circular3.\n")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
