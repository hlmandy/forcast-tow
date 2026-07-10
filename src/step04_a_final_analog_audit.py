"""Step 04 — Final-scenario-analog backtest and normal-target robustness for Task A.

Step 03's eight overlapping 7-day windows all cover the 2018-01-13..01-18 data
quality anomaly, so its overall median SSE cannot by itself choose the final
training policy. This step adds a single final-analog fold (train
2018-01-01..01-18, valid 2018-01-19..01-24, all-normal validation) and re-uses
the Step 03 predictions to score the same 24 (policy, method) combos on:

  - the final-analog fold (normal future targets, anomaly inside training)
  - fold_08 restricted to its normal validation dates (2018-01-19..01-24)
  - all Step 03 folds restricted to normal-quality validation days

Reuses Step 03 building blocks (A labels, METHODS, POLICIES, POLICY_WEIGHTS,
classify_fold_quality, predict_weighted, scale_train_mean) by import; Step 03
is not modified and its main() does not run on import. No new models, no B task,
no submission files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from step03_a_rolling_backtest import (  # noqa: E402
    METHODS,
    POLICIES,
    POLICY_ORDER,
    METHOD_ORDER,
    POLICY_WEIGHTS,
    REGION_RANK,
    TRAIN_REL,
    STEP01_DAILY,
    STEP02_DAILY,
    build_a_frame,
    classify_fold_quality,
    load_ais,
    metrics,
    predict_weighted,
    scale_train_mean,
)
from optimized_baseline import MethodSpec, predict_base, REGION_ORDER  # noqa: E402

STEP03_PREDS = Path("outputs/step03_a_rolling_backtest/a_backtest_predictions.csv")
STEP03_MODEL_SUMMARY = Path("outputs/step03_a_rolling_backtest/model_summary.csv")
OUT_REL = Path("outputs/step04_a_final_analog_audit")

# Final-analog fold: training fully contains the outage+recovery; validation is
# the all-normal tail of the available data.
TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2018-01-18")
VALID_START = pd.Timestamp("2018-01-19")
VALID_END = pd.Timestamp("2018-01-24")
EXPECTED_VALID_DATES = set(pd.date_range(VALID_START, VALID_END, freq="D").normalize())

PRED_COLUMNS = [
    "training_policy", "method", "hour", "date", "forecast_horizon", "hour_of_day", "dayofweek",
    "region", "validation_unique_vessel_count", "y_true",
    "pred_float", "pred_rounded", "error_float", "error_rounded",
]
FINAL_SCORE_COLUMNS = [
    "training_policy", "method", "train_day_count_used", "excluded_train_day_count",
    "effective_train_day_weight", "sse_float", "sse_rounded", "mae_rounded",
    "mean_bias_rounded", "actual_total", "predicted_total_rounded",
    "core_sse_rounded", "near_sse_rounded", "outer_sse_rounded", "rank_by_sse_rounded",
]
FOLD08_COLUMNS = [
    "training_policy", "method", "n_dates", "n_rows", "sse_rounded", "mse_rounded",
    "mae_rounded", "mean_bias_rounded", "core_sse_rounded", "near_sse_rounded",
    "outer_sse_rounded", "rank_by_sse_rounded",
]
ALL_NORMAL_COLUMNS = [
    "training_policy", "method", "n_prediction_rows", "n_unique_target_dates",
    "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded", "rank_by_mse_rounded",
]
DECISION_COLUMNS = [
    "training_policy", "method",
    "step03_median_sse", "step03_last_fold_sse",
    "fold08_normal_sse", "fold08_normal_rank",
    "final_analog_sse", "final_analog_rank",
    "all_folds_normal_mse", "all_folds_normal_rank",
    "final_analog_mean_bias", "final_analog_core_sse", "final_analog_near_sse", "final_analog_outer_sse",
]

CONSISTENCY_TOL = 1e-10
RUN_CMD = "python " + " ".join(sys.argv)


def region_sse(sub: pd.DataFrame, region: str) -> float:
    er = sub.loc[sub["region"] == region, "error_rounded"].to_numpy(dtype="float64")
    return float((er ** 2).sum()) if len(er) else 0.0


def main() -> None:
    out_dir = ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = ROOT / TRAIN_REL
    if not train_path.exists():
        raise FileNotFoundError(f"Training AIS file not found: {train_path}")

    print(f"train_path = {train_path}")
    print(f"out_dir    = {out_dir}")

    df = load_ais(train_path)
    daily = pd.read_csv(ROOT / STEP01_DAILY)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    audit = pd.read_csv(ROOT / STEP02_DAILY)
    audit["date"] = pd.to_datetime(audit["date"]).dt.normalize()
    china_series = dict(zip(audit["date"], audit["china_coastal_record_count"]))

    a_frame = build_a_frame(df, daily, audit)

    # --- final-analog split --------------------------------------------------
    train_full = a_frame[(a_frame["date"] >= TRAIN_START) & (a_frame["date"] <= TRAIN_END)].copy()
    valid = a_frame[(a_frame["date"] >= VALID_START) & (a_frame["date"] <= VALID_END)].copy()
    valid["forecast_horizon"] = (valid["date"] - TRAIN_END).dt.days
    valid = valid.sort_values(["hour", "region_rank"]).reset_index(drop=True)

    # assertions
    assert len(valid) == 432, f"final_analog validation must have 432 rows, got {len(valid)}"
    assert set(valid["date"].dt.normalize().unique()) == EXPECTED_VALID_DATES, "validation dates not 01-19..01-24"
    assert valid["vessel_count"].notna().all(), "missing validation unique_vessel_count"
    assert (valid["audit_quality_regime"] == "normal").all(), "validation dates must all be normal"

    train_dates = sorted(pd.date_range(TRAIN_START, TRAIN_END, freq="D").normalize())
    classes, _ = classify_fold_quality(train_dates, china_series)

    pred_frames: list[pd.DataFrame] = []
    score_rows: list[dict] = []

    for policy in POLICIES:
        pw = POLICY_WEIGHTS[policy]
        weights = {d: pw[classes[d]] for d in train_dates}
        train_w = train_full.copy()
        train_w["weight"] = train_w["date"].map(weights).astype("float64")
        smean = scale_train_mean(train_dates, dict(zip(daily["date"], daily["unique_vessel_count"])), weights, policy)
        used = int(sum(1 for d in train_dates if weights[d] > 0))
        excluded = int(sum(1 for d in train_dates if weights[d] == 0))
        eff_weight = float(sum(weights.values()))

        for mname, spec in METHODS:
            pred = np.asarray(predict_weighted(train_w, valid, spec, smean), dtype="float64")

            if policy == "all":
                ref = np.asarray(predict_base(train_w, valid, "A", spec), dtype="float64")
                diff = float(np.max(np.abs(pred - ref)))
                if diff >= CONSISTENCY_TOL:
                    raise AssertionError(f"consistency fail final_analog all {mname}: {diff:.3e}")

            assert pred.min() >= 0.0, f"negative prediction for {policy}/{mname}"
            pred_rounded = np.clip(np.rint(pred), 0, None).astype(np.int64)
            y_true = valid["y"].to_numpy(dtype="float64")

            combo = pd.DataFrame({
                "training_policy": policy,
                "method": mname,
                "hour": valid["hour"].to_numpy(),
                "date": valid["date"].to_numpy(),
                "forecast_horizon": valid["forecast_horizon"].to_numpy(),
                "hour_of_day": valid["hour_of_day"].to_numpy(),
                "dayofweek": valid["dayofweek"].to_numpy(),
                "region": valid["region"].to_numpy(),
                "validation_unique_vessel_count": valid["vessel_count"].to_numpy(),
                "y_true": y_true,
                "pred_float": pred,
                "pred_rounded": pred_rounded,
                "region_rank": valid["region_rank"].to_numpy(),
            })
            combo["error_float"] = combo["pred_float"] - combo["y_true"]
            combo["error_rounded"] = combo["pred_rounded"] - combo["y_true"]
            assert len(combo) == 432
            pred_frames.append(combo)

            m = metrics(y_true, pred, pred_rounded)
            score_rows.append({
                "training_policy": policy, "method": mname,
                "train_day_count_used": used,
                "excluded_train_day_count": excluded,
                "effective_train_day_weight": eff_weight,
                "sse_float": m["sse_float"], "sse_rounded": m["sse_rounded"],
                "mae_rounded": m["mae_rounded"], "mean_bias_rounded": m["mean_bias_rounded"],
                "actual_total": m["actual_total"], "predicted_total_rounded": m["predicted_total_rounded"],
                "core_sse_rounded": region_sse(combo, "core"),
                "near_sse_rounded": region_sse(combo, "near"),
                "outer_sse_rounded": region_sse(combo, "outer"),
            })

    preds = pd.concat(pred_frames, ignore_index=True)
    preds["_p"] = preds["training_policy"].map(POLICY_ORDER)
    preds["_m"] = preds["method"].map(METHOD_ORDER)
    preds = preds.sort_values(["_p", "_m", "hour", "region_rank"]).reset_index(drop=True)
    assert len(preds) == 10368, f"final_analog_predictions must have 10368 rows, got {len(preds)}"
    assert (preds["pred_float"] >= 0).all()
    assert preds["pred_rounded"].apply(lambda x: float(x).is_integer()).all()

    final_scores = pd.DataFrame(score_rows)
    final_scores["rank_by_sse_rounded"] = final_scores["sse_rounded"].rank(method="min", ascending=True).astype(int)
    final_scores = final_scores.sort_values(["rank_by_sse_rounded", "training_policy", "method"]).reset_index(drop=True)

    # --- fold_08 normal-date scores (from Step 03 predictions) ---------------
    s3 = pd.read_csv(ROOT / STEP03_PREDS, encoding="utf-8-sig")
    f8 = s3[(s3["fold_id"] == "fold_08") & (s3["date"] >= "2018-01-19") & (s3["date"] <= "2018-01-24")].copy()
    fold08_rows = []
    for policy in POLICIES:
        for mname, _ in METHODS:
            sub = f8[(f8["training_policy"] == policy) & (f8["method"] == mname)]
            er = sub["error_rounded"].astype("float64").to_numpy()
            fold08_rows.append({
                "training_policy": policy, "method": mname,
                "n_dates": int(sub["date"].nunique()),
                "n_rows": int(len(sub)),
                "sse_rounded": float((er ** 2).sum()),
                "mse_rounded": float((er ** 2).mean()),
                "mae_rounded": float(np.abs(er).mean()),
                "mean_bias_rounded": float(er.mean()),
                "core_sse_rounded": float((sub.loc[sub["region"] == "core", "error_rounded"].astype("float64") ** 2).sum()),
                "near_sse_rounded": float((sub.loc[sub["region"] == "near", "error_rounded"].astype("float64") ** 2).sum()),
                "outer_sse_rounded": float((sub.loc[sub["region"] == "outer", "error_rounded"].astype("float64") ** 2).sum()),
            })
    fold08_scores = pd.DataFrame(fold08_rows)
    fold08_scores["rank_by_sse_rounded"] = fold08_scores["sse_rounded"].rank(method="min", ascending=True).astype(int)
    fold08_scores = fold08_scores.sort_values(["rank_by_sse_rounded", "training_policy", "method"]).reset_index(drop=True)

    # --- all-folds normal-date scores (from Step 03 predictions) -------------
    fn = s3[s3["audit_quality_regime"] == "normal"].copy()
    allnorm_rows = []
    for policy in POLICIES:
        for mname, _ in METHODS:
            sub = fn[(fn["training_policy"] == policy) & (fn["method"] == mname)]
            er = sub["error_rounded"].astype("float64").to_numpy()
            allnorm_rows.append({
                "training_policy": policy, "method": mname,
                "n_prediction_rows": int(len(sub)),
                "n_unique_target_dates": int(sub["date"].nunique()),
                "sse_rounded": float((er ** 2).sum()),
                "mse_rounded": float((er ** 2).mean()),
                "mae_rounded": float(np.abs(er).mean()),
                "mean_bias_rounded": float(er.mean()),
            })
    allnorm_scores = pd.DataFrame(allnorm_rows)
    allnorm_scores["rank_by_mse_rounded"] = allnorm_scores["mse_rounded"].rank(method="min", ascending=True).astype(int)
    allnorm_scores = allnorm_scores.sort_values(["rank_by_mse_rounded", "training_policy", "method"]).reset_index(drop=True)

    # --- decision table ------------------------------------------------------
    ms3 = pd.read_csv(ROOT / STEP03_MODEL_SUMMARY, encoding="utf-8-sig")
    dt = ms3[["training_policy", "method", "median_sse_rounded", "last_fold_sse_rounded"]].rename(
        columns={"median_sse_rounded": "step03_median_sse", "last_fold_sse_rounded": "step03_last_fold_sse"}
    )
    dt = dt.merge(
        fold08_scores[["training_policy", "method", "sse_rounded", "rank_by_sse_rounded"]].rename(
            columns={"sse_rounded": "fold08_normal_sse", "rank_by_sse_rounded": "fold08_normal_rank"}
        ), on=["training_policy", "method"], how="left"
    ).merge(
        final_scores[["training_policy", "method", "sse_rounded", "rank_by_sse_rounded", "mean_bias_rounded",
                      "core_sse_rounded", "near_sse_rounded", "outer_sse_rounded"]].rename(
            columns={"sse_rounded": "final_analog_sse", "rank_by_sse_rounded": "final_analog_rank",
                     "mean_bias_rounded": "final_analog_mean_bias", "core_sse_rounded": "final_analog_core_sse",
                     "near_sse_rounded": "final_analog_near_sse", "outer_sse_rounded": "final_analog_outer_sse"}
        ), on=["training_policy", "method"], how="left"
    ).merge(
        allnorm_scores[["training_policy", "method", "mse_rounded", "rank_by_mse_rounded"]].rename(
            columns={"mse_rounded": "all_folds_normal_mse", "rank_by_mse_rounded": "all_folds_normal_rank"}
        ), on=["training_policy", "method"], how="left"
    )
    decision = dt[DECISION_COLUMNS].sort_values(["training_policy", "method"]).reset_index(drop=True)

    # --- write CSVs (UTF-8 with BOM) ----------------------------------------
    p_pred = out_dir / "final_analog_predictions.csv"
    p_fscore = out_dir / "final_analog_scores.csv"
    p_f8 = out_dir / "fold08_normal_scores.csv"
    p_an = out_dir / "all_folds_normal_scores.csv"
    p_dec = out_dir / "decision_table.csv"
    p_sum = out_dir / "summary.md"

    pred_out = preds[PRED_COLUMNS].copy()
    pred_out["hour"] = pd.to_datetime(pred_out["hour"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    pred_out["date"] = pd.to_datetime(pred_out["date"]).dt.strftime("%Y-%m-%d")
    pred_out.to_csv(p_pred, index=False, encoding="utf-8-sig")

    final_scores[FINAL_SCORE_COLUMNS].to_csv(p_fscore, index=False, encoding="utf-8-sig")
    fold08_scores[FOLD08_COLUMNS].to_csv(p_f8, index=False, encoding="utf-8-sig")
    allnorm_scores[ALL_NORMAL_COLUMNS].to_csv(p_an, index=False, encoding="utf-8-sig")
    decision.to_csv(p_dec, index=False, encoding="utf-8-sig")

    write_summary(p_sum, final_scores, fold08_scores, allnorm_scores, decision)

    # --- console report ------------------------------------------------------
    print("\n=== Output files ===")
    for path in [p_pred, p_fscore, p_f8, p_an, p_dec, p_sum]:
        if path.suffix == ".csv":
            rows = len(pd.read_csv(path, encoding="utf-8-sig"))
        else:
            rows = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {path.relative_to(ROOT)}  rows={rows}")

    print("\n=== final_analog top 10 by sse_rounded ===")
    for _, r in final_scores.head(10).iterrows():
        print(f"  rank={int(r['rank_by_sse_rounded']):2d}  {r['training_policy']:22s} {r['method']:20s} "
              f"sse={r['sse_rounded']:.0f} bias={r['mean_bias_rounded']:.2f} "
              f"core={r['core_sse_rounded']:.0f}/near={r['near_sse_rounded']:.0f}/outer={r['outer_sse_rounded']:.0f}")

    print("\n=== fold08_normal top 10 by sse_rounded ===")
    for _, r in fold08_scores.head(10).iterrows():
        print(f"  rank={int(r['rank_by_sse_rounded']):2d}  {r['training_policy']:22s} {r['method']:20s} "
              f"sse={r['sse_rounded']:.0f} mse={r['mse_rounded']:.3f} bias={r['mean_bias_rounded']:.2f}")
    print("\nDone.")


def write_summary(path: Path, final_scores: pd.DataFrame, fold08_scores: pd.DataFrame,
                  allnorm_scores: pd.DataFrame, decision: pd.DataFrame) -> None:
    L: list[str] = []
    L.append("# Step 04 A Final-Analog Audit")
    L.append("")
    L.append("> Task A only. No new models, no B task, no submission files. Reuses Step 03 building blocks.")
    L.append("")
    L.append(f"Run command: `{RUN_CMD}`")
    L.append("")

    # 1
    L.append("## 1. Why the Step 03 overall ranking is insufficient")
    L.append("")
    L.append("The eight 7-day validation windows overlap heavily and every one of them covers the 2018-01-13..01-18 data-quality anomaly. The same anomalous validation dates are therefore counted multiple times, and the early folds are dominated by outage/severe days. Step 03's overall median SSE consequently reflects performance on anomalous targets and cannot by itself decide the final training policy.")
    L.append("")

    # 2
    L.append("## 2. Final-analog design")
    L.append("")
    L.append(f"The final-analog fold trains on 2018-01-01~2018-01-18 (18d, which fully contains the china_coastal outage and the initial recovery) and validates on 2018-01-19~2018-01-24 (6d — the available data end on 01-24, so the window is not padded to 7). All six validation dates are `normal` in the Step 02 audit.")
    L.append("")
    L.append("Leakage controls: prediction uses only the validation dates' `unique_vessel_count`; validation AIS record counts, china_coastal record counts, quality variables, and A labels are not used in prediction. The training quality class is recomputed from only 2018-01-01~2018-01-18. For the `all` policy the weighted prediction reproduces `optimized_baseline.predict_base` with max abs diff < 1e-10 (asserted). All predictions are non-negative and rounded predictions are integers.")
    L.append("")

    # 3
    L.append("## 3. Training-quality policy comparison")
    L.append("")
    L.append("Median `sse_rounded` across the 8 methods, and best method, under each policy on three normal-target views:")
    L.append("")
    fa_best = final_scores.sort_values("sse_rounded").iloc[0]
    f8_best = fold08_scores.sort_values("sse_rounded").iloc[0]
    L.append("| view | all (median / best sse) | exclude_outage_severe (median / best) | quality_weighted (median / best) |")
    L.append("| --- | --- | --- | --- |")
    for label, frame, agg in [
        ("final_analog (sse)", final_scores, "sse_rounded"),
        ("fold_08 normal (sse)", fold08_scores, "sse_rounded"),
        ("all-fold normal (mse)", allnorm_scores, "mse_rounded"),
    ]:
        cells = []
        for p in POLICIES:
            sub = frame[frame["training_policy"] == p]
            med = sub[agg].median()
            best = sub[agg].min()
            bestmethod = sub.loc[sub[agg].idxmin(), "method"]
            cells.append(f"{med:.1f} / {best:.1f} ({bestmethod})")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    # per-method policy winner on final_analog and fold08_normal
    fa_piv = final_scores.pivot_table(index="method", columns="training_policy", values="sse_rounded")
    f8_piv = fold08_scores.pivot_table(index="method", columns="training_policy", values="sse_rounded")
    fa_win = fa_piv.idxmin(axis=1).value_counts().to_dict()
    f8_win = f8_piv.idxmin(axis=1).value_counts().to_dict()
    L.append(f"- Per-method best policy on final_analog: {fa_win}.")
    L.append(f"- Per-method best policy on fold_08 normal dates: {f8_win}.")
    L.append("- These normal-target views (not the Step 03 overall median) are the basis for judging whether excluding or down-weighting the anomalous training days helps when future targets are normal.")
    L.append("")

    # 4 scaling
    L.append("## 4. Daily vessel-count scaling comparison")
    L.append("")
    scal = ["hour_mean", "hour_scale_p025", "hour_scale_p05"]
    L.append("`hour_mean` vs `hour_scale_p025` vs `hour_scale_p05` (lower is better). Scaling is worth keeping only if p025/p05 beat hour_mean on a majority of these normal-target views.")
    L.append("")
    for label, frame, agg, policies in [
        ("final_analog (sse)", final_scores, "sse_rounded", POLICIES),
        ("fold_08 normal (sse)", fold08_scores, "sse_rounded", POLICIES),
        ("all-fold normal (mse)", allnorm_scores, "mse_rounded", POLICIES),
    ]:
        L.append(f"**{label}**")
        L.append("")
        L.append("| policy | hour_mean | hour_scale_p025 | hour_scale_p05 |")
        L.append("| --- | --- | --- | --- |")
        for p in policies:
            row = []
            for mname in scal:
                val = frame[(frame["training_policy"] == p) & (frame["method"] == mname)][agg]
                row.append(f"{val.iloc[0]:.1f}" if len(val) else "n/a")
            L.append(f"| {p} | " + " | ".join(row) + " |")
        L.append("")

    # 5 region
    L.append("## 5. Error by region")
    L.append("")
    L.append("Per-(policy, method) region SSE on the final-analog fold (region errors are reported per model, not averaged across models):")
    L.append("")
    L.append("| policy | method | core | near | outer | total |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    fs_sorted = final_scores.sort_values(["training_policy", "method"])
    for _, r in fs_sorted.iterrows():
        L.append(f"| {r['training_policy']} | {r['method']} | {r['core_sse_rounded']:.0f} | "
                 f"{r['near_sse_rounded']:.0f} | {r['outer_sse_rounded']:.0f} | {r['sse_rounded']:.0f} |")
    L.append("")

    # 6 decision
    L.append("## 6. Decision for the next modeling stage")
    L.append("")
    fa_top = final_scores.sort_values("sse_rounded").head(3)
    L.append("Top-3 (policy, method) on the final-analog fold:")
    for _, r in fa_top.iterrows():
        L.append(f"- {r['training_policy']} / {r['method']}: sse={r['sse_rounded']:.0f}, bias={r['mean_bias_rounded']:.2f}")
    L.append("")
    # scaling verdict: does p05/p025 beat hour_mean on >=2 of 3 normal views, for policy all?
    scaling_wins = {m: 0 for m in ["hour_scale_p025", "hour_scale_p05"]}
    for frame, agg in [(final_scores, "sse_rounded"), (fold08_scores, "sse_rounded"), (allnorm_scores, "mse_rounded")]:
        hm = frame[(frame["training_policy"] == "all") & (frame["method"] == "hour_mean")][agg]
        hm_val = hm.iloc[0] if len(hm) else np.inf
        for mname in scaling_wins:
            v = frame[(frame["training_policy"] == "all") & (frame["method"] == mname)][agg]
            if len(v) and v.iloc[0] < hm_val:
                scaling_wins[mname] += 1
    L.append(f"- Scaling vs hour_mean (policy=all, normal-target views where the scaler wins): hour_scale_p025={scaling_wins['hour_scale_p025']}/3, hour_scale_p05={scaling_wins['hour_scale_p05']}/3. Keep the scaler only if it wins a majority.")
    L.append("- Candidate training policies and methods should be carried forward based on the normal-target views in sections 3-4, not the Step 03 overall median. The decision table (decision_table.csv) lists the separate rankings without any composite score or single champion.")
    L.append("- This stage does not develop GLM / GAM / LightGBM / neural-net or any new model.")
    L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
