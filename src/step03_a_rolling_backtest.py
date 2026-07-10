"""Step 03 — Strict 7-day rolling backtest for Task A.

Build an expanding-window rolling backtest (8 folds) and compare the existing
simple baselines from ``optimized_baseline`` under three training-data quality
strategies (all / exclude_outage_severe / quality_weighted).

A-label construction is reused verbatim (``add_regions`` / ``make_a_labels``);
baseline method semantics are reused via ``MethodSpec`` / ``predict_base``. A
weighted prediction function is implemented here so date-level quality weights
can be applied; when all weights are 1 (the ``all`` policy) it must reproduce
``predict_base`` exactly (asserted per fold and method to < 1e-10).

Outputs (under outputs/step03_a_rolling_backtest/):
  1. fold_definitions.csv
  2. a_backtest_predictions.csv   (8 folds x 8 methods x 3 policies x 504 = 96768)
  3. a_fold_scores.csv            (192)
  4. a_component_scores.csv
  5. model_summary.csv            (24)
  6. summary.md

No B task, no new models, no submission files, no random numbers.
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

from optimized_baseline import (  # noqa: E402
    MethodSpec,
    REGION_ORDER,
    add_regions,
    make_a_labels,
    predict_base,
)

TRAIN_REL = Path("数据备份/训练集_20180101-0124_拖轮AIS.csv")
STEP01_DAILY = Path("outputs/step01_data_audit/train_daily_overview.csv")
STEP02_DAILY = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
OUT_REL = Path("outputs/step03_a_rolling_backtest")

FIXED_DATES = pd.date_range("2018-01-01", "2018-01-24", freq="D").normalize()
REGION_RANK = {r: i for i, r in enumerate(REGION_ORDER)}

# 8 baseline methods; MethodSpec objects match optimized_baseline.method_specs()
METHODS = [
    ("global_mean", MethodSpec("global_mean", "global")),
    ("hour_mean", MethodSpec("hour_mean", "hour")),
    ("dow_hour_mean", MethodSpec("dow_hour_mean", "dow_hour")),
    ("recent_7d_hour", MethodSpec("recent_7d_hour", "recent_hour", recent_days=7)),
    ("recent_14d_hour", MethodSpec("recent_14d_hour", "recent_hour", recent_days=14)),
    ("mixed_075_hour_dow", MethodSpec("mixed_075_hour_dow", "mixed", alpha=0.75)),
    ("hour_scale_p025", MethodSpec("hour_scale_p025", "hour", scale_power=0.25)),
    ("hour_scale_p05", MethodSpec("hour_scale_p05", "hour", scale_power=0.5)),
]
METHOD_ORDER = {name: i for i, (name, _) in enumerate(METHODS)}

# Three training-data quality policies expressed as per-date weight maps keyed
# by the fold-local quality class. "all" is uniform 1 -> reproduces predict_base.
POLICIES = ["all", "exclude_outage_severe", "quality_weighted"]
POLICY_ORDER = {p: i for i, p in enumerate(POLICIES)}
POLICY_WEIGHTS = {
    "all": {"normal": 1.0, "reduced": 1.0, "severe": 1.0, "outage": 1.0},
    "exclude_outage_severe": {"normal": 1.0, "reduced": 1.0, "severe": 0.0, "outage": 0.0},
    "quality_weighted": {"normal": 1.0, "reduced": 0.5, "severe": 0.1, "outage": 0.0},
}

RUN_CMD = "python " + " ".join(sys.argv)

PRED_COLUMNS = [
    "fold_id", "training_policy", "method", "hour", "date", "hour_of_day", "dayofweek",
    "region", "validation_unique_vessel_count", "audit_quality_regime",
    "y_true", "pred_float", "pred_rounded", "error_float", "error_rounded",
]
FOLD_SCORE_COLUMNS = [
    "fold_id", "training_policy", "method", "train_day_count_used",
    "excluded_train_day_count", "effective_train_day_weight",
    "sse_float", "sse_rounded", "mae_rounded", "mean_bias_rounded",
    "actual_total", "predicted_total_rounded",
]
COMPONENT_COLUMNS = [
    "fold_id", "training_policy", "method", "component_type", "component",
    "n_rows", "sse_float", "sse_rounded", "mae_rounded", "mean_bias_rounded",
]
MODEL_SUMMARY_COLUMNS = [
    "training_policy", "method", "n_folds",
    "mean_sse_rounded", "median_sse_rounded", "min_sse_rounded", "max_sse_rounded",
    "std_sse_rounded", "last_fold_sse_rounded", "mean_mae_rounded", "mean_bias_rounded",
    "fold_win_count", "rank_by_median_sse",
]

CONSISTENCY_TOL = 1e-10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_ais(train_path: Path) -> pd.DataFrame:
    usecols = ["mmsi", "x", "y", "sog", "time", "source_dataset"]
    dtypes = {"mmsi": "string", "x": "float64", "y": "float64", "sog": "float32", "source_dataset": "string"}
    df = pd.read_csv(train_path, usecols=usecols, dtype=dtypes, parse_dates=["time"])
    df["hour"] = df["time"].dt.floor("h")
    df["date"] = df["time"].dt.normalize()
    df = add_regions(df)
    return df


def build_a_frame(df: pd.DataFrame, daily: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    a_lab = make_a_labels(df)  # hour, region, y, task_key
    a_lab["date"] = a_lab["hour"].dt.normalize()
    a_lab["hour_of_day"] = a_lab["hour"].dt.hour
    a_lab["dayofweek"] = a_lab["hour"].dt.dayofweek
    a_lab = a_lab[["hour", "date", "hour_of_day", "dayofweek", "region", "y"]]

    vc = daily[["date", "unique_vessel_count"]].rename(columns={"unique_vessel_count": "vessel_count"})
    a_frame = a_lab.merge(vc, on="date", how="left")
    aq = audit[["date", "quality_regime"]].rename(columns={"quality_regime": "audit_quality_regime"})
    a_frame = a_frame.merge(aq, on="date", how="left")
    a_frame["region_rank"] = a_frame["region"].map(REGION_RANK)
    return a_frame


# ---------------------------------------------------------------------------
# Fold + per-fold quality classification
# ---------------------------------------------------------------------------
def build_folds() -> list[dict]:
    folds = []
    for k in range(8):
        train_end_day = 10 + k      # 10..17
        valid_start_day = 11 + k    # 11..18
        valid_end_day = 17 + k      # 17..24
        folds.append({
            "fold_id": f"fold_{k + 1:02d}",
            "train_start": pd.Timestamp("2018-01-01"),
            "train_end": pd.Timestamp(f"2018-01-{train_end_day:02d}"),
            "valid_start": pd.Timestamp(f"2018-01-{valid_start_day:02d}"),
            "valid_end": pd.Timestamp(f"2018-01-{valid_end_day:02d}"),
        })
    return folds


def classify_fold_quality(train_dates: list[pd.Timestamp], china_series: dict) -> tuple[dict, float]:
    """Recompute outage/severe/reduced/normal using ONLY this fold's training dates."""
    counts = [china_series[d] for d in train_dates]
    nonzero = [c for c in counts if c > 0]
    m_train = float(np.median(nonzero)) if nonzero else float("nan")
    classes: dict = {}
    for d in train_dates:
        c = china_series[d]
        if c == 0:
            classes[d] = "outage"
        else:
            ratio = c / m_train
            if ratio < 0.10:
                classes[d] = "severe"
            elif ratio < 0.50:
                classes[d] = "reduced"
            else:
                classes[d] = "normal"
    return classes, m_train


def scale_train_mean(train_dates, vc_series, weights, policy) -> float:
    """Per-policy training mean of daily vessel_count used by the scaling methods."""
    if policy == "quality_weighted":
        num = sum(weights[d] * vc_series[d] for d in train_dates)
        den = sum(weights[d] for d in train_dates)
        return num / den if den > 0 else float("nan")
    inc = [vc_series[d] for d in train_dates if weights[d] > 0]
    return float(np.mean(inc)) if inc else float("nan")


# ---------------------------------------------------------------------------
# Weighted prediction (mirrors optimized_baseline.predict_base with weights)
# ---------------------------------------------------------------------------
def _wmean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Weighted mean of y grouped by cols -> Series indexed by cols."""
    if len(df) == 0:
        return pd.Series(dtype="float64")
    tmp = df.assign(_wy=df["y"].astype("float64") * df["weight"].astype("float64"))
    agg = tmp.groupby(cols, observed=True).agg(_wy=("_wy", "sum"), _w=("weight", "sum"))
    return agg["_wy"] / agg["_w"]


def predict_weighted(train: pd.DataFrame, future: pd.DataFrame, spec: MethodSpec, scale_mean: float) -> pd.Series:
    cols_key = ["region"]
    train = train.copy()
    future = future.copy().reset_index(drop=True)

    group_mean = _wmean(train, cols_key)
    wsum = float(train["weight"].sum())
    global_mean = float((train["y"].astype("float64") * train["weight"].astype("float64")).sum() / wsum) if wsum > 0 else 0.0

    def with_key_mean(merged: pd.DataFrame) -> pd.DataFrame:
        return merged.merge(group_mean.rename("key_mean").reset_index(), on=cols_key, how="left")

    if spec.base == "global":
        m = future.merge(group_mean.rename("key_mean").reset_index(), on=cols_key, how="left")
        pred = m["key_mean"].fillna(global_mean)
    elif spec.base == "hour":
        cols = cols_key + ["hour_of_day"]
        mean = _wmean(train, cols)
        m = future.merge(mean.rename("pred").reset_index(), on=cols, how="left")
        m = with_key_mean(m)
        pred = m["pred"].fillna(m["key_mean"]).fillna(global_mean)
    elif spec.base == "dow_hour":
        cols = cols_key + ["dayofweek", "hour_of_day"]
        mean = _wmean(train, cols)
        m = future.merge(mean.rename("pred").reset_index(), on=cols, how="left")
        m = with_key_mean(m)
        pred = m["pred"].fillna(m["key_mean"]).fillna(global_mean)
    elif spec.base == "recent_hour":
        cutoff = train["hour"].max() - pd.Timedelta(days=spec.recent_days)
        recent = train[train["hour"] > cutoff]
        cols = cols_key + ["hour_of_day"]
        mean = _wmean(recent, cols)
        m = future.merge(mean.rename("pred").reset_index(), on=cols, how="left")
        m = with_key_mean(m)
        pred = m["pred"].fillna(m["key_mean"]).fillna(global_mean)
    elif spec.base == "mixed":
        hour = predict_weighted(train, future, MethodSpec("hour_mean", "hour"), scale_mean)
        dow_hour = predict_weighted(train, future, MethodSpec("dow_hour_mean", "dow_hour"), scale_mean)
        pred = spec.alpha * hour + (1.0 - spec.alpha) * dow_hour
    else:
        raise ValueError(spec.base)

    pred = pd.Series(pred).reset_index(drop=True)
    if spec.scale_power:
        scale = (future["vessel_count"].reset_index(drop=True) / scale_mean).fillna(1.0)
        pred = pred * np.power(scale, spec.scale_power)
    if len(pred) != len(future):
        raise ValueError(f"{spec.name} produced {len(pred)} preds for {len(future)} rows")
    return pred.clip(lower=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(y_true: np.ndarray, pred_float: np.ndarray, pred_rounded: np.ndarray) -> dict:
    ef = pred_float - y_true
    er = pred_rounded - y_true
    return {
        "sse_float": float((ef ** 2).sum()),
        "sse_rounded": float((er ** 2).sum()),
        "mae_rounded": float(np.abs(er).mean()),
        "mean_bias_rounded": float(er.mean()),
        "actual_total": float(y_true.sum()),
        "predicted_total_rounded": float(pred_rounded.sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    train_path = ROOT / TRAIN_REL
    out_dir = ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    if not train_path.exists():
        raise FileNotFoundError(f"Training AIS file not found: {train_path}")

    print(f"train_path = {train_path}")
    print(f"out_dir    = {out_dir}")

    df = load_ais(train_path)

    daily = pd.read_csv(ROOT / STEP01_DAILY)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    audit = pd.read_csv(ROOT / STEP02_DAILY)
    audit["date"] = pd.to_datetime(audit["date"]).dt.normalize()

    # --- assertions on inputs ------------------------------------------------
    expected = set(FIXED_DATES)
    assert set(daily["date"]) == expected, "step01 daily does not cover 2018-01-01..01-24"
    assert daily["unique_vessel_count"].notna().all(), "missing unique_vessel_count in step01"
    assert set(audit["date"]) == expected, "step02 daily does not cover 2018-01-01..01-24"

    a_frame = build_a_frame(df, daily, audit)
    assert len(a_frame) == 1728, f"A labels expected 1728 rows, got {len(a_frame)}"
    per_date = a_frame.groupby("date").size()
    assert (per_date == 72).all(), "each date must have 72 A-label rows (24h x 3 regions)"
    assert a_frame["vessel_count"].notna().all()
    assert a_frame["audit_quality_regime"].notna().all()

    china_series = dict(zip(audit["date"], audit["china_coastal_record_count"]))
    vc_series = dict(zip(daily["date"], daily["unique_vessel_count"]))

    folds = build_folds()

    pred_frames: list[pd.DataFrame] = []
    fold_score_rows: list[dict] = []
    component_rows: list[dict] = []

    for fold in folds:
        fid = fold["fold_id"]
        train_mask = (a_frame["date"] >= fold["train_start"]) & (a_frame["date"] <= fold["train_end"])
        valid_mask = (a_frame["date"] >= fold["valid_start"]) & (a_frame["date"] <= fold["valid_end"])
        train_full = a_frame[train_mask].copy()
        valid = a_frame[valid_mask].copy().sort_values(["hour", "region_rank"]).reset_index(drop=True)
        assert len(valid) == 504, f"{fid}: valid window must have 504 rows, got {len(valid)}"

        train_dates = sorted(pd.date_range(fold["train_start"], fold["train_end"], freq="D").normalize())
        classes, _ = classify_fold_quality(train_dates, china_series)

        for policy in POLICIES:
            pw = POLICY_WEIGHTS[policy]
            weights = {d: pw[classes[d]] for d in train_dates}
            train_w = train_full.copy()
            train_w["weight"] = train_w["date"].map(weights).astype("float64")
            smean = scale_train_mean(train_dates, vc_series, weights, policy)

            used = int(sum(1 for d in train_dates if weights[d] > 0))
            excluded = int(sum(1 for d in train_dates if weights[d] == 0))
            eff_weight = float(sum(weights.values()))

            for mname, spec in METHODS:
                pred = predict_weighted(train_w, valid, spec, smean)
                pred = np.asarray(pred, dtype="float64")

                # consistency: "all" policy must reproduce optimized_baseline.predict_base
                if policy == "all":
                    ref = np.asarray(predict_base(train_w, valid, "A", spec), dtype="float64")
                    diff = float(np.max(np.abs(pred - ref)))
                    if diff >= CONSISTENCY_TOL:
                        raise AssertionError(
                            f"consistency fail {fid} all {mname}: max abs diff {diff:.3e} >= {CONSISTENCY_TOL}"
                        )

                pred_rounded = np.clip(np.rint(pred), 0, None).astype(np.int64)
                y_true = valid["y"].to_numpy(dtype="float64")

                combo = pd.DataFrame({
                    "fold_id": fid,
                    "training_policy": policy,
                    "method": mname,
                    "hour": valid["hour"].to_numpy(),
                    "date": valid["date"].to_numpy(),
                    "hour_of_day": valid["hour_of_day"].to_numpy(),
                    "dayofweek": valid["dayofweek"].to_numpy(),
                    "region": valid["region"].to_numpy(),
                    "validation_unique_vessel_count": valid["vessel_count"].to_numpy(),
                    "audit_quality_regime": valid["audit_quality_regime"].to_numpy(),
                    "y_true": y_true,
                    "pred_float": pred,
                    "pred_rounded": pred_rounded,
                    "region_rank": valid["region_rank"].to_numpy(),
                })
                combo["error_float"] = combo["pred_float"] - combo["y_true"]
                combo["error_rounded"] = combo["pred_rounded"] - combo["y_true"]
                pred_frames.append(combo)

                m = metrics(y_true, pred, pred_rounded)
                fold_score_rows.append({
                    "fold_id": fid, "training_policy": policy, "method": mname,
                    "train_day_count_used": used,
                    "excluded_train_day_count": excluded,
                    "effective_train_day_weight": eff_weight,
                    **{k: m[k] for k in ["sse_float", "sse_rounded", "mae_rounded",
                                         "mean_bias_rounded", "actual_total", "predicted_total_rounded"]},
                })

                # component scores: by region and by validation audit quality
                for ctype, groupcol in [("region", "region"), ("validation_quality", "audit_quality_regime")]:
                    if ctype == "region":
                        keys = REGION_ORDER
                    else:
                        present = combo[groupcol].unique().tolist()
                        keys = [q for q in ["normal", "reduced", "severe", "outage"] if q in present]
                    for comp in keys:
                        sub = combo[combo[groupcol] == comp]
                        er = sub["error_rounded"].to_numpy(dtype="float64")
                        ef = sub["error_float"].to_numpy(dtype="float64")
                        component_rows.append({
                            "fold_id": fid, "training_policy": policy, "method": mname,
                            "component_type": ctype, "component": comp, "n_rows": int(len(sub)),
                            "sse_float": float((ef ** 2).sum()),
                            "sse_rounded": float((er ** 2).sum()),
                            "mae_rounded": float(np.abs(er).mean()) if len(er) else 0.0,
                            "mean_bias_rounded": float(er.mean()) if len(er) else 0.0,
                        })

    preds = pd.concat(pred_frames, ignore_index=True)
    preds["_fold_order"] = preds["fold_id"].map({f["fold_id"]: i for i, f in enumerate(folds)})
    preds["_policy_order"] = preds["training_policy"].map(POLICY_ORDER)
    preds["_method_order"] = preds["method"].map(METHOD_ORDER)
    preds = preds.sort_values(["_fold_order", "_policy_order", "_method_order", "hour", "region_rank"]).reset_index(drop=True)

    fold_scores = pd.DataFrame(fold_score_rows)
    components = pd.DataFrame(component_rows)

    # --- fold_definitions ----------------------------------------------------
    fold_def_rows = []
    regime_order = ["normal", "reduced", "severe", "outage"]
    for fold in folds:
        vdates = pd.date_range(fold["valid_start"], fold["valid_end"], freq="D").normalize()
        vreg = [audit.loc[audit["date"] == d, "quality_regime"].iloc[0] for d in vdates]
        counts = {r: vreg.count(r) for r in regime_order}
        fold_def_rows.append({
            "fold_id": fold["fold_id"],
            "train_start": f"{fold['train_start']:%Y-%m-%d}",
            "train_end": f"{fold['train_end']:%Y-%m-%d}",
            "valid_start": f"{fold['valid_start']:%Y-%m-%d}",
            "valid_end": f"{fold['valid_end']:%Y-%m-%d}",
            "train_day_count": int((fold["train_end"] - fold["train_start"]).days) + 1,
            "valid_day_count": int((fold["valid_end"] - fold["valid_start"]).days) + 1,
            "valid_normal_day_count": counts["normal"],
            "valid_reduced_day_count": counts["reduced"],
            "valid_severe_day_count": counts["severe"],
            "valid_outage_day_count": counts["outage"],
        })
    fold_defs = pd.DataFrame(fold_def_rows)

    # --- model_summary -------------------------------------------------------
    win_pivot = fold_scores.pivot_table(index="fold_id", columns=["training_policy", "method"], values="sse_rounded", aggfunc="first")
    min_per_fold = win_pivot.min(axis=1)
    win_matrix = win_pivot.eq(min_per_fold, axis=0)
    win_counts = win_matrix.sum(axis=0)  # MultiIndex (policy, method) -> count

    summary_rows = []
    for policy in POLICIES:
        for mname, _ in METHODS:
            sub = fold_scores[(fold_scores["training_policy"] == policy) & (fold_scores["method"] == mname)].copy()
            sub = sub.sort_values("fold_id")
            sse = sub["sse_rounded"].to_numpy(dtype="float64")
            summary_rows.append({
                "training_policy": policy, "method": mname, "n_folds": int(len(sub)),
                "mean_sse_rounded": float(np.mean(sse)),
                "median_sse_rounded": float(np.median(sse)),
                "min_sse_rounded": float(np.min(sse)),
                "max_sse_rounded": float(np.max(sse)),
                "std_sse_rounded": float(np.std(sse, ddof=1)) if len(sse) > 1 else 0.0,
                "last_fold_sse_rounded": float(sse[-1]) if len(sse) else float("nan"),
                "mean_mae_rounded": float(sub["mae_rounded"].mean()),
                "mean_bias_rounded": float(sub["mean_bias_rounded"].mean()),
                "fold_win_count": int(win_counts.get((policy, mname), 0)),
            })
    model_summary = pd.DataFrame(summary_rows)
    model_summary["rank_by_median_sse"] = model_summary["median_sse_rounded"].rank(method="min", ascending=True).astype(int)
    model_summary = model_summary.sort_values(["median_sse_rounded", "training_policy", "method"]).reset_index(drop=True)

    # --- write CSVs (UTF-8 with BOM) ----------------------------------------
    p_folddef = out_dir / "fold_definitions.csv"
    p_pred = out_dir / "a_backtest_predictions.csv"
    p_fscore = out_dir / "a_fold_scores.csv"
    p_comp = out_dir / "a_component_scores.csv"
    p_msum = out_dir / "model_summary.csv"
    p_summary = out_dir / "summary.md"

    fold_defs.to_csv(p_folddef, index=False, encoding="utf-8-sig")

    pred_out = preds[PRED_COLUMNS].copy()
    pred_out["hour"] = pd.to_datetime(pred_out["hour"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    pred_out["date"] = pd.to_datetime(pred_out["date"]).dt.strftime("%Y-%m-%d")
    pred_out.to_csv(p_pred, index=False, encoding="utf-8-sig")

    fold_scores = fold_scores.sort_values(["fold_id", "training_policy", "method"]).reset_index(drop=True)
    fold_scores.to_csv(p_fscore, index=False, encoding="utf-8-sig")

    comp_order = components.assign(
        _p=components["training_policy"].map(POLICY_ORDER),
        _m=components["method"].map(METHOD_ORDER),
        _ct=components["component_type"].map({"region": 0, "validation_quality": 1}),
        _c=components["component"].map({**{r: i for i, r in enumerate(REGION_ORDER)},
                                       **{q: i for i, q in enumerate(["normal", "reduced", "severe", "outage"])}}),
    )
    comp_order = comp_order.sort_values(["fold_id", "_p", "_m", "_ct", "_c"]).drop(columns=["_p", "_m", "_ct", "_c"])
    comp_order[COMPONENT_COLUMNS].to_csv(p_comp, index=False, encoding="utf-8-sig")

    model_summary_out = model_summary[MODEL_SUMMARY_COLUMNS].sort_values(
        ["training_policy", "method"]
    ).reset_index(drop=True)
    model_summary_out.to_csv(p_msum, index=False, encoding="utf-8-sig")

    # --- summary.md ----------------------------------------------------------
    write_summary(p_summary, fold_defs, fold_scores, model_summary, preds, components)

    # --- console report ------------------------------------------------------
    print("\n=== Output files ===")
    for path in [p_folddef, p_pred, p_fscore, p_comp, p_msum, p_summary]:
        if path.suffix == ".csv":
            rows = len(pd.read_csv(path))
        else:
            rows = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {path.relative_to(ROOT)}  rows={rows}")

    print("\n=== Top 10 (policy, method) by median_sse_rounded ===")
    top = model_summary.head(10)
    for _, r in top.iterrows():
        print(f"  rank={int(r['rank_by_median_sse']):2d}  {r['training_policy']:22s} {r['method']:20s} "
              f"median={r['median_sse_rounded']:.1f} mean={r['mean_sse_rounded']:.1f} "
              f"max={r['max_sse_rounded']:.1f} last={r['last_fold_sse_rounded']:.1f} wins={int(r['fold_win_count'])}")
    print("\nDone.")


def write_summary(path: Path, fold_defs: pd.DataFrame, fold_scores: pd.DataFrame,
                  model_summary: pd.DataFrame, preds: pd.DataFrame, components: pd.DataFrame) -> None:
    L: list[str] = []
    L.append("# Step 03 A Rolling Backtest")
    L.append("")
    L.append("> Diagnostic backtest for Task A only. No B task, no new models, no submission files.")
    L.append("")
    L.append(f"Run command: `{RUN_CMD}`")
    L.append("")

    # 1. Backtest design
    L.append("## 1. Backtest design")
    L.append("")
    L.append("Eight expanding-window folds. The training start is fixed at 2018-01-01 and the training "
             "end rolls forward by one day each fold; each validation window is the next 7 calendar days. "
             "A model in a given fold may use only data dated on or before that fold's training end.")
    L.append("")
    L.append("| fold | train | valid |")
    L.append("| --- | --- | --- |")
    for _, r in fold_defs.iterrows():
        L.append(f"| {r['fold_id']} | {r['train_start']}~{r['train_end']} ({int(r['train_day_count'])}d) | "
                 f"{r['valid_start']}~{r['valid_end']} ({int(r['valid_day_count'])}d) |")
    L.append("")

    # 2. Leakage checks
    L.append("## 2. Leakage checks")
    L.append("")
    L.append("- Validation A labels never enter prediction (they are only used to score after predicting).")
    L.append("- Validation AIS-quality variables (ais_record_count, china_coastal_record_count, audit_quality_regime) never enter prediction.")
    L.append("- The only exogenous variable used from the validation period is `unique_vessel_count` (it simulates the future daily vessel count that the competition provides).")
    L.append("- The training quality class (outage/severe/reduced/normal) is recomputed inside each fold from only that fold's training dates (`M_china_train` = median of non-zero china_coastal daily records within the fold's training span). The global Step 02 `quality_regime` is used only for post-hoc validation grouping, never for training or prediction.")
    L.append("- Consistency: for the `all` policy, the new weighted prediction function reproduces `optimized_baseline.predict_base` for all 8 folds and all 8 methods with max absolute difference < 1e-10. **PASS** (asserted in code).")
    L.append("")

    # 3. Fold composition
    L.append("## 3. Fold composition")
    L.append("")
    L.append("Validation-day counts by global audit quality (post-hoc description only):")
    L.append("")
    L.append("| fold | valid normal | reduced | severe | outage |")
    L.append("| --- | --- | --- | --- | --- |")
    for _, r in fold_defs.iterrows():
        L.append(f"| {r['fold_id']} | {r['valid_normal_day_count']} | {r['valid_reduced_day_count']} | "
                 f"{r['valid_severe_day_count']} | {r['valid_outage_day_count']} |")
    L.append("")

    # 4. Baseline ranking
    L.append("## 4. Baseline ranking")
    L.append("")
    L.append("All 24 (training_policy, method) combinations ranked by `median_sse_rounded` across the 8 folds. "
             "Prediction errors are integers (rounded), so `sse_rounded` is the ranking metric used for real submission.")
    L.append("")
    L.append("| rank | policy | method | median | mean | worst | last fold | wins |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, r in model_summary.iterrows():
        L.append(f"| {int(r['rank_by_median_sse'])} | {r['training_policy']} | {r['method']} | "
                 f"{r['median_sse_rounded']:.1f} | {r['mean_sse_rounded']:.1f} | {r['max_sse_rounded']:.1f} | "
                 f"{r['last_fold_sse_rounded']:.1f} | {int(r['fold_win_count'])} |")
    L.append("")

    # 5. Effect of training-quality treatment
    L.append("## 5. Effect of training-quality treatment")
    L.append("")
    L.append("Median `sse_rounded` by method under each training policy:")
    L.append("")
    piv = model_summary.pivot_table(index="method", columns="training_policy", values="median_sse_rounded")
    piv = piv.reindex([m for m, _ in METHODS])[[c for c in ["all", "exclude_outage_severe", "quality_weighted"] if c in piv.columns]]
    header = "| method | " + " | ".join(piv.columns) + " |"
    L.append(header)
    L.append("| --- | " + " | ".join(["---"] * len(piv.columns)) + " |")
    for mname, row in piv.iterrows():
        L.append(f"| {mname} | " + " | ".join(f"{row[c]:.1f}" for c in piv.columns) + " |")
    L.append("")
    best_policy_counts = {p: 0 for p in piv.columns}
    for mname, row in piv.iterrows():
        bp = row.idxmin()
        best_policy_counts[bp] += 1
    detail = ", ".join(f"{p}: {c}" for p, c in best_policy_counts.items())
    L.append(f"- Per-method best policy count (by median sse_rounded) — {detail}.")
    L.append("- Whether excluding or down-weighting the outage/severe training days helps should be judged across methods and folds, not from a single fold.")
    L.append("")

    # 6. Effect of daily vessel-count scaling
    L.append("## 6. Effect of daily vessel-count scaling")
    L.append("")
    scal = ["hour_mean", "hour_scale_p025", "hour_scale_p05"]
    sp = model_summary[model_summary["method"].isin(scal)].pivot_table(index="training_policy", columns="method", values="median_sse_rounded")
    sp = sp.reindex(columns=scal)
    L.append("Median `sse_rounded` for the scaling family by policy:")
    L.append("")
    L.append("| policy | hour_mean | hour_scale_p025 | hour_scale_p05 |")
    L.append("| --- | --- | --- | --- |")
    for policy, row in sp.iterrows():
        L.append(f"| {policy} | {row['hour_mean']:.1f} | {row['hour_scale_p025']:.1f} | {row['hour_scale_p05']:.1f} |")
    L.append("")
    # normal vs degraded validation-day MSE for the scaling family, policy all
    scal_pred = preds[(preds["training_policy"] == "all") & (preds["method"].isin(scal))].copy()
    scal_pred["sq_err"] = scal_pred["error_rounded"].astype(float) ** 2
    scal_pred["qgroup"] = scal_pred["audit_quality_regime"].map(
        lambda q: "normal" if q == "normal" else ("degraded" if q in ("severe", "outage") else "reduced")
    )
    mse_t = scal_pred.groupby(["method", "qgroup"])["sq_err"].mean().unstack()
    L.append("Mean squared rounded error on validation days by quality group (policy=all; folds overlap, so this is descriptive, not independent):")
    L.append("")
    cols = [c for c in ["normal", "reduced", "degraded"] if c in mse_t.columns]
    L.append("| method | " + " | ".join(cols) + " |")
    L.append("| --- | " + " | ".join(["---"] * len(cols)) + " |")
    for mname, row in mse_t.iterrows():
        L.append(f"| {mname} | " + " | ".join(f"{row[c]:.3f}" for c in cols) + " |")
    L.append("")
    L.append("- A single correlation is not causal evidence; these numbers only describe whether the daily vessel-count multiplier changes error across folds and quality groups.")
    L.append("")

    # 7. Error by region and validation quality
    L.append("## 7. Error by region and validation quality")
    L.append("")
    allpred = preds[preds["training_policy"] == "all"].copy()
    allpred["sq_err"] = allpred["error_rounded"].astype(float) ** 2
    reg_mse = allpred.groupby("region")["sq_err"].mean().reindex(REGION_ORDER)
    L.append("Mean squared rounded error by region (policy=all, averaged across the 8 methods; folds overlap):")
    L.append("")
    L.append("| region | MSE | mean bias |")
    L.append("| --- | --- | --- |")
    for r in REGION_ORDER:
        sub = allpred[allpred["region"] == r]
        L.append(f"| {r} | {reg_mse[r]:.3f} | {sub['error_rounded'].astype(float).mean():.3f} |")
    L.append("")
    q_order = [q for q in ["normal", "reduced", "severe", "outage"] if q in allpred["audit_quality_regime"].unique()]
    L.append("Mean squared rounded error by validation-day audit quality (policy=all, averaged across methods; folds overlap):")
    L.append("")
    L.append("| quality | MSE | mean bias | n_rows |")
    L.append("| --- | --- | --- | --- |")
    for q in q_order:
        sub = allpred[allpred["audit_quality_regime"] == q]
        L.append(f"| {q} | {(sub['error_rounded'].astype(float) ** 2).mean():.3f} | {sub['error_rounded'].astype(float).mean():.3f} | {len(sub)} |")
    L.append("")
    L.append("- The 8 folds overlap heavily (each validation day appears in up to 7 folds), so they must not be treated as 8 independent samples. Aggregates here are descriptive only.")
    L.append("")

    # 8. Conclusions for next stage
    L.append("## 8. Conclusions for the next stage")
    L.append("")
    top_combo = model_summary.iloc[0]
    L.append(f"- Best (policy, method) by median `sse_rounded`: **{top_combo['training_policy']} / {top_combo['method']}** (median {top_combo['median_sse_rounded']:.1f}).")
    L.append("- The backtest framework is internally consistent (the `all` policy reproduces the existing baseline exactly), so cross-method comparisons on the same folds are meaningful.")
    best_policy = max(best_policy_counts, key=best_policy_counts.get)
    L.append(f"- Training-quality treatment: the policy that is best for the most methods is **{best_policy}**; confirm this is stable across folds before adopting it.")
    L.append("- Daily vessel-count scaling: judge from section 6 whether the multiplier improves error stably; do not infer causation from a single correlation.")
    hardest = reg_mse.idxmax()
    L.append(f"- Hardest region to predict (highest MSE, policy=all): **{hardest}**.")
    L.append("- No new models are trained in this stage and no final prediction is produced.")
    L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
