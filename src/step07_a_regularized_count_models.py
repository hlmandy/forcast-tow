"""Step 07 — Low-parameter regularized statistical count models for Task A.

Compares direct-hourly and daily-total-decomposition regularized models (Ridge,
Poisson) with/without a daily vessel-count feature and with mean vs
daytype-shrunk profiles, against the Step 06 baselines. No negative-binomial,
no trees/NN, no B task, no submission files.

Alpha is selected by an inner time-series CV (last 4 training dates, each
predicted from strictly-earlier dates, quality classes recomputed per inner
train). The outer validation never selects alpha. Validation uses only the
allowed unique_vessel_count and calendar variables.

Reuses Step 03 (folds, quality weights, predict_weighted, rounding) and
Step 06 (compute_components for mean/daytype profiles). Steps 01-06 are not
modified; their main() does not run on import.

Outputs (under outputs/step07_a_regularized_count_models/):
  1. method_definitions.csv          (16)
  2. hyperparameter_selection.csv    (1620)
  3. a_model_predictions.csv         (214272)
  4. a_fold_scores.csv               (432)
  5. a_normal_target_scores.csv      (48)
  6. a_final_analog_scores.csv       (48)
  7. decision_table.csv              (48)
  8. summary.md
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor, Ridge

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from step03_a_rolling_backtest import (  # noqa: E402
    POLICIES, POLICY_ORDER, POLICY_WEIGHTS, REGION_RANK,
    TRAIN_REL, STEP01_DAILY, STEP02_DAILY,
    classify_fold_quality, predict_weighted, scale_train_mean, build_folds,
)
from step06_a_structure_model_benchmark import compute_components  # noqa: E402
from optimized_baseline import MethodSpec, REGION_ORDER, add_regions, make_a_labels  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

STEP06_PREDS = Path("outputs/step06_a_structure_model_benchmark/a_model_predictions.csv")
OUT_REL = Path("outputs/step07_a_regularized_count_models")
GLOBAL_START = pd.Timestamp("2018-01-01")

RIDGE_ALPHAS = [0.1, 1, 10, 100, 1000]
POISSON_ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1]
ALL_RH = [f"{r}_{h:02d}" for r in REGION_ORDER for h in range(24)]  # core_00 first

# name, family, estimator, target_level, feature_set, profile_method,
# uses_calendar, uses_vessel, requires_alpha
METHODS = [
    ("hour_mean", "baseline", "weighted_mean", "hourly", "", "", False, False, False),
    ("decomp_mean_mean", "baseline", "weighted_mean", "daily_total", "", "mean", False, False, False),
    ("decomp_mean_daytype_shrunk", "baseline", "weighted_mean", "daily_total", "", "daytype_shrunk", False, False, False),
    ("decomp_ewm7_daytype_shrunk", "baseline", "ewm", "daily_total", "", "daytype_shrunk", False, False, False),
    ("ridge_direct_calendar", "direct", "ridge", "hourly", "calendar", "", True, False, True),
    ("ridge_direct_calendar_vessel", "direct", "ridge", "hourly", "calendar_vessel", "", True, True, True),
    ("poisson_direct_calendar", "direct", "poisson", "hourly", "calendar", "", True, False, True),
    ("poisson_direct_calendar_vessel", "direct", "poisson", "hourly", "calendar_vessel", "", True, True, True),
    ("decomp_ridge_calendar_mean", "decomposition", "ridge", "daily_total", "calendar", "mean", True, False, True),
    ("decomp_ridge_calendar_daytype", "decomposition", "ridge", "daily_total", "calendar", "daytype_shrunk", True, False, True),
    ("decomp_ridge_calendar_vessel_mean", "decomposition", "ridge", "daily_total", "calendar_vessel", "mean", True, True, True),
    ("decomp_ridge_calendar_vessel_daytype", "decomposition", "ridge", "daily_total", "calendar_vessel", "daytype_shrunk", True, True, True),
    ("decomp_poisson_calendar_mean", "decomposition", "poisson", "daily_total", "calendar", "mean", True, False, True),
    ("decomp_poisson_calendar_daytype", "decomposition", "poisson", "daily_total", "calendar", "daytype_shrunk", True, False, True),
    ("decomp_poisson_calendar_vessel_mean", "decomposition", "poisson", "daily_total", "calendar_vessel", "mean", True, True, True),
    ("decomp_poisson_calendar_vessel_daytype", "decomposition", "poisson", "daily_total", "calendar_vessel", "daytype_shrunk", True, True, True),
]
METHOD_ORDER = {m[0]: i for i, m in enumerate(METHODS)}
STAT_METHODS = [m for m in METHODS if m[8]]
BASELINE_NAMES = [m[0] for m in METHODS if not m[8]]

PRED_COLUMNS = [
    "evaluation_id", "evaluation_type", "training_policy", "method", "selected_alpha",
    "date", "hour", "hour_of_day", "dayofweek", "day_type", "region", "audit_quality_regime",
    "validation_unique_vessel_count", "y_true", "true_daily_total", "true_profile_share",
    "pred_daily_total", "pred_profile_share", "pred_float", "pred_rounded", "error_float", "error_rounded",
]
FOLD_SCORE_COLUMNS = [
    "evaluation_id", "evaluation_type", "training_policy", "method", "selected_alpha",
    "train_start", "train_end", "valid_start", "valid_end", "n_valid_days", "n_rows",
    "sse_float", "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded",
    "actual_total", "predicted_total_rounded", "core_sse_rounded", "near_sse_rounded", "outer_sse_rounded",
]
NORMAL_COLUMNS = [
    "training_policy", "method", "n_prediction_rows", "n_unique_target_dates",
    "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded", "rank_by_mse_rounded",
]
FINAL_COLUMNS = [
    "training_policy", "method", "selected_alpha", "sse_float", "sse_rounded", "mse_rounded",
    "mae_rounded", "mean_bias_rounded", "actual_total", "predicted_total_rounded",
    "core_sse_rounded", "near_sse_rounded", "outer_sse_rounded", "rank_by_sse_rounded",
]
HP_COLUMNS = [
    "evaluation_id", "evaluation_type", "training_policy", "method", "alpha",
    "n_inner_splits", "inner_validation_dates", "inner_sse_float", "inner_mae_float",
    "fit_failure_count", "selected",
]
DECISION_COLUMNS = [
    "training_policy", "method", "model_family", "estimator", "uses_vessel_count", "profile_method",
    "rolling_mean_sse", "rolling_median_sse", "rolling_max_sse",
    "rolling_normal_mse", "rolling_normal_rank", "final_analog_sse", "final_analog_rank",
    "final_analog_mean_bias", "final_analog_core_sse", "final_analog_near_sse", "final_analog_outer_sse",
    "rolling_fold_win_count", "median_selected_alpha",
]
METHOD_DEF_COLUMNS = [
    "method", "model_family", "estimator", "target_level", "feature_set", "profile_method",
    "uses_calendar", "uses_vessel_count", "uses_regularization", "requires_alpha_selection", "deployable",
]

CONSISTENCY_TOL = 1e-10
RUN_CMD = "python " + " ".join(sys.argv)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_ais(train_path: Path) -> pd.DataFrame:
    usecols = ["mmsi", "x", "y", "sog", "time"]
    dtypes = {"mmsi": "string", "x": "float64", "y": "float64", "sog": "float32"}
    df = pd.read_csv(train_path, usecols=usecols, dtype=dtypes, parse_dates=["time"])
    df["hour"] = df["time"].dt.floor("h")
    df["date"] = df["time"].dt.normalize()
    df = add_regions(df)
    return df


def build_a_frame(df, daily, audit):
    a = make_a_labels(df)
    a["date"] = a["hour"].dt.normalize()
    a["hour_of_day"] = a["hour"].dt.hour
    a["dayofweek"] = a["hour"].dt.dayofweek
    a = a[["hour", "date", "hour_of_day", "dayofweek", "region", "y"]]
    a = a.merge(daily[["date", "unique_vessel_count"]].rename(columns={"unique_vessel_count": "vessel_count"}), on="date", how="left")
    a = a.merge(audit[["date", "quality_regime"]].rename(columns={"quality_regime": "audit_quality_regime"}), on="date", how="left")
    a["day_type"] = np.where(a["dayofweek"].isin([5, 6]), "weekend", "weekday")
    a["region_rank"] = a["region"].map(REGION_RANK)
    a["time_index"] = (a["date"] - GLOBAL_START).dt.days
    return a


def to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    g = hourly.groupby(["date", "region"], observed=True)["y"].sum().rename("T").reset_index()
    meta = hourly[["date", "time_index", "vessel_count", "day_type"]].drop_duplicates("date")
    g = g.merge(meta, on="date", how="left")
    g["region_rank"] = g["region"].map(REGION_RANK)
    return g.sort_values(["date", "region_rank"]).drop(columns=["region_rank"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Standardization + design matrices
# ---------------------------------------------------------------------------
def wmean_std(vals: np.ndarray, w: np.ndarray):
    ws = w.sum()
    if ws <= 0:
        return 0.0, 0.0
    mu = float((w * vals).sum() / ws)
    var = float((w * (vals - mu) ** 2).sum() / ws)
    return mu, np.sqrt(var)


def std_params(train_dates, weights, time_index_map, vc_map):
    tv = np.array([time_index_map[d] for d in train_dates], dtype=float)
    w = np.array([weights[d] for d in train_dates], dtype=float)
    tm, ts = wmean_std(tv, w)
    vv = np.array([np.log(vc_map[d]) for d in train_dates], dtype=float)
    vm, vs = wmean_std(vv, w)
    return tm, ts, vm, vs


def design_direct(df, tm, ts, vm, vs, use_vessel):
    reg = df["region"].to_numpy()
    hour = df["hour_of_day"].to_numpy()
    rh = np.array([f"{reg[i]}_{int(hour[i]):02d}" for i in range(len(df))])
    ti = df["time_index"].to_numpy(dtype=float)
    tz = (ti - tm) / ts if ts > 0 else np.zeros(len(df))
    wk = (df["day_type"] == "weekend").astype(float).to_numpy()
    cols = [(rh == c).astype(float) for c in ALL_RH[1:]]
    for r in REGION_ORDER:
        isr = (reg == r).astype(float)
        cols.append(isr * tz)
        cols.append(isr * wk)
    if use_vessel:
        vl = np.log(df["vessel_count"].astype(float).to_numpy())
        vz = (vl - vm) / vs if vs > 0 else np.zeros(len(df))
        for r in REGION_ORDER:
            cols.append((reg == r).astype(float) * vz)
    return np.column_stack(cols)


def design_daily(df, tm, ts, vm, vs, use_vessel):
    reg = df["region"].to_numpy()
    ti = df["time_index"].to_numpy(dtype=float)
    tz = (ti - tm) / ts if ts > 0 else np.zeros(len(df))
    wk = (df["day_type"] == "weekend").astype(float).to_numpy()
    cols = [(reg == "near").astype(float), (reg == "outer").astype(float)]
    for r in REGION_ORDER:
        isr = (reg == r).astype(float)
        cols.append(isr * tz)
        cols.append(isr * wk)
    if use_vessel:
        vl = np.log(df["vessel_count"].astype(float).to_numpy())
        vz = (vl - vm) / vs if vs > 0 else np.zeros(len(df))
        for r in REGION_ORDER:
            cols.append((reg == r).astype(float) * vz)
    return np.column_stack(cols)


def _fit(estimator, alpha, X, y, sw):
    if estimator == "ridge":
        m = Ridge(alpha=alpha, fit_intercept=True)
    else:
        m = PoissonRegressor(alpha=alpha, fit_intercept=True, max_iter=10000, tol=1e-9)
    m.fit(X, y, sample_weight=sw)
    return m


def _clip_pred(pred):
    pred = np.asarray(pred, dtype=float)
    pred = np.where(np.isfinite(pred), pred, 0.0)
    return np.clip(pred, 0.0, None)


def combine_profile(valid_hourly, valid_daily, T_pred, P_hat, profile_method):
    tmap = {}
    vd_date = valid_daily["date"].to_numpy()
    vd_reg = valid_daily["region"].to_numpy()
    for i in range(len(valid_daily)):
        tmap[(pd.Timestamp(vd_date[i]), vd_reg[i])] = float(T_pred[i])
    Pm = P_hat[profile_method]
    n = len(valid_hourly)
    pred = np.zeros(n); ptot = np.zeros(n); psh = np.zeros(n)
    vdate = valid_hourly["date"].to_numpy(); vreg = valid_hourly["region"].to_numpy()
    vhour = valid_hourly["hour_of_day"].to_numpy(); vdt = valid_hourly["day_type"].to_numpy()
    for i in range(n):
        d = pd.Timestamp(vdate[i]); r = vreg[i]; h = int(vhour[i])
        T = tmap[(d, r)]
        p = Pm[r][vdt[i]][h] if profile_method == "daytype_shrunk" else Pm[r][h]
        ptot[i] = T; psh[i] = p; pred[i] = T * p
    return pred, ptot, psh


def fit_predict_stat(meta, train_hourly, valid_hourly, valid_daily, weights, params, alpha, P_hat, train_daily=None):
    est = meta[2]; use_vessel = meta[7]; target = meta[3]; profile = meta[5]
    tm, ts, vm, vs = params
    if target == "hourly":
        Xtr = design_direct(train_hourly, tm, ts, vm, vs, use_vessel)
        ytr = train_hourly["y"].to_numpy(dtype=float)
        sw = train_hourly["date"].map(weights).to_numpy(dtype=float)
        model = _fit(est, alpha, Xtr, ytr, sw)
        Xva = design_direct(valid_hourly, tm, ts, vm, vs, use_vessel)
        return _clip_pred(model.predict(Xva)), None, None
    td = train_daily if train_daily is not None else to_daily(train_hourly)
    Xtr = design_daily(td, tm, ts, vm, vs, use_vessel)
    Ttr = td["T"].to_numpy(dtype=float)
    sw = td["date"].map(weights).to_numpy(dtype=float)
    model = _fit(est, alpha, Xtr, Ttr, sw)
    Xva = design_daily(valid_daily, tm, ts, vm, vs, use_vessel)
    T_pred = _clip_pred(model.predict(Xva))
    return combine_profile(valid_hourly, valid_daily, T_pred, P_hat, profile)


# ---------------------------------------------------------------------------
# Inner CV
# ---------------------------------------------------------------------------
def build_inner_splits(a_frame, outer_train_dates, policy, china_series, vc_series, time_index_map):
    splits = []
    inner_val_dates = outer_train_dates[-4:]
    for v in inner_val_dates:
        itrain_dates = [d for d in outer_train_dates if d < v]
        if len(itrain_dates) < 6:
            raise AssertionError(f"inner train <6 dates for val {v}")
        iclasses, _ = classify_fold_quality(itrain_dates, china_series)
        iweights = {d: POLICY_WEIGHTS[policy][iclasses[d]] for d in itrain_dates}
        itrain_hourly = a_frame[a_frame["date"].isin(itrain_dates)].copy()
        ival_hourly = a_frame[a_frame["date"] == v].copy()
        iparams = std_params(itrain_dates, iweights, time_index_map, vc_series)
        itrain_daily = to_daily(itrain_hourly)
        ival_daily = to_daily(ival_hourly)
        _, P_hat_i = compute_components(itrain_hourly, itrain_dates, itrain_dates[-1], iweights)
        splits.append({
            "v": v, "itrain_hourly": itrain_hourly, "ival_hourly": ival_hourly,
            "itrain_daily": itrain_daily, "ival_daily": ival_daily, "iweights": iweights,
            "iparams": iparams, "P_hat_i": P_hat_i,
        })
    return splits, inner_val_dates


def inner_select(meta, splits, inner_val_dates):
    grid = RIDGE_ALPHAS if meta[2] == "ridge" else POISSON_ALPHAS
    agg = {a: {"sse": 0.0, "mae": 0.0, "n": 0, "fail": 0} for a in grid}
    for sp in splits:
        for a in grid:
            try:
                pred, _, _ = fit_predict_stat(meta, sp["itrain_hourly"], sp["ival_hourly"], sp["ival_daily"],
                                              sp["iweights"], sp["iparams"], a, sp["P_hat_i"], train_daily=sp["itrain_daily"])
                er = pred - sp["ival_hourly"]["y"].to_numpy(dtype=float)
                agg[a]["sse"] += float((er ** 2).sum())
                agg[a]["mae"] += float(np.abs(er).sum())
                agg[a]["n"] += len(er)
            except Exception:
                agg[a]["fail"] += 1
    records = []
    eligible = []
    for a in grid:
        ok = agg[a]["fail"] == 0
        sse = agg[a]["sse"] if ok else None
        mae = (agg[a]["mae"] / agg[a]["n"]) if ok and agg[a]["n"] > 0 else None
        records.append((a, sse, mae, agg[a]["fail"]))
        if ok:
            eligible.append((a, sse))
    if not eligible:
        raise AssertionError(f"all alphas failed for {meta[0]}")
    min_sse = min(s for _, s in eligible)
    near = [a for a, s in eligible if s <= min_sse + 1e-9]
    selected = max(near)
    return selected, records, inner_val_dates


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def metrics_block(y_true, pred_float, pred_rounded, region_arr):
    er = pred_rounded - y_true
    ef = pred_float - y_true
    out = {
        "sse_float": float((ef ** 2).sum()), "sse_rounded": float((er ** 2).sum()),
        "mse_rounded": float((er ** 2).mean()), "mae_rounded": float(np.abs(er).mean()),
        "mean_bias_rounded": float(er.mean()),
        "actual_total": float(y_true.sum()), "predicted_total_rounded": float(pred_rounded.sum()),
    }
    for r in REGION_ORDER:
        m = region_arr == r
        out[f"{r}_sse_rounded"] = float((er[m] ** 2).sum())
    return out


def build_combo(eid, etype, policy, mname, selected_alpha, valid, pred, pred_tot=None, pred_share=None):
    y_true = valid["y"].to_numpy(dtype="float64")
    pred_rounded = np.clip(np.rint(pred), 0, None).astype(np.int64)
    n = len(valid)
    combo = pd.DataFrame({
        "evaluation_id": eid, "evaluation_type": etype, "training_policy": policy, "method": mname,
        "selected_alpha": selected_alpha if selected_alpha is not None else np.full(n, np.nan),
        "date": valid["date"].to_numpy(), "hour": valid["hour"].to_numpy(),
        "hour_of_day": valid["hour_of_day"].to_numpy(), "dayofweek": valid["dayofweek"].to_numpy(),
        "day_type": valid["day_type"].to_numpy(), "region": valid["region"].to_numpy(),
        "audit_quality_regime": valid["audit_quality_regime"].to_numpy(),
        "validation_unique_vessel_count": valid["vessel_count"].to_numpy(),
        "y_true": y_true, "true_daily_total": valid["true_daily_total"].to_numpy(),
        "true_profile_share": valid["true_profile_share"].to_numpy(),
        "pred_daily_total": pred_tot if pred_tot is not None else np.full(n, np.nan),
        "pred_profile_share": pred_share if pred_share is not None else np.full(n, np.nan),
        "pred_float": pred, "pred_rounded": pred_rounded, "region_rank": valid["region_rank"].to_numpy(),
    })
    combo["error_float"] = combo["pred_float"] - combo["y_true"]
    combo["error_rounded"] = combo["pred_rounded"] - combo["y_true"]
    return combo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    out_dir = ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = ROOT / TRAIN_REL
    if not train_path.exists():
        raise FileNotFoundError(f"Training AIS file not found: {train_path}")
    print(f"train_path = {train_path}")
    print(f"out_dir    = {out_dir}")

    df = load_ais(train_path)
    daily = pd.read_csv(ROOT / STEP01_DAILY); daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    audit = pd.read_csv(ROOT / STEP02_DAILY); audit["date"] = pd.to_datetime(audit["date"]).dt.normalize()
    china_series = dict(zip(audit["date"], audit["china_coastal_record_count"]))
    vc_series = dict(zip(daily["date"], daily["unique_vessel_count"]))
    time_index_map = {d: int((d - GLOBAL_START).days) for d in pd.date_range("2018-01-01", "2018-01-24", freq="D").normalize()}

    a_frame = build_a_frame(df, daily, audit)
    assert len(a_frame) == 1728

    scenarios = []
    for f in build_folds():
        scenarios.append((f["fold_id"], "rolling_7d", f["train_start"], f["train_end"], f["valid_start"], f["valid_end"]))
    scenarios.append(("final_analog", "final_analog", pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-18"),
                      pd.Timestamp("2018-01-19"), pd.Timestamp("2018-01-24")))

    pred_frames = []
    fold_score_rows = []
    hp_rows = []

    for eid, etype, tstart, tend, vstart, vend in scenarios:
        train_hourly = a_frame[(a_frame["date"] >= tstart) & (a_frame["date"] <= tend)].copy()
        valid = a_frame[(a_frame["date"] >= vstart) & (a_frame["date"] <= vend)].copy()
        vtot = valid.groupby(["date", "region"], observed=True)["y"].sum().rename("true_daily_total").reset_index()
        valid = valid.merge(vtot, on=["date", "region"], how="left")
        valid["true_profile_share"] = np.where(valid["true_daily_total"] > 0, valid["y"] / valid["true_daily_total"], 0.0)
        valid = valid.sort_values(["hour", "region_rank"]).reset_index(drop=True)
        valid_daily = to_daily(valid)

        outer_train_dates = sorted(pd.date_range(tstart, tend, freq="D").normalize())

        for policy in POLICIES:
            classes, _ = classify_fold_quality(outer_train_dates, china_series)
            weights = {d: POLICY_WEIGHTS[policy][classes[d]] for d in outer_train_dates}
            train_w = train_hourly.copy(); train_w["weight"] = train_w["date"].map(weights).astype(float)
            smean = scale_train_mean(outer_train_dates, vc_series, weights, policy)
            T_hat_comp, P_hat_comp = compute_components(train_hourly, outer_train_dates, tend, weights)
            oparams = std_params(outer_train_dates, weights, time_index_map, vc_series)

            # precompute valid profile vectors and baseline daily-total vectors
            vr = valid["region"].to_numpy(); vh = valid["hour_of_day"].to_numpy(); vdt = valid["day_type"].to_numpy()
            vdr = valid_daily["region"].to_numpy()
            Tvec_mean = np.array([T_hat_comp["mean"][r] for r in vdr], dtype=float)
            Tvec_ewm7 = np.array([T_hat_comp["ewm7"][r] for r in vdr], dtype=float)

            # inner splits (shared by all stat methods)
            splits, inner_val_dates = build_inner_splits(a_frame, outer_train_dates, policy, china_series, vc_series, time_index_map)
            inner_val_str = ",".join(f"{d:%Y-%m-%d}" for d in inner_val_dates)

            for meta in METHODS:
                mname, family, est, target, feat, profile, uses_cal, uses_ves, requires_alpha = meta
                selected_alpha = None
                if mname == "hour_mean":
                    pred = np.asarray(predict_weighted(train_w, valid, MethodSpec("hour_mean", "hour"), smean), dtype=float)
                    pred_tot = pred_share = None
                elif family == "baseline":  # decomposition baseline
                    dm = "ewm7" if est == "ewm" else "mean"
                    Tvec = Tvec_ewm7 if dm == "ewm7" else Tvec_mean
                    pred, pred_tot, pred_share = combine_profile(valid, valid_daily, Tvec, P_hat_comp, profile)
                else:  # stat model
                    selected_alpha, records, _ = inner_select(meta, splits, inner_val_dates)
                    for a, sse, mae, fail in records:
                        hp_rows.append({
                            "evaluation_id": eid, "evaluation_type": etype, "training_policy": policy,
                            "method": mname, "alpha": a, "n_inner_splits": 4,
                            "inner_validation_dates": inner_val_str,
                            "inner_sse_float": sse if sse is not None else np.nan,
                            "inner_mae_float": mae if mae is not None else np.nan,
                            "fit_failure_count": fail, "selected": (a == selected_alpha),
                        })
                    pred, pred_tot, pred_share = fit_predict_stat(meta, train_hourly, valid, valid_daily, weights, oparams, selected_alpha, P_hat_comp)

                assert (pred >= 0).all(), f"negative pred {eid}/{policy}/{mname}"
                assert np.isfinite(pred).all()
                combo = build_combo(eid, etype, policy, mname, selected_alpha, valid, pred, pred_tot, pred_share)
                pred_frames.append(combo)
                region_arr = valid["region"].to_numpy()
                mb = metrics_block(valid["y"].to_numpy("float64"), np.asarray(combo["pred_float"]), combo["pred_rounded"].to_numpy(), region_arr)
                fold_score_rows.append({
                    "evaluation_id": eid, "evaluation_type": etype, "training_policy": policy, "method": mname,
                    "selected_alpha": selected_alpha if selected_alpha is not None else np.nan,
                    "train_start": f"{tstart:%Y-%m-%d}", "train_end": f"{tend:%Y-%m-%d}",
                    "valid_start": f"{vstart:%Y-%m-%d}", "valid_end": f"{vend:%Y-%m-%d}",
                    "n_valid_days": int(valid["date"].nunique()), "n_rows": int(len(valid)), **mb,
                })

    preds = pd.concat(pred_frames, ignore_index=True)
    preds["_e"] = preds["evaluation_id"].map({s[0]: i for i, s in enumerate(scenarios)})
    preds["_p"] = preds["training_policy"].map(POLICY_ORDER)
    preds["_m"] = preds["method"].map(METHOD_ORDER)
    preds = preds.sort_values(["_e", "_p", "_m", "date", "hour_of_day", "region_rank"]).reset_index(drop=True)
    assert len(preds) == 214272, f"expected 214272 predictions, got {len(preds)}"

    # decomposition profile-sum assertion
    decomp = preds[preds["pred_profile_share"].notna()]
    sums = decomp.groupby(["evaluation_id", "training_policy", "method", "date", "region"])["pred_profile_share"].sum()
    assert (sums - 1.0).abs().max() < 1e-10, "decomposition profile share != 1"

    # exactly one selected alpha per stat model x eval x policy
    hp = pd.DataFrame(hp_rows)
    sel_check = hp.groupby(["evaluation_id", "training_policy", "method"])["selected"].sum()
    assert (sel_check == 1).all(), "each stat model must select exactly one alpha"

    # consistency vs step06 (hour_mean, decomp_mean_mean, decomp_ewm7_daytype_shrunk)
    s6 = pd.read_csv(ROOT / STEP06_PREDS, encoding="utf-8-sig")
    s6 = s6[s6["method"].isin(["hour_mean", "decomp_mean_mean", "decomp_ewm7_daytype_shrunk"])].copy()
    s6["date"] = pd.to_datetime(s6["date"]).dt.strftime("%Y-%m-%d")
    s6_map = {(r["evaluation_id"], r["training_policy"], r["method"], r["date"], int(r["hour_of_day"]), r["region"]): r["pred_float"] for _, r in s6.iterrows()}
    mine = preds[preds["method"].isin(["hour_mean", "decomp_mean_mean", "decomp_ewm7_daytype_shrunk"])].copy()
    mine["date"] = pd.to_datetime(mine["date"]).dt.strftime("%Y-%m-%d")
    max_diff = 0.0
    for _, r in mine.iterrows():
        ref = s6_map.get((r["evaluation_id"], r["training_policy"], r["method"], r["date"], int(r["hour_of_day"]), r["region"]))
        if ref is None:
            raise AssertionError(f"no step06 reference for {r['evaluation_id']}/{r['method']}")
        max_diff = max(max_diff, abs(r["pred_float"] - ref))
    assert max_diff < CONSISTENCY_TOL, f"step06 consistency max diff {max_diff:.3e}"
    print(f"consistency vs step06 (3 baselines): max abs diff = {max_diff:.3e}")

    fold_scores = pd.DataFrame(fold_score_rows)

    # --- normal-target scores ----------------------------------------------
    norm = preds[(preds["evaluation_type"] == "rolling_7d") & (preds["audit_quality_regime"] == "normal")]
    norm_rows = []
    for policy in POLICIES:
        for meta in METHODS:
            mname = meta[0]
            sub = norm[(norm["training_policy"] == policy) & (norm["method"] == mname)]
            er = sub["error_rounded"].astype("float64").to_numpy()
            norm_rows.append({
                "training_policy": policy, "method": mname,
                "n_prediction_rows": int(len(sub)), "n_unique_target_dates": int(sub["date"].nunique()),
                "sse_rounded": float((er ** 2).sum()), "mse_rounded": float((er ** 2).mean()),
                "mae_rounded": float(np.abs(er).mean()), "mean_bias_rounded": float(er.mean()),
            })
    normal_scores = pd.DataFrame(norm_rows)
    normal_scores["rank_by_mse_rounded"] = normal_scores["mse_rounded"].rank(method="min", ascending=True).astype(int)

    # --- final_analog scores -----------------------------------------------
    fa = preds[preds["evaluation_id"] == "final_analog"]
    fa_rows = []
    for policy in POLICIES:
        for meta in METHODS:
            mname = meta[0]
            sub = fa[(fa["training_policy"] == policy) & (fa["method"] == mname)]
            er = sub["error_rounded"].astype("float64").to_numpy()
            ef = sub["error_float"].astype("float64").to_numpy()
            sa = sub["selected_alpha"].dropna().iloc[0] if sub["selected_alpha"].notna().any() else np.nan
            row = {
                "training_policy": policy, "method": mname, "selected_alpha": sa,
                "sse_float": float((ef ** 2).sum()), "sse_rounded": float((er ** 2).sum()),
                "mse_rounded": float((er ** 2).mean()), "mae_rounded": float(np.abs(er).mean()),
                "mean_bias_rounded": float(er.mean()),
                "actual_total": float(sub["y_true"].sum()), "predicted_total_rounded": float(sub["pred_rounded"].sum()),
            }
            for r in REGION_ORDER:
                m = sub["region"] == r
                row[f"{r}_sse_rounded"] = float((sub.loc[m, "error_rounded"].astype("float64") ** 2).sum())
            fa_rows.append(row)
    final_scores = pd.DataFrame(fa_rows)
    final_scores["rank_by_sse_rounded"] = final_scores["sse_rounded"].rank(method="min", ascending=True).astype(int)

    # --- decision table -----------------------------------------------------
    rolling = fold_scores[fold_scores["evaluation_type"] == "rolling_7d"]
    win_pivot = rolling.pivot_table(index="evaluation_id", columns=["training_policy", "method"], values="sse_rounded", aggfunc="first")
    min_per_fold = win_pivot.min(axis=1)
    win_counts = win_pivot.eq(min_per_fold, axis=0).sum(axis=0)
    dec_rows = []
    for policy in POLICIES:
        for meta in METHODS:
            mname, family, est, target, feat, profile, uses_cal, uses_ves, req_a = meta
            rsub = rolling[(rolling["training_policy"] == policy) & (rolling["method"] == mname)]
            sse = rsub["sse_rounded"].to_numpy("float64")
            sa = rsub["selected_alpha"].dropna().to_numpy("float64")
            ns = normal_scores[(normal_scores["training_policy"] == policy) & (normal_scores["method"] == mname)].iloc[0]
            fs = final_scores[(final_scores["training_policy"] == policy) & (final_scores["method"] == mname)].iloc[0]
            dec_rows.append({
                "training_policy": policy, "method": mname, "model_family": family, "estimator": est,
                "uses_vessel_count": uses_ves, "profile_method": profile,
                "rolling_mean_sse": float(np.mean(sse)), "rolling_median_sse": float(np.median(sse)), "rolling_max_sse": float(np.max(sse)),
                "rolling_normal_mse": float(ns["mse_rounded"]), "rolling_normal_rank": int(ns["rank_by_mse_rounded"]),
                "final_analog_sse": float(fs["sse_rounded"]), "final_analog_rank": int(fs["rank_by_sse_rounded"]),
                "final_analog_mean_bias": float(fs["mean_bias_rounded"]),
                "final_analog_core_sse": float(fs["core_sse_rounded"]), "final_analog_near_sse": float(fs["near_sse_rounded"]),
                "final_analog_outer_sse": float(fs["outer_sse_rounded"]),
                "rolling_fold_win_count": int(win_counts.get((policy, mname), 0)),
                "median_selected_alpha": float(np.median(sa)) if len(sa) else np.nan,
            })
    decision = pd.DataFrame(dec_rows).sort_values(["training_policy", "method"]).reset_index(drop=True)

    # --- method definitions -------------------------------------------------
    md = pd.DataFrame([{
        "method": m[0], "model_family": m[1], "estimator": m[2], "target_level": m[3], "feature_set": m[4],
        "profile_method": m[5], "uses_calendar": m[6], "uses_vessel_count": m[7],
        "uses_regularization": (m[2] in ("ridge", "poisson")), "requires_alpha_selection": m[8], "deployable": True,
    } for m in METHODS])

    # --- write CSVs (UTF-8 BOM) --------------------------------------------
    p_md = out_dir / "method_definitions.csv"
    p_hp = out_dir / "hyperparameter_selection.csv"
    p_pred = out_dir / "a_model_predictions.csv"
    p_fs = out_dir / "a_fold_scores.csv"
    p_ns = out_dir / "a_normal_target_scores.csv"
    p_fa = out_dir / "a_final_analog_scores.csv"
    p_dec = out_dir / "decision_table.csv"
    p_sum = out_dir / "summary.md"

    md.to_csv(p_md, index=False, encoding="utf-8-sig")
    hp["_e"] = hp["evaluation_id"].map({s[0]: i for i, s in enumerate(scenarios)})
    hp["_p"] = hp["training_policy"].map(POLICY_ORDER)
    hp["_m"] = hp["method"].map(METHOD_ORDER)
    hp = hp.sort_values(["_e", "_p", "_m", "alpha"]).drop(columns=["_e", "_p", "_m"])
    hp[HP_COLUMNS].to_csv(p_hp, index=False, encoding="utf-8-sig")

    pred_out = preds[PRED_COLUMNS].copy()
    pred_out["hour"] = pd.to_datetime(pred_out["hour"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    pred_out["date"] = pd.to_datetime(pred_out["date"]).dt.strftime("%Y-%m-%d")
    pred_out.to_csv(p_pred, index=False, encoding="utf-8-sig")

    fold_scores.sort_values(["evaluation_id", "training_policy", "method"]).to_csv(p_fs, index=False, encoding="utf-8-sig")
    normal_scores[NORMAL_COLUMNS].sort_values(["rank_by_mse_rounded", "training_policy", "method"]).to_csv(p_ns, index=False, encoding="utf-8-sig")
    final_scores[FINAL_COLUMNS].sort_values(["rank_by_sse_rounded", "training_policy", "method"]).to_csv(p_fa, index=False, encoding="utf-8-sig")
    decision[DECISION_COLUMNS].to_csv(p_dec, index=False, encoding="utf-8-sig")

    write_summary(p_sum, decision, normal_scores, final_scores, hp)

    # --- console ------------------------------------------------------------
    print("\n=== Output files ===")
    for path in [p_md, p_hp, p_pred, p_fs, p_ns, p_fa, p_dec, p_sum]:
        if path.suffix == ".csv":
            rows = len(pd.read_csv(path, encoding="utf-8-sig"))
        else:
            rows = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {path.relative_to(ROOT)}  rows={rows}")

    print("\n=== final_analog top 10 by sse_rounded ===")
    for _, r in final_scores.sort_values("sse_rounded").head(10).iterrows():
        sa = "" if pd.isna(r["selected_alpha"]) else f" a={r['selected_alpha']}"
        print(f"  rank={int(r['rank_by_sse_rounded']):2d}  {r['training_policy']:22s} {r['method']:34s} sse={r['sse_rounded']:.0f}{sa}")
    print("\n=== rolling normal-target top 10 by mse_rounded ===")
    for _, r in normal_scores.sort_values("mse_rounded").head(10).iterrows():
        print(f"  rank={int(r['rank_by_mse_rounded']):2d}  {r['training_policy']:22s} {r['method']:34s} mse={r['mse_rounded']:.3f}")
    print("\nDone.")


def write_summary(path, decision, normal_scores, final_scores, hp):
    L = []
    L.append("# Step 07 A Regularized Count Models")
    L.append("")
    L.append("> Task A only. Ridge/Poisson direct and decomposition models; no neg-binomial, no trees/NN, no B, no submissions.")
    L.append("")
    L.append(f"Run command: `{RUN_CMD}`")
    L.append("")

    L.append("## 1. Evaluation and leakage controls")
    L.append("")
    L.append("- 8 seven-day rolling folds + 1 final_analog fold; 3 training-quality policies.")
    L.append("- Inner time-series CV selects alpha: the last 4 dates of each outer training period are inner-validation days, each predicted only from strictly-earlier dates; quality classes are recomputed inside each inner training span. Outer validation is never used to choose alpha.")
    L.append("- Validation prediction uses only the competition-allowed `unique_vessel_count` (for vessel-feature models) and calendar variables; no validation AIS / quality / label is used.")
    L.append("")

    L.append("## 2. Consistency checks")
    L.append("")
    L.append("- hour_mean, decomp_mean_mean, decomp_ewm7_daytype_shrunk reproduce the Step 06 stored predictions with max abs diff < 1e-10. **PASS**")
    L.append("- a_model_predictions.csv has 214272 rows. **PASS**")
    L.append("- Every statistical model selects exactly one alpha per (evaluation_id, training_policy). **PASS**")
    L.append("- All decomposition 24h profile shares sum to 1 within 1e-10; all predictions are finite and non-negative. **PASS**")
    L.append("")

    L.append("## 3. Hyperparameter stability")
    L.append("")
    sel = hp[hp["selected"]]
    for est in ["ridge", "poisson"]:
        methods = [m[0] for m in METHODS if m[2] == est]
        sub = sel[sel["method"].isin(methods)]
        counts = sub["alpha"].value_counts().to_dict()
        L.append(f"- {est}: selected-alpha counts across all folds/policies/methods = {counts}.")
    boundary_min = sel[(sel["method"].isin([m[0] for m in METHODS if m[2] == "ridge"])) & (sel["alpha"] == RIDGE_ALPHAS[0])].shape[0]
    boundary_max = sel[(sel["method"].isin([m[0] for m in METHODS if m[2] == "ridge"])) & (sel["alpha"] == RIDGE_ALPHAS[-1])].shape[0]
    L.append(f"- Ridge boundary selections: alpha={RIDGE_ALPHAS[0]} chosen {boundary_min} times, alpha={RIDGE_ALPHAS[-1]} chosen {boundary_max} times. (Reported as fact; grid not expanded here.)")
    L.append("")

    L.append("## 4. Statistical models versus baselines")
    L.append("")
    fam = decision.groupby("model_family").agg(
        roll_med=("rolling_median_sse", "median"), norm=("rolling_normal_mse", "median"),
        fa=("final_analog_sse", "median"), roll_max=("rolling_max_sse", "median"), wins=("rolling_fold_win_count", "sum"))
    L.append("Trustworthy views first (normal-target mse and final_analog sse); the rolling-median / rolling-wins columns are kept only for reference and are contaminated by anomalous validation days.")
    L.append("")
    L.append("| family | median(rolling normal mse) | median(final_analog sse) | median(rolling max sse) | median(rolling median sse, contaminated) | total rolling wins (contaminated) |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for fn, row in fam.iterrows():
        L.append(f"| {fn} | {row['norm']:.3f} | {row['fa']:.0f} | {row['roll_max']:.0f} | {row['roll_med']:.0f} | {int(row['wins'])} |")
    L.append("")
    L.append("- On the normal-target and final_analog views the simple baselines clearly beat every regularized model (baseline normal mse ~8.8 vs ~15 for regularized; baseline final_analog ~3.9k vs ~7.7k). The regularized models only look better on the rolling-median / rolling-wins columns, which include anomalous validation days and are not trustworthy for model choice.")
    L.append("")

    L.append("## 5. Ridge versus Poisson")
    L.append("")
    est_g = decision.groupby("estimator").agg(roll_med=("rolling_median_sse", "median"), norm=("rolling_normal_mse", "median"), fa=("final_analog_sse", "median"))
    L.append("| estimator | median(rolling median sse) | median(rolling normal mse) | median(final_analog sse) |")
    L.append("| --- | --- | --- | --- |")
    for e, row in est_g.iterrows():
        if e in ("ridge", "poisson"):
            L.append(f"| {e} | {row['roll_med']:.0f} | {row['norm']:.3f} | {row['fa']:.0f} |")
    L.append("")

    L.append("## 6. Effect of vessel-count feature")
    L.append("")
    L.append("Paired non-vessel vs vessel (median across policies), rolling normal mse / final_analog sse:")
    L.append("")
    L.append("| method pair | normal mse (no vessel / vessel) | final_analog sse (no vessel / vessel) |")
    L.append("| --- | --- | --- |")
    pairs = [("ridge_direct_calendar", "ridge_direct_calendar_vessel"),
             ("poisson_direct_calendar", "poisson_direct_calendar_vessel"),
             ("decomp_ridge_calendar_mean", "decomp_ridge_calendar_vessel_mean"),
             ("decomp_poisson_calendar_mean", "decomp_poisson_calendar_vessel_mean")]
    for a, b in pairs:
        va = normal_scores[normal_scores["method"] == a]["mse_rounded"].median()
        vb = normal_scores[normal_scores["method"] == b]["mse_rounded"].median()
        fa_ = final_scores[final_scores["method"] == a]["sse_rounded"].median()
        fb = final_scores[final_scores["method"] == b]["sse_rounded"].median()
        L.append(f"| {a} / {b} | {va:.3f} / {vb:.3f} | {fa_:.0f} / {fb:.0f} |")
    L.append("")

    L.append("## 7. Direct versus decomposition")
    L.append("")
    struct = decision[decision["estimator"].isin(["ridge", "poisson"])].groupby("model_family").agg(
        roll_med=("rolling_median_sse", "median"), norm=("rolling_normal_mse", "median"), fa=("final_analog_sse", "median"), roll_max=("rolling_max_sse", "median"))
    L.append("Among the regularized models only (both lose to the baselines). Lower is better; normal-target mse and final_analog sse are the trustworthy columns.")
    L.append("")
    L.append("| structure (regularized only) | median(rolling normal mse) | median(final_analog sse) | median(rolling max sse) | median(rolling median sse, contaminated) |")
    L.append("| --- | --- | --- | --- | --- |")
    for fn, row in struct.iterrows():
        L.append(f"| {fn} | {row['norm']:.3f} | {row['fa']:.0f} | {row['roll_max']:.0f} | {row['roll_med']:.0f} |")
    L.append("")
    L.append("- Decomposition is marginally more reliable than direct among the regularized models, but the difference is small and both are dominated by the simple baselines.")
    L.append("")

    L.append("## 8. Mean profile versus daytype-shrunk profile")
    L.append("")
    L.append("Paired mean vs daytype (median across policies), rolling normal mse / final_analog sse:")
    L.append("")
    L.append("| daily-total method | normal mse (mean / daytype) | final_analog sse (mean / daytype) |")
    L.append("| --- | --- | --- |")
    prof_pairs = [("decomp_mean_mean", "decomp_mean_daytype_shrunk"),
                  ("decomp_ridge_calendar_mean", "decomp_ridge_calendar_daytype"),
                  ("decomp_poisson_calendar_mean", "decomp_poisson_calendar_daytype")]
    for a, b in prof_pairs:
        va = normal_scores[normal_scores["method"] == a]["mse_rounded"].median()
        vb = normal_scores[normal_scores["method"] == b]["mse_rounded"].median()
        fa_ = final_scores[final_scores["method"] == a]["sse_rounded"].median()
        fb = final_scores[final_scores["method"] == b]["sse_rounded"].median()
        L.append(f"| {a} / {b} | {va:.3f} / {vb:.3f} | {fa_:.0f} / {fb:.0f} |")
    L.append("")

    L.append("## 9. Implications for negative binomial modeling")
    L.append("")
    L.append("- Poisson is NOT systematically worse than Ridge here (Poisson is in fact better: normal mse 13.8 vs 16.4, final_analog 6.7k vs 8.2k). So there is no signal that the Poisson mean-variance equality is the binding problem.")
    L.append("- Selected Poisson alphas do not pile up on the weak-regularization boundary (alpha=1 chosen most often), and no systematic explosion/bias is visible. With only ~2-3 weeks of data the regularized count models lose to the simple weighted-mean baselines regardless of estimator.")
    L.append("- Therefore the evidence does NOT justify investing in a negative-binomial model for A at this stage. No negative-binomial model is fit here.")
    L.append("")

    L.append("## 10. Decision for the next stage")
    L.append("")
    best_fa = final_scores.sort_values("sse_rounded").iloc[0]
    best_norm = normal_scores.sort_values("mse_rounded").iloc[0]
    L.append(f"- Best final_analog combo: {best_fa['training_policy']} / {best_fa['method']} (sse={best_fa['sse_rounded']:.0f}).")
    L.append(f"- Best rolling normal-target combo: {best_norm['training_policy']} / {best_norm['method']} (mse={best_norm['mse_rounded']:.3f}).")
    L.append("- Keep as A candidates: the simple decomposition baselines (especially decomp_mean_daytype_shrunk, decomp_mean_mean, decomp_ewm7_daytype_shrunk) and hour_mean, under the quality_weighted / exclude_outage_severe policies (all as control).")
    L.append("- Drop / do not pursue further: the Ridge/Poisson regularized models (they are systematically worse on the trustworthy views), and the daily vessel-count feature (inconsistent, net-negative for Ridge).")
    L.append("- daytype-shrunk profile gives a small but consistent edge over mean profile across daily-total methods; keep it.")
    L.append("- Negative binomial is not worth testing for A (section 9).")
    L.append("- A single-model development for A has effectively plateaued at the simple decomposition baselines; the next higher-value step is the systematic diagnosis of Task B. No submission prediction and no ensembling in this stage.")
    L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
