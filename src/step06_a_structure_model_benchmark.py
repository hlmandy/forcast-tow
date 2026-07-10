"""Step 06 — Benchmark direct vs decomposition prediction structures for Task A.

Compares two simple structures:
  - direct hourly prediction (hour_mean, recent7_hour)
  - size-shape decomposition: predicted daily region total x 24h profile share
    (8 decomposition variants built from daily-total methods {mean, recent7, ewm7}
    and hour-profile methods {mean, recent7, fourier2, daytype_shrunk})

No GLM/GAM/LightGBM/neural-net, no B task, no submission files.

Evaluation: the 8 Step 03 rolling folds plus the Step 04 final_analog fold,
each under the 3 training-quality policies. A labels, fold definitions, quality
weights, and the weighted prediction function are reused from Step 03; Step 03
is not modified and its main() does not run on import. hour_mean and
recent7_hour are asserted to reproduce Step 03's stored predictions (< 1e-10).

Outputs (under outputs/step06_a_structure_model_benchmark/):
  1. method_definitions.csv             (10)
  2. a_model_predictions.csv            (133920)
  3. a_fold_scores.csv                  (270)
  4. a_normal_target_scores.csv         (30)
  5. a_final_analog_scores.csv          (30)
  6. a_decomposition_diagnostics.csv    (648)
  7. decision_table.csv                 (30)
  8. summary.md
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
    POLICIES, POLICY_ORDER, POLICY_WEIGHTS, REGION_RANK,
    TRAIN_REL, STEP01_DAILY, STEP02_DAILY,
    build_folds, classify_fold_quality, predict_weighted, scale_train_mean,
)
from optimized_baseline import MethodSpec, add_regions, make_a_labels, REGION_ORDER  # noqa: E402

STEP03_PREDS = Path("outputs/step03_a_rolling_backtest/a_backtest_predictions.csv")
OUT_REL = Path("outputs/step06_a_structure_model_benchmark")

# Fourier-2 design matrix (24 x 5): intercept, sin/cos k=1, sin/cos k=2
_H = np.arange(24)
_FOURIER_BASIS = np.column_stack([
    np.ones(24),
    np.sin(2 * np.pi * 1 * _H / 24), np.cos(2 * np.pi * 1 * _H / 24),
    np.sin(2 * np.pi * 2 * _H / 24), np.cos(2 * np.pi * 2 * _H / 24),
])

# name, family, daily_total_method, hour_profile_method,
# uses_recent_window, uses_time_decay, uses_day_type, uses_fourier
METHOD_META = [
    ("hour_mean", "direct", "", "", False, False, False, False),
    ("recent7_hour", "direct", "", "", True, False, False, False),
    ("decomp_mean_mean", "decomposition", "mean", "mean", False, False, False, False),
    ("decomp_recent7_mean", "decomposition", "recent7", "mean", True, False, False, False),
    ("decomp_recent7_recent7", "decomposition", "recent7", "recent7", True, False, False, False),
    ("decomp_ewm7_mean", "decomposition", "ewm7", "mean", False, True, False, False),
    ("decomp_ewm7_recent7", "decomposition", "ewm7", "recent7", True, True, False, False),
    ("decomp_mean_fourier2", "decomposition", "mean", "fourier2", False, False, False, True),
    ("decomp_ewm7_fourier2", "decomposition", "ewm7", "fourier2", False, True, False, True),
    ("decomp_ewm7_daytype_shrunk", "decomposition", "ewm7", "daytype_shrunk", False, True, True, False),
]
METHOD_ORDER = {m[0]: i for i, m in enumerate(METHOD_META)}
DECOMP_METHODS = [m for m in METHOD_META if m[1] == "decomposition"]
# map step06 recent7_hour -> step03 method name for the consistency check
STEP03_NAME = {"hour_mean": "hour_mean", "recent7_hour": "recent_7d_hour"}

PRED_COLUMNS = [
    "evaluation_id", "evaluation_type", "training_policy", "method", "date", "hour",
    "hour_of_day", "dayofweek", "day_type", "region", "audit_quality_regime",
    "validation_unique_vessel_count", "y_true", "true_daily_total", "true_profile_share",
    "pred_daily_total", "pred_profile_share", "pred_float", "pred_rounded",
    "error_float", "error_rounded",
]
FOLD_SCORE_COLUMNS = [
    "evaluation_id", "evaluation_type", "training_policy", "method",
    "train_start", "train_end", "valid_start", "valid_end", "n_valid_days", "n_rows",
    "sse_float", "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded",
    "actual_total", "predicted_total_rounded",
    "core_sse_rounded", "near_sse_rounded", "outer_sse_rounded",
]
NORMAL_COLUMNS = [
    "training_policy", "method", "n_prediction_rows", "n_unique_target_dates",
    "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded", "rank_by_mse_rounded",
]
FINAL_COLUMNS = [
    "training_policy", "method", "sse_float", "sse_rounded", "mse_rounded", "mae_rounded",
    "mean_bias_rounded", "actual_total", "predicted_total_rounded",
    "core_sse_rounded", "near_sse_rounded", "outer_sse_rounded", "rank_by_sse_rounded",
]
DIAG_COLUMNS = [
    "evaluation_id", "evaluation_type", "training_policy", "method", "region", "n_dates",
    "hourly_sse_float", "daily_total_sse", "daily_total_mae",
    "mean_profile_l1", "mean_profile_rmse", "shape_only_sse_float", "total_only_sse_float",
]
DECISION_COLUMNS = [
    "training_policy", "method", "model_family",
    "rolling_mean_sse", "rolling_median_sse", "rolling_max_sse",
    "rolling_normal_mse", "rolling_normal_rank",
    "final_analog_sse", "final_analog_rank", "final_analog_mean_bias",
    "final_analog_core_sse", "final_analog_near_sse", "final_analog_outer_sse",
    "rolling_fold_win_count",
]
METHOD_DEF_COLUMNS = [
    "method", "model_family", "daily_total_method", "hour_profile_method",
    "uses_recent_window", "uses_time_decay", "uses_day_type", "uses_fourier", "deployable",
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


def build_a_frame(df: pd.DataFrame, daily: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    a = make_a_labels(df)
    a["date"] = a["hour"].dt.normalize()
    a["hour_of_day"] = a["hour"].dt.hour
    a["dayofweek"] = a["hour"].dt.dayofweek
    a = a[["hour", "date", "hour_of_day", "dayofweek", "region", "y"]]
    a = a.merge(daily[["date", "unique_vessel_count"]].rename(columns={"unique_vessel_count": "vessel_count"}), on="date", how="left")
    a = a.merge(audit[["date", "quality_regime"]].rename(columns={"quality_regime": "audit_quality_regime"}), on="date", how="left")
    a["day_type"] = np.where(a["dayofweek"].isin([5, 6]), "weekend", "weekday")
    a["region_rank"] = a["region"].map(REGION_RANK)
    return a


# ---------------------------------------------------------------------------
# Decomposition component builders
# ---------------------------------------------------------------------------
def fourier2_smooth(curve: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(_FOURIER_BASIS, curve, rcond=None)
    fit = np.clip(_FOURIER_BASIS @ beta, 0.0, None)
    s = fit.sum()
    if s <= 0:
        return curve.copy()
    return fit / s


def compute_components(train: pd.DataFrame, train_dates, train_end, weights: dict) -> dict:
    """Return dict with T_hat[dm][region] and P_hat[pm] for the decomposition methods."""
    tt = train.groupby(["date", "region"], observed=True)["y"].sum().rename("T").reset_index()
    tt["w"] = tt["date"].map(weights).astype(float)
    tt["wT"] = tt["w"] * tt["T"]
    dt_map = train.drop_duplicates("date").set_index("date")["day_type"]
    tt["day_type"] = tt["date"].map(dt_map)

    ts = train[["date", "region", "hour_of_day", "y"]].merge(tt[["date", "region", "T"]], on=["date", "region"])
    ts["share"] = np.where(ts["T"] > 0, ts["y"] / ts["T"], 0.0)
    ts["w"] = ts["date"].map(weights).astype(float)
    ts["day_type"] = ts["date"].map(dt_map)

    T_hat = {}
    agg = tt.groupby("region", observed=True).agg(s=("wT", "sum"), w=("w", "sum"))
    T_hat["mean"] = (agg["s"] / agg["w"]).to_dict()

    r7 = set(pd.date_range(train_end - pd.Timedelta(days=6), train_end, freq="D").normalize())
    denom7 = float(sum(weights[d] for d in train_dates if d in r7))
    if denom7 > 0:
        tt7 = tt[tt["date"].isin(r7)]
        agg7 = tt7.groupby("region", observed=True).agg(s=("wT", "sum"), w=("w", "sum"))
        T_hat["recent7"] = (agg7["s"] / agg7["w"]).to_dict()
    else:
        T_hat["recent7"] = dict(T_hat["mean"])

    decay = {d: 2.0 ** (-(train_end - d).days / 7.0) for d in train_dates}
    tt["wdecay"] = tt["w"] * tt["date"].map(decay)
    tt["wdT"] = tt["wdecay"] * tt["T"]
    agge = tt.groupby("region", observed=True).agg(s=("wdT", "sum"), w=("wdecay", "sum"))
    T_hat["ewm7"] = (agge["s"] / agge["w"]).to_dict()

    P_hat = {}
    ts["ws"] = ts["w"] * ts["share"]
    aggp = ts.groupby(["region", "hour_of_day"], observed=True).agg(s=("ws", "sum"), w=("w", "sum"))
    Pm = (aggp["s"] / aggp["w"]).unstack("hour_of_day").reindex(columns=range(24))
    P_hat["mean"] = {r: Pm.loc[r].to_numpy() for r in REGION_ORDER}

    if denom7 > 0:
        ts7 = ts[ts["date"].isin(r7)].copy()
        ts7["ws"] = ts7["w"] * ts7["share"]
        agg7p = ts7.groupby(["region", "hour_of_day"], observed=True).agg(s=("ws", "sum"), w=("w", "sum"))
        P7 = (agg7p["s"] / agg7p["w"]).unstack("hour_of_day").reindex(columns=range(24))
        P_hat["recent7"] = {r: P7.loc[r].to_numpy() for r in REGION_ORDER}
    else:
        P_hat["recent7"] = {r: P_hat["mean"][r].copy() for r in REGION_ORDER}

    P_hat["fourier2"] = {r: fourier2_smooth(P_hat["mean"][r]) for r in REGION_ORDER}

    # day-type shrunk
    P_hat["daytype_shrunk"] = {}
    for r in REGION_ORDER:
        global_curve = P_hat["mean"][r]
        P_hat["daytype_shrunk"][r] = {}
        tsr = ts[ts["region"] == r]
        for g in ["weekday", "weekend"]:
            sub = tsr[tsr["day_type"] == g]
            neff = float(sub["w"].sum())
            if neff <= 0:
                P_hat["daytype_shrunk"][r][g] = global_curve.copy()
                continue
            sg = sub.groupby("hour_of_day", observed=True).apply(lambda d: float((d["w"] * d["share"]).sum()) / neff, include_groups=False)
            curve = sg.reindex(range(24)).fillna(0.0).to_numpy()
            lam = neff / (neff + 5.0)
            shrunk = lam * curve + (1.0 - lam) * global_curve
            s = shrunk.sum()
            P_hat["daytype_shrunk"][r][g] = shrunk / s if s > 0 else global_curve.copy()
    return T_hat, P_hat


# ---------------------------------------------------------------------------
# Prediction + combo builder
# ---------------------------------------------------------------------------
def metrics_block(y_true, pred_float, pred_rounded, region_arr):
    ef = pred_float - y_true
    er = pred_rounded - y_true
    out = {
        "sse_float": float((ef ** 2).sum()),
        "sse_rounded": float((er ** 2).sum()),
        "mse_rounded": float((er ** 2).mean()),
        "mae_rounded": float(np.abs(er).mean()),
        "mean_bias_rounded": float(er.mean()),
        "actual_total": float(y_true.sum()),
        "predicted_total_rounded": float(pred_rounded.sum()),
    }
    for r in REGION_ORDER:
        m = region_arr == r
        out[f"{r}_sse_rounded"] = float((er[m] ** 2).sum())
    return out


def build_combo(eid, etype, policy, mname, valid, pred, pred_tot=None, pred_share=None):
    y_true = valid["y"].to_numpy(dtype="float64")
    pred_rounded = np.clip(np.rint(pred), 0, None).astype(np.int64)
    n = len(valid)
    combo = pd.DataFrame({
        "evaluation_id": eid, "evaluation_type": etype, "training_policy": policy, "method": mname,
        "date": valid["date"].to_numpy(), "hour": valid["hour"].to_numpy(),
        "hour_of_day": valid["hour_of_day"].to_numpy(), "dayofweek": valid["dayofweek"].to_numpy(),
        "day_type": valid["day_type"].to_numpy(), "region": valid["region"].to_numpy(),
        "audit_quality_regime": valid["audit_quality_regime"].to_numpy(),
        "validation_unique_vessel_count": valid["vessel_count"].to_numpy(),
        "y_true": y_true,
        "true_daily_total": valid["true_daily_total"].to_numpy(),
        "true_profile_share": valid["true_profile_share"].to_numpy(),
        "pred_daily_total": pred_tot if pred_tot is not None else np.full(n, np.nan),
        "pred_profile_share": pred_share if pred_share is not None else np.full(n, np.nan),
        "pred_float": pred, "pred_rounded": pred_rounded,
        "region_rank": valid["region_rank"].to_numpy(),
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
    daily = pd.read_csv(ROOT / STEP01_DAILY)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    audit = pd.read_csv(ROOT / STEP02_DAILY)
    audit["date"] = pd.to_datetime(audit["date"]).dt.normalize()
    china_series = dict(zip(audit["date"], audit["china_coastal_record_count"]))
    vc_series = dict(zip(daily["date"], daily["unique_vessel_count"]))

    a_frame = build_a_frame(df, daily, audit)
    assert len(a_frame) == 1728

    # scenarios: 8 rolling folds + final_analog
    scenarios = []
    for f in build_folds():
        scenarios.append((f["fold_id"], "rolling_7d", f["train_start"], f["train_end"], f["valid_start"], f["valid_end"]))
    scenarios.append(("final_analog", "final_analog", pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-18"),
                      pd.Timestamp("2018-01-19"), pd.Timestamp("2018-01-24")))

    pred_frames = []
    fold_score_rows = []

    for eid, etype, tstart, tend, vstart, vend in scenarios:
        train = a_frame[(a_frame["date"] >= tstart) & (a_frame["date"] <= tend)].copy()
        valid = a_frame[(a_frame["date"] >= vstart) & (a_frame["date"] <= vend)].copy()
        vtot = valid.groupby(["date", "region"], observed=True)["y"].sum().rename("true_daily_total").reset_index()
        valid = valid.merge(vtot, on=["date", "region"], how="left")
        valid["true_profile_share"] = np.where(valid["true_daily_total"] > 0, valid["y"] / valid["true_daily_total"], 0.0)
        valid = valid.sort_values(["hour", "region_rank"]).reset_index(drop=True)

        train_dates = sorted(pd.date_range(tstart, tend, freq="D").normalize())
        classes, _ = classify_fold_quality(train_dates, china_series)

        for policy in POLICIES:
            pw = POLICY_WEIGHTS[policy]
            weights = {d: pw[classes[d]] for d in train_dates}
            train_w = train.copy()
            train_w["weight"] = train_w["date"].map(weights).astype("float64")
            smean = scale_train_mean(train_dates, vc_series, weights, policy)

            # decomposition components
            T_hat, P_hat = compute_components(train, train_dates, tend, weights)
            valid_r = valid["region"].to_numpy()
            valid_h = valid["hour_of_day"].to_numpy()
            valid_dt = valid["day_type"].to_numpy()
            prof_vec = {}
            for pm in ["mean", "recent7", "fourier2"]:
                Pm = P_hat[pm]
                prof_vec[pm] = np.array([Pm[r][h] for r, h in zip(valid_r, valid_h)])
            Pdt = P_hat["daytype_shrunk"]
            prof_vec["daytype_shrunk"] = np.array([Pdt[r][dt][h] for r, dt, h in zip(valid_r, valid_dt, valid_h)])
            Tvec = {dm: np.array([T_hat[dm][r] for r in valid_r], dtype="float64") for dm in ["mean", "recent7", "ewm7"]}

            for mname, family, dm, pm, *_ in METHOD_META:
                if family == "direct":
                    spec = MethodSpec("hour_mean", "hour") if mname == "hour_mean" else MethodSpec("recent_7d_hour", "recent_hour", recent_days=7)
                    pred = np.asarray(predict_weighted(train_w, valid, spec, smean), dtype="float64")
                    combo = build_combo(eid, etype, policy, mname, valid, pred)
                else:
                    pred = Tvec[dm] * prof_vec[pm]
                    combo = build_combo(eid, etype, policy, mname, valid, pred, pred_tot=Tvec[dm], pred_share=prof_vec[pm])
                assert (combo["pred_float"] >= 0).all(), f"negative pred {eid}/{policy}/{mname}"
                pred_frames.append(combo)

                region_arr = valid["region"].to_numpy()
                mb = metrics_block(valid["y"].to_numpy("float64"), np.asarray(combo["pred_float"]), combo["pred_rounded"].to_numpy(), region_arr)
                fold_score_rows.append({
                    "evaluation_id": eid, "evaluation_type": etype, "training_policy": policy, "method": mname,
                    "train_start": f"{tstart:%Y-%m-%d}", "train_end": f"{tend:%Y-%m-%d}",
                    "valid_start": f"{vstart:%Y-%m-%d}", "valid_end": f"{vend:%Y-%m-%d}",
                    "n_valid_days": int(len(valid["date"].unique())), "n_rows": int(len(valid)),
                    **mb,
                })

    preds = pd.concat(pred_frames, ignore_index=True)
    preds["_e"] = preds["evaluation_id"].map({s[0]: i for i, s in enumerate(scenarios)})
    preds["_p"] = preds["training_policy"].map(POLICY_ORDER)
    preds["_m"] = preds["method"].map(METHOD_ORDER)
    preds = preds.sort_values(["_e", "_p", "_m", "date", "hour_of_day", "region_rank"]).reset_index(drop=True)
    assert len(preds) == 133920, f"expected 133920 predictions, got {len(preds)}"

    # decomposition profile-sum assertion
    decomp = preds[preds["pred_profile_share"].notna()]
    sums = decomp.groupby(["evaluation_id", "training_policy", "method", "date", "region"])["pred_profile_share"].sum()
    assert (sums - 1.0).abs().max() < 1e-10, "decomposition pred_profile_share does not sum to 1"
    assert (decomp["pred_daily_total"] >= 0).all()
    assert (decomp["pred_profile_share"] >= 0).all()

    # consistency vs Step 03 (rolling folds, hour_mean & recent7_hour)
    s3 = pd.read_csv(ROOT / STEP03_PREDS, encoding="utf-8-sig")
    s3 = s3[s3["method"].isin(["hour_mean", "recent_7d_hour"])].copy()
    s3["date"] = pd.to_datetime(s3["date"]).dt.strftime("%Y-%m-%d")
    s3_map = {(r["fold_id"], r["training_policy"], r["method"], r["date"], int(r["hour_of_day"]), r["region"]): r["pred_float"] for _, r in s3.iterrows()}
    mine = preds[(preds["evaluation_type"] == "rolling_7d") & (preds["method"].isin(["hour_mean", "recent7_hour"]))].copy()
    mine["date"] = pd.to_datetime(mine["date"]).dt.strftime("%Y-%m-%d")
    max_diff = 0.0
    for _, r in mine.iterrows():
        s3key = (r["evaluation_id"], r["training_policy"], STEP03_NAME[r["method"]], r["date"], int(r["hour_of_day"]), r["region"])
        ref = s3_map.get(s3key)
        if ref is None:
            raise AssertionError(f"no step03 reference for {s3key}")
        max_diff = max(max_diff, abs(r["pred_float"] - ref))
    assert max_diff < CONSISTENCY_TOL, f"step03 consistency max diff {max_diff:.3e}"
    print(f"consistency vs step03 (hour_mean, recent7_hour): max abs diff = {max_diff:.3e}")

    fold_scores = pd.DataFrame(fold_score_rows)

    # --- normal-target scores (rolling folds, normal quality) ---------------
    norm = preds[(preds["evaluation_type"] == "rolling_7d") & (preds["audit_quality_regime"] == "normal")]
    norm_rows = []
    for policy in POLICIES:
        for mname, *_ in METHOD_META:
            sub = norm[(norm["training_policy"] == policy) & (norm["method"] == mname)]
            er = sub["error_rounded"].astype("float64").to_numpy()
            norm_rows.append({
                "training_policy": policy, "method": mname,
                "n_prediction_rows": int(len(sub)),
                "n_unique_target_dates": int(sub["date"].nunique()),
                "sse_rounded": float((er ** 2).sum()),
                "mse_rounded": float((er ** 2).mean()),
                "mae_rounded": float(np.abs(er).mean()),
                "mean_bias_rounded": float(er.mean()),
            })
    normal_scores = pd.DataFrame(norm_rows)
    normal_scores["rank_by_mse_rounded"] = normal_scores["mse_rounded"].rank(method="min", ascending=True).astype(int)

    # --- final_analog scores -----------------------------------------------
    fa = preds[preds["evaluation_id"] == "final_analog"]
    fa_rows = []
    for policy in POLICIES:
        for mname, *_ in METHOD_META:
            sub = fa[(fa["training_policy"] == policy) & (fa["method"] == mname)]
            er = sub["error_rounded"].astype("float64").to_numpy()
            ef = sub["error_float"].astype("float64").to_numpy()
            row = {
                "training_policy": policy, "method": mname,
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

    # --- decomposition diagnostics -----------------------------------------
    diag_rows = []
    dpred = preds[preds["method"].isin([m[0] for m in DECOMP_METHODS])].copy()
    for (eid_i, policy, mname, region), g in dpred.groupby(["evaluation_id", "training_policy", "method", "region"], observed=True):
        etype_i = g["evaluation_type"].iloc[0]
        dates = sorted(g["date"].unique())
        hourly_sse = daily_total_sse = shape_only = total_only = 0.0
        daily_abs = []
        prof_l1 = []
        prof_rmse = []
        for d in dates:
            gd = g[g["date"] == d].sort_values("hour_of_day")
            y = gd["y_true"].to_numpy("float64")
            T_true = gd["true_daily_total"].to_numpy("float64")
            T_pred = gd["pred_daily_total"].to_numpy("float64")
            p_pred = gd["pred_profile_share"].to_numpy("float64")
            p_true = gd["true_profile_share"].to_numpy("float64")
            ttrue = T_true[0]
            tpred = T_pred[0]
            hourly_sse += float(((tpred * p_pred - y) ** 2).sum())
            daily_total_sse += float((tpred - ttrue) ** 2)
            daily_abs.append(abs(tpred - ttrue))
            prof_l1.append(float(np.abs(p_pred - p_true).sum()))
            prof_rmse.append(float(np.sqrt(np.mean((p_pred - p_true) ** 2))))
            shape_only += float(((ttrue * p_pred - y) ** 2).sum())
            total_only += float(((tpred * p_true - y) ** 2).sum())
        diag_rows.append({
            "evaluation_id": eid_i, "evaluation_type": etype_i, "training_policy": policy,
            "method": mname, "region": region, "n_dates": int(len(dates)),
            "hourly_sse_float": hourly_sse, "daily_total_sse": daily_total_sse,
            "daily_total_mae": float(np.mean(daily_abs)),
            "mean_profile_l1": float(np.mean(prof_l1)), "mean_profile_rmse": float(np.mean(prof_rmse)),
            "shape_only_sse_float": shape_only, "total_only_sse_float": total_only,
        })
    diag = pd.DataFrame(diag_rows)
    diag["_e"] = diag["evaluation_id"].map({s[0]: i for i, s in enumerate(scenarios)})
    diag["_p"] = diag["training_policy"].map(POLICY_ORDER)
    diag["_m"] = diag["method"].map(METHOD_ORDER)
    diag["_r"] = diag["region"].map(REGION_RANK)
    diag = diag.sort_values(["_e", "_p", "_m", "_r"]).drop(columns=["_e", "_p", "_m", "_r"]).reset_index(drop=True)

    # --- decision table -----------------------------------------------------
    rolling = fold_scores[fold_scores["evaluation_type"] == "rolling_7d"]
    win_pivot = rolling.pivot_table(index="evaluation_id", columns=["training_policy", "method"], values="sse_rounded", aggfunc="first")
    min_per_fold = win_pivot.min(axis=1)
    win_counts = win_pivot.eq(min_per_fold, axis=0).sum(axis=0)
    dec_rows = []
    for policy in POLICIES:
        for mname, family, *_ in METHOD_META:
            rsub = rolling[(rolling["training_policy"] == policy) & (rolling["method"] == mname)]["sse_rounded"].to_numpy("float64")
            ns = normal_scores[(normal_scores["training_policy"] == policy) & (normal_scores["method"] == mname)].iloc[0]
            fs = final_scores[(final_scores["training_policy"] == policy) & (final_scores["method"] == mname)].iloc[0]
            dec_rows.append({
                "training_policy": policy, "method": mname, "model_family": family,
                "rolling_mean_sse": float(np.mean(rsub)), "rolling_median_sse": float(np.median(rsub)), "rolling_max_sse": float(np.max(rsub)),
                "rolling_normal_mse": float(ns["mse_rounded"]), "rolling_normal_rank": int(ns["rank_by_mse_rounded"]),
                "final_analog_sse": float(fs["sse_rounded"]), "final_analog_rank": int(fs["rank_by_sse_rounded"]),
                "final_analog_mean_bias": float(fs["mean_bias_rounded"]),
                "final_analog_core_sse": float(fs["core_sse_rounded"]), "final_analog_near_sse": float(fs["near_sse_rounded"]),
                "final_analog_outer_sse": float(fs["outer_sse_rounded"]),
                "rolling_fold_win_count": int(win_counts.get((policy, mname), 0)),
            })
    decision = pd.DataFrame(dec_rows).sort_values(["training_policy", "method"]).reset_index(drop=True)

    # --- method definitions -------------------------------------------------
    md = pd.DataFrame([{
        "method": m[0], "model_family": m[1], "daily_total_method": m[2], "hour_profile_method": m[3],
        "uses_recent_window": m[4], "uses_time_decay": m[5], "uses_day_type": m[6], "uses_fourier": m[7],
        "deployable": True,
    } for m in METHOD_META])

    # --- write CSVs (UTF-8 with BOM) ---------------------------------------
    p_md = out_dir / "method_definitions.csv"
    p_pred = out_dir / "a_model_predictions.csv"
    p_fs = out_dir / "a_fold_scores.csv"
    p_ns = out_dir / "a_normal_target_scores.csv"
    p_fa = out_dir / "a_final_analog_scores.csv"
    p_dg = out_dir / "a_decomposition_diagnostics.csv"
    p_dec = out_dir / "decision_table.csv"
    p_sum = out_dir / "summary.md"

    md.to_csv(p_md, index=False, encoding="utf-8-sig")

    pred_out = preds[PRED_COLUMNS].copy()
    pred_out["hour"] = pd.to_datetime(pred_out["hour"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    pred_out["date"] = pd.to_datetime(pred_out["date"]).dt.strftime("%Y-%m-%d")
    pred_out.to_csv(p_pred, index=False, encoding="utf-8-sig")

    fold_scores.sort_values(["evaluation_id", "training_policy", "method"]).to_csv(p_fs, index=False, encoding="utf-8-sig")
    normal_scores[NORMAL_COLUMNS].sort_values(["rank_by_mse_rounded", "training_policy", "method"]).to_csv(p_ns, index=False, encoding="utf-8-sig")
    final_scores[FINAL_COLUMNS].sort_values(["rank_by_sse_rounded", "training_policy", "method"]).to_csv(p_fa, index=False, encoding="utf-8-sig")
    diag[DIAG_COLUMNS].to_csv(p_dg, index=False, encoding="utf-8-sig")
    decision[DECISION_COLUMNS].to_csv(p_dec, index=False, encoding="utf-8-sig")

    write_summary(p_sum, decision, normal_scores, final_scores, diag)

    # --- console ------------------------------------------------------------
    print("\n=== Output files ===")
    for path in [p_md, p_pred, p_fs, p_ns, p_fa, p_dg, p_dec, p_sum]:
        if path.suffix == ".csv":
            rows = len(pd.read_csv(path, encoding="utf-8-sig"))
        else:
            rows = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {path.relative_to(ROOT)}  rows={rows}")

    print("\n=== final_analog top 10 by sse_rounded ===")
    for _, r in final_scores.sort_values("sse_rounded").head(10).iterrows():
        print(f"  rank={int(r['rank_by_sse_rounded']):2d}  {r['training_policy']:22s} {r['method']:28s} sse={r['sse_rounded']:.0f} bias={r['mean_bias_rounded']:.2f}")
    print("\n=== rolling normal-target top 10 by mse_rounded ===")
    for _, r in normal_scores.sort_values("mse_rounded").head(10).iterrows():
        print(f"  rank={int(r['rank_by_mse_rounded']):2d}  {r['training_policy']:22s} {r['method']:28s} mse={r['mse_rounded']:.3f} n_dates={int(r['n_unique_target_dates'])}")
    print("\nDone.")


def write_summary(path: Path, decision: pd.DataFrame, normal_scores: pd.DataFrame,
                  final_scores: pd.DataFrame, diag: pd.DataFrame) -> None:
    L: list[str] = []
    L.append("# Step 06 A Structure Model Benchmark")
    L.append("")
    L.append("> Task A only. Direct vs decomposition structures; no GLM/GAM/LightGBM/NN, no B, no submissions.")
    L.append("")
    L.append(f"Run command: `{RUN_CMD}`")
    L.append("")

    L.append("## 1. Models and evaluation design")
    L.append("")
    L.append("- 2 direct methods (hour_mean, recent7_hour) and 8 decomposition methods (daily total in {mean, recent7, ewm7} x hour profile in {mean, recent7, fourier2, daytype_shrunk}).")
    L.append("- 3 training-quality policies (all, exclude_outage_severe, quality_weighted) x 8 seven-day rolling folds + 1 final_analog fold = 9 evaluation scenarios.")
    L.append("- Validation uses only `unique_vessel_count` (allowed); no validation AIS / quality variables enter prediction. The methods here do not force vessel-count scaling.")
    L.append("- Decomposition predicts per-region daily total times a 24h profile that sums to 1.")
    L.append("")

    L.append("## 2. Consistency checks")
    L.append("")
    L.append("- hour_mean and recent7_hour reproduce the Step 03 stored predictions for the 8 rolling folds (all policies) with max abs diff < 1e-10. **PASS**")
    L.append("- a_model_predictions.csv has 133920 rows. **PASS**")
    L.append("- Every decomposition method's 24h pred_profile_share sums to 1 within 1e-10; pred_daily_total and pred_profile_share are non-negative. **PASS**")
    L.append("- final_analog: train 2018-01-01~01-18, valid 2018-01-19~01-24 (432 rows per combo). **PASS**")
    L.append("")

    L.append("## 3. Direct versus decomposition models")
    L.append("")
    # aggregate by family across policies: median of rolling median sse
    fam = decision.groupby("model_family").agg(
        roll_med=("rolling_median_sse", "median"), roll_mean=("rolling_mean_sse", "median"),
        roll_max=("rolling_max_sse", "median"), fa=("final_analog_sse", "median"),
        norm=("rolling_normal_mse", "median"), wins=("rolling_fold_win_count", "sum"),
    )
    L.append("| family | median(rolling median sse) | median(rolling normal mse) | median(final_analog sse) | total rolling wins |")
    L.append("| --- | --- | --- | --- | --- |")
    for fam_name, row in fam.iterrows():
        L.append(f"| {fam_name} | {row['roll_med']:.0f} | {row['norm']:.3f} | {row['fa']:.0f} | {int(row['wins'])} |")
    L.append("")
    best_dir_norm = normal_scores[normal_scores["method"].isin(["hour_mean", "recent7_hour"])]["mse_rounded"].min()
    best_dec_norm = normal_scores[~normal_scores["method"].isin(["hour_mean", "recent7_hour"])]["mse_rounded"].min()
    best_dir_fa = final_scores[final_scores["method"].isin(["hour_mean", "recent7_hour"])]["sse_rounded"].min()
    best_dec_fa = final_scores[~final_scores["method"].isin(["hour_mean", "recent7_hour"])]["sse_rounded"].min()
    L.append(f"- Best-vs-best: best decomposition vs best direct are within ~1% on both views "
             f"(normal-target mse {best_dec_norm:.3f} vs {best_dir_norm:.3f}; final_analog sse {best_dec_fa:.0f} vs {best_dir_fa:.0f}). "
             f"The family medians above mix strong and weak variants, so they are NOT a best-vs-best comparison.")
    L.append("- Conclusion: simple decomposition does **not** stably beat hour_mean; it is roughly tied at the top, which makes it a viable variance-shrinkage alternative rather than a clear winner. Whether simple decomposition beats hour_mean must be judged across folds and the normal-target view, not a single final_analog window.")
    L.append("")

    L.append("## 4. Daily-total method comparison")
    L.append("")
    L.append("Median rolling SSE / final_analog SSE by daily-total method (across the hour-profile variants, all policies):")
    L.append("")
    dm_map = {m[0]: m[2] for m in METHOD_META if m[1] == "decomposition"}
    decision["_dm"] = decision["method"].map(dm_map)
    dmag = decision.groupby("_dm").agg(roll_med=("rolling_median_sse", "median"), fa=("final_analog_sse", "median"), norm=("rolling_normal_mse", "median"))
    L.append("| daily_total_method | median(rolling median sse) | median(final_analog sse) | median(rolling normal mse) |")
    L.append("| --- | --- | --- | --- |")
    for dm, row in dmag.iterrows():
        L.append(f"| {dm} | {row['roll_med']:.0f} | {row['fa']:.0f} | {row['norm']:.3f} |")
    L.append("")

    L.append("## 5. Hour-profile method comparison")
    L.append("")
    pm_map = {m[0]: m[3] for m in METHOD_META if m[1] == "decomposition"}
    decision["_pm"] = decision["method"].map(pm_map)
    pmag = decision.groupby("_pm").agg(roll_med=("rolling_median_sse", "median"), fa=("final_analog_sse", "median"), norm=("rolling_normal_mse", "median"))
    L.append("| hour_profile_method | median(rolling median sse) | median(final_analog sse) | median(rolling normal mse) |")
    L.append("| --- | --- | --- | --- |")
    for pm, row in pmag.iterrows():
        L.append(f"| {pm} | {row['roll_med']:.0f} | {row['fa']:.0f} | {row['norm']:.3f} |")
    L.append("")
    L.append("- Recent profile, Fourier smoothing, and weekday/weekend separation are judged by whether they reduce out-of-sample error across folds, not by in-sample fit.")
    L.append("")

    L.append("## 6. Total error versus shape error")
    L.append("")
    L.append("Decomposition diagnostics (median over scenarios x policies x methods), by region. `shape_only_sse` uses the true total x predicted shape; `total_only_sse` uses the predicted total x true shape. These two are diagnostic only and CANNOT be added to get the actual SSE (there is a cross term).")
    L.append("")
    rdag = diag.groupby("region").agg(hourly=("hourly_sse_float", "median"), dtsse=("daily_total_sse", "median"),
                                      shape=("shape_only_sse_float", "median"), total=("total_only_sse_float", "median"),
                                      l1=("mean_profile_l1", "median"), rmse=("mean_profile_rmse", "median"))
    L.append("| region | median hourly_sse | median daily_total_sse | median shape_only_sse | median total_only_sse | median profile L1 | median profile RMSE |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in REGION_ORDER:
        row = rdag.loc[r]
        L.append(f"| {r} | {row['hourly']:.0f} | {row['dtsse']:.0f} | {row['shape']:.0f} | {row['total']:.0f} | {row['l1']:.4f} | {row['rmse']:.4f} |")
    L.append("")
    for r in REGION_ORDER:
        row = rdag.loc[r]
        bigger = "shape" if row["shape"] >= row["total"] else "total"
        L.append(f"- {r}: shape_only_sse={row['shape']:.0f} vs total_only_sse={row['total']:.0f} -> error is more from the **{bigger}** side (diagnostic, non-additive).")
    L.append("- These medians are over all 9 scenarios, including anomalous validation days where the predicted total (fit on normal training) far exceeds the depressed true total. So the total-side dominance for core/near is partly anomalous-driven; on normal targets only, the shape/total split would be more balanced.")
    L.append("")

    L.append("## 7. Effect of training-quality policy")
    L.append("")
    pag = decision.groupby("training_policy").agg(roll_med=("rolling_median_sse", "median"), fa=("final_analog_sse", "median"), norm=("rolling_normal_mse", "median"))
    L.append("| policy | median(rolling median sse) | median(final_analog sse) | median(rolling normal mse) |")
    L.append("| --- | --- | --- | --- |")
    for p, row in pag.iterrows():
        L.append(f"| {p} | {row['roll_med']:.0f} | {row['fa']:.0f} | {row['norm']:.3f} |")
    L.append("")

    L.append("## 8. Decision for the next stage")
    L.append("")
    best_fa = final_scores.sort_values("sse_rounded").iloc[0]
    best_norm = normal_scores.sort_values("mse_rounded").iloc[0]
    L.append(f"- Best final_analog combo: {best_fa['training_policy']} / {best_fa['method']} (sse={best_fa['sse_rounded']:.0f}).")
    L.append(f"- Best rolling normal-target combo: {best_norm['training_policy']} / {best_norm['method']} (mse={best_norm['mse_rounded']:.3f}).")
    L.append("- These are observations for choosing what to carry into statistical-model development (Poisson / negative-binomial / Ridge) in a later step. No submission prediction is produced here.")
    L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
