from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import train_internal_forecast as tif


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    params: dict


def add_ml_features(df: pd.DataFrame, task: str) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = pd.to_datetime(out["hour"])
    out["date"] = out["hour"].dt.normalize()
    out["hour_of_day"] = out["hour"].dt.hour
    out["dayofweek"] = out["hour"].dt.dayofweek
    out["day"] = out["hour"].dt.day
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour_of_day"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_of_day"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["vessel_count_sq"] = out["vessel_count"] ** 2
    if task == "A":
        out["key"] = out["region"]
        out["source_region"] = "none"
        out["target_region"] = "none"
    else:
        out["key"] = out["source_region"] + "->" + out["target_region"]
        out["region"] = "none"
    out["key_hour"] = out["key"] + "_h" + out["hour_of_day"].astype(str)
    out["key_dow"] = out["key"] + "_d" + out["dayofweek"].astype(str)
    return out


def feature_columns(task: str) -> tuple[list[str], list[str]]:
    numeric = [
        "hour_of_day",
        "dayofweek",
        "day",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "vessel_count",
        "vessel_count_sq",
    ]
    categorical = ["key", "key_hour", "key_dow"]
    if task == "A":
        categorical += ["region"]
    else:
        categorical += ["source_region", "target_region"]
    return numeric, categorical


def make_model(spec: ModelSpec, task: str) -> Pipeline:
    numeric, categorical = feature_columns(task)
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )
    if spec.kind == "ridge":
        model = Ridge(**spec.params)
    elif spec.kind == "rf":
        model = RandomForestRegressor(**spec.params)
    elif spec.kind == "hgb":
        model = HistGradientBoostingRegressor(**spec.params)
    elif spec.kind == "lgbm":
        model = LGBMRegressor(**spec.params)
    else:
        raise ValueError(spec)
    return Pipeline([("pre", pre), ("model", model)])


def model_specs(task: str) -> list[ModelSpec]:
    return [
        ModelSpec("ridge_a1", "ridge", {"alpha": 1.0}),
        ModelSpec("ridge_a10", "ridge", {"alpha": 10.0}),
        ModelSpec("ridge_a100", "ridge", {"alpha": 100.0}),
        ModelSpec("rf_200_d4", "rf", {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 4, "random_state": 42, "n_jobs": -1}),
        ModelSpec("rf_300_d6", "rf", {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 3, "random_state": 43, "n_jobs": -1}),
        ModelSpec("hgb_l2_0.1", "hgb", {"max_iter": 200, "learning_rate": 0.03, "l2_regularization": 0.1, "max_leaf_nodes": 15, "random_state": 42}),
        ModelSpec("hgb_l2_1", "hgb", {"max_iter": 250, "learning_rate": 0.03, "l2_regularization": 1.0, "max_leaf_nodes": 15, "random_state": 43}),
        ModelSpec(
            "lgbm_small",
            "lgbm",
            {
                "n_estimators": 150,
                "learning_rate": 0.03,
                "num_leaves": 7,
                "min_child_samples": 15,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
                "random_state": 42,
                "verbosity": -1,
            },
        ),
        ModelSpec(
            "lgbm_medium",
            "lgbm",
            {
                "n_estimators": 250,
                "learning_rate": 0.025,
                "num_leaves": 15,
                "min_child_samples": 10,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 3.0,
                "random_state": 44,
                "verbosity": -1,
            },
        ),
    ]


def train_predict(train: pd.DataFrame, valid: pd.DataFrame, task: str, spec: ModelSpec) -> np.ndarray:
    train_f = add_ml_features(train, task)
    valid_f = add_ml_features(valid, task)
    numeric, categorical = feature_columns(task)
    cols = numeric + categorical
    pipe = make_model(spec, task)
    pipe.fit(train_f[cols], train_f["y"])
    pred = pipe.predict(valid_f[cols])
    return np.clip(pred, 0, None)


def backtest(frame: pd.DataFrame, task: str, specs: list[ModelSpec]) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    splits = [
        ("2018-01-01", "2018-01-11", "2018-01-17"),
        ("2018-01-01", "2018-01-18", "2018-01-24"),
        ("2018-01-08", "2018-01-18", "2018-01-24"),
        ("2018-01-12", "2018-01-19", "2018-01-24"),
    ]
    rows = []
    preds_by_method: dict[str, list[dict]] = {s.name: [] for s in specs}
    for train_start, valid_start, valid_end in splits:
        ts = pd.Timestamp(train_start)
        vs = pd.Timestamp(valid_start)
        ve = pd.Timestamp(valid_end) + pd.Timedelta(days=1)
        train = frame[(frame["hour"] >= ts) & (frame["hour"] < vs)].copy()
        valid = frame[(frame["hour"] >= vs) & (frame["hour"] < ve)].copy()
        split = f"{train_start}_{valid_start}_{valid_end}"
        for spec in specs:
            pred = train_predict(train, valid.drop(columns=["y"]), task, spec)
            sse = float(np.sum((valid["y"].to_numpy(float) - pred) ** 2))
            rows.append({"task": task, "split": split, "method": spec.name, "sse": sse})
            preds_by_method[spec.name].append({"split": split, "y": valid["y"].to_numpy(float), "pred": pred})
    return pd.DataFrame(rows), preds_by_method


def summarize(bt: pd.DataFrame) -> pd.DataFrame:
    out = (
        bt.groupby("method")
        .agg(mean_sse=("sse", "mean"), median_sse=("sse", "median"), max_sse=("sse", "max"), std_sse=("sse", "std"))
        .reset_index()
    )
    out["robust_score"] = out["mean_sse"] + 0.15 * out["max_sse"] + 0.05 * out["std_sse"].fillna(0)
    return out.sort_values("robust_score")


def ensemble_search(preds_by_method: dict[str, list[dict]], top_methods: list[str], step: float = 0.1) -> pd.DataFrame:
    def weights(n: int):
        vals = np.arange(0, 1 + 1e-9, step)
        def rec(k: int, rem: float, arr: list[float]):
            if k == n - 1:
                yield np.array(arr + [rem])
            else:
                for v in vals:
                    if v <= rem + 1e-9:
                        yield from rec(k + 1, round(rem - v, 10), arr + [float(v)])
        yield from rec(0, 1.0, [])

    rows = []
    n = len(top_methods)
    for w in weights(n):
        split_sses = []
        for i in range(len(next(iter(preds_by_method.values())))):
            y = preds_by_method[top_methods[0]][i]["y"]
            pred = sum(w[j] * preds_by_method[m][i]["pred"] for j, m in enumerate(top_methods))
            split_sses.append(float(np.sum((y - pred) ** 2)))
        rows.append(
            {
                "methods": "|".join(top_methods),
                "weights": json.dumps(w.round(3).tolist()),
                "mean_sse": float(np.mean(split_sses)),
                "max_sse": float(np.max(split_sses)),
                "std_sse": float(np.std(split_sses, ddof=1)),
                "robust_score": float(np.mean(split_sses) + 0.15 * np.max(split_sses) + 0.05 * np.std(split_sses, ddof=1)),
                "split_sses": json.dumps([round(x, 3) for x in split_sses]),
            }
        )
    return pd.DataFrame(rows).sort_values("robust_score")


def fit_final_predict(frame: pd.DataFrame, future: pd.DataFrame, task: str, spec_names: list[str], weights: list[float]) -> np.ndarray:
    spec_map = {s.name: s for s in model_specs(task)}
    pred = np.zeros(len(future), dtype=float)
    for name, weight in zip(spec_names, weights):
        p = train_predict(frame, future, task, spec_map[name])
        pred += weight * p
    return np.clip(pred, 0, None)


def write_official_float(run_dir: Path, template_dir: Path, a_pred: pd.DataFrame, b_pred: pd.DataFrame) -> None:
    def read_template(zip_name: str, csv_name: str) -> pd.DataFrame:
        with zipfile.ZipFile(template_dir / zip_name) as z:
            with z.open(csv_name) as f:
                return pd.read_csv(f)

    a_t = read_template("提交结果1_区域活跃拖轮数量.zip", "提交结果1_区域活跃拖轮数量.csv")
    b_t = read_template("提交结果2_圈层间拖轮迁移量.zip", "提交结果2_圈层间拖轮迁移量.csv")
    a_t["time_dt"] = pd.to_datetime(a_t["time_window"])
    b_t["time_dt"] = pd.to_datetime(b_t["time_window"])
    a = a_pred.copy()
    a["time_dt"] = a["hour"]
    a["zone"] = a["region"].map(tif.REGION_CN)
    b = b_pred.copy()
    b["time_dt"] = b["hour"]
    b["source_zone"] = b["source_region"].map(tif.REGION_CN)
    b["target_zone"] = b["target_region"].map(tif.REGION_CN)
    af = a_t.drop(columns=["vessel_count"]).merge(
        a[["time_dt", "zone", "prediction"]], on=["time_dt", "zone"], how="left", validate="one_to_one"
    )
    bf = b_t.drop(columns=["vessel_count"]).merge(
        b[["time_dt", "source_zone", "target_zone", "prediction"]],
        on=["time_dt", "source_zone", "target_zone"],
        how="left",
        validate="one_to_one",
    )
    if af["prediction"].isna().any() or bf["prediction"].isna().any():
        raise ValueError("template mismatch")
    a_out = af[["time_window", "zone"]].copy()
    a_out["vessel_count"] = af["prediction"].round(6)
    b_out = bf[["time_window", "source_zone", "target_zone"]].copy()
    b_out["vessel_count"] = bf["prediction"].round(6)
    for df, csv_name, zip_name in [
        (a_out, "提交结果1_区域活跃拖轮数量.csv", "提交结果1_区域活跃拖轮数量.zip"),
        (b_out, "提交结果2_圈层间拖轮迁移量.csv", "提交结果2_圈层间拖轮迁移量.zip"),
    ]:
        csv = run_dir / csv_name
        zpath = run_dir / zip_name
        df.to_csv(csv, index=False, encoding="utf-8-sig")
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(csv, arcname=csv_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = tif.find_project_root(args.root)
    quality = tif.latest_quality_run(root)
    train_csv = tif.find_train_csv(root)
    train_daily = tif.compute_train_daily_vessels(train_csv)
    val_daily = pd.read_csv(tif.find_val_daily(root), parse_dates=["date"])
    a, b = tif.load_labels(quality, "raw", train_daily)
    run_dir = root / "outputs" / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}_ml_forecast"
    run_dir.mkdir(parents=True, exist_ok=True)

    bt_a, pred_a = backtest(a, "A", model_specs("A"))
    bt_b, pred_b = backtest(b, "B", model_specs("B"))
    sum_a = summarize(bt_a)
    sum_b = summarize(bt_b)
    top_a = sum_a.head(3)["method"].tolist()
    top_b = sum_b.head(3)["method"].tolist()
    ens_a = ensemble_search(pred_a, top_a, step=0.1)
    ens_b = ensemble_search(pred_b, top_b, step=0.1)

    bt_a.to_csv(run_dir / "backtest_A_detail.csv", index=False, encoding="utf-8-sig")
    bt_b.to_csv(run_dir / "backtest_B_detail.csv", index=False, encoding="utf-8-sig")
    sum_a.to_csv(run_dir / "backtest_A_summary.csv", index=False, encoding="utf-8-sig")
    sum_b.to_csv(run_dir / "backtest_B_summary.csv", index=False, encoding="utf-8-sig")
    ens_a.to_csv(run_dir / "ensemble_A_search.csv", index=False, encoding="utf-8-sig")
    ens_b.to_csv(run_dir / "ensemble_B_search.csv", index=False, encoding="utf-8-sig")

    fa = tif.future_grid("A", val_daily)
    fb = tif.future_grid("B", val_daily)
    best_a_methods = top_a
    best_b_methods = top_b
    best_a_weights = json.loads(ens_a.iloc[0]["weights"])
    best_b_weights = json.loads(ens_b.iloc[0]["weights"])
    fa["prediction"] = fit_final_predict(a, fa, "A", best_a_methods, best_a_weights)
    fb["prediction"] = fit_final_predict(b, fb, "B", best_b_methods, best_b_weights)
    fa.to_csv(run_dir / "submit_A_debug.csv", index=False, encoding="utf-8-sig")
    fb.to_csv(run_dir / "submit_B_debug.csv", index=False, encoding="utf-8-sig")
    write_official_float(run_dir, root / "outputs" / "_official_examples_backup", fa, fb)

    metadata = {
        "A_top_methods": best_a_methods,
        "A_weights": best_a_weights,
        "B_top_methods": best_b_methods,
        "B_weights": best_b_weights,
        "A_best_single_mean_sse": float(sum_a.iloc[0]["mean_sse"]),
        "B_best_single_mean_sse": float(sum_b.iloc[0]["mean_sse"]),
        "A_ensemble_mean_sse": float(ens_a.iloc[0]["mean_sse"]),
        "B_ensemble_mean_sse": float(ens_b.iloc[0]["mean_sse"]),
        "combined_ensemble_score": float(ens_a.iloc[0]["mean_sse"] + 3 * ens_b.iloc[0]["mean_sse"]),
        "A_sum": float(fa["prediction"].sum()),
        "B_sum": float(fb["prediction"].sum()),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
