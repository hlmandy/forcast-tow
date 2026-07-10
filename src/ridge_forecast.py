from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import train_internal_forecast as tif


@dataclass(frozen=True)
class RidgeSpec:
    name: str
    alpha: float
    include_key_hour: bool = True
    include_key_dow: bool = True
    include_vessel: bool = True
    include_recent_blend: bool = False
    recent_days: int = 14
    recent_weight: float = 0.25


def make_feature_frame(df: pd.DataFrame, task: str) -> pd.DataFrame:
    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out["hour_of_day"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_of_day"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    if task == "A":
        out["key"] = out["region"]
    else:
        out["key"] = out["source_region"] + "->" + out["target_region"]
    out["key_hour"] = out["key"] + "_h" + out["hour_of_day"].astype(str)
    out["key_dow"] = out["key"] + "_d" + out["dayofweek"].astype(str)
    return out


def fit_encoder(train: pd.DataFrame, task: str, spec: RidgeSpec) -> list[str]:
    cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    if spec.include_vessel:
        cols.append("vessel_count")
    cat_cols = ["key"]
    if spec.include_key_hour:
        cat_cols.append("key_hour")
    if spec.include_key_dow:
        cat_cols.append("key_dow")
    encoded_names: list[str] = ["intercept", *cols]
    for col in cat_cols:
        for val in sorted(train[col].astype(str).unique()):
            encoded_names.append(f"{col}={val}")
    return encoded_names


def transform(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    n = len(df)
    x = np.zeros((n, len(feature_names)), dtype=float)
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    x[:, name_to_idx["intercept"]] = 1.0
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "vessel_count"]:
        idx = name_to_idx.get(col)
        if idx is not None:
            vals = df[col].to_numpy(float)
            if col == "vessel_count":
                vals = (vals - np.nanmean(vals)) / (np.nanstd(vals) or 1.0)
            x[:, idx] = np.nan_to_num(vals)
    for col in ["key", "key_hour", "key_dow"]:
        vals = df[col].astype(str).to_numpy()
        for row, val in enumerate(vals):
            idx = name_to_idx.get(f"{col}={val}")
            if idx is not None:
                x[row, idx] = 1.0
    return x


def ridge_fit_predict(train: pd.DataFrame, future: pd.DataFrame, task: str, spec: RidgeSpec) -> np.ndarray:
    train_f = make_feature_frame(train, task)
    future_f = make_feature_frame(future, task)
    names = fit_encoder(train_f, task, spec)
    x = transform(train_f, names)
    xf = transform(future_f, names)
    y = train_f["y"].to_numpy(float)
    penalty = np.eye(x.shape[1]) * spec.alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    pred = xf @ beta
    pred = np.clip(pred, 0, None)
    if spec.include_recent_blend:
        recent_spec = tif.ForecastSpec("recent", "hour_mean", recent_days=spec.recent_days)
        recent = tif.predict(train, future, task, recent_spec).to_numpy()
        pred = (1 - spec.recent_weight) * pred + spec.recent_weight * recent
    return np.clip(pred, 0, None)


def specs(task: str) -> list[RidgeSpec]:
    alphas = [0.01, 0.1, 1, 10, 100, 1000]
    out = []
    for alpha in alphas:
        out.append(RidgeSpec(f"ridge_full_a{alpha:g}", alpha))
        out.append(RidgeSpec(f"ridge_no_dow_a{alpha:g}", alpha, include_key_dow=False))
        out.append(RidgeSpec(f"ridge_no_keyhour_a{alpha:g}", alpha, include_key_hour=False))
        out.append(RidgeSpec(f"ridge_recentblend_a{alpha:g}", alpha, include_recent_blend=True, recent_days=14, recent_weight=0.25))
    return out


def sse(y: pd.Series, p: np.ndarray) -> float:
    return float(np.sum((pd.Series(y).to_numpy(float) - p) ** 2))


def backtest(frame: pd.DataFrame, task: str, spec_list: list[RidgeSpec]) -> pd.DataFrame:
    splits = [
        ("2018-01-01", "2018-01-11", "2018-01-17"),
        ("2018-01-01", "2018-01-18", "2018-01-24"),
        ("2018-01-08", "2018-01-18", "2018-01-24"),
        ("2018-01-12", "2018-01-19", "2018-01-24"),
    ]
    rows = []
    for train_start, valid_start, valid_end in splits:
        ts = pd.Timestamp(train_start)
        vs = pd.Timestamp(valid_start)
        ve = pd.Timestamp(valid_end) + pd.Timedelta(days=1)
        train = frame[(frame["hour"] >= ts) & (frame["hour"] < vs)].copy()
        valid = frame[(frame["hour"] >= vs) & (frame["hour"] < ve)].copy()
        split = f"{train_start}_{valid_start}_{valid_end}"
        for spec in spec_list:
            pred = ridge_fit_predict(train, valid.drop(columns=["y"]), task, spec)
            rows.append({"task": task, "split": split, "method": spec.name, "sse": sse(valid["y"], pred)})
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> pd.DataFrame:
    out = (
        bt.groupby("method")
        .agg(mean_sse=("sse", "mean"), median_sse=("sse", "median"), max_sse=("sse", "max"), std_sse=("sse", "std"))
        .reset_index()
    )
    out["robust_score"] = out["mean_sse"] + 0.15 * out["max_sse"] + 0.05 * out["std_sse"].fillna(0)
    return out.sort_values("robust_score")


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
    af = a_t.drop(columns=["vessel_count"]).merge(
        a[["time_dt", "zone", "prediction"]], on=["time_dt", "zone"], how="left", validate="one_to_one"
    )
    b = b_pred.copy()
    b["time_dt"] = b["hour"]
    b["source_zone"] = b["source_region"].map(tif.REGION_CN)
    b["target_zone"] = b["target_region"].map(tif.REGION_CN)
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
    run_dir = root / "outputs" / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}_ridge_forecast"
    run_dir.mkdir(parents=True, exist_ok=True)

    bt_a = backtest(a, "A", specs("A"))
    bt_b = backtest(b, "B", specs("B"))
    sum_a = summarize(bt_a)
    sum_b = summarize(bt_b)
    best_a = sum_a.iloc[0]["method"]
    best_b = sum_b.iloc[0]["method"]
    spec_a = {s.name: s for s in specs("A")}[best_a]
    spec_b = {s.name: s for s in specs("B")}[best_b]
    fa = tif.future_grid("A", val_daily)
    fb = tif.future_grid("B", val_daily)
    fa["prediction"] = ridge_fit_predict(a, fa, "A", spec_a)
    fb["prediction"] = ridge_fit_predict(b, fb, "B", spec_b)

    bt_a.to_csv(run_dir / "backtest_A_detail.csv", index=False, encoding="utf-8-sig")
    bt_b.to_csv(run_dir / "backtest_B_detail.csv", index=False, encoding="utf-8-sig")
    sum_a.to_csv(run_dir / "backtest_A_summary.csv", index=False, encoding="utf-8-sig")
    sum_b.to_csv(run_dir / "backtest_B_summary.csv", index=False, encoding="utf-8-sig")
    fa.to_csv(run_dir / "submit_A_debug.csv", index=False, encoding="utf-8-sig")
    fb.to_csv(run_dir / "submit_B_debug.csv", index=False, encoding="utf-8-sig")
    write_official_float(run_dir, root / "outputs" / "_official_examples_backup", fa, fb)
    metadata = {
        "best_A_method": str(best_a),
        "best_B_method": str(best_b),
        "A_mean_sse": float(sum_a.iloc[0]["mean_sse"]),
        "B_mean_sse": float(sum_b.iloc[0]["mean_sse"]),
        "combined_mean_score": float(sum_a.iloc[0]["mean_sse"] + 3 * sum_b.iloc[0]["mean_sse"]),
        "A_sum": float(fa["prediction"].sum()),
        "B_sum": float(fb["prediction"].sum()),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
