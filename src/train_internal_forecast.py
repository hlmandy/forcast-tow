from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REGION_ORDER = ["core", "near", "outer"]
REGION_CN = {"core": "核心区", "near": "近港区", "outer": "外围区"}
PAIRS = [(s, t) for s in REGION_ORDER for t in REGION_ORDER if s != t]


@dataclass(frozen=True)
class ForecastSpec:
    name: str
    family: str
    recent_days: int | None = None
    vessel_power: float = 0.0
    blend: float = 0.0
    shrink: float = 0.0
    use_dow: bool = False


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    candidates = [start, *start.parents, Path(r"C:\Users\mandy\hl_Documents\2026CTS")]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "outputs").exists() and ((candidate / "PLAN.md").exists() or (candidate / "notebooks").exists()):
            return candidate
    raise FileNotFoundError("Could not find project root")


def latest_quality_run(root: Path) -> Path:
    runs = sorted((root / "outputs").glob("*_quality_cleaning"))
    if not runs:
        raise FileNotFoundError("No *_quality_cleaning directory found")
    return runs[-1]


def find_train_csv(root: Path) -> Path:
    csvs = [p for p in root.rglob("*.csv") if p.is_file() and "outputs" not in p.parts]
    if not csvs:
        raise FileNotFoundError("Could not find raw training CSV")
    return max(csvs, key=lambda p: p.stat().st_size)


def find_val_daily(root: Path) -> Path:
    csvs = [p for p in root.rglob("*.csv") if p.is_file() and "outputs" not in p.parts and p.stat().st_size < 100_000]
    for path in csvs:
        try:
            sample = pd.read_csv(path, nrows=10)
        except Exception:
            continue
        if list(sample.columns) == ["date", "vessel_count"]:
            return path
    raise FileNotFoundError("Could not find validation daily vessel count CSV")


def compute_train_daily_vessels(train_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv, usecols=["mmsi", "time"], dtype={"mmsi": "string"}, parse_dates=["time"])
    out = df.groupby(df["time"].dt.normalize())["mmsi"].nunique().rename("vessel_count").reset_index()
    out = out.rename(columns={"time": "date"})
    return out


def load_labels(quality_run: Path, version: str, train_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv(quality_run / f"a_labels_{version}.csv", parse_dates=["hour"])
    b = pd.read_csv(quality_run / f"b_labels_{version}.csv", parse_dates=["hour"])
    a["task_key"] = a["region"]
    b["task_key"] = b["source_region"] + "->" + b["target_region"]
    return add_features(a, train_daily), add_features(b, train_daily)


def add_features(frame: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["hour"] = pd.to_datetime(out["hour"])
    out["date"] = out["hour"].dt.normalize()
    out["hour_of_day"] = out["hour"].dt.hour
    out["dayofweek"] = out["hour"].dt.dayofweek
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    return out.merge(daily[["date", "vessel_count"]], on="date", how="left")


def future_grid(task: str, val_daily: pd.DataFrame) -> pd.DataFrame:
    val_daily = val_daily.copy()
    val_daily["date"] = pd.to_datetime(val_daily["date"]).dt.normalize()
    hours = pd.date_range(val_daily["date"].min(), val_daily["date"].max() + pd.Timedelta(hours=23), freq="h")
    if task == "A":
        frame = pd.MultiIndex.from_product([hours, REGION_ORDER], names=["hour", "region"]).to_frame(index=False)
        frame["task_key"] = frame["region"]
    else:
        frame = pd.MultiIndex.from_tuples(
            [(h, s, t) for h in hours for s, t in PAIRS],
            names=["hour", "source_region", "target_region"],
        ).to_frame(index=False)
        frame["task_key"] = frame["source_region"] + "->" + frame["target_region"]
    return add_features(frame, val_daily)


def key_cols(task: str) -> list[str]:
    return ["region"] if task == "A" else ["source_region", "target_region"]


def specs(task: str) -> list[ForecastSpec]:
    out: list[ForecastSpec] = []
    vessel_powers = [0.0, 0.25, 0.5, 0.75, 1.0] if task == "A" else [0.0, 0.25, 0.5]
    for vp in vessel_powers:
        out.append(ForecastSpec(f"hour_mean_vp{vp:g}", "hour_mean", vessel_power=vp))
        out.append(ForecastSpec(f"dow_hour_mean_vp{vp:g}", "hour_mean", vessel_power=vp, use_dow=True))
        for days in [3, 5, 7, 10, 14]:
            out.append(ForecastSpec(f"recent{days}_hour_vp{vp:g}", "hour_mean", recent_days=days, vessel_power=vp))
    # Structure: predict daily total per key, allocate by hour shape.
    for vp in vessel_powers:
        for days in [None, 7, 14]:
            suffix = "all" if days is None else f"recent{days}"
            out.append(ForecastSpec(f"daily_total_shape_{suffix}_vp{vp:g}", "daily_total_shape", recent_days=days, vessel_power=vp))
            out.append(ForecastSpec(f"daily_total_shape_{suffix}_dow_vp{vp:g}", "daily_total_shape", recent_days=days, vessel_power=vp, use_dow=True))
    # Blend daily-total structure with hourly mean.
    for blend in [0.25, 0.5, 0.75]:
        out.append(ForecastSpec(f"blend_shape_hour_b{blend:g}", "blend", recent_days=14, blend=blend, vessel_power=0.5 if task == "A" else 0.25))
    # Shrink sparse B directions; also useful for A spikes.
    for shrink in ([0.15, 0.3] if task == "A" else [0.25, 0.5, 0.75]):
        out.append(ForecastSpec(f"recent14_hour_shrink{shrink:g}", "hour_mean", recent_days=14, shrink=shrink))
    return list({s.name: s for s in out}.values())


def group_mean(df: pd.DataFrame, cols: list[str], value: str = "y", name: str = "pred") -> pd.DataFrame:
    return df.groupby(cols, observed=True)[value].mean().rename(name).reset_index().drop_duplicates(cols)


def linear_daily_prediction(train_daily_key: pd.DataFrame, future_daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in train_daily_key.groupby("task_key", observed=True):
        x = g["vessel_count"].to_numpy(float)
        y = g["daily_y"].to_numpy(float)
        if len(g) >= 3 and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope, intercept = 0.0, float(np.mean(y) if len(y) else 0)
        # Conservative blend between linear fit and mean to reduce tiny-sample instability.
        mean_y = float(np.mean(y) if len(y) else 0)
        for _, row in future_daily.iterrows():
            pred = 0.5 * (intercept + slope * row["vessel_count"]) + 0.5 * mean_y
            rows.append({"task_key": key, "date": row["date"], "daily_pred": max(0.0, pred)})
    return pd.DataFrame(rows)


def predict_hour_mean(train: pd.DataFrame, future: pd.DataFrame, task: str, spec: ForecastSpec) -> pd.Series:
    keys = key_cols(task)
    source = train
    if spec.recent_days is not None:
        cutoff = train["hour"].max() - pd.Timedelta(days=spec.recent_days)
        source = train[train["hour"] > cutoff]
    cols = keys + (["dayofweek"] if spec.use_dow else []) + ["hour_of_day"]
    mean = group_mean(source, cols)
    fallback_hour = group_mean(train, keys + ["hour_of_day"], name="fallback_hour")
    key_mean = group_mean(train, keys, name="key_mean")
    merged = (
        future.merge(mean, on=cols, how="left", validate="many_to_one")
        .merge(fallback_hour, on=keys + ["hour_of_day"], how="left", validate="many_to_one")
        .merge(key_mean, on=keys, how="left", validate="many_to_one")
    )
    pred = merged["pred"].fillna(merged["fallback_hour"]).fillna(merged["key_mean"]).fillna(train["y"].mean())
    pred = pd.Series(pred).reset_index(drop=True)
    if spec.vessel_power:
        scale = (future["vessel_count"].reset_index(drop=True) / train.drop_duplicates("date")["vessel_count"].mean()).fillna(1.0)
        pred = pred * np.power(scale, spec.vessel_power)
    if spec.shrink:
        key_pred = merged["key_mean"].fillna(train["y"].mean()).reset_index(drop=True)
        pred = (1 - spec.shrink) * pred + spec.shrink * key_pred
    return pred.clip(lower=0)


def predict_daily_total_shape(train: pd.DataFrame, future: pd.DataFrame, task: str, spec: ForecastSpec) -> pd.Series:
    keys = key_cols(task)
    source = train
    if spec.recent_days is not None:
        cutoff = train["hour"].max() - pd.Timedelta(days=spec.recent_days)
        source = train[train["hour"] > cutoff]

    train_daily_key = (
        source.groupby(["task_key", "date"], observed=True)
        .agg(daily_y=("y", "sum"), vessel_count=("vessel_count", "first"))
        .reset_index()
    )
    future_daily = future[["date", "vessel_count"]].drop_duplicates("date")
    daily_pred = linear_daily_prediction(train_daily_key, future_daily)

    shape_cols = ["task_key"] + (["dayofweek"] if spec.use_dow else []) + ["hour_of_day"]
    shape_base = source.copy()
    day_totals = shape_base.groupby(["task_key", "date"], observed=True)["y"].sum().rename("daily_y").reset_index()
    shape_base = shape_base.merge(day_totals, on=["task_key", "date"], how="left")
    shape_base["share"] = np.where(shape_base["daily_y"] > 0, shape_base["y"] / shape_base["daily_y"], np.nan)
    shape = shape_base.groupby(shape_cols, observed=True)["share"].mean().rename("share").reset_index()
    fallback = shape_base.groupby(["task_key", "hour_of_day"], observed=True)["share"].mean().rename("fallback_share").reset_index()
    merged = (
        future.merge(daily_pred, on=["task_key", "date"], how="left", validate="many_to_one")
        .merge(shape, on=shape_cols, how="left", validate="many_to_one")
        .merge(fallback, on=["task_key", "hour_of_day"], how="left", validate="many_to_one")
    )
    # If a daily total is zero in history, equal zero is fine. Missing shares fallback to 0.
    pred = merged["daily_pred"].fillna(0) * merged["share"].fillna(merged["fallback_share"]).fillna(0)
    pred = pd.Series(pred).reset_index(drop=True)
    if spec.vessel_power:
        scale = (future["vessel_count"].reset_index(drop=True) / train.drop_duplicates("date")["vessel_count"].mean()).fillna(1.0)
        pred = pred * np.power(scale, spec.vessel_power)
    return pred.clip(lower=0)


def predict(train: pd.DataFrame, future: pd.DataFrame, task: str, spec: ForecastSpec) -> pd.Series:
    future = future.copy().reset_index(drop=True)
    if spec.family == "hour_mean":
        pred = predict_hour_mean(train, future, task, spec)
    elif spec.family == "daily_total_shape":
        pred = predict_daily_total_shape(train, future, task, spec)
    elif spec.family == "blend":
        shape = predict_daily_total_shape(train, future, task, ForecastSpec("shape", "daily_total_shape", recent_days=spec.recent_days, vessel_power=spec.vessel_power))
        hour = predict_hour_mean(train, future, task, ForecastSpec("hour", "hour_mean", recent_days=spec.recent_days, vessel_power=spec.vessel_power))
        pred = spec.blend * shape + (1 - spec.blend) * hour
    else:
        raise ValueError(spec)
    if len(pred) != len(future):
        raise ValueError(f"{spec.name}: {len(pred)} predictions for {len(future)} rows")
    return pd.Series(pred).clip(lower=0)


def sse(y: pd.Series, p: pd.Series) -> float:
    yv = pd.Series(y).reset_index(drop=True).to_numpy(float)
    pv = pd.Series(p).reset_index(drop=True).to_numpy(float)
    return float(np.sum((yv - pv) ** 2))


def backtest(frame: pd.DataFrame, task: str, spec_list: list[ForecastSpec]) -> pd.DataFrame:
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
            p = predict(train, valid.drop(columns=["y"]), task, spec)
            rows.append({"task": task, "split": split, "method": spec.name, "sse": sse(valid["y"], p)})
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> pd.DataFrame:
    out = (
        bt.groupby("method")
        .agg(mean_sse=("sse", "mean"), median_sse=("sse", "median"), max_sse=("sse", "max"), std_sse=("sse", "std"))
        .reset_index()
    )
    out["robust_score"] = out["mean_sse"] + 0.15 * out["max_sse"] + 0.05 * out["std_sse"].fillna(0)
    return out.sort_values("robust_score")


def read_template(template_dir: Path, zip_name: str, csv_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(template_dir / zip_name) as z:
        with z.open(csv_name) as f:
            return pd.read_csv(f)


def write_official(run_dir: Path, template_dir: Path, a_pred: pd.DataFrame, b_pred: pd.DataFrame) -> None:
    a_template = read_template(template_dir, "提交结果1_区域活跃拖轮数量.zip", "提交结果1_区域活跃拖轮数量.csv")
    b_template = read_template(template_dir, "提交结果2_圈层间拖轮迁移量.zip", "提交结果2_圈层间拖轮迁移量.csv")
    a_template["time_dt"] = pd.to_datetime(a_template["time_window"])
    b_template["time_dt"] = pd.to_datetime(b_template["time_window"])

    a = a_pred.copy()
    a["time_dt"] = a["hour"]
    a["zone"] = a["region"].map(REGION_CN)
    af = a_template.drop(columns=["vessel_count"]).merge(
        a[["time_dt", "zone", "prediction_rounded"]], on=["time_dt", "zone"], how="left", validate="one_to_one"
    )
    if af["prediction_rounded"].isna().any():
        raise ValueError("A template mismatch")
    a_out = af[["time_window", "zone"]].copy()
    a_out["vessel_count"] = af["prediction_rounded"].astype(int)

    b = b_pred.copy()
    b["time_dt"] = b["hour"]
    b["source_zone"] = b["source_region"].map(REGION_CN)
    b["target_zone"] = b["target_region"].map(REGION_CN)
    bf = b_template.drop(columns=["vessel_count"]).merge(
        b[["time_dt", "source_zone", "target_zone", "prediction_rounded"]],
        on=["time_dt", "source_zone", "target_zone"],
        how="left",
        validate="one_to_one",
    )
    if bf["prediction_rounded"].isna().any():
        raise ValueError("B template mismatch")
    b_out = bf[["time_window", "source_zone", "target_zone"]].copy()
    b_out["vessel_count"] = bf["prediction_rounded"].astype(int)

    for data, csv_name, zip_name in [
        (a_out, "提交结果1_区域活跃拖轮数量.csv", "提交结果1_区域活跃拖轮数量.zip"),
        (b_out, "提交结果2_圈层间拖轮迁移量.csv", "提交结果2_圈层间拖轮迁移量.zip"),
    ]:
        csv_path = run_dir / csv_name
        zip_path = run_dir / zip_name
        data.to_csv(csv_path, index=False, encoding="utf-8-sig")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(csv_path, arcname=csv_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--label-version", default="raw")
    args = parser.parse_args()

    root = find_project_root(args.root)
    quality = latest_quality_run(root)
    train_csv = find_train_csv(root)
    val_daily_path = find_val_daily(root)
    train_daily = compute_train_daily_vessels(train_csv)
    val_daily = pd.read_csv(val_daily_path, parse_dates=["date"])
    a, b = load_labels(quality, args.label_version, train_daily)

    run_dir = root / "outputs" / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}_train_internal_forecast"
    run_dir.mkdir(parents=True, exist_ok=True)

    a_specs = specs("A")
    b_specs = specs("B")
    bt_a = backtest(a, "A", a_specs)
    bt_b = backtest(b, "B", b_specs)
    summary_a = summarize(bt_a)
    summary_b = summarize(bt_b)
    best_a = summary_a.iloc[0]["method"]
    best_b = summary_b.iloc[0]["method"]
    spec_a = {s.name: s for s in a_specs}[best_a]
    spec_b = {s.name: s for s in b_specs}[best_b]

    fa = future_grid("A", val_daily)
    fb = future_grid("B", val_daily)
    fa["prediction"] = predict(a, fa, "A", spec_a)
    fb["prediction"] = predict(b, fb, "B", spec_b)
    fa["prediction_rounded"] = fa["prediction"].round().clip(lower=0).astype(int)
    fb["prediction_rounded"] = fb["prediction"].round().clip(lower=0).astype(int)

    bt_a.to_csv(run_dir / "backtest_A_detail.csv", index=False, encoding="utf-8-sig")
    bt_b.to_csv(run_dir / "backtest_B_detail.csv", index=False, encoding="utf-8-sig")
    summary_a.to_csv(run_dir / "backtest_A_summary.csv", index=False, encoding="utf-8-sig")
    summary_b.to_csv(run_dir / "backtest_B_summary.csv", index=False, encoding="utf-8-sig")
    train_daily.to_csv(run_dir / "train_daily_vessel_count.csv", index=False, encoding="utf-8-sig")
    fa.to_csv(run_dir / "submit_A_debug.csv", index=False, encoding="utf-8-sig")
    fb.to_csv(run_dir / "submit_B_debug.csv", index=False, encoding="utf-8-sig")
    write_official(run_dir, root / "outputs" / "_official_examples_backup", fa, fb)

    metadata = {
        "quality_run": str(quality),
        "train_csv": str(train_csv),
        "val_daily_path": str(val_daily_path),
        "best_A_method": str(best_a),
        "best_B_method": str(best_b),
        "A_mean_sse": float(summary_a.iloc[0]["mean_sse"]),
        "B_mean_sse": float(summary_b.iloc[0]["mean_sse"]),
        "combined_mean_score": float(summary_a.iloc[0]["mean_sse"] + 3 * summary_b.iloc[0]["mean_sse"]),
        "A_sum": int(fa["prediction_rounded"].sum()),
        "B_sum": int(fb["prediction_rounded"].sum()),
        "A_rows": int(len(fa)),
        "B_rows": int(len(fb)),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
