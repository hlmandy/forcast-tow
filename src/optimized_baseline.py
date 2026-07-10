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
REGION_CN = {
    "core": "核心区",
    "near": "近港区",
    "outer": "外围区",
}
CN_TO_REGION = {v: k for k, v in REGION_CN.items()}
PAIRS = [(s, t) for s in REGION_ORDER for t in REGION_ORDER if s != t]

CENTER_LON = 117.79
CENTER_LAT = 38.97
KM_PER_LAT = 111.32
KM_PER_LON = 111.32 * math.cos(math.radians(CENTER_LAT))


@dataclass(frozen=True)
class MethodSpec:
    name: str
    base: str
    recent_days: int | None = None
    scale_power: float = 0.0
    alpha: float = 0.75


def find_project_root(start: Path) -> Path:
    candidates = [start.resolve(), *start.resolve().parents, Path(r"C:\Users\mandy\hl_Documents\2026CTS")]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "outputs").exists() and ((candidate / "notebooks").exists() or (candidate / "PLAN.md").exists()):
            return candidate
    raise FileNotFoundError("Could not find project root")


def find_data_files(root: Path) -> tuple[Path, Path]:
    csv_files = [p for p in root.rglob("*.csv") if p.is_file()]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {root}")

    # The raw AIS training file is by far the largest CSV. Generated label and
    # submission files also live under outputs/, so do not use path names alone
    # to identify the small validation hint file.
    train_candidates = [p for p in csv_files if "outputs" not in p.parts]
    train_path = sorted(train_candidates or csv_files, key=lambda p: p.stat().st_size, reverse=True)[0]

    val_candidates: list[Path] = []
    for path in csv_files:
        if path == train_path or path.stat().st_size >= 100_000:
            continue
        try:
            sample = pd.read_csv(path, nrows=20)
        except Exception:
            continue
        if list(sample.columns) == ["date", "vessel_count"] and 1 <= len(sample) <= 10:
            val_candidates.append(path)
    if not val_candidates:
        raise FileNotFoundError("Could not find validation daily vessel count CSV")

    val_candidates = sorted(
        val_candidates,
        key=lambda p: (
            "outputs" in p.parts,
            "验证集" not in str(p),
            len(p.parts),
            str(p),
        ),
    )
    return train_path, val_candidates[0]


def add_regions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dx = (out["x"] - CENTER_LON) * KM_PER_LON
    dy = (out["y"] - CENTER_LAT) * KM_PER_LAT
    out["dist_km"] = np.sqrt(dx * dx + dy * dy)
    out["region"] = np.select(
        [
            out["dist_km"] < 3,
            out["dist_km"].between(3, 10, inclusive="left"),
            out["dist_km"].between(10, 30, inclusive="left"),
        ],
        REGION_ORDER,
        default="outside",
    )
    return out


def load_data(train_path: Path, val_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usecols = ["mmsi", "x", "y", "sog", "time"]
    dtypes = {"mmsi": "string", "x": "float64", "y": "float64", "sog": "float32"}
    df = pd.read_csv(train_path, usecols=usecols, dtype=dtypes, parse_dates=["time"])
    df["hour"] = df["time"].dt.floor("h")
    df["date"] = df["time"].dt.normalize()
    df["hour_of_day"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek
    df = add_regions(df)
    val_daily = pd.read_csv(val_path, parse_dates=["date"])
    val_daily["date"] = val_daily["date"].dt.normalize()
    train_daily = df.groupby("date")["mmsi"].nunique().rename("vessel_count").reset_index()
    return df, train_daily, val_daily


def make_a_labels(df: pd.DataFrame) -> pd.DataFrame:
    active = df[df["region"].isin(REGION_ORDER) & df["sog"].between(2, 10, inclusive="both")]
    ship_region_counts = (
        active.groupby(["hour", "region", "mmsi"], observed=True)
        .size()
        .rename("ais_points")
        .reset_index()
    )
    qualified = ship_region_counts[ship_region_counts["ais_points"] >= 3]
    labels = (
        qualified.groupby(["hour", "region"], observed=True)["mmsi"]
        .nunique()
        .rename("y")
        .reset_index()
    )
    hours = pd.date_range(df["hour"].min(), df["hour"].max(), freq="h")
    full_index = pd.MultiIndex.from_product([hours, REGION_ORDER], names=["hour", "region"])
    labels = labels.set_index(["hour", "region"]).reindex(full_index, fill_value=0).reset_index()
    labels["task_key"] = labels["region"]
    return labels


def make_b_labels(df: pd.DataFrame) -> pd.DataFrame:
    in_region = df[df["region"].isin(REGION_ORDER)]
    agg = (
        in_region.groupby(["mmsi", "hour", "region"], observed=True)
        .agg(points=("time", "size"), last_time=("time", "max"))
        .reset_index()
    )
    representative = (
        agg.sort_values(["mmsi", "hour", "points", "last_time"], ascending=[True, True, False, False])
        .drop_duplicates(["mmsi", "hour"], keep="first")
        .sort_values(["mmsi", "hour"])
        .rename(columns={"region": "source_region"})
    )
    representative["next_hour"] = representative.groupby("mmsi")["hour"].shift(-1)
    representative["target_region"] = representative.groupby("mmsi")["source_region"].shift(-1)
    moved = representative[
        (representative["next_hour"] == representative["hour"] + pd.Timedelta(hours=1))
        & (representative["source_region"].astype(str) != representative["target_region"].astype(str))
    ]
    labels = (
        moved.groupby(["hour", "source_region", "target_region"], observed=True)["mmsi"]
        .nunique()
        .rename("y")
        .reset_index()
    )
    hours = pd.date_range(df["hour"].min(), df["hour"].max(), freq="h")
    full_index = pd.MultiIndex.from_tuples(
        [(h, s, t) for h in hours for s, t in PAIRS],
        names=["hour", "source_region", "target_region"],
    )
    labels = labels.set_index(["hour", "source_region", "target_region"]).reindex(full_index, fill_value=0).reset_index()
    labels["task_key"] = labels["source_region"] + "->" + labels["target_region"]
    return labels


def add_time_features(frame: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = out["hour"].dt.normalize()
    out["hour_of_day"] = out["hour"].dt.hour
    out["dayofweek"] = out["hour"].dt.dayofweek
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    hint = daily.copy()
    hint["date"] = pd.to_datetime(hint["date"]).dt.normalize()
    return out.merge(hint[["date", "vessel_count"]], on="date", how="left")


def key_cols(task: str) -> list[str]:
    return ["region"] if task == "A" else ["source_region", "target_region"]


def future_grid(task: str, val_daily: pd.DataFrame) -> pd.DataFrame:
    start = val_daily["date"].min()
    end = val_daily["date"].max() + pd.Timedelta(hours=23)
    hours = pd.date_range(start, end, freq="h")
    if task == "A":
        frame = pd.MultiIndex.from_product([hours, REGION_ORDER], names=["hour", "region"]).to_frame(index=False)
        frame["task_key"] = frame["region"]
    else:
        frame = pd.MultiIndex.from_tuples(
            [(h, s, t) for h in hours for s, t in PAIRS],
            names=["hour", "source_region", "target_region"],
        ).to_frame(index=False)
        frame["task_key"] = frame["source_region"] + "->" + frame["target_region"]
    return add_time_features(frame, val_daily)


def method_specs() -> list[MethodSpec]:
    specs: list[MethodSpec] = [
        MethodSpec("global_mean", "global"),
        MethodSpec("hour_mean", "hour"),
        MethodSpec("dow_hour_mean", "dow_hour"),
        MethodSpec("recent_7d_hour", "recent_hour", recent_days=7),
        MethodSpec("recent_14d_hour", "recent_hour", recent_days=14),
        MethodSpec("mixed_050_hour_dow", "mixed", alpha=0.50),
        MethodSpec("mixed_075_hour_dow", "mixed", alpha=0.75),
    ]
    for base, recent_days in [("hour", None), ("recent_hour", 7), ("recent_hour", 14)]:
        base_name = "hour" if base == "hour" else f"recent_{recent_days}d_hour"
        for power in [0.25, 0.5, 0.75, 1.0]:
            specs.append(MethodSpec(f"{base_name}_scale_p{str(power).replace('.', '')}", base, recent_days=recent_days, scale_power=power))
    return specs


def predict_base(train: pd.DataFrame, future: pd.DataFrame, task: str, spec: MethodSpec) -> pd.Series:
    cols_key = key_cols(task)
    train = train.copy()
    future = future.copy().reset_index(drop=True)
    group_mean = train.groupby(cols_key, observed=True)["y"].mean().rename("key_mean").reset_index().drop_duplicates(cols_key)
    global_mean = train["y"].mean()

    if spec.base == "global":
        merged = future.merge(group_mean, on=cols_key, how="left", validate="many_to_one")
        pred = merged["key_mean"].fillna(global_mean)
    elif spec.base == "hour":
        cols = cols_key + ["hour_of_day"]
        mean = train.groupby(cols, observed=True)["y"].mean().rename("pred").reset_index().drop_duplicates(cols)
        merged = future.merge(mean, on=cols, how="left", validate="many_to_one").merge(group_mean, on=cols_key, how="left", validate="many_to_one")
        pred = merged["pred"].fillna(merged["key_mean"]).fillna(global_mean)
    elif spec.base == "dow_hour":
        cols = cols_key + ["dayofweek", "hour_of_day"]
        mean = train.groupby(cols, observed=True)["y"].mean().rename("pred").reset_index().drop_duplicates(cols)
        merged = future.merge(mean, on=cols, how="left", validate="many_to_one").merge(group_mean, on=cols_key, how="left", validate="many_to_one")
        pred = merged["pred"].fillna(merged["key_mean"]).fillna(global_mean)
    elif spec.base == "recent_hour":
        assert spec.recent_days is not None
        cutoff = train["hour"].max() - pd.Timedelta(days=spec.recent_days)
        recent = train[train["hour"] > cutoff]
        cols = cols_key + ["hour_of_day"]
        mean = recent.groupby(cols, observed=True)["y"].mean().rename("pred").reset_index().drop_duplicates(cols)
        merged = future.merge(mean, on=cols, how="left", validate="many_to_one").merge(group_mean, on=cols_key, how="left", validate="many_to_one")
        pred = merged["pred"].fillna(merged["key_mean"]).fillna(global_mean)
    elif spec.base == "mixed":
        hour = predict_base(train, future, task, MethodSpec("hour_mean", "hour"))
        dow_hour = predict_base(train, future, task, MethodSpec("dow_hour_mean", "dow_hour"))
        pred = spec.alpha * hour + (1 - spec.alpha) * dow_hour
    else:
        raise ValueError(spec)

    pred = pd.Series(pred).reset_index(drop=True)
    if spec.scale_power:
        train_daily_mean = train.drop_duplicates("date")["vessel_count"].mean()
        scale = (future["vessel_count"].reset_index(drop=True) / train_daily_mean).fillna(1.0)
        pred = pred * np.power(scale, spec.scale_power)
    if len(pred) != len(future):
        raise ValueError(f"{spec.name} produced {len(pred)} predictions for {len(future)} rows")
    return pred.clip(lower=0)


def sse(y: pd.Series, pred: pd.Series) -> float:
    y_arr = pd.Series(y).reset_index(drop=True).to_numpy()
    p_arr = pd.Series(pred).reset_index(drop=True).to_numpy()
    if len(y_arr) != len(p_arr):
        raise ValueError(f"Length mismatch: {len(y_arr)} vs {len(p_arr)}")
    return float(np.sum((y_arr - p_arr) ** 2))


def backtest(frame: pd.DataFrame, task: str, specs: list[MethodSpec]) -> pd.DataFrame:
    splits = [
        ("2018-01-01", "2018-01-11", "2018-01-17"),
        ("2018-01-01", "2018-01-18", "2018-01-24"),
        ("2018-01-08", "2018-01-18", "2018-01-24"),
    ]
    rows = []
    for train_start, valid_start, valid_end in splits:
        train_start_ts = pd.Timestamp(train_start)
        valid_start_ts = pd.Timestamp(valid_start)
        valid_end_excl = pd.Timestamp(valid_end) + pd.Timedelta(days=1)
        train = frame[(frame["hour"] >= train_start_ts) & (frame["hour"] < valid_start_ts)].copy()
        valid = frame[(frame["hour"] >= valid_start_ts) & (frame["hour"] < valid_end_excl)].copy()
        split_name = f"train {train_start}~{valid_start_ts - pd.Timedelta(days=1):%Y-%m-%d}, valid {valid_start}~{valid_end}"
        for spec in specs:
            pred = predict_base(train, valid.drop(columns=["y"]), task, spec)
            rows.append(
                {
                    "task": task,
                    "split": split_name,
                    "method": spec.name,
                    "sse": sse(valid["y"], pred),
                }
            )
    return pd.DataFrame(rows)


def read_template(template_dir: Path, zip_name: str, csv_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(template_dir / zip_name) as zf:
        with zf.open(csv_name) as fh:
            return pd.read_csv(fh)


def write_zip(csv_path: Path, zip_path: Path, arcname: str) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=arcname)


def build_official_outputs(
    run_dir: Path,
    template_dir: Path,
    a_pred: pd.DataFrame,
    b_pred: pd.DataFrame,
) -> None:
    a_template = read_template(template_dir, "提交结果1_区域活跃拖轮数量.zip", "提交结果1_区域活跃拖轮数量.csv")
    b_template = read_template(template_dir, "提交结果2_圈层间拖轮迁移量.zip", "提交结果2_圈层间拖轮迁移量.csv")
    a_template["time_dt"] = pd.to_datetime(a_template["time_window"])
    b_template["time_dt"] = pd.to_datetime(b_template["time_window"])

    a_pred = a_pred.copy()
    a_pred["time_dt"] = a_pred["hour"]
    a_pred["zone"] = a_pred["region"].map(REGION_CN)
    a_fill = a_template.drop(columns=["vessel_count"]).merge(
        a_pred[["time_dt", "zone", "prediction_rounded"]],
        on=["time_dt", "zone"],
        how="left",
        validate="one_to_one",
    )
    if a_fill["prediction_rounded"].isna().any():
        raise ValueError("A official template has unmatched rows")
    a_official = a_fill[["time_window", "zone"]].copy()
    a_official["vessel_count"] = a_fill["prediction_rounded"].astype(int)

    b_pred = b_pred.copy()
    b_pred["time_dt"] = b_pred["hour"]
    b_pred["source_zone"] = b_pred["source_region"].map(REGION_CN)
    b_pred["target_zone"] = b_pred["target_region"].map(REGION_CN)
    b_fill = b_template.drop(columns=["vessel_count"]).merge(
        b_pred[["time_dt", "source_zone", "target_zone", "prediction_rounded"]],
        on=["time_dt", "source_zone", "target_zone"],
        how="left",
        validate="one_to_one",
    )
    if b_fill["prediction_rounded"].isna().any():
        raise ValueError("B official template has unmatched rows")
    b_official = b_fill[["time_window", "source_zone", "target_zone"]].copy()
    b_official["vessel_count"] = b_fill["prediction_rounded"].astype(int)

    a_csv = run_dir / "提交结果1_区域活跃拖轮数量.csv"
    b_csv = run_dir / "提交结果2_圈层间拖轮迁移量.csv"
    a_zip = run_dir / "提交结果1_区域活跃拖轮数量.zip"
    b_zip = run_dir / "提交结果2_圈层间拖轮迁移量.zip"
    a_official.to_csv(a_csv, index=False, encoding="utf-8-sig")
    b_official.to_csv(b_csv, index=False, encoding="utf-8-sig")
    write_zip(a_csv, a_zip, "提交结果1_区域活跃拖轮数量.csv")
    write_zip(b_csv, b_zip, "提交结果2_圈层间拖轮迁移量.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = find_project_root(args.root)
    train_path, val_path = find_data_files(root)
    output_dir = root / "outputs"
    template_dir = output_dir / "_official_examples_backup"
    run_dir = output_dir / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_optimized")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"root={root}")
    print(f"train={train_path}")
    print(f"val_daily={val_path}")
    print(f"run_dir={run_dir}")

    df, train_daily, val_daily = load_data(train_path, val_path)
    a_frame = add_time_features(make_a_labels(df), train_daily)
    b_frame = add_time_features(make_b_labels(df), train_daily)
    specs = method_specs()

    bt_a = backtest(a_frame, "A", specs)
    bt_b = backtest(b_frame, "B", specs)
    summary_a = bt_a.groupby("method").agg(mean_sse=("sse", "mean"), median_sse=("sse", "median"), max_sse=("sse", "max")).sort_values("mean_sse")
    summary_b = bt_b.groupby("method").agg(mean_sse=("sse", "mean"), median_sse=("sse", "median"), max_sse=("sse", "max")).sort_values("mean_sse")

    best_a_name = str(summary_a.index[0])
    best_b_name = str(summary_b.index[0])
    spec_by_name = {spec.name: spec for spec in specs}
    best_a = spec_by_name[best_a_name]
    best_b = spec_by_name[best_b_name]
    combined_mean_score = float(summary_a.loc[best_a_name, "mean_sse"] + 3 * summary_b.loc[best_b_name, "mean_sse"])

    future_a = future_grid("A", val_daily)
    future_b = future_grid("B", val_daily)
    future_a["prediction"] = predict_base(a_frame, future_a, "A", best_a)
    future_b["prediction"] = predict_base(b_frame, future_b, "B", best_b)
    future_a["prediction_rounded"] = future_a["prediction"].round().astype(int)
    future_b["prediction_rounded"] = future_b["prediction"].round().astype(int)

    a_frame.to_csv(run_dir / "a_train_labels.csv", index=False, encoding="utf-8-sig")
    b_frame.to_csv(run_dir / "b_train_labels.csv", index=False, encoding="utf-8-sig")
    future_a.to_csv(run_dir / "submit_A_debug.csv", index=False, encoding="utf-8-sig")
    future_b.to_csv(run_dir / "submit_B_debug.csv", index=False, encoding="utf-8-sig")
    bt_a.to_csv(run_dir / "backtest_A_detail.csv", index=False, encoding="utf-8-sig")
    bt_b.to_csv(run_dir / "backtest_B_detail.csv", index=False, encoding="utf-8-sig")
    summary_a.to_csv(run_dir / "backtest_A_summary.csv", encoding="utf-8-sig")
    summary_b.to_csv(run_dir / "backtest_B_summary.csv", encoding="utf-8-sig")

    build_official_outputs(run_dir, template_dir, future_a, future_b)

    metadata = {
        "train_path": str(train_path),
        "val_daily_path": str(val_path),
        "best_A_method": best_a_name,
        "best_B_method": best_b_name,
        "mean_sse_A": float(summary_a.loc[best_a_name, "mean_sse"]),
        "mean_sse_B": float(summary_b.loc[best_b_name, "mean_sse"]),
        "combined_mean_score": combined_mean_score,
        "rows_A": int(len(future_a)),
        "rows_B": int(len(future_b)),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
