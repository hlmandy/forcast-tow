from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REGION_CN = {"core": "核心区", "near": "近港区", "outer": "外围区"}
CN_REGION = {v: k for k, v in REGION_CN.items()}

VARIANTS = [
    {
        "dir": "submit_01_A100_B085_anchor",
        "constraint_level": "none",
        "blend_to_daily_constraint": 0.0,
        "reason": "Current online-component anchor: old A100 shape plus B085.",
    },
    {
        "dir": "submit_02_A_daily_total_constraint25_B085",
        "constraint_level": "daily_total",
        "blend_to_daily_constraint": 0.25,
        "reason": "Move each daily A total 25% toward validation-vessel-count regression target.",
    },
    {
        "dir": "submit_03_A_daily_total_constraint50_B085",
        "constraint_level": "daily_total",
        "blend_to_daily_constraint": 0.50,
        "reason": "Move each daily A total 50% toward validation-vessel-count regression target.",
    },
]


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "outputs").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not find project root")


def find_val_daily(root: Path) -> Path:
    for path in root.rglob("*.csv"):
        if "outputs" in path.parts:
            continue
        if path.stat().st_size > 100_000:
            continue
        try:
            sample = pd.read_csv(path, nrows=5)
        except Exception:
            continue
        if list(sample.columns) == ["date", "vessel_count"]:
            return path
    raise FileNotFoundError("Could not find validation daily vessel_count CSV")


def find_train_csv(root: Path) -> Path:
    candidates = [p for p in root.rglob("*.csv") if "outputs" not in p.parts and p.stat().st_size > 1_000_000]
    if not candidates:
        raise FileNotFoundError("Could not find training AIS CSV")
    return max(candidates, key=lambda p: p.stat().st_size)


def train_daily_vessels(train_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv, usecols=["mmsi", "time"], dtype={"mmsi": "string"}, parse_dates=["time"])
    return (
        df.groupby(df["time"].dt.normalize())["mmsi"]
        .nunique()
        .rename("vessel_count")
        .reset_index()
        .rename(columns={"time": "date"})
    )


def date_region_targets(root: Path, val_daily: pd.DataFrame, train_daily: pd.DataFrame) -> pd.DataFrame:
    labels = pd.read_csv(root / "outputs" / "20260709_112307_quality_cleaning" / "a_labels_raw.csv", parse_dates=["hour"])
    labels["date"] = labels["hour"].dt.normalize()
    train = (
        labels.groupby(["date", "region"], observed=True)["y"]
        .sum()
        .rename("daily_region_y")
        .reset_index()
        .merge(train_daily, on="date", how="left", validate="many_to_one")
    )
    val_dates = val_daily.copy()
    val_dates["date"] = pd.to_datetime(val_dates["date"]).dt.normalize()
    rows = []
    for region, group in train.groupby("region", observed=True):
        x = group["vessel_count"].to_numpy(float)
        y = group["daily_region_y"].to_numpy(float)
        if len(group) >= 3 and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope, intercept = 0.0, float(y.mean())
        mean_y = float(y.mean())
        for _, row in val_dates.iterrows():
            linear = intercept + slope * float(row["vessel_count"])
            # Shrink the regression because the train correlation is only moderate.
            target = max(0.0, 0.5 * linear + 0.5 * mean_y)
            rows.append(
                {
                    "date": row["date"],
                    "region": region,
                    "target_daily_region_y": target,
                    "regression_slope": float(slope),
                    "train_region_mean": mean_y,
                }
            )
    return pd.DataFrame(rows)


def daily_total_targets(root: Path, val_daily: pd.DataFrame, train_daily: pd.DataFrame) -> pd.DataFrame:
    labels = pd.read_csv(root / "outputs" / "20260709_112307_quality_cleaning" / "a_labels_raw.csv", parse_dates=["hour"])
    labels["date"] = labels["hour"].dt.normalize()
    train = (
        labels.groupby("date", observed=True)["y"]
        .sum()
        .rename("daily_y")
        .reset_index()
        .merge(train_daily, on="date", how="left", validate="one_to_one")
    )
    x = train["vessel_count"].to_numpy(float)
    y = train["daily_y"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    val = val_daily.copy()
    val["date"] = pd.to_datetime(val["date"]).dt.normalize()
    val["target_daily_y"] = (intercept + slope * val["vessel_count"].astype(float)).clip(lower=0)
    val["regression_slope"] = float(slope)
    val["train_daily_mean"] = float(y.mean())
    return val[["date", "target_daily_y", "regression_slope", "train_daily_mean"]]


def largest_remainder_round(values: pd.Series, target_total: int) -> pd.Series:
    raw = values.clip(lower=0).to_numpy(float)
    floor = np.floor(raw).astype(int)
    remainder = raw - floor
    need = int(target_total - floor.sum())
    if need > 0:
        order = np.argsort(-remainder)
        floor[order[:need]] += 1
    elif need < 0:
        order = np.argsort(remainder)
        removable = np.where(floor[order] > 0)[0]
        floor[order[removable[: -need]]] -= 1
    return pd.Series(floor, index=values.index)


def constrain_a(old_a: pd.DataFrame, targets: pd.DataFrame, blend: float) -> pd.DataFrame:
    out = old_a.copy()
    out["time_dt"] = pd.to_datetime(out["time_window"])
    out["date"] = out["time_dt"].dt.normalize()
    out["region"] = out["zone"].map(CN_REGION)
    current = (
        out.groupby(["date", "region"], observed=True)["vessel_count"]
        .sum()
        .rename("current_total")
        .reset_index()
    )
    scaling = current.merge(targets, on=["date", "region"], how="left", validate="one_to_one")
    scaling["final_total"] = (1 - blend) * scaling["current_total"] + blend * scaling["target_daily_region_y"]
    scaling["final_total_int"] = scaling["final_total"].round().clip(lower=0).astype(int)
    out = out.merge(scaling[["date", "region", "current_total", "final_total_int"]], on=["date", "region"], how="left", validate="many_to_one")
    out["scaled"] = np.where(
        out["current_total"] > 0,
        out["vessel_count"] * out["final_total_int"] / out["current_total"],
        0.0,
    )
    pieces = []
    for _, group in out.groupby(["date", "region"], observed=True):
        group = group.copy()
        group["vessel_count"] = largest_remainder_round(group["scaled"], int(group["final_total_int"].iloc[0]))
        pieces.append(group)
    final = pd.concat(pieces, ignore_index=True).sort_values(["time_dt", "zone"])
    return final[["time_window", "zone", "vessel_count"]]


def constrain_a_daily_total(old_a: pd.DataFrame, targets: pd.DataFrame, blend: float) -> pd.DataFrame:
    out = old_a.copy()
    out["time_dt"] = pd.to_datetime(out["time_window"])
    out["date"] = out["time_dt"].dt.normalize()
    current = out.groupby("date", observed=True)["vessel_count"].sum().rename("current_total").reset_index()
    scaling = current.merge(targets, on="date", how="left", validate="one_to_one")
    scaling["final_total"] = (1 - blend) * scaling["current_total"] + blend * scaling["target_daily_y"]
    scaling["final_total_int"] = scaling["final_total"].round().clip(lower=0).astype(int)
    out = out.merge(scaling[["date", "current_total", "final_total_int"]], on="date", how="left", validate="many_to_one")
    out["scaled"] = np.where(
        out["current_total"] > 0,
        out["vessel_count"] * out["final_total_int"] / out["current_total"],
        0.0,
    )
    pieces = []
    for _, group in out.groupby("date", observed=True):
        group = group.copy()
        group["vessel_count"] = largest_remainder_round(group["scaled"], int(group["final_total_int"].iloc[0]))
        pieces.append(group)
    final = pd.concat(pieces, ignore_index=True).sort_values(["time_dt", "zone"])
    return final[["time_window", "zone", "vessel_count"]]


def validate_csv(df: pd.DataFrame, expected_rows: int, cols: list[str]) -> None:
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(df)}")
    if list(df.columns) != cols:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    vc = df["vessel_count"]
    if not ((vc >= 0).all() and (vc == vc.astype(int)).all()):
        raise ValueError("vessel_count must be non-negative integer")


def main() -> None:
    root = find_project_root(Path.cwd())
    source_dir = root / "outputs" / "_model_sources"
    out_dir = root / "outputs" / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}_daily_count_constrained_csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_daily = pd.read_csv(find_val_daily(root), parse_dates=["date"])
    train_daily = train_daily_vessels(find_train_csv(root))
    region_targets = date_region_targets(root, val_daily, train_daily)
    total_targets = daily_total_targets(root, val_daily, train_daily)
    old_a = pd.read_csv(source_dir / "A100_anchor_区域活跃拖轮数量.csv")
    b_csv = pd.read_csv(source_dir / "B085_anchor_圈层间拖轮迁移量.csv")

    validate_csv(old_a, 504, ["time_window", "zone", "vessel_count"])
    validate_csv(b_csv, 1008, ["time_window", "source_zone", "target_zone", "vessel_count"])

    summary = []
    for variant in VARIANTS:
        run_dir = out_dir / variant["dir"]
        run_dir.mkdir()
        if variant["constraint_level"] == "daily_total":
            a_csv = constrain_a_daily_total(old_a, total_targets, variant["blend_to_daily_constraint"])
        elif variant["constraint_level"] == "none":
            a_csv = old_a.copy()
        else:
            a_csv = constrain_a(old_a, region_targets, variant["blend_to_daily_constraint"])
        validate_csv(a_csv, 504, ["time_window", "zone", "vessel_count"])
        a_csv.to_csv(run_dir / "提交结果1_区域活跃拖轮数量.csv", index=False, encoding="utf-8-sig")
        b_csv.to_csv(run_dir / "提交结果2_圈层间拖轮迁移量.csv", index=False, encoding="utf-8-sig")
        meta = {
            **variant,
            "A_sum": int(a_csv["vessel_count"].sum()),
            "B_sum": int(b_csv["vessel_count"].sum()),
            "output_format": "csv-only",
            "uses_validation_daily_vessel_count": True,
        }
        (run_dir / "variant_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append({k: meta[k] for k in ["dir", "constraint_level", "blend_to_daily_constraint", "A_sum", "B_sum", "reason"]})

    region_targets.to_csv(out_dir / "daily_region_targets.csv", index=False, encoding="utf-8-sig")
    total_targets.to_csv(out_dir / "daily_total_targets.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(out_dir / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    print(out_dir)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
