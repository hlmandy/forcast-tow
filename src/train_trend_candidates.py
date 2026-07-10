from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import train_internal_forecast as tif


REGION_CN = tif.REGION_CN
REGION_ORDER = tif.REGION_ORDER
PAIRS = tif.PAIRS


def recent_daily_shape_predict(
    train: pd.DataFrame,
    future: pd.DataFrame,
    task: str,
    recent_days: int,
    blend_with_hour: float = 0.0,
) -> pd.Series:
    keys = tif.key_cols(task)
    cutoff = train["date"].max() - pd.Timedelta(days=recent_days - 1)
    recent = train[train["date"] >= cutoff].copy()

    # Predict future daily total per key from recent daily average, scaled only
    # by validation daily vessel count relative to recent vessel count.
    daily_key = (
        recent.groupby(["task_key", "date"], observed=True)
        .agg(daily_y=("y", "sum"), vessel_count=("vessel_count", "first"))
        .reset_index()
    )
    recent_vessel_mean = recent.drop_duplicates("date")["vessel_count"].mean()
    future_daily = future[["date", "vessel_count"]].drop_duplicates("date")
    daily_rows = []
    for key, g in daily_key.groupby("task_key", observed=True):
        base = g["daily_y"].mean()
        for _, row in future_daily.iterrows():
            scale = row["vessel_count"] / recent_vessel_mean if recent_vessel_mean else 1.0
            daily_rows.append({"task_key": key, "date": row["date"], "daily_pred": max(0.0, base * scale)})
    daily_pred = pd.DataFrame(daily_rows)

    # Allocate daily total using recent average hour share per key.
    day_totals = recent.groupby(["task_key", "date"], observed=True)["y"].sum().rename("daily_y").reset_index()
    shape_base = recent.merge(day_totals, on=["task_key", "date"], how="left")
    shape_base["share"] = np.where(shape_base["daily_y"] > 0, shape_base["y"] / shape_base["daily_y"], np.nan)
    shape = shape_base.groupby(["task_key", "hour_of_day"], observed=True)["share"].mean().rename("share").reset_index()
    pred_frame = future.merge(daily_pred, on=["task_key", "date"], how="left").merge(
        shape, on=["task_key", "hour_of_day"], how="left"
    )
    pred = (pred_frame["daily_pred"].fillna(0) * pred_frame["share"].fillna(0)).reset_index(drop=True)

    if blend_with_hour:
        hour_spec = tif.ForecastSpec("hour_mean_vp1", "hour_mean", vessel_power=1.0)
        hour_pred = tif.predict(train, future, task, hour_spec).reset_index(drop=True)
        pred = (1 - blend_with_hour) * pred + blend_with_hour * hour_pred
    return pd.Series(pred).clip(lower=0)


def write_official(run_dir: Path, template_dir: Path, a_pred: pd.DataFrame, b_pred: pd.DataFrame) -> None:
    tif.write_official(run_dir, template_dir, a_pred, b_pred)


def main() -> None:
    root = tif.find_project_root(Path.cwd())
    quality = tif.latest_quality_run(root)
    train_csv = tif.find_train_csv(root)
    val_daily_path = tif.find_val_daily(root)
    train_daily = tif.compute_train_daily_vessels(train_csv)
    val_daily = pd.read_csv(val_daily_path, parse_dates=["date"])
    a, b = tif.load_labels(quality, "raw", train_daily)

    base_run = root / "outputs" / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}_train_trend_candidates"
    base_run.mkdir(parents=True, exist_ok=True)
    template_dir = root / "outputs" / "_official_examples_backup"

    fa_base = tif.future_grid("A", val_daily)
    fb_base = tif.future_grid("B", val_daily)
    b_spec = tif.ForecastSpec("recent14_hour_shrink0.25", "hour_mean", recent_days=14, shrink=0.25)
    b_pred = fb_base.copy()
    b_pred["prediction"] = tif.predict(b, fb_base, "B", b_spec)
    b_pred["prediction_rounded"] = b_pred["prediction"].round().clip(lower=0).astype(int)

    variants = [
        ("A_recent7_shape_B_recent14shrink", 7, 0.0),
        ("A_recent7_shape_blend25hour_B_recent14shrink", 7, 0.25),
        ("A_recent5_shape_blend25hour_B_recent14shrink", 5, 0.25),
    ]
    summary = []
    for name, recent_days, blend in variants:
        run_dir = base_run / name
        run_dir.mkdir(parents=True, exist_ok=True)
        a_pred = fa_base.copy()
        a_pred["prediction"] = recent_daily_shape_predict(a, fa_base, "A", recent_days=recent_days, blend_with_hour=blend)
        a_pred["prediction_rounded"] = a_pred["prediction"].round().clip(lower=0).astype(int)
        b_out = b_pred.copy()

        a_pred.to_csv(run_dir / "submit_A_debug.csv", index=False, encoding="utf-8-sig")
        b_out.to_csv(run_dir / "submit_B_debug.csv", index=False, encoding="utf-8-sig")
        write_official(run_dir, template_dir, a_pred, b_out)

        meta = {
            "name": name,
            "A_recent_days": recent_days,
            "A_blend_with_hour": blend,
            "B_method": b_spec.name,
            "A_sum": int(a_pred["prediction_rounded"].sum()),
            "B_sum": int(b_out["prediction_rounded"].sum()),
            "A_zip": str(run_dir / "提交结果1_区域活跃拖轮数量.zip"),
            "B_zip": str(run_dir / "提交结果2_圈层间拖轮迁移量.zip"),
        }
        (run_dir / "variant_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(meta)

    pd.DataFrame(summary).to_csv(base_run / "candidate_summary.csv", index=False, encoding="utf-8-sig")

    # Diagnostics for why these candidates exist.
    a_daily = a.groupby("date")["y"].sum().rename("A_total").reset_index().merge(train_daily, on="date")
    b_daily = b.groupby("date")["y"].sum().rename("B_total").reset_index().merge(train_daily, on="date")
    a_daily.to_csv(base_run / "train_A_daily_totals.csv", index=False, encoding="utf-8-sig")
    b_daily.to_csv(base_run / "train_B_daily_totals.csv", index=False, encoding="utf-8-sig")
    readme = f"""# Train Trend Candidates

These candidates are generated from training-set evidence only, without using online score calibration.

Key observation:

- A daily total last 7-day mean: {a_daily.tail(7)['A_total'].mean():.2f}
- A daily total full-period mean: {a_daily['A_total'].mean():.2f}
- B daily total last 7-day mean: {b_daily.tail(7)['B_total'].mean():.2f}
- B daily total full-period mean: {b_daily['B_total'].mean():.2f}

A shows a strong late-period upward trend while validation daily vessel counts are close to the last training week. Therefore these candidates use recent daily A totals plus recent hourly shape. B is comparatively stable, so it uses the best train-internal recent14 shrink method.
"""
    (base_run / "README.md").write_text(readme, encoding="utf-8")
    print(base_run)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
