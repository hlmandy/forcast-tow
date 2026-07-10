from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import train_internal_forecast as tif


def last_week_shift(
    labels: pd.DataFrame,
    future: pd.DataFrame,
    task: str,
    scale_power: float = 1.0,
    blend_hour: float = 0.0,
) -> pd.Series:
    train = labels.copy()
    keys = tif.key_cols(task)
    train["future_hour"] = train["hour"] + pd.Timedelta(days=7)
    lag = train[keys + ["future_hour", "y", "vessel_count"]].rename(
        columns={"future_hour": "hour", "y": "lag_y", "vessel_count": "lag_vessel_count"}
    )
    pred_frame = future.merge(lag, on=keys + ["hour"], how="left", validate="one_to_one")
    hour_mean = (
        train.groupby(keys + ["hour_of_day"], observed=True)["y"]
        .mean()
        .rename("hour_mean")
        .reset_index()
        .drop_duplicates(keys + ["hour_of_day"])
    )
    pred_frame = pred_frame.merge(hour_mean, on=keys + ["hour_of_day"], how="left", validate="many_to_one")
    pred = pred_frame["lag_y"].fillna(pred_frame["hour_mean"]).fillna(train["y"].mean())
    if scale_power:
        scale = (pred_frame["vessel_count"] / pred_frame["lag_vessel_count"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        pred = pred * np.power(scale, scale_power)
    if blend_hour:
        pred = (1 - blend_hour) * pred + blend_hour * pred_frame["hour_mean"].fillna(train["y"].mean())
    return pd.Series(pred).clip(lower=0).reset_index(drop=True)


def read_template(template_dir: Path, zip_name: str, csv_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(template_dir / zip_name) as z:
        with z.open(csv_name) as f:
            return pd.read_csv(f)


def write_official(run_dir: Path, template_dir: Path, a_pred: pd.DataFrame, b_pred: pd.DataFrame, float_output: bool) -> None:
    a_t = read_template(template_dir, "提交结果1_区域活跃拖轮数量.zip", "提交结果1_区域活跃拖轮数量.csv")
    b_t = read_template(template_dir, "提交结果2_圈层间拖轮迁移量.zip", "提交结果2_圈层间拖轮迁移量.csv")
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
    b_out = bf[["time_window", "source_zone", "target_zone"]].copy()
    if float_output:
        a_out["vessel_count"] = af["prediction"].round(6)
        b_out["vessel_count"] = bf["prediction"].round(6)
    else:
        a_out["vessel_count"] = af["prediction"].round().astype(int)
        b_out["vessel_count"] = bf["prediction"].round().astype(int)

    for df, csv_name, zip_name in [
        (a_out, "提交结果1_区域活跃拖轮数量.csv", "提交结果1_区域活跃拖轮数量.zip"),
        (b_out, "提交结果2_圈层间拖轮迁移量.csv", "提交结果2_圈层间拖轮迁移量.zip"),
    ]:
        csv_path = run_dir / csv_name
        zip_path = run_dir / zip_name
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(csv_path, arcname=csv_name)


def main() -> None:
    root = tif.find_project_root(Path.cwd())
    quality = tif.latest_quality_run(root)
    train_csv = tif.find_train_csv(root)
    train_daily = tif.compute_train_daily_vessels(train_csv)
    val_daily = pd.read_csv(tif.find_val_daily(root), parse_dates=["date"])
    a, b = tif.load_labels(quality, "raw", train_daily)
    fa = tif.future_grid("A", val_daily)
    fb = tif.future_grid("B", val_daily)
    template_dir = root / "outputs" / "_official_examples_backup"
    base_dir = root / "outputs" / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}_last_week_shift_candidates"
    base_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("lastweek_A_scale1_B_scale1_float", 1.0, 1.0, 0.0, True),
        ("lastweek_A_scale1_blend25_B_scale1_float", 1.0, 1.0, 0.25, True),
        ("lastweek_A_scale075_B_scale095_float", 0.75, 0.95, 0.25, True),
        ("lastweek_A_scale1_blend25_B_scale1_int", 1.0, 1.0, 0.25, False),
    ]
    rows = []
    for name, a_power, b_power, a_blend, float_output in variants:
        run_dir = base_dir / name
        run_dir.mkdir(exist_ok=True)
        ap = fa.copy()
        bp = fb.copy()
        ap["prediction"] = last_week_shift(a, fa, "A", scale_power=a_power, blend_hour=a_blend)
        bp["prediction"] = last_week_shift(b, fb, "B", scale_power=b_power, blend_hour=0.0)
        ap.to_csv(run_dir / "submit_A_debug.csv", index=False, encoding="utf-8-sig")
        bp.to_csv(run_dir / "submit_B_debug.csv", index=False, encoding="utf-8-sig")
        write_official(run_dir, template_dir, ap, bp, float_output=float_output)
        meta = {
            "name": name,
            "A_scale_power": a_power,
            "B_scale_power": b_power,
            "A_blend_hour": a_blend,
            "float_output": float_output,
            "A_sum": float(ap["prediction"].sum()),
            "B_sum": float(bp["prediction"].sum()),
        }
        (run_dir / "variant_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(meta)

    pd.DataFrame(rows).to_csv(base_dir / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    readme = """# Last Week Shift Candidates

These candidates copy the last training week (2018-01-18 to 2018-01-24)
to the validation week (2018-01-25 to 2018-01-31), with optional scaling by
the provided daily vessel counts.

This is a calendar-structure candidate, not the best average CV method.
It is useful because the validation window is exactly the next week after
training.
"""
    (base_dir / "README.md").write_text(readme, encoding="utf-8")
    print(base_dir)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
