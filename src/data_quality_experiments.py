from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REGION_ORDER = ["core", "near", "outer"]
PAIRS = [(s, t) for s in REGION_ORDER for t in REGION_ORDER if s != t]
CENTER_LON = 117.79
CENTER_LAT = 38.97
KM_PER_LAT = 111.32
KM_PER_LON = 111.32 * math.cos(math.radians(CENTER_LAT))


@dataclass(frozen=True)
class CleanSpec:
    name: str
    dedup_mmsi_time: bool = False
    max_step_knots: float | None = None
    max_sog: float | None = None
    drop_outside_for_labels_only: bool = True


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    candidates = [start, *start.parents, Path(r"C:\Users\mandy\hl_Documents\2026CTS")]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "outputs").exists() and ((candidate / "notebooks").exists() or (candidate / "PLAN.md").exists()):
            return candidate
    raise FileNotFoundError("Could not find project root")


def find_train_csv(root: Path) -> Path:
    csvs = [p for p in root.rglob("*.csv") if p.is_file() and "outputs" not in p.parts]
    if not csvs:
        raise FileNotFoundError("Could not find raw training CSV")
    return max(csvs, key=lambda p: p.stat().st_size)


def add_region(df: pd.DataFrame) -> pd.DataFrame:
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


def load_raw(train_path: Path) -> pd.DataFrame:
    usecols = ["mmsi", "x", "y", "sog", "cog", "true_heading", "rot", "time", "ship_type"]
    df = pd.read_csv(train_path, usecols=usecols, dtype={"mmsi": "string"}, parse_dates=["time"])
    df["hour"] = df["time"].dt.floor("h")
    df["date"] = df["time"].dt.normalize()
    df = add_region(df)
    return df


def add_trajectory_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["mmsi", "time", "x", "y"]).copy()
    out["prev_time"] = out.groupby("mmsi")["time"].shift()
    out["prev_x"] = out.groupby("mmsi")["x"].shift()
    out["prev_y"] = out.groupby("mmsi")["y"].shift()
    out["dt_s"] = (out["time"] - out["prev_time"]).dt.total_seconds()
    dx = (out["x"] - out["prev_x"]) * KM_PER_LON
    dy = (out["y"] - out["prev_y"]) * KM_PER_LAT
    out["step_km"] = np.sqrt(dx * dx + dy * dy)
    out["calc_knots"] = out["step_km"] / (out["dt_s"] / 3600) / 1.852
    out.loc[~np.isfinite(out["calc_knots"]), "calc_knots"] = np.nan
    out["duplicate_mmsi_time"] = out.duplicated(["mmsi", "time"], keep=False)
    out["duplicate_mmsi_time_drop"] = out.duplicated(["mmsi", "time"], keep="first")
    out["flag_sog_gt_15"] = out["sog"] > 15
    out["flag_sog_gt_30"] = out["sog"] > 30
    out["flag_true_heading_invalid"] = out["true_heading"].isna() | (out["true_heading"] == 511) | ~out["true_heading"].between(0, 359)
    out["flag_cog_invalid"] = out["cog"].isna() | ~out["cog"].between(0, 360, inclusive="left")
    out["flag_rot_missing"] = out["rot"].isna()
    out["flag_dt_gap_gt_1h"] = out["dt_s"] > 3600
    for threshold in [30, 50, 100]:
        out[f"flag_step_speed_gt_{threshold}"] = (out["dt_s"] > 0) & (out["dt_s"] <= 3600) & (out["calc_knots"] > threshold)
    return out


def diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("rows", len(df)),
        ("unique_mmsi", df["mmsi"].nunique()),
        ("missing_true_heading", df["true_heading"].isna().sum()),
        ("missing_rot", df["rot"].isna().sum()),
        ("true_heading_invalid", df["flag_true_heading_invalid"].sum()),
        ("cog_invalid", df["flag_cog_invalid"].sum()),
        ("sog_gt_15", df["flag_sog_gt_15"].sum()),
        ("sog_gt_30", df["flag_sog_gt_30"].sum()),
        ("outside_30km", (df["region"] == "outside").sum()),
        ("duplicate_mmsi_time_rows", df["duplicate_mmsi_time"].sum()),
        ("duplicate_mmsi_time_rows_to_drop", df["duplicate_mmsi_time_drop"].sum()),
        ("gap_gt_1h", df["flag_dt_gap_gt_1h"].sum()),
        ("step_speed_gt_30", df["flag_step_speed_gt_30"].sum()),
        ("step_speed_gt_50", df["flag_step_speed_gt_50"].sum()),
        ("step_speed_gt_100", df["flag_step_speed_gt_100"].sum()),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def clean_data(df: pd.DataFrame, spec: CleanSpec) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if spec.dedup_mmsi_time:
        mask &= ~df["duplicate_mmsi_time_drop"]
    if spec.max_sog is not None:
        mask &= df["sog"] <= spec.max_sog
    if spec.max_step_knots is not None:
        mask &= ~((df["dt_s"] > 0) & (df["dt_s"] <= 3600) & (df["calc_knots"] > spec.max_step_knots))
    return df.loc[mask].copy()


def make_a_labels(df: pd.DataFrame) -> pd.DataFrame:
    active = df[df["region"].isin(REGION_ORDER) & df["sog"].between(2, 10, inclusive="both")]
    ship_counts = active.groupby(["hour", "region", "mmsi"], observed=True).size().rename("ais_points").reset_index()
    qualified = ship_counts[ship_counts["ais_points"] >= 3]
    labels = qualified.groupby(["hour", "region"], observed=True)["mmsi"].nunique().rename("y").reset_index()
    hours = pd.date_range(df["hour"].min(), df["hour"].max(), freq="h")
    idx = pd.MultiIndex.from_product([hours, REGION_ORDER], names=["hour", "region"])
    return labels.set_index(["hour", "region"]).reindex(idx, fill_value=0).reset_index()


def make_b_labels(df: pd.DataFrame) -> pd.DataFrame:
    in_region = df[df["region"].isin(REGION_ORDER)]
    agg = (
        in_region.groupby(["mmsi", "hour", "region"], observed=True)
        .agg(points=("time", "size"), last_time=("time", "max"))
        .reset_index()
    )
    rep = (
        agg.sort_values(["mmsi", "hour", "points", "last_time"], ascending=[True, True, False, False])
        .drop_duplicates(["mmsi", "hour"], keep="first")
        .sort_values(["mmsi", "hour"])
        .rename(columns={"region": "source_region"})
    )
    rep["next_hour"] = rep.groupby("mmsi")["hour"].shift(-1)
    rep["target_region"] = rep.groupby("mmsi")["source_region"].shift(-1)
    moved = rep[
        (rep["next_hour"] == rep["hour"] + pd.Timedelta(hours=1))
        & (rep["source_region"].astype(str) != rep["target_region"].astype(str))
    ]
    labels = moved.groupby(["hour", "source_region", "target_region"], observed=True)["mmsi"].nunique().rename("y").reset_index()
    hours = pd.date_range(df["hour"].min(), df["hour"].max(), freq="h")
    idx = pd.MultiIndex.from_tuples(
        [(h, s, t) for h in hours for s, t in PAIRS],
        names=["hour", "source_region", "target_region"],
    )
    return labels.set_index(["hour", "source_region", "target_region"]).reindex(idx, fill_value=0).reset_index()


def compare_labels(raw: pd.DataFrame, other: pd.DataFrame, keys: list[str], label_name: str, version: str) -> dict[str, float | int | str]:
    merged = raw.merge(other, on=keys, how="outer", suffixes=("_raw", "_clean")).fillna(0)
    diff = merged["y_clean"] - merged["y_raw"]
    return {
        "version": version,
        "label": label_name,
        "rows": int(len(merged)),
        "changed_rows": int((diff != 0).sum()),
        "sum_raw": float(merged["y_raw"].sum()),
        "sum_clean": float(merged["y_clean"].sum()),
        "sum_delta": float(diff.sum()),
        "sum_abs_delta": float(diff.abs().sum()),
        "sse_between": float(np.sum(diff.to_numpy() ** 2)),
        "max_abs_delta": float(diff.abs().max()),
    }


def top_changes(raw: pd.DataFrame, other: pd.DataFrame, keys: list[str], version: str, label_name: str, n: int = 50) -> pd.DataFrame:
    merged = raw.merge(other, on=keys, how="outer", suffixes=("_raw", "_clean")).fillna(0)
    merged["delta"] = merged["y_clean"] - merged["y_raw"]
    merged["abs_delta"] = merged["delta"].abs()
    merged["version"] = version
    merged["label"] = label_name
    return merged.sort_values("abs_delta", ascending=False).head(n)


def version_specs() -> list[CleanSpec]:
    return [
        CleanSpec("raw"),
        CleanSpec("dedup_mmsi_time", dedup_mmsi_time=True),
        CleanSpec("drop_step_gt_100", max_step_knots=100),
        CleanSpec("drop_step_gt_50", max_step_knots=50),
        CleanSpec("drop_step_gt_30", max_step_knots=30),
        CleanSpec("dedup_drop_step_gt_50", dedup_mmsi_time=True, max_step_knots=50),
        CleanSpec("dedup_drop_step_gt_50_sog_le_30", dedup_mmsi_time=True, max_step_knots=50, max_sog=30),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = find_project_root(args.root)
    train_path = find_train_csv(root)
    run_dir = root / "outputs" / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}_quality_cleaning"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"root={root}")
    print(f"train={train_path}")
    print(f"run_dir={run_dir}")

    df = add_trajectory_flags(load_raw(train_path))
    diagnostics(df).to_csv(run_dir / "quality_diagnostics.csv", index=False, encoding="utf-8-sig")

    raw_a = make_a_labels(df)
    raw_b = make_b_labels(df)
    label_summaries = []
    version_summaries = []
    top_frames = []

    for spec in version_specs():
        cleaned = df if spec.name == "raw" else clean_data(df, spec)
        removed = len(df) - len(cleaned)
        version_summaries.append(
            {
                "version": spec.name,
                "rows": int(len(cleaned)),
                "removed_rows": int(removed),
                "removed_ratio": float(removed / len(df)),
                "unique_mmsi": int(cleaned["mmsi"].nunique()),
                "outside_rows": int((cleaned["region"] == "outside").sum()),
                "active_sog_2_10_rows": int(cleaned["sog"].between(2, 10, inclusive="both").sum()),
            }
        )
        a_labels = make_a_labels(cleaned)
        b_labels = make_b_labels(cleaned)
        a_labels.to_csv(run_dir / f"a_labels_{spec.name}.csv", index=False, encoding="utf-8-sig")
        b_labels.to_csv(run_dir / f"b_labels_{spec.name}.csv", index=False, encoding="utf-8-sig")
        label_summaries.append(compare_labels(raw_a, a_labels, ["hour", "region"], "A", spec.name))
        label_summaries.append(compare_labels(raw_b, b_labels, ["hour", "source_region", "target_region"], "B", spec.name))
        if spec.name != "raw":
            top_frames.append(top_changes(raw_a, a_labels, ["hour", "region"], spec.name, "A"))
            top_frames.append(top_changes(raw_b, b_labels, ["hour", "source_region", "target_region"], spec.name, "B"))

    pd.DataFrame(version_summaries).to_csv(run_dir / "clean_version_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(label_summaries).to_csv(run_dir / "label_diff_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(top_frames, ignore_index=True).to_csv(run_dir / "top_label_changes.csv", index=False, encoding="utf-8-sig")

    # Hour/day concentration of severe jump points.
    severe = df[df["flag_step_speed_gt_50"]].copy()
    if not severe.empty:
        severe.groupby(["date", "region"]).size().rename("rows").reset_index().to_csv(
            run_dir / "jump_gt_50_by_date_region.csv", index=False, encoding="utf-8-sig"
        )
        severe.groupby(["mmsi"]).size().rename("rows").sort_values(ascending=False).head(30).reset_index().to_csv(
            run_dir / "jump_gt_50_top_mmsi.csv", index=False, encoding="utf-8-sig"
        )

    metadata = {
        "train_path": str(train_path),
        "run_dir": str(run_dir),
        "rows": int(len(df)),
        "unique_mmsi": int(df["mmsi"].nunique()),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
