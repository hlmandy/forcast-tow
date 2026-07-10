"""Step 01 — Data audit for the 2026 CTS tugboat AIS task.

This script reads the raw training AIS file and the validation daily vessel
count file, reuses the OFFICIAL A/B label construction from
``src.optimized_baseline`` (``add_regions`` / ``make_a_labels`` /
``make_b_labels``) so the label caliber is identical to the rest of the
project, and writes five audit artifacts under ``outputs/step01_data_audit/``:

  1. train_daily_overview.csv     — one row per training day
  2. train_hourly_overview.csv    — one row per (day, hour)
  3. train_daily_by_source.csv    — one row per (day, data source)
  4. validation_daily_vessels.csv — standardized validation daily vessel counts
  5. summary.md                   — objective statistics only

It does NOT modify any prediction code, train models, delete anomalies, or
generate submission files. It uses no randomness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make the sibling modules under ``src/`` importable when this file is run as
# ``python src/step01_data_audit.py`` from the project root.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimized_baseline import (  # noqa: E402  (import after sys.path tweak)
    PAIRS,
    REGION_ORDER,
    add_regions,
    make_a_labels,
    make_b_labels,
)

# ---------------------------------------------------------------------------
# Fixed paths and column orders (per spec).
# ---------------------------------------------------------------------------
TRAIN_REL = Path("数据备份/训练集_20180101-0124_拖轮AIS.csv")
VAL_REL = Path("数据备份/验证集_20180125-0131_每日拖轮数量.csv")
OUT_REL = Path("outputs/step01_data_audit")

# B migration directions in the exact order required by the spec; this is the
# same order as optimized_baseline.PAIRS (source in REGION_ORDER, then target).
B_DIRECTIONS = list(PAIRS)  # [(core,near),(core,outer),(near,core),...]
B_COLS = [f"b_{s}_to_{t}" for s, t in B_DIRECTIONS]

DAILY_COLUMNS = [
    "date",
    "ais_record_count",
    "unique_vessel_count",
    "observed_hour_count",
    "min_hourly_record_count",
    "max_hourly_record_count",
    "median_records_per_vessel_hour",
    "a_core_total",
    "a_near_total",
    "a_outer_total",
    "a_total",
    *B_COLS,
    "b_total",
]

HOURLY_COLUMNS = [
    "date",
    "hour",
    "ais_record_count",
    "unique_vessel_count",
    "median_records_per_vessel",
    "vessels_with_at_least_3_records",
    "a_core",
    "a_near",
    "a_outer",
    "a_total",
    *B_COLS,
    "b_total",
]

SOURCE_COLUMNS = ["date", "source", "ais_record_count", "unique_vessel_count"]
VALIDATION_COLUMNS = ["date", "unique_vessel_count"]

RUN_CMD = "python " + " ".join(sys.argv)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_ais(train_path: Path) -> pd.DataFrame:
    """Read raw AIS using the same approach as optimized_baseline.load_data,
    extended with the ``source_dataset`` column needed for the by-source audit.
    """
    usecols = ["mmsi", "x", "y", "sog", "time", "source_dataset"]
    dtypes = {
        "mmsi": "string",
        "x": "float64",
        "y": "float64",
        "sog": "float32",
        "source_dataset": "string",
    }
    df = pd.read_csv(train_path, usecols=usecols, dtype=dtypes, parse_dates=["time"])
    df["hour"] = df["time"].dt.floor("h")
    df["date"] = df["time"].dt.normalize()
    df["hour_of_day"] = df["time"].dt.hour
    df = add_regions(df)
    return df


def load_validation(val_path: Path) -> pd.DataFrame:
    val = pd.read_csv(val_path)
    val["date"] = pd.to_datetime(val["date"]).dt.normalize()
    # The raw column is ``vessel_count``; standardize to the required name.
    val = val.rename(columns={"vessel_count": "unique_vessel_count"})
    val = val[["date", "unique_vessel_count"]].sort_values("date").reset_index(drop=True)
    return val


# ---------------------------------------------------------------------------
# Raw-AIS aggregations (no region/speed filtering — true data completeness).
# ---------------------------------------------------------------------------
def daily_raw_overview(df: pd.DataFrame, all_dates: pd.Series) -> pd.DataFrame:
    # ais records + unique vessels per day (unique_vessel_count matches the
    # validation caliber: distinct vessels seen at least once that day).
    base = (
        df.groupby("date", observed=True)
        .agg(ais_record_count=("time", "size"), unique_vessel_count=("mmsi", "nunique"))
        .reset_index()
    )

    # Distinct hours actually present each day.
    obs_hours = df.groupby("date", observed=True)["hour_of_day"].nunique().rename("observed_hour_count").reset_index()

    # Record counts on the full 24-hour grid (missing hours -> 0) so that the
    # min/max reveal genuine gaps instead of ignoring empty hours.
    hourly_recs = df.groupby(["date", "hour_of_day"], observed=True).size().rename("recs").reset_index()
    grid = pd.MultiIndex.from_product([all_dates, range(24)], names=["date", "hour_of_day"]).to_frame(index=False)
    hourly_recs = grid.merge(hourly_recs, on=["date", "hour_of_day"], how="left").fillna({"recs": 0})
    hourly_recs["recs"] = hourly_recs["recs"].astype(np.int64)
    min_hourly = hourly_recs.groupby("date", observed=True)["recs"].min().rename("min_hourly_record_count").reset_index()
    max_hourly = hourly_recs.groupby("date", observed=True)["recs"].max().rename("max_hourly_record_count").reset_index()

    # Median records per (vessel, hour) cell, within each day.
    vh = df.groupby(["date", "hour_of_day", "mmsi"], observed=True).size().rename("n").reset_index()
    med_vh = vh.groupby("date", observed=True)["n"].median().rename("median_records_per_vessel_hour").reset_index()

    out = (
        base.merge(obs_hours, on="date", how="outer")
        .merge(min_hourly, on="date", how="outer")
        .merge(max_hourly, on="date", how="outer")
        .merge(med_vh, on="date", how="outer")
        .fillna({"ais_record_count": 0, "unique_vessel_count": 0, "observed_hour_count": 0,
                 "min_hourly_record_count": 0, "max_hourly_record_count": 0,
                 "median_records_per_vessel_hour": 0.0})
    )
    # Reindex onto the full daily span so any entirely-missing day still appears.
    out = pd.DataFrame({"date": all_dates}).merge(out, on="date", how="left").fillna(
        {"ais_record_count": 0, "unique_vessel_count": 0, "observed_hour_count": 0,
         "min_hourly_record_count": 0, "max_hourly_record_count": 0,
         "median_records_per_vessel_hour": 0.0}
    )
    for c in ["ais_record_count", "unique_vessel_count", "observed_hour_count",
              "min_hourly_record_count", "max_hourly_record_count"]:
        out[c] = out[c].astype(np.int64)
    return out


def hourly_raw_overview(df: pd.DataFrame, all_dates: pd.Series) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product([all_dates, range(24)], names=["date", "hour_of_day"]).to_frame(index=False)
    grid = grid.rename(columns={"hour_of_day": "hour"})

    base = (
        df.groupby(["date", "hour_of_day"], observed=True)
        .agg(ais_record_count=("time", "size"), unique_vessel_count=("mmsi", "nunique"))
        .reset_index()
        .rename(columns={"hour_of_day": "hour"})
    )

    vh = df.groupby(["date", "hour_of_day", "mmsi"], observed=True).size().rename("n").reset_index()
    med = vh.groupby(["date", "hour_of_day"], observed=True)["n"].median().rename("median_records_per_vessel").reset_index().rename(columns={"hour_of_day": "hour"})
    ge3 = vh[vh["n"] >= 3].groupby(["date", "hour_of_day"], observed=True)["mmsi"].nunique().rename("vessels_with_at_least_3_records").reset_index().rename(columns={"hour_of_day": "hour"})

    out = (
        grid.merge(base, on=["date", "hour"], how="left")
        .merge(med, on=["date", "hour"], how="left")
        .merge(ge3, on=["date", "hour"], how="left")
        .fillna({"ais_record_count": 0, "unique_vessel_count": 0,
                 "median_records_per_vessel": 0.0, "vessels_with_at_least_3_records": 0})
    )
    for c in ["ais_record_count", "unique_vessel_count", "vessels_with_at_least_3_records"]:
        out[c] = out[c].astype(np.int64)
    return out


def daily_by_source(df: pd.DataFrame, all_dates: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    sources = sorted(df["source_dataset"].dropna().unique().tolist())
    if len(sources) == 1:
        sources = ["all"]
        df_src = df.assign(source_dataset="all")
    else:
        df_src = df

    agg = (
        df_src.groupby(["date", "source_dataset"], observed=True)
        .agg(ais_record_count=("time", "size"), unique_vessel_count=("mmsi", "nunique"))
        .reset_index()
        .rename(columns={"source_dataset": "source"})
    )

    grid = pd.MultiIndex.from_product([all_dates, sources], names=["date", "source"]).to_frame(index=False)
    out = grid.merge(agg, on=["date", "source"], how="left").fillna(
        {"ais_record_count": 0, "unique_vessel_count": 0}
    )
    for c in ["ais_record_count", "unique_vessel_count"]:
        out[c] = out[c].astype(np.int64)
    out = out.sort_values(["date", "source"]).reset_index(drop=True)
    return out, sources


# ---------------------------------------------------------------------------
# A/B label aggregation (reusing official label construction).
# ---------------------------------------------------------------------------
def a_daily_totals(a_lab: pd.DataFrame) -> pd.DataFrame:
    work = a_lab.copy()
    work["date"] = work["hour"].dt.normalize()
    g = work.groupby(["date", "region"], observed=True)["y"].sum().unstack("region")
    for r in REGION_ORDER:
        if r not in g.columns:
            g[r] = 0
    g = g[REGION_ORDER].fillna(0)
    out = g.rename(columns={r: f"a_{r}_total" for r in REGION_ORDER}).reset_index()
    out["a_total"] = out[[f"a_{r}_total" for r in REGION_ORDER]].sum(axis=1)
    return out


def a_hourly(a_lab: pd.DataFrame) -> pd.DataFrame:
    work = a_lab.copy()
    work["date"] = work["hour"].dt.normalize()
    work["hour"] = work["hour"].dt.hour
    g = work.groupby(["date", "hour", "region"], observed=True)["y"].sum().unstack("region")
    for r in REGION_ORDER:
        if r not in g.columns:
            g[r] = 0
    g = g[REGION_ORDER].fillna(0)
    out = g.rename(columns={r: f"a_{r}" for r in REGION_ORDER}).reset_index()
    out["a_total"] = out[[f"a_{r}" for r in REGION_ORDER]].sum(axis=1)
    return out


def b_daily_totals(b_lab: pd.DataFrame) -> pd.DataFrame:
    work = b_lab.copy()
    work["date"] = work["hour"].dt.normalize()
    work["dir"] = "b_" + work["source_region"] + "_to_" + work["target_region"]
    g = work.groupby(["date", "dir"], observed=True)["y"].sum().unstack("dir")
    for c in B_COLS:
        if c not in g.columns:
            g[c] = 0
    g = g[B_COLS].fillna(0)
    out = g.reset_index()
    out["b_total"] = out[B_COLS].sum(axis=1)
    return out


def b_hourly(b_lab: pd.DataFrame) -> pd.DataFrame:
    work = b_lab.copy()
    work["date"] = work["hour"].dt.normalize()
    work["hour"] = work["hour"].dt.hour
    work["dir"] = "b_" + work["source_region"] + "_to_" + work["target_region"]
    g = work.groupby(["date", "hour", "dir"], observed=True)["y"].sum().unstack("dir")
    for c in B_COLS:
        if c not in g.columns:
            g[c] = 0
    g = g[B_COLS].fillna(0)
    out = g.reset_index()
    out["b_total"] = out[B_COLS].sum(axis=1)
    return out


# ---------------------------------------------------------------------------
# Correlations.
# ---------------------------------------------------------------------------
def correlation_table(daily: pd.DataFrame) -> pd.DataFrame:
    target = daily["unique_vessel_count"]
    feats = (
        ["a_total"] + [f"a_{r}_total" for r in REGION_ORDER] + ["b_total"] + B_COLS
    )
    rows = []
    for f in feats:
        col = daily[f]
        if col.nunique() > 1 and target.nunique() > 1:
            pearson = float(col.corr(target, method="pearson"))
            spearman = float(col.corr(target, method="spearman"))
        else:
            pearson = float("nan")
            spearman = float("nan")
        rows.append({"feature": f, "pearson": pearson, "spearman": spearman})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output helpers.
# ---------------------------------------------------------------------------
def write_csv(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    frame = frame.copy()
    if "date" in frame.columns and pd.api.types.is_datetime64_any_dtype(frame["date"]):
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    frame = frame[columns]
    frame.to_csv(path, index=False, encoding="utf-8")


def md_num(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:.4f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    train_path = ROOT / TRAIN_REL
    val_path = ROOT / VAL_REL
    out_dir = ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"Training AIS file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    print(f"train_path = {train_path}")
    print(f"val_path   = {val_path}")
    print(f"out_dir    = {out_dir}")

    # --- Load ----------------------------------------------------------------
    df = load_ais(train_path)
    val = load_validation(val_path)

    all_dates = pd.Series(pd.date_range(df["date"].min(), df["date"].max(), freq="D").normalize())
    n_records = len(df)
    n_vessels = df["mmsi"].nunique()
    sources = sorted(df["source_dataset"].dropna().unique().tolist())
    train_date_min = df["date"].min()
    train_date_max = df["date"].max()

    # --- Official labels (reused) -------------------------------------------
    a_lab = make_a_labels(df)
    b_lab = make_b_labels(df)

    # --- Raw-AIS overviews --------------------------------------------------
    raw_daily = daily_raw_overview(df, all_dates)
    raw_hourly = hourly_raw_overview(df, all_dates)
    by_source, source_names = daily_by_source(df, all_dates)

    # --- A/B overviews ------------------------------------------------------
    a_dly = a_daily_totals(a_lab)
    a_hly = a_hourly(a_lab)
    b_dly = b_daily_totals(b_lab)
    b_hly = b_hourly(b_lab)

    daily = (
        raw_daily.merge(a_dly, on="date", how="left")
        .merge(b_dly, on="date", how="left")
        .fillna({c: 0 for c in DAILY_COLUMNS if c not in ("date", "median_records_per_vessel_hour")})
    )
    daily["median_records_per_vessel_hour"] = daily["median_records_per_vessel_hour"].fillna(0.0)
    int_cols_daily = [c for c in DAILY_COLUMNS if c not in ("date", "median_records_per_vessel_hour")]
    for c in int_cols_daily:
        daily[c] = daily[c].astype(np.int64)
    daily = daily.sort_values("date").reset_index(drop=True)

    hourly = (
        raw_hourly.merge(a_hly, on=["date", "hour"], how="left")
        .merge(b_hly, on=["date", "hour"], how="left")
        .fillna({c: 0 for c in HOURLY_COLUMNS if c not in ("date", "hour", "median_records_per_vessel")})
    )
    hourly["median_records_per_vessel"] = hourly["median_records_per_vessel"].fillna(0.0)
    int_cols_hourly = [c for c in HOURLY_COLUMNS if c not in ("date", "hour", "median_records_per_vessel")]
    for c in int_cols_hourly:
        hourly[c] = hourly[c].astype(np.int64)
    hourly = hourly.sort_values(["date", "hour"]).reset_index(drop=True)

    # --- Validation output --------------------------------------------------
    val_out = val.copy()

    # --- Correlations -------------------------------------------------------
    corr = correlation_table(daily)

    # --- Write CSVs ---------------------------------------------------------
    p_daily = out_dir / "train_daily_overview.csv"
    p_hourly = out_dir / "train_hourly_overview.csv"
    p_source = out_dir / "train_daily_by_source.csv"
    p_val = out_dir / "validation_daily_vessels.csv"
    p_summary = out_dir / "summary.md"

    write_csv(daily, p_daily, DAILY_COLUMNS)
    write_csv(hourly, p_hourly, HOURLY_COLUMNS)
    write_csv(by_source, p_source, SOURCE_COLUMNS)
    write_csv(val_out, p_val, VALIDATION_COLUMNS)

    # --- Summary statistics for summary.md ----------------------------------
    expected_dates = pd.Series(pd.date_range("2018-01-01", "2018-01-24", freq="D").normalize())
    present_dates = set(daily["date"].tolist())
    missing_dates = [d for d in expected_dates if d not in present_dates]

    dates_under_24h = daily[daily["observed_hour_count"] < 24][["date", "observed_hour_count"]]
    zero_record_hours = hourly[hourly["ais_record_count"] == 0][["date", "hour"]]
    lowest_ais_days = daily.nsmallest(5, "ais_record_count")[["date", "ais_record_count"]]
    lowest_density_days = daily.nsmallest(5, "median_records_per_vessel_hour")[
        ["date", "median_records_per_vessel_hour"]
    ]
    source_zero = by_source[by_source["ais_record_count"] == 0][["date", "source"]]

    train_vc = daily["unique_vessel_count"]
    val_vc = val["unique_vessel_count"]
    train_min, train_max, train_mean = int(train_vc.min()), int(train_vc.max()), float(train_vc.mean())
    val_min, val_max, val_mean = int(val_vc.min()), int(val_vc.max()), float(val_vc.mean())
    val_exceeds = (val_vc.max() > train_max) or (val_vc.min() < train_min)

    # --- summary.md ---------------------------------------------------------
    lines: list[str] = []
    lines.append("# Step 01 Data Audit")
    lines.append("")
    lines.append("> Objective statistics only. No model decisions, no anomaly deletion, no submissions.")
    lines.append("")
    lines.append(f"Run command: `{RUN_CMD}`")
    lines.append("")

    lines.append("## 1. Input information")
    lines.append("")
    lines.append(f"- Training date range: {train_date_min:%Y-%m-%d} ~ {train_date_max:%Y-%m-%d}")
    lines.append(f"- Number of AIS records: {n_records:,}")
    lines.append(f"- Number of unique vessels (whole training period): {n_vessels}")
    lines.append(f"- Number of data sources: {len(source_names)} ({', '.join(source_names)})")
    lines.append(f"- Validation date range: {val['date'].min():%Y-%m-%d} ~ {val['date'].max():%Y-%m-%d}")
    lines.append("")

    lines.append("## 2. Completeness checks")
    lines.append("")
    if missing_dates:
        lines.append("- Missing training dates: " + ", ".join(f"{d:%Y-%m-%d}" for d in missing_dates))
    else:
        lines.append("- Missing training dates: none")
    if len(dates_under_24h):
        detail = ", ".join(f"{r['date']:%Y-%m-%d}({int(r['observed_hour_count'])}h)" for _, r in dates_under_24h.iterrows())
        lines.append(f"- Dates with fewer than 24 observed hours: {len(dates_under_24h)} — {detail}")
    else:
        lines.append("- Dates with fewer than 24 observed hours: none")
    lines.append(f"- Hours with zero AIS records (on the 24x24 grid): {len(zero_record_hours)}")
    if len(zero_record_hours):
        detail = ", ".join(f"{r['date']:%Y-%m-%d} {int(r['hour'])}:00" for _, r in zero_record_hours.head(40).iterrows())
        suffix = "" if len(zero_record_hours) <= 40 else " ..."
        lines.append(f"  - {detail}{suffix}")
    lines.append("- Five dates with the lowest AIS record counts:")
    for _, r in lowest_ais_days.iterrows():
        lines.append(f"  - {r['date']:%Y-%m-%d}: {int(r['ais_record_count']):,} records")
    lines.append("- Five dates with the lowest median records per vessel-hour:")
    for _, r in lowest_density_days.iterrows():
        lines.append(f"  - {r['date']:%Y-%m-%d}: {r['median_records_per_vessel_hour']:.2f}")
    if len(source_zero):
        lines.append(f"- Source-level zero-record (date, source) cells: {len(source_zero)}")
        detail = ", ".join(f"({r['date']:%Y-%m-%d}, {r['source']})" for _, r in source_zero.head(40).iterrows())
        suffix = "" if len(source_zero) <= 40 else " ..."
        lines.append(f"  - {detail}{suffix}")
    else:
        lines.append("- Source-level zero-record periods: none")
    lines.append("")

    lines.append("## 3. Daily vessel-count relationships")
    lines.append("")
    lines.append("Pearson and Spearman correlations between daily `unique_vessel_count` and A/B totals.")
    lines.append("")
    lines.append("| feature | pearson | spearman |")
    lines.append("| --- | --- | --- |")
    for _, r in corr.iterrows():
        lines.append(f"| {r['feature']} | {md_num(r['pearson'])} | {md_num(r['spearman'])} |")
    lines.append("")
    lines.append("These correlations are descriptive only and do not fix any model choice.")
    lines.append("")

    lines.append("## 4. Validation comparison")
    lines.append("")
    lines.append("| set | min | max | mean |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| training | {train_min} | {train_max} | {train_mean:.2f} |")
    lines.append(f"| validation | {val_min} | {val_max} | {val_mean:.2f} |")
    lines.append("")
    lines.append(
        f"- Validation values exceed the training range: {'YES' if val_exceeds else 'no'}"
        f" (training {train_min}-{train_max}, validation {val_min}-{val_max})."
    )
    lines.append("")

    lines.append("## 5. Detected anomalies")
    lines.append("")
    lines.append("Dates and hours with obvious missing or abnormally sparse data. Nothing is deleted here.")
    lines.append("")
    if len(dates_under_24h):
        lines.append("- Dates with gaps (fewer than 24 observed hours):")
        for _, r in dates_under_24h.iterrows():
            lines.append(f"  - {r['date']:%Y-%m-%d}: only {int(r['observed_hour_count'])} hours observed")
    else:
        lines.append("- Dates with gaps: none")
    if len(zero_record_hours):
        lines.append(f"- (date, hour) cells with zero AIS records: {len(zero_record_hours)} (listed in section 2)")
    else:
        lines.append("- (date, hour) cells with zero AIS records: none")
    low_density_all = daily[daily["median_records_per_vessel_hour"] <= daily["median_records_per_vessel_hour"].quantile(0.25)]
    if len(low_density_all):
        lines.append("- Low-sampling-density days (lowest quartile of median records per vessel-hour):")
        for _, r in low_density_all.iterrows():
            lines.append(f"  - {r['date']:%Y-%m-%d}: {r['median_records_per_vessel_hour']:.2f}")
    lines.append("")

    p_summary.write_text("\n".join(lines), encoding="utf-8")

    # --- Console report -----------------------------------------------------
    print("\n=== Output files ===")
    for path in [p_daily, p_hourly, p_source, p_val, p_summary]:
        rows = len(pd.read_csv(path)) if path.suffix == ".csv" else sum(1 for _ in open(path, encoding="utf-8"))
        rel = path.relative_to(ROOT)
        print(f"  {rel}  rows={rows}")

    print("\nDone.")


if __name__ == "__main__":
    main()
