"""Step 02 — Diagnose the effect of AIS sampling quality on Task A labels.

Question: does the drop in the A label come from a real change in vessel
activity, or from insufficient AIS sampling?

This script reads the raw training AIS file, reuses the OFFICIAL region
assignment and A-label construction from ``src.optimized_baseline``
(``add_regions`` / ``make_a_labels``) so the A caliber is identical to the rest
of the project, and writes four diagnostic artifacts under
``outputs/step02_a_quality_effect/``:

  1. a_filter_funnel_hourly.csv     — A filter funnel per (day, hour, region)
  2. a_daily_quality_effect.csv     — per-day quality + funnel rollup
  3. a_relationship_by_quality.csv  — correlations under quality subsets
  4. summary.md

It does NOT modify prediction code, train models, generate submissions, or
delete / impute any training data. No randomness is used.
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

from optimized_baseline import (  # noqa: E402
    REGION_ORDER,
    add_regions,
    make_a_labels,
)

# ---------------------------------------------------------------------------
# Fixed configuration (per spec — do NOT infer the date range from raw data).
# ---------------------------------------------------------------------------
TRAIN_REL = Path("数据备份/训练集_20180101-0124_拖轮AIS.csv")
OUT_REL = Path("outputs/step02_a_quality_effect")

FIXED_DATES = pd.date_range("2018-01-01", "2018-01-24", freq="D").normalize()
HOURS = list(range(24))
SOURCES = ["china_coastal", "e_globe_daily", "f_globe_dynamic"]

SOG_LO, SOG_HI = 2, 10  # inclusive on both ends, matching make_a_labels

RUN_CMD = "python " + " ".join(sys.argv)

REGION_RANK = {r: i for i, r in enumerate(REGION_ORDER)}

FUNNEL_COLUMNS = [
    "date", "hour", "region",
    "raw_region_vessel_count",
    "active_region_vessel_count",
    "active_vessels_ge2",
    "active_vessels_ge3",
    "active_vessels_ge4",
    "a_label",
    "active_ais_record_count",
    "median_active_records_per_vessel",
    "china_coastal_active_record_count",
    "e_globe_daily_active_record_count",
    "f_globe_dynamic_active_record_count",
    "active_source_count",
    "ge3_rate_among_active",
]

DAILY_COLUMNS = [
    "date", "unique_vessel_count", "ais_record_count",
    "china_coastal_record_count", "e_globe_daily_record_count", "f_globe_dynamic_record_count",
    "china_record_ratio", "quality_regime", "zero_ais_hour_count",
    "raw_region_vessel_hours", "active_region_vessel_hours",
    "qualified_ge2_vessel_hours", "qualified_ge3_vessel_hours", "qualified_ge4_vessel_hours",
    "a_core_total", "a_near_total", "a_outer_total", "a_total",
    "a_qualification_rate",
]

RELATION_COLUMNS = [
    "subset", "target", "n_days",
    "pearson_daily_vessels", "spearman_daily_vessels",
    "pearson_ais_records", "spearman_ais_records",
    "pearson_active_records", "spearman_active_records",
    "pearson_qualification_rate", "spearman_qualification_rate",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_ais(train_path: Path) -> pd.DataFrame:
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


def grid_cell_frame() -> pd.DataFrame:
    """Full cartesian product: 24 dates x 24 hours x 3 regions = 1728 rows."""
    rows = [
        (d, h, r)
        for d in FIXED_DATES
        for h in HOURS
        for r in REGION_ORDER
    ]
    g = pd.DataFrame(rows, columns=["date", "hour", "region"])
    g["date"] = pd.to_datetime(g["date"]).dt.normalize()
    g["hour"] = g["hour"].astype(np.int64)
    return g


# ---------------------------------------------------------------------------
# Funnel (file 1)
# ---------------------------------------------------------------------------
def build_funnel(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (funnel_hourly keyed by [date,hour,region], a_label frame)."""
    in_rings = df["region"].isin(REGION_ORDER)
    active_mask = in_rings & df["sog"].between(SOG_LO, SOG_HI, inclusive="both")
    active = df[active_mask].copy()

    key = ["date", "hour_of_day", "region"]

    # vessel-level active record counts within (date, hour, region)
    vc = active.groupby(key + ["mmsi"], observed=True).size().rename("n").reset_index()

    # raw vessel presence (no speed filter)
    raw = df[in_rings]
    rrv = raw.groupby(key, observed=True)["mmsi"].nunique().rename("raw_region_vessel_count").reset_index()

    # active vessel presence (>=1 active record)
    arv = vc.groupby(key, observed=True)["mmsi"].nunique().rename("active_region_vessel_count").reset_index()

    # active records total + median records per active vessel
    arc = active.groupby(key, observed=True).size().rename("active_ais_record_count").reset_index()
    medn = vc.groupby(key, observed=True)["n"].median().rename("median_active_records_per_vessel").reset_index()

    # per-source active records + source count
    src = active.groupby(key + ["source_dataset"], observed=True).size().rename("n").reset_index()
    src_cols = {}
    for sname in SOURCES:
        s = src[src["source_dataset"] == sname].groupby(key, observed=True)["n"].sum().rename(f"{sname}_active_record_count").reset_index()
        src_cols[sname] = s
    asc = src.groupby(key, observed=True)["source_dataset"].nunique().rename("active_source_count").reset_index()

    # a_label from official make_a_labels
    a_lab = make_a_labels(df)
    a_lab["date"] = a_lab["hour"].dt.normalize()
    a_lab["hour_of_day"] = a_lab["hour"].dt.hour
    a_lab = a_lab[["date", "hour_of_day", "region", "y"]].rename(columns={"y": "a_label"})

    # start assembling onto the active-presence table
    funnel = arv
    for thr, col in [(2, "active_vessels_ge2"), (3, "active_vessels_ge3"), (4, "active_vessels_ge4")]:
        s = vc[vc["n"] >= thr].groupby(key, observed=True)["mmsi"].nunique().rename(col).reset_index()
        funnel = funnel.merge(s, on=key, how="outer")
    funnel = (
        funnel.merge(rrv, on=key, how="outer")
        .merge(arc, on=key, how="outer")
        .merge(medn, on=key, how="outer")
    )
    for sname in SOURCES:
        funnel = funnel.merge(src_cols[sname], on=key, how="outer")
    funnel = funnel.merge(asc, on=key, how="outer")
    funnel = funnel.merge(a_lab, on=key, how="outer")

    # attach to full grid (guarantees 1728 rows)
    grid = grid_cell_frame()
    funnel = grid.merge(funnel, left_on=["date", "hour", "region"], right_on=key, how="left")

    count_cols = [
        "raw_region_vessel_count", "active_region_vessel_count",
        "active_vessels_ge2", "active_vessels_ge3", "active_vessels_ge4",
        "a_label", "active_ais_record_count",
        "china_coastal_active_record_count", "e_globe_daily_active_record_count",
        "f_globe_dynamic_active_record_count", "active_source_count",
    ]
    for c in count_cols:
        funnel[c] = funnel[c].fillna(0).astype(np.int64)
    funnel["median_active_records_per_vessel"] = funnel["median_active_records_per_vessel"].fillna(0.0).astype(float)

    # consistency assertion: official label must equal the >=3 funnel exactly
    mismatch = (funnel["a_label"] != funnel["active_vessels_ge3"]).sum()
    if mismatch:
        raise AssertionError(f"a_label != active_vessels_ge3 in {mismatch} rows")

    funnel["ge3_rate_among_active"] = np.where(
        funnel["active_region_vessel_count"] > 0,
        funnel["active_vessels_ge3"] / funnel["active_region_vessel_count"],
        0.0,
    )

    funnel["_rank"] = funnel["region"].map(REGION_RANK)
    funnel = funnel.drop(columns=["hour_of_day"]).sort_values(["date", "hour", "_rank"]).drop(columns=["_rank"]).reset_index(drop=True)
    return funnel, a_lab


# ---------------------------------------------------------------------------
# Daily (file 2)
# ---------------------------------------------------------------------------
def quality_regime(china_count: int, china_ratio: float) -> str:
    if china_count == 0:
        return "outage"
    if china_ratio < 0.10:
        return "severe"
    if china_ratio < 0.50:
        return "reduced"
    return "normal"


def build_daily(df: pd.DataFrame, funnel: pd.DataFrame) -> pd.DataFrame:
    key = ["date", "hour_of_day", "region"]
    f = funnel.copy()
    f["hour_of_day"] = f["hour"]

    roll = f.groupby("date", observed=True).agg(
        raw_region_vessel_hours=("raw_region_vessel_count", "sum"),
        active_region_vessel_hours=("active_region_vessel_count", "sum"),
        qualified_ge2_vessel_hours=("active_vessels_ge2", "sum"),
        qualified_ge3_vessel_hours=("active_vessels_ge3", "sum"),
        qualified_ge4_vessel_hours=("active_vessels_ge4", "sum"),
        active_record_count=("active_ais_record_count", "sum"),
    ).reset_index()

    # A totals per region per day
    areg = f.groupby(["date", "region"], observed=True)["a_label"].sum().unstack("region")
    for r in REGION_ORDER:
        if r not in areg.columns:
            areg[r] = 0
    areg = areg[REGION_ORDER].fillna(0).rename(columns={r: f"a_{r}_total" for r in REGION_ORDER}).reset_index()
    areg["a_total"] = areg[[f"a_{r}_total" for r in REGION_ORDER]].sum(axis=1)

    # raw daily counts
    base = df.groupby("date", observed=True).agg(
        unique_vessel_count=("mmsi", "nunique"),
        ais_record_count=("time", "size"),
    ).reset_index()
    src_daily = {}
    for sname in SOURCES:
        src_daily[sname] = df[df["source_dataset"] == sname].groupby("date", observed=True).size().rename(f"{sname}_record_count").reset_index()

    # zero-ais hours per day (full 24h grid)
    hourly_recs = df.groupby(["date", "hour_of_day"], observed=True).size().rename("recs").reset_index()
    g2 = pd.MultiIndex.from_product([FIXED_DATES, HOURS], names=["date", "hour_of_day"]).to_frame(index=False)
    hourly_recs = g2.merge(hourly_recs, on=["date", "hour_of_day"], how="left").fillna({"recs": 0})
    hourly_recs["recs"] = hourly_recs["recs"].astype(np.int64)
    zero_hours = hourly_recs.groupby("date", observed=True)["recs"].apply(lambda s: int((s == 0).sum())).rename("zero_ais_hour_count").reset_index()

    daily = (
        pd.DataFrame({"date": FIXED_DATES})
        .merge(base, on="date", how="left")
    )
    for sname in SOURCES:
        daily = daily.merge(src_daily[sname], on="date", how="left")
    daily = (
        daily.merge(zero_hours, on="date", how="left")
        .merge(roll, on="date", how="left")
        .merge(areg, on="date", how="left")
    )

    fill_zero = [
        "unique_vessel_count", "ais_record_count",
        "china_coastal_record_count", "e_globe_daily_record_count", "f_globe_dynamic_record_count",
        "zero_ais_hour_count",
        "raw_region_vessel_hours", "active_region_vessel_hours",
        "qualified_ge2_vessel_hours", "qualified_ge3_vessel_hours", "qualified_ge4_vessel_hours",
        "active_record_count",
        "a_core_total", "a_near_total", "a_outer_total", "a_total",
    ]
    for c in fill_zero:
        daily[c] = daily[c].fillna(0)
    for c in fill_zero:
        if c != "active_record_count":
            daily[c] = daily[c].astype(np.int64)

    # china_record_ratio using median over non-zero china days
    china_vals = daily.loc[daily["china_coastal_record_count"] > 0, "china_coastal_record_count"]
    m_china = float(china_vals.median()) if len(china_vals) else float("nan")
    daily["china_record_ratio"] = daily["china_coastal_record_count"].astype(float) / m_china
    daily["china_record_ratio"] = daily["china_record_ratio"].fillna(0.0)

    daily["quality_regime"] = [
        quality_regime(int(c), float(r))
        for c, r in zip(daily["china_coastal_record_count"], daily["china_record_ratio"])
    ]

    # consistency assertion: qualified_ge3 == a_total
    mismatch = (daily["qualified_ge3_vessel_hours"] != daily["a_total"]).sum()
    if mismatch:
        raise AssertionError(f"qualified_ge3_vessel_hours != a_total in {mismatch} days")

    daily["a_qualification_rate"] = np.where(
        daily["active_region_vessel_hours"] > 0,
        daily["qualified_ge3_vessel_hours"] / daily["active_region_vessel_hours"],
        0.0,
    )

    daily = daily.sort_values("date").reset_index(drop=True)
    # stash m_china via attribute for the summary
    daily.attrs["m_china"] = m_china
    return daily


# ---------------------------------------------------------------------------
# Relationship table (file 3)
# ---------------------------------------------------------------------------
def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if len(x) < 3:
        return float("nan")
    xs = pd.Series(x)
    ys = pd.Series(y)
    if xs.nunique() <= 1 or ys.nunique() <= 1:
        return float("nan")
    return float(xs.corr(ys, method=method))


def build_relationships(daily: pd.DataFrame) -> pd.DataFrame:
    subsets = {
        "all": np.ones(len(daily), dtype=bool),
        "normal": daily["quality_regime"] == "normal",
        "usable": daily["quality_regime"].isin(["normal", "reduced"]),
        "degraded": daily["quality_regime"].isin(["outage", "severe"]),
    }
    targets = ["a_total", "a_core_total", "a_near_total", "a_outer_total"]
    predictors = {
        "daily_vessels": "unique_vessel_count",
        "ais_records": "ais_record_count",
        "active_records": "active_record_count",
        "qualification_rate": "a_qualification_rate",
    }
    rows = []
    for sname, mask in subsets.items():
        sub = daily[mask]
        for tgt in targets:
            y = sub[tgt].to_numpy(dtype=float)
            row = {"subset": sname, "target": tgt, "n_days": int(mask.sum())}
            for pname, col in predictors.items():
                x = sub[col].to_numpy(dtype=float)
                row[f"pearson_{pname}"] = safe_corr(x, y, "pearson")
                row[f"spearman_{pname}"] = safe_corr(x, y, "spearman")
            rows.append(row)
    return pd.DataFrame(rows, columns=RELATION_COLUMNS)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def write_csv(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    frame = frame.copy()
    if "date" in frame.columns and pd.api.types.is_datetime64_any_dtype(frame["date"]):
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    frame = frame[columns]
    frame.to_csv(path, index=False, encoding="utf-8")


def md_num(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.4f}"


def consecutive_zero_runs(daily_src: pd.Series) -> list[tuple[str, str, int]]:
    """Given a Series of per-date record counts (indexed by sorted date Timestamp),
    return list of (start, end, length) for maximal runs of zero days."""
    runs = []
    start = None
    prev = None
    length = 0
    dates = daily_src.index.tolist()
    vals = daily_src.values.tolist()
    ordered = sorted(zip(dates, vals))
    for d, v in ordered:
        if v == 0:
            if start is None:
                start = d
                length = 1
            else:
                # consecutive calendar day?
                if prev is not None and (d - prev) == pd.Timedelta(days=1):
                    length += 1
                else:
                    runs.append((start, prev, length))
                    start = d
                    length = 1
            prev = d
        else:
            if start is not None:
                runs.append((start, prev, length))
                start = None
                length = 0
            prev = d
    if start is not None:
        runs.append((start, prev, length))
    return [(f"{s:%Y-%m-%d}", f"{e:%Y-%m-%d}", n) for s, e, n in runs]


def main() -> None:
    train_path = ROOT / TRAIN_REL
    out_dir = ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    if not train_path.exists():
        raise FileNotFoundError(f"Training AIS file not found: {train_path}")

    print(f"train_path = {train_path}")
    print(f"out_dir    = {out_dir}")

    df = load_ais(train_path)

    actual_sources = sorted(df["source_dataset"].dropna().unique().tolist())
    if set(actual_sources) != set(SOURCES):
        raise AssertionError(f"Unexpected sources: {actual_sources} (expected {SOURCES})")

    # missing-date check based on RAW AIS dates vs the fixed range
    raw_dates = set(df["date"].dt.normalize().unique())
    missing_dates = [d for d in FIXED_DATES if d not in raw_dates]

    funnel, _ = build_funnel(df)
    daily = build_daily(df, funnel)
    rel = build_relationships(daily)

    # --- write CSVs ---------------------------------------------------------
    p_funnel = out_dir / "a_filter_funnel_hourly.csv"
    p_daily = out_dir / "a_daily_quality_effect.csv"
    p_rel = out_dir / "a_relationship_by_quality.csv"
    p_summary = out_dir / "summary.md"

    write_csv(funnel, p_funnel, FUNNEL_COLUMNS)
    write_csv(daily, p_daily, DAILY_COLUMNS)
    write_csv(rel, p_rel, RELATION_COLUMNS)

    # --- numbers for summary ------------------------------------------------
    n_records = len(df)
    n_vessels = df["mmsi"].nunique()
    m_china = daily.attrs["m_china"]

    # source-level completeness
    src_totals = {s: int((df["source_dataset"] == s).sum()) for s in SOURCES}
    src_daily_series = {
        s: df[df["source_dataset"] == s].groupby("date", observed=True).size().reindex(FIXED_DATES, fill_value=0)
        for s in SOURCES
    }
    src_stats = {
        s: (int(v.min()), float(v.median()), int(v.max())) for s, v in src_daily_series.items()
    }
    src_zero_runs = {s: consecutive_zero_runs(src_daily_series[s]) for s in SOURCES}

    # per-region funnel (aggregate over all 1728 cells)
    region_funnel = funnel.groupby("region", observed=True).agg(
        raw=("raw_region_vessel_count", "sum"),
        active=("active_region_vessel_count", "sum"),
        ge2=("active_vessels_ge2", "sum"),
        ge3=("active_vessels_ge3", "sum"),
        ge4=("active_vessels_ge4", "sum"),
    )
    region_funnel = region_funnel.reindex(REGION_ORDER)
    region_funnel["retention_active_to_ge3"] = region_funnel["ge3"] / region_funnel["active"]

    # quality regimes
    regime_dates: dict[str, list[str]] = {r: [] for r in ["outage", "severe", "reduced", "normal"]}
    for _, r in daily.iterrows():
        regime_dates[r["quality_regime"]].append(f"{r['date']:%Y-%m-%d}")

    normal_mask = daily["quality_regime"] == "normal"
    degraded_mask = daily["quality_regime"].isin(["outage", "severe"])
    normal_a_mean = float(daily.loc[normal_mask, "a_total"].mean()) if normal_mask.any() else float("nan")
    degraded_a_mean = float(daily.loc[degraded_mask, "a_total"].mean()) if degraded_mask.any() else float("nan")

    def rel_val(subset: str, target: str, field: str) -> float:
        m = (rel["subset"] == subset) & (rel["target"] == target)
        if not m.any():
            return float("nan")
        return float(rel.loc[m, field].iloc[0])

    # --- summary.md ---------------------------------------------------------
    L: list[str] = []
    L.append("# Step 02 A Quality Effect")
    L.append("")
    L.append("> Diagnostic only. No deletions, no imputation, no model selection.")
    L.append("")
    L.append(f"Run command: `{RUN_CMD}`")
    L.append("")

    L.append("## 1. Input and consistency checks")
    L.append("")
    L.append(f"- Fixed training date range: 2018-01-01 ~ 2018-01-24 ({len(FIXED_DATES)} days)")
    L.append(f"- Number of AIS records: {n_records:,}")
    L.append(f"- Number of unique vessels (whole period): {n_vessels}")
    L.append(f"- Data sources: {', '.join(SOURCES)}")
    if missing_dates:
        L.append("- Missing training dates (from raw AIS): " + ", ".join(f"{d:%Y-%m-%d}" for d in missing_dates))
    else:
        L.append("- Missing training dates (from raw AIS): none")
    L.append("- Consistency assertion `a_label == active_vessels_ge3` over all 1728 (day,hour,region) cells: **PASS**")
    L.append("- Consistency assertion `qualified_ge3_vessel_hours == a_total` over all 24 days: **PASS**")
    L.append("")

    L.append("## 2. Source-level completeness")
    L.append("")
    L.append("Total records and per-day record-count statistics (over the fixed 24-day range, zero days included).")
    L.append("")
    L.append("| source | total records | daily min | daily median | daily max |")
    L.append("| --- | --- | --- | --- | --- |")
    for s in SOURCES:
        mn, md, mx = src_stats[s]
        L.append(f"| {s} | {src_totals[s]:,} | {mn:,} | {md:.1f} | {mx:,} |")
    L.append("")
    L.append(f"- Median china_coastal record count over non-zero days (used as `M_china`): {m_china:,.1f}")
    L.append("- Consecutive zero-record intervals per source:")
    for s in SOURCES:
        runs = src_zero_runs[s]
        if runs:
            detail = "; ".join(f"{a}~{b} ({n}d)" for a, b, n in runs)
            L.append(f"  - {s}: {detail}")
        else:
            L.append(f"  - {s}: none")
    L.append("")

    L.append("## 3. A-filter funnel")
    L.append("")
    L.append("Aggregate vessel-hours per region over all 24 days x 24 hours. "
             "`retention_active_to_ge3` = sum(active_vessels_ge3) / sum(active_region_vessel_count).")
    L.append("")
    L.append("| region | raw_vessel_hours | active_vessel_hours | ge2 | ge3 | ge4 | retention active->ge3 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in REGION_ORDER:
        row = region_funnel.loc[r]
        L.append(f"| {r} | {int(row['raw'])} | {int(row['active'])} | {int(row['ge2'])} | {int(row['ge3'])} | {int(row['ge4'])} | {row['retention_active_to_ge3']:.4f} |")
    L.append("")
    overall_active = int(region_funnel["active"].sum())
    overall_ge3 = int(region_funnel["ge3"].sum())
    L.append(f"- Overall retention active->ge3: {overall_ge3}/{overall_active} = {overall_ge3 / overall_active:.4f}")
    L.append("")

    L.append("## 4. Quality regimes")
    L.append("")
    L.append(f"`M_china` = {m_china:,.1f} (median of china_coastal daily records over non-zero days).")
    L.append("")
    for regime in ["outage", "severe", "reduced", "normal"]:
        dates = regime_dates[regime]
        label = "day" if len(dates) == 1 else "days"
        L.append(f"- **{regime}** ({len(dates)} {label}): {', '.join(dates) if dates else 'none'}")
    L.append("")
    L.append(f"- Mean A total on `normal` days: {normal_a_mean:.2f}")
    L.append(f"- Mean A total on `degraded` (outage+severe) days: {degraded_a_mean:.2f}")
    L.append("")

    L.append("## 5. Relationships under different quality subsets")
    L.append("")
    L.append("Pearson / Spearman correlations between each predictor and each A target, within each day subset. "
             "Empty cell = too few days or a constant variable.")
    L.append("")
    hdr = "| subset | target | n_days | pearson daily_vessels | spearman daily_vessels | pearson ais_records | pearson active_records | pearson qualification_rate |"
    L.append(hdr)
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, r in rel.iterrows():
        L.append(
            f"| {r['subset']} | {r['target']} | {int(r['n_days'])} | "
            f"{md_num(r['pearson_daily_vessels'])} | {md_num(r['spearman_daily_vessels'])} | "
            f"{md_num(r['pearson_ais_records'])} | {md_num(r['pearson_active_records'])} | "
            f"{md_num(r['pearson_qualification_rate'])} |"
        )
    L.append("")
    L.append("Key comparisons (predictor vs `a_total`):")
    for subset, label in [("all", "all days"), ("normal", "normal days")]:
        L.append(
            f"- {label}: daily_vessels pearson={md_num(rel_val(subset, 'a_total', 'pearson_daily_vessels'))} "
            f"spearman={md_num(rel_val(subset, 'a_total', 'spearman_daily_vessels'))}; "
            f"ais_records pearson={md_num(rel_val(subset, 'a_total', 'pearson_ais_records'))}; "
            f"active_records pearson={md_num(rel_val(subset, 'a_total', 'pearson_active_records'))}; "
            f"qualification_rate pearson={md_num(rel_val(subset, 'a_total', 'pearson_qualification_rate'))}"
        )
    L.append("")

    L.append("## 6. Conclusions supported by this audit")
    L.append("")
    L.append("- The A label and the `>=3 active records` funnel are identical by construction (both assertions PASS), so all A variation below is real label variation, not a reconstruction artifact.")
    if normal_mask.any() and degraded_mask.any() and not pd.isna(normal_a_mean) and not pd.isna(degraded_a_mean):
        ratio = degraded_a_mean / normal_a_mean if normal_a_mean else float("nan")
        L.append(f"- Mean A total drops from {normal_a_mean:.1f} on normal days to {degraded_a_mean:.1f} on degraded days ({ratio:.2f}x).")
    L.append("- Source-level completeness shows the china_coastal outage that drives the degraded regime (see section 2).")
    L.append("- These results diagnose the data-quality effect only. They do NOT decide which days to delete, down-weight, or repair, and they do NOT choose a prediction model. Those decisions belong to later steps.")
    L.append("")

    p_summary.write_text("\n".join(L), encoding="utf-8")

    # --- console report -----------------------------------------------------
    print("\n=== Output files ===")
    for path in [p_funnel, p_daily, p_rel, p_summary]:
        if path.suffix == ".csv":
            rows = len(pd.read_csv(path))
        else:
            rows = sum(1 for _ in open(path, encoding="utf-8"))
        relpath = path.relative_to(ROOT)
        print(f"  {relpath}  rows={rows}")
    print("\nDone.")


if __name__ == "__main__":
    main()
