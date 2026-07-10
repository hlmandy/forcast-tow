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
class Strategy:
    name: str
    base: str
    recent_days: int | None = None
    trend_weight: float = 0.0
    vessel_power: float = 0.0
    shrink_weight: float = 0.0
    clip_quantile: float | None = None


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
        raise FileNotFoundError("No *_quality_cleaning output found")
    return runs[-1]


def find_val_daily(root: Path) -> Path:
    csvs = [p for p in root.rglob("*.csv") if p.is_file() and "outputs" not in p.parts and p.stat().st_size < 100_000]
    for p in csvs:
        try:
            sample = pd.read_csv(p, nrows=10)
        except Exception:
            continue
        if list(sample.columns) == ["date", "vessel_count"]:
            return p
    raise FileNotFoundError("Could not find validation daily count CSV")


def load_labels(quality_run: Path, version: str = "raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv(quality_run / f"a_labels_{version}.csv", parse_dates=["hour"])
    b = pd.read_csv(quality_run / f"b_labels_{version}.csv", parse_dates=["hour"])
    a["task_key"] = a["region"]
    b["task_key"] = b["source_region"] + "->" + b["target_region"]
    return add_time(a), add_time(b)


def add_time(df: pd.DataFrame, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = pd.to_datetime(out["hour"])
    out["date"] = out["hour"].dt.normalize()
    out["hour_of_day"] = out["hour"].dt.hour
    out["dayofweek"] = out["hour"].dt.dayofweek
    out["day_index"] = (out["date"] - out["date"].min()).dt.days
    if daily is not None:
        hint = daily.copy()
        hint["date"] = pd.to_datetime(hint["date"]).dt.normalize()
        out = out.merge(hint[["date", "vessel_count"]], on="date", how="left")
    return out


def train_daily_from_labels(a: pd.DataFrame) -> pd.DataFrame:
    # This is not unique-vessel count; it is only a fallback. Prefer external
    # daily counts when available. For backtest scaling we derive it from labels
    # consistently to avoid raw AIS reload.
    return a.groupby("date")["y"].sum().rename("daily_activity_sum").reset_index()


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
    return add_time(frame, val_daily)


def key_cols(task: str) -> list[str]:
    return ["region"] if task == "A" else ["source_region", "target_region"]


def make_strategies(task: str) -> list[Strategy]:
    strategies: list[Strategy] = []
    bases = [
        ("global", None),
        ("hour", None),
        ("dow_hour", None),
        ("recent_hour", 3),
        ("recent_hour", 5),
        ("recent_hour", 7),
        ("recent_hour", 10),
        ("recent_hour", 14),
        ("recent_same_dow_hour", 14),
    ]
    vessel_powers = [0.0, 0.25, 0.5, 0.75, 1.0] if task == "A" else [0.0, 0.25, 0.5]
    trend_weights = [0.0, 0.25, 0.5]
    shrink_weights = [0.0, 0.25, 0.5] if task == "B" else [0.0, 0.15, 0.3]
    clip_quantiles = [None, 0.95, 0.98] if task == "A" else [None, 0.98]
    for base, days in bases:
        for vp in vessel_powers:
            for tw in trend_weights:
                for sw in shrink_weights:
                    for cq in clip_quantiles:
                        if base == "global" and (tw or days or cq):
                            continue
                        if base in {"global", "dow_hour"} and days is not None:
                            continue
                        name = base
                        if days:
                            name += f"_{days}d"
                        if vp:
                            name += f"_vp{vp:g}"
                        if tw:
                            name += f"_tw{tw:g}"
                        if sw:
                            name += f"_sw{sw:g}"
                        if cq:
                            name += f"_clip{cq:g}"
                        strategies.append(Strategy(name, base, days, tw, vp, sw, cq))
    # Add a few explicit shape-only variants.
    strategies.append(Strategy("recent_7d_hour_no_scale", "recent_hour", 7))
    strategies.append(Strategy("recent_14d_hour_no_scale", "recent_hour", 14))
    return list({s.name: s for s in strategies}.values())


def group_mean(train: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return train.groupby(cols, observed=True)["y"].mean().rename("pred").reset_index().drop_duplicates(cols)


def predict(train: pd.DataFrame, future: pd.DataFrame, task: str, strategy: Strategy) -> pd.Series:
    train = train.copy()
    future = future.copy().reset_index(drop=True)
    keys = key_cols(task)
    key_mean = group_mean(train, keys).rename(columns={"pred": "key_mean"})
    global_mean = float(train["y"].mean())

    if strategy.base == "global":
        merged = future.merge(key_mean, on=keys, how="left", validate="many_to_one")
        pred = merged["key_mean"].fillna(global_mean)
    elif strategy.base == "hour":
        cols = keys + ["hour_of_day"]
        mean = group_mean(train, cols)
        merged = future.merge(mean, on=cols, how="left", validate="many_to_one").merge(key_mean, on=keys, how="left", validate="many_to_one")
        pred = merged["pred"].fillna(merged["key_mean"]).fillna(global_mean)
    elif strategy.base == "dow_hour":
        cols = keys + ["dayofweek", "hour_of_day"]
        mean = group_mean(train, cols)
        merged = future.merge(mean, on=cols, how="left", validate="many_to_one").merge(key_mean, on=keys, how="left", validate="many_to_one")
        pred = merged["pred"].fillna(merged["key_mean"]).fillna(global_mean)
    elif strategy.base == "recent_hour":
        assert strategy.recent_days is not None
        cutoff = train["hour"].max() - pd.Timedelta(days=strategy.recent_days)
        recent = train[train["hour"] > cutoff]
        cols = keys + ["hour_of_day"]
        mean = group_mean(recent, cols)
        merged = future.merge(mean, on=cols, how="left", validate="many_to_one").merge(key_mean, on=keys, how="left", validate="many_to_one")
        pred = merged["pred"].fillna(merged["key_mean"]).fillna(global_mean)
    elif strategy.base == "recent_same_dow_hour":
        assert strategy.recent_days is not None
        cutoff = train["hour"].max() - pd.Timedelta(days=strategy.recent_days)
        recent = train[train["hour"] > cutoff]
        cols = keys + ["dayofweek", "hour_of_day"]
        mean = group_mean(recent, cols)
        fallback = group_mean(train, keys + ["hour_of_day"])
        merged = (
            future.merge(mean, on=cols, how="left", validate="many_to_one")
            .merge(fallback.rename(columns={"pred": "fallback"}), on=keys + ["hour_of_day"], how="left", validate="many_to_one")
            .merge(key_mean, on=keys, how="left", validate="many_to_one")
        )
        pred = merged["pred"].fillna(merged["fallback"]).fillna(merged["key_mean"]).fillna(global_mean)
    else:
        raise ValueError(strategy)

    pred = pd.Series(pred).reset_index(drop=True).astype(float)

    # Blend toward per-key mean to reduce sparse B overreaction or A spikes.
    if strategy.shrink_weight:
        km = future.merge(key_mean, on=keys, how="left", validate="many_to_one")["key_mean"].fillna(global_mean).reset_index(drop=True)
        pred = (1 - strategy.shrink_weight) * pred + strategy.shrink_weight * km

    # Apply daily vessel-count calibration if the future frame has vessel_count.
    if strategy.vessel_power and "vessel_count" in future.columns:
        train_vessel_mean = train.drop_duplicates("date")["vessel_count"].mean() if "vessel_count" in train.columns else future["vessel_count"].mean()
        scale = (future["vessel_count"].reset_index(drop=True) / train_vessel_mean).fillna(1.0)
        pred = pred * np.power(scale, strategy.vessel_power)

    # Add simple trend adjustment from recent daily total vs all daily total.
    if strategy.trend_weight:
        recent_cut = train["hour"].max() - pd.Timedelta(days=7)
        recent_daily = train[train["hour"] > recent_cut].groupby("date")["y"].sum().mean()
        all_daily = train.groupby("date")["y"].sum().mean()
        if all_daily and np.isfinite(recent_daily):
            trend_scale = recent_daily / all_daily
            pred = pred * ((1 - strategy.trend_weight) + strategy.trend_weight * trend_scale)

    if strategy.clip_quantile:
        cap = train.groupby(keys, observed=True)["y"].quantile(strategy.clip_quantile).rename("cap").reset_index()
        caps = future.merge(cap, on=keys, how="left", validate="many_to_one")["cap"].fillna(train["y"].quantile(strategy.clip_quantile)).reset_index(drop=True)
        pred = np.minimum(pred, caps)

    if len(pred) != len(future):
        raise ValueError(f"{strategy.name}: predicted {len(pred)} rows for {len(future)} future rows")
    return pd.Series(pred).clip(lower=0)


def sse(y: pd.Series, p: pd.Series) -> float:
    yv = pd.Series(y).reset_index(drop=True).to_numpy(float)
    pv = pd.Series(p).reset_index(drop=True).to_numpy(float)
    return float(np.sum((yv - pv) ** 2))


def backtest(frame: pd.DataFrame, task: str, strategies: list[Strategy]) -> pd.DataFrame:
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
        for st in strategies:
            p = predict(train, valid.drop(columns=["y"]), task, st)
            rows.append({"task": task, "split": split, "method": st.name, "sse": sse(valid["y"], p)})
    return pd.DataFrame(rows)


def select_best(bt: pd.DataFrame) -> pd.DataFrame:
    return (
        bt.groupby("method")
        .agg(mean_sse=("sse", "mean"), median_sse=("sse", "median"), max_sse=("sse", "max"), std_sse=("sse", "std"))
        .assign(robust_score=lambda x: x["mean_sse"] + 0.15 * x["max_sse"])
        .sort_values("robust_score")
    )


def read_template(template_dir: Path, zip_name: str, csv_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(template_dir / zip_name) as zf:
        with zf.open(csv_name) as fh:
            return pd.read_csv(fh)


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

    for df, csv_name, zip_name in [
        (a_out, "提交结果1_区域活跃拖轮数量.csv", "提交结果1_区域活跃拖轮数量.zip"),
        (b_out, "提交结果2_圈层间拖轮迁移量.csv", "提交结果2_圈层间拖轮迁移量.zip"),
    ]:
        csv_path = run_dir / csv_name
        zip_path = run_dir / zip_name
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, arcname=csv_name)


def load_online_feedback(root: Path) -> dict[str, float]:
    feedback = {}
    anchor = root / "outputs" / "20260709_110524_optimized" / "online_score.json"
    if anchor.exists():
        data = json.loads(anchor.read_text(encoding="utf-8"))
        feedback["anchor_total"] = data.get("online_total_score")
        feedback["anchor_sse_A"] = data.get("online_sse_A")
        feedback["anchor_sse_B"] = data.get("online_sse_B")
    joint = root / "outputs" / "20260709_submission_candidates_joint_AB" / "online_results_so_far.csv"
    if joint.exists():
        df = pd.read_csv(joint)
        for _, row in df.iterrows():
            feedback[f"{row['candidate']}_A"] = row["SSE_A"]
            feedback[f"{row['candidate']}_B"] = row["SSE_B"]
    return feedback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--label-version", default="raw")
    args = parser.parse_args()

    root = find_project_root(args.root)
    quality = latest_quality_run(root)
    val_daily_path = find_val_daily(root)
    val_daily = pd.read_csv(val_daily_path, parse_dates=["date"])
    template_dir = root / "outputs" / "_official_examples_backup"
    run_dir = root / "outputs" / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}_forecast_search"
    run_dir.mkdir(parents=True, exist_ok=True)

    a, b = load_labels(quality, args.label_version)
    # Use true daily vessel counts from validation for future, and derive train
    # vessel_count from the original validation hint scale proxy is impossible
    # here, so use daily total label scale for CV. This avoids reloading AIS.
    a_daily = a.groupby("date")["y"].sum().rename("vessel_count").reset_index()
    b_daily = b.groupby("date")["y"].sum().rename("vessel_count").reset_index()
    a = add_time(a.drop(columns=[c for c in ["date", "hour_of_day", "dayofweek", "day_index"] if c in a.columns]), a_daily)
    b = add_time(b.drop(columns=[c for c in ["date", "hour_of_day", "dayofweek", "day_index"] if c in b.columns]), b_daily)

    a_strategies = make_strategies("A")
    b_strategies = make_strategies("B")
    bt_a = backtest(a, "A", a_strategies)
    bt_b = backtest(b, "B", b_strategies)
    summary_a = select_best(bt_a)
    summary_b = select_best(bt_b)

    best_a_name = str(summary_a.index[0])
    best_b_name = str(summary_b.index[0])
    strat_a = {s.name: s for s in a_strategies}[best_a_name]
    strat_b = {s.name: s for s in b_strategies}[best_b_name]

    # For final future vessel scaling, use validation daily count by replacing
    # the train frame's vessel_count scale with actual training daily unique
    # count unavailable here. We normalize validation counts by their mean and
    # keep strategy vessel scaling primarily learned by CV.
    future_a = future_grid("A", val_daily)
    future_b = future_grid("B", val_daily)
    future_a["prediction"] = predict(a, future_a, "A", strat_a)
    future_b["prediction"] = predict(b, future_b, "B", strat_b)

    # Do not tune final predictions from sparse online feedback here. The
    # leaderboard points are useful records, but two submissions are not enough
    # to identify a stable correction. Keep this script train/CV driven.
    feedback = load_online_feedback(root)
    online_calibration = {"A_multiplier": 1.0, "B_multiplier": 1.0}
    future_a["prediction"] *= online_calibration["A_multiplier"]
    future_b["prediction"] *= online_calibration["B_multiplier"]

    future_a["prediction_rounded"] = future_a["prediction"].clip(lower=0).round().astype(int)
    future_b["prediction_rounded"] = future_b["prediction"].clip(lower=0).round().astype(int)

    bt_a.to_csv(run_dir / "backtest_A_detail.csv", index=False, encoding="utf-8-sig")
    bt_b.to_csv(run_dir / "backtest_B_detail.csv", index=False, encoding="utf-8-sig")
    summary_a.to_csv(run_dir / "backtest_A_summary.csv", encoding="utf-8-sig")
    summary_b.to_csv(run_dir / "backtest_B_summary.csv", encoding="utf-8-sig")
    future_a.to_csv(run_dir / "submit_A_debug.csv", index=False, encoding="utf-8-sig")
    future_b.to_csv(run_dir / "submit_B_debug.csv", index=False, encoding="utf-8-sig")
    write_official(run_dir, template_dir, future_a, future_b)

    metadata = {
        "quality_run": str(quality),
        "val_daily_path": str(val_daily_path),
        "label_version": args.label_version,
        "best_A_method": best_a_name,
        "best_B_method": best_b_name,
        "cv_A_mean_sse": float(summary_a.loc[best_a_name, "mean_sse"]),
        "cv_B_mean_sse": float(summary_b.loc[best_b_name, "mean_sse"]),
        "cv_combined_mean_score": float(summary_a.loc[best_a_name, "mean_sse"] + 3 * summary_b.loc[best_b_name, "mean_sse"]),
        "online_feedback": feedback,
        "online_calibration": online_calibration,
        "rows_A": int(len(future_a)),
        "rows_B": int(len(future_b)),
        "sum_A": int(future_a["prediction_rounded"].sum()),
        "sum_B": int(future_b["prediction_rounded"].sum()),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
