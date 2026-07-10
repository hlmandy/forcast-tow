from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REGION_CN = {"core": "核心区", "near": "近港区", "outer": "外围区"}

VARIANTS = [
    {
        "dir": "submit_01_A100_B085_known_best_combo",
        "a_multiplier": 1.00,
        "reason": "Combine best known A100 with best known B085 from separate online components.",
    },
    {
        "dir": "submit_02_A105_B085_small_up",
        "a_multiplier": 1.05,
        "reason": "Small A upward test on old stable A shape; B stays at known-better B085.",
    },
    {
        "dir": "submit_03_A095_B085_small_down",
        "a_multiplier": 0.95,
        "reason": "Small A downward guardrail test; B stays at known-better B085.",
    },
]


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "outputs").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not find project root")


def build_a_csv(source_dir: Path, template_dir: Path, multiplier: float) -> pd.DataFrame:
    if multiplier == 1.0:
        return pd.read_csv(source_dir / "提交结果1_区域活跃拖轮数量.csv")

    template = pd.read_csv(template_dir / "提交结果1_区域活跃拖轮数量.csv")
    template["time_dt"] = pd.to_datetime(template["time_window"])
    debug = pd.read_csv(source_dir / "submit_A_debug.csv", parse_dates=["hour"])
    debug["time_dt"] = debug["hour"]
    debug["zone"] = debug["region"].map(REGION_CN)
    debug["prediction_rounded"] = (debug["prediction"] * multiplier).round().clip(lower=0).astype(int)
    merged = template.drop(columns=["vessel_count"]).merge(
        debug[["time_dt", "zone", "prediction_rounded"]],
        on=["time_dt", "zone"],
        how="left",
        validate="one_to_one",
    )
    if merged["prediction_rounded"].isna().any():
        raise ValueError("A template mismatch")
    out = merged[["time_window", "zone"]].copy()
    out["vessel_count"] = merged["prediction_rounded"].astype(int)
    return out


def main() -> None:
    root = find_project_root(Path.cwd())
    source_a_dir = root / "outputs" / "20260709_110524_optimized"
    source_b_dir = root / "outputs" / "20260709_submission_candidates_joint_AB" / "candidate_01_A085_B085"
    template_dir = root / "outputs" / "_official_examples_backup"
    out_dir = root / "outputs" / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}_online_constrained_csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    b_csv = pd.read_csv(source_b_dir / "提交结果2_圈层间拖轮迁移量.csv")
    summary = []
    for variant in VARIANTS:
        run_dir = out_dir / variant["dir"]
        run_dir.mkdir()
        a_csv = build_a_csv(source_a_dir, template_dir, variant["a_multiplier"])
        a_csv.to_csv(run_dir / "提交结果1_区域活跃拖轮数量.csv", index=False, encoding="utf-8-sig")
        b_csv.to_csv(run_dir / "提交结果2_圈层间拖轮迁移量.csv", index=False, encoding="utf-8-sig")

        meta = {
            **variant,
            "source_A": str(source_a_dir),
            "source_B": str(source_b_dir),
            "A_sum": int(a_csv["vessel_count"].sum()),
            "B_sum": int(b_csv["vessel_count"].sum()),
            "A_nonzero": int((a_csv["vessel_count"] > 0).sum()),
            "B_nonzero": int((b_csv["vessel_count"] > 0).sum()),
            "output_format": "csv-only",
        }
        (run_dir / "variant_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append({key: meta[key] for key in ["dir", "a_multiplier", "A_sum", "B_sum", "A_nonzero", "B_nonzero", "reason"]})

    pd.DataFrame(summary).to_csv(out_dir / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    print(out_dir)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
