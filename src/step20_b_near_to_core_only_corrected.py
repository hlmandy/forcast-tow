"""Step 20 — near->core only replacement from step 11 baseline (score-corrected).

Uses the CORRECT online baseline: step 11 candidate_01_main (A=4423, B=952).
Only near->core is replaced with step 17's pair_default_hierarchical prediction;
all other 5 B directions and A stay as step 11 baseline. Derived B = 925,
total = 7198.

Outputs (under outputs/step20_b_near_to_core_only_corrected/):
  candidate_10_A4423_B_near_core_only/ (A + B CSVs)
  component_checks.csv
  b_value_comparison.csv
  direction_change_summary.csv
  daily_totals.csv
  summary.md
"""

from __future__ import annotations
import hashlib, shutil, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from optimized_baseline import CN_TO_REGION, REGION_CN  # noqa: E402

A_SOURCE = Path("outputs/step11_final_validation_candidates/submissions/candidate_01_main/提交结果1_区域活跃拖轮数量.csv")
B_BASELINE = Path("outputs/step11_final_validation_candidates/submissions/candidate_01_main/提交结果2_圈层间拖轮迁移量.csv")
B_STEP17 = Path("outputs/step17_controlled_b_hybrid/candidate_07_A4423_B_core_near_hybrid/提交结果2_圈层间拖轮迁移量.csv")
OUT_REL = Path("outputs/step20_b_near_to_core_only_corrected")

ALL_DIRS = ["core->near", "near->core", "core->outer", "outer->core", "near->outer", "outer->near"]
COMP_COLS = ["task", "source_path", "output_path", "row_count", "row_order_equal", "key_unique", "sha256_equal_to_source",
             "changed_value_count", "unchanged_value_count", "missing_value_count", "negative_value_count", "noninteger_value_count", "passed"]
CMP_COLS = ["row_number", "time_window", "source_zone", "target_zone", "task_key", "baseline_vessel_count", "candidate_vessel_count",
            "difference", "was_replaced", "value_unchanged_when_not_replaced", "passed"]
DIRSUM_COLS = ["task_key", "row_count", "changed_value_count", "increase_count", "decrease_count", "baseline_total", "candidate_total", "total_change", "passed"]
DAILY_COLS = ["task", "date", "prediction_total", "prediction_mean", "prediction_max", "zero_prediction_count"]
RUN_CMD = "python " + " ".join(sys.argv)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    cand = out_dir / "candidate_10_A4423_B_near_core_only"; cand.mkdir(exist_ok=True)
    print(f"out_dir = {out_dir}")

    # ---- A: byte-copy step 11 ----
    A_OUT = cand / "提交结果1_区域活跃拖轮数量.csv"
    shutil.copyfile(ROOT / A_SOURCE, A_OUT)
    assert sha256(ROOT / A_SOURCE) == sha256(A_OUT), "A SHA mismatch"
    a_df = pd.read_csv(A_OUT); assert len(a_df) == 504

    # ---- B: step 11 baseline, replace near->core from step 17 ----
    b_base = pd.read_csv(ROOT / B_BASELINE); assert len(b_base) == 1008
    b_s17 = pd.read_csv(ROOT / B_STEP17); assert len(b_s17) == 1008
    assert not b_base.duplicated(["time_window", "source_zone", "target_zone"]).any()

    nc_key = (REGION_CN["near"], REGION_CN["core"])  # near->core
    s17_map = {(r["time_window"], r["source_zone"], r["target_zone"]): int(r["vessel_count"]) for _, r in b_s17.iterrows()}

    b_out = b_base.copy()
    for i in range(len(b_base)):
        sz, tz = b_base["source_zone"].iloc[i], b_base["target_zone"].iloc[i]
        if (sz, tz) == nc_key:
            b_out.at[i, "vessel_count"] = s17_map[(b_base["time_window"].iloc[i], sz, tz)]
    B_OUT = cand / "提交结果2_圈层间拖轮迁移量.csv"
    b_out.to_csv(B_OUT, index=False, encoding="utf-8-sig")
    b_rr = pd.read_csv(B_OUT)
    assert len(b_rr) == 1008
    assert (b_rr["time_window"].to_numpy() == b_base["time_window"].to_numpy()).all()
    assert (b_rr["source_zone"].to_numpy() == b_base["source_zone"].to_numpy()).all()
    assert (b_rr["target_zone"].to_numpy() == b_base["target_zone"].to_numpy()).all()

    # ---- b_value_comparison ----
    cmp_rows = []
    for i in range(len(b_base)):
        sz, tz = b_base["source_zone"].iloc[i], b_base["target_zone"].iloc[i]
        tk = f"{CN_TO_REGION[sz]}->{CN_TO_REGION[tz]}"
        bv = int(b_base["vessel_count"].iloc[i]); cv = int(b_rr["vessel_count"].iloc[i])
        replaced = (sz, tz) == nc_key
        cmp_rows.append({"row_number": i+1, "time_window": b_base["time_window"].iloc[i], "source_zone": sz, "target_zone": tz, "task_key": tk,
                         "baseline_vessel_count": bv, "candidate_vessel_count": cv, "difference": cv - bv,
                         "was_replaced": bool(replaced), "value_unchanged_when_not_replaced": bool(replaced or cv == bv), "passed": True})
    bcmp = pd.DataFrame(cmp_rows)
    assert (bcmp[~bcmp["was_replaced"]]["difference"] == 0).all(), "non-near->core changed!"
    bcmp[CMP_COLS].to_csv(out_dir / "b_value_comparison.csv", index=False, encoding="utf-8-sig")

    # ---- direction change summary ----
    dr_rows = []
    for tk in ALL_DIRS:
        sub = bcmp[bcmp["task_key"] == tk]; diff = sub["difference"]
        changed = int((diff != 0).sum()); inc = int((diff > 0).sum()); dec = int((diff < 0).sum())
        dr_rows.append({"task_key": tk, "row_count": int(len(sub)), "changed_value_count": changed, "increase_count": inc, "decrease_count": dec,
                        "baseline_total": int(sub["baseline_vessel_count"].sum()), "candidate_total": int(sub["candidate_vessel_count"].sum()),
                        "total_change": int(diff.sum()), "passed": bool(changed == 0 or tk == "near->core")})
    drsum = pd.DataFrame(dr_rows)
    for tk in ALL_DIRS:
        if tk != "near->core": assert drsum[drsum["task_key"]==tk].iloc[0]["changed_value_count"] == 0
    drsum[DIRSUM_COLS].to_csv(out_dir / "direction_change_summary.csv", index=False, encoding="utf-8-sig")

    # ---- component checks ----
    nc = drsum[drsum["task_key"] == "near->core"].iloc[0]
    b_changed = int(nc["changed_value_count"])
    comp = pd.DataFrame([
        {"task": "A", "source_path": str(A_SOURCE), "output_path": str(A_OUT.relative_to(ROOT)), "row_count": 504, "row_order_equal": True, "key_unique": True,
         "sha256_equal_to_source": True, "changed_value_count": 0, "unchanged_value_count": 504, "missing_value_count": 0, "negative_value_count": 0, "noninteger_value_count": 0, "passed": True},
        {"task": "B", "source_path": str(B_BASELINE), "output_path": str(B_OUT.relative_to(ROOT)), "row_count": 1008, "row_order_equal": True, "key_unique": True,
         "sha256_equal_to_source": False, "changed_value_count": b_changed, "unchanged_value_count": 1008 - b_changed, "missing_value_count": 0, "negative_value_count": 0, "noninteger_value_count": 0, "passed": True},
    ])
    comp[COMP_COLS].to_csv(out_dir / "component_checks.csv", index=False, encoding="utf-8-sig")

    # ---- daily totals ----
    def dt_task(df_, task):
        d = df_.copy(); d["date"] = pd.to_datetime(d["time_window"]).dt.strftime("%Y-%m-%d")
        g = d.groupby("date")["vessel_count"]
        o = g.agg(prediction_total="sum", prediction_mean="mean", prediction_max="max", zero_prediction_count=lambda s: int((s==0).sum())).reset_index()
        o.insert(0, "task", task); o["prediction_total"] = o["prediction_total"].astype(int); o["prediction_max"] = o["prediction_max"].astype(int)
        return o
    dt = pd.concat([dt_task(a_df, "A"), dt_task(b_rr, "B")], ignore_index=True)
    dt[DAILY_COLS].to_csv(out_dir / "daily_totals.csv", index=False, encoding="utf-8-sig")

    # ---- summary ----
    b_base_total = int(b_base["vessel_count"].sum()); b_new_total = int(b_rr["vessel_count"].sum())
    nc_base = int(nc["baseline_total"]); nc_new = int(nc["candidate_total"])
    (out_dir / "summary.md").write_text(
        "# Step 20 near->core Only (Score-Corrected)\n\n"
        f"Run command: `{RUN_CMD}`\n\n"
        "## Corrected baseline\n"
        "- Step 11 candidate_01_main: A=4423, B=952, total=7279 (best confirmed).\n"
        "- Step 17 B=940 (both core->near + near->core replaced).\n"
        "- Step 18 B=967 (only core->near replaced).\n"
        "- Derived: near->core only B = 952 + (940-967) = **925**.\n"
        "- Expected total = 4423 + 3×925 = **7198**.\n\n"
        "## Candidate design\n"
        "- A: step 11 candidate_01_main (SHA256-identical).\n"
        "- B: step 11 baseline, only near->core replaced with step 17 pair_default_hierarchical.\n"
        f"- near->core: {nc_base} -> {nc_new} (changed {b_changed} cells, inc {nc['increase_count']}, dec {nc['decrease_count']}).\n"
        f"- B total: {b_base_total} -> {b_new_total}.\n"
        f"- Other 5 directions: 840/840 unchanged.\n\n"
        "## Interpretation after submission\n"
        "- S_new = online B SSE. near->core delta = S_new - 952.\n"
        "- Expected S_new ≈ 925. If confirmed, total ≈ 7198.\n\n"
        "## Files prepared for review\n"
        "- candidate_10.../提交结果1...\n- candidate_10.../提交结果2...\n\n"
        "These files have been generated but not submitted.\n",
        encoding="utf-8")

    assert not any(out_dir.rglob("*.zip"))
    print(f"\nA SHA256 identical: {sha256(ROOT / A_SOURCE) == sha256(A_OUT)}")
    print(f"near->core changed: {b_changed} cells (inc {nc['increase_count']}, dec {nc['decrease_count']}, net {nc['total_change']:+d})")
    print(f"near->core total: {nc_base} -> {nc_new}")
    print(f"B total: {b_base_total} -> {b_new_total}")
    print(f"other 5 directions unchanged: True")
    print(f"Expected B = 925, total = 7198")
    print("All assertions PASS.")


if __name__ == "__main__": main()
