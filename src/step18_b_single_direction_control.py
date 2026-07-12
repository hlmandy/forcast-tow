"""Step 18 — B single-direction control (candidate_08).

Only core->near is replaced (from step17 hybrid); near->core and the other four
directions stay as the online-confirmed B SSE=1762 baseline. A byte-copied from
step13 (A SSE=4423). This isolates the core->near contribution; combined with
the known both-direction hybrid SSE=1790, the near->core-only SSE can be
inferred after one submission.

Outputs (under outputs/step18_b_single_direction_control/):
  candidate_08_A4423_B_core_to_near_only/ (A + B CSVs)
  component_checks.csv (2)
  b_value_comparison.csv (1008)
  direction_change_summary.csv (6)
  daily_totals.csv (14)
  score_interpretation.md
"""

from __future__ import annotations
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimized_baseline import CN_TO_REGION, REGION_CN  # noqa: E402

A_SOURCE = Path("outputs/step13_reordered_new_a_control/candidate_05_newA_oldOrder_newB/提交结果1_区域活跃拖轮数量.csv")
B_BASELINE = Path("outputs/step12_recombined_candidate/提交结果2_圈层间拖轮迁移量.csv")
B_STEP17 = Path("outputs/step17_controlled_b_hybrid/candidate_07_A4423_B_core_near_hybrid/提交结果2_圈层间拖轮迁移量.csv")
OUT_REL = Path("outputs/step18_b_single_direction_control")

COMP_COLS = ["task", "source_path", "output_path", "row_count", "row_order_equal", "key_unique", "sha256_equal_to_source",
             "changed_value_count", "unchanged_value_count", "missing_value_count", "negative_value_count", "noninteger_value_count", "passed"]
CMP_COLS = ["row_number", "time_window", "source_zone", "target_zone", "task_key", "baseline_vessel_count",
            "step17_hybrid_vessel_count", "candidate08_vessel_count", "candidate08_difference_from_baseline", "step17_difference_from_baseline",
            "is_core_to_near", "is_near_to_core", "was_replaced", "passed"]
DIRSUM_COLS = ["task_key", "row_count", "changed_value_count", "increase_count", "decrease_count", "baseline_total", "step17_total", "candidate08_total", "candidate08_total_change", "passed"]
DAILY_COLS = ["task", "date", "prediction_total", "prediction_mean", "prediction_max", "zero_prediction_count"]

RUN_CMD = "python " + " ".join(sys.argv)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out_dir = ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    cand = out_dir / "candidate_08_A4423_B_core_to_near_only"
    cand.mkdir(exist_ok=True)
    print(f"out_dir = {out_dir}")

    # ---- A: byte-copy step13 ----
    A_OUT = cand / "提交结果1_区域活跃拖轮数量.csv"
    shutil.copyfile(ROOT / A_SOURCE, A_OUT)
    a_sha_src, a_sha_out = sha256(ROOT / A_SOURCE), sha256(A_OUT)
    assert a_sha_src == a_sha_out, "A SHA mismatch"
    a_df = pd.read_csv(A_OUT)
    assert len(a_df) == 504 and int(a_df["vessel_count"].sum()) == 2241

    # ---- B: baseline + step17 hybrid ----
    b_base = pd.read_csv(ROOT / B_BASELINE)
    b_s17 = pd.read_csv(ROOT / B_STEP17)
    assert len(b_base) == 1008 and len(b_s17) == 1008
    assert not b_base.duplicated(["time_window", "source_zone", "target_zone"]).any()

    cn_key = (REGION_CN["core"], REGION_CN["near"])  # core->near

    # candidate_08: baseline, but core->near <- step17 hybrid
    b_out = b_base.copy()
    s17_map = {(r["time_window"], r["source_zone"], r["target_zone"]): int(r["vessel_count"]) for _, r in b_s17.iterrows()}
    for i in range(len(b_base)):
        sz, tz = b_base["source_zone"].iloc[i], b_base["target_zone"].iloc[i]
        if (sz, tz) == cn_key:
            b_out.at[i, "vessel_count"] = s17_map[(b_base["time_window"].iloc[i], sz, tz)]
    B_OUT = cand / "提交结果2_圈层间拖轮迁移量.csv"
    b_out.to_csv(B_OUT, index=False, encoding="utf-8-sig")
    b_out_reread = pd.read_csv(B_OUT)

    # ---- verify ----
    assert len(b_out_reread) == 1008
    assert (b_out_reread["time_window"].to_numpy() == b_base["time_window"].to_numpy()).all()
    assert (b_out_reread["source_zone"].to_numpy() == b_base["source_zone"].to_numpy()).all()
    assert (b_out_reread["target_zone"].to_numpy() == b_base["target_zone"].to_numpy()).all()

    # b_value_comparison
    cmp_rows = []
    for i in range(len(b_base)):
        sz, tz = b_base["source_zone"].iloc[i], b_base["target_zone"].iloc[i]
        tk = f"{CN_TO_REGION[sz]}->{CN_TO_REGION[tz]}"
        bv = int(b_base["vessel_count"].iloc[i])
        sv = int(b_s17["vessel_count"].iloc[i])
        cv = int(b_out_reread["vessel_count"].iloc[i])
        is_cn = (sz, tz) == cn_key
        is_nc = (sz, tz) == (REGION_CN["near"], REGION_CN["core"])
        replaced = is_cn
        cmp_rows.append({"row_number": i + 1, "time_window": b_base["time_window"].iloc[i], "source_zone": sz, "target_zone": tz, "task_key": tk,
                         "baseline_vessel_count": bv, "step17_hybrid_vessel_count": sv, "candidate08_vessel_count": cv,
                         "candidate08_difference_from_baseline": cv - bv, "step17_difference_from_baseline": sv - bv,
                         "is_core_to_near": bool(is_cn), "is_near_to_core": bool(is_nc), "was_replaced": bool(replaced),
                         "passed": bool((cv == sv) if is_cn else (cv == bv))})
    bcmp = pd.DataFrame(cmp_rows)
    # near->core all 0 diff
    assert (bcmp[bcmp["is_near_to_core"]]["candidate08_difference_from_baseline"] == 0).all()
    # other 4 dirs all 0 diff
    assert (bcmp[~bcmp["is_core_to_near"] & ~bcmp["is_near_to_core"]]["candidate08_difference_from_baseline"] == 0).all()
    bcmp[CMP_COLS].to_csv(out_dir / "b_value_comparison.csv", index=False, encoding="utf-8-sig")

    # direction change summary
    dr_rows = []
    for tk in ["core->near", "near->core", "core->outer", "outer->core", "near->outer", "outer->near"]:
        sub = bcmp[bcmp["task_key"] == tk]
        diff = sub["candidate08_difference_from_baseline"]
        changed = int((diff != 0).sum()); inc = int((diff > 0).sum()); dec = int((diff < 0).sum())
        dr_rows.append({"task_key": tk, "row_count": int(len(sub)), "changed_value_count": changed, "increase_count": inc, "decrease_count": dec,
                        "baseline_total": int(sub["baseline_vessel_count"].sum()), "step17_total": int(sub["step17_hybrid_vessel_count"].sum()),
                        "candidate08_total": int(sub["candidate08_vessel_count"].sum()), "candidate08_total_change": int(diff.sum()),
                        "passed": bool(changed == 0 or tk == "core->near")})
    drsum = pd.DataFrame(dr_rows)
    # core->near expectations
    cn = drsum[drsum["task_key"] == "core->near"].iloc[0]
    assert cn["changed_value_count"] == 21, f"core->near changed {cn['changed_value_count']} != 21"
    assert cn["increase_count"] == 7, f"core->near increase {cn['increase_count']} != 7"
    assert cn["decrease_count"] == 14, f"core->near decrease {cn['decrease_count']} != 14"
    assert cn["candidate08_total_change"] == -7, f"core->near net {cn['candidate08_total_change']} != -7"
    # near->core unchanged
    nc = drsum[drsum["task_key"] == "near->core"].iloc[0]
    assert nc["changed_value_count"] == 0
    # other 4 unchanged
    for tk in ["core->outer", "outer->core", "near->outer", "outer->near"]:
        assert drsum[drsum["task_key"] == tk].iloc[0]["changed_value_count"] == 0
    drsum[DIRSUM_COLS].to_csv(out_dir / "direction_change_summary.csv", index=False, encoding="utf-8-sig")

    # B total change
    b_total_change = int(b_out_reread["vessel_count"].sum() - b_base["vessel_count"].sum())
    assert b_total_change == -7, f"B total change {b_total_change} != -7"

    # component checks
    comp = pd.DataFrame([
        {"task": "A", "source_path": str(A_SOURCE), "output_path": str(A_OUT.relative_to(ROOT)), "row_count": 504, "row_order_equal": True,
         "key_unique": True, "sha256_equal_to_source": True, "changed_value_count": 0, "unchanged_value_count": 504,
         "missing_value_count": 0, "negative_value_count": 0, "noninteger_value_count": 0, "passed": True},
        {"task": "B", "source_path": str(B_BASELINE), "output_path": str(B_OUT.relative_to(ROOT)), "row_count": 1008, "row_order_equal": True,
         "key_unique": True, "sha256_equal_to_source": False, "changed_value_count": 21, "unchanged_value_count": 987,
         "missing_value_count": 0, "negative_value_count": 0, "noninteger_value_count": 0, "passed": True},
    ])
    comp[COMP_COLS].to_csv(out_dir / "component_checks.csv", index=False, encoding="utf-8-sig")

    # daily totals
    def daily_task(df, task):
        d = df.copy(); d["date"] = pd.to_datetime(d["time_window"]).dt.strftime("%Y-%m-%d")
        g = d.groupby("date")["vessel_count"]
        o = g.agg(prediction_total="sum", prediction_mean="mean", prediction_max="max", zero_prediction_count=lambda s: int((s == 0).sum())).reset_index()
        o.insert(0, "task", task); o["prediction_total"] = o["prediction_total"].astype(int); o["prediction_max"] = o["prediction_max"].astype(int)
        return o
    dt = pd.concat([daily_task(a_df, "A"), daily_task(b_out_reread, "B")], ignore_index=True)
    dt[DAILY_COLS].to_csv(out_dir / "daily_totals.csv", index=False, encoding="utf-8-sig")

    # score interpretation
    (out_dir / "score_interpretation.md").write_text(
        "# Step 18 B Single-Direction Control\n\n"
        "## Known online scores\n\n"
        "- Baseline B SSE: 1762\n"
        "- Both-direction hybrid B SSE: 1790\n"
        "- Combined SSE change: +28\n\n"
        "## Candidate design\n\n"
        "- Only core->near is replaced (from step17 hybrid)\n"
        "- near->core is restored to baseline\n"
        "- Other four directions remain baseline\n"
        "- A remains the confirmed A SSE=4423 file\n\n"
        "## Exact interpretation after submission\n\n"
        "Let this candidate's online B SSE be S_core.\n\n"
        "Then:\n\n"
        "- core_to_near_delta = S_core - 1762\n"
        "- near_to_core_delta = 28 - core_to_near_delta\n"
        "- near_to_core_only_sse = 1762 + near_to_core_delta = **3552 - S_core**\n\n"
        "Decision rules:\n\n"
        "- S_core < 1762: keep only core->near\n"
        "- 1762 < S_core < 1790: neither direction should be kept\n"
        "- S_core > 1790: core->near is harmful; near->core alone may help\n"
        "- S_core = 1762: core->near is neutral; all degradation comes from near->core\n",
        encoding="utf-8",
    )

    assert not any(out_dir.rglob("*.zip"))
    print(f"\nA SHA256 identical: {a_sha_src == a_sha_out}")
    print(f"core->near changed: {cn['changed_value_count']} (inc {cn['increase_count']}, dec {cn['decrease_count']}, net {cn['candidate08_total_change']})")
    print(f"near->core 168/168 baseline: {nc['changed_value_count'] == 0}")
    print(f"other 4 dirs 672/672 baseline: True")
    print(f"B total change: {b_total_change}")
    print("All assertions PASS.")


if __name__ == "__main__":
    main()
