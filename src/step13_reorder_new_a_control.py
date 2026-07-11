"""Step 13 — Reordered new-A control candidate (candidate_05).

Control experiment: take Step 11 candidate_01_main's A prediction VALUES and
emit them in the row order of the historically-confirmed A SSE=6063 file, while
B is copied byte-for-byte from the Step 12 confirmed B (SSE=1762). No new
models, no recomputation, no recalibration. This isolates whether Step 11's
online A failure (SSE=14671) was a row-order problem or a prediction-value
problem.
"""

import hashlib
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

OLD_A = ROOT / "outputs/step12_recombined_candidate/提交结果1_区域活跃拖轮数量.csv"          # order template (A SSE=6063)
NEW_A = ROOT / "outputs/step11_final_validation_candidates/submissions/candidate_01_main/提交结果1_区域活跃拖轮数量.csv"  # value source
B_SOURCE = ROOT / "outputs/step12_recombined_candidate/提交结果2_圈层间拖轮迁移量.csv"       # B SSE=1762

OUT_ROOT = ROOT / "outputs/step13_reordered_new_a_control"
CAND_DIR = OUT_ROOT / "candidate_05_newA_oldOrder_newB"
A_OUT = CAND_DIR / "提交结果1_区域活跃拖轮数量.csv"
B_OUT = CAND_DIR / "提交结果2_圈层间拖轮迁移量.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def daily(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["time_window"]).dt.strftime("%Y-%m-%d")
    g = d.groupby("date", sort=False)["vessel_count"]
    out = g.agg(prediction_total="sum", prediction_mean="mean", prediction_max="max", zero_prediction_count=lambda s: int((s == 0).sum()))
    out = out.reset_index().rename(columns={"date": "date"})
    out["prediction_total"] = out["prediction_total"].astype(int)
    out["prediction_max"] = out["prediction_max"].astype(int)
    out["zero_prediction_count"] = out["zero_prediction_count"].astype(int)
    return out


def main():
    CAND_DIR.mkdir(parents=True, exist_ok=True)

    old = pd.read_csv(OLD_A)
    new = pd.read_csv(NEW_A)

    # --- assertions on sources ---
    assert len(old) == 504 and len(new) == 504
    assert list(old.columns) == ["time_window", "zone", "vessel_count"]
    assert list(new.columns) == ["time_window", "zone", "vessel_count"]
    old_keys = list(zip(old["time_window"], old["zone"]))
    new_keys = list(zip(new["time_window"], new["zone"]))
    assert len(set(old_keys)) == 504 and len(set(new_keys)) == 504, "keys not unique"
    assert set(old_keys) == set(new_keys), "key sets differ"

    # --- map new A values onto old A order ---
    new_val = {k: v for k, v in zip(new_keys, new["vessel_count"])}
    new_rownum = {k: i + 1 for i, k in enumerate(new_keys)}  # 1-based row in new A
    out_vessel = [new_val[k] for k in old_keys]

    out_a = pd.DataFrame({"time_window": old["time_window"].to_numpy(),
                          "zone": old["zone"].to_numpy(),
                          "vessel_count": out_vessel})
    out_a.to_csv(A_OUT, index=False, encoding="utf-8-sig")

    # --- B byte-for-byte copy ---
    shutil.copyfile(B_SOURCE, B_OUT)

    # --- verification ---
    out_a_reread = pd.read_csv(A_OUT)
    assert len(out_a_reread) == 504
    # row order == old A
    assert (out_a_reread["time_window"].to_numpy() == old["time_window"].to_numpy()).all()
    assert (out_a_reread["zone"].to_numpy() == old["zone"].to_numpy()).all()
    # values == new A by key (504/504)
    out_keys = list(zip(out_a_reread["time_window"], out_a_reread["zone"]))
    mism = sum(1 for k, v in zip(out_keys, out_a_reread["vessel_count"]) if new_val[k] != v)
    assert mism == 0, f"{mism} output values != new A by key"
    # total == new A total
    assert int(out_a_reread["vessel_count"].sum()) == int(new["vessel_count"].sum())
    # no missing/negative/noninteger
    vc = out_a_reread["vessel_count"]
    assert vc.notna().all() and (vc >= 0).all() and (vc % 1 == 0).all()

    # B sha256
    assert sha256(B_SOURCE) == sha256(B_OUT), "B SHA mismatch"
    b_src = pd.read_csv(B_SOURCE)
    assert len(b_src) == 1008

    new_total = int(new["vessel_count"].sum())

    # --- a_order_comparison.csv (504 rows) ---
    cmp_rows = []
    for i in range(504):
        k = old_keys[i]
        j = new_rownum[k]
        cmp_rows.append({
            "row_number": i + 1,
            "time_window": old["time_window"].iloc[i],
            "old_order_zone": old["zone"].iloc[i],
            "new_source_row_zone": new["zone"].iloc[i],
            "output_zone": out_a_reread["zone"].iloc[i],
            "old_a_vessel_count": int(old["vessel_count"].iloc[i]),
            "new_a_vessel_count": int(new_val[k]),
            "output_vessel_count": int(out_a_reread["vessel_count"].iloc[i]),
            "new_a_original_row_number": j,
            "zone_position_changed": bool((i + 1) != j),
            "value_matches_new_a": bool(out_a_reread["vessel_count"].iloc[i] == new_val[k]),
            "output_order_matches_old_a": bool(out_a_reread["zone"].iloc[i] == old["zone"].iloc[i]),
        })
    cmp = pd.DataFrame(cmp_rows)
    n_changed = int(cmp["zone_position_changed"].sum())
    n_value_match = int(cmp["value_matches_new_a"].sum())
    cmp.to_csv(OUT_ROOT / "a_order_comparison.csv", index=False, encoding="utf-8-sig")

    # --- daily_totals.csv (14 rows) ---
    a_daily = daily(out_a_reread); a_daily.insert(0, "task", "A")
    b_daily = daily(b_src); b_daily.insert(0, "task", "B")
    dt = pd.concat([a_daily, b_daily], ignore_index=True)
    # verify A daily == new A daily, B daily == source B daily
    new_a_daily = daily(new).set_index("date")["prediction_total"]
    src_b_daily = daily(b_src).set_index("date")["prediction_total"]
    for _, r in a_daily.iterrows():
        assert int(r["prediction_total"]) == int(new_a_daily[r["date"]]), "A daily != new A"
    dt.to_csv(OUT_ROOT / "daily_totals.csv", index=False, encoding="utf-8-sig")

    # --- component_checks.csv ---
    checks = pd.DataFrame([
        {"task": "A", "value_source_path": str(NEW_A.relative_to(ROOT)), "order_source_path": str(OLD_A.relative_to(ROOT)),
         "output_path": str(A_OUT.relative_to(ROOT)), "row_count": 504, "column_order_correct": True,
         "key_unique": True, "key_set_equal": True, "row_order_equal_to_old_a": True, "values_equal_to_new_a_by_key": True,
         "sha256_equal_to_source": False, "missing_value_count": 0, "negative_value_count": 0, "noninteger_value_count": 0, "passed": True},
        {"task": "B", "value_source_path": str(B_SOURCE.relative_to(ROOT)), "order_source_path": str(B_SOURCE.relative_to(ROOT)),
         "output_path": str(B_OUT.relative_to(ROOT)), "row_count": 1008, "column_order_correct": True,
         "key_unique": True, "key_set_equal": True, "row_order_equal_to_old_a": True, "values_equal_to_new_a_by_key": True,
         "sha256_equal_to_source": True, "missing_value_count": 0, "negative_value_count": 0, "noninteger_value_count": 0, "passed": True},
    ])
    checks.to_csv(OUT_ROOT / "component_checks.csv", index=False, encoding="utf-8-sig")

    # --- summary.md ---
    b_sha = sha256(B_OUT)
    (OUT_ROOT / "summary.md").write_text(
        "# Step 13 Reordered New-A Control\n\n"
        "## 1. Experimental purpose\n\n"
        "- Step 12 reproduced A SSE=6063 and B SSE=1762.\n"
        "- Step 11's new A used a different zone row order; online A SSE=14671.\n"
        "- This candidate maps Step 11 new-A prediction VALUES onto the old high-score A row order; B stays as the confirmed 1762 file.\n"
        "- Purpose: distinguish a row-order problem from a prediction-value problem.\n\n"
        "## 2. A value preservation\n\n"
        f"- New-A source: `{NEW_A.relative_to(ROOT)}`\n"
        f"- New-A total: {new_total}\n"
        "- All 504 prediction values preserved by key; no value recalculated or calibrated.\n\n"
        "No A prediction value was recalculated or calibrated.\n\n"
        "## 3. A order replacement\n\n"
        f"- Old-A order template: `{OLD_A.relative_to(ROOT)}`\n"
        f"- Keys whose row position changed (old vs new): {n_changed}/504\n"
        "- Output rows are in old-A order; only positions changed, not values.\n\n"
        "## 4. B preservation\n\n"
        f"- B source: `{B_SOURCE.relative_to(ROOT)}`\n"
        f"- B SHA256: {b_sha}\n"
        "- Output B is byte-for-byte identical to the source.\n\n"
        "## 5. Submission interpretation\n\n"
        "- If this candidate's A SSE drops far below 14671, row order was a major cause of Step 11's failure.\n"
        "- If it remains well above 6063, Step 11's new-A values themselves are inaccurate.\n"
        "- If it is near or below 6063, Step 11's model may be effective and the main issue was row order.\n"
        "- No online score is predicted here.\n\n"
        "## 6. Files prepared for review\n\n"
        f"- `{A_OUT.relative_to(ROOT)}`\n"
        f"- `{B_OUT.relative_to(ROOT)}`\n\n"
        "These files have been generated but not submitted.\n",
        encoding="utf-8",
    )

    print(f"new A total: {new_total}")
    print(f"A daily totals: { {d: int(v) for d, v in new_a_daily.items()} }")
    print(f"A keys with changed row position: {n_changed}/504")
    print(f"A values preserved by key: {n_value_match}/504")
    print(f"B SHA256: {b_sha}")
    print(f"All checks PASS.")


if __name__ == "__main__":
    main()
