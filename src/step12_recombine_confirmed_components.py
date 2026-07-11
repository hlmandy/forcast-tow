"""Step 12 — Recombine confirmed A and B submission files into candidate_04.

Copies the historically-submitted A file (A SSE=6063, from the
daily_count_constrained submit_02 dir) and the step11 candidate_01_main B file
(B SSE=1762) byte-for-byte into outputs/step12_recombined_candidate/, with no
reordering, no recomputation, and no modification. Asserts content/order/SHA256
identity with the sources.
"""

import hashlib
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

A_SOURCE = ROOT / "outputs/20260710_144658_daily_count_constrained_csv/submit_02_A_daily_total_constraint25_B085/提交结果1_区域活跃拖轮数量.csv"
B_SOURCE = ROOT / "outputs/step11_final_validation_candidates/submissions/candidate_01_main/提交结果2_圈层间拖轮迁移量.csv"

OUT_DIR = ROOT / "outputs/step12_recombined_candidate"
A_OUT = OUT_DIR / "提交结果1_区域活跃拖轮数量.csv"
B_OUT = OUT_DIR / "提交结果2_圈层间拖轮迁移量.csv"

# Expected A daily totals (2018-01-25 .. 2018-01-31), verified from source.
EXPECTED_A_DAILY = {
    "2018-01-25": 238, "2018-01-26": 262, "2018-01-27": 288,
    "2018-01-28": 266, "2018-01-29": 258, "2018-01-30": 258, "2018-01-31": 247,
}
EXPECTED_A_TOTAL = 1817


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def daily_totals(df: pd.DataFrame) -> dict:
    """Map YYYY-MM-DD date string -> sum of vessel_count, preserving file row order."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["time_window"]).dt.strftime("%Y-%m-%d")
    g = df.groupby("date", sort=False)["vessel_count"].sum()
    return {d: int(v) for d, v in g.items()}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Read sources (for verification only; copy is byte-for-byte) ---
    a_src = pd.read_csv(A_SOURCE)
    b_src = pd.read_csv(B_SOURCE)

    # --- A assertions ---
    assert len(a_src) == 504, f"A rows {len(a_src)} != 504"
    a_total = int(a_src["vessel_count"].sum())
    assert a_total == EXPECTED_A_TOTAL, f"A total {a_total} != {EXPECTED_A_TOTAL}"
    a_daily = daily_totals(a_src)
    for d, v in EXPECTED_A_DAILY.items():
        assert a_daily.get(d) == v, f"A daily {d}: {a_daily.get(d)} != {v}"

    # --- B assertions ---
    assert len(b_src) == 1008, f"B rows {len(b_src)} != 1008"

    # --- Byte-for-byte copy (preserves order, encoding, everything) ---
    shutil.copyfile(A_SOURCE, A_OUT)
    shutil.copyfile(B_SOURCE, B_OUT)

    # --- SHA256 identity ---
    a_sha_src, a_sha_out = sha256(A_SOURCE), sha256(A_OUT)
    b_sha_src, b_sha_out = sha256(B_SOURCE), sha256(B_OUT)
    assert a_sha_src == a_sha_out, f"A SHA mismatch: {a_sha_src} vs {a_sha_out}"
    assert b_sha_src == b_sha_out, f"B SHA mismatch: {b_sha_src} vs {b_sha_out}"

    # --- Normalized content identity (re-read output, compare to source) ---
    a_out = pd.read_csv(A_OUT)
    b_out = pd.read_csv(B_OUT)
    assert a_src.equals(a_out), "A normalized content/order mismatch"
    assert b_src.equals(b_out), "B normalized content/order mismatch"

    # --- daily_totals.csv ---
    b_daily = daily_totals(b_src)
    dt_rows = []
    for d, v in a_daily.items():
        dt_rows.append({"task": "A", "date": d, "prediction_total": v})
    for d, v in b_daily.items():
        dt_rows.append({"task": "B", "date": d, "prediction_total": v})
    dt = pd.DataFrame(dt_rows)
    dt.to_csv(OUT_DIR / "daily_totals.csv", index=False)

    # --- component_checks.csv ---
    checks = pd.DataFrame([
        {"task": "A", "source": str(A_SOURCE.relative_to(ROOT)), "output": str(A_OUT.relative_to(ROOT)),
         "row_count": len(a_src), "sha256_equal": a_sha_src == a_sha_out,
         "normalized_equal": a_src.equals(a_out)},
        {"task": "B", "source": str(B_SOURCE.relative_to(ROOT)), "output": str(B_OUT.relative_to(ROOT)),
         "row_count": len(b_src), "sha256_equal": b_sha_src == b_sha_out,
         "normalized_equal": b_src.equals(b_out)},
    ])
    checks.to_csv(OUT_DIR / "component_checks.csv", index=False)

    # --- summary.md ---
    b_total = int(b_src["vessel_count"].sum())
    (OUT_DIR / "summary.md").write_text(
        f"# Step 12 — Recombined candidate_04\n\n"
        f"Combines two already-submitted files with no retraining, reordering, or modification.\n\n"
        f"## Sources\n\n"
        f"- **A** (A SSE=6063 online): `{A_SOURCE.relative_to(ROOT)}` — copied byte-for-byte.\n"
        f"- **B** (B SSE=1762 online, from step11 candidate_01_main): `{B_SOURCE.relative_to(ROOT)}` — copied byte-for-byte.\n\n"
        f"## Verification\n\n"
        f"- A rows: {len(a_src)} (expected 504). **PASS**\n"
        f"- A total vessel_count: {a_total} (expected {EXPECTED_A_TOTAL}). **PASS**\n"
        f"- A daily totals match the 7 expected values exactly. **PASS**\n"
        f"- B rows: {len(b_src)} (expected 1008). **PASS**\n"
        f"- A SHA256 source == output: {a_sha_src == a_sha_out}. **PASS**\n"
        f"- B SHA256 source == output: {b_sha_src == b_sha_out}. **PASS**\n"
        f"- A normalized content/order identical: {a_src.equals(a_out)}. **PASS**\n"
        f"- B normalized content/order identical: {b_src.equals(b_out)}. **PASS**\n\n"
        f"## Daily totals (B)\n\n"
        f"B total = {b_total}.\n\n"
        f"## Note\n\n"
        f"This candidate is a pure recombination of two confirmed submission files. "
        f"No model was retrained, no value recomputed, no row reordered.\n",
        encoding="utf-8",
    )

    print(f"A source: {A_SOURCE.relative_to(ROOT)}")
    print(f"B source: {B_SOURCE.relative_to(ROOT)}")
    print(f"A total: {a_total} | B total: {b_total}")
    print(f"A SHA256: {a_sha_out}")
    print(f"B SHA256: {b_sha_out}")
    print(f"Output dir: {OUT_DIR.relative_to(ROOT)}")
    print("All assertions PASS.")


if __name__ == "__main__":
    main()
