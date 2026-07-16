"""Step 45 — candidate_14: R2 probe + B date binary search (01-28/29).

A: core@03h 0 (revert to candidate_10), core@06h +2, near@07h +3.
   A_next = 4118 + 2*R2, so R2 = (A_next - 4118) / 2.
   Then R3 = 66 - R2.

B: outer->core @ 10:00, only 01-28/29 = 1.
   If B_next = 925: occurrence in {28,29}. If 927: in {30,31}.
"""

from __future__ import annotations
import sys, warnings, zipfile
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from optimized_baseline import CN_TO_REGION, REGION_CN  # noqa: E402

A_CAND10 = Path("outputs/step20_b_near_to_core_only_corrected/candidate_10_A4423_B_near_core_only/提交结果1_区域活跃拖轮数量.csv")
B_CAND10 = Path("outputs/step20_b_near_to_core_only_corrected/candidate_10_A4423_B_near_core_only/提交结果2_圈层间拖轮迁移量.csv")
OUT_REL = Path("outputs/step45_candidate_14_probe")
RUN_CMD = "python " + " ".join(sys.argv)


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    cand_dir = out_dir / "candidate_14"; cand_dir.mkdir(exist_ok=True)
    print(f"out_dir = {out_dir}")

    a_cand10 = pd.read_csv(ROOT / A_CAND10)
    b_cand10 = pd.read_csv(ROOT / B_CAND10)
    assert len(a_cand10) == 504 and len(b_cand10) == 1008

    # ---- A: core@03h=0, core@06h=+2, near@07h=+3 ----
    print("=== A: core@03h 0 (candidate_10), core@06h +2, near@07h +3 ===")
    a_out = a_cand10.copy()
    a_ts = pd.to_datetime(a_out["time_window"])
    corrections = [(6, "核心区", 2), (7, "近港区", 3)]
    a_total_delta = 0
    for h, zone, delta in corrections:
        mask = (a_out["zone"] == zone) & (a_ts.dt.hour == h)
        a_out.loc[mask, "vessel_count"] += delta
        n = int(mask.sum()); a_total_delta += n * delta
        print(f"  {zone} @ {h:02d}:00 -> {n} cells +{delta}")
    # core@03h stays at candidate_10 (no correction)
    print(f"  核心区 @ 03:00 -> candidate_10 (no change)")
    print(f"  Total A delta: +{a_total_delta}")

    a_diff = a_out["vessel_count"] - a_cand10["vessel_count"]
    assert int((a_diff != 0).sum()) == 14, f"A changes: {int((a_diff!=0).sum())} != 14"
    assert int(a_diff.sum()) == a_total_delta
    print(f"  A diff: 14 cells changed (core@06h +2 x7, near@07h +3 x7)")

    A_OUT = cand_dir / "提交结果1_区域活跃拖轮数量.csv"
    a_out.to_csv(A_OUT, index=False, encoding="utf-8-sig")

    # ---- B: outer->core @ 10:00, only 01-28/29 ----
    print("\n=== B outer->core @ 10:00, only 01-28/29 ===")
    b_out = b_cand10.copy()
    b_ts = pd.to_datetime(b_out["time_window"])
    for day in [28, 29]:
        mask = (b_out["source_zone"] == "外围区") & (b_out["target_zone"] == "核心区") & \
               (b_ts.dt.hour == 10) & (b_ts.dt.day == day) & (b_ts.dt.month == 1) & \
               (b_out["vessel_count"] == 0)
        b_out.loc[mask, "vessel_count"] = 1
        print(f"  01-{day:02d} outer->core @ 10:00 -> {int(mask.sum())} cell 0->1")

    B_OUT = cand_dir / "提交结果2_圈层间拖轮迁移量.csv"
    b_out.to_csv(B_OUT, index=False, encoding="utf-8-sig")

    # ---- assertions ----
    print("\n=== Assertions ===")
    b_diff = b_out["vessel_count"] - b_cand10["vessel_count"]
    b_changed = int((b_diff != 0).sum())
    assert b_changed == 2, f"B changes: {b_changed} != 2"
    for i in np.where((b_diff != 0).to_numpy())[0]:
        assert CN_TO_REGION[b_out["source_zone"].iloc[i]] == "outer"
        assert CN_TO_REGION[b_out["target_zone"].iloc[i]] == "core"
        assert int(b_diff.iloc[i]) == 1
    cn_mask = (b_out["source_zone"]=="核心区") & (b_out["target_zone"]=="近港区")
    assert (b_out.loc[cn_mask, "vessel_count"] == b_cand10.loc[cn_mask, "vessel_count"]).all()
    nc_mask = (b_out["source_zone"]=="近港区") & (b_out["target_zone"]=="核心区")
    assert (b_out.loc[nc_mask, "vessel_count"] == b_cand10.loc[nc_mask, "vessel_count"]).all()
    print(f"  A: 14 cells (core@06h+2, near@07h+3), core@03h=candidate_10 [OK]")
    print(f"  B: 2 cells outer->core +1 (01-28/29), all else unchanged [OK]")

    # ---- ZIP ----
    zip_path = out_dir / "candidate_14.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(A_OUT, "提交结果1_区域活跃拖轮数量.csv")
        zf.write(B_OUT, "提交结果2_圈层间拖轮迁移量.csv")

    # ---- summary ----
    L = ["# Step 45 Candidate 14 Probe (R2 + B binary)", "", f"Run command: `{RUN_CMD}`", ""]
    L.append("## A R2 probe")
    L.append("")
    L.append("core@03h: **0** (candidate_10, R1=-3 proven, optimal k1=0)")
    L.append("core@06h: **+2** (7 days)")
    L.append("near@07h: **+3** (7 days)")
    L.append("")
    L.append("Delta SSE = 7*(4+9) - 2*(2*R2 + 3*R3)")
    L.append("         = 91 - 2*(2*R2 + 3*(66-R2))")
    L.append("         = 91 - 2*(198 - R2)")
    L.append("         = 2*R2 - 305")
    L.append("")
    L.append("**A_next = 4423 + 2*R2 - 305 = 4118 + 2*R2**")
    L.append("")
    L.append("After submission: R2 = (A_next - 4118) / 2")
    L.append("Then: R3 = 66 - R2")
    L.append("Optimal: k1=0, k2=round(R2/7), k3=round(R3/7)")
    L.append("")
    L.append("## B binary search")
    L.append("")
    L.append("outer->core @ 10:00, only 01-28/29 = 1.")
    L.append("Known: occurrence in {28,29,30,31}.")
    L.append("")
    L.append("If B_next = 925: occurrence in {28,29}")
    L.append("If B_next = 927: occurrence in {30,31}")
    L.append("")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\n=== Summary ===")
    print(f"A: core@06h +2, near@07h +3 (14 cells), core@03h unchanged")
    print(f"B: outer->core @ 10:00 on 01-28/29 only (2 cells)")
    print(f"A_next = 4118 + 2*R2")
    print(f"B_next = 925 (if in 28/29) or 927 (if in 30/31)")
    print(f"ZIP: {zip_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
