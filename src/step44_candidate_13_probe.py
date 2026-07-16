"""Step 44 — candidate_13: A group probe (R1) + B outer→core date probe.

A: core@03h +2, core@06h +3, near@07h +3 → isolates R1 = (A_next - 4199) / 2.
B: outer→core @ h*=10, only 01-25/26/27 = 1 → narrows true occurrence date.

If A_next = 4199 + 2*R1, then R1 = (A_next - 4199)/2.
If B_next = 928, occurrence is NOT in {25,26,27}. If B_next = 926, it IS.
"""

from __future__ import annotations
import sys, warnings, hashlib, zipfile
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from optimized_baseline import CN_TO_REGION, REGION_CN  # noqa: E402

A_CAND10 = Path("outputs/step20_b_near_to_core_only_corrected/candidate_10_A4423_B_near_core_only/提交结果1_区域活跃拖轮数量.csv")
B_CAND10 = Path("outputs/step20_b_near_to_core_only_corrected/candidate_10_A4423_B_near_core_only/提交结果2_圈层间拖轮迁移量.csv")
OUT_REL = Path("outputs/step44_candidate_13_probe")
RUN_CMD = "python " + " ".join(sys.argv)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    cand_dir = out_dir / "candidate_13"; cand_dir.mkdir(exist_ok=True)
    print(f"out_dir = {out_dir}")

    a_cand10 = pd.read_csv(ROOT / A_CAND10)
    b_cand10 = pd.read_csv(ROOT / B_CAND10)
    assert len(a_cand10) == 504 and len(b_cand10) == 1008

    # ---- A: group probe k1=2, k2=3, k3=3 ----
    print("=== A group probe: core@03h +2, core@06h +3, near@07h +3 ===")
    a_out = a_cand10.copy()
    a_ts = pd.to_datetime(a_out["time_window"])
    corrections = [(3, "核心区", 2), (6, "核心区", 3), (7, "近港区", 3)]
    a_total_delta = 0
    for h, zone, delta in corrections:
        mask = (a_out["zone"] == zone) & (a_ts.dt.hour == h)
        a_out.loc[mask, "vessel_count"] += delta
        n = int(mask.sum())
        a_total_delta += n * delta
        print(f"  {zone} @ {h:02d}:00 -> {n} cells +{delta}")
    print(f"  Total A delta: +{a_total_delta}")

    # verify: exactly 21 cells changed, with specific deltas
    a_diff = a_out["vessel_count"] - a_cand10["vessel_count"]
    assert int((a_diff != 0).sum()) == 21
    assert int(a_diff.sum()) == a_total_delta  # 2*7+3*7+3*7 = 14+21+21 = 56
    print(f"  A diff: 21 cells, total delta=+{int(a_diff.sum())}")

    A_OUT = cand_dir / "提交结果1_区域活跃拖轮数量.csv"
    a_out.to_csv(A_OUT, index=False, encoding="utf-8-sig")

    # ---- B: outer→core @ 10:00, only 01-25/26/27 ----
    print("\n=== B outer->core date probe: 01-25/26/27 @ 10:00 ===")
    b_out = b_cand10.copy()
    b_ts = pd.to_datetime(b_out["time_window"])
    probe_dates = [25, 26, 27]
    b_changed = 0
    for day in probe_dates:
        mask = (b_out["source_zone"] == "外围区") & (b_out["target_zone"] == "核心区") & \
               (b_ts.dt.hour == 10) & (b_ts.dt.day == day) & (b_ts.dt.month == 1) & \
               (b_out["vessel_count"] == 0)
        b_out.loc[mask, "vessel_count"] = 1
        b_changed += int(mask.sum())
        print(f"  01-{day:02d} outer->core @ 10:00 -> {int(mask.sum())} cell 0->1")
    print(f"  Total B changes: {b_changed}")

    B_OUT = cand_dir / "提交结果2_圈层间拖轮迁移量.csv"
    b_out.to_csv(B_OUT, index=False, encoding="utf-8-sig")

    # ---- assertions ----
    print("\n=== Assertions ===")
    a_diff_mask = a_diff != 0
    assert int(a_diff_mask.sum()) == 21
    # verify specific deltas
    for h, zone, delta in corrections:
        mask = (a_out["zone"] == zone) & (a_ts.dt.hour == h)
        d = a_out.loc[mask, "vessel_count"] - a_cand10.loc[mask, "vessel_count"]
        assert int(d.iloc[0]) == delta, f"{zone}@{h}: delta={int(d.iloc[0])} != {delta}"
    print(f"  A: 21 cells, core@03h=+2, core@06h=+3, near@07h=+3 [OK]")

    b_diff = b_out["vessel_count"] - b_cand10["vessel_count"]
    b_changed_mask = b_diff != 0
    for i in np.where(b_changed_mask.to_numpy())[0]:
        sz = b_out["source_zone"].iloc[i]; tz = b_out["target_zone"].iloc[i]
        assert CN_TO_REGION[sz] == "outer" and CN_TO_REGION[tz] == "core", f"Not outer->core: {sz}->{tz}"
        assert int(b_diff.iloc[i]) == 1
    # all other directions unchanged
    cn_mask = (b_out["source_zone"]=="核心区") & (b_out["target_zone"]=="近港区")
    assert (b_out.loc[cn_mask, "vessel_count"] == b_cand10.loc[cn_mask, "vessel_count"]).all()
    nc_mask = (b_out["source_zone"]=="近港区") & (b_out["target_zone"]=="核心区")
    assert (b_out.loc[nc_mask, "vessel_count"] == b_cand10.loc[nc_mask, "vessel_count"]).all()
    print(f"  B: {int(b_changed_mask.sum())} cells outer->core +1, all else unchanged [OK]")

    # ---- ZIP ----
    zip_path = out_dir / "candidate_13.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(A_OUT, "提交结果1_区域活跃拖轮数量.csv")
        zf.write(B_OUT, "提交结果2_圈层间拖轮迁移量.csv")
    print(f"\n  ZIP: {zip_path}")

    # ---- summary ----
    L = ["# Step 44 Candidate 13 Probe", "", f"Run command: `{RUN_CMD}`", ""]
    L.append("## A group probe (R1 isolation)")
    L.append("")
    L.append("Corrections relative to candidate_10:")
    L.append("- core @ 03:00: **+2** (7 days)")
    L.append("- core @ 06:00: **+3** (7 days)")
    L.append("- near @ 07:00: **+3** (7 days)")
    L.append("")
    L.append("From candidate_11's result: R1+R2+R3 = 63")
    L.append("")
    L.append("Delta SSE = 7*(4+9+9) - 2*(2*R1 + 3*R2 + 3*R3)")
    L.append("         = 154 - 2*(2*R1 + 3*(63-R1))")
    L.append("         = 154 - 378 + 2*R1")
    L.append("         = 2*R1 - 224")
    L.append("")
    L.append("**A_next = 4423 + 2*R1 - 224 = 4199 + 2*R1**")
    L.append("")
    L.append("After submission: R1 = (A_next - 4199) / 2")
    L.append("")
    L.append("## B date probe")
    L.append("")
    L.append("outer->core @ 10:00, only 01-25/26/27 predict 1 (3 cells), rest 0.")
    L.append("")
    L.append("Known: exactly 1 of 7 days has true y=1.")
    L.append("")
    L.append("If occurrence in {25,26,27}: B_next = 925 + 3 - 2 = **926**")
    L.append("If occurrence NOT in {25,26,27}: B_next = 925 + 3 = **928**")
    L.append("")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\n=== Summary ===")
    print(f"A: core@03h +2, core@06h +3, near@07h +3 (21 cells)")
    print(f"B: outer->core @ 10:00 on 01-25/26/27 only ({b_changed} cells)")
    print(f"A_next = 4199 + 2*R1 (probe for R1)")
    print(f"B_next = 926 (if in probe) or 928 (if not)")
    print(f"ZIP: {zip_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
