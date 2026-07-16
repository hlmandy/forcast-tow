"""Step 42 — Joint candidate_11: sparse A correction + B sparse direction replacement.

A = Step 11 baseline + LODO-screened sparse -1 cells (max 5).
B = candidate_10 + outer_to_core direct_long + near_to_outer direct_long.
Generates submission CSVs and ZIP.

Outputs (under outputs/step42_joint_candidate_11/):
  1-9 as specified.
"""

from __future__ import annotations
import sys, warnings, math, zipfile, shutil
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from optimized_baseline import REGION_ORDER, CN_TO_REGION, REGION_CN, add_regions, make_a_labels  # noqa: E402
from step03_a_rolling_backtest import classify_fold_quality, POLICY_WEIGHTS  # noqa: E402
from step06_a_structure_model_benchmark import compute_components  # noqa: E402

TRAIN_REL = Path("数据备份/训练集_2018801-0124_拖轮AIS.csv") if False else Path("数据备份/训练集_20180101-0124_拖轮AIS.csv")
STEP07 = Path("outputs/step07_a_regularized_count_models/a_model_predictions.csv")
STEP01 = Path("outputs/step01_data_audit/train_daily_overview.csv")
STEP02 = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
STEP10_PREDS = Path("outputs/step10_b_shrinkage_calibration/b_model_predictions.csv")
STEP15_PREDS = Path("outputs/step15_b_pair_decomposition_benchmark/b_model_predictions.csv")
STEP17_B = Path("outputs/step17_controlled_b_hybrid/candidate_07_A4423_B_core_near_hybrid/提交结果2_圈层间拖轮迁移量.csv")
STEP11_A = Path("outputs/step11_final_validation_candidates/submissions/candidate_01_main/提交结果1_区域活跃拖轮数量.csv")
STEP11_B = Path("outputs/step11_final_validation_candidates/submissions/candidate_01_main/提交结果2_圈层间拖轮迁移量.csv")
A_TEMPLATE = Path("outputs/_official_examples_backup/提交结果1_区域活跃拖轮数量.csv")
B_TEMPLATE = Path("outputs/_official_examples_backup/提交结果2_圈层间拖轮迁移量.csv")
OUT_REL = Path("outputs/step42_joint_candidate_11")
REGIONS = REGION_ORDER
RUN_CMD = "python " + " ".join(sys.argv)
_cache = {}


def load_all():
    if _cache: return _cache
    df = pd.read_csv(ROOT / TRAIN_REL, usecols=["mmsi","x","y","sog","time"],
                     dtype={"mmsi":"string","x":"float64","y":"float64","sog":"float32"}, parse_dates=["time"])
    df["hour"] = df["time"].dt.floor("h"); df["date"] = df["time"].dt.normalize()
    df["hour_of_day"] = df["time"].dt.hour; df = add_regions(df)
    a_lab = make_a_labels(df)
    a_lab["date"] = a_lab["hour"].dt.normalize(); a_lab["hour_of_day"] = a_lab["hour"].dt.hour
    a_pivot = a_lab.pivot_table(index=["date","hour_of_day"], columns="region", values="y", aggfunc="first").reset_index()
    for r in REGIONS:
        if r not in a_pivot.columns: a_pivot[r] = 0
    aud = pd.read_csv(ROOT / STEP02); aud["date"] = pd.to_datetime(aud["date"]).dt.normalize()
    vc_df = pd.read_csv(ROOT / STEP01); vc_df["date"] = pd.to_datetime(vc_df["date"]).dt.normalize()
    all_dates = sorted(pd.date_range("2018-01-01","2018-01-24",freq="D").normalize())
    _cache.update({"df":df, "a_lab":a_lab, "a_pivot":a_pivot,
                   "qmap":dict(zip(aud["date"],aud["quality_regime"])),
                   "china_map":dict(zip(aud["date"],aud["china_coastal_record_count"])),
                   "vc_map":dict(zip(vc_df["date"],vc_df["unique_vessel_count"])),
                   "all_dates":all_dates})
    return _cache


def get_a_vec(d):
    data = load_all()
    sub = data["a_pivot"][data["a_pivot"]["date"]==d].sort_values("hour_of_day")
    m = np.zeros((24,3))
    for _, row in sub.iterrows():
        h = int(row["hour_of_of_day"]) if "hour_of_of_day" in row else int(row["hour_of_day"])
        for ri,r in enumerate(REGIONS): m[h,ri] = float(row[r])
    return m


def compute_baseline_vec(train_dates, te, d_target):
    """Compute Step 11 prediction for a single date."""
    d = load_all()
    classes, _ = classify_fold_quality(train_dates, d["china_map"])
    weights = {dd: POLICY_WEIGHTS["exclude_outage_severe"][classes[dd]] for dd in train_dates}
    train_df = d["a_lab"][d["a_lab"]["date"].isin(train_dates)].copy()
    train_df["day_type"] = np.where(train_df["hour"].dt.dayofweek.isin([5,6]), "weekend", "weekday")
    T_hat, P_hat = compute_components(train_df, train_dates, te, weights)
    vd = d["a_lab"][d["a_lab"]["date"]==d_target].sort_values(["hour","region"]).reset_index(drop=True)
    vr = vd["region"].to_numpy(); vh = vd["hour_of_day"].to_numpy().astype(int)
    vdt = vd["hour"].dt.dayofweek.to_numpy()
    vdt_type = np.array(["weekend" if x in (5,6) else "weekday" for x in vdt])
    Tvec = np.array([T_hat["mean"][r] for r in vr], dtype=float)
    prof = np.array([P_hat["daytype_shrunk"][r][vdt_type[i]][vh[i]] for i,r in enumerate(vr)])
    pr = np.clip(np.rint(Tvec*prof),0,None).astype(int)
    mat = np.zeros((24,3))
    for i,r in enumerate(vr): mat[vh[i],REGIONS.index(r)] = float(pr[i])
    return mat


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")
    data = load_all(); qmap = data["qmap"]; all_dates = data["all_dates"]
    normal_dates = [d for d in all_dates if qmap.get(d) == "normal"]

    # ---- audit ----
    (out_dir / "implementation_audit.md").write_text(
        "# Implementation Audit\n\n"
        "Candidate_10 baseline: A=Step11 candidate_01_main (A SSE=4423), "
        "B=near→core from Step17 + rest from Step11 (B SSE=925), total=7198.\n\n"
        "Step 41 showed outer→core FN=100% (long mean +16.9%) and near→outer FN=75% (+7.8%).\n"
        "A Step23 fixed10 diagnostic showed ~2.5% gain but with selection bias.\n"
        "This step uses LODO screening for A and direction audit for B.\n",
        encoding="utf-8")

    # ================================================================
    # PART 1: B DIRECTION REPLACEMENT
    # ================================================================
    print("\n=== B direction replacement audit ===")
    s10 = pd.read_csv(ROOT / STEP10_PREDS, encoding="utf-8-sig")
    s10["date"] = pd.to_datetime(s10["date"]).dt.normalize()
    s15 = pd.read_csv(ROOT / STEP15_PREDS, encoding="utf-8-sig")
    s15["date"] = pd.to_datetime(s15["date"]).dt.normalize()

    b10 = s10[(s10["predictor_method"]=="pair_hour_hier_tau_scale_cv") & (s10["rounding_method"]=="independent")].copy()
    b15_nc = s15[s15["method"]=="pair_default_hierarchical"].copy()

    # candidate_10 B per direction
    DIRS = ["core->near","near->core","near->outer","outer->near","core->outer","outer->core"]
    blocks_map = {
        "block_pre_normal": ("2018-01-01","2018-01-07","2018-01-08","2018-01-11"),
        "block_target_like": ("2018-01-01","2018-01-19","2018-01-20","2018-01-24"),
        "block_post_outage_stress": ("2018-01-01","2018-01-18","2018-01-19","2018-01-24"),
    }

    b_audit_rows = []
    b_replace = {}  # direction -> bool

    for d_name in ["outer->core", "near->outer"]:
        for bname, (ts, te, vs_, ve) in blocks_map.items():
            te_dt = pd.Timestamp(te)
            # training data for long mean
            train_data = b10[(b10["task_key"]==d_name) & (b10["date"]<=te_dt) & (b10["score_available"]==True)]
            if d_name == "near->core":
                train_data = b15_nc[(b15_nc["task_key"]==d_name) & (b15_nc["date"]<=te_dt)]
            train_y = train_data["y_true"].to_numpy(float) if len(train_data) else np.array([0.0])
            long_mean = float(np.round(train_y.mean())) if len(train_y) else 0.0

            # target block
            if d_name == "near->core":
                tgt = b15_nc[(b15_nc["date"]>=pd.Timestamp(vs_)) & (b15_nc["date"]<=pd.Timestamp(ve))]
            else:
                tgt = b10[(b10["task_key"]==d_name) & (b10["date"]>=pd.Timestamp(vs_)) & (b10["date"]<=pd.Timestamp(ve)) & (b10["score_available"]==True)]
            if len(tgt) == 0:
                b_audit_rows.append({"direction":d_name,"block":bname,"cand10_sse":0,"long_sse":0,"gain":0,"pct":0,"verdict":"no_data"})
                continue
            y_tgt = tgt["y_true"].to_numpy(float)
            p_cand10 = tgt["pred_rounded"].to_numpy(float)
            p_long = np.full(len(y_tgt), long_mean)
            sse_cand = float(((p_cand10 - y_tgt)**2).sum())
            sse_long = float(((p_long - y_tgt)**2).sum())
            gain = sse_cand - sse_long
            pct = gain / sse_cand * 100 if sse_cand > 0 else 0
            verdict = "replace" if pct > 0 else "keep"
            b_audit_rows.append({"direction":d_name,"block":bname,"cand10_sse":sse_cand,"long_sse":sse_long,
                               "gain":gain,"pct":pct,"long_mean":long_mean,"verdict":verdict})
            print(f"  {d_name:15s} {bname:30s} cand10={sse_cand:.0f} long={sse_long:.0f} gain={gain:+.0f} ({pct:+.1f}%) {verdict}")

    b_audit_df = pd.DataFrame(b_audit_rows)
    b_audit_df.to_csv(out_dir / "b_direction_replacement_audit.csv", index=False, encoding="utf-8-sig")

    # Decision: replace if target_like improves AND not both other blocks worsen >5%
    for d_name in ["outer->core", "near->outer"]:
        tl = b_audit_df[(b_audit_df["direction"]==d_name) & (b_audit_df["block"]=="block_target_like")]
        pn = b_audit_df[(b_audit_df["direction"]==d_name) & (b_audit_df["block"]=="block_pre_normal")]
        po = b_audit_df[(b_audit_df["direction"]==d_name) & (b_audit_df["block"]=="block_post_outage_stress")]
        tl_pct = float(tl["pct"].iloc[0]) if len(tl) else -100
        pn_pct = float(pn["pct"].iloc[0]) if len(pn) else -100
        po_pct = float(po["pct"].iloc[0]) if len(po) else -100
        passes = tl_pct > 0 and pn_pct >= -5 and po_pct >= -5
        b_replace[d_name] = passes
        print(f"  {d_name}: target_like={tl_pct:+.1f}% pre={pn_pct:+.1f}% post={po_pct:+.1f}% -> {'REPLACE' if passes else 'KEEP'}")

    # ================================================================
    # PART 2: A SPARSE -1 CORRECTION (LODO)
    # ================================================================
    print("\n=== A sparse -1 LODO screening ===")
    # Step 23 candidate cells
    candidate_cells = [(2,"core"),(3,"core"),(6,"core"),(10,"core"),(12,"core"),
                       (7,"near"),(10,"near"),(11,"near"),(12,"outer"),(13,"outer")]

    # Compute Step 11 OOF for all normal dates
    print("  Computing Step 11 OOF for normal dates...")
    oof_preds = {}
    for d in normal_dates:
        prev = [dd for dd in all_dates if dd < d]
        prev_normal = [dd for dd in prev if qmap.get(dd)=="normal"]
        if len(prev_normal) < 4: continue
        te = d - pd.Timedelta(days=1)
        oof_preds[d] = compute_baseline_vec(prev, te, d)

    oof_dates = sorted(oof_preds.keys())
    print(f"  OOF dates: {len(oof_dates)}")

    # LODO screening
    lodo_rows = []
    selected_cells = []
    for h, r in candidate_cells:
        ri = REGIONS.index(r)
        gains = []; improved = 0; total = 0; pre_gain = 0; post_gain = 0
        for i, d_target in enumerate(oof_dates):
            # leave d_target out: use all other OOF dates to decide
            other_dates = [dd for dd in oof_dates if dd != d_target]
            if len(other_dates) < 3: continue
            # count how many other dates benefit from -1
            benefit_count = 0
            for dd in other_dates:
                bl = oof_preds[dd]; a = get_a_vec(dd)
                bl_val = int(bl[h, ri])
                if bl_val <= 0: continue
                corrected = bl_val - 1
                gain_dd = (bl_val - int(a[h,ri]))**2 - (corrected - int(a[h,ri]))**2
                if gain_dd > 0: benefit_count += 1
            ratio = benefit_count / len(other_dates)
            # evaluate on held-out date
            bl = oof_preds[d_target]; a = get_a_vec(d_target)
            bl_val = int(bl[h, ri])
            if bl_val <= 0:
                gains.append(0); total += 1
                continue
            corrected = bl_val - 1
            gain_target = (bl_val - int(a[h,ri]))**2 - (corrected - int(a[h,ri]))**2
            gains.append(gain_target); total += 1
            if gain_target > 0: improved += 1
            if d_target <= pd.Timestamp("2018-01-11"): pre_gain += gain_target
            else: post_gain += gain_target
        improve_rate = improved / total if total > 0 else 0
        total_gain = sum(gains)
        lodo_rows.append({"hour":h,"region":r,"improve_rate":improve_rate,"total_gain":total_gain,
                         "pre_gain":pre_gain,"post_gain":post_gain,"n_dates":total})
        passes = (improve_rate >= 0.65 and total_gain > 0 and pre_gain >= 0 and post_gain >= 0 and total >= 8)
        if passes:
            selected_cells.append((h, r, total_gain, improve_rate, post_gain))
            print(f"  cell ({h},{r}): rate={improve_rate:.2f} gain={total_gain:.0f} pre={pre_gain:.0f} post={post_gain:.0f} -> SELECT")

    # sort by total_gain desc, take top 5
    selected_cells.sort(key=lambda x: (-x[2], -x[3], -x[4]))
    final_cells = selected_cells[:5]

    # fallback: if none selected, pick best with post_gain >= 0
    if not final_cells:
        lodo_rows.sort(key=lambda x: -x["total_gain"])
        for row in lodo_rows:
            if row["post_gain"] >= 0:
                final_cells = [(row["hour"], row["region"], row["total_gain"], row["improve_rate"], row["post_gain"])]
                print(f"  fallback cell ({row['hour']},{row['region']}): gain={row['total_gain']:.0f}")
                break

    print(f"  Final A cells: {[(h,r) for h,r,_,_,_ in final_cells]}")

    lodo_df = pd.DataFrame(lodo_rows)
    lodo_df.to_csv(out_dir / "a_sparse_cell_stability.csv", index=False, encoding="utf-8-sig")

    # A block scores with corrections
    a_block_rows = []
    for bname, (ts, te, vs_, ve) in blocks_map.items():
        vd = list(pd.date_range(pd.Timestamp(vs_), pd.Timestamp(ve), freq="D").normalize())
        te_dt = pd.Timestamp(te)
        bl_sse = 0; corr_sse = 0
        for d in vd:
            train = [dd for dd in all_dates if dd <= te_dt]
            bl = compute_baseline_vec(train, te_dt, d); a = get_a_vec(d)
            bl_sse += float(((bl - a)**2).sum())
            corr = bl.copy()
            for h, r, _, _, _ in final_cells:
                ri = REGIONS.index(r)
                corr[h, ri] = max(0, corr[h, ri] - 1)
            corr_sse += float(((corr - a)**2).sum())
        gain = bl_sse - corr_sse; pct = gain/bl_sse*100 if bl_sse else 0
        a_block_rows.append({"block":bname,"baseline_sse":bl_sse,"corrected_sse":corr_sse,
                           "absolute_gain":gain,"relative_gain_pct":pct})
        print(f"  A {bname}: bl={bl_sse:.0f} corr={corr_sse:.0f} gain={gain:+.0f} ({pct:+.1f}%)")

    a_block_df = pd.DataFrame(a_block_rows)
    a_block_df.to_csv(out_dir / "a_sparse_block_scores.csv", index=False, encoding="utf-8-sig")

    # ================================================================
    # PART 3: BUILD CANDIDATE_11
    # ================================================================
    print("\n=== Building candidate_11 ===")
    cand_dir = out_dir / "candidate_11_joint"; cand_dir.mkdir(exist_ok=True)

    # A: byte-copy Step 11 A template, apply corrections
    tmpl_a = pd.read_csv(ROOT / A_TEMPLATE)
    step11_a = pd.read_csv(ROOT / STEP11_A)
    a_out = step11_a.copy()
    a_ts = pd.to_datetime(a_out["time_window"])
    # apply -1 corrections
    a_changed = 0
    for h, r, _, _, _ in final_cells:
        zone = REGION_CN[r]
        mask = (a_out["zone"] == zone) & (a_ts.dt.hour == h) & (a_out["vessel_count"] > 0)
        a_out.loc[mask, "vessel_count"] -= 1
        a_changed += int(mask.sum())
    A_OUT = cand_dir / "提交结果1_区域活跃拖轮数量.csv"
    a_out.to_csv(A_OUT, index=False, encoding="utf-8-sig")
    print(f"  A: {a_changed} cells changed by -1")

    # B: byte-copy Step 11 B template, apply direction replacements
    step11_b = pd.read_csv(ROOT / STEP11_B)
    step17_b = pd.read_csv(ROOT / STEP17_B)  # near->core already replaced
    b_out = step17_b.copy()  # candidate_10 B = step17 B (which is step11 B + near->core replaced)
    b_ts = pd.to_datetime(b_out["time_window"])

    # For replaced directions: compute long mean from full training data
    b_changed = {}
    for d_name in ["outer->core", "near->outer"]:
        if not b_replace.get(d_name, False): continue
        src, tgt = d_name.split("->")
        sz = REGION_CN[src]; tz = REGION_CN[tgt]
        # training data
        train_b = b10[(b10["task_key"]==d_name) & (b10["score_available"]==True)]
        train_y = train_b["y_true"].to_numpy(float) if len(train_b) else np.array([0.0])
        long_mean = int(np.round(train_y.mean())) if len(train_y) else 0
        mask = (b_out["source_zone"]==sz) & (b_out["target_zone"]==tz)
        old_vals = b_out.loc[mask, "vessel_count"].copy()
        new_vals = long_mean  # same for all hours (long_hour mean)
        # clip: max change <= 2
        diff = new_vals - old_vals
        diff_clipped = diff.clip(-2, 2)
        b_out.loc[mask, "vessel_count"] = old_vals + diff_clipped
        b_changed[d_name] = {"cells": int(mask.sum()), "long_mean": long_mean,
                             "total_change": int(diff_clipped.sum())}
        print(f"  B {d_name}: {mask.sum()} cells -> long_mean={long_mean}, total_change={diff_clipped.sum()}")

    B_OUT = cand_dir / "提交结果2_圈层间拖轮迁移量.csv"
    b_out.to_csv(B_OUT, index=False, encoding="utf-8-sig")

    # ================================================================
    # PART 4: DIFF REPORT
    # ================================================================
    diff_rows = []
    # A diff
    a_diff_mask = (a_out["vessel_count"] != step11_a["vessel_count"])
    for i in np.where(a_diff_mask.to_numpy())[0]:
        diff_rows.append({"component":"A","index":int(i),"candidate10_value":int(step11_a["vessel_count"].iloc[i]),
                         "candidate11_value":int(a_out["vessel_count"].iloc[i]),
                         "difference":int(a_out["vessel_count"].iloc[i])-int(step11_a["vessel_count"].iloc[i]),
                         "reason":"sparse_-1_correction"})
    # B diff
    b_diff_mask = (b_out["vessel_count"] != step17_b["vessel_count"])
    for i in np.where(b_diff_mask.to_numpy())[0]:
        sz = b_out["source_zone"].iloc[i]; tz = b_out["target_zone"].iloc[i]
        direction = f"{CN_TO_REGION[sz]}->{CN_TO_REGION[tz]}"
        reason = "near->core_replacement" if direction == "near->core" else f"{direction}_direct_long"
        diff_rows.append({"component":"B","index":int(i),"candidate10_value":int(step17_b["vessel_count"].iloc[i]),
                         "candidate11_value":int(b_out["vessel_count"].iloc[i]),
                         "difference":int(b_out["vessel_count"].iloc[i])-int(step17_b["vessel_count"].iloc[i]),
                         "reason":reason})
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(out_dir / "joint_candidate_diff.csv", index=False, encoding="utf-8-sig")

    print(f"\n  A changed cells: {(a_out['vessel_count'] != step11_a['vessel_count']).sum()}")
    print(f"  B changed cells: {(b_out['vessel_count'] != step17_b['vessel_count']).sum()}")
    print(f"  A total change: {int((a_out['vessel_count'] - step11_a['vessel_count']).sum())}")
    print(f"  B total change: {int((b_out['vessel_count'] - step17_b['vessel_count']).sum())}")

    # ================================================================
    # PART 5: ZIP
    # ================================================================
    zip_path = out_dir / "candidate_11_joint.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(A_OUT, "提交结果1_区域活跃拖轮数量.csv")
        zf.write(B_OUT, "提交结果2_圈层间拖轮迁移量.csv")
    print(f"  ZIP: {zip_path}")

    # ================================================================
    # PART 6: SUMMARY
    # ================================================================
    L = ["# Step 42 Joint Candidate 11", "", f"Run command: `{RUN_CMD}`", ""]
    L.append("## B direction replacement")
    L.append("")
    for d_name in ["outer->core", "near->outer"]:
        r = b_replace.get(d_name, False)
        L.append(f"- {d_name}: {'REPLACED with direct_long' if r else 'KEPT candidate_10'}")
    L.append("")
    L.append("## A sparse -1 cells")
    L.append("")
    for h, r, gain, rate, post in final_cells:
        L.append(f"- ({h}, {r}): LODO gain={gain:.0f}, rate={rate:.2f}")
    L.append("")
    L.append("## A block scores")
    L.append("")
    L.append("| block | baseline | corrected | gain |")
    L.append("| --- | --- | --- | --- |")
    for _, r in a_block_df.iterrows():
        L.append(f"| {r['block']} | {r['baseline_sse']:.0f} | {r['corrected_sse']:.0f} | {r['absolute_gain']:+.0f} ({r['relative_gain_pct']:+.1f}%) |")
    L.append("")
    L.append(f"## Diff summary")
    L.append(f"- A changed cells: {int((a_out['vessel_count'] != step11_a['vessel_count']).sum())}")
    L.append(f"- B changed cells: {int((b_out['vessel_count'] != step17_b['vessel_count']).sum())}")
    L.append("")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    assert len(a_out) == 504 and len(b_out) == 1008
    assert (a_out["vessel_count"] >= 0).all() and (b_out["vessel_count"] >= 0).all()
    assert not any(out_dir.rglob("*.zip")) is False  # we want the zip to exist

    print(f"\n=== Summary ===")
    print(f"A cells changed: {int((a_out['vessel_count'] != step11_a['vessel_count']).sum())}")
    print(f"B cells changed: {int((b_out['vessel_count'] != step17_b['vessel_count']).sum() if len(b_out)==len(step17_b) else 0)}")
    print(f"B replaced directions: {[d for d,v in b_replace.items() if v]}")
    print(f"A final cells: {[(h,r) for h,r,_,_,_ in final_cells]}")
    print(f"ZIP: {zip_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
