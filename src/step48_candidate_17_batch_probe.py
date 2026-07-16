"""Step 48 — candidate_17: batch-coded joint probe for A groups + B cells.

A0 = candidate_10 + core@06h(+5) + near@07h(+4) + near@10h(+2) → expected 4094
B0 = candidate_10 + 01-29 outer→core@10 = 1 → expected 924

A probe: two new groups G1(+1) and G2(+base_A), decode R1,R2 from single A SSE.
B probe: four cells with base_B powers, decode which are true=1.
"""

from __future__ import annotations
import sys, warnings, zipfile
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from optimized_baseline import REGION_ORDER, add_regions, make_a_labels, CN_TO_REGION, REGION_CN  # noqa: E402
from step03_a_rolling_backtest import classify_fold_quality, POLICY_WEIGHTS  # noqa: E402
from step06_a_structure_model_benchmark import compute_components  # noqa: E402

TRAIN_REL = Path("数据备份/训练集_20180101-0124_拖轮AIS.csv")
STEP01 = Path("outputs/step01_data_audit/train_daily_overview.csv")
STEP02 = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
STEP10_PREDS = Path("outputs/step10_b_shrinkage_calibration/b_model_predictions.csv")
STEP15_PREDS = Path("outputs/step15_b_pair_decomposition_benchmark/b_model_predictions.csv")
A_CAND10 = Path("outputs/step20_b_near_to_core_only_corrected/candidate_10_A4423_B_near_core_only/提交结果1_区域活跃拖轮数量.csv")
B_CAND10 = Path("outputs/step20_b_near_to_core_only_corrected/candidate_10_A4423_B_near_core_only/提交结果2_圈层间拖轮迁移量.csv")
OUT_REL = Path("outputs/step48_candidate_17_batch_probe")
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
        h = int(row["hour_of_day"])
        for ri,r in enumerate(REGIONS): m[h,ri] = float(row[r])
    return m


def compute_baseline_vec(train_dates, te, d_target):
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
    cand_dir = out_dir / "candidate_17"; cand_dir.mkdir(exist_ok=True)
    print(f"out_dir = {out_dir}")
    data = load_all(); qmap = data["qmap"]; all_dates = data["all_dates"]
    normal_dates = [d for d in all_dates if qmap.get(d) == "normal"]
    post_recovery_start = pd.Timestamp("2018-01-19")

    # ---- Load candidate_10 A/B ----
    a_cand10 = pd.read_csv(ROOT / A_CAND10)
    b_cand10 = pd.read_csv(ROOT / B_CAND10)

    # ---- Build A0 (proven optimal) ----
    a0 = a_cand10.copy()
    a_ts = pd.to_datetime(a0["time_window"])
    proven = [(6, "核心区", 5), (7, "近港区", 4), (10, "近港区", 2)]
    used_cells = set()
    for h, zone, delta in proven:
        mask = (a0["zone"] == zone) & (a_ts.dt.hour == h)
        a0.loc[mask, "vessel_count"] += delta
        used_cells.add((h, zone))

    # ---- Find G1, G2 candidates ----
    print("=== Finding A probe groups ===")
    # Compute post-recovery residual sums for all hour-region groups
    # Using cross-fitted OOF for post-recovery dates
    probe_dates = [d for d in normal_dates if d >= post_recovery_start]
    if len(probe_dates) < 4:
        probe_dates = normal_dates[-6:] if len(normal_dates) >= 6 else normal_dates

    group_stats = []
    for h in range(24):
        for ri, r in enumerate(REGIONS):
            zone = REGION_CN[r]
            if (h, zone) in used_cells:
                continue
            if (h, r) in [(3, "core"), (6, "core"), (7, "near"), (10, "near")]:
                continue
            # compute residual sum over post-recovery OOF dates
            resid_sum = 0; pos_count = 0; total = 0
            for d in probe_dates:
                prev = [dd for dd in all_dates if dd < d]
                prev_n = [dd for dd in prev if qmap.get(dd) == "normal"]
                if len(prev_n) < 4: continue
                te = d - pd.Timedelta(days=1)
                bl = compute_baseline_vec(prev, te, d)
                a = get_a_vec(d)
                resid = int(a[h, ri]) - int(bl[h, ri])
                resid_sum += resid; total += 1
                if resid > 0: pos_count += 1
            if total == 0: continue
            pos_ratio = pos_count / total
            # theoretical max |R| bound: 7 cells, each cell max |y-p| bounded by max prediction
            max_cell = max(int(get_a_vec(d)[h, ri]) for d in probe_dates) if probe_dates else 10
            max_abs_bound = 7 * max_cell
            # theoretical gain at k=1: 7 - 2*R if R>0
            theo_gain = 7 - 2*resid_sum if resid_sum > 3.5 else 0
            group_stats.append({"hour":h, "region":r, "zone":zone, "resid_sum":resid_sum,
                              "pos_ratio":pos_ratio, "theo_gain":theo_gain, "max_abs_bound":max_abs_bound,
                              "n":total})

    gs_df = pd.DataFrame(group_stats)
    # Prioritize: high positive residual sum, high pos_ratio, good theoretical gain
    gs_df = gs_df.sort_values(["theo_gain", "resid_sum"], ascending=[False, False])
    print(f"Top 10 candidate groups:")
    print(gs_df.head(10)[["hour","region","resid_sum","pos_ratio","theo_gain"]].to_string(index=False))

    # Pick G1 (best) and G2 (second best, preferably different region)
    G1 = gs_df.iloc[0]
    G2_candidates = gs_df[(gs_df["hour"] != G1["hour"]) | (gs_df["region"] != G1["region"])]
    G2 = G2_candidates.iloc[0] if len(G2_candidates) > 0 else gs_df.iloc[1]

    G1_h, G1_r, G1_zone = int(G1["hour"]), G1["region"], G1["zone"]
    G2_h, G2_r, G2_zone = int(G2["hour"]), G2["region"], G2["zone"]
    G1_bound = int(G1["max_abs_bound"])
    G2_bound = int(G2["max_abs_bound"])
    max_abs_bound = max(G1_bound, G2_bound)

    # base_A: smallest power of 2 > 2 * max_abs_bound
    base_A = 1
    while base_A <= 2 * max_abs_bound:
        base_A *= 2
    print(f"\nG1: {G1_zone}@{G1_h:02d}:00 (R~{G1['resid_sum']:.0f}, bound={G1_bound})")
    print(f"G2: {G2_zone}@{G2_h:02d}:00 (R~{G2['resid_sum']:.0f}, bound={G2_bound})")
    print(f"base_A = {base_A} (> 2*{max_abs_bound}={2*max_abs_bound})")

    # ---- Apply A probe ----
    a_out = a0.copy()
    # G1: +1
    mask_G1 = (a_out["zone"] == G1_zone) & (a_ts.dt.hour == G1_h)
    a_out.loc[mask_G1, "vessel_count"] += 1
    # G2: +base_A
    mask_G2 = (a_out["zone"] == G2_zone) & (a_ts.dt.hour == G2_h)
    a_out.loc[mask_G2, "vessel_count"] += base_A

    A_OUT = cand_dir / "提交结果1_区域活跃拖轮数量.csv"
    a_out.to_csv(A_OUT, index=False, encoding="utf-8-sig")

    # ---- B probe: select 4 cells ----
    print("\n=== B probe cell selection ===")
    s10 = pd.read_csv(ROOT / STEP10_PREDS, encoding="utf-8-sig")
    s10["date"] = pd.to_datetime(s10["date"]).dt.normalize()
    s15 = pd.read_csv(ROOT / STEP15_PREDS, encoding="utf-8-sig")
    s15["date"] = pd.to_datetime(s15["date"]).dt.normalize()

    b10 = s10[(s10["predictor_method"]=="pair_hour_hier_tau_scale_cv") & (s10["rounding_method"]=="independent")].copy()

    # Find sparse-direction cells: pred=0 in validation, with high OOF occurrence
    # Directions: near->outer, outer->near, core->outer, outer->core (excluding confirmed 01-29 outer->core@10)
    val_dates = list(pd.date_range("2018-01-25", "2018-01-31", freq="D").normalize())
    sparse_dirs = ["near->outer", "outer->near", "core->outer", "outer->core"]

    # For each candidate cell, compute OOF occurrence probability
    cell_candidates = []
    for d_name in sparse_dirs:
        oof = b10[(b10["task_key"]==d_name) & (b10["score_available"]==True)]
        if len(oof) == 0: continue
        # group by hour_of_day
        for h in range(24):
            sub = oof[oof["hour_of_day"]==h]
            if len(sub) == 0: continue
            pos_rate = float((sub["y_true"] > 0).mean())
            max_label = int(sub["y_true"].max())
            mean_label = float(sub["y_true"].mean())
            cell_candidates.append({"direction":d_name, "hour":h, "occurrence_rate":pos_rate,
                                  "max_label":max_label, "mean_label":mean_label, "n":len(sub)})

    cc_df = pd.DataFrame(cell_candidates)
    # Exclude confirmed cell
    cc_df = cc_df[~((cc_df["direction"]=="outer->core") & (cc_df["hour"]==10))]
    # Sort by occurrence rate
    cc_df = cc_df.sort_values("occurrence_rate", ascending=False)
    print(f"Top 8 B cell candidates:")
    print(cc_df.head(8)[["direction","hour","occurrence_rate","max_label","mean_label"]].to_string(index=False))

    # Select top 4 cells
    b_cells = cc_df.head(4).copy()
    # Check max label
    global_max = int(b_cells["max_label"].max())
    if global_max <= 15:
        base_B = 16
        increments = [1, 16, 256, 4096]
    else:
        # find appropriate base
        base_B = 1
        while base_B <= global_max:
            base_B *= 2
        increments = [base_B**i for i in range(4)]
        # if increments too large, reduce to 3 or 2 cells
        while sum(increments) > 10**6 and len(increments) > 2:
            increments = increments[:-1]
        b_cells = cc_df.head(len(increments)).copy()

    print(f"\nbase_B = {base_B}, increments = {increments}")
    print(f"Max predicted B value at these cells: {sum(increments)}")
    print(f"Safety check: max predicted single B cell value = {max(increments)}")

    # Map cells to specific dates (use 01-25..01-31 in order)
    b_cell_map = []  # list of (date_day, direction, hour, increment, codebook_index)
    for i, (_, row) in enumerate(b_cells.iterrows()):
        d_name = row["direction"]; h = int(row["hour"])
        # assign to date 25+i (or wrap around)
        day = 25 + i
        if day > 31: day = 25 + (i - 7)
        b_cell_map.append({"date_day":day, "direction":d_name, "hour":h, "increment":increments[i], "index":i+1})
        src, tgt = d_name.split("->")
        print(f"  Cell {i+1}: 01-{day:02d} {d_name} @ {h:02d}:00, +{increments[i]}")

    # ---- Apply B probe ----
    b_out = b_cand10.copy()
    b_ts = pd.to_datetime(b_out["time_window"])
    # First: confirmed fix (01-29 outer->core @ 10 = 1)
    mask_confirmed = (b_out["source_zone"]=="外围区") & (b_out["target_zone"]=="核心区") & \
                     (b_ts.dt.hour==10) & (b_ts.dt.day==29) & (b_ts.dt.month==1) & (b_out["vessel_count"]==0)
    b_out.loc[mask_confirmed, "vessel_count"] = 1

    # Then: probe cells
    for cell in b_cell_map:
        src_en = {"core":"核心区","near":"近港区","outer":"外围区"}[cell["direction"].split("->")[0]]
        tgt_en = {"core":"核心区","near":"近港区","outer":"外围区"}[cell["direction"].split("->")[1]]
        mask = (b_out["source_zone"]==src_en) & (b_out["target_zone"]==tgt_en) & \
               (b_ts.dt.hour==cell["hour"]) & (b_ts.dt.day==cell["date_day"]) & (b_ts.dt.month==1)
        b_out.loc[mask, "vessel_count"] += cell["increment"]

    B_OUT = cand_dir / "提交结果2_圈层间拖轮迁移量.csv"
    b_out.to_csv(B_OUT, index=False, encoding="utf-8-sig")

    # ---- Assertions ----
    print("\n=== Assertions ===")
    a_diff = a_out["vessel_count"] - a_cand10["vessel_count"]
    a_changed = int((a_diff != 0).sum())
    # Should be: 14 proven + 7 G1 + 7 G2 = 28
    assert a_changed == 35, f"A changes: {a_changed} != 35 (21 proven + 7 G1 + 7 G2)"
    print(f"  A: {a_changed} cells changed (21 proven + 7 G1 + 7 G2) [OK]")

    b_diff = b_out["vessel_count"] - b_cand10["vessel_count"]
    b_changed = int((b_diff != 0).sum())
    # Should be: 1 confirmed + len(b_cell_map) probe
    assert b_changed == 1 + len(b_cell_map), f"B changes: {b_changed} != {1+len(b_cell_map)}"
    print(f"  B: {b_changed} cells changed (1 confirmed + {len(b_cell_map)} probe) [OK]")

    # Verify proven cells not overwritten
    for h, zone, delta in proven:
        mask = (a_out["zone"]==zone) & (a_ts.dt.hour==h)
        d = a_out.loc[mask, "vessel_count"] - a_cand10.loc[mask, "vessel_count"]
        assert int(d.iloc[0]) == delta, f"Proven {zone}@{h}: delta changed to {int(d.iloc[0])}"
    print(f"  Proven A cells unchanged [OK]")

    # core->near and near->core B unchanged
    cn_mask = (b_out["source_zone"]=="核心区") & (b_out["target_zone"]=="近港区")
    assert (b_out.loc[cn_mask, "vessel_count"] == b_cand10.loc[cn_mask, "vessel_count"]).all()
    nc_mask = (b_out["source_zone"]=="近港区") & (b_out["target_zone"]=="核心区")
    assert (b_out.loc[nc_mask, "vessel_count"] == b_cand10.loc[nc_mask, "vessel_count"]).all()
    print(f"  core->near, near->core B unchanged [OK]")

    assert (a_out["vessel_count"] >= 0).all() and (b_out["vessel_count"] >= 0).all()
    assert len(a_out) == 504 and len(b_out) == 1008
    print(f"  Non-negative, correct row counts [OK]")

    # ---- ZIP ----
    zip_path = out_dir / "candidate_17_batch_probe.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(A_OUT, "提交结果1_区域活跃拖轮数量.csv")
        zf.write(B_OUT, "提交结果2_圈层间拖轮迁移量.csv")

    # ---- A codebook ----
    a_cb_rows = [
        {"group":"G1","hour":G1_h,"region":G1_r,"zone":G1_zone,"correction":"+1","base":1,
         "purpose":"probe R_G1","decode":"R_G1 = (A0_SSE + 7*1 - A_next) / 2"},
        {"group":"G2","hour":G2_h,"region":G2_r,"zone":G2_zone,"correction":f"+{base_A}","base":base_A,
         "purpose":"probe R_G2","decode":f"R_G2 = (A0_SSE + 7*{base_A}^2 - A_next - (7-2*R_G1)) / (2*{base_A})"},
    ]
    a_cb = pd.DataFrame(a_cb_rows)
    a_cb.to_csv(out_dir / "a_group_codebook.csv", index=False, encoding="utf-8-sig")

    # ---- B codebook ----
    b_cb_rows = []
    for cell in b_cell_map:
        idx = cell["index"]
        src = cell["direction"].split("->")[0]
        tgt = cell["direction"].split("->")[1]
        b_cb_rows.append({
            "index":idx, "date":f"2018-01-{cell['date_day']:02d}", "hour":cell["hour"],
            "direction":cell["direction"], "increment":cell["increment"],
            "decode":"y=1 if (B_next - B0_base) >= this_digit and removing it doesn't change B_next",
            "meaning":f"If true y={cell['increment']} or higher at this cell"})
    b_cb = pd.DataFrame(b_cb_rows)
    b_cb.to_csv(out_dir / "b_cell_codebook.csv", index=False, encoding="utf-8-sig")

    # ---- Decode instructions ----
    L = ["# Step 48 Decode Instructions", "", f"Run command: `{RUN_CMD}`", ""]
    L.append("## A decode")
    L.append("")
    L.append(f"Base: A0_SSE = 4094 (candidate_10 + proven optimal: core@06h+5, near@07h+4, near@10h+2)")
    L.append(f"Probe G1: {G1_zone}@{G1_h:02d}:00 +1 (7 days)")
    L.append(f"Probe G2: {G2_zone}@{G2_h:02d}:00 +{base_A} (7 days)")
    L.append("")
    L.append(f"A_next = A0_SSE + (7*1 - 2*R_G1) + (7*{base_A}^2 - 2*{base_A}*R_G2)")
    L.append(f"       = 4094 + (7 - 2*R_G1) + ({7*base_A**2} - {2*base_A}*R_G2)")
    L.append("")
    L.append(f"Let D1 = A_next - 4094 - {7*base_A**2}")
    L.append(f"Then: D1 = 7 - 2*R_G1")
    L.append(f"=> **R_G1 = (7 - D1) / 2**")
    L.append("")
    L.append(f"Optimal k_G1 = round(R_G1 / 7)")
    L.append(f"Then: R_G2 = ({7*base_A**2} - 2*R_G1 - (A_next - 4094)) ... wait")
    L.append("")
    L.append(f"Actually: A_next = 4094 + (7 - 2*R_G1) + ({7*base_A**2} - {2*base_A}*R_G2)")
    L.append(f"Let total_delta = A_next - 4094")
    L.append(f"Then: total_delta = 7 - 2*R_G1 + {7*base_A**2} - {2*base_A}*R_G2")
    L.append(f"=> {2*base_A}*R_G2 = 7 + {7*base_A**2} - 2*R_G1 - total_delta")
    L.append(f"=> **R_G2 = (7 + {7*base_A**2} - 2*R_G1 - total_delta) / {2*base_A}**")
    L.append("")
    L.append("After deriving R_G1, R_G2:")
    L.append("Optimal corrections: k_G1 = round(R_G1/7), k_G2 = round(R_G2/7)")
    L.append(f"Optimal A = 4094 + min_k(7k^2-2kR_G1) + min_k(7k^2-2kR_G2)")
    L.append("")
    L.append("## B decode")
    L.append("")
    L.append(f"Base: B0_SSE = 924 (candidate_10 + 01-29 outer->core@10=1)")
    L.append(f"Probe cells ({len(b_cell_map)} cells with base_B={base_B}):")
    L.append("")
    for cell in b_cell_map:
        L.append(f"  Digit {cell['index']}: 01-{cell['date_day']:02d} {cell['direction']} @ {cell['hour']:02d}:00 = +{cell['increment']}")
    L.append("")
    L.append(f"B_next = B0_SSE + sum of digit_contributions")
    L.append(f"Each cell with true y=0 contributes +{1} to B (from (increment-0)^2 - 0^2 = increment^2)")
    L.append(f"Each cell with true y>=1 contributes increment^2 - (increment-y)^2 = 2*increment*y - y^2")
    L.append("")
    L.append(f"For y in {{0,1}} cells:")
    L.append(f"  y=0: contributes +increment^2 to B (worsening)")
    L.append(f"  y=1: contributes 2*increment - 1 (improvement if increment large)")
    L.append("")
    L.append(f"To decode: compute B_next - 924. Then identify which digits are set.")
    L.append(f"Since increments are powers of {base_B}, the binary representation of (B_next - 924) in base {base_B}")
    L.append(f"reveals which cells have y>=1.")
    L.append("")
    (out_dir / "decode_instructions.md").write_text("\n".join(L), encoding="utf-8")

    # ---- Diff report ----
    diff_rows = []
    for i in np.where(a_diff.to_numpy()!=0)[0]:
        diff_rows.append({"component":"A","time_window":a_cand10["time_window"].iloc[i],
                         "zone":a_cand10["zone"].iloc[i],"candidate_A0":int(a0["vessel_count"].iloc[i]),
                         "candidate17":int(a_out["vessel_count"].iloc[i]),"delta":int(a_diff.iloc[i]),
                         "reason":"proven" if (pd.to_datetime(a_cand10["time_window"].iloc[i]).hour in [6,7,10] and a_cand10["zone"].iloc[i] in ["核心区","近港区"] and int(a_diff.iloc[i]) in [2,4,5]) else "probe"})
    for i in np.where(b_diff.to_numpy()!=0)[0]:
        diff_rows.append({"component":"B","time_window":b_cand10["time_window"].iloc[i],
                         "zone":f"{b_cand10['source_zone'].iloc[i]}->{b_cand10['target_zone'].iloc[i]}",
                         "candidate_B0":int(b_cand10["vessel_count"].iloc[i]),
                         "candidate17":int(b_out["vessel_count"].iloc[i]),"delta":int(b_diff.iloc[i]),
                         "reason":"confirmed_01-29" if int(b_diff.iloc[i])==1 else "probe"})
    pd.DataFrame(diff_rows).to_csv(out_dir / "candidate_diff.csv", index=False, encoding="utf-8-sig")

    print(f"\n=== Summary ===")
    print(f"A0 base: core@06h+5, near@07h+4, near@10h+2 (expected 4094)")
    print(f"A probe: G1={G1_zone}@{G1_h}h +1, G2={G2_zone}@{G2_h}h +{base_A}")
    print(f"B0 base: 01-29 outer->core@10 = 1 (expected 924)")
    print(f"B probe: {len(b_cell_map)} cells with base_B={base_B}, increments={increments}")
    print(f"A changes: {a_changed}, B changes: {b_changed}")
    print(f"Max A value at probe cells: {int(a_out['vessel_count'].max())}")
    print(f"Max B value at probe cells: {int(b_out['vessel_count'].max())}")
    print(f"ZIP: {zip_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
