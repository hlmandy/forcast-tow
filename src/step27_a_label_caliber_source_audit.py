"""Step 27 — A label caliber and multi-source consistency audit.

Audits whether training A labels are contaminated by cross-source duplicates,
distance approximation, or time misalignment. Constructs 6 label variants (V0-V5)
and compares them. No models trained, no submissions.

Outputs (under outputs/step27_a_label_caliber_source_audit/):
  1-12 as specified in the plan.
"""

from __future__ import annotations
import sys, warnings, math
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from optimized_baseline import REGION_ORDER, CENTER_LON, CENTER_LAT, KM_PER_LAT, KM_PER_LON, add_regions, make_a_labels  # noqa: E402

TRAIN_REL = Path("数据备份/训练集_20180101-0124_拖轮AIS.csv")
STEP01 = Path("outputs/step01_data_audit/train_daily_overview.csv")
STEP02 = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
OUT_REL = Path("outputs/step27_a_label_caliber_source_audit")
REGIONS = REGION_ORDER
RUN_CMD = "python " + " ".join(sys.argv)
PRE_END = pd.Timestamp("2018-01-11"); DEG_END = pd.Timestamp("2018-01-18")


def haversine_km(lon, lat, clon=CENTER_LON, clat=CENTER_LAT):
    R = 6371.0
    lon, lat, clon, clat = map(np.radians, [lon, lat, clon, clat])
    dlon = lon - clon; dlat = lat - clat
    a = np.sin(dlat/2)**2 + np.cos(lat)*np.cos(clat)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def assign_region_haversine(df):
    d = haversine_km(df["x"].values, df["y"].values)
    return np.select([d < 3, np.logical_and(d >= 3, d < 10), np.logical_and(d >= 10, d < 30)],
                     REGIONS, default="outside")


def make_a_labels_variant(df, variant="V0"):
    """Build A labels with different caliber rules."""
    if variant in ("V0", "V1"):
        d = df.copy()
        if variant == "V1":
            dist = haversine_km(d["x"].values, d["y"].values)
            d = d.assign(dist_km=dist)
            d["region"] = np.select([d["dist_km"]<3, d["dist_km"].between(3,10,inclusive="left"),
                                     d["dist_km"].between(10,30,inclusive="left")], REGIONS, default="outside")
        else:
            d = add_regions(d.copy())
        active = d[d["region"].isin(REGIONS) & d["sog"].between(2, 10, inclusive="both")]
    elif variant == "V2":  # exact message dedup + haversine
        d = df.drop_duplicates(subset=["mmsi","time","x","y","sog"]).copy()
        dist = haversine_km(d["x"].values, d["y"].values)
        d["region"] = np.select([dist<3, np.logical_and(dist>=3,dist<10), np.logical_and(dist>=10,dist<30)], REGIONS, default="outside")
        active = d[d["region"].isin(REGIONS) & d["sog"].between(2, 10, inclusive="both")]
    elif variant == "V3":  # vessel-timestamp dedup (median coords/sog) + haversine
        d = df.copy()
        # group by mmsi+time, take median of x, y, sog; keep source as first
        agg = d.groupby(["mmsi","time"]).agg(x=("x","median"), y=("y","median"), sog=("sog","median"),
                                              source_dataset=("source_dataset","first"), hour=("hour","first"),
                                              date=("date","first")).reset_index()
        dist = haversine_km(agg["x"].values, agg["y"].values)
        agg["region"] = np.select([dist<3, np.logical_and(dist>=3,dist<10), np.logical_and(dist>=10,dist<30)], REGIONS, default="outside")
        active = agg[agg["region"].isin(REGIONS) & agg["sog"].between(2, 10, inclusive="both")]
    elif variant == "V4":  # unique timestamp threshold + haversine
        d = df.copy()
        dist = haversine_km(d["x"].values, d["y"].values)
        d["region"] = np.select([dist<3, np.logical_and(dist>=3,dist<10), np.logical_and(dist>=10,dist<30)], REGIONS, default="outside")
        active = d[d["region"].isin(REGIONS) & d["sog"].between(2, 10, inclusive="both")]
        # count unique timestamps per (hour, region, mmsi) instead of rows
        active = active.copy()
        active["unique_ts"] = active.groupby(["hour","region","mmsi"])["time"].transform("nunique")
        qualified = active[active["unique_ts"] >= 3]
        labels = qualified.groupby(["hour","region"], observed=True)["mmsi"].nunique().rename("y").reset_index()
        return _full_grid(labels, df)
    elif variant == "V5":  # per-source threshold union + haversine
        d = df.copy()
        dist = haversine_km(d["x"].values, d["y"].values)
        d["region"] = np.select([dist<3, np.logical_and(dist>=3,dist<10), np.logical_and(dist>=10,dist<30)], REGIONS, default="outside")
        active = d[d["region"].isin(REGIONS) & d["sog"].between(2, 10, inclusive="both")]
        # per source ≥3
        src_counts = active.groupby(["hour","region","mmsi","source_dataset"]).size().rename("n").reset_index()
        qualified_vessels = src_counts[src_counts["n"] >= 3][["hour","region","mmsi"]].drop_duplicates()
        labels = qualified_vessels.groupby(["hour","region"], observed=True)["mmsi"].nunique().rename("y").reset_index()
        return _full_grid(labels, df)

    # V0-V3: standard ≥3 rows threshold
    src_counts = active.groupby(["hour","region","mmsi"], observed=True).size().rename("ais_points").reset_index()
    qualified = src_counts[src_counts["ais_points"] >= 3]
    labels = qualified.groupby(["hour","region"], observed=True)["mmsi"].nunique().rename("y").reset_index()
    return _full_grid(labels, df)


def _full_grid(labels, df):
    hours = pd.date_range(df["hour"].min(), df["hour"].max(), freq="h")
    idx = pd.MultiIndex.from_product([hours, REGIONS], names=["hour","region"])
    return labels.set_index(["hour","region"]).reindex(idx, fill_value=0).reset_index()


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    # ---- load raw AIS ----
    df = pd.read_csv(ROOT / TRAIN_REL,
                     dtype={"mmsi":"string","x":"float64","y":"float64","sog":"float32","source_dataset":"string"},
                     parse_dates=["time"])
    df["hour"] = df["time"].dt.floor("h"); df["date"] = df["time"].dt.normalize()
    df["hour_of_day"] = df["time"].dt.hour
    print(f"Total AIS rows: {len(df)}")

    aud = pd.read_csv(ROOT / STEP02); aud["date"] = pd.to_datetime(aud["date"]).dt.normalize()
    qmap = dict(zip(aud["date"], aud["quality_regime"]))
    all_dates = sorted(df["date"].unique())

    # ========== 1. source daily coverage ==========
    src_stats = df.groupby(["source_dataset","date"]).agg(
        ais_rows=("mmsi","size"), unique_mmsi=("mmsi","nunique")).reset_index()
    src_summary = df.groupby("source_dataset").agg(
        total_rows=("mmsi","size"), unique_mmsi=("mmsi","nunique"),
        sog_missing=("sog", lambda s: s.isna().sum()),
        x_missing=("x", lambda s: s.isna().sum())).reset_index()
    src_summary.to_csv(out_dir / "source_daily_coverage.csv", index=False, encoding="utf-8-sig")
    print("\nSource summary:"); print(src_summary.to_string(index=False))

    # ========== 2. vessel source membership ==========
    vessel_sources = df.groupby("mmsi")["source_dataset"].apply(lambda s: tuple(sorted(s.unique()))).reset_index()
    vessel_sources.columns = ["mmsi","source_combo"]
    membership_counts = vessel_sources["source_combo"].value_counts()
    print(f"\nVessel source membership:\n{membership_counts}")
    vessel_sources.to_csv(out_dir / "vessel_source_membership.csv", index=False, encoding="utf-8-sig")

    # ========== 3. duplicate audit ==========
    # exact message duplicate
    exact_dup = df.duplicated(subset=["mmsi","time","x","y","sog"]).sum()
    # same vessel timestamp
    ts_dup = df.duplicated(subset=["mmsi","time"]).sum()
    # same vessel second
    df["second"] = df["time"].dt.floor("s")
    sec_dup = df.duplicated(subset=["mmsi","second"]).sum()

    print(f"\nDuplicates: exact={exact_dup} ({exact_dup/len(df)*100:.2f}%), "
          f"timestamp={ts_dup} ({ts_dup/len(df)*100:.2f}%), second={sec_dup} ({sec_dup/len(df)*100:.2f}%)")

    # cross-source timestamp duplicates
    cross_src = df[df.duplicated(subset=["mmsi","time"], keep=False)]
    cross_src_groups = cross_src.groupby(["mmsi","time"])["source_dataset"].nunique()
    n_cross = int((cross_src_groups > 1).sum())
    print(f"Cross-source timestamp duplicates: {n_cross}")

    # threshold impact: how many vessel-hour-region groups cross 3-boundary due to cross-source dup
    if "region" not in df.columns: df = add_regions(df)
    active = df[df["region"].isin(REGIONS) & df["sog"].between(2, 10, inclusive="both")]
    vrc = active.groupby(["hour","region","mmsi","source_dataset"]).size().rename("n_src").reset_index()
    vrc_combined = vrc.groupby(["hour","region","mmsi"])["n_src"].sum().rename("n_combined").reset_index()
    vrc_max = vrc.groupby(["hour","region","mmsi"])["n_src"].max().rename("n_max_src").reset_index()
    vrc_merged = vrc_combined.merge(vrc_max, on=["hour","region","mmsi"])

    # groups that pass ≥3 combined but fail per-source
    cross_threshold = vrc_merged[(vrc_merged["n_combined"] >= 3) & (vrc_merged["n_max_src"] < 3)]
    n_cross_threshold = len(cross_threshold)
    n_total_qualified = len(vrc_merged[vrc_merged["n_combined"] >= 3])
    pct_cross = n_cross_threshold / max(1, n_total_qualified) * 100
    print(f"Groups passing ≥3 combined but <3 per-source: {n_cross_threshold} / {n_total_qualified} ({pct_cross:.1f}%)")

    dup_summary = pd.DataFrame([
        {"duplicate_type": "exact_message", "duplicate_rows": int(exact_dup), "pct": exact_dup/len(df)*100},
        {"duplicate_type": "same_vessel_timestamp", "duplicate_rows": int(ts_dup), "pct": ts_dup/len(df)*100},
        {"duplicate_type": "same_vessel_second", "duplicate_rows": int(sec_dup), "pct": sec_dup/len(df)*100},
        {"duplicate_type": "cross_source_threshold_groups", "duplicate_rows": n_cross_threshold,
         "pct": pct_cross, "total_qualified": n_total_qualified},
    ])
    dup_summary.to_csv(out_dir / "exact_duplicate_summary.csv", index=False, encoding="utf-8-sig")

    # ========== 4. planar vs haversine ==========
    dist_planar = np.sqrt(((df["x"] - CENTER_LON) * KM_PER_LON)**2 + ((df["y"] - CENTER_LAT) * KM_PER_LAT)**2)
    dist_hav = haversine_km(df["x"].values, df["y"].values)
    region_planar = np.select([dist_planar<3, dist_planar.between(3,10,inclusive="left"), dist_planar.between(10,30,inclusive="left")],
                              REGIONS, default="outside")
    region_hav = np.select([dist_hav<3, np.logical_and(dist_hav>=3,dist_hav<10), np.logical_and(dist_hav>=10,dist_hav<30)],
                           REGIONS, default="outside")
    region_diff = (region_planar != region_hav).sum()
    # near-boundary points
    for lo, hi, name in [(2.8,3.2,"3km"), (9.8,10.2,"10km"), (29.8,30.2,"30km")]:
        near = np.logical_and(dist_planar >= lo, dist_planar < hi)
        n_near = int(near.sum())
        n_diff = int(((region_planar != region_hav) & near).sum())
        print(f"  Boundary {name}: {n_near} points near, {n_diff} reclassified")
    print(f"Total planar vs haversine reclassifications: {region_diff} ({region_diff/len(df)*100:.2f}%)")

    # ========== 5. label variants V0-V5 ==========
    print("\n=== Building label variants ===")
    variants = {}
    for v in ["V0","V1","V2","V3","V4","V5"]:
        lab = make_a_labels_variant(df, v)
        lab["date"] = lab["hour"].dt.normalize()
        lab["hour_of_day"] = lab["hour"].dt.hour
        variants[v] = lab
        print(f"  {v}: total A = {int(lab['y'].sum())}")

    # verify V0 matches make_a_labels
    official = make_a_labels(df)
    v0 = variants["V0"]
    merged = official.merge(v0, on=["hour","region"], how="outer", suffixes=("_off","_v0"))
    merged["y_off"] = merged["y_off"].fillna(0).astype(int)
    merged["y_v0"] = merged["y_v0"].fillna(0).astype(int)
    mism = int((merged["y_off"] != merged["y_v0"]).sum())
    assert mism == 0, f"V0 mismatch with make_a_labels: {mism} cells"
    print("V0 verified against make_a_labels: PASS")

    # variant comparison
    comp_rows = []
    for v1 in ["V0","V1","V2","V3","V4","V5"]:
        for v2 in ["V0"]:
            if v1 == v2: continue
            l1 = variants[v1].set_index(["hour","region"])["y"]
            l2 = variants[v2].set_index(["hour","region"])["y"]
            diff = (l1 - l2).abs()
            comp_rows.append({"variant_i": v1, "variant_j": v2,
                              "different_cells": int((diff > 0).sum()),
                              "total_cells": len(diff),
                              "pct_different": float((diff > 0).mean() * 100),
                              "mean_abs_diff": float(diff.mean()),
                              "max_abs_diff": int(diff.max()),
                              "total_diff": int((l1 - l2).sum())})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out_dir / "label_variant_cell_comparison.csv", index=False, encoding="utf-8-sig")
    print("\nVariant comparison vs V0:")
    print(comp[["variant_i","different_cells","pct_different","max_abs_diff","total_diff"]].to_string(index=False))

    # ========== 6. time alignment ==========
    # For vessels in multiple sources, match nearby records
    vessel_sources["combo_str"] = vessel_sources["source_combo"].apply(lambda t: ",".join(t) if isinstance(t, tuple) else str(t))
    multi_src_vessels = vessel_sources[vessel_sources["combo_str"].str.contains(",")]["mmsi"].tolist()[:50]
    time_diffs = []
    for m in multi_src_vessels[:20]:
        md = df[df["mmsi"] == m].sort_values("time")
        for s1, s2 in [("china_coastal","e_globe_daily"), ("china_coastal","f_globe_dynamic"), ("e_globe_daily","f_globe_dynamic")]:
            d1 = md[md["source_dataset"] == s1]
            d2 = md[md["source_dataset"] == s2]
            if len(d1) == 0 or len(d2) == 0: continue
            # for each d1 record, find nearest d2 by time
            for _, r1 in d1.head(50).iterrows():
                diffs = (d2["time"] - r1["time"]).dt.total_seconds().abs()
                if len(diffs) > 0:
                    min_idx = diffs.idxmin()
                    if diffs[min_idx] < 300:  # within 5 min
                        time_diffs.append({"mmsi": m, "source_pair": f"{s1}->{s2}",
                                           "time_diff_seconds": float((d2.loc[min_idx,"time"] - r1["time"]).total_seconds())})
    ta = pd.DataFrame(time_diffs) if time_diffs else pd.DataFrame(columns=["mmsi","source_pair","time_diff_seconds"])
    ta.to_csv(out_dir / "timestamp_alignment.csv", index=False, encoding="utf-8-sig")
    if len(ta) > 0:
        print(f"\nTime alignment: {len(ta)} matched pairs, median diff = {ta['time_diff_seconds'].median():.1f}s")
    else:
        print("\nTime alignment: no matched pairs found")

    # ========== 7. source cohort analysis ==========
    cohort_map = {}
    for _, r in vessel_sources.iterrows():
        cohort_map[r["mmsi"]] = r["source_combo"]
    df["cohort"] = df["mmsi"].map(cohort_map)
    # A contribution per cohort
    active_cohort = df[df["region"].isin(REGIONS) & df["sog"].between(2, 10, inclusive="both")].copy()
    ac_counts = active_cohort.groupby(["hour","region","mmsi"]).size().rename("n").reset_index()
    ac_qualified = ac_counts[ac_counts["n"] >= 3]
    ac_qualified["cohort"] = ac_qualified["mmsi"].map(cohort_map)
    cohort_daily = ac_qualified.groupby(["cohort", ac_qualified["hour"].dt.normalize()])["mmsi"].nunique().rename("vessels").reset_index()
    cohort_daily.columns = ["cohort","date","vessels"]
    # normal-day CV per cohort
    cohort_stats = []
    for c in cohort_daily["cohort"].unique():
        cd = cohort_daily[(cohort_daily["cohort"]==c)]
        cd_normal = cd[cd["date"].map(qmap)=="normal"]
        if len(cd_normal) >= 3:
            cohort_stats.append({"cohort": c, "n_vessels": int(vessel_sources[vessel_sources["source_combo"]==c].shape[0]),
                                 "normal_daily_mean": float(cd_normal["vessels"].mean()),
                                 "normal_daily_cv": float(cd_normal["vessels"].std(ddof=1)/cd_normal["vessels"].mean()) if cd_normal["vessels"].mean() else np.nan})
    cs = pd.DataFrame(cohort_stats).sort_values("normal_daily_mean", ascending=False) if cohort_stats else pd.DataFrame()
    print(f"\nCohort stats:")
    if len(cs): print(cs.to_string(index=False))

    # ========== 8. boundary sensitivity ==========
    boundary_rows = []
    for lo, hi, name in [(2.8,3.2,"3km"),(9.8,10.2,"10km"),(29.8,30.2,"30km")]:
        near = np.logical_and(dist_planar >= lo, dist_planar < hi)
        n_near = int(near.sum())
        n_diff = int(((region_planar != region_hav) & near).sum())
        boundary_rows.append({"boundary": name, "nearby_points": n_near, "reclassified": n_diff, "pct": n_diff/max(1,n_near)*100})
    bs = pd.DataFrame(boundary_rows)
    bs.to_csv(out_dir / "boundary_sensitivity.csv", index=False, encoding="utf-8-sig")

    # ========== 9. variant definitions ==========
    vd = pd.DataFrame([
        {"variant": "V0", "name": "current_raw_planar", "description": "exact make_a_labels reproduction"},
        {"variant": "V1", "name": "raw_haversine", "description": "haversine distance, no dedup"},
        {"variant": "V2", "name": "exact_message_dedup_haversine", "description": "dedup exact messages + haversine"},
        {"variant": "V3", "name": "vessel_timestamp_dedup_haversine", "description": "dedup per mmsi+timestamp, median coords + haversine"},
        {"variant": "V4", "name": "unique_timestamp_threshold_haversine", "description": "≥3 unique timestamps instead of ≥3 rows + haversine"},
        {"variant": "V5", "name": "per_source_threshold_union_haversine", "description": "≥3 per source, union + haversine"},
    ])
    vd.to_csv(out_dir / "label_variant_definitions.csv", index=False, encoding="utf-8-sig")

    # ========== 10. decision table ==========
    dec_rows = [
        {"issue": "planar_vs_haversine", "magnitude": f"{region_diff} reclassifications ({region_diff/len(df)*100:.2f}%)",
         "affected_a_cells": int(comp[comp["variant_i"]=="V1"]["different_cells"].iloc[0]) if len(comp[comp["variant_i"]=="V1"]) else 0,
         "likely_label_caliber_risk": "low" if region_diff/len(df) < 0.01 else "medium",
         "recommended_action": "switch_to_haversine" if region_diff/len(df) > 0.005 else "keep_current",
         "reason": f"{region_diff/len(df)*100:.2f}% of points reclassified"},
        {"issue": "exact_duplicate_effect", "magnitude": f"{exact_dup} duplicates ({exact_dup/len(df)*100:.2f}%)",
         "affected_a_cells": int(comp[comp["variant_i"]=="V2"]["different_cells"].iloc[0]) if len(comp[comp["variant_i"]=="V2"]) else 0,
         "likely_label_caliber_risk": "low" if exact_dup/len(df) < 0.01 else "medium",
         "recommended_action": "deduplicate_exact_messages" if exact_dup/len(df) > 0.001 else "keep_current",
         "reason": f"{exact_dup} exact duplicate rows"},
        {"issue": "timestamp_duplicate_effect", "magnitude": f"{ts_dup} duplicates ({ts_dup/len(df)*100:.2f}%)",
         "affected_a_cells": int(comp[comp["variant_i"]=="V3"]["different_cells"].iloc[0]) if len(comp[comp["variant_i"]=="V3"]) else 0,
         "likely_label_caliber_risk": "low" if ts_dup/len(df) < 0.05 else "medium",
         "recommended_action": "keep_current" if ts_dup/len(df) < 0.05 else "investigate_time_offset",
         "reason": f"{ts_dup} same-timestamp duplicates"},
        {"issue": "cross_source_threshold_effect", "magnitude": f"{n_cross_threshold} groups ({pct_cross:.1f}%)",
         "affected_a_cells": int(comp[comp["variant_i"]=="V5"]["different_cells"].iloc[0]) if len(comp[comp["variant_i"]=="V5"]) else 0,
         "likely_label_caliber_risk": "high" if pct_cross > 5 else ("medium" if pct_cross > 1 else "low"),
         "recommended_action": "use_per_source_threshold" if pct_cross > 5 else "keep_current",
         "reason": f"{pct_cross:.1f}% of qualified groups rely on cross-source merging"},
        {"issue": "source_time_alignment", "magnitude": f"{len(ta)} matched pairs" + (f", median {ta['time_diff_seconds'].median():.1f}s" if len(ta) else ""),
         "affected_a_cells": 0, "likely_label_caliber_risk": "unknown",
         "recommended_action": "investigate_time_offset" if len(ta) > 0 and abs(ta["time_diff_seconds"].median()) > 60 else "no_action",
         "reason": "see timestamp_alignment.csv"},
        {"issue": "source_cohort_stability", "magnitude": f"{len(cs)} cohorts" if len(cs) else "n/a",
         "affected_a_cells": 0, "likely_label_caliber_risk": "low",
         "recommended_action": "build_source_cohort_model" if len(cs) > 0 and cs["normal_daily_cv"].min() < 0.3 else "no_action",
         "reason": "see cohort analysis"},
    ]
    decision = pd.DataFrame(dec_rows)
    decision.to_csv(out_dir / "decision_table.csv", index=False, encoding="utf-8-sig")

    # ========== summary ==========
    L = ["# Step 27 A Label Caliber and Source Audit", "", f"Run command: `{RUN_CMD}`", ""]
    L.append("## Duplicate rates")
    L.append(f"- Exact message: {exact_dup} ({exact_dup/len(df)*100:.2f}%)")
    L.append(f"- Same vessel timestamp: {ts_dup} ({ts_dup/len(df)*100:.2f}%)")
    L.append(f"- Cross-source threshold groups: {n_cross_threshold}/{n_total_qualified} ({pct_cross:.1f}%)")
    L.append("")
    L.append("## Label variants vs V0 (current)")
    L.append("")
    L.append("| variant | different cells | % | max diff | total diff |")
    L.append("| --- | --- | --- | --- | --- |")
    for _, r in comp.iterrows():
        L.append(f"| {r['variant_i']} | {int(r['different_cells'])} | {r['pct_different']:.1f}% | {int(r['max_abs_diff'])} | {int(r['total_diff'])} |")
    L.append("")
    L.append(f"## Planar vs haversine\n- {region_diff} reclassifications ({region_diff/len(df)*100:.2f}% of points)\n")
    if len(ta):
        L.append(f"## Time alignment\n- {len(ta)} matched pairs, median diff {ta['time_diff_seconds'].median():.1f}s\n")
    if len(cs):
        L.append("## Source cohort stability")
        L.append("")
        L.append("| cohort | n_vessels | normal daily mean | normal CV |")
        L.append("| --- | --- | --- | --- |")
        for _, r in cs.head(7).iterrows():
            L.append(f"| {r['cohort']} | {int(r['n_vessels'])} | {r['normal_daily_mean']:.1f} | {r['normal_daily_cv']:.3f} |")
        L.append("")
    L.append("## Decision summary")
    L.append("")
    for _, r in decision.iterrows():
        L.append(f"- **{r['issue']}**: {r['recommended_action']} — {r['reason']}")
    L.append("")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    assert not any(out_dir.rglob("*.zip"))
    print(f"\n=== Key results ===")
    print(f"Exact duplicates: {exact_dup} ({exact_dup/len(df)*100:.2f}%)")
    print(f"Cross-source threshold groups: {n_cross_threshold}/{n_total_qualified} ({pct_cross:.1f}%)")
    print(f"Planar vs haversine: {region_diff} ({region_diff/len(df)*100:.2f}%)")
    print(f"V1 different cells: {comp[comp['variant_i']=='V1']['different_cells'].iloc[0]}")
    print(f"V5 different cells: {comp[comp['variant_i']=='V5']['different_cells'].iloc[0]}")
    print("\nDone.")


if __name__ == "__main__":
    main()
