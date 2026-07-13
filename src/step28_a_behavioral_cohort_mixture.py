"""Step 28 — A behavioral cohort mixture model benchmark.

Tests whether grouping vessels by behavioral patterns (dominant region, activity
intensity, day/night shift) and predicting cohort-level composition (using U_d)
outperforms the aggregate baseline. This is the missing middle ground between
single-vessel models (too noisy) and aggregate means (too homogeneous).

Outputs (under outputs/step28_a_behavioral_cohort_mixture/):
  1-11 as specified.
"""

from __future__ import annotations
import sys, warnings, math
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from optimized_baseline import REGION_ORDER, add_regions, make_a_labels  # noqa: E402

TRAIN_REL = Path("数据备份/训练集_20180101-0124_拖轮AIS.csv")
STEP07 = Path("outputs/step07_a_regularized_count_models/a_model_predictions.csv")
STEP01 = Path("outputs/step01_data_audit/train_daily_overview.csv")
STEP02 = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
OUT_REL = Path("outputs/step28_a_behavioral_cohort_mixture")
REGIONS = REGION_ORDER
RUN_CMD = "python " + " ".join(sys.argv)


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    # ---- load ----
    df = pd.read_csv(ROOT / TRAIN_REL, usecols=["mmsi","x","y","sog","time","source_dataset"],
                     dtype={"mmsi":"string","x":"float64","y":"float64","sog":"float32","source_dataset":"string"}, parse_dates=["time"])
    df["hour"] = df["time"].dt.floor("h"); df["date"] = df["time"].dt.normalize()
    df["hour_of_day"] = df["time"].dt.hour
    df = add_regions(df)

    daily = pd.read_csv(ROOT / STEP01); daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    aud = pd.read_csv(ROOT / STEP02); aud["date"] = pd.to_datetime(aud["date"]).dt.normalize()
    qmap = dict(zip(aud["date"], aud["quality_regime"]))
    vc_map = dict(zip(daily["date"], daily["unique_vessel_count"]))
    all_dates = sorted(pd.date_range("2018-01-01", "2018-01-24", freq="D").normalize())

    # ---- A labels ----
    a_lab = make_a_labels(df)
    a_lab["date"] = a_lab["hour"].dt.normalize(); a_lab["hour_of_day"] = a_lab["hour"].dt.hour
    a_pivot = a_lab.pivot_table(index=["date","hour_of_day"], columns="region", values="y", aggfunc="first").reset_index()
    for r in REGIONS:
        if r not in a_pivot.columns: a_pivot[r] = 0

    def get_a_vec(d):
        sub = a_pivot[a_pivot["date"]==d].sort_values("hour_of_day")
        m = np.zeros((24,3))
        for _, row in sub.iterrows():
            h = int(row["hour_of_day"])
            for ri, r in enumerate(REGIONS): m[h, ri] = float(row[r])
        return m

    # ---- vessel active records ----
    active = df[df["region"].isin(REGIONS) & df["sog"].between(2, 10, inclusive="both")]
    vrc = active.groupby(["mmsi","date","hour_of_day","region"]).size().rename("n").reset_index()
    va = vrc[vrc["n"] >= 3].copy()  # qualified active

    # ---- exact baseline from Step 07 ----
    a7 = pd.read_csv(ROOT / STEP07, encoding="utf-8-sig")
    a7["date"] = pd.to_datetime(a7["date"]).dt.normalize()
    bp_fa = a7[(a7["method"]=="decomp_mean_daytype_shrunk")&(a7["training_policy"]=="exclude_outage_severe")&(a7["evaluation_id"]=="final_analog")]
    bp_pivot = bp_fa.pivot_table(index=["date","hour_of_day"], columns="region", values="pred_rounded", aggfunc="first").reset_index()
    for r in REGIONS:
        if r not in bp_pivot.columns: bp_pivot[r] = 0

    def get_bl_pred(d):
        sub = bp_pivot[bp_pivot["date"]==d].sort_values("hour_of_day")
        m = np.zeros((24,3))
        for _, row in sub.iterrows():
            h = int(row["hour_of_day"])
            for ri, r in enumerate(REGIONS): m[h, ri] = float(row[r])
        return m

    # ---- post-outage block setup ----
    po_dates = list(pd.date_range("2018-01-19", "2018-01-24", freq="D").normalize())
    train_end = pd.Timestamp("2018-01-18")
    train_dates = [d for d in all_dates if d <= train_end]
    train_normal = [d for d in train_dates if qmap.get(d) == "normal"]
    all_mmsi = sorted(df["mmsi"].unique())
    print(f"Vessels: {len(all_mmsi)}, train normal dates: {len(train_normal)}")

    # ---- compute vessel behavioral features (training only) ----
    vfeat = {}
    for m in all_mmsi:
        mv = va[(va["mmsi"]==m) & (va["date"] <= train_end)]
        present_days = set()
        for d in train_dates:
            if m in df[df["date"]==d]["mmsi"].values: present_days.add(d)
        if not present_days: continue

        # dominant region
        if len(mv) > 0:
            reg_counts = mv.groupby("region").size()
            dom = reg_counts.idxmax()
            total_active = len(mv)
            shares = {r: float(reg_counts.get(r, 0)) / total_active for r in REGIONS}
        else:
            dom = "none"; shares = {r: 0 for r in REGIONS}; total_active = 0

        # active hours per present day
        active_days = mv["date"].nunique() if len(mv) else 0
        mah = total_active / active_days if active_days > 0 else 0

        # time-of-day
        if len(mv) > 0:
            night_mask = (mv["hour_of_day"] >= 18) | (mv["hour_of_day"] < 6)
            day_mask = (mv["hour_of_day"] >= 6) & (mv["hour_of_day"] < 18)
            night_share = float(night_mask.sum()) / len(mv)
            morning = ((mv["hour_of_day"]>=6)&(mv["hour_of_day"]<12)).sum() / len(mv)
            afternoon = ((mv["hour_of_day"]>=12)&(mv["hour_of_day"]<18)).sum() / len(mv)
        else:
            night_share = morning = afternoon = 0.5

        # presence frequency
        pres_freq = len(present_days) / len(train_dates)
        rec7_start = train_end - pd.Timedelta(days=7)
        rec7_days = len([d for d in present_days if d > rec7_start])
        rec7_freq = rec7_days / 7.0

        # source membership
        sources = tuple(sorted(df[df["mmsi"]==m]["source_dataset"].unique()))

        vfeat[m] = {"dominant_region": dom, "core_share": shares["core"], "near_share": shares["near"],
                     "outer_share": shares["outer"], "mean_active_hours": mah, "night_share": night_share,
                     "morning_share": morning, "afternoon_share": afternoon,
                     "presence_freq": pres_freq, "recent7_freq": rec7_freq, "source_group": sources,
                     "active_days": active_days, "present_days": len(present_days)}

    # ---- cohort assignment ----
    def assign_cohorts(scheme):
        assignments = {}
        for m, f in vfeat.items():
            if scheme == "aggregate_1":
                assignments[m] = "all"
            elif scheme == "dominant_region_3":
                assignments[m] = f["dominant_region"] if f["dominant_region"] in REGIONS else "outer"
            elif scheme == "region_intensity_6":
                base = f["dominant_region"] if f["dominant_region"] in REGIONS else "outer"
                assignments[m] = f"{base}_hi"  # placeholder, will split below
            elif scheme == "region_shift_6":
                base = f["dominant_region"] if f["dominant_region"] in REGIONS else "outer"
                shift = "night" if f["night_share"] > 0.5 else "day"
                assignments[m] = f"{base}_{shift}"
        # intensity split (needs median per region)
        if scheme == "region_intensity_6":
            for base_r in REGIONS:
                region_vessels = [m for m, f in vfeat.items() if f["dominant_region"] == base_r]
                if not region_vessels: continue
                mah_vals = [vfeat[m]["mean_active_hours"] for m in region_vessels]
                med = np.median(mah_vals)
                for m in region_vessels:
                    assignments[m] = f"{base_r}_{'hi' if vfeat[m]['mean_active_hours'] >= med else 'lo'}"
        return assignments

    # ---- compute cohort profiles ----
    def compute_cohort_profile(cohort_vessels, tau=20):
        """θ_{g,h,z}: contribution per vessel-present-day."""
        total_vpd = 0
        profile = np.zeros((24, 3))
        for m in cohort_vessels:
            mv = va[(va["mmsi"]==m) & (va["date"] <= train_end)]
            pres_days = vfeat.get(m, {}).get("present_days", 0)
            if pres_days == 0: continue
            total_vpd += pres_days
            for _, row in mv.iterrows():
                h = int(row["hour_of_day"]); ri = REGIONS.index(row["region"])
                profile[h, ri] += 1
        if total_vpd == 0: return np.zeros((24, 3)), 0
        return profile / total_vpd, total_vpd

    # global profile
    all_vessels = list(vfeat.keys())
    global_profile, global_vpd = compute_cohort_profile(all_vessels, tau=0)

    # new vessel ratio
    new_ratios = []
    for i in range(1, len(train_normal)):
        d = train_normal[i]; prev = set()
        for pd_ in train_normal[:i]: prev.update(df[df["date"]==pd_]["mmsi"].values)
        today = set(df[df["date"]==d]["mmsi"].values)
        if len(today) > 0: new_ratios.append(len(today - prev) / len(today))
    rho_new = float(np.median(new_ratios)) if new_ratios else 0.1

    # new vessel profile
    new_profile = np.zeros((24, 3)); n_new = 0
    for i in range(1, len(train_normal)):
        d = train_normal[i]; prev = set()
        for pd_ in train_normal[:i]: prev.update(df[df["date"]==pd_]["mmsi"].values)
        today = set(df[df["date"]==d]["mmsi"].values)
        new_v = today - prev
        for m in new_v:
            n_new += 1
            mv = va[(va["mmsi"]==m) & (va["date"]==d)]
            for _, row in mv.iterrows():
                h = int(row["hour_of_day"]); ri = REGIONS.index(row["region"])
                new_profile[h, ri] += 1
    new_profile = new_profile / n_new if n_new > 0 else global_profile.copy()

    # ---- cohort daily counts (training normal) ----
    def get_cohort_counts(assignments):
        """For each training normal date, count vessels per cohort."""
        counts = {}
        for d in train_normal:
            present = set(df[df["date"]==d]["mmsi"].values)
            c = {}
            for m in present:
                g = assignments.get(m, "unknown")
                c[g] = c.get(g, 0) + 1
            counts[d] = c
        return counts

    # ---- predict target cohort counts ----
    def predict_counts(assignments, method="long", Ud=50, use_new_pool=True):
        counts = get_cohort_counts(assignments)
        cohorts = sorted(set(assignments.values()))
        # compute shares
        if method == "long":
            share_sums = {g: 0 for g in cohorts}; total = 0
            for d in train_normal:
                c = counts.get(d, {})
                t = sum(c.values())
                if t > 0:
                    for g in cohorts: share_sums[g] += c.get(g, 0) / t
                    total += 1
            shares = {g: share_sums[g] / total if total else 1/len(cohorts) for g in cohorts}
        elif method in ("recent3", "recent7"):
            n_rec = 3 if method == "recent3" else 7
            rec_dates = train_normal[-n_rec:] if len(train_normal) >= n_rec else train_normal
            share_sums = {g: 0 for g in cohorts}; total = 0
            for d in rec_dates:
                c = counts.get(d, {}); t = sum(c.values())
                if t > 0:
                    for g in cohorts: share_sums[g] += c.get(g, 0) / t
                    total += 1
            shares = {g: share_sums[g] / total if total else 1/len(cohorts) for g in cohorts}
        elif method == "blend":
            # long
            sl = {g: 0 for g in cohorts}; tl = 0
            for d in train_normal:
                c = counts.get(d, {}); t = sum(c.values())
                if t > 0:
                    for g in cohorts: sl[g] += c.get(g,0)/t; tl += 1
            sl = {g: sl[g]/tl if tl else 0 for g in cohorts}
            # recent7
            rec_dates = train_normal[-7:]; sr = {g: 0 for g in cohorts}; tr = 0
            for d in rec_dates:
                c = counts.get(d, {}); t = sum(c.values())
                if t > 0:
                    for g in cohorts: sr[g] += c.get(g,0)/t; tr += 1
            sr = {g: sr[g]/tr if tr else 0 for g in cohorts}
            shares = {g: 0.5*sl[g] + 0.5*sr[g] for g in cohorts}
        else:
            shares = {g: 1/len(cohorts) for g in cohorts}
        # normalize
        total_share = sum(shares.values())
        if total_share > 0: shares = {g: s/total_share for g, s in shares.items()}
        # allocate U_d
        U_new = Ud * rho_new if use_new_pool else 0
        U_hist = Ud - U_new
        N_g = {g: U_hist * shares.get(g, 0) for g in cohorts}
        return N_g, U_new

    # ---- predict A for a target date ----
    def predict_cohort_a(d, scheme, share_method="long", tau=20, use_new_pool=True, use_true_counts=False):
        assignments = assign_cohorts(scheme)
        cohorts = sorted(set(assignments.values()))
        Ud = vc_map.get(d, 50)

        # cohort profiles with shrinkage
        profiles = {}; vpds = {}
        for g in cohorts:
            gv = [m for m, a in assignments.items() if a == g]
            raw_prof, vpd = compute_cohort_profile(gv, tau=0)
            shrunk = (vpd * raw_prof + tau * global_profile) / (vpd + tau) if vpd > 0 else global_profile
            profiles[g] = shrunk; vpds[g] = vpd

        # counts
        if use_true_counts:
            # oracle: true cohort counts from validation day
            present = set(df[df["date"]==d]["mmsi"].values)
            true_counts = {}
            for m in present:
                g = assignments.get(m, "unknown")
                true_counts[g] = true_counts.get(g, 0) + 1
            N_g = {g: float(true_counts.get(g, 0)) for g in cohorts}
            U_new = 0  # oracle sees all vessels
        else:
            N_g, U_new = predict_counts(assignments, share_method, Ud, use_new_pool)

        # aggregate
        pred = np.zeros((24, 3))
        for g in cohorts:
            pred += N_g.get(g, 0) * profiles[g]
        if use_new_pool and U_new > 0:
            pred += U_new * new_profile

        return np.clip(np.round(pred), 0, None).astype(int), N_g, U_new

    # ---- score methods ----
    schemes = ["aggregate_1", "dominant_region_3", "region_intensity_6", "region_shift_6"]
    share_methods = ["long", "recent3", "recent7", "blend"]
    methods = ["exact_step11_baseline"]
    for sc in schemes:
        for sm in share_methods:
            methods.append(f"cohort_{sc}_{sm}")
    methods += ["oracle_true_cohort_counts_dominant3"]

    print(f"\n=== Evaluating {len(methods)} methods on post-outage block ===")
    results = {}
    for method in methods:
        total_sse = 0; region_sses = np.zeros(3); daily = []
        for d in po_dates:
            true_vec = get_a_vec(d)
            if method == "exact_step11_baseline":
                pred = get_bl_pred(d)
            elif method.startswith("oracle"):
                pred, _, _ = predict_cohort_a(d, "dominant_region_3", "long", 20, False, use_true_counts=True)
            else:
                parts = method.split("_")
                # scheme is everything between "cohort_" and the share method
                sm = parts[-1]  # long, recent3, recent7, blend
                sc = "_".join(parts[1:-1])
                pred, _, _ = predict_cohort_a(d, sc, sm, 20, use_new_pool=True)
            sse = float(((pred - true_vec)**2).sum())
            total_sse += sse; daily.append({"date": f"{d:%Y-%m-%d}", "sse": sse})
            for ri in range(3): region_sses[ri] += float(((pred[:, ri] - true_vec[:, ri])**2).sum())
        results[method] = {"total_sse": total_sse, "daily": daily, "region_sses": region_sses}
        print(f"  {method:52s} SSE={total_sse:.0f}")

    bl_sse = results["exact_step11_baseline"]["total_sse"]
    print(f"\nBaseline: {bl_sse}")

    # ---- oracle with global profile ----
    oracle_global_sse = 0
    assignments = assign_cohorts("dominant_region_3")
    cohorts = sorted(set(assignments.values()))
    for d in po_dates:
        true_vec = get_a_vec(d)
        present = set(df[df["date"]==d]["mmsi"].values)
        true_counts = {}
        for m in present:
            g = assignments.get(m, "unknown")
            true_counts[g] = true_counts.get(g, 0) + 1
        pred = np.zeros((24, 3))
        for g in cohorts:
            pred += float(true_counts.get(g, 0)) * global_profile
        oracle_global_sse += float(((pred - true_vec)**2).sum())
    print(f"  oracle_true_counts_global_profile               SSE={oracle_global_sse:.0f}")

    # ---- write outputs ----
    # method definitions
    md_rows = [{"method": "exact_step11_baseline", "deployable": True}]
    for m in methods[1:]:
        dep = not m.startswith("oracle")
        md_rows.append({"method": m, "deployable": dep})
    md_rows.append({"method": "oracle_true_counts_global_profile", "deployable": False})
    pd.DataFrame(md_rows).to_csv(out_dir / "method_definitions.csv", index=False, encoding="utf-8-sig")

    # daily scores
    ds_rows = []
    for method in methods:
        for dr in results[method]["daily"]:
            bl_daily = [x["sse"] for x in results["exact_step11_baseline"]["daily"] if x["date"]==dr["date"]][0]
            ds_rows.append({"method": method, "date": dr["date"], "sse": dr["sse"], "delta_vs_baseline": dr["sse"] - bl_daily})
    pd.DataFrame(ds_rows).to_csv(out_dir / "daily_scores.csv", index=False, encoding="utf-8-sig")

    # region scores
    rs_rows = []
    for method in methods:
        for ri, r in enumerate(REGIONS):
            rs_rows.append({"method": method, "region": r, "sse": float(results[method]["region_sses"][ri])})
    pd.DataFrame(rs_rows).to_csv(out_dir / "region_scores.csv", index=False, encoding="utf-8-sig")

    # oracle gap
    og = pd.DataFrame([
        {"method": "oracle_true_cohort_counts_dominant3", "sse": results["oracle_true_cohort_counts_dominant3"]["total_sse"],
         "baseline_sse": bl_sse, "reduction_ratio": 1 - results["oracle_true_cohort_counts_dominant3"]["total_sse"]/bl_sse},
        {"method": "oracle_true_counts_global_profile", "sse": oracle_global_sse,
         "baseline_sse": bl_sse, "reduction_ratio": 1 - oracle_global_sse/bl_sse},
    ])
    og.to_csv(out_dir / "oracle_gap.csv", index=False, encoding="utf-8-sig")

    # decision table
    dec_rows = []
    for method in methods:
        r = results[method]; dep = not method.startswith("oracle")
        red = 1 - r["total_sse"]/bl_sse if bl_sse else 0
        dec_rows.append({"method": method, "deployable": dep, "total_sse": r["total_sse"],
                         "baseline_sse": bl_sse, "reduction_ratio": red, "passes_5pct": dep and red >= 0.05})
    decision = pd.DataFrame(dec_rows)
    decision.to_csv(out_dir / "decision_table.csv", index=False, encoding="utf-8-sig")

    # cohort assignments (dominant_region_3)
    ca_rows = [{"mmsi": m, "cohort_dominant3": assign_cohorts("dominant_region_3").get(m, "unknown"),
                **f} for m, f in vfeat.items()]
    pd.DataFrame(ca_rows).to_csv(out_dir / "cohort_assignments.csv", index=False, encoding="utf-8-sig")

    # summary
    L = ["# Step 28 A Behavioral Cohort Mixture", "", f"Run command: `{RUN_CMD}`", ""]
    L.append("## Results (post-outage block, Jan 19-24)")
    L.append("")
    L.append("| method | SSE | reduction | deployable |")
    L.append("| --- | --- | --- | --- |")
    for _, r in decision.iterrows():
        L.append(f"| {r['method']} | {r['total_sse']:.0f} | {r['reduction_ratio']*100:.1f}% | {r['deployable']} |")
    L.append(f"| oracle_true_counts_global_profile | {oracle_global_sse:.0f} | {(1-oracle_global_sse/bl_sse)*100:.1f}% | False |")
    L.append("")
    n_pass = int(decision["passes_5pct"].sum())
    oracle_red = (1 - results["oracle_true_cohort_counts_dominant3"]["total_sse"]/bl_sse) * 100
    oracle_glob_red = (1 - oracle_global_sse/bl_sse) * 100
    L.append(f"## Key findings")
    L.append(f"- Deployable methods passing 5%: {n_pass}")
    L.append(f"- Oracle true cohort counts (dominant3) reduction: {oracle_red:.1f}%")
    L.append(f"- Oracle true counts + global profile reduction: {oracle_glob_red:.1f}%")
    L.append(f"- New vessel ratio: {rho_new:.3f}")
    L.append("")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    assert not any(out_dir.rglob("*.zip"))
    print(f"\n=== Summary ===")
    for _, r in decision.iterrows():
        print(f"  {r['method']:52s} SSE={r['total_sse']:.0f} red={r['reduction_ratio']*100:.1f}% dep={r['deployable']}")
    print(f"  oracle_global: {oracle_global_sse:.0f} ({oracle_glob_red:.1f}%)")
    print(f"Deployable passing 5%: {n_pass}")
    print("\nDone.")


if __name__ == "__main__":
    main()
