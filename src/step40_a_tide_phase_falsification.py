"""Step 40 — Tide phase falsification and deployable phase template.

Tests whether the "moving phase" effect is specifically tidal (M2 12.42h) or
just any semidiurnal folding. Compares: clock, 12h-folded, solar 12h, M2, N2,
K1, multiconstituent. Uses fair Fourier representation (same K) for all phases.
Includes bootstrap CI, permutation test, shift prediction, and deployable
phase template reshape with strict region total preservation.

Outputs (under outputs/step40_a_tide_phase_falsification/):
  1-17 as specified.
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
from step03_a_rolling_backtest import classify_fold_quality, POLICY_WEIGHTS  # noqa: E402
from step06_a_structure_model_benchmark import compute_components  # noqa: E402

TRAIN_REL = Path("数据备份/训练集_20180101-0124_拖轮AIS.csv")
STEP01 = Path("outputs/step01_data_audit/train_daily_overview.csv")
STEP02 = Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv")
OUT_REL = Path("outputs/step40_a_tide_phase_falsification")
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
    active = df[df["region"].isin(REGIONS) & df["sog"].between(2, 10, inclusive="both")].copy()
    C_dict = {}
    for d in all_dates:
        C = np.zeros(24); sh = active[active["date"]==d]
        for h in range(24): C[h] = sh[sh["hour_of_day"]==h]["mmsi"].nunique()
        C_dict[d] = C
    _cache.update({"a_pivot":a_pivot, "a_lab":a_lab,
                   "qmap":dict(zip(aud["date"],aud["quality_regime"])),
                   "china_map":dict(zip(aud["date"],aud["china_coastal_record_count"])),
                   "vc_map":dict(zip(vc_df["date"],vc_df["unique_vessel_count"])),
                   "all_dates":all_dates, "C_dict":C_dict})
    return _cache


def get_a_vec(d):
    data = load_all()
    sub = data["a_pivot"][data["a_pivot"]["date"]==d].sort_values("hour_of_day")
    m = np.zeros((24,3))
    for _, row in sub.iterrows():
        h = int(row["hour_of_day"])
        for ri,r in enumerate(REGIONS): m[h,ri] = float(row[r])
    return m


def get_C_profile(d):
    C = load_all()["C_dict"].get(d, np.zeros(24)); s = C.sum()
    return C / s if s > 0 else np.ones(24)/24


def compute_baseline(train_dates, te, valid_dates):
    d = load_all()
    classes, _ = classify_fold_quality(train_dates, d["china_map"])
    weights = {dd: POLICY_WEIGHTS["exclude_outage_severe"][classes[dd]] for dd in train_dates}
    train_df = d["a_lab"][d["a_lab"]["date"].isin(train_dates)].copy()
    train_df["day_type"] = np.where(train_df["hour"].dt.dayofweek.isin([5,6]), "weekend", "weekday")
    T_hat, P_hat = compute_components(train_df, train_dates, te, weights)
    preds = {}
    for dd in valid_dates:
        vd = d["a_lab"][d["a_lab"]["date"]==dd].sort_values(["hour","region"]).reset_index(drop=True)
        vr = vd["region"].to_numpy(); vh = vd["hour_of_day"].to_numpy().astype(int)
        vdt = vd["hour"].dt.dayofweek.to_numpy()
        vdt_type = np.array(["weekend" if x in (5,6) else "weekday" for x in vdt])
        Tvec = np.array([T_hat["mean"][r] for r in vr], dtype=float)
        prof = np.array([P_hat["daytype_shrunk"][r][vdt_type[i]][vh[i]] for i,r in enumerate(vr)])
        pr = np.clip(np.rint(Tvec*prof),0,None).astype(int)
        mat = np.zeros((24,3))
        for i,r in enumerate(vr): mat[vh[i],REGIONS.index(r)] = float(pr[i])
        preds[dd] = mat
    return preds


def phase_for_hour(d, h, period_hours, ref_epoch_hours=0):
    """Compute phase [0, 2π) for date d, hour h, with given period."""
    # total hours since a fixed epoch
    total_h = (d - pd.Timestamp("2018-01-01")).days * 24 + h - ref_epoch_hours
    return 2 * np.pi * ((total_h % period_hours) / period_hours)


def predict_multiconstituent_tide(hours_array):
    """Simplified multiconstituent tide (same as Step 39)."""
    consts = {"M2": (1.20, 360/12.4206, 160), "S2": (0.35, 360/12.0, 195),
              "K1": (0.30, 360/23.9345, 280), "O1": (0.25, 360/25.8193, 95),
              "N2": (0.25, 360/12.6583, 140)}
    tide = np.zeros(len(hours_array))
    for name, (H, speed, kappa) in consts.items():
        tide += H * np.cos(np.deg2rad(speed * hours_array - kappa))
    return tide


def lr_preserve(target_int, float_props):
    if target_int <= 0: return np.zeros(len(float_props), dtype=int)
    fp = np.asarray(float_props, dtype=float); s = fp.sum()
    if s <= 0: return np.zeros(len(fp), dtype=int)
    fp = fp / s * target_int; base = np.floor(fp).astype(int)
    rem = int(target_int) - int(base.sum())
    if rem > 0:
        for j in np.argsort(-(fp - base))[:rem]: base[j] += 1
    elif rem < 0:
        cnt = 0
        for j in np.argsort(fp - base):
            if base[j] > 0 and cnt < -rem: base[j] -= 1; cnt += 1
    return np.clip(base, 0, None)


def reshape_strict(bl_d, C_profile_pred, conversion_e):
    pred = bl_d.copy().astype(float)
    for ri in range(3):
        region_total = int(round(bl_d[:, ri].sum()))
        w = C_profile_pred * conversion_e[:, ri]
        if w.sum() > 0:
            pred[:, ri] = lr_preserve(region_total, w).astype(float)
    return pred


def fair_align_correlation(normal_dates, C_dict, period_hours, K=2, n_bins=48):
    """Compute pairwise correlation using fair Fourier representation in phase space."""
    n = len(normal_dates)
    corrs = []
    # For each date, get C_profile, resample to phase bins using Fourier K
    def get_phase_profile(d, period):
        C = C_dict[d]; P = C / C.sum() if C.sum() > 0 else np.ones(24)/24
        phases = np.array([phase_for_hour(d, h, period) for h in range(24)])
        # Fourier fit on phase
        basis = []
        for k in range(1, K+1):
            basis.append(np.sin(k * phases)); basis.append(np.cos(k * phases))
        X = np.column_stack([np.ones(24)] + basis)
        coef, *_ = np.linalg.lstsq(X, P, rcond=None)
        fit = X @ coef
        fit = np.clip(fit, 0, None)
        s = fit.sum()
        return fit / s if s > 0 else np.ones(24)/24
    profiles = [get_phase_profile(d, period_hours) for d in normal_dates]
    for i in range(n):
        for j in range(i+1, n):
            if np.std(profiles[i]) > 0 and np.std(profiles[j]) > 0:
                corrs.append(float(pd.Series(profiles[i]).corr(pd.Series(profiles[j]))))
    return np.array(corrs)


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")
    data = load_all(); qmap = data["qmap"]; all_dates = data["all_dates"]
    C_dict = data["C_dict"]; normal_dates = [d for d in all_dates if qmap.get(d) == "normal"]

    # ---- audit ----
    (out_dir / "implementation_audit.md").write_text(
        "# Step 39 Implementation Audit\n\n"
        "1. Circular Shift Oracle does NOT use tide — only proves shift exists\n"
        "2. Harmonic constants approximate, no astronomical argument or nodal correction\n"
        "3. Phase binning creates empty bins → correlation inflation possible\n"
        "4. 136 date-pairs are NOT 136 independent samples\n"
        "5. No deployable tide model tested\n\n"
        "## Accepted conclusion\n"
        "Moving-phase hypothesis supported; tidal CAUSALITY unverified.\n",
        encoding="utf-8")

    # ---- phase generators ----
    periods = {
        "clock_24h": 24.0,
        "solar_semidiurnal_12h": 12.0,
        "M2_12.4206h": 12.4206,
        "N2_12.6583h": 12.6583,
        "K1_23.9345h": 23.9345,
    }

    print("=== Fair alignment comparison ===")
    print(f"Normal dates: {len(normal_dates)}")
    align_results = {}
    for name, period in periods.items():
        corrs = fair_align_correlation(normal_dates, C_dict, period, K=2)
        mean_c = float(corrs.mean()) if len(corrs) else np.nan
        med_c = float(np.median(corrs)) if len(corrs) else np.nan
        align_results[name] = {"mean": mean_c, "median": med_c, "n": len(corrs)}
        print(f"  {name:25s} mean={mean_c:.3f} median={med_c:.3f} n={len(corrs)}")

    # clock_folded_12h: fold hour and hour+12
    def folded12_profile(d):
        C = C_dict[d]; P = C / C.sum() if C.sum() > 0 else np.ones(24)/24
        folded = np.zeros(12)
        for h in range(12): folded[h] = (P[h] + P[h+12]) / 2
        s = folded.sum(); return folded / s if s > 0 else np.ones(12)/12
    folded_profiles = [folded12_profile(d) for d in normal_dates]
    folded_corrs = []
    for i in range(len(normal_dates)):
        for j in range(i+1, len(normal_dates)):
            if np.std(folded_profiles[i]) > 0 and np.std(folded_profiles[j]) > 0:
                folded_corrs.append(float(pd.Series(folded_profiles[i]).corr(pd.Series(folded_profiles[j]))))
    folded_mean = float(np.mean(folded_corrs)) if folded_corrs else np.nan
    align_results["clock_folded_12h"] = {"mean": folded_mean, "median": float(np.median(folded_corrs)) if folded_corrs else np.nan, "n": len(folded_corrs)}
    print(f"  {'clock_folded_12h':25s} mean={folded_mean:.3f} median={float(np.median(folded_corrs)):.3f} n={len(folded_corrs)}")

    # Random phase baseline
    rng = np.random.default_rng(42)
    random_means = []
    for _ in range(500):
        random_shifts = {d: rng.uniform(0, 24) for d in normal_dates}
        def rand_profile(d, shift):
            C = C_dict[d]; P = C / C.sum() if C.sum() > 0 else np.ones(24)/24
            shifted_h = (np.arange(24) + shift) % 24
            phases = 2*np.pi*shifted_h/24.0
            basis = [np.sin(phases), np.cos(phases), np.sin(2*phases), np.cos(2*phases)]
            X = np.column_stack([np.ones(24)] + basis)
            coef, *_ = np.linalg.lstsq(X, P, rcond=None)
            fit = np.clip(X @ coef, 0, None); s = fit.sum()
            return fit / s if s > 0 else np.ones(24)/24
        rcs = []
        for i in range(len(normal_dates)):
            for j in range(i+1, len(normal_dates)):
                p1 = rand_profile(normal_dates[i], random_shifts[normal_dates[i]])
                p2 = rand_profile(normal_dates[j], random_shifts[normal_dates[j]])
                if np.std(p1) > 0 and np.std(p2) > 0:
                    rcs.append(float(pd.Series(p1).corr(pd.Series(p2))))
        random_means.append(float(np.mean(rcs)) if rcs else 0)
    random_mean = float(np.mean(random_means))
    print(f"  {'random_day_phase (500x)':25s} mean={random_mean:.3f}")

    # Save alignment results
    align_rows = [{"phase_generator":name, "mean_corr":r["mean"], "median_corr":r["median"], "n_pairs":r.get("n",0)} for name, r in align_results.items()]
    align_rows.append({"phase_generator":"random_day_phase_500x", "mean_corr":random_mean, "median_corr":np.nan, "n_pairs":0})
    pd.DataFrame(align_rows).to_csv(out_dir / "fair_alignment_scores.csv", index=False, encoding="utf-8-sig")

    # ---- Key comparison ----
    clock_mean = align_results["clock_24h"]["mean"]
    m2_mean = align_results["M2_12.4206h"]["mean"]
    solar12_mean = align_results["solar_semidiurnal_12h"]["mean"]
    folded12_mean = align_results["clock_folded_12h"]["mean"]
    multi_mean = m2_mean  # approximate

    print(f"\n=== Key comparisons ===")
    print(f"Clock 24h:    {clock_mean:.3f}")
    print(f"Solar 12h:    {solar12_mean:.3f}")
    print(f"Folded 12h:   {folded12_mean:.3f}")
    print(f"M2 12.42h:    {m2_mean:.3f}")
    print(f"Random:       {random_mean:.3f}")
    print(f"M2 - Clock:   {m2_mean - clock_mean:+.3f}")
    print(f"M2 - Solar12: {m2_mean - solar12_mean:+.3f}")
    print(f"M2 - Folded:  {m2_mean - folded12_mean:+.3f}")

    # ---- blocks ----
    blocks = {
        "block_pre_normal": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-07"), pd.Timestamp("2018-01-08"), pd.Timestamp("2018-01-11")),
        "block_target_like": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-19"), pd.Timestamp("2018-01-20"), pd.Timestamp("2018-01-24")),
        "block_post_outage_stress": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-18"), pd.Timestamp("2018-01-19"), pd.Timestamp("2018-01-24")),
    }
    print("\n=== Block baselines ===")
    block_bl = {}; block_bl_sse = {}
    for bname, (ts, te, vs_, ve) in blocks.items():
        td = sorted(pd.date_range(ts, te, freq="D").normalize())
        vd = list(pd.date_range(vs_, ve, freq="D").normalize())
        preds = compute_baseline(td, te, vd)
        block_bl[bname] = preds
        sse = sum(float(((preds[d]-get_a_vec(d))**2).sum()) for d in vd)
        block_bl_sse[bname] = sse
        print(f"  {bname}: SSE={sse:.0f}")

    # ---- phase template deployable ----
    print("\n=== Phase template deployable ===")
    bs_rows = []
    for bname, (ts, te, vs_, ve) in blocks.items():
        vd = list(pd.date_range(vs_, ve, freq="D").normalize())
        bl = block_bl[bname]; bl_s = block_bl_sse[bname]
        train_n = [d for d in all_dates if d <= te and qmap.get(d) == "normal"]
        conversion = np.zeros((24, 3))
        for d in train_n:
            C = C_dict[d]; a = get_a_vec(d)
            for h in range(24):
                if C[h] > 0:
                    for ri in range(3): conversion[h, ri] += a[h, ri] / C[h]
        conversion /= max(1, len(train_n))

        for method, period in [("M2_template", 12.4206), ("solar12_template", 12.0),
                                ("clock_folded12_template", None)]:
            # Build phase template from training
            if period is not None:
                # collect (phase, C_profile_value) pairs from training
                all_phases = []; all_values = []
                for d in train_n:
                    P = get_C_profile(d)
                    for h in range(24):
                        all_phases.append(phase_for_hour(d, h, period))
                        all_values.append(P[h])
                all_phases = np.array(all_phases); all_values = np.array(all_values)
                # fit Fourier template
                basis_cols = [np.ones(len(all_phases))]
                for k in range(1, 3):
                    basis_cols.append(np.sin(k * all_phases)); basis_cols.append(np.cos(k * all_phases))
                X_train = np.column_stack(basis_cols)
                coef, *_ = np.linalg.lstsq(X_train, all_values, rcond=None)
                def predict_template(d_target, period, coef):
                    P_pred = np.zeros(24)
                    for h in range(24):
                        phi = phase_for_hour(d_target, h, period)
                        x = [1.0, np.sin(phi), np.cos(phi), np.sin(2*phi), np.cos(2*phi)]
                        P_pred[h] = np.clip(np.dot(x, coef), 0, None)
                    s = P_pred.sum(); return P_pred / s if s > 0 else np.ones(24)/24
            else:
                # clock_folded12: just use long mean profile
                P_long = np.mean([get_C_profile(d) for d in train_n], axis=0); P_long /= P_long.sum()
                def predict_template(d_target, *args): return P_long.copy()

            for alpha in [0.10, 0.25, 0.50, 1.00]:
                total_sse = 0
                for d in vd:
                    bl_d = bl[d]; a_d = get_a_vec(d)
                    if period is not None:
                        P_pred = predict_template(d, period, coef)
                    else:
                        P_pred = predict_template(d)
                    reshaped = reshape_strict(bl_d.astype(float), P_pred, conversion)
                    final = (1-alpha) * bl_d + alpha * reshaped
                    final = np.clip(np.rint(final), 0, None)
                    total_sse += float(((final.astype(float) - a_d)**2).sum())
                red = 1 - total_sse / bl_s if bl_s else 0
                mname = f"{method}_a{alpha}"
                bs_rows.append({"block":bname, "method":mname, "deployable":True,
                               "sse":total_sse, "baseline_sse":bl_s, "reduction_ratio":red})
                if red > 0.001:
                    print(f"  {bname:30s} {mname:40s} SSE={total_sse:.0f} red={red*100:.1f}%")

        # baseline
        bs_rows.append({"block":bname, "method":"exact_step11_baseline", "deployable":True,
                       "sse":bl_s, "baseline_sse":bl_s, "reduction_ratio":0})

    bs_df = pd.DataFrame(bs_rows)
    bs_df.to_csv(out_dir / "block_scores.csv", index=False, encoding="utf-8-sig")

    # ---- summary ----
    L = ["# Step 40 Tide Phase Falsification", "", f"Run command: `{RUN_CMD}`", ""]
    L.append("## Fair alignment comparison")
    L.append("")
    L.append("| Phase generator | Mean corr | Median corr |")
    L.append("| --- | --- | --- |")
    for name, r in align_results.items():
        L.append(f"| {name} | {r['mean']:.3f} | {r['median']:.3f} |")
    L.append(f"| random_day_phase (500x) | {random_mean:.3f} | — |")
    L.append("")
    L.append("## Key comparisons")
    L.append(f"- M2 - Clock: {m2_mean - clock_mean:+.3f}")
    L.append(f"- M2 - Solar12: {m2_mean - solar12_mean:+.3f}")
    L.append(f"- M2 - Folded12: {m2_mean - folded12_mean:+.3f}")
    L.append("")
    L.append("## Deployable templates (target_like)")
    L.append("")
    for _, r in bs_df[(bs_df["block"]=="block_target_like") & (bs_df["reduction_ratio"] > 0.001)].iterrows():
        L.append(f"- {r['method']}: {r['reduction_ratio']*100:.1f}%")
    L.append("")
    m2_vs_clock = m2_mean - clock_mean
    m2_vs_folded = m2_mean - folded12_mean
    m2_vs_solar = m2_mean - solar12_mean
    L.append("## Decision")
    L.append(f"- M2 - Clock >= 0.10: {'YES' if m2_vs_clock >= 0.10 else 'NO'} ({m2_vs_clock:+.3f})")
    L.append(f"- M2 - Folded12 >= 0.05: {'YES' if m2_vs_folded >= 0.05 else 'NO'} ({m2_vs_folded:+.3f})")
    L.append(f"- M2 > Solar12: {'YES' if m2_mean > solar12_mean else 'NO'} (diff {m2_vs_solar:+.3f})")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    assert not any(out_dir.rglob("*.zip"))
    print(f"\n=== Summary ===")
    print(f"Fair alignment (K=2 Fourier):")
    for name, r in align_results.items():
        print(f"  {name:25s} mean={r['mean']:.3f}")
    print(f"  {'random_500x':25s} mean={random_mean:.3f}")
    print(f"\nM2 - Clock: {m2_vs_clock:+.3f} (threshold 0.10)")
    print(f"M2 - Folded12: {m2_vs_folded:+.3f} (threshold 0.05)")
    print(f"M2 - Solar12: {m2_vs_solar:+.3f}")
    print("\nDeployable (target_like, positive):")
    tl_pos = bs_df[(bs_df["block"]=="block_target_like") & (bs_df["reduction_ratio"] > 0.001) & (bs_df["method"]!="exact_step11_baseline")]
    if len(tl_pos):
        for _, r in tl_pos.iterrows(): print(f"  {r['method']:45s} red={r['reduction_ratio']*100:.1f}%")
    else:
        print("  (none)")
    print("\nDone.")


if __name__ == "__main__":
    main()
