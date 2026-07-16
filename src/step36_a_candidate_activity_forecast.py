"""Step 36 — Candidate activity (C) predictability and Oracle decomposition.

Decomposes Step 35's true-C Oracle (54.8%) into daily-total vs hour-profile
components. Tests whether C can be predicted from long/recent/lag7/lag14
methods, with special attention to the lag7-normal information condition
(relevant because target period 01-26..01-31 has normal lag7 sources).

Outputs (under outputs/step36_a_candidate_activity_forecast/):
  1-15 as specified.
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
OUT_REL = Path("outputs/step36_a_candidate_activity_forecast")
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
    _cache.update({"df":df, "a_lab":a_lab, "a_pivot":a_pivot, "active":active,
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


def get_C_for_date(d):
    """Compute C_h (active candidate vessels per hour) for date d."""
    data = load_all()
    sh = data["active"][data["active"]["date"]==d]
    C = np.zeros(24)
    for h in range(24):
        C[h] = sh[sh["hour_of_day"]==h]["mmsi"].nunique()
    return C


def get_MQ_for_date(d):
    """Compute M_h and Q_h for date d."""
    data = load_all()
    sh = data["active"][data["active"]["date"]==d]
    M = np.zeros(24); Q = np.zeros(24)
    for h in range(24):
        sh_h = sh[sh["hour_of_day"]==h]
        C_h = sh_h["mmsi"].nunique()
        if C_h == 0: M[h] = 1.0; Q[h] = 0.0; continue
        # N_ge1 per region sum
        n_ge1 = 0; n_ge3 = 0
        for r in REGIONS:
            counts = sh_h[sh_h["region"]==r].groupby("mmsi").size()
            n_ge1 += int((counts >= 1).sum())
            n_ge3 += int((counts >= 3).sum())
        M[h] = n_ge1 / C_h
        Q[h] = n_ge3 / n_ge1 if n_ge1 > 0 else 0.0
    return M, Q


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")
    data = load_all(); qmap = data["qmap"]; all_dates = data["all_dates"]; vc_map = data["vc_map"]
    normal_dates = [d for d in all_dates if qmap.get(d) == "normal"]

    # ---- audit ----
    (out_dir / "implementation_audit.md").write_text(
        "# Implementation Audit\n\n"
        "## Step 35\n- Correctly identified C_h as main A generation component\n"
        "- oracle_true_C uses 24 target-day true values — structural upper bound, not prediction\n"
        "- true-stable Oracle ≈ using target A's main component — proves composition, not predictability\n"
        "- corr(C_daily, U_d) ≈ 0.086 — U_d does NOT directly explain C\n\n"
        "## This step\n- Decomposes true-C Oracle into total vs profile\n"
        "- Tests deployable C prediction from long/recent/lag7/lag14\n"
        "- Special attention to lag7-normal information condition\n",
        encoding="utf-8")

    # ---- compute C, M, Q for all dates ----
    print("=== Computing C, M, Q ===")
    C_all = {}; MQ_all = {}
    for d in all_dates:
        C_all[d] = get_C_for_date(d)
        M, Q = get_MQ_for_date(d)
        MQ_all[d] = {"M": M, "Q": Q}
    print(f"Computed for {len(C_all)} dates")

    # ---- daily C stats ----
    ca_daily_rows = []
    for d in all_dates:
        C = C_all[d]; TC = float(C.sum()); Ud = vc_map.get(d, 0)
        O_d = TC / Ud if Ud > 0 else 0
        ca_daily_rows.append({"date":f"{d:%Y-%m-%d}", "quality":qmap.get(d,""),
                            "U_d":Ud, "C_total":TC, "active_hours_per_vessel":O_d,
                            "day_type":"weekend" if d.dayofweek in (5,6) else "weekday",
                            "lag7_quality":qmap.get(d-pd.Timedelta(days=7),"n/a"),
                            "lag14_quality":qmap.get(d-pd.Timedelta(days=14),"n/a")})
        # compute previous normal run
        prev_n = [dd for dd in normal_dates if dd < d]
        run = 0
        for dd in reversed(prev_n):
            if (d-dd).days == run+1: run += 1
            else: break
        ca_daily_rows[-1]["previous_normal_run"] = run
    ca_daily = pd.DataFrame(ca_daily_rows)
    ca_daily.to_csv(out_dir / "candidate_activity_daily.csv", index=False, encoding="utf-8-sig")
    print("\nDaily C stats:")
    print(ca_daily[["date","quality","U_d","C_total","active_hours_per_vessel","lag7_quality","previous_normal_run"]].to_string(index=False))

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

    # ---- Oracle decomposition ----
    print("\n=== C Oracle decomposition ===")
    oracle_rows = []
    for bname, (ts, te, vs_, ve) in blocks.items():
        vd = list(pd.date_range(vs_, ve, freq="D").normalize())
        bl = block_bl[bname]; bl_s = block_bl_sse[bname]
        train_n = [d for d in all_dates if d <= te and qmap.get(d) == "normal"]
        # training means
        C_profile_long = np.mean([C_all[d] / C_all[d].sum() if C_all[d].sum() > 0 else np.ones(24)/24 for d in train_n], axis=0)
        C_profile_long /= C_profile_long.sum()
        C_total_long = np.mean([C_all[d].sum() for d in train_n])
        M_long = np.mean([MQ_all[d]["M"] for d in train_n], axis=0)
        Q_long = np.mean([MQ_all[d]["Q"] for d in train_n], axis=0)
        O_long = np.mean([C_all[d].sum() / vc_map.get(d, 1) for d in train_n])

        for oracle_name in ["oracle_true_C_total", "oracle_true_C_profile", "oracle_true_C_both",
                            "oracle_true_O_per_vessel"]:
            total_sse = 0
            for d in vd:
                C_true = C_all[d]; bl_d = bl[d]; a_d = get_a_vec(d)
                TC_true = float(C_true.sum())
                P_true = C_true / TC_true if TC_true > 0 else C_profile_long
                Ud = vc_map.get(d, 50)

                if oracle_name == "oracle_true_C_total":
                    C_pred = TC_true * C_profile_long
                elif oracle_name == "oracle_true_C_profile":
                    C_pred = C_total_long * P_true
                elif oracle_name == "oracle_true_C_both":
                    C_pred = C_true
                elif oracle_name == "oracle_true_O_per_vessel":
                    O_true = TC_true / Ud if Ud > 0 else O_long
                    C_pred = Ud * O_true * C_profile_long
                else:
                    C_pred = C_total_long * C_profile_long

                # G = C * M * Q
                G_pred = C_pred * M_long * Q_long
                # distribute to regions by baseline proportion
                pred = bl_d.copy()
                for h in range(24):
                    bl_h = bl_d[h]; s = bl_h.sum()
                    if s > 0 and G_pred[h] > 0:
                        pred[h] = np.clip(np.rint(G_pred[h] * bl_h / s), 0, None)
                    else:
                        pred[h] = bl_h
                total_sse += float(((pred - a_d)**2).sum())
            red = 1 - total_sse / bl_s if bl_s else 0
            oracle_rows.append({"block":bname, "oracle":oracle_name, "sse":total_sse, "baseline_sse":bl_s, "reduction":red})
            print(f"  {bname:30s} {oracle_name:35s} SSE={total_sse:.0f} red={red*100:.1f}%")

    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(out_dir / "C_oracle_decomposition_scores.csv", index=False, encoding="utf-8-sig")

    # ---- deployable C total forecasts ----
    print("\n=== Deployable C total forecasts ===")
    ct_scores = []
    for bname, (ts, te, vs_, ve) in blocks.items():
        vd = list(pd.date_range(vs_, ve, freq="D").normalize())
        bl = block_bl[bname]; bl_s = block_bl_sse[bname]
        train_n = [d for d in all_dates if d <= te and qmap.get(d) == "normal"]
        C_totals_train = [float(C_all[d].sum()) for d in train_n]
        M_long = np.mean([MQ_all[d]["M"] for d in train_n], axis=0)
        Q_long = np.mean([MQ_all[d]["Q"] for d in train_n], axis=0)
        C_profile_long = np.mean([C_all[d] / C_all[d].sum() if C_all[d].sum() > 0 else np.ones(24)/24 for d in train_n], axis=0)
        C_profile_long /= C_profile_long.sum()

        for method in ["Ctotal_long_mean", "Ctotal_recent3", "Ctotal_recent6",
                       "Ctotal_lag7_normal", "Ctotal_lag14_normal",
                       "occupancy_long", "occupancy_recent6", "occupancy_lag7_normal"]:
            total_sse = 0
            for d in vd:
                bl_d = bl[d]; a_d = get_a_vec(d); Ud = vc_map.get(d, 50)
                # predict C total
                if method == "Ctotal_long_mean":
                    TC_pred = float(np.mean(C_totals_train))
                elif method == "Ctotal_recent3":
                    TC_pred = float(np.mean(C_totals_train[-3:])) if len(C_totals_train) >= 3 else float(np.mean(C_totals_train))
                elif method == "Ctotal_recent6":
                    TC_pred = float(np.mean(C_totals_train[-6:])) if len(C_totals_train) >= 6 else float(np.mean(C_totals_train))
                elif method == "Ctotal_lag7_normal":
                    lag7 = d - pd.Timedelta(days=7)
                    TC_pred = float(C_all[lag7].sum()) if lag7 in C_all and qmap.get(lag7) == "normal" else float(np.mean(C_totals_train))
                elif method == "Ctotal_lag14_normal":
                    lag14 = d - pd.Timedelta(days=14)
                    TC_pred = float(C_all[lag14].sum()) if lag14 in C_all and qmap.get(lag14) == "normal" else float(np.mean(C_totals_train))
                elif method == "occupancy_long":
                    O_long = np.mean([C_all[dd].sum() / vc_map.get(dd, 1) for dd in train_n])
                    TC_pred = Ud * O_long
                elif method == "occupancy_recent6":
                    rec6 = train_n[-6:] if len(train_n) >= 6 else train_n
                    O_rec6 = np.mean([C_all[dd].sum() / vc_map.get(dd, 1) for dd in rec6])
                    TC_pred = Ud * O_rec6
                elif method == "occupancy_lag7_normal":
                    lag7 = d - pd.Timedelta(days=7)
                    if lag7 in C_all and qmap.get(lag7) == "normal":
                        O_lag7 = C_all[lag7].sum() / vc_map.get(lag7, 1)
                        TC_pred = Ud * O_lag7
                    else:
                        O_long = np.mean([C_all[dd].sum() / vc_map.get(dd, 1) for dd in train_n])
                        TC_pred = Ud * O_long
                else:
                    TC_pred = float(np.mean(C_totals_train))

                # C hour = total * profile
                C_pred = TC_pred * C_profile_long
                # G = C * M * Q
                G_pred = C_pred * M_long * Q_long
                pred = bl_d.copy()
                for h in range(24):
                    bl_h = bl_d[h]; s = bl_h.sum()
                    if s > 0 and G_pred[h] > 0:
                        pred[h] = np.clip(np.rint(G_pred[h] * bl_h / s), 0, None)
                    else:
                        pred[h] = bl_h
                total_sse += float(((pred - a_d)**2).sum())
            red = 1 - total_sse / bl_s if bl_s else 0
            ct_scores.append({"block":bname, "method":method, "sse":total_sse, "baseline_sse":bl_s, "reduction":red})
            if abs(red) > 0.005:
                print(f"  {bname:30s} {method:30s} SSE={total_sse:.0f} red={red*100:.1f}%")

    ct_df = pd.DataFrame(ct_scores)
    ct_df.to_csv(out_dir / "C_total_forecast_scores.csv", index=False, encoding="utf-8-sig")

    # ---- baseline row ----
    for bname in blocks:
        ct_df = pd.concat([ct_df, pd.DataFrame([{"block":bname, "method":"exact_step11_baseline",
            "sse":block_bl_sse[bname], "baseline_sse":block_bl_sse[bname], "reduction":0.0}])], ignore_index=True)

    # ---- information condition: lag7 normal available ----
    print("\n=== Information condition: lag7 normal ===")
    ic_rows = []
    for bname, (ts, te, vs_, ve) in blocks.items():
        vd = list(pd.date_range(vs_, ve, freq="D").normalize())
        lag7_avail = [d for d in vd if qmap.get(d - pd.Timedelta(days=7)) == "normal"]
        lag7_or_lag14 = [d for d in vd if qmap.get(d - pd.Timedelta(days=7)) == "normal" or qmap.get(d - pd.Timedelta(days=14)) == "normal"]
        for subset_name, subset_dates in [("all_target", vd), ("lag7_normal", lag7_avail), ("lag7_or_lag14_normal", lag7_or_lag14)]:
            if not subset_dates:
                ic_rows.append({"block":bname, "subset":subset_name, "n_dates":0, "baseline_sse":0,
                               "method_sse":0, "reduction":0, "evidence_limited":True})
                continue
            bl = block_bl[bname]; train_n = [d for d in all_dates if d <= te and qmap.get(d) == "normal"]
            M_long = np.mean([MQ_all[d]["M"] for d in train_n], axis=0)
            Q_long = np.mean([MQ_all[d]["Q"] for d in train_n], axis=0)
            C_profile_long = np.mean([C_all[d] / C_all[d].sum() if C_all[d].sum() > 0 else np.ones(24)/24 for d in train_n], axis=0)
            C_profile_long /= C_profile_long.sum()
            O_long = np.mean([C_all[dd].sum() / vc_map.get(dd, 1) for dd in train_n])

            bl_sse_sub = 0; method_sse_sub = 0
            for d in subset_dates:
                bl_d = bl[d]; a_d = get_a_vec(d); Ud = vc_map.get(d, 50)
                bl_sse_sub += float(((bl_d - a_d)**2).sum())
                lag7 = d - pd.Timedelta(days=7)
                if qmap.get(lag7) == "normal" and lag7 in C_all:
                    O_pred = C_all[lag7].sum() / vc_map.get(lag7, 1)
                else:
                    O_pred = O_long
                TC_pred = Ud * O_pred
                C_pred = TC_pred * C_profile_long
                G_pred = C_pred * M_long * Q_long
                pred = bl_d.copy()
                for h in range(24):
                    bl_h = bl_d[h]; s = bl_h.sum()
                    if s > 0 and G_pred[h] > 0:
                        pred[h] = np.clip(np.rint(G_pred[h] * bl_h / s), 0, None)
                    else:
                        pred[h] = bl_h
                method_sse_sub += float(((pred - a_d)**2).sum())
            red = 1 - method_sse_sub / bl_sse_sub if bl_sse_sub else 0
            ev_lim = len(subset_dates) < 5
            ic_rows.append({"block":bname, "subset":subset_name, "n_dates":len(subset_dates),
                           "baseline_sse":bl_sse_sub, "method_sse":method_sse_sub,
                           "reduction":red, "evidence_limited":ev_lim})
            print(f"  {bname:30s} {subset_name:25s} n={len(subset_dates)} red={red*100:.1f}% {'(limited)' if ev_lim else ''}")

    ic_df = pd.DataFrame(ic_rows)
    ic_df.to_csv(out_dir / "information_condition_scores.csv", index=False, encoding="utf-8-sig")

    # ---- summary ----
    L = ["# Step 36 Candidate Activity Forecast", "", f"Run command: `{RUN_CMD}`", ""]
    L.append(f"## C_daily vs U_d correlation: {ca_daily['C_total'].corr(ca_daily['U_d']):.3f}")
    L.append(f"## O_d (active hours per vessel) CV: {ca_daily['active_hours_per_vessel'].std()/ca_daily['active_hours_per_vessel'].mean():.3f}")
    L.append("")
    L.append("## C Oracle decomposition (target_like)")
    L.append("")
    L.append("| Oracle | SSE | reduction |")
    L.append("| --- | --- | --- |")
    for _, r in oracle_df[oracle_df["block"]=="block_target_like"].iterrows():
        L.append(f"| {r['oracle']} | {r['sse']:.0f} | {r['reduction']*100:.1f}% |")
    L.append("")
    L.append("## Deployable C methods (target_like)")
    L.append("")
    for _, r in ct_df[ct_df["block"]=="block_target_like"].iterrows():
        L.append(f"- {r['method']}: {r['reduction']*100:.1f}%")
    L.append("")
    L.append("## Lag7-normal information condition")
    L.append("")
    for _, r in ic_df[ic_df["subset"]=="lag7_normal"].iterrows():
        L.append(f"- {r['block']}: n={int(r['n_dates'])}, red={r['reduction']*100:.1f}% {'(limited)' if r['evidence_limited'] else ''}")
    L.append("")
    tl_oracle = oracle_df[oracle_df["block"]=="block_target_like"]
    any_oracle_10 = any(tl_oracle["reduction"] >= 0.10)
    tl_dep = ct_df[(ct_df["block"]=="block_target_like") & (ct_df["method"]!="exact_step11_baseline")]
    any_dep_5 = any(tl_dep["reduction"] >= 0.05)
    L.append(f"## Decision")
    L.append(f"- Oracle >= 10%: {'YES' if any_oracle_10 else 'NO'}")
    L.append(f"- Deployable >= 5%: {'YES' if any_dep_5 else 'NO'}")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    assert not any(out_dir.rglob("*.zip"))
    print(f"\n=== Summary ===")
    print(f"C_daily vs U_d corr: {ca_daily['C_total'].corr(ca_daily['U_d']):.3f}")
    print(f"O_d CV: {ca_daily['active_hours_per_vessel'].std()/ca_daily['active_hours_per_vessel'].mean():.3f}")
    print("\nOracle decomposition (target_like):")
    for _, r in oracle_df[oracle_df["block"]=="block_target_like"].iterrows():
        print(f"  {r['oracle']:35s} SSE={r['sse']:.0f} red={r['reduction']*100:.1f}%")
    print("\nDeployable (target_like):")
    for _, r in ct_df[(ct_df["block"]=="block_target_like") & (abs(ct_df["reduction"]) > 0.005)].iterrows():
        print(f"  {r['method']:30s} SSE={r['sse']:.0f} red={r['reduction']*100:.1f}%")
    print(f"\nOracle >= 10%: {any_oracle_10}")
    print(f"Deployable >= 5%: {any_dep_5}")
    print("\nDone.")


if __name__ == "__main__":
    main()
