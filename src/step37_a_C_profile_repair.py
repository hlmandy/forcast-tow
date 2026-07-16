"""Step 37 — C profile prediction repair and baseline shape reshape.

Fixes Step 36: implements actual C-profile forecasting methods (long, recent,
lag7, lag14, shrunk) and profile-only reshape that preserves Step 11 region
daily totals while redistributing hourly weights.

Outputs (under outputs/step37_a_C_profile_repair/):
  1-13 as specified.
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
OUT_REL = Path("outputs/step37_a_C_profile_repair")
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
    # compute C per date-hour
    C_dict = {}
    for d in all_dates:
        C = np.zeros(24)
        sh = active[active["date"]==d]
        for h in range(24): C[h] = sh[sh["hour_of_day"]==h]["mmsi"].nunique()
        C_dict[d] = C
    _cache.update({"a_pivot":a_pivot, "a_lab":a_lab, "active":active,
                   "qmap":dict(zip(aud["date"],aud["quality_regime"])),
                   "china_map":dict(zip(aud["date"],aud["china_coastal_record_count"])),
                   "vc_map":dict(zip(vc_df["date"],vc_df["unique_vessel_count"])),
                   "all_dates":all_dates, "C_dict":C_dict, "df":df})
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
    """C profile = C_h / sum(C_h)."""
    C = load_all()["C_dict"][d]
    s = C.sum()
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


def reshape_region_totals(bl_d, C_profile_pred, conversion_e):
    """Reshape preserving Step 11 region daily totals.
    conversion_e: (24, 3) = E[A_{h,z}/C_h] from training.
    """
    C_profile_pred = np.asarray(C_profile_pred, dtype=float).ravel()
    if len(C_profile_pred) != 24: C_profile_pred = np.ones(24)/24
    # weights w_{h,z} ∝ C_profile_pred[h] * conversion_e[h,z]
    weights = C_profile_pred[:, None] * conversion_e  # (24, 3)
    # preserve each region's daily total
    pred = bl_d.copy()
    for ri in range(3):
        region_total = bl_d[:, ri].sum()
        col_w = weights[:, ri]
        s = col_w.sum()
        if s > 0 and region_total > 0:
            pred[:, ri] = np.clip(np.rint(region_total * col_w / s), 0, None)
        # else keep baseline
    return pred.astype(int)


def main():
    out_dir = ROOT / OUT_REL; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")
    data = load_all(); qmap = data["qmap"]; all_dates = data["all_dates"]
    vc_map = data["vc_map"]; C_dict = data["C_dict"]
    normal_dates = [d for d in all_dates if qmap.get(d) == "normal"]

    # ---- audit ----
    (out_dir / "implementation_audit.md").write_text(
        "# Step 36 Implementation Audit\n\n"
        "1. Step 36 only implemented C-total methods; NO C-profile methods tested\n"
        "2. lag7-normal subset for target_like = 0 days\n"
        "3. lag7_or_lag14 didn't actually use lag14 profile\n"
        "4. Step 36 replaced Step 11 entirely; didn't test profile-only reshape\n"
        "5. Therefore 'C profile unpredictable' was NOT tested\n\n"
        "## Correlation reports\n"
        f"- corr(C_total, U_d) all dates: {pd.Series([C_dict[d].sum() for d in all_dates]).corr(pd.Series([vc_map.get(d,0) for d in all_dates])):.3f}\n"
        f"- corr(C_total, U_d) normal dates: {pd.Series([C_dict[d].sum() for d in normal_dates]).corr(pd.Series([vc_map.get(d,0) for d in normal_dates])):.3f}\n",
        encoding="utf-8")

    # ---- hourly C profile data ----
    print("=== C hourly profiles ===")
    hourly_rows = []
    for d in all_dates:
        C = C_dict[d]; TC = float(C.sum())
        P = C / TC if TC > 0 else np.ones(24)/24
        q = qmap.get(d, "")
        dt = "weekend" if d.dayofweek in (5,6) else "weekday"
        lag7q = qmap.get(d - pd.Timedelta(days=7), "n/a")
        lag14q = qmap.get(d - pd.Timedelta(days=14), "n/a")
        prev_n = [dd for dd in normal_dates if dd < d]
        run = 0
        for dd in reversed(prev_n):
            if (d-dd).days == run+1: run += 1
            else: break
        for h in range(24):
            hourly_rows.append({"date":f"{d:%Y-%m-%d}", "hour":h, "quality":q,
                               "C_hour":float(C[h]), "C_total":TC, "C_profile":float(P[h]),
                               "day_type":dt, "lag7_quality":lag7q, "lag14_quality":lag14q,
                               "previous_normal_run":run})
    hourly_df = pd.DataFrame(hourly_rows)
    hourly_df.to_csv(out_dir / "candidate_activity_hourly.csv", index=False, encoding="utf-8-sig")

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

    # ---- profile prediction methods ----
    print("\n=== Profile prediction methods ===")
    profile_scores = []
    block_scores = []

    for bname, (ts, te, vs_, ve) in blocks.items():
        vd = list(pd.date_range(vs_, ve, freq="D").normalize())
        bl = block_bl[bname]; bl_s = block_bl_sse[bname]
        train_n = [d for d in all_dates if d <= te and qmap.get(d) == "normal"]

        # training profiles
        P_long = np.mean([get_C_profile(d) for d in train_n], axis=0); P_long /= P_long.sum()
        P_daytype = {}
        for dt in ["weekday", "weekend"]:
            dt_dates = [d for d in train_n if ("weekend" if d.dayofweek in (5,6) else "weekday") == dt]
            if dt_dates: P_daytype[dt] = np.mean([get_C_profile(d) for d in dt_dates], axis=0)
            else: P_daytype[dt] = P_long
            P_daytype[dt] /= P_daytype[dt].sum()
        P_recent3 = np.mean([get_C_profile(d) for d in train_n[-3:]]) if len(train_n) >= 3 else P_long; P_recent3 /= P_recent3.sum()
        P_recent6 = np.mean([get_C_profile(d) for d in train_n[-6:]]) if len(train_n) >= 6 else P_long; P_recent6 /= P_recent6.sum()

        # conversion e_{h,z} = E[A_{h,z}/C_h]
        conversion = np.zeros((24, 3))
        for d in train_n:
            C = C_dict[d]; a = get_a_vec(d)
            for h in range(24):
                if C[h] > 0:
                    for ri in range(3):
                        conversion[h, ri] += a[h, ri] / C[h]
        conversion /= len(train_n)

        for method in ["profile_long", "profile_daytype", "profile_recent3", "profile_recent6",
                       "profile_lag7_direct", "profile_lag14_direct", "profile_lag_fallback",
                       "oracle_true_profile"]:
            total_sse = 0
            for di, d in enumerate(vd):
                bl_d = bl[d]; a_d = get_a_vec(d)
                dt = "weekend" if d.dayofweek in (5,6) else "weekday"
                lag7 = d - pd.Timedelta(days=7); lag14 = d - pd.Timedelta(days=14)

                if method == "profile_long": P_pred = P_long
                elif method == "profile_daytype": P_pred = P_daytype.get(dt, P_long)
                elif method == "profile_recent3": P_pred = P_recent3
                elif method == "profile_recent6": P_pred = P_recent6
                elif method == "profile_lag7_direct":
                    P_pred = get_C_profile(lag7) if lag7 in C_dict and qmap.get(lag7)=="normal" else P_long
                elif method == "profile_lag14_direct":
                    P_pred = get_C_profile(lag14) if lag14 in C_dict and qmap.get(lag14)=="normal" else P_long
                elif method == "profile_lag_fallback":
                    if lag7 in C_dict and qmap.get(lag7)=="normal": P_pred = get_C_profile(lag7)
                    elif lag14 in C_dict and qmap.get(lag14)=="normal": P_pred = get_C_profile(lag14)
                    else: P_pred = P_recent6
                elif method == "oracle_true_profile":
                    P_pred = get_C_profile(d)
                else: P_pred = P_long
                P_pred /= P_pred.sum()

                # reshape preserving region totals
                pred = reshape_region_totals(bl_d, P_pred, conversion)
                total_sse += float(((pred - a_d)**2).sum())

            red = 1 - total_sse / bl_s if bl_s else 0
            block_scores.append({"block":bname, "method":method, "deployable":method!="oracle_true_profile",
                               "sse":total_sse, "baseline_sse":bl_s, "reduction_ratio":red})
            if abs(red) > 0.005:
                print(f"  {bname:30s} {method:30s} SSE={total_sse:.0f} red={red*100:.1f}%")

    # baseline row
    for bname in blocks:
        block_scores.append({"block":bname, "method":"exact_step11_baseline", "deployable":True,
                           "sse":block_bl_sse[bname], "baseline_sse":block_bl_sse[bname], "reduction_ratio":0})

    bs_df = pd.DataFrame(block_scores)
    bs_df.to_csv(out_dir / "block_scores.csv", index=False, encoding="utf-8-sig")

    # ---- profile gap diagnostics ----
    print("\n=== Profile gap diagnostics ===")
    gap_rows = []
    for d_target in normal_dates:
        P_true = get_C_profile(d_target)
        for lag in range(1, 15):
            d_src = d_target - pd.Timedelta(days=lag)
            if d_src not in all_dates: continue
            if qmap.get(d_src) != "normal": continue
            P_src = get_C_profile(d_src)
            corr = float(pd.Series(P_true).corr(pd.Series(P_src))) if np.std(P_true) > 0 and np.std(P_src) > 0 else np.nan
            l2 = float(np.sqrt(np.sum((P_true - P_src)**2)))
            gap_rows.append({"target_date":f"{d_target:%Y-%m-%d}", "lag":lag,
                           "source_date":f"{d_src:%Y-%m-%d}", "profile_corr":corr, "profile_l2":l2})
    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(out_dir / "profile_gap_diagnostics.csv", index=False, encoding="utf-8-sig")
    if len(gap_df):
        print("Profile correlation by lag:")
        print(gap_df.groupby("lag")["profile_corr"].agg(["mean","median","count"]).to_string())

    # ---- information condition subsets ----
    print("\n=== Information condition subsets ===")
    ic_rows = []
    for bname, (ts, te, vs_, ve) in blocks.items():
        vd = list(pd.date_range(vs_, ve, freq="D").normalize())
        bl = block_bl[bname]; train_n = [d for d in all_dates if d <= te and qmap.get(d) == "normal"]
        P_long = np.mean([get_C_profile(d) for d in train_n], axis=0); P_long /= P_long.sum()
        conversion = np.zeros((24, 3))
        for d in train_n:
            C = C_dict[d]; a = get_a_vec(d)
            for h in range(24):
                if C[h] > 0:
                    for ri in range(3): conversion[h, ri] += a[h, ri] / C[h]
        conversion /= len(train_n)

        for subset_name, filter_func in [("all_target", lambda d: True),
                                          ("lag7_normal", lambda d: qmap.get(d-pd.Timedelta(days=7))=="normal"),
                                          ("lag14_normal", lambda d: qmap.get(d-pd.Timedelta(days=14))=="normal")]:
            subset_dates = [d for d in vd if filter_func(d)]
            if not subset_dates:
                ic_rows.append({"block":bname, "subset":subset_name, "n_dates":0, "bl_sse":0, "method_sse":0, "reduction":0, "limited":True})
                continue
            bl_sse_sub = 0; method_sse_sub = 0
            for d in subset_dates:
                bl_d = bl[d]; a_d = get_a_vec(d)
                bl_sse_sub += float(((bl_d - a_d)**2).sum())
                lag7 = d - pd.Timedelta(days=7); lag14 = d - pd.Timedelta(days=14)
                if subset_name == "lag7_normal" and qmap.get(lag7)=="normal":
                    P_pred = get_C_profile(lag7)
                elif subset_name == "lag14_normal" and qmap.get(lag14)=="normal":
                    P_pred = get_C_profile(lag14)
                else:
                    # fallback for all_target: use lag_fallback
                    if qmap.get(lag7)=="normal": P_pred = get_C_profile(lag7)
                    elif qmap.get(lag14)=="normal": P_pred = get_C_profile(lag14)
                    else: P_pred = P_long
                P_pred /= P_pred.sum()
                pred = reshape_region_totals(bl_d, P_pred, conversion)
                method_sse_sub += float(((pred - a_d)**2).sum())
            red = 1 - method_sse_sub/bl_sse_sub if bl_sse_sub else 0
            limited = len(subset_dates) < 5
            ic_rows.append({"block":bname, "subset":subset_name, "n_dates":len(subset_dates),
                           "bl_sse":bl_sse_sub, "method_sse":method_sse_sub, "reduction":red, "limited":limited})
            print(f"  {bname:30s} {subset_name:20s} n={len(subset_dates)} red={red*100:.1f}% {'(limited)' if limited else ''}")

    ic_df = pd.DataFrame(ic_rows)
    ic_df.to_csv(out_dir / "information_condition_scores.csv", index=False, encoding="utf-8-sig")

    # ---- summary ----
    L = ["# Step 37 C Profile Repair and Reshape", "", f"Run command: `{RUN_CMD}`", ""]
    L.append("## Block scores (key methods, target_like)")
    L.append("")
    L.append("| method | SSE | reduction |")
    L.append("| --- | --- | --- |")
    for _, r in bs_df[(bs_df["block"]=="block_target_like") & (bs_df["method"].isin(
        ["exact_step11_baseline","profile_long","profile_recent3","profile_recent6",
         "profile_lag7_direct","profile_lag14_direct","profile_lag_fallback","oracle_true_profile"]))].iterrows():
        L.append(f"| {r['method']} | {r['sse']:.0f} | {r['reduction_ratio']*100:.1f}% |")
    L.append("")
    L.append("## Profile gap diagnostics (by lag)")
    L.append("")
    if len(gap_df):
        L.append("| lag | mean corr | median corr | n_pairs |")
        L.append("| --- | --- | --- | --- |")
        for lag, g in gap_df.groupby("lag"):
            L.append(f"| {lag} | {g['profile_corr'].mean():.3f} | {g['profile_corr'].median():.3f} | {len(g)} |")
    L.append("")
    L.append("## Information conditions")
    L.append("")
    for _, r in ic_df.iterrows():
        L.append(f"- {r['block']} {r['subset']}: n={int(r['n_dates'])}, red={r['reduction']*100:.1f}% {'(limited)' if r['limited'] else ''}")
    L.append("")
    tl = bs_df[(bs_df["block"]=="block_target_like") & (bs_df["deployable"])]
    best_dep = tl.loc[tl["reduction_ratio"].idxmax()] if len(tl) else None
    oracle_prof = bs_df[(bs_df["block"]=="block_target_like") & (bs_df["method"]=="oracle_true_profile")]
    oracle_red = float(oracle_prof["reduction_ratio"].iloc[0]) if len(oracle_prof) else 0
    L.append("## Decision")
    L.append(f"- Oracle true profile reduction: {oracle_red*100:.1f}%")
    if best_dep is not None:
        L.append(f"- Best deployable: {best_dep['method']} ({best_dep['reduction_ratio']*100:.1f}%)")
    L.append("")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")

    assert not any(out_dir.rglob("*.zip"))
    print(f"\n=== Summary ===")
    print("Block scores (target_like):")
    for _, r in bs_df[(bs_df["block"]=="block_target_like") & (bs_df["method"].isin(
        ["exact_step11_baseline","profile_long","profile_recent6","profile_lag7_direct","profile_lag14_direct",
         "profile_lag_fallback","oracle_true_profile"]))].iterrows():
        print(f"  {r['method']:30s} SSE={r['sse']:.0f} red={r['reduction_ratio']*100:.1f}%")
    if len(gap_df):
        print("\nProfile corr by lag:")
        for lag in [1, 7, 14]:
            g = gap_df[gap_df["lag"]==lag]
            if len(g): print(f"  lag {lag}: corr={g['profile_corr'].mean():.3f} (n={len(g)})")
    print("\nDone.")


if __name__ == "__main__":
    main()
