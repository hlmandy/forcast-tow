"""Step 11 — Generate final validation-set candidates (2018-01-25..01-31).

Freeze 3 A components and 3 B components, reproduce each against the Step 07 /
Step 10 final_analog predictions (<1e-10), then retrain on the full training
period (2018-01-01..01-24) and predict the validation period. Package 3
candidates into official-format CSVs (UTF-8 BOM, no ZIP). No new models, no
ensembling, no validation AIS / labels / vessel-count scaling.

Outputs (under outputs/step11_final_validation_candidates/):
  1. candidate_manifest.csv
  2. a_component_predictions.csv     (1512)
  3. b_component_predictions.csv     (3024)
  4. final_b_hyperparameter_selection.csv (51)
  5. reproduction_checks.csv         (6)
  6. candidate_daily_totals.csv      (42)
  7. candidate_pairwise_comparison.csv (3)
  8. submission_checks.csv           (6)
  9. summary.md
  + submissions/candidate_{01_main,02_a_sensitivity,03_b_robustness}/ (2 CSVs each)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimized_baseline import (  # noqa: E402
    CN_TO_REGION, REGION_CN, REGION_ORDER, add_regions, make_a_labels,
)
from step03_a_rolling_backtest import (  # noqa: E402
    POLICY_WEIGHTS, REGION_RANK, classify_fold_quality, predict_weighted, scale_train_mean,
)
from step06_a_structure_model_benchmark import compute_components  # noqa: E402
from step10_b_shrinkage_calibration import (  # noqa: E402
    SCALES, SMOOTHERS, TAUS, candidates_for, cell_groups_of, compute_aggregates, compute_bases,
    inner_folds, round_pred, score_candidate, smooth_circular,
)

TRAIN_REL = Path("数据备份/训练集_20180101-0124_拖轮AIS.csv")
VAL_VESSELS = Path("outputs/step01_data_audit/validation_daily_vessels.csv")
STEP07_PREDS = Path("outputs/step07_a_regularized_count_models/a_model_predictions.csv")
STEP10_PREDS = Path("outputs/step10_b_shrinkage_calibration/b_model_predictions.csv")
SS_PATH = Path("outputs/step08_b_state_transition_audit/b_hourly_source_state.csv")
PF_PATH = Path("outputs/step08_b_state_transition_audit/b_hourly_pair_flow.csv")
A_TEMPLATE = Path("outputs/_official_examples_backup/提交结果1_区域活跃拖轮数量.csv")
B_TEMPLATE = Path("outputs/_official_examples_backup/提交结果2_圈层间拖轮迁移量.csv")
STEP07_NORMAL = Path("outputs/step07_a_regularized_count_models/a_normal_target_scores.csv")
STEP07_FINAL = Path("outputs/step07_a_regularized_count_models/a_final_analog_scores.csv")
STEP10_DECISION = Path("outputs/step10_b_shrinkage_calibration/decision_table.csv")
OUT_REL = Path("outputs/step11_final_validation_candidates")

GLOBAL_START = pd.Timestamp("2018-01-01")
FULL_TRAIN_END = pd.Timestamp("2018-01-24")
FA_TRAIN_END = pd.Timestamp("2018-01-18")
VAL_START = pd.Timestamp("2018-01-25")
VAL_END = pd.Timestamp("2018-01-31")
MIG = {"core": ("near", "outer"), "near": ("core", "outer"), "outer": ("core", "near")}
DIRS = [("core", "near"), ("core", "outer"), ("near", "core"), ("near", "outer"), ("outer", "core"), ("outer", "near")]
DIR_NAMES = [f"{s}->{t}" for s, t in DIRS]
DIR_RANK = {d: i for i, d in enumerate(DIR_NAMES)}

A_COMPONENTS = [
    ("a_exclude_daytype", "exclude_outage_severe", "decomp_mean_daytype_shrunk"),
    ("a_quality_hour", "quality_weighted", "hour_mean"),
    ("a_quality_mean_profile", "quality_weighted", "decomp_mean_mean"),
]
B_COMPONENTS = [
    ("b_hier_scale", "pair_hour_hier_tau_scale_cv"),
    ("b_circular", "pair_hour_circular_cv"),
    ("b_fixed085", "pair_hour_fixed085"),
]
CANDIDATES = [
    ("candidate_01_main", "main", "a_exclude_daytype", "b_hier_scale", "final_analog-oriented main candidate"),
    ("candidate_02_a_sensitivity", "a_sensitivity", "a_quality_hour", "b_hier_scale", "only A changes (vs 01); isolates A training/structure effect"),
    ("candidate_03_b_robustness", "b_robustness", "a_exclude_daytype", "b_circular", "only B changes (vs 01); rolling-fold robust B"),
]
A_PRED_COLS = ["component_id", "training_policy", "method", "date", "hour", "hour_of_day", "dayofweek", "day_type",
               "region", "validation_unique_vessel_count", "pred_daily_total", "pred_profile_share", "pred_float", "pred_rounded"]
B_PRED_COLS = ["component_id", "method", "rounding_method", "selected_tau", "selected_smoother", "selected_scale",
               "date", "hour", "hour_of_day", "dayofweek", "day_type", "source_region", "target_region", "task_key",
               "validation_unique_vessel_count", "pred_float", "pred_rounded"]

RUN_CMD = "python " + " ".join(sys.argv)


def _select_idx_local(method, scored):
    def tie_key(i):
        cand, sse = scored[i][0], scored[i][1]
        _typ, tau, sm, scale = cand
        scale_dist = abs(scale - 1.0) if scale is not None else 0.0
        if method == "pair_hour_hier_tau_cv":
            return (sse, -tau)
        if method == "pair_hour_hier_tau_scale_cv":
            return (sse, scale_dist, -tau)
        if method == "pair_hour_circular_cv":
            return (sse, ["raw", "triangular3", "triangular5"].index(sm))
        if method == "pair_hour_circular_scale_cv":
            return (sse, scale_dist, ["raw", "triangular3", "triangular5"].index(sm))
        return (sse,)
    return min(range(len(scored)), key=tie_key)


def load_ais(train_path):
    usecols = ["mmsi", "x", "y", "sog", "time"]
    dtypes = {"mmsi": "string", "x": "float64", "y": "float64", "sog": "float32"}
    df = pd.read_csv(train_path, usecols=usecols, dtype=dtypes, parse_dates=["time"])
    df["hour"] = df["time"].dt.floor("h")
    df["date"] = df["time"].dt.normalize()
    df = add_regions(df)
    return df


def build_a_frame(df, daily, audit):
    a = make_a_labels(df)
    a["date"] = a["hour"].dt.normalize()
    a["hour_of_day"] = a["hour"].dt.hour
    a["dayofweek"] = a["hour"].dt.dayofweek
    a = a[["hour", "date", "hour_of_day", "dayofweek", "region", "y"]]
    a = a.merge(daily[["date", "unique_vessel_count"]].rename(columns={"unique_vessel_count": "vessel_count"}), on="date", how="left")
    a = a.merge(audit[["date", "quality_regime"]].rename(columns={"quality_regime": "audit_quality_regime"}), on="date", how="left")
    a["day_type"] = np.where(a["dayofweek"].isin([5, 6]), "weekend", "weekday")
    a["region_rank"] = a["region"].map(REGION_RANK)
    return a


def build_a_valid_grid(val_vc):
    rows = []
    for d in pd.date_range(VAL_START, VAL_END, freq="D").normalize():
        for h in range(24):
            hr = d + pd.Timedelta(hours=h)
            for r in REGION_ORDER:
                rows.append((d, hr, h, hr.dayofweek, "weekend" if hr.dayofweek in (5, 6) else "weekday", r))
    g = pd.DataFrame(rows, columns=["date", "hour", "hour_of_day", "dayofweek", "day_type", "region"])
    g["vessel_count"] = g["date"].map(val_vc)
    g["region_rank"] = g["region"].map(REGION_RANK)
    return g


def build_b_valid_grid():
    rows = []
    for d in pd.date_range(VAL_START, VAL_END, freq="D").normalize():
        for h in range(24):
            hr = d + pd.Timedelta(hours=h)
            for (s, t) in DIRS:
                rows.append((d, hr, h, hr.dayofweek, "weekend" if hr.dayofweek in (5, 6) else "weekday", s, t, f"{s}->{t}"))
    g = pd.DataFrame(rows, columns=["date", "hour", "hour_of_day", "dayofweek", "day_type", "source_region", "target_region", "task_key"])
    g["dir_order"] = g["task_key"].map(DIR_RANK)
    g["source_rank"] = g["source_region"].map(REGION_RANK)
    g["target_rank"] = g["target_region"].map(REGION_RANK)
    return g


def predict_a(comp_id, policy, method, a_frame, train_start, train_end, valid, china_series, vc_series):
    from optimized_baseline import MethodSpec
    train = a_frame[(a_frame["date"] >= train_start) & (a_frame["date"] <= train_end)].copy()
    train_dates = sorted(pd.date_range(train_start, train_end, freq="D").normalize())
    classes, _ = classify_fold_quality(train_dates, china_series)
    weights = {d: POLICY_WEIGHTS[policy][classes[d]] for d in train_dates}
    train_w = train.copy(); train_w["weight"] = train_w["date"].map(weights).astype(float)
    smean = scale_train_mean(train_dates, vc_series, weights, policy)
    if method == "hour_mean":
        pred = np.asarray(predict_weighted(train_w, valid, MethodSpec("hour_mean", "hour"), smean), dtype=float)
        return pred, np.full(len(valid), np.nan), np.full(len(valid), np.nan)
    T_hat, P_hat = compute_components(train, train_dates, train_end, weights)
    vr = valid["region"].to_numpy(); vh = valid["hour_of_day"].to_numpy().astype(int); vdt = valid["day_type"].to_numpy()
    Tvec = np.array([T_hat["mean"][r] for r in vr], dtype=float)
    if method == "decomp_mean_mean":
        prof = np.array([P_hat["mean"][r][h] for r, h in zip(vr, vh)])
    elif method == "decomp_mean_daytype_shrunk":
        prof = np.array([P_hat["daytype_shrunk"][r][dt][h] for r, dt, h in zip(vr, vdt, vh)])
    else:
        raise ValueError(method)
    pred = Tvec * prof
    return pred, Tvec, prof


def select_b(method, train_start, train_end, ss, pf):
    folds = inner_folds(train_start, train_end)
    inner_data = []
    inner_dates = []
    for (vdate, its, ite) in folds:
        iagg = compute_aggregates(ss, pf, its, ite)
        iv = pf[pf["date"] == vdate].sort_values(["hour", "source_region", "target_region"]).reset_index(drop=True)
        ib = compute_bases(iagg, iv)
        inner_data.append((ib, cell_groups_of(iv), iv["dir_order"].to_numpy(), iv["y"].to_numpy(float), iv["task_key"].to_numpy()))
        inner_dates.append(vdate)
    cands = candidates_for(method)
    scored = []
    for cand in cands:
        _typ, tau, sm, scale = cand
        sse_sum = flo_sum = mae_sum = bias_sum = 0.0; nb = 0
        for (ib, ig, iod, iy, itask) in inner_data:
            base = ib["hier"][tau] if method.startswith("pair_hour_hier") else ib["circ"][sm]
            sarr = np.full(len(base), scale if scale is not None else 1.0)
            sse, flo, mae, bias, _, _ = score_candidate(base, sarr, False, "independent", ib, ig, iod, iy, ib["pres_stock"])
            sse_sum += sse; flo_sum += flo; mae_sum += mae; bias_sum += bias; nb += 1
        scored.append((cand, sse_sum, flo_sum, mae_sum / nb, bias_sum / nb))
    sel_idx = _select_idx_local(method, scored)
    return scored[sel_idx][0], scored, inner_dates


def predict_b(method, train_start, train_end, valid, ss, pf):
    agg = compute_aggregates(ss, pf, train_start, train_end)
    bases = compute_bases(agg, valid)
    groups = cell_groups_of(valid); dir_order = valid["dir_order"].to_numpy()
    sel_tau = sel_sm = sel_scale = np.nan
    if method == "pair_hour_fixed085":
        pred = bases["hm"] * 0.85; sel_scale = 0.85
    else:
        sel_cand, _scored, _ = select_b(method, train_start, train_end, ss, pf)
        _typ, tau, sm, scale = sel_cand
        if method.startswith("pair_hour_hier"):
            base = bases["hier"][tau]; pred = base * scale; sel_tau = tau; sel_scale = scale
        else:
            base = bases["circ"][sm]; pred = base; sel_sm = sm
    pred = np.clip(pred, 0, None)
    rounded = round_pred(pred, "independent", groups, dir_order)
    return pred, rounded, sel_tau, sel_sm, sel_scale


def main():
    out_dir = ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir = out_dir / "submissions"
    sub_dir.mkdir(exist_ok=True)
    print(f"out_dir = {out_dir}")

    # --- load inputs ---
    df = load_ais(ROOT / TRAIN_REL)
    daily = pd.read_csv(ROOT / Path("outputs/step01_data_audit/train_daily_overview.csv"))
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    audit = pd.read_csv(ROOT / Path("outputs/step02_a_quality_effect/a_daily_quality_effect.csv"))
    audit["date"] = pd.to_datetime(audit["date"]).dt.normalize()
    china_series = dict(zip(audit["date"], audit["china_coastal_record_count"]))
    vc_series = dict(zip(daily["date"], daily["unique_vessel_count"]))
    a_frame = build_a_frame(df, daily, audit)
    assert len(a_frame) == 1728

    val_v = pd.read_csv(ROOT / VAL_VESSELS)
    val_v["date"] = pd.to_datetime(val_v["date"]).dt.normalize()
    assert set(val_v["date"]) == set(pd.date_range(VAL_START, VAL_END, freq="D").normalize())
    assert val_v["unique_vessel_count"].notna().all()
    val_vc = dict(zip(val_v["date"], val_v["unique_vessel_count"]))

    ss = pd.read_csv(ROOT / SS_PATH, encoding="utf-8-sig")
    ss["hour"] = pd.to_datetime(ss["hour"]); ss["date"] = pd.to_datetime(ss["date"]).dt.normalize(); ss["hour_of_day"] = ss["hour"].dt.hour
    pf = pd.read_csv(ROOT / PF_PATH, encoding="utf-8-sig")
    pf["hour"] = pd.to_datetime(pf["hour"]); pf["date"] = pd.to_datetime(pf["date"]).dt.normalize(); pf["hour_of_day"] = pf["hour"].dt.hour
    pf["dir_order"] = pf["task_key"].map(DIR_RANK)

    tmpl_a = pd.read_csv(ROOT / A_TEMPLATE, encoding="utf-8-sig")
    tmpl_b = pd.read_csv(ROOT / B_TEMPLATE, encoding="utf-8-sig")

    # --- reproduction references ---
    s7 = pd.read_csv(ROOT / STEP07_PREDS, encoding="utf-8-sig")
    s7["date"] = pd.to_datetime(s7["date"]).dt.strftime("%Y-%m-%d")
    s10 = pd.read_csv(ROOT / STEP10_PREDS, encoding="utf-8-sig")
    s10["date"] = pd.to_datetime(s10["date"]).dt.strftime("%Y-%m-%d")

    # --- A components: reproduction + final ---
    a_grid = build_a_valid_grid(val_vc)
    fa_a_valid = a_frame[(a_frame["date"] >= pd.Timestamp("2018-01-19")) & (a_frame["date"] <= pd.Timestamp("2018-01-24"))].copy()
    fa_a_valid = fa_a_valid.sort_values(["hour", "region_rank"]).reset_index(drop=True)
    a_frames = []
    repro_rows = []
    a_final_preds = {}
    for comp_id, policy, method in A_COMPONENTS:
        # reproduction on final_analog
        pred_fa, _, _ = predict_a(comp_id, policy, method, a_frame, GLOBAL_START, FA_TRAIN_END, fa_a_valid, china_series, vc_series)
        ref = s7[(s7["evaluation_id"] == "final_analog") & (s7["training_policy"] == policy) & (s7["method"] == method)].copy()
        ref = ref.sort_values(["date", "hour_of_day", "region"]).reset_index(drop=True)
        mine = pd.DataFrame({"date": fa_a_valid["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                             "hour_of_day": fa_a_valid["hour_of_day"].to_numpy(), "region": fa_a_valid["region"].to_numpy(), "pred": pred_fa})
        m = ref[["date", "hour_of_day", "region", "pred_float"]].merge(mine, on=["date", "hour_of_day", "region"], how="inner")
        max_diff = float((m["pred"] - m["pred_float"]).abs().max())
        exact = float((np.rint(m["pred"]) == np.rint(m["pred_float"])).mean())
        repro_rows.append({"task": "A", "component_id": comp_id, "reference_step": "step07", "reference_evaluation_id": "final_analog",
                           "reference_training_policy": policy, "reference_method": method, "reference_rounding_method": "independent",
                           "n_rows": int(len(m)), "max_abs_diff_float": max_diff, "rounded_exact_match_rate": exact,
                           "passed": bool(max_diff < 1e-10 and exact == 1.0)})
        assert max_diff < 1e-10 and exact == 1.0, f"A reproduction failed {comp_id}: diff={max_diff}"
        # final prediction
        pred_f, ptot, pshare = predict_a(comp_id, policy, method, a_frame, GLOBAL_START, FULL_TRAIN_END, a_grid, china_series, vc_series)
        assert np.isfinite(pred_f).all() and (pred_f >= 0).all()
        rounded = np.clip(np.rint(pred_f), 0, None).astype(np.int64)
        assert len(rounded) == 504
        assert (rounded <= a_grid["vessel_count"].to_numpy()).all(), f"A {comp_id}: cell pred > vessel_count"
        a_final_preds[comp_id] = (pred_f, rounded, ptot, pshare)
        af = pd.DataFrame({
            "component_id": comp_id, "training_policy": policy, "method": method,
            "date": a_grid["date"].dt.strftime("%Y-%m-%d").to_numpy(), "hour": a_grid["hour"].to_numpy(),
            "hour_of_day": a_grid["hour_of_day"].to_numpy(), "dayofweek": a_grid["dayofweek"].to_numpy(),
            "day_type": a_grid["day_type"].to_numpy(), "region": a_grid["region"].to_numpy(),
            "validation_unique_vessel_count": a_grid["vessel_count"].to_numpy(),
            "pred_daily_total": ptot, "pred_profile_share": pshare, "pred_float": pred_f, "pred_rounded": rounded,
        })
        a_frames.append(af)
    print("A components: reproduction + final prediction done.")

    # --- B components: reproduction + final ---
    b_grid = build_b_valid_grid()
    fa_b_valid = pf[(pf["date"] >= pd.Timestamp("2018-01-19")) & (pf["date"] <= pd.Timestamp("2018-01-24"))].copy()
    fa_b_valid = fa_b_valid.sort_values(["hour", "source_region", "target_region"]).reset_index(drop=True)
    b_frames = []
    b_hp_rows = []
    b_final_preds = {}
    b_final_params = {}
    for comp_id, method in B_COMPONENTS:
        # reproduction on final_analog
        pred_fa, _, _, _, _ = predict_b(method, GLOBAL_START, FA_TRAIN_END, fa_b_valid, ss, pf)
        ref = s10[(s10["evaluation_id"] == "final_analog") & (s10["predictor_method"] == method) & (s10["rounding_method"] == "independent")].copy()
        ref = ref.sort_values(["date", "hour_of_day", "source_region", "target_region"]).reset_index(drop=True)
        mine = pd.DataFrame({"date": fa_b_valid["date"].dt.strftime("%Y-%m-%d").to_numpy(), "hour_of_day": fa_b_valid["hour_of_day"].to_numpy(),
                             "source_region": fa_b_valid["source_region"].to_numpy(), "target_region": fa_b_valid["target_region"].to_numpy(), "pred": pred_fa})
        m = ref[["date", "hour_of_day", "source_region", "target_region", "pred_float"]].merge(mine, on=["date", "hour_of_day", "source_region", "target_region"], how="inner")
        max_diff = float((m["pred"] - m["pred_float"]).abs().max())
        exact = float((np.rint(m["pred"]) == np.rint(m["pred_float"])).mean())
        repro_rows.append({"task": "B", "component_id": comp_id, "reference_step": "step10", "reference_evaluation_id": "final_analog",
                           "reference_training_policy": "", "reference_method": method, "reference_rounding_method": "independent",
                           "n_rows": int(len(m)), "max_abs_diff_float": max_diff, "rounded_exact_match_rate": exact,
                           "passed": bool(max_diff < 1e-10 and exact == 1.0)})
        assert max_diff < 1e-10 and exact == 1.0, f"B reproduction failed {comp_id}: diff={max_diff}"
        # final prediction on full training
        if method != "pair_hour_fixed085":
            sel_cand, scored, inner_dates = select_b(method, GLOBAL_START, FULL_TRAIN_END, ss, pf)
            inner_dates_str = ",".join(f"{d:%Y-%m-%d}" for d in inner_dates)
            assert len(inner_dates) == 17, f"B inner folds {len(inner_dates)} != 17"
            for (cand, sse, flo, mae, bias) in scored:
                _typ, tau, sm, scale = cand
                b_hp_rows.append({"component_id": comp_id, "tau": tau if tau is not None else np.nan, "smoother": sm if sm else "",
                                  "scale": scale if scale is not None else np.nan, "n_inner_splits": 17,
                                  "inner_validation_start": inner_dates_str.split(",")[0], "inner_validation_end": inner_dates_str.split(",")[-1],
                                  "inner_sse_rounded": sse, "inner_sse_float": flo, "inner_mae_rounded": mae, "mean_bias_rounded": bias,
                                  "selected": cand == sel_cand})
        pred_f, rounded, rtau, rsm, rscale = predict_b(method, GLOBAL_START, FULL_TRAIN_END, b_grid, ss, pf)
        assert np.isfinite(pred_f).all() and (pred_f >= 0).all() and len(rounded) == 1008
        b_final_preds[comp_id] = (pred_f, rounded)
        b_final_params[comp_id] = (rtau, rsm, rscale)
        bf = pd.DataFrame({
            "component_id": comp_id, "method": method, "rounding_method": "independent",
            "selected_tau": rtau, "selected_smoother": rsm, "selected_scale": rscale,
            "date": b_grid["date"].dt.strftime("%Y-%m-%d").to_numpy(), "hour": b_grid["hour"].to_numpy(),
            "hour_of_day": b_grid["hour_of_day"].to_numpy(), "dayofweek": b_grid["dayofweek"].to_numpy(),
            "day_type": b_grid["day_type"].to_numpy(), "source_region": b_grid["source_region"].to_numpy(),
            "target_region": b_grid["target_region"].to_numpy(), "task_key": b_grid["task_key"].to_numpy(),
            "validation_unique_vessel_count": b_grid["date"].map(val_vc).to_numpy(), "pred_float": pred_f, "pred_rounded": rounded,
        })
        b_frames.append(bf)
    print("B components: reproduction + final prediction done.")
    b_hp = pd.DataFrame(b_hp_rows)
    assert len(b_hp) == 51, f"b_hp {len(b_hp)} != 51"
    for comp in ["b_hier_scale", "b_circular"]:
        assert int(b_hp[b_hp["component_id"] == comp]["selected"].sum()) == 1

    repro = pd.DataFrame(repro_rows)
    assert len(repro) == 6 and repro["passed"].all()

    # --- historical metrics for manifest ---
    s7n = pd.read_csv(ROOT / STEP07_NORMAL, encoding="utf-8-sig")
    s7f = pd.read_csv(ROOT / STEP07_FINAL, encoding="utf-8-sig")
    s10d = pd.read_csv(ROOT / STEP10_DECISION, encoding="utf-8-sig")
    a_metrics = {}
    for comp_id, policy, method in A_COMPONENTS:
        nr = s7n[(s7n["training_policy"] == policy) & (s7n["method"] == method)]["mse_rounded"].iloc[0]
        fr = s7f[(s7f["training_policy"] == policy) & (s7f["method"] == method)]["sse_rounded"].iloc[0]
        a_metrics[comp_id] = (float(nr), float(fr))
    b_metrics = {}
    for comp_id, method in B_COMPONENTS:
        r = s10d[(s10d["predictor_method"] == method) & (s10d["rounding_method"] == "independent")].iloc[0]
        b_metrics[comp_id] = (float(r["rolling_mean_mse"]), float(r["rolling_median_mse"]), float(r["rolling_max_mse"]), float(r["final_analog_sse"]))

    # --- package candidate submission CSVs + manifest + checks ---
    manifest_rows = []
    daily_rows = []
    sub_rows = []
    pred_a_by_comp = {c: a_final_preds[c] for c in a_final_preds}
    pred_b_by_comp = {c: b_final_preds[c] for c in b_final_preds}

    for cand_id, prio, a_comp, b_comp, purpose in CANDIDATES:
        cdir = sub_dir / cand_id
        cdir.mkdir(exist_ok=True)
        _, a_rounded, _, _ = pred_a_by_comp[a_comp]
        _, b_rounded = pred_b_by_comp[b_comp]
        a_csv = fill_template_a(tmpl_a, a_grid, a_rounded)
        b_csv = fill_template_b(tmpl_b, b_grid, b_rounded)
        a_path = cdir / "提交结果1_区域活跃拖轮数量.csv"
        b_path = cdir / "提交结果2_圈层间拖轮迁移量.csv"
        a_csv.to_csv(a_path, index=False, encoding="utf-8-sig")
        b_csv.to_csv(b_path, index=False, encoding="utf-8-sig")

        an, af = a_metrics[a_comp]; bm, bmd, bmx, bf = b_metrics[b_comp]
        manifest_rows.append({
            "candidate_id": cand_id, "priority": prio, "a_component_id": a_comp, "b_component_id": b_comp,
            "a_training_policy": dict(A_COMPONENTS)[a_comp] if False else [c[1] for c in A_COMPONENTS if c[0] == a_comp][0],
            "a_method": [c[2] for c in A_COMPONENTS if c[0] == a_comp][0], "b_method": b_comp, "b_rounding_method": "independent",
            "a_rolling_normal_mse": an, "a_final_analog_sse": af, "b_rolling_mean_mse": bm, "b_rolling_median_mse": bmd,
            "b_rolling_max_mse": bmx, "b_final_analog_sse": bf, "diagnostic_combined_final_analog": af + 3 * bf,
            "submission_directory": f"submissions/{cand_id}/", "purpose": purpose,
        })
        # daily totals
        adf = a_frames[[i for i, c in enumerate(A_COMPONENTS) if c[0] == a_comp][0]]
        bdf = b_frames[[i for i, c in enumerate(B_COMPONENTS) if c[0] == b_comp][0]]
        for d in pd.date_range(VAL_START, VAL_END, freq="D").normalize():
            ds = f"{d:%Y-%m-%d}"
            ar = adf[adf["date"] == ds]["pred_rounded"].to_numpy()
            br = bdf[bdf["date"] == ds]["pred_rounded"].to_numpy()
            daily_rows.append({"candidate_id": cand_id, "task": "A", "date": ds, "prediction_total": int(ar.sum()),
                               "prediction_mean": float(ar.mean()), "prediction_max": int(ar.max()), "zero_prediction_count": int((ar == 0).sum())})
            daily_rows.append({"candidate_id": cand_id, "task": "B", "date": ds, "prediction_total": int(br.sum()),
                               "prediction_mean": float(br.mean()), "prediction_max": int(br.max()), "zero_prediction_count": int((br == 0).sum())})
        # submission checks
        sub_rows.append(check_submission(cand_id, "A", a_path, tmpl_a))
        sub_rows.append(check_submission(cand_id, "B", b_path, tmpl_b))

    manifest = pd.DataFrame(manifest_rows)
    daily = pd.DataFrame(daily_rows)

    # --- pairwise comparison ---
    pair_rows = []
    for (ci, cj) in [("candidate_01_main", "candidate_02_a_sensitivity"), ("candidate_01_main", "candidate_03_b_robustness"), ("candidate_02_a_sensitivity", "candidate_03_b_robustness")]:
        ai = [c for c in CANDIDATES if c[0] == ci][0][2]; aj = [c for c in CANDIDATES if c[0] == cj][0][2]
        bi = [c for c in CANDIDATES if c[0] == ci][0][3]; bj = [c for c in CANDIDATES if c[0] == cj][0][3]
        ai_r = pred_a_by_comp[ai][1]; aj_r = pred_a_by_comp[aj][1]; bi_r = pred_b_by_comp[bi][1]; bj_r = pred_b_by_comp[bj][1]
        ad = ai_r - aj_r; bd = bi_r - bj_r
        pair_rows.append({
            "candidate_i": ci, "candidate_j": cj,
            "a_differing_cell_count": int((ad != 0).sum()), "a_mean_abs_difference": float(np.abs(ad).mean()),
            "a_max_abs_difference": int(np.abs(ad).max()), "a_total_difference": int(ad.sum()),
            "b_differing_cell_count": int((bd != 0).sum()), "b_mean_abs_difference": float(np.abs(bd).mean()),
            "b_max_abs_difference": int(np.abs(bd).max()), "b_total_difference": int(bd.sum()),
        })
    pairwise = pd.DataFrame(pair_rows)

    # --- write top-level files ---
    a_comp_df = pd.concat(a_frames, ignore_index=True)
    a_comp_df["_c"] = a_comp_df["component_id"].map({c[0]: i for i, c in enumerate(A_COMPONENTS)})
    a_comp_df = a_comp_df.sort_values(["_c", "date", "hour_of_day", "region"]).drop(columns=["_c"])
    b_comp_df = pd.concat(b_frames, ignore_index=True)
    b_comp_df["_c"] = b_comp_df["component_id"].map({c[0]: i for i, c in enumerate(B_COMPONENTS)})
    b_comp_df = b_comp_df.sort_values(["_c", "date", "hour_of_day", "source_region", "target_region"]).drop(columns=["_c"])

    p_manifest = out_dir / "candidate_manifest.csv"
    p_a = out_dir / "a_component_predictions.csv"
    p_b = out_dir / "b_component_predictions.csv"
    p_hp = out_dir / "final_b_hyperparameter_selection.csv"
    p_rep = out_dir / "reproduction_checks.csv"
    p_dt = out_dir / "candidate_daily_totals.csv"
    p_pw = out_dir / "candidate_pairwise_comparison.csv"
    p_sc = out_dir / "submission_checks.csv"
    p_md = out_dir / "summary.md"

    manifest.to_csv(p_manifest, index=False, encoding="utf-8-sig")
    a_comp_df[A_PRED_COLS].to_csv(p_a, index=False, encoding="utf-8-sig")
    b_comp_df[B_PRED_COLS].to_csv(p_b, index=False, encoding="utf-8-sig")
    hp_out = b_hp.copy()
    hp_out["_c"] = hp_out["component_id"].map({c[0]: i for i, c in enumerate(B_COMPONENTS)})
    hp_out = hp_out.sort_values(["_c", "tau", "smoother", "scale"]).drop(columns=["_c"])
    hp_out.to_csv(p_hp, index=False, encoding="utf-8-sig")
    repro.to_csv(p_rep, index=False, encoding="utf-8-sig")
    daily.sort_values(["candidate_id", "task", "date"]).to_csv(p_dt, index=False, encoding="utf-8-sig")
    pairwise.to_csv(p_pw, index=False, encoding="utf-8-sig")
    pd.DataFrame(sub_rows).to_csv(p_sc, index=False, encoding="utf-8-sig")

    # --- assertions ---
    assert len(manifest) == 3 and len(a_comp_df) == 1512 and len(b_comp_df) == 3024 and len(b_hp) == 51
    assert len(repro) == 6 and len(daily) == 42 and len(pairwise) == 3 and len(sub_rows) == 6
    # no zip
    assert not any(out_dir.rglob("*.zip"))
    for cand_id, *_ in CANDIDATES:
        assert (sub_dir / cand_id / "提交结果1_区域活跃拖轮数量.csv").exists()
        assert (sub_dir / cand_id / "提交结果2_圈层间拖轮迁移量.csv").exists()

    write_summary(p_md, manifest, repro, b_hp, daily, pairwise, a_comp_df, b_comp_df, b_final_params)

    # --- console ---
    print("\n=== Output files ===")
    for path in [p_manifest, p_a, p_b, p_hp, p_rep, p_dt, p_pw, p_sc, p_md]:
        rows = len(pd.read_csv(path, encoding="utf-8-sig")) if path.suffix == ".csv" else sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {path.relative_to(ROOT)}  rows={rows}")
    print("\n=== candidate dirs ===")
    for cand_id, *_ in CANDIDATES:
        print(f"  submissions/{cand_id}/")
    rt, rs, rm = b_final_params["b_hier_scale"], b_final_params["b_circular"], b_final_params["b_fixed085"]
    print(f"\nB selected: b_hier_scale tau={rt[0]} scale={rt[2]} | b_circular smoother={rs[1]} | b_fixed085 scale={rm[2]}")
    for cand_id, prio, a_comp, b_comp, _ in CANDIDATES:
        at = int(pred_a_by_comp[a_comp][1].sum()); bt = int(pred_b_by_comp[b_comp][1].sum())
        print(f"  {cand_id}: A total={at}, B total={bt}")
    print("\nDone.")


def fill_template_a(tmpl, grid, rounded):
    out = tmpl.copy()
    ts = pd.to_datetime(tmpl["time_window"])
    region = tmpl["zone"].map(CN_TO_REGION)
    pred_map = {}
    for i in range(len(grid)):
        hr = grid["hour"].iloc[i]; r = grid["region"].iloc[i]
        pred_map[(pd.Timestamp(hr), r)] = int(rounded[i])
    vals = [pred_map.get((pd.Timestamp(ts.iloc[i]), region.iloc[i])) for i in range(len(tmpl))]
    assert all(v is not None for v in vals), "A template rows not fully matched"
    out["vessel_count"] = vals
    return out[["time_window", "zone", "vessel_count"]]


def fill_template_b(tmpl, grid, rounded):
    out = tmpl.copy()
    ts = pd.to_datetime(tmpl["time_window"])
    src = tmpl["source_zone"].map(CN_TO_REGION); tgt = tmpl["target_zone"].map(CN_TO_REGION)
    pred_map = {}
    for i in range(len(grid)):
        hr = grid["hour"].iloc[i]; s = grid["source_region"].iloc[i]; t = grid["target_region"].iloc[i]
        pred_map[(pd.Timestamp(hr), s, t)] = int(rounded[i])
    vals = [pred_map.get((pd.Timestamp(ts.iloc[i]), src.iloc[i], tgt.iloc[i])) for i in range(len(tmpl))]
    assert all(v is not None for v in vals), "B template rows not fully matched"
    out["vessel_count"] = vals
    return out[["time_window", "source_zone", "target_zone", "vessel_count"]]


def check_submission(cand_id, task, path, tmpl):
    df = pd.read_csv(path, encoding="utf-8-sig")
    exp = 504 if task == "A" else 1008
    cols = list(df.columns)
    tmpl_cols = list(tmpl.columns)
    cols_exact = cols == tmpl_cols
    # template order: compare time_window+zone(s) sequence
    if task == "A":
        order_exact = (df["time_window"].to_numpy() == tmpl["time_window"].to_numpy()).all() and (df["zone"].to_numpy() == tmpl["zone"].to_numpy()).all()
        key_unique = not df[["time_window", "zone"]].duplicated().any()
    else:
        order_exact = (df["time_window"].to_numpy() == tmpl["time_window"].to_numpy()).all() and (df["source_zone"].to_numpy() == tmpl["source_zone"].to_numpy()).all() and (df["target_zone"].to_numpy() == tmpl["target_zone"].to_numpy()).all()
        key_unique = not df[["time_window", "source_zone", "target_zone"]].duplicated().any()
    vc = df["vessel_count"]
    return {
        "candidate_id": cand_id, "task": task, "file_path": str(path.relative_to(ROOT)), "row_count": int(len(df)),
        "expected_row_count": exp, "columns_exact": bool(cols_exact), "template_order_exact": bool(order_exact),
        "key_unique": bool(key_unique), "date_start": str(pd.to_datetime(df["time_window"]).min().date()),
        "date_end": str(pd.to_datetime(df["time_window"]).max().date()),
        "missing_value_count": int(vc.isna().sum()), "negative_value_count": int((vc < 0).sum()),
        "noninteger_value_count": int((vc % 1 != 0).sum()), "passed": bool(len(df) == exp and cols_exact and order_exact and key_unique and vc.notna().all() and (vc >= 0).all()),
    }


def write_summary(path, manifest, repro, b_hp, daily, pairwise, a_comp_df, b_comp_df, b_params):
    L = []
    L.append("# Step 11 Final Validation Candidates")
    L.append("")
    L.append("> Frozen A/B components retrained on the full training period; predictions for 2018-01-25..01-31. Generated, NOT submitted.")
    L.append("")
    L.append(f"Run command: `{RUN_CMD}`")
    L.append("")

    L.append("## 1. Frozen components")
    L.append("")
    L.append("- A: a_exclude_daytype (exclude_outage_severe / decomp_mean_daytype_shrunk), a_quality_hour (quality_weighted / hour_mean), a_quality_mean_profile (quality_weighted / decomp_mean_mean).")
    L.append("- B: b_hier_scale (pair_hour_hier_tau_scale_cv / independent), b_circular (pair_hour_circular_cv / independent), b_fixed085 (pair_hour_fixed085 / independent).")
    L.append("- No new models, no ensembling. No validation AIS, no validation labels. validation daily vessel count is retained only in the debug tables; frozen models do NOT scale by it.")
    L.append("")

    L.append("## 2. Historical reproduction checks")
    L.append("")
    L.append("| component | ref step | max abs diff | rounded exact | passed |")
    L.append("| --- | --- | --- | --- | --- |")
    for _, r in repro.iterrows():
        L.append(f"| {r['component_id']} | {r['reference_step']} | {r['max_abs_diff_float']:.2e} | {r['rounded_exact_match_rate']:.3f} | {r['passed']} |")
    L.append("")

    L.append("## 3. Final A training")
    L.append("")
    L.append("- Quality groups recomputed on the full 2018-01-01..01-24 training period; exclude_outage_severe keeps normal+reduced dates; quality_weighted uses weights normal=1/reduced=0.5/severe=0.1/outage=0.")
    for comp_id, policy, method in A_COMPONENTS:
        sub = a_comp_df[a_comp_df["component_id"] == comp_id]
        L.append(f"- {comp_id}: validation-period predicted total = {int(sub['pred_rounded'].sum())}.")
    L.append("")

    L.append("## 4. Final B hyperparameter selection")
    L.append("")
    rt = b_params["b_hier_scale"]; rs = b_params["b_circular"]
    L.append(f"- 17 inner validation dates (2018-01-07..01-23).")
    L.append(f"- b_hier_scale selected: tau={rt[0]}, scale={rt[2]}.")
    L.append(f"- b_circular selected: smoother={rs[1]}.")
    for comp in ["b_hier_scale", "b_circular"]:
        sub = b_hp[b_hp["component_id"] == comp].sort_values("inner_sse_rounded")
        best = sub.iloc[0]; second = sub.iloc[1]
        gap = second["inner_sse_rounded"] - best["inner_sse_rounded"]
        L.append(f"- {comp}: best inner_sse={best['inner_sse_rounded']:.0f}, second={second['inner_sse_rounded']:.0f}, gap={gap:.1f}{' (small -> selection unstable)' if gap < 1e-6 else ''}.")
    L.append("")

    L.append("## 5. Final B training boundary")
    L.append("")
    L.append("- 2018-01-24 23:00 six migration labels are right-censored (no 2018-01-25 00:00 state) and did NOT enter training; last observable training migration hour is 2018-01-24 22:00. Final prediction still covers all 1008 rows 2018-01-25..01-31 (including 01-31 23:00) and uses no validation AIS state.")
    L.append("")

    L.append("## 6. Candidate definitions")
    L.append("")
    for cand_id, prio, a_c, b_c, purpose in CANDIDATES:
        L.append(f"- {cand_id}: A={a_c}, B={b_c} — {purpose}")
    L.append("- candidate_02 changes only A; candidate_03 changes only B (vs candidate_01), so online sub-component results are easier to attribute.")
    L.append("")

    L.append("## 7. Prediction totals and dispersion")
    L.append("")
    L.append("| candidate | A total | B total | A daily range | B daily range |")
    L.append("| --- | --- | --- | --- | --- |")
    for cand_id, prio, a_c, b_c, _ in CANDIDATES:
        at = int(a_comp_df[a_comp_df["component_id"] == a_c]["pred_rounded"].sum())
        bt = int(b_comp_df[b_comp_df["component_id"] == b_c]["pred_rounded"].sum())
        ad = daily[(daily["candidate_id"] == cand_id) & (daily["task"] == "A")]["prediction_total"]
        bd = daily[(daily["candidate_id"] == cand_id) & (daily["task"] == "B")]["prediction_total"]
        L.append(f"| {cand_id} | {at} | {bt} | {int(ad.min())}-{int(ad.max())} | {int(bd.min())}-{int(bd.max())} |")
    L.append("")
    L.append("| pair | A differing cells | A mean |diff| | B differing cells | B mean |diff| |")
    L.append("| --- | --- | --- | --- | --- |")
    for _, r in pairwise.iterrows():
        L.append(f"| {r['candidate_i']} vs {r['candidate_j']} | {int(r['a_differing_cell_count'])} | {r['a_mean_abs_difference']:.3f} | {int(r['b_differing_cell_count'])} | {r['b_mean_abs_difference']:.3f} |")
    L.append("")

    L.append("## 8. Official-file validation")
    L.append("")
    L.append("- Each A file has 504 rows; each B file 1008 rows; columns match the template exactly; row order matches the template; keys unique; no missing/negative values; all integers; UTF-8 with BOM; no ZIP generated. See submission_checks.csv (all passed).")
    L.append("")

    L.append("## 9. Files prepared for review")
    L.append("")
    for cand_id, *_ in CANDIDATES:
        L.append(f"- submissions/{cand_id}/提交结果1_区域活跃拖轮数量.csv")
        L.append(f"- submissions/{cand_id}/提交结果2_圈层间拖轮迁移量.csv")
    L.append("")
    L.append("These files have been generated but not submitted.")
    L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
