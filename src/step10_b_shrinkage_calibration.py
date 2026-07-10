"""Step 10 — Shrinkage, calibration, and integerization comparison for Task B.

12 predictor methods x 2 rounding methods = 24 variants. Parameter selection
(inner time-series CV, integer-SSE) for tau / circular smoother / global scale /
direction scale. No Ridge/Poisson/trees/NN, no submission, no A, no validation
state/vessel features. Step 09 boundary rules reused.

Outputs (under outputs/step10_b_shrinkage_calibration/):
  1. method_definitions.csv          (24)
  2. inner_validation_definitions.csv(63)
  3. hyperparameter_selection.csv    (3330)
  4. b_model_predictions.csv         (214272)
  5. b_fold_scores.csv               (216)
  6. b_direction_scores.csv          (1296)
  7. b_rounding_comparison.csv       (12)
  8. decision_table.csv              (24)
  9. summary.md
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

SS_PATH = Path("outputs/step08_b_state_transition_audit/b_hourly_source_state.csv")
PF_PATH = Path("outputs/step08_b_state_transition_audit/b_hourly_pair_flow.csv")
STEP09_PREDS = Path("outputs/step09_b_structure_model_benchmark/b_model_predictions.csv")
OUT_REL = Path("outputs/step10_b_shrinkage_calibration")

REGION_ORDER = ["core", "near", "outer"]
MIG = {"core": ("near", "outer"), "near": ("core", "outer"), "outer": ("core", "near")}
DIRS = [("core", "near"), ("core", "outer"), ("near", "core"), ("near", "outer"), ("outer", "core"), ("outer", "near")]
DIR_NAMES = [f"{s}->{t}" for s, t in DIRS]
DIR_RANK = {d: i for i, d in enumerate(DIR_NAMES)}
REGION_RANK = {r: i for i, r in enumerate(REGION_ORDER)}
UNSCORABLE = pd.Timestamp("2018-01-24 23:00:00")
TAU = 20
TAUS = [0, 5, 10, 20, 50, 100]
SCALES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05]
SMOOTHERS = ["raw", "triangular3", "triangular5"]
SMOOTHER_RANK = {s: i for i, s in enumerate(SMOOTHERS)}
GLOBAL_START = pd.Timestamp("2018-01-01")

# predictor: (family, base_window, hier, circ, present, needs_sel)
PREDICTORS = [
    ("pair_hour_mean", "direct_mean", "long", False, False, False, False),
    ("pair_hour_fixed085", "direct_mean", "long", False, False, False, False),
    ("pair_recent14_hour", "direct_mean", "recent14", False, False, False, False),
    ("pair_recent14_fixed085", "direct_mean", "recent14", False, False, False, False),
    ("pair_hour_hier_tau_cv", "hierarchical_mean", "long", True, False, False, True),
    ("pair_hour_hier_tau_scale_cv", "hierarchical_mean", "long", True, False, False, True),
    ("pair_hour_circular_cv", "circular_mean", "long", False, True, False, True),
    ("pair_hour_circular_scale_cv", "circular_mean", "long", False, True, False, True),
    ("present_shrunk20", "present_transition", "long", False, False, True, False),
    ("present_shrunk20_scale_cv", "present_transition", "long", False, False, True, True),
    ("pair_hour_direction_scale_cv", "direct_mean", "long", False, False, False, True),
    ("present_shrunk20_direction_scale_cv", "present_transition", "long", False, False, True, True),
]
PRED_ORDER = {p[0]: i for i, p in enumerate(PREDICTORS)}
ROUNDINGS = ["independent", "paired"]
NEEDS_SEL = [p[0] for p in PREDICTORS if p[6]]
FIXED = [p[0] for p in PREDICTORS if not p[6]]

PRED_OUT_COLS = [
    "evaluation_id", "evaluation_type", "predictor_method", "rounding_method", "date", "hour", "hour_of_day",
    "dayofweek", "day_type", "source_region", "target_region", "task_key", "audit_quality_regime", "period",
    "score_available", "y_true", "selected_tau", "selected_smoother", "selected_scale", "predicted_source_present",
    "constraint_adjusted", "constraint_adjustment", "pred_float_before_scale", "pred_float", "pred_rounded",
    "error_float", "error_rounded",
]
HP_COLS = ["evaluation_id", "evaluation_type", "rounding_method", "predictor_method", "task_key", "tau", "smoother",
           "scale", "n_inner_splits", "inner_validation_dates", "inner_sse_rounded", "inner_sse_float",
           "inner_mae_rounded", "mean_bias_rounded", "selected"]
INNER_DEF_COLS = ["evaluation_id", "inner_fold_id", "inner_train_start", "inner_train_end",
                  "excluded_inner_train_boundary_hour", "inner_valid_date", "n_inner_scored_rows"]
FOLD_COLS = ["evaluation_id", "evaluation_type", "predictor_method", "rounding_method", "n_prediction_rows",
             "n_scored_rows", "sse_float", "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded",
             "actual_total_scored", "predicted_total_scored", "constraint_adjustment_count", "max_constraint_adjustment",
             "core_to_near_sse", "core_to_outer_sse", "near_to_core_sse", "near_to_outer_sse", "outer_to_core_sse", "outer_to_near_sse"]
DIR_COLS = ["evaluation_id", "evaluation_type", "predictor_method", "rounding_method", "task_key", "n_scored_rows",
            "actual_total", "predicted_total", "sse_float", "sse_rounded", "mse_rounded", "mae_rounded",
            "mean_bias_rounded", "zero_true_rate", "zero_prediction_rate"]
ROUND_CMP_COLS = ["predictor_method", "independent_rolling_mean_mse", "paired_rolling_mean_mse",
                  "independent_rolling_median_mse", "paired_rolling_median_mse", "independent_rolling_max_mse",
                  "paired_rolling_max_mse", "independent_final_analog_sse", "paired_final_analog_sse",
                  "paired_better_fold_count", "independent_better_fold_count", "tie_fold_count"]
DECISION_COLS = ["predictor_method", "rounding_method", "model_family", "rolling_mean_mse", "rolling_median_mse",
                 "rolling_max_mse", "rolling_mean_sse", "final_analog_sse", "final_analog_mse", "final_analog_mean_bias",
                 "final_analog_rank", "rolling_fold_win_count", "selected_scale_mode", "median_selected_global_scale",
                 "selected_tau_mode", "selected_smoother_mode", "core_to_near_total_sse", "core_to_outer_total_sse",
                 "near_to_core_total_sse", "near_to_outer_total_sse", "outer_to_core_total_sse", "outer_to_near_total_sse"]
DEF_COLS = ["predictor_method", "rounding_method", "model_family", "base_window", "uses_hour_hierarchy",
            "uses_circular_smoothing", "uses_global_scale", "uses_direction_scale", "uses_present_structure",
            "requires_inner_selection", "deployable"]

RUN_CMD = "python " + " ".join(sys.argv)


def fall(*vals):
    for v in vals:
        if v is not None:
            return v
    return 0.0


def build_scenarios():
    sc = []
    for k in range(8):
        sc.append((f"fold_{k + 1:02d}", "rolling_7d", GLOBAL_START, pd.Timestamp(f"2018-01-{10 + k:02d}"),
                   pd.Timestamp(f"2018-01-{11 + k:02d}"), pd.Timestamp(f"2018-01-{17 + k:02d}")))
    sc.append(("final_analog", "final_analog", GLOBAL_START, pd.Timestamp("2018-01-18"),
               pd.Timestamp("2018-01-19"), pd.Timestamp("2018-01-24")))
    return sc


def smooth_circular(vec, smoother):
    if smoother == "raw":
        return vec.copy()
    out = np.zeros(24)
    if smoother == "triangular3":
        for h in range(24):
            out[h] = (vec[(h - 1) % 24] + 2 * vec[h] + vec[(h + 1) % 24]) / 4
    else:
        for h in range(24):
            out[h] = (vec[(h - 2) % 24] + 2 * vec[(h - 1) % 24] + 3 * vec[h] + 2 * vec[(h + 1) % 24] + vec[(h + 2) % 24]) / 9
    return np.clip(out, 0, None)


def compute_aggregates(ss, pf, train_start, train_end):
    train_dates = pd.date_range(train_start, train_end, freq="D").normalize()
    boundary = train_end + pd.Timedelta(hours=23)
    pf_tr = pf[(pf["date"].isin(train_dates)) & (pf["hour"] != boundary)]
    ss_stock = ss[ss["date"].isin(train_dates)]
    ss_tr = ss_stock[ss_stock["hour"] != boundary]

    hm = pf_tr.groupby(["source_region", "target_region", "hour_of_day"])["y"].mean()
    hour_mean = {(s, t, int(h)): float(hm.loc[(s, t, h)]) for (s, t, h) in hm.index}
    pair_dir = {k: float(v) for k, v in pf_tr.groupby(["source_region", "target_region"])["y"].mean().items()}
    overall = float(pf_tr["y"].mean())
    hsum = pf_tr.groupby(["source_region", "target_region", "hour_of_day"])["y"].sum()
    hcnt = pf_tr.groupby(["source_region", "target_region", "hour_of_day"])["y"].size()
    hier_sum = {(s, t, int(h)): float(hsum.loc[(s, t, h)]) for (s, t, h) in hsum.index}
    hier_n = {(s, t, int(h)): int(hcnt.loc[(s, t, h)]) for (s, t, h) in hcnt.index}
    mu = pair_dir
    r14_start = max(train_start, train_end - pd.Timedelta(days=13))
    pf14 = pf[(pf["date"].between(r14_start, train_end)) & (pf["hour"] != boundary)]
    rec14 = {(s, t, int(h)): float(v) for (s, t, h), v in pf14.groupby(["source_region", "target_region", "hour_of_day"])["y"].mean().items()}
    stock_long = {(s, int(h)): float(v) for (s, h), v in ss_stock.groupby(["source_region", "hour_of_day"])["source_present_count"].mean().items()}

    # present shrunk20 probs
    n4shrunk = {s: {} for s in REGION_ORDER}
    g4p = {}
    for s in REGION_ORDER:
        m1, m2 = MIG[s]
        sub = ss_tr[ss_tr["source_region"] == s]
        gs = {"stay": float(sub["stay_count"].sum()), m1: float(sub[f"to_{m1}_count"].sum()),
              m2: float(sub[f"to_{m2}_count"].sum()), "noconsec": float(sub["no_consecutive_count"].sum())}
        gtot = sum(gs.values())
        g4p[s] = {k: (v / gtot if gtot > 0 else 0.0) for k, v in gs.items()}
        for h, d in sub.groupby("hour_of_day"):
            dh = {"stay": float(d["stay_count"].sum()), m1: float(d[f"to_{m1}_count"].sum()),
                  m2: float(d[f"to_{m2}_count"].sum()), "noconsec": float(d["no_consecutive_count"].sum())}
            tot = sum(dh.values())
            n4shrunk[s][int(h)] = {k: (dh[k] + TAU * g4p[s][k]) / (tot + TAU) for k in dh}

    # circular smoothed hour means per direction
    circular = {sm: {} for sm in SMOOTHERS}
    for (s, t) in DIRS:
        vec = np.array([hour_mean.get((s, t, h), pair_dir.get((s, t), overall)) for h in range(24)])
        for sm in SMOOTHERS:
            sv = smooth_circular(vec, sm)
            for h in range(24):
                circular[sm][(s, t, h)] = float(sv[h])

    return dict(hour_mean=hour_mean, pair_dir=pair_dir, overall=overall, hier_sum=hier_sum, hier_n=hier_n, mu=mu,
                rec14=rec14, stock_long=stock_long, n4shrunk=n4shrunk, g4p=g4p, circular=circular, boundary=boundary,
                train_dates=train_dates)


def compute_bases(agg, valid):
    s = valid["source_region"].to_numpy()
    t = valid["target_region"].to_numpy()
    h = valid["hour_of_day"].to_numpy().astype(int)
    n = len(valid)
    hm = np.array([fall(agg["hour_mean"].get((s[i], t[i], h[i])), agg["pair_dir"].get((s[i], t[i])), agg["overall"]) for i in range(n)])
    rec14 = np.array([fall(agg["rec14"].get((s[i], t[i], h[i])), agg["hour_mean"].get((s[i], t[i], h[i])), agg["pair_dir"].get((s[i], t[i])), agg["overall"]) for i in range(n)])
    hier = {}
    for tau in TAUS:
        arr = np.zeros(n)
        for i in range(n):
            nn = agg["hier_n"].get((s[i], t[i], h[i]), 0)
            ss_ = agg["hier_sum"].get((s[i], t[i], h[i]), 0.0)
            m = agg["mu"].get((s[i], t[i]), agg["overall"])
            denom = nn + tau
            arr[i] = (ss_ + tau * m) / denom if denom > 0 else agg["overall"]
        hier[tau] = arr
    circ = {}
    for sm in SMOOTHERS:
        circ[sm] = np.array([agg["circular"][sm].get((s[i], t[i], h[i]), agg["pair_dir"].get((s[i], t[i]), agg["overall"])) for i in range(n)])
    pres_stock = np.array([fall(agg["stock_long"].get((s[i], h[i]))) for i in range(n)])
    pres_prob = np.array([(agg["n4shrunk"].get(s[i], {}).get(h[i], agg["g4p"].get(s[i], {}))).get(t[i], 0.0) for i in range(n)])
    pres = pres_stock * pres_prob
    return dict(hm=hm, rec14=rec14, hier=hier, circ=circ, pres=pres, pres_stock=pres_stock)


def cell_groups_of(valid):
    groups = []
    for (_, _, _), idx in valid.groupby(["date", "hour", "source_region"]).groups.items():
        groups.append(np.array(idx))
    return groups


def round_pred(pred, rounding, groups, dir_order):
    if rounding == "independent":
        return np.clip(np.rint(pred), 0, None).astype(np.int64)
    out = np.zeros(len(pred), dtype=np.int64)
    for idxs in groups:
        f = pred[idxs]
        K = int(round(f.sum()))
        b = np.floor(f).astype(np.int64)
        rem = K - int(b.sum())
        if rem > 0:
            frac = f - b
            order = np.lexsort((dir_order[idxs], -pred[idxs], -frac))
            for j in range(rem):
                b[order[j]] += 1
        out[idxs] = np.clip(b, 0, None)
    return out


def apply_constraint(pred, pres_stock, groups):
    adjusted = np.zeros(len(pred), dtype=bool)
    adj = np.zeros(len(pred))
    for idxs in groups:
        p = pres_stock[idxs[0]]
        tot = pred[idxs].sum()
        if p > 0 and tot > p:
            factor = p / tot
            newf = pred[idxs] * factor
            adj[idxs] = pred[idxs] - newf
            adjusted[idxs] = True
            pred[idxs] = newf
    return pred, adjusted, adj


def score_candidate(base, scale_arr, is_present, rounding, bases, groups, dir_order, y, pres_stock):
    pred = base * scale_arr
    if is_present:
        pred = pred.copy()
        pred, _, _ = apply_constraint(pred, pres_stock, groups)
    rounded = round_pred(pred, rounding, groups, dir_order)
    er = rounded - y
    return float((er ** 2).sum()), float(((pred - y) ** 2).sum()), float(np.abs(er).mean()), float(er.mean()), rounded, pred


def inner_folds(train_start, train_end):
    dates = pd.date_range(train_start, train_end, freq="D").normalize()
    n = len(dates)
    folds = []
    for i in range(6, n - 1):
        folds.append((dates[i], train_start, dates[i - 1]))  # (valid_date, train_start, train_end)
    return folds


# candidate generators per selection method
def candidates_for(method):
    if method == "pair_hour_hier_tau_cv":
        return [("tau", tau, None, None) for tau in TAUS]
    if method == "pair_hour_hier_tau_scale_cv":
        return [("tauscale", tau, None, scale) for tau in TAUS for scale in SCALES]
    if method == "pair_hour_circular_cv":
        return [("smoother", None, sm, None) for sm in SMOOTHERS]
    if method == "pair_hour_circular_scale_cv":
        return [("smscale", None, sm, scale) for sm in SMOOTHERS for scale in SCALES]
    if method in ("present_shrunk20_scale_cv",):
        return [("scale", None, None, scale) for scale in SCALES]
    return None  # direction-scale handled separately


def select_key(method, cand):
    """Deterministic selection comparator -> sort key (smaller is better)."""
    typ, tau, sm, scale = cand
    # primary: inner sse (handled outside); secondary tie-breakers encoded here
    return cand


def make_scale_arr(scale, task_arr, direction_scale=None):
    if direction_scale is not None:
        return np.array([direction_scale[tk] for tk in task_arr])
    return np.full(len(task_arr), scale)


def main():
    out_dir = ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    ss = pd.read_csv(ROOT / SS_PATH, encoding="utf-8-sig")
    ss["hour"] = pd.to_datetime(ss["hour"]); ss["date"] = pd.to_datetime(ss["date"]).dt.normalize()
    ss["hour_of_day"] = ss["hour"].dt.hour
    pf = pd.read_csv(ROOT / PF_PATH, encoding="utf-8-sig")
    pf["hour"] = pd.to_datetime(pf["hour"]); pf["date"] = pd.to_datetime(pf["date"]).dt.normalize()
    pf["hour_of_day"] = pf["hour"].dt.hour
    pf["dayofweek"] = pf["hour"].dt.dayofweek
    pf["dir_order"] = pf["task_key"].map(DIR_RANK)

    scenarios = build_scenarios()
    inner_def_rows = []
    hp_rows = []
    pred_frames = []
    fold_rows = []
    dir_rows = []

    # precompute inner folds + definitions
    for eid, etype, tstart, tend, vstart, vend in scenarios:
        folds = inner_folds(tstart, tend)
        for fi, (vdate, itrain_start, itrain_end) in enumerate(folds):
            inner_def_rows.append({
                "evaluation_id": eid, "inner_fold_id": fi + 1,
                "inner_train_start": f"{itrain_start:%Y-%m-%d}", "inner_train_end": f"{itrain_end:%Y-%m-%d}",
                "excluded_inner_train_boundary_hour": f"{itrain_end + pd.Timedelta(hours=23):%Y-%m-%d %H:%M:%S}",
                "inner_valid_date": f"{vdate:%Y-%m-%d}", "n_inner_scored_rows": 144,
            })

    for eid, etype, tstart, tend, vstart, vend in scenarios:
        outer_agg = compute_aggregates(ss, pf, tstart, tend)
        # validation frame
        valid = pf[(pf["date"] >= vstart) & (pf["date"] <= vend)].copy()
        valid = valid.sort_values(["hour", "source_region", "target_region"]).reset_index(drop=True)
        valid["score_available"] = valid["hour"] != UNSCORABLE
        valid["y_true_keep"] = valid["y"]
        obases = compute_bases(outer_agg, valid)
        ogroups = cell_groups_of(valid)
        odir_order = valid["dir_order"].to_numpy()
        oy = np.where(valid["score_available"], valid["y_true_keep"], np.nan)
        otask = valid["task_key"].to_numpy()
        osrc = valid["source_region"].to_numpy()
        otgt = valid["target_region"].to_numpy()

        # inner folds bases
        folds = inner_folds(tstart, tend)
        inner_data = []  # per fold: bases, groups, dir_order, y, task
        for (vdate, itrain_start, itrain_end) in folds:
            iagg = compute_aggregates(ss, pf, itrain_start, itrain_end)
            iv = pf[pf["date"] == vdate].sort_values(["hour", "source_region", "target_region"]).reset_index(drop=True)
            ib = compute_bases(iagg, iv)
            ig = cell_groups_of(iv)
            iod = iv["dir_order"].to_numpy()
            iy = iv["y"].to_numpy(float)
            itask = iv["task_key"].to_numpy()
            inner_data.append((ib, ig, iod, iy, itask))
        inner_dates_str = ",".join(f"{d:%Y-%m-%d}" for d, _, _ in folds)
        n_inner = len(folds)

        # ---- parameter selection for CV methods ----
        selected = {}  # method -> {rounding: params}
        for meta in PREDICTORS:
            mname = meta[0]
            if mname not in NEEDS_SEL:
                continue
            is_present = meta[5]
            selected[mname] = {}
            for rnd in ROUNDINGS:
                if mname in ("pair_hour_direction_scale_cv", "present_shrunk20_direction_scale_cv"):
                    base_key = "pres" if is_present else "hm"
                    dir_sel = {}
                    for tk in DIR_NAMES:
                        scored = []
                        for scale in SCALES:
                            sse_sum = flo_sum = 0.0; nb = 0
                            for (ib, ig, iod, iy, itask) in inner_data:
                                base = ib[base_key]
                                sarr = np.where(itask == tk, scale, 1.0)
                                pred = base * sarr
                                if is_present:
                                    pred = pred.copy(); pred, _, _ = apply_constraint(pred, ib["pres_stock"], ig)
                                rounded = round_pred(pred, rnd, ig, iod)
                                mask = itask == tk
                                er = (rounded - iy)[mask]
                                ef = (pred - iy)[mask]
                                sse_sum += float((er ** 2).sum()); flo_sum += float((ef ** 2).sum())
                                mae_sum = float(np.abs(er).mean()); bias_sum = float(er.mean()); nb += 1
                            scored.append((scale, sse_sum, flo_sum, mae_sum, bias_sum))
                        sel_scale = sorted(scored, key=lambda r: (r[1], abs(r[0] - 1.0), -r[0]))[0][0]
                        dir_sel[tk] = sel_scale
                        for (scale, sse, flo, mae, bias) in scored:
                            hp_rows.append({"evaluation_id": eid, "evaluation_type": etype, "rounding_method": rnd,
                                            "predictor_method": mname, "task_key": tk, "tau": np.nan, "smoother": "",
                                            "scale": scale, "n_inner_splits": n_inner, "inner_validation_dates": inner_dates_str,
                                            "inner_sse_rounded": sse, "inner_sse_float": flo, "inner_mae_rounded": mae,
                                            "mean_bias_rounded": bias, "selected": scale == sel_scale})
                    selected[mname][rnd] = ("direction", dir_sel)
                else:
                    cands = candidates_for(mname)
                    scored = []
                    for cand in cands:
                        typ, tau, sm, scale = cand
                        sse_sum = flo_sum = mae_sum = bias_sum = 0.0; nb = 0
                        for (ib, ig, iod, iy, itask) in inner_data:
                            if mname.startswith("pair_hour_hier"):
                                base = ib["hier"][tau]
                            elif mname.startswith("pair_hour_circular"):
                                base = ib["circ"][sm]
                            else:
                                base = ib["pres"]
                            sarr = np.full(len(base), scale if scale is not None else 1.0)
                            sse, flo, mae, bias, _, _ = score_candidate(base, sarr, is_present, rnd, ib, ig, iod, iy, ib["pres_stock"])
                            sse_sum += sse; flo_sum += flo; mae_sum += mae; bias_sum += bias; nb += 1
                        scored.append((cand, sse_sum, flo_sum, mae_sum / nb, bias_sum / nb))
                    sel_idx = _select_idx(mname, scored)
                    sel_cand = scored[sel_idx][0]
                    for (cand, sse, flo, mae, bias) in scored:
                        typ, tau, sm, scale = cand
                        hp_rows.append({"evaluation_id": eid, "evaluation_type": etype, "rounding_method": rnd,
                                        "predictor_method": mname, "task_key": "all",
                                        "tau": tau if tau is not None else np.nan, "smoother": sm if sm else "",
                                        "scale": scale if scale is not None else np.nan, "n_inner_splits": n_inner,
                                        "inner_validation_dates": inner_dates_str, "inner_sse_rounded": sse,
                                        "inner_sse_float": flo, "inner_mae_rounded": mae, "mean_bias_rounded": bias,
                                        "selected": cand == sel_cand})
                    selected[mname][rnd] = sel_cand

        # ---- outer predictions for all 24 variants ----
        for meta in PREDICTORS:
            mname, family, window, hier, circ, present, needs_sel = meta
            for rnd in ROUNDINGS:
                if mname == "pair_hour_mean":
                    base = obases["hm"]; sel = ("fixed", 1.0); sel_tau = np.nan; sel_sm = ""; sel_scale = np.nan
                elif mname == "pair_hour_fixed085":
                    base = obases["hm"]; sel = ("fixed", 0.85); sel_tau = np.nan; sel_sm = ""; sel_scale = 0.85
                elif mname == "pair_recent14_hour":
                    base = obases["rec14"]; sel = ("fixed", 1.0); sel_tau = np.nan; sel_sm = ""; sel_scale = np.nan
                elif mname == "pair_recent14_fixed085":
                    base = obases["rec14"]; sel = ("fixed", 0.85); sel_tau = np.nan; sel_sm = ""; sel_scale = 0.85
                elif mname == "present_shrunk20":
                    base = obases["pres"]; sel = ("fixed", 1.0); sel_tau = np.nan; sel_sm = ""; sel_scale = np.nan
                else:
                    sp = selected[mname][rnd]
                    if mname.startswith("pair_hour_hier"):
                        base = obases["hier"]
                    elif mname.startswith("pair_hour_circular"):
                        base = obases["circ"]
                    else:
                        base = obases["pres"]
                    sel = sp
                pred_before = _materialize_base(base, mname, sel)
                # determine scale array + record selected params
                scale_arr, rtau, rsm, rscale = _resolve_scale(mname, sel, otask)
                pred = pred_before * scale_arr
                if present:
                    pred = pred.copy()
                    pred, adjusted, adj_amount = apply_constraint(pred, obases["pres_stock"], ogroups)
                else:
                    adjusted = np.zeros(len(pred), dtype=bool); adj_amount = np.zeros(len(pred))
                rounded = round_pred(pred, rnd, ogroups, odir_order)
                combo = pd.DataFrame({
                    "evaluation_id": eid, "evaluation_type": etype, "predictor_method": mname, "rounding_method": rnd,
                    "date": valid["date"].to_numpy(), "hour": valid["hour"].to_numpy(), "hour_of_day": valid["hour_of_day"].to_numpy(),
                    "dayofweek": valid["dayofweek"].to_numpy(), "day_type": valid["day_type"].to_numpy(),
                    "source_region": osrc, "target_region": otgt, "task_key": otask,
                    "audit_quality_regime": valid["audit_quality_regime"].to_numpy(), "period": valid["period"].to_numpy(),
                    "score_available": valid["score_available"].to_numpy(),
                    "y_true": np.where(valid["score_available"], valid["y_true_keep"], np.nan),
                    "selected_tau": rtau, "selected_smoother": rsm, "selected_scale": rscale,
                    "predicted_source_present": obases["pres_stock"] if present else np.nan,
                    "constraint_adjusted": adjusted, "constraint_adjustment": adj_amount,
                    "pred_float_before_scale": pred_before, "pred_float": pred, "pred_rounded": rounded,
                    "source_rank": valid["source_region"].map(REGION_RANK).to_numpy(),
                    "target_rank": valid["target_region"].map(REGION_RANK).to_numpy(),
                })
                combo["error_float"] = combo["pred_float"] - combo["y_true"]
                combo["error_rounded"] = combo["pred_rounded"] - combo["y_true"]
                assert np.isfinite(combo["pred_float"]).all() and (combo["pred_float"] >= 0).all()
                pred_frames.append(combo)

                sc = combo[combo["score_available"]]
                er = sc["error_rounded"].astype(float).to_numpy()
                ef = sc["error_float"].astype(float).to_numpy()
                frow = {
                    "evaluation_id": eid, "evaluation_type": etype, "predictor_method": mname, "rounding_method": rnd,
                    "n_prediction_rows": int(len(combo)), "n_scored_rows": int(len(sc)),
                    "sse_float": float((ef ** 2).sum()), "sse_rounded": float((er ** 2).sum()),
                    "mse_rounded": float((er ** 2).mean()), "mae_rounded": float(np.abs(er).mean()),
                    "mean_bias_rounded": float(er.mean()),
                    "actual_total_scored": float(sc["y_true"].sum()), "predicted_total_scored": float(sc["pred_rounded"].sum()),
                    "constraint_adjustment_count": int(sc["constraint_adjusted"].sum()), "max_constraint_adjustment": float(sc["constraint_adjustment"].max() if len(sc) else 0.0),
                }
                for (s, t) in DIRS:
                    m = (sc["source_region"] == s) & (sc["target_region"] == t)
                    frow[f"{s}_to_{t}_sse"] = float((sc.loc[m, "error_rounded"].astype(float) ** 2).sum())
                fold_rows.append(frow)

                for (s, t) in DIRS:
                    d = sc[(sc["source_region"] == s) & (sc["target_region"] == t)]
                    erd = d["error_rounded"].astype(float).to_numpy()
                    efd = d["error_float"].astype(float).to_numpy()
                    dir_rows.append({
                        "evaluation_id": eid, "evaluation_type": etype, "predictor_method": mname, "rounding_method": rnd,
                        "task_key": f"{s}->{t}", "n_scored_rows": int(len(d)), "actual_total": float(d["y_true"].sum()),
                        "predicted_total": float(d["pred_rounded"].sum()), "sse_float": float((efd ** 2).sum()),
                        "sse_rounded": float((erd ** 2).sum()), "mse_rounded": float((erd ** 2).mean()) if len(erd) else np.nan,
                        "mae_rounded": float(np.abs(erd).mean()) if len(erd) else np.nan, "mean_bias_rounded": float(erd.mean()) if len(erd) else np.nan,
                        "zero_true_rate": float((d["y_true"] == 0).mean()) if len(d) else np.nan,
                        "zero_prediction_rate": float((d["pred_rounded"] == 0).mean()) if len(d) else np.nan,
                    })

    preds = pd.concat(pred_frames, ignore_index=True)
    preds["_e"] = preds["evaluation_id"].map({s[0]: i for i, s in enumerate(scenarios)})
    preds["_p"] = preds["predictor_method"].map(PRED_ORDER)
    preds["_r"] = preds["rounding_method"].map({r: i for i, r in enumerate(ROUNDINGS)})
    preds = preds.sort_values(["_e", "_p", "_r", "hour", "source_rank", "target_rank"]).reset_index(drop=True)
    assert len(preds) == 214272, f"preds {len(preds)} != 214272"
    unscore = preds[~preds["score_available"]]
    assert unscore["y_true"].isna().all() and unscore["error_rounded"].isna().all()
    fold_scores = pd.DataFrame(fold_rows)
    direction_scores = pd.DataFrame(dir_rows)
    inner_def = pd.DataFrame(inner_def_rows)
    hp = pd.DataFrame(hp_rows)
    print(f"preds {len(preds)}; inner_def {len(inner_def)}; hp {len(hp)}")
    assert len(inner_def) == 63
    assert len(hp) == 3330, f"hp {len(hp)} != 3330"

    # consistency vs step9
    s9 = pd.read_csv(ROOT / STEP09_PREDS, encoding="utf-8-sig")
    s9_map = {
        "pair_hour_mean": "pair_hour_mean",
        "pair_hour_fixed085": "pair_hour_mean_x085",
        "pair_recent14_hour": "pair_recent14_hour",
        "present_shrunk20": "present_hour_prob_shrunk20",
    }
    max_diff = 0.0
    for my, s9m in s9_map.items():
        mine = preds[(preds["predictor_method"] == my) & (preds["rounding_method"] == "independent")].copy()
        mine["date"] = pd.to_datetime(mine["date"]).dt.strftime("%Y-%m-%d")
        ref = s9[s9["method"] == s9m].copy()
        ref["date"] = pd.to_datetime(ref["date"]).dt.strftime("%Y-%m-%d")
        ref_s = ref.set_index(["evaluation_id", "date", "hour_of_day", "source_region", "target_region"])["pred_float"]
        for _, r in mine.iterrows():
            key = (r["evaluation_id"], r["date"], int(r["hour_of_day"]), r["source_region"], r["target_region"])
            rv = ref_s.get(key)
            if rv is not None:
                max_diff = max(max_diff, abs(r["pred_float"] - rv))
    assert max_diff < 1e-10, f"step9 consistency {max_diff}"
    print(f"consistency vs step9 (4 methods): max abs diff = {max_diff:.3e}")

    # paired rounding assertion (per method x cell)
    paired = preds[preds["rounding_method"] == "paired"]
    for (eid, mname, date, hour, sr), g in paired.groupby(["evaluation_id", "predictor_method", "date", "hour", "source_region"]):
        f = g["pred_float"].to_numpy()
        s = g["pred_rounded"].to_numpy()
        assert (g["pred_rounded"] >= 0).all()
        assert abs(s.sum() - round(f.sum())) <= 1e-9, f"paired sum violation {eid}/{mname}"

    write_outputs(out_dir, preds, fold_scores, direction_scores, inner_def, hp, scenarios)
    print("\nDone.")


def _select_idx(method, scored):
    """Return index into scored of the best candidate, with deterministic tie rules."""
    def tie_key(i):
        cand, sse = scored[i][0], scored[i][1]
        typ, tau, sm, scale = cand
        scale_dist = abs(scale - 1.0) if scale is not None else 0.0
        if method == "pair_hour_hier_tau_cv":
            return (sse, -tau)  # smaller sse, then larger tau
        if method == "pair_hour_hier_tau_scale_cv":
            return (sse, scale_dist, -tau)
        if method == "pair_hour_circular_cv":
            return (sse, SMOOTHER_RANK[sm])
        if method == "pair_hour_circular_scale_cv":
            return (sse, scale_dist, SMOOTHER_RANK[sm])
        if method == "present_shrunk20_scale_cv":
            return (sse, scale_dist)
        return (sse,)
    return min(range(len(scored)), key=tie_key)


def _materialize_base(base, mname, sel):
    if mname in ("pair_hour_hier_tau_cv", "pair_hour_hier_tau_scale_cv"):
        tau = sel[1]  # ("tau"/"tascale", tau, sm, scale)
        return base[tau].copy()
    if mname in ("pair_hour_circular_cv", "pair_hour_circular_scale_cv"):
        sm = sel[2]
        return base[sm].copy()
    if mname in ("present_shrunk20_scale_cv",):
        return base.copy()
    if mname in ("pair_hour_direction_scale_cv", "present_shrunk20_direction_scale_cv"):
        return base.copy()
    return base.copy()


def _resolve_scale(mname, sel, task_arr):
    # returns (scale_arr, rtau, rsm, rscale)
    if mname in ("pair_hour_mean", "pair_recent14_hour", "present_shrunk20"):
        return np.ones(len(task_arr)), np.nan, "", np.nan
    if mname in ("pair_hour_fixed085", "pair_recent14_fixed085"):
        return np.full(len(task_arr), 0.85), np.nan, "", 0.85
    if mname == "pair_hour_hier_tau_cv":
        return np.ones(len(task_arr)), sel[1], "", np.nan
    if mname == "pair_hour_hier_tau_scale_cv":
        return np.full(len(task_arr), sel[3]), sel[1], "", sel[3]
    if mname == "pair_hour_circular_cv":
        return np.ones(len(task_arr)), np.nan, sel[2], np.nan
    if mname == "pair_hour_circular_scale_cv":
        return np.full(len(task_arr), sel[3]), np.nan, sel[2], sel[3]
    if mname == "present_shrunk20_scale_cv":
        return np.full(len(task_arr), sel[3]), np.nan, "", sel[3]
    if mname in ("pair_hour_direction_scale_cv", "present_shrunk20_direction_scale_cv"):
        dir_sel = sel[1]
        arr = np.array([dir_sel[tk] for tk in task_arr])
        return arr, np.nan, "", np.nan
    return np.ones(len(task_arr)), np.nan, "", np.nan


def write_outputs(out_dir, preds, fold_scores, direction_scores, inner_def, hp, scenarios):
    # method_definitions
    p_def = out_dir / "method_definitions.csv"
    p_indef = out_dir / "inner_validation_definitions.csv"
    p_hp = out_dir / "hyperparameter_selection.csv"
    p_pred = out_dir / "b_model_predictions.csv"
    p_fs = out_dir / "b_fold_scores.csv"
    p_ds = out_dir / "b_direction_scores.csv"
    p_rc = out_dir / "b_rounding_comparison.csv"
    p_dec = out_dir / "decision_table.csv"
    p_sum = out_dir / "summary.md"

    md_rows = []
    for meta in PREDICTORS:
        mname, family, window, hier, circ, present, needs_sel = meta
        uses_global = mname in ("pair_hour_fixed085", "pair_recent14_fixed085", "pair_hour_hier_tau_scale_cv",
                                "pair_hour_circular_scale_cv", "present_shrunk20_scale_cv")
        uses_dir = mname in ("pair_hour_direction_scale_cv", "present_shrunk20_direction_scale_cv")
        for rnd in ROUNDINGS:
            md_rows.append({"predictor_method": mname, "rounding_method": rnd, "model_family": family, "base_window": window,
                            "uses_hour_hierarchy": hier, "uses_circular_smoothing": circ, "uses_global_scale": uses_global,
                            "uses_direction_scale": uses_dir, "uses_present_structure": present,
                            "requires_inner_selection": needs_sel, "deployable": True})
    pd.DataFrame(md_rows)[DEF_COLS].to_csv(p_def, index=False, encoding="utf-8-sig")
    inner_def[INNER_DEF_COLS].to_csv(p_indef, index=False, encoding="utf-8-sig")

    hp_out = hp.copy()
    hp_out["_e"] = hp_out["evaluation_id"].map({s[0]: i for i, s in enumerate(scenarios)})
    hp_out["_p"] = hp_out["predictor_method"].map(PRED_ORDER)
    hp_out["_r"] = hp_out["rounding_method"].map({r: i for i, r in enumerate(ROUNDINGS)})
    hp_out = hp_out.sort_values(["_e", "_p", "_r", "task_key", "tau", "smoother", "scale"])
    hp_out[HP_COLS].to_csv(p_hp, index=False, encoding="utf-8-sig")
    # assert exactly one selected per (eval,rnd,method,task)
    sel_check = hp.groupby(["evaluation_id", "rounding_method", "predictor_method", "task_key"])["selected"].sum()
    assert (sel_check == 1).all(), f"selection count != 1: {sel_check[sel_check!=1]}"

    pred_out = preds[PRED_OUT_COLS].copy()
    pred_out["hour"] = pd.to_datetime(pred_out["hour"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    pred_out["date"] = pd.to_datetime(pred_out["date"]).dt.strftime("%Y-%m-%d")
    pred_out.to_csv(p_pred, index=False, encoding="utf-8-sig")

    fs_out = fold_scores.sort_values(["evaluation_id", "predictor_method", "rounding_method"])
    fs_out[FOLD_COLS].to_csv(p_fs, index=False, encoding="utf-8-sig")
    ds_out = direction_scores.copy()
    ds_out["_e"] = ds_out["evaluation_id"].map({s[0]: i for i, s in enumerate(scenarios)})
    ds_out["_p"] = ds_out["predictor_method"].map(PRED_ORDER)
    ds_out["_r"] = ds_out["rounding_method"].map({r: i for i, r in enumerate(ROUNDINGS)})
    ds_out["_d"] = ds_out["task_key"].map(DIR_RANK)
    ds_out.sort_values(["_e", "_p", "_r", "_d"])[DIR_COLS].to_csv(p_ds, index=False, encoding="utf-8-sig")

    # rounding comparison
    rc_rows = []
    for meta in PREDICTORS:
        mname = meta[0]
        row = {"predictor_method": mname}
        rolling = fold_scores[(fold_scores["evaluation_type"] == "rolling_7d") & (fold_scores["predictor_method"] == mname)]
        fa = fold_scores[(fold_scores["evaluation_id"] == "final_analog") & (fold_scores["predictor_method"] == mname)]
        for rnd in ROUNDINGS:
            rsub = rolling[rolling["rounding_method"] == rnd]["mse_rounded"].to_numpy(float)
            fsub = fa[fa["rounding_method"] == rnd]["sse_rounded"].to_numpy(float)
            row[f"{rnd}_rolling_mean_mse"] = float(np.mean(rsub))
            row[f"{rnd}_rolling_median_mse"] = float(np.median(rsub))
            row[f"{rnd}_rolling_max_mse"] = float(np.max(rsub))
            row[f"{rnd}_final_analog_sse"] = float(fsub[0]) if len(fsub) else np.nan
        # per-fold paired vs independent winner
        pw = rolling.pivot_table(index="evaluation_id", columns="rounding_method", values="sse_rounded")
        paired_better = int((pw["paired"] < pw["independent"]).sum()) if "paired" in pw and "independent" in pw else 0
        ind_better = int((pw["independent"] < pw["paired"]).sum()) if "paired" in pw and "independent" in pw else 0
        tie = int(((pw["paired"] == pw["independent"])).sum()) if "paired" in pw and "independent" in pw else 0
        row["paired_better_fold_count"] = paired_better
        row["independent_better_fold_count"] = ind_better
        row["tie_fold_count"] = tie
        rc_rows.append(row)
    rc = pd.DataFrame(rc_rows)
    rc[ROUND_CMP_COLS].to_csv(p_rc, index=False, encoding="utf-8-sig")

    # decision table
    dec_rows = []
    fa_set = fold_scores[fold_scores["evaluation_id"] == "final_analog"]
    fa_rank = fa_set.sort_values("sse_rounded")  # rank across 24 variants
    rank_map = {r["predictor_method"] + "|" + r["rounding_method"]: i + 1 for i, (_, r) in enumerate(fa_set.sort_values("sse_rounded").iterrows())}
    rolling = fold_scores[fold_scores["evaluation_type"] == "rolling_7d"]
    # fold wins across 24 variants
    wp = rolling.pivot_table(index="evaluation_id", columns=["predictor_method", "rounding_method"], values="sse_rounded", aggfunc="first")
    minpf = wp.min(axis=1)
    winc = wp.eq(minpf, axis=0).sum(axis=0)
    for meta in PREDICTORS:
        mname = meta[0]
        for rnd in ROUNDINGS:
            rsub = rolling[(rolling["predictor_method"] == mname) & (rolling["rounding_method"] == rnd)]
            mse = rsub["mse_rounded"].to_numpy(float); sse = rsub["sse_rounded"].to_numpy(float)
            far = fa_set[(fa_set["predictor_method"] == mname) & (fa_set["rounding_method"] == rnd)].iloc[0]
            dec_rows.append({
                "predictor_method": mname, "rounding_method": rnd, "model_family": meta[1],
                "rolling_mean_mse": float(np.mean(mse)), "rolling_median_mse": float(np.median(mse)),
                "rolling_max_mse": float(np.max(mse)), "rolling_mean_sse": float(np.mean(sse)),
                "final_analog_sse": float(far["sse_rounded"]), "final_analog_mse": float(far["mse_rounded"]),
                "final_analog_mean_bias": float(far["mean_bias_rounded"]),
                "final_analog_rank": int(rank_map[mname + "|" + rnd]),
                "rolling_fold_win_count": int(winc.get((mname, rnd), 0)),
                "selected_scale_mode": "", "median_selected_global_scale": np.nan,
                "selected_tau_mode": "", "selected_smoother_mode": "",
                "core_to_near_total_sse": float(rsub["core_to_near_sse"].sum()), "core_to_outer_total_sse": float(rsub["core_to_outer_sse"].sum()),
                "near_to_core_total_sse": float(rsub["near_to_core_sse"].sum()), "near_to_outer_total_sse": float(rsub["near_to_outer_sse"].sum()),
                "outer_to_core_total_sse": float(rsub["outer_to_core_sse"].sum()), "outer_to_near_total_sse": float(rsub["outer_to_near_sse"].sum()),
            })
    decision = pd.DataFrame(dec_rows)
    decision[DECISION_COLS].to_csv(p_dec, index=False, encoding="utf-8-sig")

    write_summary(p_sum, decision, fold_scores, hp, rc)

    print("\n=== Output files ===")
    for path in [p_def, p_indef, p_hp, p_pred, p_fs, p_ds, p_rc, p_dec, p_sum]:
        if path.suffix == ".csv":
            rows = len(pd.read_csv(path, encoding="utf-8-sig"))
        else:
            rows = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {path.relative_to(ROOT)}  rows={rows}")
    print("\n=== final_analog top 10 (by sse_rounded) ===")
    fa_top = fa_set.sort_values("sse_rounded").head(10)
    for _, r in fa_top.iterrows():
        print(f"  {r['predictor_method']:38s} {r['rounding_method']:10s} sse={r['sse_rounded']:.0f} mse={r['mse_rounded']:.3f}")
    print("\n=== rolling median MSE top 10 ===")
    for _, r in decision.sort_values("rolling_median_mse").head(10).iterrows():
        print(f"  {r['predictor_method']:38s} {r['rounding_method']:10s} roll_med_mse={r['rolling_median_mse']:.3f} wins={int(r['rolling_fold_win_count'])}")


def write_summary(path, decision, fold_scores, hp, rc):
    L = []
    L.append("# Step 10 B Shrinkage and Calibration")
    L.append("")
    L.append("> Task B only. Nested inner-CV shrinkage/calibration/integerization. No Ridge/Poisson/trees/NN, no submission, no A.")
    L.append("")
    L.append(f"Run command: `{RUN_CMD}`")
    L.append("")

    L.append("## 1. Boundary-safe nested evaluation")
    L.append("")
    L.append("- 9 outer evaluation scenarios; 63 inner time-order folds (3..11 per scenario). Inner-train boundary 23:00 labels excluded; inner-validation day 23:00 is scoreable; outer-train last day is never an inner-validation day. Outer validation state never read. 2018-01-24 23:00 stays right-censored. Parameter selection uses integer-prediction SSE.")
    L.append("")

    L.append("## 2. Step 09 consistency checks")
    L.append("")
    L.append("- pair_hour_mean, pair_hour_fixed085, pair_recent14_hour, present_shrunk20 (all independent round) reproduce Step 09 within 1e-10. 214272 prediction rows complete. **PASS**")
    L.append("")

    L.append("## 3. Fixed 0.85 versus selected global scale")
    L.append("")
    dd = decision.set_index(["predictor_method", "rounding_method"])
    for m in ["pair_hour_mean", "pair_hour_fixed085", "pair_hour_hier_tau_scale_cv", "pair_hour_circular_scale_cv"]:
        r = dd.loc[(m, "independent")]
        L.append(f"- {m}: rolling med MSE {r['rolling_median_mse']:.3f}, final SSE {r['final_analog_sse']:.0f}.")
    # selected global scale distribution
    gsel = hp[(hp["predictor_method"].isin(["pair_hour_hier_tau_scale_cv", "pair_hour_circular_scale_cv", "present_shrunk20_scale_cv"])) & (hp["selected"])]
    if len(gsel):
        L.append(f"- Selected global scale distribution (independent): {gsel[gsel['rounding_method']=='independent']['scale'].value_counts().to_dict()}.")
    L.append("")

    L.append("## 4. Hour hierarchy")
    L.append("")
    for m in ["pair_hour_mean", "pair_hour_hier_tau_cv"]:
        r = dd.loc[(m, "independent")]
        L.append(f"- {m}: rolling med MSE {r['rolling_median_mse']:.3f}, final SSE {r['final_analog_sse']:.0f}.")
    tausel = hp[(hp["predictor_method"] == "pair_hour_hier_tau_cv") & (hp["selected"]) & (hp["rounding_method"] == "independent")]
    L.append(f"- Selected tau distribution: {tausel['tau'].value_counts().to_dict()}.")
    L.append("")

    L.append("## 5. Circular hour smoothing")
    L.append("")
    smsel = hp[(hp["predictor_method"] == "pair_hour_circular_cv") & (hp["selected"]) & (hp["rounding_method"] == "independent")]
    L.append(f"- Selected smoother distribution: {smsel['smoother'].value_counts().to_dict()}.")
    for m in ["pair_hour_mean", "pair_hour_circular_cv"]:
        r = dd.loc[(m, "independent")]
        L.append(f"- {m}: rolling med MSE {r['rolling_median_mse']:.3f}, final SSE {r['final_analog_sse']:.0f}.")
    L.append("")

    L.append("## 6. Global versus direction-specific scale")
    L.append("")
    for m in ["pair_hour_hier_tau_scale_cv", "pair_hour_direction_scale_cv", "present_shrunk20_scale_cv", "present_shrunk20_direction_scale_cv"]:
        r = dd.loc[(m, "independent")]
        L.append(f"- {m}: rolling med MSE {r['rolling_median_mse']:.3f}, rolling max MSE {r['rolling_max_mse']:.3f}, final SSE {r['final_analog_sse']:.0f}.")
    L.append("")

    L.append("## 7. Independent versus paired rounding")
    L.append("")
    L.append("| predictor | ind roll med MSE | paired roll med MSE | paired better folds | ind better folds | ties |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    rc_i = rc.set_index("predictor_method")
    for meta in PREDICTORS:
        r = rc_i.loc[meta[0]]
        L.append(f"| {meta[0]} | {r['independent_rolling_median_mse']:.3f} | {r['paired_rolling_median_mse']:.3f} | {int(r['paired_better_fold_count'])} | {int(r['independent_better_fold_count'])} | {int(r['tie_fold_count'])} |")
    L.append("")

    L.append("## 8. Direct hierarchy versus present structure")
    L.append("")
    L.append("- raw present and direction-hour-mean are algebraically close, so only shrinkage/calibration value is judged; linkable models are not discussed.")
    L.append("")

    L.append("## 9. Hyperparameter stability")
    L.append("")
    L.append(f"- tau selections (hier): {tausel['tau'].value_counts().to_dict()}.")
    L.append(f"- smoother selections: {smsel['smoother'].value_counts().to_dict()}.")
    L.append("")

    L.append("## 10. Decision for final B candidates")
    L.append("")
    fa = fold_scores[fold_scores["evaluation_id"] == "final_analog"].sort_values("sse_rounded")
    L.append("- Top-5 final_analog variants:")
    for _, r in fa.head(5).iterrows():
        L.append(f"  - {r['predictor_method']} / {r['rounding_method']}: SSE={r['sse_rounded']:.0f}")
    L.append("- At most 3 B candidates are retained, judged across rolling folds + final_analog (not a single window).")
    L.append("- When training on all 2018-01-01..01-24 for the real 01-25..01-31 forecast, 2018-01-24 23:00 stays right-censored (no 01-25 00:00 state), and no validation AIS state is used.")
    L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
