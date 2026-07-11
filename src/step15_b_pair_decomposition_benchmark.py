"""Step 15 (B2) — Three layer-pair decomposition benchmark for Task B.

Predicts each bidirectional layer-pair (core<->near, near<->outer, core<->outer)
in three layers: daily pair total T, 24h share p, direction ratio r (Beta
shrinkage). Compares 7 model variants (circular3 baseline; fixed; global-CV;
by-pair-CV; each x independent / pair-hierarchical rounding) under 4 evaluation
views. No hurdle, no net-flow shrinkage, no ensembling, no vessel scaling.

Boundary: pair model uses only COMPLETE training dates (outer train_end-1;
inner valid_date-2); circular3 baseline reuses the step10 observable-hour rule.
2018-01-24 23:00 right-censored.

Outputs (under outputs/step15_b_pair_decomposition_benchmark/):
  1. method_definitions.csv (7)
  2. inner_validation_definitions.csv (63)
  3. hyperparameter_selection.csv (6912)
  4. b_model_predictions.csv (62496)
  5. b_fold_scores.csv (63)
  6. b_pair_scores.csv (189)
  7. b_direction_scores.csv (378)
  8. b_horizon_scores.csv (434)
  9. b_nonoverlap_scores.csv (21)
  10. decision_table.csv (7)
  11. summary.md
"""

from __future__ import annotations

import itertools
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

from step10_b_shrinkage_calibration import (  # noqa: E402
    DIRS, DIR_NAMES, DIR_RANK, REGION_RANK, SMOOTHERS, UNSCORABLE,
    cell_groups_of, compute_aggregates, compute_bases, inner_folds,
    round_pred, score_candidate, smooth_circular,
)

PF_PATH = Path("outputs/step08_b_state_transition_audit/b_hourly_pair_flow.csv")
SS_PATH = Path("outputs/step08_b_state_transition_audit/b_hourly_source_state.csv")
STEP10_PREDS = Path("outputs/step10_b_shrinkage_calibration/b_model_predictions.csv")
OUT_REL = Path("outputs/step15_b_pair_decomposition_benchmark")

GLOBAL_START = pd.Timestamp("2018-01-01")
PAIRS = [
    ("core_near", "core", "near", "core->near", "near->core"),
    ("near_outer", "near", "outer", "near->outer", "outer->near"),
    ("core_outer", "core", "outer", "core->outer", "outer->core"),
]
PAIR_ORDER = {p[0]: i for i, p in enumerate(PAIRS)}
# map (source,target) -> (pair_name, role)
PAIR_OF = {}
for name, a, b, dab, dba in PAIRS:
    PAIR_OF[(a, b)] = (name, "forward")
    PAIR_OF[(b, a)] = (name, "reverse")
DAILY_METHODS = ["long_mean", "recent14_mean", "blend25", "median"]
PROFILE_METHODS = ["day_equal_raw", "day_equal_triangular3", "flow_weighted_raw", "flow_weighted_triangular3"]
TAUS = [0, 5, 10, 20, 50, 100]
DAILY_RANK = {m: i for i, m in enumerate(DAILY_METHODS)}
PROFILE_RANK = {m: i for i, m in enumerate(PROFILE_METHODS)}

# 7 variants: name, family, scope, rounding, (fixed params or None)
VARIANTS = [
    ("circular3_independent", "direct_baseline", "fixed", "independent", None),
    ("pair_default_independent", "pair_decomposition", "fixed", "independent", ("long_mean", "day_equal_triangular3", 20)),
    ("pair_default_hierarchical", "pair_decomposition", "fixed", "paired", ("long_mean", "day_equal_triangular3", 20)),
    ("pair_global_cv_independent", "pair_decomposition", "global", "independent", None),
    ("pair_global_cv_hierarchical", "pair_decomposition", "global", "paired", None),
    ("pair_bypair_cv_independent", "pair_decomposition", "by_pair", "independent", None),
    ("pair_bypair_cv_hierarchical", "pair_decomposition", "by_pair", "paired", None),
]
VARIANT_ORDER = {v[0]: i for i, v in enumerate(VARIANTS)}
ROUND_LABEL = {"independent": "independent", "paired": "pair_hierarchical"}

PRED_COLS = ["evaluation_id", "evaluation_type", "method", "date", "hour", "hour_of_day", "forecast_horizon",
             "source_region", "target_region", "task_key", "pair", "direction_role", "score_available", "y_true",
             "selected_daily_total_method", "selected_profile_method", "selected_tau",
             "predicted_pair_daily_total", "predicted_pair_hour_share", "predicted_pair_hour_total_float",
             "predicted_direction_ratio", "pred_float", "pred_rounded", "error_float", "error_rounded"]
FOLD_COLS = ["evaluation_id", "evaluation_type", "method", "n_prediction_rows", "n_scored_rows", "sse_float",
             "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded", "actual_total", "predicted_total", "rolling_rank"]
PAIRSCORE_COLS = ["evaluation_id", "evaluation_type", "method", "pair", "n_scored_rows", "actual_total", "predicted_total",
                  "sse_float", "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded", "zero_true_rate", "zero_prediction_rate"]
DIRSCORE_COLS = ["evaluation_id", "evaluation_type", "method", "task_key", "pair", "direction_role", "n_scored_rows",
                 "actual_total", "predicted_total", "sse_float", "sse_rounded", "mse_rounded", "mae_rounded",
                 "mean_bias_rounded", "zero_true_rate", "zero_prediction_rate"]
HORIZON_COLS = ["evaluation_id", "evaluation_type", "method", "forecast_horizon", "n_scored_rows", "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded"]
NONOVERLAP_COLS = ["method", "block", "target_start", "target_end", "n_scored_rows", "sse_rounded", "mse_rounded", "mae_rounded", "mean_bias_rounded"]
DECISION_COLS = ["method", "model_family", "parameter_scope", "rounding_method", "rolling_mean_mse", "rolling_median_mse",
                 "rolling_max_mse", "rolling_mean_sse", "rolling_fold_win_count", "final_analog_sse", "final_analog_mse",
                 "nonoverlap_combined_mse", "horizon_1_mse", "horizon_7_mse",
                 "core_near_total_sse", "near_outer_total_sse", "core_outer_total_sse",
                 "core_to_near_total_sse", "core_to_outer_total_sse", "near_to_core_total_sse", "near_to_outer_total_sse",
                 "outer_to_core_total_sse", "outer_to_near_total_sse"]
METHOD_DEF_COLS = ["method", "model_family", "parameter_scope", "rounding_method", "daily_total_method",
                   "profile_method", "tau", "requires_inner_selection", "deployable"]
INNER_DEF_COLS = ["evaluation_id", "inner_fold_id", "inner_train_start", "inner_train_end", "last_complete_inner_train_date",
                  "inner_valid_date", "n_complete_train_days", "n_inner_scored_rows"]
HP_COLS = ["evaluation_id", "evaluation_type", "rounding_method", "parameter_scope", "pair", "daily_total_method",
           "profile_method", "tau", "n_inner_splits", "inner_validation_dates", "inner_sse_rounded", "inner_sse_float",
           "inner_mae_rounded", "mean_bias_rounded", "selected"]

RUN_CMD = "python " + " ".join(sys.argv)


def build_scenarios():
    sc = []
    for k in range(8):
        sc.append((f"fold_{k + 1:02d}", "rolling_7d", GLOBAL_START, pd.Timestamp(f"2018-01-{10 + k:02d}"),
                   pd.Timestamp(f"2018-01-{11 + k:02d}"), pd.Timestamp(f"2018-01-{17 + k:02d}")))
    sc.append(("final_analog", "final_analog", GLOBAL_START, pd.Timestamp("2018-01-18"),
               pd.Timestamp("2018-01-19"), pd.Timestamp("2018-01-24")))
    return sc


def smooth3(vec):
    out = np.zeros(24)
    for h in range(24):
        out[h] = (vec[(h - 1) % 24] + 2 * vec[h] + vec[(h + 1) % 24]) / 4
    return np.clip(out, 0, None)


def pair_base(pf, complete_dates, pair):
    """Return T_hat(4 methods), p_hat(4 methods, 24-vec), r_hat(6 tau, 24-vec), and T array, for one pair."""
    name, a, b, dab, dba = pair
    sub = pf[pf["date"].isin(complete_dates) & pf["task_key"].isin([dab, dba])]
    fwd_df = sub[sub["task_key"] == dab].pivot_table(index="date", columns="hour_of_day", values="y")
    rev_df = sub[sub["task_key"] == dba].pivot_table(index="date", columns="hour_of_day", values="y")
    fwd_df = fwd_df.reindex(index=complete_dates, columns=range(24)).fillna(0)
    rev_df = rev_df.reindex(index=complete_dates, columns=range(24)).fillna(0)
    fwd = fwd_df.to_numpy(float)
    rev = rev_df.to_numpy(float)
    X = fwd + rev
    T = X.sum(axis=1)
    X_h = X.sum(axis=0)
    Yfwd_h = fwd.sum(axis=0)
    sum_X = X_h.sum()
    r_global = float(Yfwd_h.sum() / sum_X) if sum_X > 0 else 0.5

    # daily totals
    long_mean = float(T.mean()) if len(T) else 0.0
    r14 = T[-14:]
    recent14 = float(r14.mean()) if len(r14) else long_mean
    blend25 = 0.75 * long_mean + 0.25 * recent14
    med = float(np.median(T)) if len(T) else 0.0
    T_hat = {"long_mean": long_mean, "recent14_mean": recent14, "blend25": blend25, "median": med}

    # profiles
    with np.errstate(divide="ignore", invalid="ignore"):
        p_day = np.where((T[:, None] > 0), X / np.where(T[:, None] > 0, T[:, None], 1.0), 0.0)
    mask = T > 0
    de_raw = p_day[mask].mean(axis=0) if mask.any() else np.full(24, 1 / 24)
    s = de_raw.sum()
    de_raw = de_raw / s if s > 0 else np.full(24, 1 / 24)
    de_tri = smooth3(de_raw); de_tri = de_tri / de_tri.sum() if de_tri.sum() > 0 else np.full(24, 1 / 24)
    fw_raw = X_h / sum_X if sum_X > 0 else np.full(24, 1 / 24)
    fw_tri = smooth3(fw_raw); fw_tri = fw_tri / fw_tri.sum() if fw_tri.sum() > 0 else np.full(24, 1 / 24)
    p_hat = {"day_equal_raw": de_raw, "day_equal_triangular3": de_tri, "flow_weighted_raw": fw_raw, "flow_weighted_triangular3": fw_tri}

    # direction ratio
    r_hat = {}
    for tau in TAUS:
        if tau == 0:
            r = np.where(X_h > 0, Yfwd_h / np.where(X_h > 0, X_h, 1.0), r_global)
        else:
            r = (Yfwd_h + tau * r_global) / (X_h + tau)
        r_hat[tau] = np.clip(r, 0, 1)
    return {"T_hat": T_hat, "p_hat": p_hat, "r_hat": r_hat, "r_global": r_global}


def predict_pair_float(pb, daily, profile, tau):
    T = pb["T_hat"][daily]
    p = pb["p_hat"][profile]
    r = pb["r_hat"][tau]
    fwd = T * p * r
    rev = T * p * (1 - r)
    return fwd, rev, T, p, r


def alloc_hours(K, p):
    fa = K * p
    base = np.floor(fa).astype(int)
    rem = K - int(base.sum())
    if rem > 0:
        frac = fa - base
        order = np.lexsort((np.arange(24), -fa, -frac))
        for j in range(rem):
            base[order[j]] += 1
    return base


def round_pair(pb, daily, profile, tau, rounding):
    fwd_f, rev_f, T, p, r = predict_pair_float(pb, daily, profile, tau)
    if rounding == "independent":
        fwd_i = np.clip(np.rint(fwd_f), 0, None).astype(int)
        rev_i = np.clip(np.rint(rev_f), 0, None).astype(int)
    else:
        K = int(round(T))
        alloc = alloc_hours(K, p)
        assert alloc.sum() == K
        fwd_i = np.clip(np.rint(alloc * r), 0, alloc).astype(int)
        rev_i = alloc - fwd_i
        assert (fwd_i + rev_i == alloc).all()
    return fwd_i, rev_i, fwd_f, rev_f


def valid_y_per_pair(pf, date, pair):
    name, a, b, dab, dba = pair
    sub = pf[(pf["date"] == date) & (pf["task_key"].isin([dab, dba]))]
    fwd = sub[sub["task_key"] == dab].set_index("hour_of_day")["y"].reindex(range(24)).fillna(0).to_numpy(float)
    rev = sub[sub["task_key"] == dba].set_index("hour_of_day")["y"].reindex(range(24)).fillna(0).to_numpy(float)
    return fwd, rev


def score_candidate_fold(pbs, y_pairs, daily, profile, tau, rounding):
    """Return per-pair SSE (list of 3) and total SSE for one candidate on one inner fold."""
    per_pair = []
    for q, pair in enumerate(PAIRS):
        fwd_i, rev_i, _, _ = round_pair(pbs[q], daily, profile, tau, rounding)
        yf, yr = y_pairs[q]
        per_pair.append(float(((fwd_i - yf) ** 2).sum() + ((rev_i - yr) ** 2).sum()))
    return per_pair, sum(per_pair)


def select_global(scored):
    """scored: list of (daily,profile,tau, total_sse). Return best per tie rules."""
    def key(r):
        d, pr, t, sse = r
        return (sse, DAILY_RANK[d], PROFILE_RANK[pr], -t)
    return min(scored, key=key)


def select_pair(scored):
    def key(r):
        d, pr, t, sse = r
        return (sse, DAILY_RANK[d], PROFILE_RANK[pr], -t)
    return min(scored, key=key)


def circular3_predict(ss, pf, train_start, train_end, valid):
    """Reproduce step10 pair_hour_circular_cv / independent."""
    folds = inner_folds(train_start, train_end)
    inner_data = []
    for (vdate, its, ite) in folds:
        iagg = compute_aggregates(ss, pf, its, ite)
        iv = pf[pf["date"] == vdate].sort_values(["hour", "source_region", "target_region"]).reset_index(drop=True)
        ib = compute_bases(iagg, iv)
        inner_data.append((ib, cell_groups_of(iv), iv["dir_order"].to_numpy(), iv["y"].to_numpy(float), iv["task_key"].to_numpy()))
    scored = []
    for sm in SMOOTHERS:
        sse_sum = 0.0
        for (ib, ig, iod, iy, itask) in inner_data:
            base = ib["circ"][sm]
            sarr = np.ones(len(base))
            sse, _, _, _, _, _ = score_candidate(base, sarr, False, "independent", ib, ig, iod, iy, ib["pres_stock"])
            sse_sum += sse
        scored.append((sm, sse_sum))
    sel_sm = min(scored, key=lambda r: (r[1], SMOOTHERS.index(r[0])))[0]
    agg = compute_aggregates(ss, pf, train_start, train_end)
    bases = compute_bases(agg, valid)
    pred = bases["circ"][sel_sm]
    groups = cell_groups_of(valid)
    rounded = round_pred(pred, "independent", groups, valid["dir_order"].to_numpy())
    return pred, rounded, sel_sm


def main():
    out_dir = ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    pf = pd.read_csv(ROOT / PF_PATH, encoding="utf-8-sig")
    pf["hour"] = pd.to_datetime(pf["hour"]); pf["date"] = pd.to_datetime(pf["date"]).dt.normalize()
    pf["hour_of_day"] = pf["hour"].dt.hour; pf["dayofweek"] = pf["hour"].dt.dayofweek
    pf["dir_order"] = pf["task_key"].map(DIR_RANK)
    ss = pd.read_csv(ROOT / SS_PATH, encoding="utf-8-sig")
    ss["hour"] = pd.to_datetime(ss["hour"]); ss["date"] = pd.to_datetime(ss["date"]).dt.normalize(); ss["hour_of_day"] = ss["hour"].dt.hour

    scenarios = build_scenarios()
    s10 = pd.read_csv(ROOT / STEP10_PREDS, encoding="utf-8-sig")
    s10["date"] = pd.to_datetime(s10["date"]).dt.strftime("%Y-%m-%d")

    inner_def_rows = []
    pred_frames = []
    fold_rows = []
    pairscore_rows = []
    dirscore_rows = []
    horizon_rows = []
    hp_rows = []
    selected_params = {}  # (eval, variant_name) -> params

    for eid, etype, tstart, tend, vstart, vend in scenarios:
        # validation frame (6 directions)
        valid = pf[(pf["date"] >= vstart) & (pf["date"] <= vend)].copy()
        valid = valid.sort_values(["hour", "source_region", "target_region"]).reset_index(drop=True)
        valid["score_available"] = valid["hour"] != UNSCORABLE
        valid["y_keep"] = valid["y"]
        valid["pair"] = valid.apply(lambda r: PAIR_OF[(r["source_region"], r["target_region"])][0], axis=1)
        valid["direction_role"] = valid.apply(lambda r: PAIR_OF[(r["source_region"], r["target_region"])][1], axis=1)
        valid["forecast_horizon"] = (valid["date"] - vstart).dt.days + 1
        valid["pair_rank"] = valid["pair"].map(PAIR_ORDER)

        # complete outer train dates for pair model
        complete_outer = list(pd.date_range(tstart, tend - pd.Timedelta(days=1), freq="D").normalize())
        pbs_outer = [pair_base(pf, complete_outer, p) for p in PAIRS]

        # inner folds
        folds = inner_folds(tstart, tend)
        for fi, (vdate, its, ite) in enumerate(folds):
            last_complete = vdate - pd.Timedelta(days=2)
            n_complete = len(pd.date_range(tstart, last_complete, freq="D").normalize())
            inner_def_rows.append({"evaluation_id": eid, "inner_fold_id": fi + 1, "inner_train_start": f"{its:%Y-%m-%d}",
                                   "inner_train_end": f"{ite:%Y-%m-%d}", "last_complete_inner_train_date": f"{last_complete:%Y-%m-%d}",
                                   "inner_valid_date": f"{vdate:%Y-%m-%d}", "n_complete_train_days": int(n_complete), "n_inner_scored_rows": 144})
        # inner pair bases per fold (complete dates up to v-2)
        inner_fold_data = []
        for (vdate, its, ite) in folds:
            last_complete = vdate - pd.Timedelta(days=2)
            comp = list(pd.date_range(tstart, last_complete, freq="D").normalize())
            assert len(comp) >= 5, f"{eid} inner complete <5"
            pbs = [pair_base(pf, comp, p) for p in PAIRS]
            y_pairs = [valid_y_per_pair(pf, vdate, p) for p in PAIRS]
            inner_fold_data.append((vdate, pbs, y_pairs))
        inner_dates_str = ",".join(f"{v:%Y-%m-%d}" for v, _, _ in inner_fold_data)

        # ---- parameter selection (global + by_pair), per rounding ----
        # precompute per-fold per-candidate per-pair SSE
        cand_grid = [(d, pr, t) for d in DAILY_METHODS for pr in PROFILE_METHODS for t in TAUS]
        sel_cache = {}  # (scope, rounding, pair_or_all) -> params
        for rounding in ["independent", "paired"]:
            # aggregate per candidate: per-pair SSE and total
            agg_pair = {q: {c: 0.0 for c in cand_grid} for q in range(3)}
            agg_total = {c: 0.0 for c in cand_grid}
            for (vdate, pbs, y_pairs) in inner_fold_data:
                for c in cand_grid:
                    d, pr, t = c
                    pp, tot = score_candidate_fold(pbs, y_pairs, d, pr, t, rounding)
                    for q in range(3):
                        agg_pair[q][c] += pp[q]
                    agg_total[c] += tot
            # global selection
            scored_g = [(d, pr, t, agg_total[(d, pr, t)]) for (d, pr, t) in cand_grid]
            gsel = select_global(scored_g)
            sel_cache[("global", rounding, "all")] = gsel
            for (d, pr, t, sse) in scored_g:
                hp_rows.append({"evaluation_id": eid, "evaluation_type": etype, "rounding_method": rounding,
                                "parameter_scope": "global", "pair": "all", "daily_total_method": d, "profile_method": pr,
                                "tau": t, "n_inner_splits": len(folds), "inner_validation_dates": inner_dates_str,
                                "inner_sse_rounded": sse, "inner_sse_float": sse, "inner_mae_rounded": np.nan,
                                "mean_bias_rounded": np.nan, "selected": ((d, pr, t) == (gsel[0], gsel[1], gsel[2]))})
            # by_pair selection
            for q, pair in enumerate(PAIRS):
                scored_p = [(d, pr, t, agg_pair[q][(d, pr, t)]) for (d, pr, t) in cand_grid]
                psel = select_pair(scored_p)
                sel_cache[("by_pair", rounding, pair[0])] = psel
                for (d, pr, t, sse) in scored_p:
                    hp_rows.append({"evaluation_id": eid, "evaluation_type": etype, "rounding_method": rounding,
                                    "parameter_scope": "by_pair", "pair": pair[0], "daily_total_method": d, "profile_method": pr,
                                    "tau": t, "n_inner_splits": len(folds), "inner_validation_dates": inner_dates_str,
                                    "inner_sse_rounded": sse, "inner_sse_float": sse, "inner_mae_rounded": np.nan,
                                    "mean_bias_rounded": np.nan, "selected": ((d, pr, t) == (psel[0], psel[1], psel[2]))})

        # ---- predictions per variant ----
        variant_preds = {}
        for vname, family, scope, rounding, fixed in VARIANTS:
            rkey = "independent" if rounding == "independent" else "paired"
            if vname == "circular3_independent":
                pred, rounded, _ = circular3_predict(ss, pf, tstart, tend, valid)
                # build per-row (no pair decomposition fields)
                combo = build_combo(eid, etype, vname, valid, pred, rounded, None)
            else:
                # determine per-pair params
                pair_params = {}
                for pair in PAIRS:
                    if scope == "fixed":
                        d, pr, t = fixed
                    elif scope == "global":
                        g = sel_cache[("global", rkey, "all")]; d, pr, t = g[0], g[1], g[2]
                    else:  # by_pair
                        p = sel_cache[("by_pair", rkey, pair[0])]; d, pr, t = p[0], p[1], p[2]
                    pair_params[pair[0]] = (d, pr, t)
                pred, rounded, decomp = predict_variant(pbs_outer, pair_params, valid, rounding)
                selected_params[(eid, vname)] = (pair_params, rounding)
                combo = build_combo(eid, etype, vname, valid, pred, rounded, decomp)
            variant_preds[vname] = (pred, rounded, combo)
            pred_frames.append(combo)
            # fold score
            sc = combo[combo["score_available"]]
            fold_rows.append(score_block(eid, etype, vname, combo, sc))
            # pair scores
            for pair in PAIRS:
                sp = sc[sc["pair"] == pair[0]]
                pairscore_rows.append(pair_score_block(eid, etype, vname, pair[0], sp))
            # direction scores
            for (s, t) in DIRS:
                sd = sc[(sc["source_region"] == s) & (sc["target_region"] == t)]
                dirscore_rows.append(dir_score_block(eid, etype, vname, s, t, sd))
            # horizon scores
            for hz in sorted(sc["forecast_horizon"].unique()):
                sh = sc[sc["forecast_horizon"] == hz]
                er = sh["error_rounded"].astype(float).to_numpy()
                horizon_rows.append({"evaluation_id": eid, "evaluation_type": etype, "method": vname, "forecast_horizon": int(hz),
                                     "n_scored_rows": int(len(sh)), "sse_rounded": float((er ** 2).sum()),
                                     "mse_rounded": float((er ** 2).mean()) if len(er) else np.nan,
                                     "mae_rounded": float(np.abs(er).mean()) if len(er) else np.nan,
                                     "mean_bias_rounded": float(er.mean()) if len(er) else np.nan})

        # circular3 reproduction check
        c3 = variant_preds["circular3_independent"][2]
        ref = s10[(s10["evaluation_id"] == eid) & (s10["predictor_method"] == "pair_hour_circular_cv") & (s10["rounding_method"] == "independent")].copy()
        ref = ref.sort_values(["date", "hour_of_day", "source_region", "target_region"]).reset_index(drop=True)
        c3s = c3.sort_values(["date", "hour_of_day", "source_region", "target_region"]).reset_index(drop=True)
        c3s["date"] = pd.to_datetime(c3s["date"]).dt.strftime("%Y-%m-%d")
        # compare pred_float
        merged = ref[["date", "hour_of_day", "source_region", "target_region", "pred_float"]].merge(
            c3s[["date", "hour_of_day", "source_region", "target_region", "pred_float"]].rename(columns={"pred_float": "mine"}),
            on=["date", "hour_of_day", "source_region", "target_region"])
        md = float((merged["pred_float"] - merged["mine"]).abs().max())
        assert md < 1e-10, f"circular3 reproduction {eid}: {md}"

    preds = pd.concat(pred_frames, ignore_index=True)
    preds["_e"] = preds["evaluation_id"].map({s[0]: i for i, s in enumerate(scenarios)})
    preds["_m"] = preds["method"].map(VARIANT_ORDER)
    preds = preds.sort_values(["_e", "_m", "hour", "source_region", "target_region"]).reset_index(drop=True)
    assert len(preds) == 62496, f"preds {len(preds)} != 62496"

    hp = pd.DataFrame(hp_rows)
    assert len(hp) == 6912, f"hp {len(hp)} != 6912"
    # selected checks
    sg = hp[(hp["parameter_scope"] == "global")].groupby(["evaluation_id", "rounding_method"])["selected"].sum()
    assert (sg == 1).all()
    sb = hp[hp["parameter_scope"] == "by_pair"].groupby(["evaluation_id", "rounding_method", "pair"])["selected"].sum()
    assert (sb == 1).all()

    inner_def = pd.DataFrame(inner_def_rows)
    assert len(inner_def) == 63
    fold_scores = pd.DataFrame(fold_rows)
    pairscores = pd.DataFrame(pairscore_rows)
    dirscores = pd.DataFrame(dirscore_rows)
    horizons = pd.DataFrame(horizon_rows)

    # rolling rank
    rolling = fold_scores[fold_scores["evaluation_type"] == "rolling_7d"].copy()
    rolling["rolling_rank"] = rolling.groupby("evaluation_id")["sse_rounded"].rank(method="min").astype(int)
    fold_scores = fold_scores.drop(columns=["rolling_rank"]).merge(rolling[["evaluation_id", "method", "rolling_rank"]], on=["evaluation_id", "method"], how="left")
    fold_scores["rolling_rank"] = fold_scores["rolling_rank"].where(fold_scores["evaluation_type"] == "rolling_7d")

    # non-overlap
    nonoverlap_rows = build_nonoverlap(fold_scores, preds, scenarios)
    nonoverlap = pd.DataFrame(nonoverlap_rows)
    assert len(nonoverlap) == 21

    # decision table
    decision = build_decision(fold_scores, pairscores, dirscores, horizons, nonoverlap, scenarios)

    # write
    write_all(out_dir, preds, fold_scores, pairscores, dirscores, horizons, nonoverlap, decision, inner_def, hp, scenarios, selected_params)
    print(f"preds {len(preds)} | hp {len(hp)} | circular3 repro < 1e-10 PASS")
    print("\nDone.")


def predict_variant(pbs_outer, pair_params, valid, rounding):
    """Predict all 6 directions for the validation frame using per-pair params."""
    n = len(valid)
    pred = np.zeros(n)
    rounded = np.zeros(n, dtype=int)
    decomp = {k: np.full(n, np.nan) for k in ["daily", "profile_share", "hour_total_float", "ratio"]}
    decomp["sel_daily"] = np.array([None] * n, dtype=object)
    decomp["sel_profile"] = np.array([None] * n, dtype=object)
    decomp["sel_tau"] = np.full(n, np.nan)
    for q, pair in enumerate(PAIRS):
        d, pr, t = pair_params[pair[0]]
        pb = pbs_outer[q]
        f_i, r_i, f_f, r_f = round_pair_full(pb, d, pr, t, rounding)
        T_v = pb["T_hat"][d]; p_v = pb["p_hat"][pr]; r_v = pb["r_hat"][t]
        mask = (valid["pair"] == pair[0]).to_numpy()
        idx = np.where(mask)[0]
        hours = valid.loc[mask, "hour_of_day"].to_numpy().astype(int)
        roles = valid.loc[mask, "direction_role"].to_numpy()
        for k_i, ii in enumerate(idx):
            h = hours[k_i]
            if roles[k_i] == "forward":
                pred[ii] = f_f[h]; rounded[ii] = f_i[h]
            else:
                pred[ii] = r_f[h]; rounded[ii] = r_i[h]
            decomp["daily"][ii] = T_v
            decomp["profile_share"][ii] = p_v[h]
            decomp["hour_total_float"][ii] = T_v * p_v[h]
            decomp["ratio"][ii] = r_v[h] if roles[k_i] == "forward" else (1 - r_v[h])
            decomp["sel_daily"][ii] = d; decomp["sel_profile"][ii] = pr; decomp["sel_tau"][ii] = t
    return pred, rounded, decomp


def round_pair_full(pb, daily, profile, tau, rounding):
    fwd_f, rev_f, T, p, r = predict_pair_float(pb, daily, profile, tau)
    if rounding == "independent":
        fwd_i = np.clip(np.rint(fwd_f), 0, None).astype(int)
        rev_i = np.clip(np.rint(rev_f), 0, None).astype(int)
    else:
        K = int(round(T))
        alloc = alloc_hours(K, p)
        assert alloc.sum() == K, f"alloc sum {alloc.sum()} != {K}"
        fwd_i = np.clip(np.rint(alloc * r), 0, alloc).astype(int)
        rev_i = alloc - fwd_i
        assert (fwd_i + rev_i == alloc).all()
    return fwd_i, rev_i, fwd_f, rev_f


def build_combo(eid, etype, vname, valid, pred, rounded, decomp):
    n = len(valid)
    y_true = np.where(valid["score_available"], valid["y_keep"], np.nan)
    combo = pd.DataFrame({
        "evaluation_id": eid, "evaluation_type": etype, "method": vname,
        "date": valid["date"].to_numpy(), "hour": valid["hour"].to_numpy(), "hour_of_day": valid["hour_of_day"].to_numpy(),
        "forecast_horizon": valid["forecast_horizon"].to_numpy(),
        "source_region": valid["source_region"].to_numpy(), "target_region": valid["target_region"].to_numpy(),
        "task_key": valid["task_key"].to_numpy(), "pair": valid["pair"].to_numpy(), "direction_role": valid["direction_role"].to_numpy(),
        "score_available": valid["score_available"].to_numpy(), "y_true": y_true,
        "pred_float": pred, "pred_rounded": rounded,
    })
    if decomp is not None:
        combo["predicted_pair_daily_total"] = decomp["daily"]
        combo["predicted_pair_hour_share"] = decomp["profile_share"]
        combo["predicted_pair_hour_total_float"] = decomp["hour_total_float"]
        combo["predicted_direction_ratio"] = decomp["ratio"]
        combo["selected_daily_total_method"] = decomp["sel_daily"]
        combo["selected_profile_method"] = decomp["sel_profile"]
        combo["selected_tau"] = decomp["sel_tau"]
    else:
        for c in ["predicted_pair_daily_total", "predicted_pair_hour_share", "predicted_pair_hour_total_float", "predicted_direction_ratio",
                  "selected_daily_total_method", "selected_profile_method", "selected_tau"]:
            combo[c] = np.nan
    combo["error_float"] = combo["pred_float"] - combo["y_true"]
    combo["error_rounded"] = combo["pred_rounded"] - combo["y_true"]
    return combo


def score_block(eid, etype, vname, combo, sc):
    er = sc["error_rounded"].astype(float).to_numpy()
    ef = sc["error_float"].astype(float).to_numpy()
    return {"evaluation_id": eid, "evaluation_type": etype, "method": vname, "n_prediction_rows": int(len(combo)),
            "n_scored_rows": int(len(sc)), "sse_float": float((ef ** 2).sum()), "sse_rounded": float((er ** 2).sum()),
            "mse_rounded": float((er ** 2).mean()), "mae_rounded": float(np.abs(er).mean()), "mean_bias_rounded": float(er.mean()),
            "actual_total": float(sc["y_true"].sum()), "predicted_total": float(sc["pred_rounded"].sum()), "rolling_rank": np.nan}


def pair_score_block(eid, etype, vname, pair_name, sp):
    er = sp["error_rounded"].astype(float).to_numpy()
    ef = sp["error_float"].astype(float).to_numpy()
    return {"evaluation_id": eid, "evaluation_type": etype, "method": vname, "pair": pair_name, "n_scored_rows": int(len(sp)),
            "actual_total": float(sp["y_true"].sum()), "predicted_total": float(sp["pred_rounded"].sum()),
            "sse_float": float((ef ** 2).sum()), "sse_rounded": float((er ** 2).sum()), "mse_rounded": float((er ** 2).mean()) if len(er) else np.nan,
            "mae_rounded": float(np.abs(er).mean()) if len(er) else np.nan, "mean_bias_rounded": float(er.mean()) if len(er) else np.nan,
            "zero_true_rate": float((sp["y_true"] == 0).mean()) if len(sp) else np.nan,
            "zero_prediction_rate": float((sp["pred_rounded"] == 0).mean()) if len(sp) else np.nan}


def dir_score_block(eid, etype, vname, s, t, sd):
    er = sd["error_rounded"].astype(float).to_numpy()
    ef = sd["error_float"].astype(float).to_numpy()
    pair = PAIR_OF[(s, t)][0]; role = PAIR_OF[(s, t)][1]
    return {"evaluation_id": eid, "evaluation_type": etype, "method": vname, "task_key": f"{s}->{t}", "pair": pair,
            "direction_role": role, "n_scored_rows": int(len(sd)), "actual_total": float(sd["y_true"].sum()),
            "predicted_total": float(sd["pred_rounded"].sum()), "sse_float": float((ef ** 2).sum()),
            "sse_rounded": float((er ** 2).sum()), "mse_rounded": float((er ** 2).mean()) if len(er) else np.nan,
            "mae_rounded": float(np.abs(er).mean()) if len(er) else np.nan, "mean_bias_rounded": float(er.mean()) if len(er) else np.nan,
            "zero_true_rate": float((sd["y_true"] == 0).mean()) if len(sd) else np.nan,
            "zero_prediction_rate": float((sd["pred_rounded"] == 0).mean()) if len(sd) else np.nan}


def build_nonoverlap(fold_scores, preds, scenarios):
    # block1 = fold_01 valid, block2 = fold_08 valid
    rows = []
    for vname, *_ in VARIANTS:
        for block, eid in [("nonoverlap_block_1", "fold_01"), ("nonoverlap_block_2", "fold_08"), ("nonoverlap_combined", None)]:
            if block == "nonoverlap_combined":
                sub = preds[(preds["method"] == vname) & (preds["evaluation_id"].isin(["fold_01", "fold_08"])) & preds["score_available"]]
            else:
                sub = preds[(preds["method"] == vname) & (preds["evaluation_id"] == eid) & preds["score_available"]]
            er = sub["error_rounded"].astype(float).to_numpy()
            dts = pd.to_datetime(sub["date"].unique())
            rows.append({"method": vname, "block": block, "target_start": f"{dts.min():%Y-%m-%d}" if len(dts) else "",
                         "target_end": f"{dts.max():%Y-%m-%d}" if len(dts) else "", "n_scored_rows": int(len(sub)),
                         "sse_rounded": float((er ** 2).sum()), "mse_rounded": float((er ** 2).mean()) if len(er) else np.nan,
                         "mae_rounded": float(np.abs(er).mean()) if len(er) else np.nan, "mean_bias_rounded": float(er.mean()) if len(er) else np.nan})
    return rows


def build_decision(fold_scores, pairscores, dirscores, horizons, nonoverlap, scenarios):
    rows = []
    rolling = fold_scores[fold_scores["evaluation_type"] == "rolling_7d"]
    win = rolling.pivot_table(index="evaluation_id", columns="method", values="sse_rounded", aggfunc="first")
    winc = win.eq(win.min(axis=1), axis=0).sum(axis=0)
    fa = fold_scores[fold_scores["evaluation_id"] == "final_analog"].set_index("method")
    no = nonoverlap.set_index(["method", "block"])
    for v in VARIANTS:
        vname, family, scope, rounding, _ = v
        rsub = rolling[rolling["method"] == vname]
        mse = rsub["mse_rounded"].to_numpy(float); sse = rsub["sse_rounded"].to_numpy(float)
        far = fa.loc[vname]
        no_c = no.loc[(vname, "nonoverlap_combined")]
        h1 = horizons[(horizons["method"] == vname) & (horizons["forecast_horizon"] == 1)]["mse_rounded"]
        h7 = horizons[(horizons["method"] == vname) & (horizons["forecast_horizon"] == 7)]["mse_rounded"]
        row = {"method": vname, "model_family": family, "parameter_scope": scope, "rounding_method": ROUND_LABEL[rounding],
               "rolling_mean_mse": float(np.mean(mse)), "rolling_median_mse": float(np.median(mse)), "rolling_max_mse": float(np.max(mse)),
               "rolling_mean_sse": float(np.mean(sse)), "rolling_fold_win_count": int(winc.get(vname, 0)),
               "final_analog_sse": float(far["sse_rounded"]), "final_analog_mse": float(far["mse_rounded"]),
               "nonoverlap_combined_mse": float(no_c["mse_rounded"]),
               "horizon_1_mse": float(h1.iloc[0]) if len(h1) else np.nan, "horizon_7_mse": float(h7.iloc[0]) if len(h7) else np.nan}
        for pair in PAIRS:
            ps = pairscores[(pairscores["method"] == vname) & (pairscores["pair"] == pair[0])]
            row[f"{pair[0]}_total_sse"] = float(ps["sse_rounded"].sum())
        for (s, t) in DIRS:
            ds = dirscores[(dirscores["method"] == vname) & (dirscores["task_key"] == f"{s}->{t}")]
            row[f"{s}_to_{t}_total_sse"] = float(ds["sse_rounded"].sum())
        rows.append(row)
    return pd.DataFrame(rows)[DECISION_COLS]


def write_all(out_dir, preds, fold_scores, pairscores, dirscores, horizons, nonoverlap, decision, inner_def, hp, scenarios, selected_params):
    p_def = out_dir / "method_definitions.csv"
    p_indef = out_dir / "inner_validation_definitions.csv"
    p_hp = out_dir / "hyperparameter_selection.csv"
    p_pred = out_dir / "b_model_predictions.csv"
    p_fs = out_dir / "b_fold_scores.csv"
    p_ps = out_dir / "b_pair_scores.csv"
    p_ds = out_dir / "b_direction_scores.csv"
    p_hz = out_dir / "b_horizon_scores.csv"
    p_no = out_dir / "b_nonoverlap_scores.csv"
    p_dec = out_dir / "decision_table.csv"
    p_md = out_dir / "summary.md"

    md_rows = []
    for v in VARIANTS:
        vname, family, scope, rounding, fixed = v
        md_rows.append({"method": vname, "model_family": family, "parameter_scope": scope, "rounding_method": ROUND_LABEL[rounding],
                        "daily_total_method": fixed[0] if fixed else "", "profile_method": fixed[1] if fixed else "",
                        "tau": fixed[2] if fixed else "", "requires_inner_selection": scope in ("global", "by_pair"), "deployable": True})
    pd.DataFrame(md_rows)[METHOD_DEF_COLS].to_csv(p_def, index=False, encoding="utf-8-sig")
    inner_def[INNER_DEF_COLS].to_csv(p_indef, index=False, encoding="utf-8-sig")

    hp_out = hp.copy()
    hp_out["_e"] = hp_out["evaluation_id"].map({s[0]: i for i, s in enumerate(scenarios)})
    hp_out["_r"] = hp_out["rounding_method"].map({"independent": 0, "paired": 1})
    hp_out["_s"] = hp_out["parameter_scope"].map({"global": 0, "by_pair": 1})
    hp_out["_p"] = hp_out["pair"].map({"all": -1, **{p[0]: i for i, p in enumerate(PAIRS)}})
    hp_out = hp_out.sort_values(["_e", "_r", "_s", "_p", "daily_total_method", "profile_method", "tau"])
    hp_out[HP_COLS].to_csv(p_hp, index=False, encoding="utf-8-sig")

    pred_out = preds[PRED_COLS].copy()
    pred_out["hour"] = pd.to_datetime(pred_out["hour"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    pred_out["date"] = pd.to_datetime(pred_out["date"]).dt.strftime("%Y-%m-%d")
    pred_out.to_csv(p_pred, index=False, encoding="utf-8-sig")

    fs_out = fold_scores.sort_values(["evaluation_id", "method"])
    fs_out[FOLD_COLS].to_csv(p_fs, index=False, encoding="utf-8-sig")
    ps_out = pairscores.copy()
    ps_out["_e"] = ps_out["evaluation_id"].map({s[0]: i for i, s in enumerate(scenarios)})
    ps_out["_m"] = ps_out["method"].map(VARIANT_ORDER); ps_out["_p"] = ps_out["pair"].map(PAIR_ORDER)
    ps_out.sort_values(["_e", "_m", "_p"])[PAIRSCORE_COLS].to_csv(p_ps, index=False, encoding="utf-8-sig")
    ds_out = dirscores.copy()
    ds_out["_e"] = ds_out["evaluation_id"].map({s[0]: i for i, s in enumerate(scenarios)})
    ds_out["_m"] = ds_out["method"].map(VARIANT_ORDER); ds_out["_d"] = ds_out["task_key"].map(DIR_RANK)
    ds_out.sort_values(["_e", "_m", "_d"])[DIRSCORE_COLS].to_csv(p_ds, index=False, encoding="utf-8-sig")
    hz_out = horizons.copy()
    hz_out["_e"] = hz_out["evaluation_id"].map({s[0]: i for i, s in enumerate(scenarios)})
    hz_out["_m"] = hz_out["method"].map(VARIANT_ORDER)
    hz_out.sort_values(["_e", "_m", "forecast_horizon"])[HORIZON_COLS].to_csv(p_hz, index=False, encoding="utf-8-sig")
    nonoverlap[NONOVERLAP_COLS].to_csv(p_no, index=False, encoding="utf-8-sig")
    decision.to_csv(p_dec, index=False, encoding="utf-8-sig")

    write_summary(p_md, decision, hp, selected_params, scenarios)

    print("\n=== Output files ===")
    for path in [p_def, p_indef, p_hp, p_pred, p_fs, p_ps, p_ds, p_hz, p_no, p_dec, p_md]:
        rows = len(pd.read_csv(path, encoding="utf-8-sig")) if path.suffix == ".csv" else sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {path.relative_to(ROOT)}  rows={rows}")
    print("\n=== rolling median MSE (all 7 variants) ===")
    for _, r in decision.sort_values("rolling_median_mse").iterrows():
        print(f"  {r['method']:38s} med={r['rolling_median_mse']:.3f} mean={r['rolling_mean_mse']:.3f} max={r['rolling_max_mse']:.3f} fa={r['final_analog_sse']:.0f} no={r['nonoverlap_combined_mse']:.3f}")


def write_summary(path, decision, hp, selected_params, scenarios):
    L = []
    L.append("# Step 15 (B2) B Pair-Decomposition Benchmark")
    L.append("")
    L.append("> Three layer-pair decomposition vs circular3 baseline. No hurdle/net-flow/ensemble/vessel-scaling. Not submitted.")
    L.append("")
    L.append(f"Run command: `{RUN_CMD}`")
    L.append("")
    L.append("## 1. Boundary-safe design\n- 9 outer scenarios; 63 inner folds; train-end 23:00 excluded; 2018-01-24 23:00 right-censored. Pair model uses COMPLETE training dates only (outer train_end-1; inner valid_date-2, >=5 dates); circular3 baseline uses the step10 observable-hour rule. No validation AIS / stock / vessel count used.\n")
    L.append("## 2. Step 10 baseline reproduction\n- circular3_independent reproduces step10 pair_hour_circular_cv/independent with max abs diff < 1e-10 (asserted in code). **PASS**\n")
    dd = decision.set_index("method")
    L.append("## 3. Fixed pair decomposition\n")
    L.append("| method | rolling med MSE | final SSE | nonoverlap MSE |")
    L.append("| --- | --- | --- | --- |")
    for m in ["circular3_independent", "pair_default_independent", "pair_default_hierarchical"]:
        r = dd.loc[m]
        L.append(f"| {m} | {r['rolling_median_mse']:.3f} | {r['final_analog_sse']:.0f} | {r['nonoverlap_combined_mse']:.3f} |")
    L.append("")
    L.append("## 4. Global vs by-pair CV\n")
    L.append("| method | rolling med | rolling max | final | nonoverlap |")
    L.append("| --- | --- | --- | --- | --- |")
    for m in ["pair_global_cv_independent", "pair_global_cv_hierarchical", "pair_bypair_cv_independent", "pair_bypair_cv_hierarchical"]:
        r = dd.loc[m]
        L.append(f"| {m} | {r['rolling_median_mse']:.3f} | {r['rolling_max_mse']:.3f} | {r['final_analog_sse']:.0f} | {r['nonoverlap_combined_mse']:.3f} |")
    L.append("")
    # selected param distributions (final_analog, independent)
    fa_hp = hp[(hp["evaluation_id"] == "final_analog") & (hp["rounding_method"] == "independent") & hp["selected"]]
    L.append("## 5-7. Selected parameters (final_analog, independent)\n")
    L.append("| scope | pair | daily | profile | tau |")
    L.append("| --- | --- | --- | --- | --- |")
    for _, r in fa_hp.sort_values(["parameter_scope", "pair"]).iterrows():
        L.append(f"| {r['parameter_scope']} | {r['pair']} | {r['daily_total_method']} | {r['profile_method']} | {int(r['tau'])} |")
    L.append("")
    L.append("## 8. Independent vs hierarchical rounding\n- See decision_table rolling/final/nonoverlap columns for pair_default and pair_global_cv under both roundings.\n")
    L.append("## 9. Forecast horizon\n- horizon_1_mse and horizon_7_mse in decision_table.\n")
    L.append("## 10. Non-overlap blocks\n- nonoverlap_combined_mse in decision_table (fold_01 + fold_08 targets, non-overlapping).\n")
    L.append("## 11. Decision for B3\n- Judged from the tables above against the pass criteria (>=5 rolling folds beat circular3; final not worse; nonoverlap improved; max MSE not >5% worse; >=4 of 6 directions improved; core->near and near->core not both clearly worse).\n")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
